"""Lower a fusion group to a straight-line scalar schedule with structural sparsity.

This is the half of the emitter that is target-independent: it turns a fusion group of
segmented-polynomial ops into a list of scalar assignments over *named* values, with every
structurally-zero element eliminated. `codegen/emit.py` renders that schedule as CuTe DSL.

Why unroll. The trailing extents here are tiny -- 9x9 Wigner blocks, 9x128 messages -- while the
segment axis is ~260k edges. So the segment axis is the parallel axis and the trailing axes are
loop bodies small enough to unroll completely. Full unrolling is what makes the next paragraph
possible.

Why sparsity is the whole point. A Wigner rotation is block-diagonal: 35 of 81 entries are
nonzero at lmax=2, and products of block-diagonals stay block-diagonal. FlashSO2 measured a
dense WGMMA tile at 0.55x/0.46x/0.33x of small-tile Triton at lmax 4/6/8 and diagnosed the cause
as "the dense N x K tile pays quadratically for the block-diagonal zeros" (DECISIONS.md D22). In
an unrolled scalar schedule that cost is not paid at all: a term whose operand is known zero is
never emitted, so the block structure costs nothing rather than costing quadratically. The
sparsity here is discovered from the IR's coefficient structure, not declared by hand.

The masks are *conservative*: an element is marked possibly-nonzero unless every term that could
write it is provably absent. Exact zeros arising from cancellation between terms are not
detected, and must not be -- that would make the schedule depend on values.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from zippel.ir import IndexType, Op, Program

# ------------------------------------------------------------------------------------------
# group analysis
# ------------------------------------------------------------------------------------------


@dataclass
class GroupSpec:
    """One fusion group, resolved into everything the emitter needs.

    `live_in` are buffers produced outside the group (or program inputs) that its ops read;
    `live_out` are buffers the group produces that something outside needs -- a declared program
    output, or an op in another group. Everything else is internal and never leaves registers.
    That set is the entire point: those are the stores the fused kernel does not make.
    """

    name: str
    segment: str
    ops: tuple[str, ...]
    live_in: tuple[str, ...]
    live_out: tuple[str, ...]
    internal: tuple[str, ...]

    def __str__(self) -> str:
        return (f"{self.name}[{self.segment}] {len(self.ops)} ops, "
                f"in {len(self.live_in)}, out {len(self.live_out)}, "
                f"internal {len(self.internal)}")


def analyze_group(prog: Program, group: list[str], name: str = "g") -> GroupSpec:
    members = set(group)
    consumers = prog.consumers()
    outputs = set(prog.outputs)

    live_in, seen = [], set()
    for n in group:
        for src in prog.ops[n].inputs:
            if src not in members and src not in seen:
                seen.add(src)
                live_in.append(src)

    live_out, internal = [], []
    for n in group:
        needed = n in outputs or any(c not in members for c in consumers.get(n, []))
        (live_out if needed else internal).append(n)

    segments = {prog.ops[n].out_type.segment for n in group}
    if len(segments) != 1:
        raise ValueError(f"group {name} spans segments {segments}; not a single loop nest")

    return GroupSpec(name=name, segment=segments.pop(), ops=tuple(group),
                     live_in=tuple(live_in), live_out=tuple(live_out),
                     internal=tuple(internal))


def index_maps_used(prog: Program, spec: GroupSpec) -> bool:
    return any(any(m is not None for m in prog.ops[n].index_maps)
               or prog.ops[n].out_index_map is not None for n in spec.ops)


# ------------------------------------------------------------------------------------------
# structural sparsity
# ------------------------------------------------------------------------------------------


def _path_assignments(path, operand_sizes: list[tuple[int, ...]], out_rank: int):
    """Enumerate every index assignment of one path.

    Yields `(out_index, [in_index per operand position])` in *sliced-local* coordinates. The
    caller adds slice offsets. Extents come from the operands, which the type checker has
    already proven consistent (zippel/ir.py `_check_contraction`).
    """
    specs, out_spec = path.parse()
    extent: dict[str, int] = {}
    for pos, spec in enumerate(specs):
        for ch, size in zip(spec, operand_sizes[pos]):
            extent[ch] = size
    letters = sorted(extent)
    if len(out_spec) != out_rank:
        raise ValueError(f"path {path.subscripts!r} output rank {len(out_spec)} != {out_rank}")
    for combo in itertools.product(*(range(extent[c]) for c in letters)):
        assign = dict(zip(letters, combo))
        yield (tuple(assign[c] for c in out_spec),
               [tuple(assign[c] for c in spec) for spec in specs])


def _offset(sl: tuple, idx: tuple[int, ...]) -> tuple[int, ...]:
    """Map sliced-local indices to buffer-absolute ones."""
    if not sl:
        return idx
    return tuple((s.start or 0) + i for s, i in zip(sl, idx))


def all_indices(sizes: tuple[int, ...]) -> set[tuple[int, ...]]:
    return set(itertools.product(*(range(s) for s in sizes))) or {()}


def nonzero_masks(prog: Program, spec: GroupSpec,
                  given: dict[str, set[tuple[int, ...]]] | None = None,
                  ) -> dict[str, set[tuple[int, ...]]]:
    """Possibly-nonzero trailing indices for every buffer the group touches.

    Live-ins default to dense; pass `given` to declare a sparser input (that is how the
    block-diagonal structure of a *materialized* Wigner operand enters). Propagation is forward
    over the group in topological order.
    """
    masks: dict[str, set[tuple[int, ...]]] = dict(given or {})

    for n in spec.live_in:
        if n in masks:
            continue
        t = prog.type_of(n)
        masks[n] = set() if isinstance(t, IndexType) else all_indices(t.sizes)

    for n in spec.ops:
        op = prog.ops[n]
        if op.kind == "scalar_map":
            # Every f' in the vocabulary maps 0 -> f(0), which is nonzero for several of them
            # (exp, sigmoid, rsqrt...). Only the ones with f(0) == 0 preserve sparsity; the rest
            # densify. Being wrong in the safe direction here costs registers, not correctness.
            src = masks[op.inputs[0]]
            masks[n] = src if op.fn in ("silu", "sin") else all_indices(op.out_type.sizes)
            continue

        live: set[tuple[int, ...]] = set()
        for p in op.paths:
            sizes = []
            for pos, j in enumerate(p.operands):
                t = prog.type_of(op.inputs[j])
                sl = p.slices_for(pos)
                full = t.sizes
                sizes.append(tuple(len(range(*s.indices(f))) for s, f in zip(sl, full))
                             if sl else full)
            for out_idx, in_idxs in _path_assignments(p, sizes, len(p.out_slice) or
                                                      op.out_type.rank):
                ok = True
                for pos, j in enumerate(p.operands):
                    abs_idx = _offset(p.slices_for(pos), in_idxs[pos])
                    if abs_idx not in masks[op.inputs[j]]:
                        ok = False
                        break
                if ok:
                    live.add(_offset(p.out_slice, out_idx))
        masks[n] = live
    return masks


# ------------------------------------------------------------------------------------------
# the scalar schedule
# ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Term:
    """One product term contributing to one output element."""

    coeff: float
    factors: tuple[tuple[str, tuple[int, ...]], ...]   # (buffer, absolute trailing index)


@dataclass
class Assign:
    """`target[index] = sum(terms)` -- or `f(source[index])` for a scalar_map."""

    target: str
    index: tuple[int, ...]
    terms: tuple[Term, ...] = ()
    fn: str | None = None
    order: int = 0
    source: tuple[str, tuple[int, ...]] | None = None


@dataclass
class Schedule:
    spec: GroupSpec
    assigns: list[Assign] = field(default_factory=list)
    masks: dict[str, set[tuple[int, ...]]] = field(default_factory=dict)
    last_use: dict[tuple[str, tuple[int, ...]], int] = field(default_factory=dict)

    @property
    def n_terms(self) -> int:
        return sum(len(a.terms) for a in self.assigns)

    @property
    def n_values(self) -> int:
        return len(self.assigns)

    def peak_live_values(self) -> int:
        """Max simultaneously-live scalar registers, by a single forward pass.

        A value dies after its last read; live-outs never die. This is the register-pressure
        number that decides whether a group is emittable as one kernel at all.
        """
        outs = set(self.spec.live_out)
        live: set[tuple[str, tuple[int, ...]]] = set()
        peak = 0
        for i, a in enumerate(self.assigns):
            live.add((a.target, a.index))
            peak = max(peak, len(live))
            for key in [k for k in live if k[0] not in outs]:
                if self.last_use.get(key, -1) <= i:
                    live.discard(key)
        return peak


def build_schedule(prog: Program, spec: GroupSpec,
                   given: dict[str, set[tuple[int, ...]]] | None = None) -> Schedule:
    """Turn a fusion group into straight-line scalar assignments, zeros elided."""
    if index_maps_used(prog, spec):
        raise NotImplementedError(
            f"group {spec.name} carries gather/scatter index maps; "
            "the v1 emitter handles register-resident groups only")

    masks = nonzero_masks(prog, spec, given)
    sched = Schedule(spec=spec, masks=masks)

    for n in spec.ops:
        op = prog.ops[n]
        if op.kind == "scalar_map":
            for idx in sorted(masks[n]):
                sched.assigns.append(Assign(target=n, index=idx, fn=op.fn, order=op.order,
                                            source=(op.inputs[0], idx)))
            continue

        # accumulate terms per output element, then emit one assignment each
        acc: dict[tuple[int, ...], list[Term]] = {idx: [] for idx in masks[n]}
        for p in op.paths:
            sizes = []
            for pos, j in enumerate(p.operands):
                t = prog.type_of(op.inputs[j])
                sl = p.slices_for(pos)
                sizes.append(tuple(len(range(*s.indices(f))) for s, f in zip(sl, t.sizes))
                             if sl else t.sizes)
            rank = len(p.out_slice) or op.out_type.rank
            for out_idx, in_idxs in _path_assignments(p, sizes, rank):
                factors, ok = [], True
                for pos, j in enumerate(p.operands):
                    abs_idx = _offset(p.slices_for(pos), in_idxs[pos])
                    if abs_idx not in masks[op.inputs[j]]:
                        ok = False
                        break
                    factors.append((op.inputs[j], abs_idx))
                if not ok:
                    continue
                acc[_offset(p.out_slice, out_idx)].append(
                    Term(coeff=p.coeff, factors=tuple(factors)))

        for idx in sorted(acc):
            sched.assigns.append(Assign(target=n, index=idx, terms=tuple(acc[idx])))

    for i, a in enumerate(sched.assigns):
        for t in a.terms:
            for factor in t.factors:
                sched.last_use[factor] = i
        if a.source is not None:
            sched.last_use[a.source] = i
    return sched


def dense_term_count(prog: Program, spec: GroupSpec) -> int:
    """Terms a schedule would have with no sparsity -- the WGMMA-equivalent work."""
    total = 0
    for n in spec.ops:
        op = prog.ops[n]
        if op.kind == "scalar_map":
            total += len(all_indices(op.out_type.sizes))
            continue
        for p in op.paths:
            sizes = []
            for pos, j in enumerate(p.operands):
                t = prog.type_of(op.inputs[j])
                sl = p.slices_for(pos)
                sizes.append(tuple(len(range(*s.indices(f))) for s, f in zip(sl, t.sizes))
                             if sl else t.sizes)
            rank = len(p.out_slice) or op.out_type.rank
            total += sum(1 for _ in _path_assignments(p, sizes, rank))
    return total


__all__ = ["GroupSpec", "Schedule", "Term", "Assign", "analyze_group", "build_schedule",
           "nonzero_masks", "dense_term_count", "all_indices", "index_maps_used"]
