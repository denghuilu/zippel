"""CSE and dead-code elimination on the segmented-polynomial IR.

Neither pass reassociates arithmetic, so both are **exact in floating point** — no tolerance
is involved and no result changes. That matters here: the derived programs are validated
against FP64 oracles, and a simplifier that perturbed values would make those comparisons
meaningless.

CSE is structural hashing over `(op kind, attributes, input ids)` in topological order
(docs/ir.md section 4). DCE is reachability from the program's declared outputs.
"""

from __future__ import annotations

from dataclasses import replace

from zippel.ir import IndexType, Op, Program


def _rewrite_inputs(op: Op, sub: dict[str, str]) -> Op:
    return replace(
        op,
        inputs=tuple(sub.get(i, i) for i in op.inputs),
        index_maps=tuple(sub.get(m, m) if m else m for m in op.index_maps),
        out_index_map=sub.get(op.out_index_map, op.out_index_map) if op.out_index_map else None,
    )


def cse(prog: Program) -> Program:
    """Common-subexpression elimination by structural hashing.

    Runs in definition order, which is already topological, so an op's inputs are canonical
    by the time it is hashed.
    """
    out = Program(inputs=dict(prog.inputs), outputs=prog.outputs, _counter=prog._counter)
    seen: dict[tuple, str] = {}
    sub: dict[str, str] = {}

    for name in prog.topo():
        op = _rewrite_inputs(prog.ops[name], sub)
        key = op.key()
        if key in seen:
            sub[name] = seen[key]
            continue
        seen[key] = name
        out.ops[name] = replace(op, name=name)

    out.outputs = tuple(sub.get(o, o) for o in prog.outputs)
    return out


def dce(prog: Program, keep: tuple[str, ...] | None = None) -> Program:
    """Drop ops no declared output depends on."""
    roots = set(keep or ()) | set(prog.outputs)
    live: set[str] = set()
    stack = [r for r in roots if r in prog.ops]
    live.update(stack)

    while stack:
        op = prog.ops[stack.pop()]
        for src in (*op.inputs, *(m for m in op.index_maps if m),
                    *( (op.out_index_map,) if op.out_index_map else () )):
            if src in prog.ops and src not in live:
                live.add(src)
                stack.append(src)

    out = Program(inputs=dict(prog.inputs), outputs=prog.outputs, _counter=prog._counter)
    for name in prog.topo():
        if name in live:
            out.ops[name] = prog.ops[name]
    return out


def simplify(prog: Program, keep: tuple[str, ...] | None = None) -> Program:
    """DCE then CSE then DCE: the second DCE collects ops orphaned by CSE's rewrites."""
    return dce(cse(dce(prog, keep)), keep)


# ----------------------------------------------------------------------------------------
# statistics
# ----------------------------------------------------------------------------------------


def op_counts(prog: Program) -> dict[str, int]:
    counts = {"total": len(prog.ops), "segmented_contraction": 0, "scalar_map": 0, "paths": 0}
    for op in prog.ops.values():
        counts[op.kind] += 1
        counts["paths"] += len(op.paths)
    return counts


def signatures(prog: Program) -> dict[tuple, int]:
    """Distinct op signatures and their multiplicity.

    The Phase 2 kernel-count proxy: ops sharing a signature can be served by one generated
    kernel with different pointers, so this is an upper bound on how many kernels hand
    scheduling has to write.
    """
    out: dict[tuple, int] = {}
    for op in prog.ops.values():
        out[op.signature()] = out.get(op.signature(), 0) + 1
    return out


def contraction_signatures(prog: Program) -> dict[tuple, int]:
    return {s: n for s, n in signatures(prog).items() if s[0] == "segmented_contraction"}


def kernel_families(prog: Program) -> dict[tuple, int]:
    """Coarser than `signatures`: slice *offsets* are abstracted away, extents kept.

    A generated kernel can take slice offsets as runtime arguments, so two ops that differ
    only in where they read and write are the same kernel. `signatures` counts them
    separately and is therefore an upper bound on how many kernels Phase 2 must write; this
    is the corresponding lower bound. The truth is in between and depends on how much the
    emitter parameterises, which is a Phase 2 decision.
    """
    def extent(sl):
        if sl.start is None and sl.stop is None:
            return None
        return (sl.stop or 0) - (sl.start or 0)

    out: dict[tuple, int] = {}
    for op in prog.ops.values():
        if op.kind != "segmented_contraction":
            continue
        key = (str(op.out_type),
               tuple(m is not None for m in op.index_maps),
               op.out_index_map is not None,
               tuple(sorted(
                   (p.subscripts, p.operands,
                    tuple(tuple(extent(x) for x in g) for g in p.in_slices),
                    tuple(extent(x) for x in p.out_slice))
                   for p in op.paths)))
        out[key] = out.get(key, 0) + 1
    return out




def _canonical_subscripts(subscripts: str) -> str:
    """Rename indices in order of first appearance, so `"ij,jc->ic"` and `"ab,bd->ad"` agree."""
    mapping: dict[str, str] = {}
    out = []
    for ch in subscripts:
        if ch in ",->":
            out.append(ch)
            continue
        if ch not in mapping:
            mapping[ch] = chr(ord("a") + len(mapping))
        out.append(mapping[ch])
    return "".join(out)


def archetypes(prog: Program) -> dict[tuple, int]:
    """Coarsest useful grouping: what an *emitter* has to know how to write.

    A generated kernel takes its coefficient and slice tables as runtime data, and its
    extents as parameters. What it cannot take at runtime is its *shape of computation*:
    which operands are gathered, whether the result is scattered, and the contraction
    pattern up to index renaming. That is the archetype.

    So: extents dropped, path coefficients and slices dropped, path *count* dropped (a
    kernel loops over a path table), and subscripts canonicalised by renaming indices in
    order of first appearance. What remains is the set of distinct contraction patterns the
    emitter must handle -- the number a human actually writes code for.
    """
    out: dict[tuple, int] = {}
    for op in prog.ops.values():
        if op.kind == "scalar_map":
            key = ("scalar_map", op.fn)
        else:
            key = ("segmented_contraction",
                   op.out_type.segment,
                   tuple(m is not None for m in op.index_maps),
                   op.out_index_map is not None,
                   tuple(sorted({_canonical_subscripts(p.subscripts) for p in op.paths})))
        out[key] = out.get(key, 0) + 1
    return out


def op_volume(prog: Program, op: "Op") -> int:
    """Index-space volume of one op: the number of scalar terms it would unroll to.

    O(paths) arithmetic -- extents multiplied, never enumerated. Used as the width proxy for the
    fusion cap, because compile time is quadratic in a group's term count (D35) and terms track
    this volume closely.
    """
    import math

    if op.kind == "scalar_map":
        return math.prod(op.out_type.sizes or (1,))
    total = 0
    for p in op.paths:
        specs, _ = p.parse()
        extent: dict[str, int] = {}
        for pos, j in enumerate(p.operands):
            t = prog.type_of(op.inputs[j])
            sl = p.slices_for(pos)
            sizes = (tuple(len(range(*x.indices(f))) for x, f in zip(sl, t.sizes))
                     if sl else t.sizes)
            for ch, size in zip(specs[pos], sizes):
                extent[ch] = size
        total += math.prod(extent.values()) if extent else 1
    return total


def fusion_groups(prog: Program, max_volume: int | None = None) -> list[list[str]]:
    """Greedy producer-consumer fusion partition, constrained to stay schedulable.

    Fusable means: the consumer reads the producer's output, both live on the same segment
    axis, their trailing extents match (so one loop nest covers both), and the consumer does
    no gather or scatter (an index map re-maps the segment axis, which breaks the shared
    loop). A `scalar_map` is always fusable into its producer under those conditions.

    **Acyclicity is a hard constraint, not a refinement.** Fusing an op into a group adds an
    edge from each of its other producers' groups into that group, and if the group can already
    reach one of them the result is two kernels each waiting on the other's output. The
    unconstrained greedy version put 36 of 42 forward groups into such cycles -- LayerNorm is
    the archetype, where `x - mean(x)` wants to fuse with `x` while `mean(x)` reduces `x` in a
    group of its own. A partition with a cycle is not a launch count at all, so the check runs
    before every merge and the merge is refused if it would close one.

    `max_volume` caps a group's index-space volume, refusing a merge that would exceed it.
    **Fusion width is not free**: `cute.compile` costs `terms^1.97` (D35), so the widest forward
    group -- 23 040 terms -- takes an extrapolated 109 minutes to compile, 13.7x the other 47
    combined, in order to elide one `[E,9,256]` intermediate. Uncapped fusion maximises bytes
    saved while ignoring what it costs to build. `None` keeps the historical behaviour, and the
    Gate 1 group counts (48/115/320) are the uncapped numbers.

    Still deliberately cheap and greedy: a Phase 2 *planning* estimate of how many kernel
    launches a straightforward fusion pass would leave, not a scheduling decision.
    """
    def fusable(prod_name: str, cons: "Op") -> bool:
        prod = prog.ops.get(prod_name)
        if prod is None:
            return False
        if any(m is not None for m in cons.index_maps) or cons.out_index_map is not None:
            return False
        if prod.out_type.segment != cons.out_type.segment:
            return False
        return prod.out_type.sizes == cons.out_type.sizes

    group_of: dict[str, int] = {}
    groups: list[list[str]] = []
    succ: dict[int, set[int]] = {}          # producer group -> consumer groups
    volume: dict[int, int] = {}             # group -> index-space volume so far

    def reaches(start: int, goal: int) -> bool:
        stack, seen = [start], {start}
        while stack:
            g = stack.pop()
            if g == goal:
                return True
            for nxt in succ[g]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return False

    def merge_is_safe(target: int, op: "Op") -> bool:
        """Would putting `op` in `target` close a cycle among groups?"""
        for src in op.inputs:
            g = group_of.get(src)
            if g is None or g == target:
                continue
            if reaches(target, g):          # target already depends on g; g -> target closes it
                return False
        return True

    def new_group(name: str) -> int:
        groups.append([name])
        succ[len(groups) - 1] = set()
        volume[len(groups) - 1] = op_volume(prog, prog.ops[name])
        return len(groups) - 1

    for name in prog.topo():
        op = prog.ops[name]
        target = None
        cost = op_volume(prog, op) if max_volume is not None else 0
        for src in op.inputs:
            g = group_of.get(src)
            if g is None or not fusable(src, op) or not merge_is_safe(g, op):
                continue
            if max_volume is not None and volume[g] + cost > max_volume:
                continue                     # would make the group too expensive to compile
            target = g
            break
        if target is None:
            target = new_group(name)
        else:
            groups[target].append(name)
            volume[target] = volume.get(target, 0) + op_volume(prog, op)
        group_of[name] = target
        for src in op.inputs:
            g = group_of.get(src)
            if g is not None and g != target:
                succ[g].add(target)
    return groups


__all__ = ["cse", "dce", "simplify", "op_counts", "signatures",
           "contraction_signatures", "kernel_families", "archetypes", "fusion_groups",
           "op_volume"]
