"""T2 — cooperative tile: channels on threads, everything else unrolled.

T1 unrolls every trailing axis into registers, which works while the trailing extents are small
(the 9x9 Wigner blocks). It does not survive a channel axis: the radial MLP needs 641 live
scalars per thread and the SO(2) conv 3329, so `codegen/emit.py` refuses them. Forcing T1 there
would spill to local memory and lose exactly the locality the fusion exists to buy.

T2 keeps the same straight-line scalar model but makes the channel axis **symbolic**: one thread
owns one channel (or a slice of them), and a value's channel coordinate is the thread's own
rather than a literal. The other trailing axes stay unrolled, and structural sparsity still
applies to them. Channels carry no structural sparsity -- they are dense by construction -- so
the mask machinery deliberately does not run over that axis.

Two access patterns fall out, and the difference is the whole template:

  aligned      every operand reads the channel the output writes (`c->c`, `c,c->c`, `,c->c`,
               every scalar_map). Purely per-thread; no communication.
  contracted   the path sums over a channel-extent index that is NOT the output's channel
               (`i,oi->o`, the Linear layers). Thread `o` needs every input channel `i`, so the
               operand is staged in shared memory and the sum is emitted over it.

A buffer that is read contracted must be in smem before any thread reads it, so each staging
point costs a barrier. Those barriers are the reason the fusion pass and this template have to
agree: a group is only worth fusing if the barriers it introduces cost less than the round trip
to memory it removes.

Correctness first, per the Phase 2 discipline: this emits fully unrolled sums (`cutlass.range_
constexpr`-equivalent, generated as Python expressions), which is bit-exact against the FP64
interpreter but makes no attempt at vectorization, double buffering, or MMA. Those are the
performance variants, and they come after the bit-exact one.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from zippel.ir import IndexType, Program

#: Sentinel for "this thread's own channel" in an index tuple.
CH = "c"


@dataclass(frozen=True)
class TileTerm:
    coeff: float
    #: (buffer, trailing index, from_smem). `from_smem` marks a read of a value that lives in
    #: another *thread's* registers, which is the only case that forces a barrier.
    factors: tuple[tuple[str, tuple, bool], ...]


@dataclass
class TileAssign:
    target: str
    index: tuple                       # trailing index, with CH at the channel position
    terms: tuple[TileTerm, ...] = ()
    fn: str | None = None
    order: int = 0
    source: tuple[str, tuple] | None = None


@dataclass
class TileSchedule:
    spec: object
    axis: int                          # position of the channel axis in the trailing axes
    extent: int                        # channel extent
    assigns: list[TileAssign] = field(default_factory=list)
    staged: list[str] = field(default_factory=list)   # buffers needing smem, in staging order
    stage_before: dict[int, list[str]] = field(default_factory=dict)

    @property
    def n_terms(self) -> int:
        return sum(len(a.terms) for a in self.assigns)

    @property
    def n_values(self) -> int:
        return len(self.assigns)


def channel_axis(prog: Program, spec, min_extent: int = 32) -> tuple[int, int] | None:
    """Pick the axis to put on threads: the largest trailing axis shared by every op output.

    Returns `(position, extent)`, or None if no axis qualifies -- in which case the group is not
    a T2 candidate and the selection rule falls through to splitting it (docs/templates.md 2).
    """
    sizes = {prog.ops[n].out_type.sizes for n in spec.ops}
    if len(sizes) != 1:
        return None
    (shape,) = sizes
    if not shape:
        return None
    pos = max(range(len(shape)), key=lambda i: shape[i])
    return (pos, shape[pos]) if shape[pos] >= min_extent else None


def _sliced(t, sl):
    return tuple(len(range(*s.indices(f))) for s, f in zip(sl, t.sizes)) if sl else t.sizes


def _offset(sl, idx):
    if not sl:
        return idx
    out = []
    for s, i in zip(sl, idx):
        out.append(i if i == CH else (s.start or 0) + i)
    return tuple(out)


def build_tile_schedule(prog: Program, spec, axis: int, extent: int) -> TileSchedule:
    """Straight-line schedule with the channel axis symbolic and everything else unrolled."""
    sched = TileSchedule(spec=spec, axis=axis, extent=extent)
    produced: set[str] = set()
    need_stage: set[str] = set()

    def unrolled_indices(sizes, ch_pos):
        """All trailing indices of a buffer, with CH pinned at its channel position."""
        ranges = [(CH,) if i == ch_pos else range(s) for i, s in enumerate(sizes)]
        return [tuple(x) for x in itertools.product(*ranges)]

    for n in spec.ops:
        op = prog.ops[n]
        out_sizes = op.out_type.sizes

        if op.kind == "scalar_map":
            for idx in unrolled_indices(out_sizes, axis):
                sched.assigns.append(TileAssign(target=n, index=idx, fn=op.fn, order=op.order,
                                                source=(op.inputs[0], idx)))
            produced.add(n)
            continue

        acc: dict[tuple, list[TileTerm]] = {}
        for p in op.paths:
            specs, out_spec = p.parse()
            sizes = [_sliced(prog.type_of(op.inputs[j]), p.slices_for(pos))
                     for pos, j in enumerate(p.operands)]
            extent_of: dict[str, int] = {}
            for pos, s in enumerate(specs):
                for ch, sz in zip(s, sizes[pos]):
                    extent_of[ch] = sz

            # the output's channel letter, and any *other* letter of channel extent that this
            # path sums over -- that one is the contraction the staging exists for
            out_rank = len(p.out_slice) or op.out_type.rank
            ch_letter = out_spec[axis] if len(out_spec) == out_rank and axis < len(out_spec) \
                else None
            summed = [c for c in extent_of if c not in out_spec]
            ch_summed = [c for c in summed if extent_of[c] == extent and c != ch_letter]

            # unroll every letter except the output's channel letter
            free = [c for c in sorted(extent_of) if c != ch_letter]
            for combo in itertools.product(*(range(extent_of[c]) for c in free)):
                assign = dict(zip(free, combo))
                if ch_letter is not None:
                    assign[ch_letter] = CH
                out_idx = _offset(p.out_slice, tuple(assign[c] for c in out_spec))
                factors = []
                for pos, j in enumerate(p.operands):
                    idx = _offset(p.slices_for(pos), tuple(assign[c] for c in specs[pos]))
                    buf = op.inputs[j]
                    # Cross-channel read: this operand is indexed by a summed channel letter, so
                    # thread `c` needs a value belonging to channel k != c.
                    cross = any(ch in ch_summed for ch in specs[pos])
                    # It only has to travel through smem if it lives in another thread's
                    # *registers*. A live-in is in gmem and every thread can just read it --
                    # correct, and it keeps the barrier count down. A weight matrix indexed
                    # [o, i] is not cross at all: thread o reads its own row.
                    from_smem = cross and buf in produced
                    if from_smem:
                        need_stage.add(buf)
                    factors.append((buf, idx, from_smem))
                acc.setdefault(out_idx, []).append(TileTerm(p.coeff, tuple(factors)))

        for idx in sorted(acc, key=str):
            sched.assigns.append(TileAssign(target=n, index=idx, terms=tuple(acc[idx])))
        produced.add(n)

    # Each staged buffer is written to smem and barriered once, immediately before the first
    # assignment that reads it from another thread.
    sched.staged = sorted(need_stage)
    first: dict[str, int] = {}
    for i, a in enumerate(sched.assigns):
        for t in a.terms:
            for buf, _idx, from_smem in t.factors:
                if from_smem and buf not in first:
                    first[buf] = i
    for buf, i in first.items():
        sched.stage_before.setdefault(i, []).append(buf)
    for i in sched.stage_before:
        sched.stage_before[i].sort()
    return sched


__all__ = ["CH", "TileSchedule", "TileAssign", "TileTerm", "build_tile_schedule", "channel_axis"]
