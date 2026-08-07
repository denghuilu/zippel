"""Per-kernel floating-point ordering bounds (DECISIONS.md D25).

A generated kernel and the FP64 interpreter compute the same mathematical quantity by summing
the same terms in different orders -- the interpreter through `einsum`'s blocked reduction, the
kernel sequentially with FMA contraction. Neither is more correct. The question a test must ask
is therefore not "are they equal" but "do they differ by more than reordering can explain", and
that quantity is computable.

Standard result: for an n-term floating-point sum, any evaluation order satisfies

    |fl(sum) - sum|  <=  (n - 1) * eps * SUM |x_i|

so two *different* orders differ by at most twice that. Note the bound is over `SUM |x_i|`, not
over the result: with cancellation the terms can be far larger than what they add up to, and a
bound written against the result would be too tight and would fire on correct kernels.

`SUM |x_i|` is data-dependent, so the emitter cannot produce a number -- it produces the
*structure* (the reduction depth of the schedule it just emitted) and this module evaluates the
magnitude against real inputs. The two together are the bound the harness asserts, and neither
half can be loosened without changing the schedule or the fixture.

Per D26 this is an upper bound by construction: chaining through internal buffers uses the
triangle inequality, so a chained group's bound over-estimates rather than under-estimates.

**What this catches, and what it does not.** Being a rigorous worst-case bound it is loose --
measured errors run 0.2-0.9 % of it in practice. It reliably catches the failures that actually
happen in a generated kernel (a dropped term, a missing barrier, a wrong slice, a transposed
index), because those move the result by O(1) rather than by ulps. It would not catch a bug that
perturbs the answer by a few ulps, and it is not offered as doing so; T1's separate bit-equality
assertion is the tight check, and it applies wherever the orders genuinely match.
"""

from __future__ import annotations

import torch

from codegen.tile import CH, Ch, TileSchedule

#: FP64 unit roundoff.
EPS = torch.finfo(torch.float64).eps


def reduction_depth(sched) -> int:
    """The most terms any single output element sums. Purely structural."""
    return max((len(a.terms) for a in sched.assigns), default=1)


def _factor_tensor(env: dict[str, torch.Tensor], buf: str, idx: tuple,
                   window: tuple[int, int] | None = None) -> tuple[torch.Tensor, bool]:
    """|env[buf]| at trailing index `idx`, and whether that index ranges over channels.

    The segment axis is always kept (a `none`-segment buffer keeps its length-1 axis, which
    broadcasts). A channel component is kept too -- it is the thread index, so the magnitude is
    evaluated over every participating channel rather than one.

    `window` is the half-open range of thread channels this assignment runs on. Thread `c` reads
    the operand at `c + offset`, so the operand slice is `[lo + offset, hi + offset)` and its
    width is `hi - lo` -- *not* the schedule's full channel extent, which is what a concatenated
    axis (320 wide, from operands of 64/128/128) does not have.
    """
    out = env[buf].abs()
    dim = 1                       # 0 is the segment axis, always kept
    has_ch = False
    for i in idx:
        if isinstance(i, Ch):
            if window is not None:
                lo, hi = window
                start, length = lo + i.offset, hi - lo
                if out.shape[dim] != length or start != 0:
                    out = out.narrow(dim, start, length)
            dim += 1
            has_ch = True
            continue
        out = out.select(dim, i)
    if out.dim() != dim:
        raise ValueError(
            f"index {idx} does not cover the trailing axes of {buf!r} "
            f"(shape {tuple(env[buf].shape)}); the bound would be computed on the wrong slice")
    return out, has_ch


def magnitude_sum(sched, env: dict[str, torch.Tensor],
                  gathers: dict[str, str] | None = None) -> float:
    """max over output elements of SUM |term|, evaluated on the real inputs.

    This is the data-dependent half of the bound. Terms are evaluated with absolute values, so
    cancellation makes the result larger than the true output -- which is the point.

    `gathers` maps a live-in to the index buffer it is read through (T3). A gathered operand
    lives on a different segment from the result -- `pos` has 216 rows where the output has 9 576
    edges -- so it must be gathered here too, or the magnitudes do not even align in shape.
    """
    gathers = gathers or {}
    worst = 0.0
    for a in sched.assigns:
        if not a.terms:
            continue
        total: torch.Tensor | None = None
        for term in a.terms:
            extent = getattr(sched, "extent", 0)
            window = getattr(a, "ch_range", None) or ((0, extent) if extent else None)
            parts = []
            gs = getattr(term, "gathers", ()) or (None,) * len(term.factors)
            for f, g in zip(term.factors, gs):
                tensor, has_ch = _factor_tensor(env, f[0], f[1], window)
                # per-factor, not per-buffer: one buffer can be read through several maps
                gmap = g if g is not None else gathers.get(f[0])
                if gmap is not None:
                    tensor = tensor[env[gmap].long()]
                parts.append((tensor, has_ch))
            if not parts:
                continue
            # Align ranks: a factor with no channel index is [seg] while one with CH is
            # [seg, C]. Give the former a trailing singleton so they broadcast.
            wide = any(has_ch for _, has_ch in parts)
            piece = None
            for tensor, has_ch in parts:
                if wide and not has_ch:
                    tensor = tensor.unsqueeze(-1)
                piece = tensor if piece is None else piece * tensor
            piece = piece * abs(term.coeff)
            total = piece if total is None else total + piece
        if total is not None:
            worst = max(worst, float(total.max()))
    return worst


#: Terms per emitted statement; must match CHUNK in the emitters.
_CHUNK = 48


def inlined_live_upper_bound(sched, interleaved_stores: bool = False) -> int:
    """Upper bound on live scalars per thread for a template that INLINES its live-in loads.

    T1 hoists live-ins into register symbols, so `Schedule.peak_live_values` counts them and is
    the right bound there. T2 (`emit_tile.py`) and T3 (`emit_reduce.py`) inline each live-in at
    its point of use, so a live-in occupies a register only across the expression that reads it
    -- applying T1's bound to them would refuse kernels that work (`cat_83` reads 4 608 live-in
    elements and needs nothing like 4 608 registers).

    **Upper bound by construction (D26).** No liveness *ordering* analysis is performed --
    every value that could still be needed is counted as live, because an estimator that gates
    refusal must not be able to under-report.

    `interleaved_stores` reflects an emitter that stores each output the moment it is produced
    (T3, `emit_reduce.py`). Such a value is written and dies immediately, so only values that a
    *later* assignment reads have to persist. Without it the bound counts all of them, which for
    a `[9,256]` T3 output is 2 304 and refuses a kernel that needs almost no registers. With an
    emitter that appends all stores at the end, the un-interleaved count is the honest one -- and
    it correctly reported that those kernels were spilling before the interleaving landed.
    """
    later_use: dict[tuple, int] = {}
    for i, a in enumerate(sched.assigns):
        for t in a.terms:
            for f in t.factors:
                later_use[(f[0], f[1])] = i
        if a.source is not None:
            later_use[a.source] = i

    out_bufs = set(getattr(sched.spec, "live_out", ()))
    persist = 0
    for i, a in enumerate(sched.assigns):
        key = (a.target, a.index)
        dies_at_once = (interleaved_stores and a.target in out_bufs
                        and later_use.get(key, -1) <= i)
        if not dies_at_once:
            persist += 1

    chunked = sum(1 for a in sched.assigns if len(a.terms) > _CHUNK)
    widest = max((len(t.factors) for a in sched.assigns for t in a.terms), default=1)
    return persist + chunked + widest + (1 if interleaved_stores else 0)


def max_factors(sched) -> int:
    """Most multiplications in any single term."""
    return max((len(t.factors) for a in sched.assigns for t in a.terms), default=1)


def scalar_map_count(sched) -> int:
    """Scalar-map evaluations in the schedule.

    Each is worth at least an ulp of its own result: `rsqrt` in the emitted kernel and `rsqrt` in
    torch need not agree to the last bit, and that difference is *not* a reordering of a sum, so
    the ordering term does not cover it.
    """
    return sum(1 for a in sched.assigns if a.fn is not None)


def output_magnitude(sched, env: dict[str, torch.Tensor]) -> float:
    """Largest magnitude among the group's live-outs.

    The term-sum magnitude is the wrong scale for a group whose result is a scalar map: only
    assignments *with terms* contribute to it, so `invstd = rsqrt(var + eps)` had its bound
    computed from the magnitude of the variance while the error was measured on its inverse
    square root. When `var` is small that inverts to something much larger, and the bound was
    correspondingly too small -- g11 exceeded it by 1.008x while its twin g7 passed.
    """
    worst = 0.0
    for buf in sched.spec.live_out:
        if buf in env:
            worst = max(worst, float(env[buf].abs().max()))
    return worst


def term_magnitudes(sched, env: dict[str, torch.Tensor],
                    gathers: dict[str, str] | None = None) -> list[tuple[float, int, int]]:
    """`(magnitude, assign index, term index)` for every term, largest first.

    Used to choose *where* to plant a fault. Naive selection does not work: the first multi-term
    assignment in the Wigner chain sums `-sin(0.g)` and `+cos(0.g)`, and the sine term is exactly
    zero, so deleting it changes nothing and the resulting test passes vacuously. A fault has to
    be planted somewhere it can actually be observed, and that is a measurable property.
    """
    gathers = gathers or {}
    out: list[tuple[float, int, int]] = []
    extent = getattr(sched, "extent", 0)
    for ai, a in enumerate(sched.assigns):
        window = getattr(a, "ch_range", None) or ((0, extent) if extent else None)
        for ti, term in enumerate(a.terms):
            piece = None
            for factor in term.factors:
                tensor, has_ch = _factor_tensor(env, factor[0], factor[1], window)
                if factor[0] in gathers:
                    tensor = tensor[env[gathers[factor[0]]].long()]
                if piece is not None and piece.dim() != tensor.dim():
                    if piece.dim() < tensor.dim():
                        piece = piece.unsqueeze(-1)
                    else:
                        tensor = tensor.unsqueeze(-1)
                piece = tensor if piece is None else piece * tensor
            if piece is not None:
                out.append((float(piece.max()) * abs(term.coeff), ai, ti))
    out.sort(reverse=True)
    return out


def _scatter_fan_in(env: dict[str, torch.Tensor], scatter: str) -> int:
    """Largest number of segment elements accumulating into one output element."""
    idx = env[scatter].long().reshape(-1)
    return int(torch.bincount(idx).max()) if idx.numel() else 1


def ordering_bound(sched, env: dict[str, torch.Tensor],
                   gathers: dict[str, str] | None = None,
                   scatter: str | None = None) -> float:
    """The largest difference re-associating this schedule's arithmetic can produce.

    `2 * (depth - 1 + factors) * eps * max SUM|terms|`. The `factors` term is not decoration: a
    depth-1 assignment has nothing to *sum* but still multiplies, and `coeff * (a * b)` may round
    differently from `(coeff * a) * b` or contract into an FMA. Without it the bound was exactly
    0.0 for single-term assignments and fired on a correct kernel differing by 1.11e-16.

    **`scatter` widens the bound by the fan-in, and must be passed for any scattering kernel.**
    A per-thread bound is the wrong bound for a scatter-add: the output element is a sum over
    every segment element that maps to it -- 216 nodes into one scalar for `E_105`, ~45 edges per
    node for `scatter_100` -- and CUDA atomics complete in nondeterministic order, so that outer
    sum is reordered too, differently on every run. Omitting this made both scatter kernels
    "fail" at 1.11e-15 and 3.55e-15 against ~1.2e-16 bounds: near machine precision, which is the
    signature of a bound that is too tight rather than a kernel that is wrong.
    """
    depth = reduction_depth(sched)
    fan_in = _scatter_fan_in(env, scatter) if scatter is not None else 1
    # the outer accumulation contributes fan_in-1 further additions, each over a magnitude of at
    # most the per-thread total, so both the depth and the magnitude scale by the fan-in
    total_depth = depth * fan_in
    # The scale is whichever is larger: the terms being summed, or the values actually produced.
    # A scalar map can make the output far larger than any term feeding it.
    magnitude = max(magnitude_sum(sched, env, gathers), output_magnitude(sched, env)) * fan_in
    ops = total_depth - 1 + max_factors(sched) + scalar_map_count(sched)
    return 2.0 * ops * EPS * magnitude


def assert_within_bound(name: str, got: torch.Tensor, want: torch.Tensor,
                        bound: float, exact: bool = False) -> float:
    """The assertion every emitted kernel gets. Returns the measured error.

    `exact=True` additionally demands bit-equality -- the stronger T1 claim, which is allowed to
    fail loudly if the reference's reduction order ever changes, rather than being silently
    absorbed by the bound.
    """
    err = float((got - want).abs().max())
    if exact and not torch.equal(got, want):
        raise AssertionError(
            f"{name}: T1 kernel is no longer bit-exact (max abs {err:.3e}). The schedule and the "
            f"reference no longer sum in the same order; that is a finding, not a tolerance to "
            f"widen.")
    if err > bound:
        raise AssertionError(
            f"{name}: max abs error {err:.3e} exceeds the ordering bound {bound:.3e}. "
            f"Reordering cannot explain this; it is an arithmetic difference.")
    return err


__all__ = ["EPS", "reduction_depth", "max_factors", "scalar_map_count",
           "inlined_live_upper_bound",
           "output_magnitude", "magnitude_sum", "ordering_bound", "term_magnitudes",
           "assert_within_bound"]
