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


def magnitude_sum(sched, env: dict[str, torch.Tensor]) -> float:
    """max over output elements of SUM |term|, evaluated on the real inputs.

    This is the data-dependent half of the bound. Terms are evaluated with absolute values, so
    cancellation makes the result larger than the true output -- which is the point.
    """
    worst = 0.0
    for a in sched.assigns:
        if not a.terms:
            continue
        total: torch.Tensor | None = None
        for term in a.terms:
            extent = getattr(sched, "extent", 0)
            window = getattr(a, "ch_range", None) or ((0, extent) if extent else None)
            parts = [_factor_tensor(env, f[0], f[1], window) for f in term.factors]
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


def max_factors(sched) -> int:
    """Most multiplications in any single term."""
    return max((len(t.factors) for a in sched.assigns for t in a.terms), default=1)


def ordering_bound(sched, env: dict[str, torch.Tensor]) -> float:
    """The largest difference re-associating this schedule's arithmetic can produce.

    `2 * (depth - 1 + factors) * eps * max SUM|terms|`. The `factors` term is not decoration: a
    depth-1 assignment has nothing to *sum* but still multiplies, and `coeff * (a * b)` may round
    differently from `(coeff * a) * b` or contract into an FMA. Without it the bound was exactly
    0.0 for single-term assignments and fired on a correct kernel differing by 1.11e-16.
    """
    depth = reduction_depth(sched)
    return 2.0 * (depth - 1 + max_factors(sched)) * EPS * magnitude_sum(sched, env)


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


__all__ = ["EPS", "reduction_depth", "max_factors", "magnitude_sum", "ordering_bound",
           "assert_within_bound"]
