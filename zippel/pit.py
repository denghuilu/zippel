"""Randomized polynomial-identity testing on the multilinear core.

A `segmented_contraction` is multilinear in its operands, so any rewrite of one is an
identity between two multilinear forms. Schwartz-Zippel says: evaluate both at a random
point; if they differ, they are different polynomials (certainly), and if they agree, they
are equal with high probability -- the failure probability is bounded by d/|S| for degree d
over a sample set S.

This is the *mechanism*, established on one nontrivial rewrite. Its honest limitation is
recorded in findings/pit-exactness.md: over floating point this is a numerical identity test
with a tolerance, not a field-exact one.
"""

from __future__ import annotations

import torch

from zippel.interp import run
from zippel.ir import Program


def pit_equal(prog_a: Program, out_a: str, prog_b: Program, out_b: str,
              make_inputs, sizes: dict[str, int], trials: int = 8,
              tol: float = 1e-11, seed: int = 0) -> tuple[bool, float]:
    """Are two programs the same polynomial? Returns (verdict, worst relative difference).

    Each trial draws a fresh random point. A single disagreement is conclusive (the forms
    differ); agreement across trials is probabilistic evidence of equality.
    """
    worst = 0.0
    for t in range(trials):
        inputs = make_inputs(seed + t)
        a = run(prog_a, inputs, sizes)[out_a]
        b = run(prog_b, inputs, sizes)[out_b]
        scale = b.abs().max().clamp_min(1e-30)
        worst = max(worst, ((a - b).abs().max() / scale).item())
        if worst > tol:
            return False, worst
    return True, worst
