# position→E is rational-plus-rsqrt: sin and cos are never used

Vocabulary v1.1 declares eight `scalar_map` functions:

```
exp, sigmoid, silu, rsqrt, reciprocal, sin, cos, poly_envelope
```

`sin` and `cos` were added at Phase 1 kickoff as the "Wigner delta" — the expectation being
that building the edge-frame rotation would need trigonometry. **It does not.** Measured on
the assembled programs (`tests/test_ir_block.py::test_derived_programs_are_closed_and_never_use_sin_or_cos`):

| program | scalar functions actually used |
|---|---|
| fwd | `exp, poly_envelope, rsqrt, sigmoid, silu` |
| force | + `reciprocal` |
| dbwd | + `reciprocal` |

**Neither `sin` nor `cos` appears in any of the three.** The effective vocabulary is six
functions, not eight.

## Why

Two independent reasons, and both have to hold:

1. **The forward never forms an angle.** The rational rewrite (DECISIONS.md D4) gives
   `cos kβ = T_k(ŷ)` and `sin kβ = r·U_{k-1}(ŷ)` as Chebyshev polynomials, and `cos kα`,
   `sin kα` by de Moivre from `ẑ·rsqrt(s2)` and `x̂·rsqrt(s2)`. Every Wigner-D entry is a
   polynomial in `(x̂, ŷ, ẑ, rsqrt(s2))`. The roll angle γ is position-independent, so its
   harmonics are per-edge input constants rather than computed trigonometry.

2. **No derivative rule introduces them.** `exp' = y`, `sigmoid' = y − y²`,
   `silu' = s + xs − xs·s`, `reciprocal' = −y²`, `poly_envelope'` increments an order, and
   `rsqrt' = −½·y·reciprocal(x)` — which is the only rule that adds a *new* function, and it
   adds `reciprocal`, not trigonometry. Since `sin`/`cos` are absent from the forward and no
   rule can create them, they are absent from `force` and `dbwd` too.

## Why it matters

It is a **representability** result, not an optimisation: the entire energy→force→double-backward
path for this block is a rational function of the atomic positions, with `rsqrt` as the sole
non-polynomial primitive. That is a stronger statement than "the vocabulary is closed" — it says
what kind of object the computation *is*.

Three consequences:

- **Verification.** `rsqrt` produces a square root of a rational — the same kind of algebraic
  number as the Clebsch-Gordan coefficients. So the whole path lives in one real algebraic
  extension `Q(sqrt(...))`, rather than the transcendental extension that `sin`/`cos`/`acos`/
  `atan2` would generate. Field-exact identity testing is therefore *conceivably* in reach,
  where with trigonometry it would not be. See `findings/pit-exactness.md`.
- **Codegen.** A rational-plus-`rsqrt` kernel needs no transcendental unit beyond the hardware
  reciprocal-square-root instruction, which on SM90 is a fast-path intrinsic.
- **The declared set was over-specified.** Recording this rather than quietly leaving two unused
  entries in the vocabulary: the closure test exists to catch the set growing, and it should
  equally catch the set being larger than the evidence requires.

**Caveat on scope.** This is measured for *this block* at lmax = 2 (and the lmax = 4 anchor
forward). It is not a claim about equivariant architectures in general — an architecture that
takes spherical harmonics of an angle directly, or uses an S² grid activation rather than a
gate, could reintroduce trigonometry. `sin`/`cos` are deliberately left in v1.1 rather than
removed, so that such a case would be expressible rather than silently rejected.
