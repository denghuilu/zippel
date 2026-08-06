# PIT: what the randomized identity check does and does not establish

`zippel/pit.py` implements a Schwartz-Zippel-style randomized polynomial identity test and
applies it to one nontrivial rewrite of the multilinear core. That establishes the
**mechanism**: a `segmented_contraction` is multilinear in its operands, so a rewrite is an
identity between two multilinear forms, and evaluating both at random points either refutes
the identity outright or supports it with probability bounded by `d/|S|`.

## The limitation, stated plainly

Over floating point this is a **numerical** identity test with a tolerance, not a field-exact
one. A rewrite that is wrong by less than the tolerance passes. Schwartz-Zippel's guarantee
is a statement about exact arithmetic over a field; we are sampling in FP64 and comparing with
`1e-11`, so what we actually establish is "these two forms agree numerically at random points",
which is weaker.

Making it exact runs into the coefficients. The path coefficients here are **algebraic
numbers, not rationals**: Clebsch-Gordan-derived factors are square roots of rationals, and the
`1/sqrt(u)` and `1/sqrt(2u)` normalisations seen in the cuEquivariance descriptor are the same
kind of quantity. Exact verification would need arithmetic in the number field they generate --
`Q(sqrt(p_1), ..., sqrt(p_k))` for the primes appearing under the radicals -- rather than in
`Q` or a prime field, which is what a textbook PIT implementation assumes.

## Why the outlook is better than it looks

The one non-polynomial primitive on the position path is `rsqrt`, and `rsqrt` produces
**exactly the same kind of algebraic number** as the CG coefficients: a square root of a
rational. So the whole position→E computation lives in a single real algebraic extension
`Q(sqrt(...))` rather than needing transcendental closure. That is a materially better
starting point for field-exact verification than it would have been had the rotation kept
`sin`/`cos`/`acos`/`atan2`, which generate a transcendental extension and put exact
verification out of reach entirely.

Concretely, an exact checker would need to: identify the finitely many radicands appearing in
a program, work in the compositum of the corresponding quadratic extensions, and run PIT over
that field (or over a prime field after a suitable reduction that preserves the extension).
None of that is attempted here.

**Status: documented, not solved.** This is future work for the P1 claim, deliberately out of
Phase 1 scope per the work order.
