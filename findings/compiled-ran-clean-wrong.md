# Compiled, ran clean, O(1) wrong: `invar_101` and its two cousins

**Status: stands.** My bug, caught before it reached a measurement — and the interesting part is
*why* it was caught here when the same failure class survived for years in two other codebases.

## The defect

`invar_101` computes the readout's per-`l` invariants. In the IR it is one op with three paths
writing **disjoint slices** of a 384-wide output channel axis, each reading a different `m`-range
of the same buffer:

```
+1  mi,m->i    in=[m 0:1, i full]  out=i 0:128      # l=0
+1  mi,mi->i   in=[m 1:4, i full]  out=i 128:256    # l=1
+1  mi,mi->i   in=[m 4:9, i full]  out=i 256:384    # l=2
```

The T2 emitter put the channel axis on threads and treated every channel index as the thread's
own `c`. For the second path, thread `c = 200` should read input channel `200 − 128 = 72` and
write output 200. It read input 200 instead — a channel that path does not own.

The kernel compiled without a warning, launched without an error, produced finite numbers of
plausible magnitude, and was wrong by **4.47e+00** in relative terms. The same defect in
`conv2_95` measured 4.03e-01.

## Why nothing else would have caught it

Every structural check in the suite passed throughout:

* the IR type checker — the *program* was well-typed; the bug was in lowering, not in the IR;
* `assert_closed` — the vocabulary was never left;
* the acyclicity guard — the partition was fine;
* the register-budget precondition — the group was legitimately T2;
* compilation — CuTe DSL had no complaint to make; the indices were in range, just wrong.

An equivariance or invariance property test would not have caught it either: a readout that
mixes up which channels it squares is still invariant under rotation.

What caught it was the per-kernel ordering bound (D25). The kernel was checked against
`2·(depth−1+factors)·eps·max Σ|terms|` ≈ 9.86e-15 and missed by fourteen orders of magnitude.

## The three-codebase table

The same failure class — code that compiles, runs, and returns a confidently wrong number —
appears three times in this project's blast radius, with very different lifetimes:

| codebase | defect | lifetime before detection | what caught it |
|---|---|---|---|
| **fairchem** | `Safeacos.forward` saves a `clamp`ed tensor under no-grad, so the second derivative loses its dependence on `x` | shipped; present in a released model family, found during Phase 0 recon | a *deliberately non-invariant* probe scalar `uᵀW(pos)v` in FP64. The obvious probes (`‖Wv‖²`, `Σ W²`) are invariant and return a vacuous ~1e-15 |
| **cuEquivariance** | `indexed_linear` does not accumulate paths that share an output segment | released in 0.11.0; the backend is silently selectable and raises nothing | a hand-written FP64 transcription of the descriptor's own semantics, run as a three-way cross-check against `fused_tp` and `naive` |
| **zippel (mine)** | T2 emitter ignored channel-slice offsets | **~40 minutes**, and never reached a benchmark | the bound the emitter ships with every kernel |

The pattern in the first two columns is identical. The difference is entirely in the third and
fourth: both upstream defects needed *someone to go looking with a purpose-built oracle*, and
both survived release because the natural checks are vacuous against them. Mine was caught by a
check that runs unbidden on every kernel, every time.

That is not a claim to have been more careful. I wrote the bug, and I wrote it in the same week
as four others. It is a claim about where the check sits: an oracle you must remember to
construct protects the code you thought to doubt, while a bound attached to the artifact protects
the code you did not.

## What it cost, and what it would have cost

It cost about forty minutes and no wrong number left the machine. Had it survived to Phase 3, the
readout invariants feed the energy, so every wall-clock and memory figure for the fused forward
would have been measured on a kernel computing the wrong energy — and the numbers would have
looked *fine*: plausible magnitudes, stable across runs, faster than eager. The failure mode this
program names as its only real one is "a fabricated or gamed positive result", and a
silently-wrong fused kernel that benchmarks well is exactly that, arrived at by accident rather
than intent.

## Consequence

`tests/test_planted_faults.py` now plants this fault deliberately, alongside a dropped term, a
transposed index, a missing barrier, a wrong `SEGMENT`, and a wrong `REDUCTION_DEPTH`, and
asserts each is rejected. A check that has never fired is indistinguishable from a check that
cannot fire, and until that battery existed the suite only demonstrated that the bound *accepts*
correct kernels.

Writing the battery immediately exposed a second-order version of the same problem: my first
attempt planted a dropped term in the first multi-term assignment of the Wigner chain, which sums
`−sin(0·γ)` and `+cos(0·γ)`. The sine term is exactly zero, deleting it changes nothing, and the
test passed while proving nothing. Fault selection is now by *measured* term magnitude
(`codegen.bounds.term_magnitudes`) — a falsification test can be vacuous in precisely the way the
assertion it is meant to validate can be.
