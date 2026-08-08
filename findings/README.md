# findings/

Self-findings and upstream findings from M1, kept whether or not they survived. An entry is
written when a claim reached a report, not when it turns out to be right.

## Status taxonomy

| status | meaning |
|---|---|
| **(none)** | Stands as written. |
| **REFUTED** | The claim itself is false. The entry stays because it was reported; the mechanism of the error is stated at the top. |
| **MECHANISM CORRECTED** | The *decision* the finding drove is upheld, but the reason given for it was wrong. Both halves are load-bearing: acting rightly for a wrong reason generalises wrongly, and the next decision that reuses the reason will be the one that breaks. |

`MECHANISM CORRECTED` exists because D24 will not be the last of its kind. Rejecting a dense-MMA
design was right; "it pays quadratically for the block-diagonal zeros" was not why, and the
measurement that showed so (padding is *cheaper* than not padding) would have been read as
confirming the original reasoning if the two had not been separated.

## Methodology lessons

Recurring failure modes in *how measurements were made*, as distinct from what they found. Each
was learned the expensive way here.

| lesson | where it bit |
|---|---|
| **An instrument must not pre-pay the cost it measures.** The dbwd emission preflight ranked groups by `n_terms`, which requires building every schedule to learn which are large — so it spent fifteen minutes paying the exact cost it was built to quantify, and produced nothing. Replaced by an analytic O(paths) size predictor. | `bench/dbwd_scale_preflight.py` → `bench/schedule_scaling.py` |
| **A good fit is not an explanation.** A constant fitted to reproduce known traffic fits it well under either physical story; the 0.4 % residual distinguished nothing, and only the device's capacity and an independent bandwidth measurement did. | `dcgm-bandwidth-constant.md` |
| **A tight correlation is not a mechanism.** Schedule cost fitted index-space volume at R² 0.976, and the inferred cause (walking the dense index space) was wrong: 97 % of it was a quadratic liveness scan, correlated with volume only because bigger spaces make more assignments. Profile before attributing. | D31 |
| **A falsification test can be vacuous in the way the assertion it validates can be.** The first planted dropped-term fault deleted `−sin(0·γ)`, which is exactly zero, so the test passed while proving nothing. | `compiled-ran-clean-wrong.md` |
| **A literal grep cannot find a name built by interpolation.** I reported "`CUTE_DSL_CACHE_DIR` appears zero times in the package" as a measurement; the name is constructed as `f"{prefix}_CACHE_DIR"` and my search was structurally incapable of finding it. Absence of evidence from a search is evidence of absence only if the search could have found the thing. | `cute-dsl-cache-dir-is-a-noop.md` |
| **An estimator that gates a decision must be conservative or carry a falsification test.** `peak_live_values` under-reported by two orders of magnitude and its guard failed by *not raising*. | D26 |

## Entries

| entry | status |
|---|---|
| `vocabulary-shrink.md` | stands |
| `pit-exactness.md` | stands |
| `self-halved-derivative.md` | stands — my bug, D21 |
| `fusion-partition-cycles.md` | stands — my bug, corrected a Gate 1 number |
| `cueq-indexed-linear-trigger.md` | stands — upstream, DRAFT issue awaiting review |
| `cueq-math-dtype.md` | **REFUTED** — my Gate 0 claim, disproved by controlled experiment |
| `dense-mma-density-argument.md` | **MECHANISM CORRECTED** — D22 upheld, its stated cause replaced (D24) |
| `traffic-model-calibration.md` | stands — what the traffic model predicts, and what it does not |
| `dcgm-bandwidth-constant.md` | **MECHANISM CORRECTED** — the fit stays in service; it is not a bandwidth |
| `compiled-ran-clean-wrong.md` | stands — my `invar_101` bug, and the three-codebase table for its failure class |
| `keyed-by-identity.md` | stands — one bug class, four instances, four different detection mechanisms |
| `cute-dsl-cache-dir-is-a-noop.md` | **REFUTED** — the variable is real; `cute.compile()` disables the cache by design, and the upstream note was withdrawn unfiled |

## Methodology ledger — two entries from the `conv1_90` diagnosis (2026-08-08)

**1. A split verdict is a verdict.** D42 claimed uncoalesced weight access explained `conv1_90`'s
714.8 ms, on a static model predicting 651 ms. `ncu` says the access pattern was **real**
(20.31 → 3.61 sectors/request) and that fixing it bought **1.228×**, not the ~4.6× the model
implied. The right verdict is neither "confirmed" nor "refuted" but **"true as a description,
false as an account of the cost"** — and the two halves have different consequences: the
description earned a ratified emission rule, the cost account earned a retraction. A taxonomy with
only CONFIRMED and REFUTED would have forced one of those to be wrong.

**2. Being wrong about why you were wrong.** D53 correctly refuted its own traffic model (it
predicted 5 611.8 ms for a kernel measured at 714.8) and then named the cause: the weights are
1.25 MB in a 60 MiB L2, so they were never DRAM traffic. `ncu` measured the L2 hit rate at
**57.4 %**, not the >80 % that claim requires. The refutation was right; **the stated mechanism was
wrong**, and the real cause was the *other* error in the same list — counting textual factor
occurrences instead of issued loads (2.24 TB actual against 22.4 TB implied, a 10× over-count).

The generalisable part: **a retraction is itself a causal claim and inherits the evidence law.**
D53's mechanism was asserted at the same moment its subject was being retracted, and the relief of
having caught an error is exactly when the next one gets waved through. What caught it was a row
in a pre-registered table written *for that purpose* before any counter existed — not vigilance.
