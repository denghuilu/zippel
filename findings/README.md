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
