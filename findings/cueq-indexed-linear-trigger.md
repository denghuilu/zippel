# Thread (b): the `indexed_linear` trigger, isolated

**Trigger: two or more paths accumulating into the same output segment.** Nothing else.

## How it was narrowed

The Gate 0 triage could only say "`indexed_linear` is wrong on multi-path descriptors", because
`escn_tp_compact` introduces three properties at once at m ≥ 1. `bench/cueq_isolate.py` builds
`SegmentedTensorProduct`s by hand that vary one property at a time, against a hand oracle
transcribed from the descriptor's own path list.

**Round 1 refuted the obvious hypotheses.** All six cases were correct to ~1e-16:

| case | `indexed_linear` |
|---|---|
| 1 path, unique weight, +c | 9.85e-17 ✅ |
| 2 paths, unique weights, +c | 1.14e-16 ✅ |
| 2 paths, **reused** weight, +c | 9.85e-17 ✅ |
| 2 paths, unique weights, **−c** | 1.14e-16 ✅ |
| 2 paths, reused weight, −c | 9.85e-17 ✅ |
| 4 paths, reused weight, −c | 9.52e-17 ✅ |

So weight reuse is not the trigger, negative coefficients are not the trigger, and path count
alone is not the trigger — contradicting all three of the candidates named at Gate 0.

**Round 2 found it** by testing the property round 1 had failed to vary — in `escn_tp_compact`
both `W1·x(−m)` and `W2·x(+m)` land on the *same* output segment, whereas every round-1 case gave
each path its own:

| case | `indexed_linear` |
|---|---|
| 2 paths → **shared** output segment | **7.69e-01** ❌ |
| 2 paths → shared output, −c | **1.31e+00** ❌ |
| 2 paths, reused weight → shared output | **1.14e+00** ❌ |
| 4 paths, reused weight, −c → shared output | **7.34e-01** ❌ |
| 2 paths, heterogeneous segment sizes, distinct outputs | 1.64e-16 ✅ |

Every failure has a shared output segment; every correct case does not. Heterogeneous segment
sizes — the other structural difference from the hand-built cases — are fine.

`fused_tp` and `naive` are correct on all eleven cases, at 0.00e+00.

## Minimal reproducer

**2 paths, unique weights, coefficient +1 each, both writing one output segment** — no weight
reuse, no negative coefficient, no size heterogeneity. Relative error **0.769**.

```python
d = cue.SegmentedTensorProduct.from_subscripts("uv,u,v")
d.add_segment(1, (4,)); d.add_segment(1, (4,))      # two input segments
d.add_segment(2, (3,))                               # ONE output segment
d.add_segment(0, (4, 3)); d.add_segment(0, (4, 3))  # two distinct weights
d.add_path(0, 0, 0, c=1.0)
d.add_path(1, 1, 0, c=1.0)                           # <- both paths -> output segment 0
poly = cue.SegmentedPolynomial.eval_last_operand(d)
```

## Likely mechanism (inferred, not confirmed from source)

The symptom is consistent with the backend **not accumulating across paths that share an output
segment** — writing rather than adding, or executing only one path per output. The relative
errors (0.73–1.31) are the size you would expect from dropping one of two comparable
contributions, rather than from a precision or indexing fault.

We have not read the `indexed_linear` kernel source, so this is the shape of the bug as seen from
outside, not a diagnosis.

## Why it matters here

`indexed_linear` is the only backend that does not densify a shared weight table
(0.25 GiB vs 24.18 GiB on the eSEN conv1 shape, ~97×). Every eSEN SO(2) block has two paths per
`|m|` landing on each output segment — that *is* the complex product — so this bug lands exactly
on the descriptor shape the architecture needs, and it is why B3 contributes no valid wall-clock
number.

Draft upstream report: `docs/upstream_cuequivariance_issue_DRAFT.md` — **not filed**.
