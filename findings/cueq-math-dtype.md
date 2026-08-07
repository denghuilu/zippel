# REFUTED — thread (c): `indexed_linear` does not silently compute in FP32

> **Status: REFUTED.** This entry records a claim *I* made at Gate 0 and then disproved. It is
> kept because the claim reached a report, not because it stands.
>
> **What I claimed:** `indexed_linear` silently downgrades FP64 operands to FP32 math.
> **Mechanism of the error:** I inferred a cause from a co-occurrence. Two facts were true — a
> `2x0 -> 3x0` escn case came out at 1.7e-08 with FP64 operands, and the backend warned that
> `math_dtype` would be ignored — and I joined them into a mechanism ("ignored `math_dtype`
> means it fell back to FP32") without varying anything. The residual and the warning had
> different causes: the case I measured was one the *accumulation* bug corrupts
> (`cueq-indexed-linear-trigger.md`), so precision was never the variable under test.
> **What is actually true:** on a descriptor `indexed_linear` handles correctly, it computes at
> operand precision exactly — 0.00e+00 in FP64 — and what it ignores is a request to *reduce*
> precision. The surviving observation is a three-way contract disagreement between backends,
> which is a documentation matter and not a correctness bug.
> **Consequence:** the `math_dtype` section was withdrawn from the upstream draft. No number in
> any table depended on it.
> **The general lesson:** an error bar plus a warning is a correlation. Diagnosing it as a
> mechanism requires an experiment where precision is the only thing that varies, on a case
> already known to be otherwise correct — which is exactly what `bench/cueq_math_dtype.py` does,
> and what I should have done before writing the claim down.

---

# Thread (c): `indexed_linear` silently ignores `math_dtype` — and my earlier reading of it was wrong

## What was claimed at Gate 0

During B3 triage, the `2x0 -> 3x0` escn case came out at 1.7e-08 with FP64 operands, and
`indexed_linear` emitted `UserWarning: indexed_linear does not support explicit math_dtype. This
will be ignored.` I inferred from the two together that the backend was **silently computing in
FP32**.

## What a controlled experiment shows

`bench/cueq_math_dtype.py` runs a descriptor `indexed_linear` handles *correctly* (one path,
unique weight, +1 coefficient — see `cueq-indexed-linear-trigger.md`), so precision is the only
variable. FP64 operands throughout, against an exact FP64 `einsum` reference:

| method | `math_dtype` | rel err | behaviour |
|---|---|---|---|
| `fused_tp` | float64 | 5.59e-16 | correct |
| `fused_tp` | float32 | — | **raises**: "Fused TP does not support float32 math_dtype with float64 inputs" |
| `indexed_linear` | float64 | **0.00e+00** | warns that `math_dtype` is ignored |
| `indexed_linear` | float32 | **0.00e+00** | warns; computes in FP64 anyway |
| `naive` | float64 | 5.59e-16 | correct |
| `naive` | float32 | 1.65e-07 | honours the request |

**The inference was wrong.** `indexed_linear` computes at *operand* precision — exactly, here —
and what it ignores is a request to **reduce** precision. It is not downgrading silently; it is
declining to downgrade, silently.

## The actual finding

A contract issue rather than a correctness one, in two parts:

1. **`indexed_linear` cannot be asked for reduced-precision math.** Passing
   `math_dtype=torch.float32` with FP64 operands is accepted, warned about, and ignored. A user
   tuning for speed gets FP64 arithmetic and no error — only a warning that is easy to filter out.
2. **The three backends disagree on the contract.** `fused_tp` *raises* on the same combination,
   `naive` *honours* it, `indexed_linear` *ignores* it. Whichever is intended, three behaviours
   for one argument is the reportable part.

## The residual that started this is still unexplained

The original 1.7e-08 on the escn `2x0 -> 3x0` case is *not* explained by precision. That
descriptor has a single output segment and one path, so it is also not the thread-(b) trigger.
It is left as an open loose end rather than attached to a hypothesis that this experiment
refutes — recording it as unexplained is more useful than the wrong cause.

## Status

Folded into `docs/upstream_cuequivariance_issue_DRAFT.md` as a secondary observation, clearly
separated from the correctness bug. **Not filed.**
