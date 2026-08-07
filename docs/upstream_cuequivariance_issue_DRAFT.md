# DRAFT — upstream issue for NVIDIA/cuEquivariance. **Not filed.** For review before submission.

---

**Title:** `method="indexed_linear"` does not accumulate paths that share an output segment

**Version:** cuequivariance 0.11.0, cuequivariance-torch 0.11.0, cuequivariance-ops-torch-cu13
0.11.0

### Summary

When two or more paths of a `SegmentedTensorProduct` accumulate into the **same output
segment**, `method="indexed_linear"` returns a different result from `method="fused_tp"` and
`method="naive"` on identical inputs. `fused_tp` and `naive` agree with each other and with a
hand-written transcription of the descriptor semantics; `indexed_linear` does not. No error or
warning is raised, so a user who selects `indexed_linear` for its (much better) memory behaviour
silently gets wrong numbers.

Relative errors up to **1.31** in FP64. Path count, weight reuse, negative coefficients and heterogeneous segment sizes are each individually fine — the shared output segment is the trigger.

### Reproducer

Self-contained; no `escn_tp_compact` and no experimental entry point. The two descriptors below
are identical except for whether the two paths write to the **same** output segment.

```python
import torch
import cuequivariance as cue
import cuequivariance_torch as cuet

DT, DEV = torch.float64, "cuda"
U, V = 4, 3


def make(shared_out: bool):
    """Two paths, unique weights, +1 coefficients. Only the output segments differ."""
    d = cue.SegmentedTensorProduct.from_subscripts("uv,u,v")
    d.add_segment(1, (U,)); d.add_segment(1, (U,))          # two input segments
    for _ in range(1 if shared_out else 2):
        d.add_segment(2, (V,))                              # one shared, or one each
    d.add_segment(0, (U, V)); d.add_segment(0, (U, V))      # a distinct weight per path
    d.add_path(0, 0, 0, c=1.0)
    d.add_path(1, 1, 0 if shared_out else 1, c=1.0)
    return cue.SegmentedPolynomial.eval_last_operand(d)


def oracle(w, x, shared_out):
    """out[oseg] += w[path] @ x[iseg], straight from the descriptor's own semantics."""
    n = x.shape[0]
    res = torch.zeros(n, V if shared_out else 2 * V, device=DEV, dtype=DT)
    for p in range(2):
        blk = w[:, p * U * V:(p + 1) * U * V].reshape(-1, U, V).expand(n, U, V)
        o = 0 if shared_out else p * V
        res[:, o:o + V] += torch.einsum("buv,bu->bv", blk, x[:, p * U:(p + 1) * U])
    return res


torch.manual_seed(0)
w = torch.randn(1, 2 * U * V, device=DEV, dtype=DT)
x = torch.randn(8, 2 * U, device=DEV, dtype=DT)
idx = torch.zeros(8, dtype=torch.long, device=DEV)

for shared_out in (True, False):
    poly = make(shared_out)
    ref = oracle(w, x, shared_out)
    label = "SHARED output segment" if shared_out else "distinct output segments"
    print(f"\n2 paths, unique weights, +1 coefficients, {label}:")
    for method in ("fused_tp", "indexed_linear", "naive"):
        mod = cuet.SegmentedPolynomial(poly, method=method, math_dtype=DT).cuda()
        got = mod([w, x], input_indices={0: idx})[0]
        rel = ((got - ref).abs().max() / ref.abs().max()).item()
        print(f"  {method:16s} rel err = {rel:.3e}")
```

Output on our machine:

```
2 paths, unique weights, +1 coefficients, SHARED output segment:
  fused_tp         rel err = 0.000e+00
  indexed_linear   rel err = 7.690e-01
  naive            rel err = 0.000e+00

2 paths, unique weights, +1 coefficients, distinct output segments:
  fused_tp         rel err = 0.000e+00
  indexed_linear   rel err = 1.144e-16
  naive            rel err = 0.000e+00
```

Flipping one flag — which output segment the second path writes to — moves `indexed_linear` from
machine precision to 77 % relative error, with `fused_tp` and `naive` exact in both.

### Scope — the trigger is a shared output segment, nothing else

Varying one structural property at a time, with hand-built `SegmentedTensorProduct`s so that
nothing else differs (all FP64, one-row weight table plus `input_indices`, `U=4`, `V=3`):

| case | paths | reused weight | negative coeff | shared output | `indexed_linear` |
|---|---|---|---|---|---|
| 1 path | 1 | no | no | – | 9.85e-17 ✅ |
| 2 paths, distinct outputs | 2 | no | no | no | 1.14e-16 ✅ |
| 2 paths, reused weight | 2 | **yes** | no | no | 9.85e-17 ✅ |
| 2 paths, negative coefficient | 2 | no | **yes** | no | 1.14e-16 ✅ |
| 2 paths, reused + negative | 2 | yes | yes | no | 9.85e-17 ✅ |
| 4 paths, reused + negative | 4 | yes | yes | no | 9.52e-17 ✅ |
| heterogeneous segment sizes | 2 | no | no | no | 1.64e-16 ✅ |
| **2 paths → one output segment** | 2 | no | no | **yes** | **7.69e-01** ❌ |
| 2 paths → one output, −c | 2 | no | yes | **yes** | **1.31e+00** ❌ |
| 2 paths, reused weight → one output | 2 | yes | yes | **yes** | **1.14e+00** ❌ |
| 4 paths, reused, −c → one output | 4 | yes | yes | **yes** | **7.34e-01** ❌ |

Every failure has two or more paths writing to the same output segment; every correct case does
not. Weight reuse, negative coefficients, path count alone, and heterogeneous segment sizes are
all ruled out. `fused_tp` and `naive` are correct on all eleven, at 0.00e+00.

The magnitudes (0.73–1.31 relative) are what dropping one of two comparable contributions would
give, which is consistent with the paths not being **accumulated** — written rather than added,
or only one executed per output. We have not read the kernel source, so that is the shape of the
symptom rather than a diagnosis.

### Why it matters

`escn_tp_compact` descends from eSCN, where the radial network emits a *per-edge* weight matrix,
so operand 0 is a batched input. For architectures that instead **share** the weights across edges
and vary only a per-edge diagonal gain (eSEN / the UMA `SO2_Convolution` family),
`fused_tp` materialises the weight operand to `[E, weight_size]` — 24 GB at 9 620 edges and OOM at
259 474 edges for the eSEN conv1 shape above. `indexed_linear` is the natural backend for that
access pattern (0.25 GB on the same case, ~97× smaller), so the correctness bug is exactly what
blocks the memory-viable path.

### Environment

torch 2.11.0+cu130, CUDA 13.0, NVIDIA GH200 (sm_90a), aarch64, Python 3.13,
cuequivariance{,-torch,-ops-torch-cu13} 0.11.0.

(Note: torch 2.11.0 rather than something newer because
`cuequivariance-ops-torch-cu13==0.11.0`'s extension does not load against torch 2.12/2.13 —
`undefined symbol: torch::Library::_def(c10::FunctionSchema&&, c10::OperatorName*, const std::vector<at::Tag>&, torch::_RegisterOrVerify)`.
Happy to file that separately if useful.)

---

## TODO before this is sendable

1. ~~Narrow the trigger~~ — **done.** It is a shared output segment; path count, weight reuse,
   negative coefficients and heterogeneous segment sizes are each individually fine.
2. ~~Reframe around the public API~~ — **done.** The reproducer builds a
   `SegmentedTensorProduct` from `from_subscripts` / `add_segment` / `add_path` only, so the
   report no longer depends on `escn_tp_compact` being a supported entry point.
   `escn_tp_compact` now appears solely as motivation under "Why it matters".
3. Re-check against the latest cuequivariance release, and on x86 as well as aarch64, to rule out
   a platform-specific build issue.
4. Search existing issues for duplicates.

A `math_dtype` observation was drafted here and has been **withdrawn**: a controlled experiment
refuted the reading it rested on (`findings/cueq-math-dtype.md`). It is not part of this report.

**Status: awaiting Denghui's review. No further edits.**
