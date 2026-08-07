# DRAFT — upstream issue for NVIDIA/cuEquivariance. **Not filed.** For review before submission.

---

**Title:** `method="indexed_linear"` does not accumulate paths that share an output segment

**Version:** cuequivariance 0.11.0, cuequivariance-torch 0.11.0, cuequivariance-ops-torch-cu13
0.11.0

### Summary

For a `SegmentedPolynomial` whose `SegmentedTensorProduct` has **more than one path**,
`method="indexed_linear"` returns a different result from `method="fused_tp"` and
`method="naive"` on identical inputs. `fused_tp` and `naive` agree with each other and with a
hand-written transcription of the descriptor semantics; `indexed_linear` does not. No error or
warning is raised, so a user who selects `indexed_linear` for its (much better) memory behaviour
silently gets wrong numbers.

Relative errors up to **0.83** in FP64.

### Reproducer

```python
import torch
import cuequivariance as cue
import cuequivariance_torch as cuet
from cuequivariance.group_theory.experimental.escn import escn_tp_compact

DT, DEV = torch.float64, "cuda"

def segment_slices(operand):
    out, off = [], 0
    for seg in operand.segments:
        size = 1
        for d in seg:
            size *= d
        out.append((off, off + size, tuple(seg)))
        off += size
    return out

def hand_oracle(poly, weights, x):
    """Literal transcription of `uv,u,v`: out[o][v] += c * sum_u w[wseg][u,v] * x[i][u]."""
    (op, stp), = poly.operations
    w_segs, in_segs, out_segs = (segment_slices(o) for o in stp.operands)
    out = torch.zeros(x.shape[0], out_segs[-1][1], device=x.device, dtype=x.dtype)
    for path in stp.paths:
        wi, xi, oi = (int(k) for k in path.indices)
        c = float(path.coefficients)
        ws, we, (u, v) = w_segs[wi]
        xs, xe, _ = in_segs[xi]
        os_, oe, _ = out_segs[oi]
        blk = weights[:, ws:we].reshape(-1, u, v).expand(x.shape[0], u, v)
        out[:, os_:oe] += c * torch.einsum("buv,bu->bv", blk, x[:, xs:xe])
    return out

# minimal failing case: one l=1 irrep, m_max=1 -> 5 paths, 12 weights, 6 inputs
poly = escn_tp_compact(cue.Irreps("SO3", "2x1"), cue.Irreps("SO3", "2x1"), m_max=1)

torch.manual_seed(0)
w = torch.randn(1, poly.inputs[0].size, device=DEV, dtype=DT)
x = torch.randn(8, poly.inputs[1].size, device=DEV, dtype=DT)
idx = torch.zeros(8, dtype=torch.long, device=DEV)
oracle = hand_oracle(poly, w, x)

for method in ("fused_tp", "indexed_linear", "naive"):
    mod = cuet.SegmentedPolynomial(poly, method=method, math_dtype=DT).cuda()
    got = mod([w, x], input_indices={0: idx})[0]
    rel = ((got - oracle).abs().max() / oracle.abs().max()).item()
    print(f"{method:16s} rel err vs descriptor semantics = {rel:.3e}")
```

Output on our machine:

```
fused_tp         rel err vs descriptor semantics = 0.000e+00
indexed_linear   rel err vs descriptor semantics = 8.298e-01
naive            rel err vs descriptor semantics = 1.173e-16
```

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

### Secondary observation

The three backends disagree on what `math_dtype` means when it conflicts with the operand dtype.
With FP64 operands on a descriptor `indexed_linear` handles correctly:

| method | `math_dtype=float32` with FP64 operands |
|---|---|
| `fused_tp` | raises `ValueError: Fused TP does not support float32 math_dtype with float64 inputs` |
| `naive` | honours it — result accurate to 1.65e-07 |
| `indexed_linear` | warns that `math_dtype` is ignored, then computes in FP64 (rel err 0.00e+00) |

So `indexed_linear` cannot be asked for reduced-precision math: the request is accepted, warned
about and dropped, and a user tuning for speed silently gets FP64. Whichever behaviour is
intended, three different ones for a single argument seems worth reconciling. This is
independent of the correctness bug above.

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

1. ~~Narrow the trigger~~ — **done**; it is a shared output segment, and the report above is
   now written against hand-built descriptors rather than `escn_tp_compact`, so item 4 below
   matters much less.
2. Re-check against the latest cuequivariance release, and on x86 as well as aarch64, to rule out
   a platform-specific build issue.
3. Search existing issues for duplicates.
4. Confirm `escn_tp_compact` is a supported entry point — it lives under
   `group_theory.experimental`, is not exported from `cue.descriptors`, and its own tests only
   construct descriptors without executing them. If it is unsupported, reframe the report around a
   descriptor built from public API instead, so it cannot be dismissed on that basis.
