# DRAFT — upstream issue for NVIDIA/cuEquivariance. **Not filed.** For review before submission.

---

**Title:** `method="indexed_linear"` gives wrong results on multi-path `SegmentedPolynomial`s
(disagrees with `fused_tp` and `naive`)

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

### Scope — it tracks path count, not size

Sweeping a ladder of descriptors (all FP64, all with a one-row weight table plus
`input_indices`):

| descriptor | paths | `fused_tp` | `indexed_linear` | `naive` |
|---|---|---|---|---|
| `1x0 -> 1x0`, m_max=0 | 1 | 0.0 | **0.0** | 0.0 |
| `2x0 -> 3x0`, m_max=0 | 1 | 0.0 | **1.7e-08** | 7.0e-17 |
| `2x1 -> 2x1`, m_max=1 | 5 | 0.0 | **8.3e-01** | 1.2e-16 |
| `2x0+2x1`, m_max=1 | 5 | 0.0 | **4.3e-01** | 0.0 |
| `2x0+2x1+2x2`, m_max=2 | 9 | 1.4e-16 | **6.2e-01** | 1.4e-16 |
| `256x0+256x1+256x2 -> 128x0+128x1+128x2`, m_max=2 | 9 | 7.7e-16 | **6.0e-01** | 0.0 |

`indexed_linear` is correct on both single-path descriptors and wrong on every multi-path one,
independently of size. At m ≥ 1 an eSCN descriptor introduces three things simultaneously —
multiple paths, **weight-segment reuse** (one weight segment is referenced by two paths with
different in/out segment pairs) and a **negative path coefficient** — so we cannot say from the
outside which of the three is the trigger.

### Secondary observation

`indexed_linear` warns `` `indexed_linear` does not support explicit `math_dtype`. This will be
ignored. `` and the `2x0 -> 3x0` row above is accurate only to ~1.7e-08, i.e. FP32, even though
both operands and `math_dtype` are FP64. Silently dropping to single precision is worth
documenting even independently of the correctness issue.

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

1. Narrow the trigger: build a hand-made `SegmentedTensorProduct` that isolates multi-path from
   weight-reuse from negative-coefficient, instead of relying on `escn_tp_compact` which
   introduces all three at once. That would make the report actionable rather than descriptive.
2. Re-check against the latest cuequivariance release, and on x86 as well as aarch64, to rule out
   a platform-specific build issue.
3. Search existing issues for duplicates.
4. Confirm `escn_tp_compact` is a supported entry point — it lives under
   `group_theory.experimental`, is not exported from `cue.descriptors`, and its own tests only
   construct descriptors without executing them. If it is unsupported, reframe the report around a
   descriptor built from public API instead, so it cannot be dismissed on that basis.
