"""The minimal reproducer for the `indexed_linear` accumulation bug.

Two paths, unique weights, +1 coefficients, homogeneous segment sizes. The ONLY difference
between the failing and passing descriptor is whether the two paths write to the same output
segment. Nothing here imports `escn_tp_compact`, so the report does not depend on an unexported
experimental entry point.

Needs the cueq-abi-test env (torch 2.11): cuequivariance-ops-torch-cu13 0.11.0 does not load
against torch 2.12/2.13.

    /iopsstor/scratch/cscs/dlu/envs/cueq-abi-test/bin/python bench/cueq_min_repro.py
"""

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
