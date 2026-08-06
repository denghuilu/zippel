"""B3 triage: is the 0.665 rel err OUR mapping bug, or a cuEquivariance bug?

Method. Build a *hand oracle* directly from the descriptor's own operand/path list, i.e.
a literal transcription of what a `SegmentedTensorProduct` with subscripts `uv,u,v` means:

    for each path (w_seg, in_seg, out_seg) with coefficient c:
        out[out_seg][v] += c * sum_u  weight[w_seg][u, v] * x[in_seg][u]

Nothing about eSEN, no layout guessing, no normalisation reasoning -- just the descriptor.
Then run each backend on the identical inputs and compare. That splits the outcome cleanly:

    fused_tp matches oracle, indexed_linear does not
        -> the two backends disagree about the same descriptor. cuEq bug.
    neither matches oracle
        -> our reading of the descriptor semantics is wrong. Our bug.
    both match at tiny sizes but diverge at eSEN sizes
        -> a size/shape-dependent bug; report the crossover.

A size ladder isolates structure from scale, and a layout sweep covers the two remaining
degrees of freedom the user flagged: cuEq's mul_ir vs ir_mul ordering *within* a segment,
and the (m, -m) real/imaginary pair order.

    $SPIR_CUEQ_PY bench/b3_triage.py
"""

from __future__ import annotations

import itertools
import json
import pathlib

import torch

DT = torch.float64
DEV = "cuda"

# (name, irreps_in, irreps_out, m_max) -- smallest first
CASES = [
    ("scalar 1x0->1x0 m=0", "1x0", "1x0", 0),
    ("scalar 2x0->3x0 m=0", "2x0", "3x0", 0),
    ("l=1 only 2x1->2x1 m=1", "2x1", "2x1", 1),
    ("l<=1 2x0+2x1 m=1", "2x0+2x1", "2x0+2x1", 1),
    ("l<=2 2x0+2x1+2x2 m=2", "2x0+2x1+2x2", "2x0+2x1+2x2", 2),
    ("eSEN conv1 shape", "256x0+256x1+256x2", "128x0+128x1+128x2", 2),
]
METHODS = ["fused_tp", "indexed_linear", "naive"]


def descriptor(irreps_in: str, irreps_out: str, m_max: int):
    import cuequivariance as cue
    from cuequivariance.group_theory.experimental.escn import escn_tp_compact

    return escn_tp_compact(cue.Irreps("SO3", irreps_in), cue.Irreps("SO3", irreps_out),
                           m_max=m_max)


def segment_slices(operand):
    """[(start, stop, shape), ...] for one operand's segments, in declaration order."""
    out, off = [], 0
    for seg in operand.segments:
        size = 1
        for d in seg:
            size *= d
        out.append((off, off + size, tuple(seg)))
        off += size
    return out


def hand_oracle(poly, weights: torch.Tensor, x: torch.Tensor,
                weight_row_major: bool = True) -> torch.Tensor:
    """Literal transcription of `uv,u,v` segmented-polynomial semantics.

    `weight_row_major=True` reads each (u, v) weight block as u-major (C order);
    False reads it v-major. That is the mul_ir / ir_mul degree of freedom.
    """
    (op, stp), = poly.operations
    del op
    w_segs = segment_slices(stp.operands[0])
    in_segs = segment_slices(stp.operands[1])
    out_segs = segment_slices(stp.operands[2])

    batch = x.shape[0]
    out = torch.zeros(batch, out_segs[-1][1], device=x.device, dtype=x.dtype)
    for path in stp.paths:
        wi, xi, oi = (int(k) for k in path.indices)
        c = float(path.coefficients)
        ws, we, wshape = w_segs[wi]
        xs, xe, _ = in_segs[xi]
        os_, oe, _ = out_segs[oi]
        u, v = wshape
        blk = weights[:, ws:we].reshape(-1, u, v) if weight_row_major else \
            weights[:, ws:we].reshape(-1, v, u).transpose(1, 2)
        out[:, os_:oe] += c * torch.einsum("buv,bu->bv", blk.expand(batch, u, v), x[:, xs:xe])
    return out


def run_case(name, irreps_in, irreps_out, m_max, batch=8):
    import cuequivariance_torch as cuet

    poly = descriptor(irreps_in, irreps_out, m_max)
    w_size, x_size = poly.inputs[0].size, poly.inputs[1].size
    torch.manual_seed(0)
    weights = torch.randn(1, w_size, device=DEV, dtype=DT)
    x = torch.randn(batch, x_size, device=DEV, dtype=DT)
    idx = torch.zeros(batch, dtype=torch.long, device=DEV)

    oracle_u = hand_oracle(poly, weights, x, weight_row_major=True)
    oracle_v = hand_oracle(poly, weights, x, weight_row_major=False)

    row = {"case": name, "w_size": int(w_size), "x_size": int(x_size),
           "n_paths": sum(len(stp.paths) for _, stp in poly.operations)}
    for method in METHODS:
        try:
            mod = cuet.SegmentedPolynomial(poly, method=method, math_dtype=DT).cuda()
            got = mod([weights, x], input_indices={0: idx})[0]
            for tag, oracle in (("u_major", oracle_u), ("v_major", oracle_v)):
                rel = ((got - oracle).abs().max() / oracle.abs().max().clamp_min(1e-30)).item()
                row[f"{method}__{tag}"] = rel
        except Exception as exc:
            row[f"{method}__error"] = f"{type(exc).__name__}: {str(exc)[:90]}"
    return row


def main():
    import torch as _t

    print(f"torch {_t.__version__}")
    try:
        import cuequivariance_ops_torch  # noqa: F401
        print("cuequivariance_ops_torch: extension LOADED (fused paths are real)")
    except Exception as exc:
        print(f"cuequivariance_ops_torch: NOT LOADED ({str(exc)[:60]}) -- everything "
              "will silently be `naive`; triage is invalid in this interpreter")

    rows = []
    for case in CASES:
        row = run_case(*case)
        rows.append(row)
        best = {k: v for k, v in row.items() if isinstance(v, float)}
        print(f"\n{row['case']:32s} w={row['w_size']:7d} x={row['x_size']:5d} "
              f"paths={row['n_paths']}")
        for k, v in sorted(best.items()):
            flag = "  <-- MATCH" if v < 1e-10 else ""
            print(f"    {k:34s} rel={v:.3e}{flag}")
        for k, v in row.items():
            if k.endswith("__error"):
                print(f"    {k:34s} {v}")

    out = pathlib.Path("bench/results/b3_triage.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))

    # ---- classification -------------------------------------------------------------
    print("\n" + "=" * 78)
    fused_ok = [r for r in rows if min(
        (v for k, v in r.items() if k.startswith("fused_tp__") and isinstance(v, float)),
        default=1.0) < 1e-10]
    idx_ok = [r for r in rows if min(
        (v for k, v in r.items() if k.startswith("indexed_linear__") and isinstance(v, float)),
        default=1.0) < 1e-10]
    print(f"fused_tp       matches the hand oracle on {len(fused_ok)}/{len(rows)} cases")
    print(f"indexed_linear matches the hand oracle on {len(idx_ok)}/{len(rows)} cases")
    if len(fused_ok) == len(rows) and len(idx_ok) < len(rows):
        print("\nVERDICT: cuEq bug. `fused_tp` implements the descriptor; `indexed_linear` "
              "does not, on the same descriptor and the same inputs.")
        if idx_ok:
            print(f"         indexed_linear is correct on: {[r['case'] for r in idx_ok]}")
            print(f"         and wrong on:                 "
                  f"{[r['case'] for r in rows if r not in idx_ok]}")
    elif not fused_ok and not idx_ok:
        print("\nVERDICT: our bug. Neither backend matches our reading of the descriptor, "
              "so the hand oracle (and hence our weight mapping) is wrong.")
    else:
        print("\nVERDICT: inconclusive -- see the per-case table above.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
