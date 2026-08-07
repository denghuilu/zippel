"""Thread (c): does `indexed_linear` silently compute in FP32 despite math_dtype=float64?

Observed incidentally during B3 triage: with FP64 operands and math_dtype=torch.float64, the
one multi-element single-path case was accurate only to ~1.7e-08 -- FP32 territory -- and the
backend emits `UserWarning: indexed_linear does not support explicit math_dtype. This will be
ignored.`

This isolates that claim on a descriptor `indexed_linear` computes CORRECTLY, so precision is
the only variable and the result cannot be confused with the correctness bug in thread (b).

    $ZIPPEL_CUEQ_PY bench/cueq_math_dtype.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import warnings

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

DEV = "cuda"
U, V, BATCH = 64, 32, 256


def single_path_poly():
    """One path, unique weight, +1 coefficient -- a case indexed_linear gets right."""
    import cuequivariance as cue

    d = cue.SegmentedTensorProduct.from_subscripts("uv,u,v")
    d.add_segment(0, (U, V))
    d.add_segment(1, (U,))
    d.add_segment(2, (V,))
    d.add_path(0, 0, 0, c=1.0)
    return cue.SegmentedPolynomial.eval_last_operand(d)


def main():
    import cuequivariance_torch as cuet
    try:
        import cuequivariance_ops_torch  # noqa: F401
    except Exception as exc:
        print(f"ops extension NOT loaded ({str(exc)[:60]}) -- results invalid")
        return

    poly = single_path_poly()
    torch.manual_seed(0)
    # FP64 operands throughout; the reference is an exact FP64 matmul.
    w = torch.randn(1, poly.inputs[0].size, device=DEV, dtype=torch.float64)
    x = torch.randn(BATCH, poly.inputs[1].size, device=DEV, dtype=torch.float64)
    idx = torch.zeros(BATCH, dtype=torch.long, device=DEV)
    want = torch.einsum("uv,bu->bv", w[0].reshape(U, V), x)

    rows = []
    print(f"{'method':16s} {'math_dtype':12s} {'rel err':>10s}  warning")
    for method in ("fused_tp", "indexed_linear", "naive"):
        for md in (torch.float64, torch.float32):
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    mod = cuet.SegmentedPolynomial(poly, method=method, math_dtype=md).cuda()
                    msgs = [str(c.message) for c in caught]
                got = mod([w, x], input_indices={0: idx})[0]
                rel = ((got - want).abs().max() / want.abs().max()).item()
            except Exception as exc:
                # fused_tp refuses float32 math with float64 inputs -- an explicit guard,
                # which is the *correct* behaviour and worth contrasting with the silent
                # ignore below.
                print(f"{method:16s} {str(md).replace('torch.',''):12s} {'--':>10s}  "
                      f"REJECTS: {str(exc)[:46]}")
                rows.append({"method": method, "math_dtype": str(md),
                             "rel_err": None, "rejected": str(exc)[:80]})
                continue
            warn = next((m for m in msgs if "math_dtype" in m), "")
            print(f"{method:16s} {str(md).replace('torch.',''):12s} {rel:10.2e}  "
                  f"{'SILENTLY IGNORES math_dtype' if warn else ''}")
            rows.append({"method": method, "math_dtype": str(md), "rel_err": rel,
                         "warned_math_dtype_ignored": bool(warn)})

    f64 = {r["method"]: r["rel_err"] for r in rows
           if r["math_dtype"] == "torch.float64" and r.get("rel_err") is not None}
    print("\nWith FP64 operands and math_dtype=float64:")
    for m, e in f64.items():
        verdict = ("FP64-accurate" if e < 1e-14 else
                   "FP32-accurate -- silently reduced precision" if e < 1e-5 else
                   "neither (see thread (b): correctness bug)")
        print(f"  {m:16s} rel {e:.2e}  -> {verdict}")

    out = pathlib.Path("bench/results/cueq_math_dtype.json")
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
