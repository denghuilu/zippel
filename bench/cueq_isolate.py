"""Thread (b): isolate which property triggers cuEquivariance's indexed_linear bug.

`escn_tp_compact` introduces three things simultaneously at m >= 1 -- multiple paths, a weight
segment reused by two paths, and a negative path coefficient -- so the earlier ladder could not
say which is the trigger. This builds SegmentedTensorProducts by hand that vary one property at
a time, against the same hand oracle used in bench/b3_triage.py.

    $ZIPPEL_CUEQ_PY bench/cueq_isolate.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

DT = torch.float64
DEV = "cuda"
U, V, BATCH = 4, 3, 8


def build(n_paths: int, reuse_weight: bool, negative_coeff: bool,
          shared_out: bool = False, hetero: bool = False):
    """A `uv,u,v` STP with exactly the requested properties and nothing else.

    `shared_out` makes every path accumulate into ONE output segment, which is what
    escn_tp_compact does (both W1*x(-m) and W2*x(+m) land on out(-m)) and what the first
    round of this experiment failed to vary. `hetero` gives the segments differing sizes,
    the other structural difference from the hand-built cases.
    """
    import cuequivariance as cue

    d = cue.SegmentedTensorProduct.from_subscripts("uv,u,v")
    us = [U + 2 * i for i in range(n_paths)] if hetero else [U] * n_paths
    vs = [V] * n_paths if shared_out else ([V + i for i in range(n_paths)] if hetero
                                           else [V] * n_paths)
    for i in range(n_paths):
        d.add_segment(1, (us[i],))
    n_out = 1 if shared_out else n_paths
    for i in range(n_out):
        d.add_segment(2, (vs[0] if shared_out else vs[i],))
    n_w = 1 if reuse_weight else n_paths
    for i in range(n_w):
        d.add_segment(0, (us[0] if reuse_weight else us[i],
                          vs[0] if (shared_out or reuse_weight) else vs[i]))
    for i in range(n_paths):
        c = -1.0 if (negative_coeff and i % 2 == 1) else 1.0
        d.add_path(0 if reuse_weight else i, i, 0 if shared_out else i, c=c)
    return cue.SegmentedPolynomial.eval_last_operand(d)


def oracle(poly, w, x):
    """Literal `uv,u,v` semantics, transcribed from the descriptor's own path list."""
    (op, stp), = poly.operations
    del op

    def slices(operand):
        out, off = [], 0
        for seg in operand.segments:
            n = 1
            for dd in seg:
                n *= dd
            out.append((off, off + n, tuple(seg)))
            off += n
        return out

    ws, ins, outs = (slices(o) for o in stp.operands)
    res = torch.zeros(x.shape[0], outs[-1][1], device=x.device, dtype=x.dtype)
    for p in stp.paths:
        wi, xi, oi = (int(k) for k in p.indices)
        c = float(p.coefficients)
        a, b, (u, v) = ws[wi]
        xa, xb, _ = ins[xi]
        oa, ob, _ = outs[oi]
        blk = w[:, a:b].reshape(-1, u, v).expand(x.shape[0], u, v)
        res[:, oa:ob] += c * torch.einsum("buv,bu->bv", blk, x[:, xa:xb])
    return res


def main():
    import cuequivariance_torch as cuet
    try:
        import cuequivariance_ops_torch  # noqa: F401
        print("ops extension: LOADED\n")
    except Exception as exc:
        print(f"ops extension NOT loaded ({str(exc)[:60]}) -- results invalid\n")
        return

    cases = [
        # round 1: none of these triggered it
        ("1 path,  uniq w, +c", 1, False, False, False, False),
        ("2 paths, uniq w, +c", 2, False, False, False, False),
        ("2 paths, REUSE w, +c", 2, True, False, False, False),
        ("2 paths, uniq w, -c", 2, False, True, False, False),
        ("2 paths, REUSE w, -c", 2, True, True, False, False),
        ("4 paths, REUSE w, -c", 4, True, True, False, False),
        # round 2: the properties round 1 failed to vary
        ("2 paths -> SHARED out", 2, False, False, True, False),
        ("2 paths -> SHARED out, -c", 2, False, True, True, False),
        ("2 paths REUSE w -> SHARED", 2, True, True, True, False),
        ("2 paths, HETERO sizes", 2, False, False, False, True),
        ("4 paths REUSE -c SHARED", 4, True, True, True, False),
    ]
    rows = []
    print(f"{'case':26s} {'fused_tp':>11s} {'indexed_linear':>15s} {'naive':>11s}")
    for name, n, reuse, neg, shared, hetero in cases:
        try:
            poly = build(n, reuse, neg, shared, hetero)
        except Exception as exc:
            print(f"{name:26s} CONSTRUCT FAILED: {type(exc).__name__}: {str(exc)[:50]}")
            continue
        torch.manual_seed(0)
        w = torch.randn(1, poly.inputs[0].size, device=DEV, dtype=DT)
        x = torch.randn(BATCH, poly.inputs[1].size, device=DEV, dtype=DT)
        idx = torch.zeros(BATCH, dtype=torch.long, device=DEV)
        want = oracle(poly, w, x)
        row = {"case": name, "n_paths": n, "reused_weight": reuse,
               "negative_coeff": neg, "shared_out": shared, "hetero": hetero}
        cells = []
        for m in ("fused_tp", "indexed_linear", "naive"):
            try:
                mod = cuet.SegmentedPolynomial(poly, method=m, math_dtype=DT).cuda()
                got = mod([w, x], input_indices={0: idx})[0]
                rel = ((got - want).abs().max() / want.abs().max().clamp_min(1e-30)).item()
                row[m] = rel
                cells.append(f"{rel:.2e}")
            except Exception as exc:
                row[m] = f"{type(exc).__name__}"
                cells.append(type(exc).__name__[:11])
        print(f"{name:26s} {cells[0]:>11s} {cells[1]:>15s} {cells[2]:>11s}")
        rows.append(row)

    ok = lambda v: isinstance(v, float) and v < 1e-10          # noqa: E731
    bad = [r for r in rows if not ok(r.get("indexed_linear"))]
    print("\nindexed_linear fails on:", [r["case"] for r in bad] or "nothing")
    if bad:
        print("  n_paths>1 in all failures:  ", all(r["n_paths"] > 1 for r in bad))
        print("  reused weight in all:       ", all(r["reused_weight"] for r in bad))
        print("  negative coeff in all:      ", all(r["negative_coeff"] for r in bad))
    out = pathlib.Path("bench/results/cueq_isolate.json")
    out.write_text(json.dumps(rows, indent=2, default=str))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
