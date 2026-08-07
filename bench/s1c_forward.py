"""S1c: drive the whole forward through emitted kernels and check it end to end.

Individually-correct kernels are not a correct program. This validates the composition against
the FP64 interpreter and, through it, against the reference block.

    python bench/s1c_forward.py --fixture si_small
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from blocks.eso2_ir import build_forward                                   # noqa: E402
from blocks.eso2_ref import BlockConfig, ESO2RefBlock                      # noqa: E402
from blocks.ir_bind import bind                                            # noqa: E402
from codegen import costs                                                  # noqa: E402
from codegen.compose import (DEFAULT_MAX_VOLUME, allocate, compile_program,  # noqa: E402
                             run_program)
from fixtures.load import load_batch                                       # noqa: E402
from zippel.interp import run                                             # noqa: E402
from zippel.simplify import simplify                                       # noqa: E402

DT = torch.float64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="si_small")
    ap.add_argument("--max-volume", type=int, default=DEFAULT_MAX_VOLUME)
    ap.add_argument("--out", default="bench/results/s1c_forward.json")
    args = ap.parse_args()

    cfg = BlockConfig()
    torch.manual_seed(0)
    jd = [j.to(DT) for j in torch.load("blocks/Jd.pt", weights_only=False)]
    block = ESO2RefBlock(cfg).to("cpu", DT)
    batch = load_batch(args.fixture, "cpu", DT, cfg)

    prog, meta = build_forward(cfg, gauss_coeff=block.gauss_coeff)
    simp = simplify(prog, keep=prog.outputs)
    inputs, sizes = bind(block, batch, jd, cfg)
    print(f"forward @ {args.fixture}: {len(simp.ops)} ops, sizes {sizes}", flush=True)

    ref = run(simp, inputs, sizes)
    print(f"interpreter done, {len(ref)} buffers", flush=True)

    cp = compile_program(prog, sizes, "s1c", dtype="f64", max_volume=args.max_volume)
    print(f"compiled {cp.n_launches} kernels "
          f"({sum(1 for g in cp.groups if g.template == 'T1')} T1, "
          f"{sum(1 for g in cp.groups if g.template == 'T2')} T2, "
          f"{sum(1 for g in cp.groups if g.template == 'T3')} T3)", flush=True)

    env = allocate(cp, inputs, dtype=DT)
    run_program(cp, env)
    print("launched\n", flush=True)

    energy = meta["energy"]
    # the interpreter may run on a different device than the composed program; compare on one
    got = env[energy]
    want = ref[energy].to(got.device)
    rel = float((got - want).abs().max() / want.abs().max().clamp_min(1e-300))
    print(f"{'buffer':>14} {'rel err':>11}")
    print(f"{energy:>14} {rel:>11.3e}   <- program output")

    # every live-out, so a wrong intermediate cannot hide behind a right total
    worst = []
    for g in cp.groups:
        for b in g.live_out:
            if b in ref and ref[b].numel():
                r = ref[b].to(env[b].device)
                scale = float(r.abs().max())
                e = float((env[b] - r).abs().max())
                worst.append((e / scale if scale else e, b, g.template))
    worst.sort(reverse=True)
    for e, b, t in worst[:5]:
        print(f"{b:>14} {e:>11.3e}   {t}")

    ok = rel < 1e-10 and (not worst or worst[0][0] < 1e-10)
    detail = (f", worst live-out {worst[0][0]:.3e} ({worst[0][1]})" if worst else "")
    print(f"\nS1C {'PASS' if ok else 'FAIL'}: energy rel {rel:.3e}{detail}")
    summary = costs.summary()
    print("cost ledger: " + "  ".join(f"{k}={v:.1f}s" for k, v in summary["totals_s"].items()),
          flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "fixture": args.fixture, "sizes": sizes, "launches": cp.n_launches,
        "max_volume": args.max_volume, "energy_rel_err": rel,
        "worst_live_out": [{"buffer": b, "rel": e, "template": t} for e, b, t in worst[:20]],
        "pass": ok, "costs": summary}, indent=2))
    print(f"wrote {out}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
