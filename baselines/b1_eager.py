"""B1 -- eager PyTorch baseline.

The reference block run as-is through the shared harness, at fp32 (strict), tf32, and
bf16-AMP. This is the "how fairchem would run it" number and the floor every other
implementation is measured against.

    python baselines/b1_eager.py --fixtures si_medium --precisions fp32 bf16
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from baselines.common import PRECISIONS, load_block_and_batch, make_step, precision_context
from bench.harness import Measurement, time_training_step

ALL_FIXTURES = ["si_small", "si_medium", "si_large", "cu_small", "cu_medium", "cu_large"]


def run_one(fixture: str, precision: str, label: str = "B1 eager") -> Measurement:
    with precision_context(precision):
        block, batch, jd, stats = load_block_and_batch(fixture, precision)
        step, zero, live = make_step(block, batch, jd, precision)
        m = time_training_step(
            step, zero, label=label, fixture=fixture, precision=precision,
            atoms=stats["atoms"], edges=stats["edges"], liveness_fn=live,
        )
    del block, batch, jd
    torch.cuda.empty_cache()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", nargs="*", default=["si_medium"])
    ap.add_argument("--precisions", nargs="*", default=list(PRECISIONS))
    ap.add_argument("--out", default="bench/results/b1_eager.json")
    args = ap.parse_args()

    results = []
    for fixture in args.fixtures:
        for precision in args.precisions:
            try:
                m = run_one(fixture, precision)
            except Exception as exc:  # a baseline that cannot run is a result, not a crash
                m = Measurement("B1 eager", fixture, precision, float("nan"), float("nan"),
                                float("nan"), float("nan"), float("nan"), 0, 0, 0,
                                error=f"{type(exc).__name__}: {str(exc)[:160]}")
                torch.cuda.empty_cache()
            results.append(m)
            status = m.error or (f"{m.median_ms:8.2f} ms  IQR {m.iqr_ms:6.2f}  "
                                 f"peak {m.peak_mem_gib:6.2f} GiB  n={m.iters}")
            print(f"{m.label:12s} {fixture:10s} {precision:5s}  {status}", flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([vars(m) for m in results], indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
