"""B2 evidence probe -- fast, no inductor autotuning, no expensive inventory.

Establishes the single load-bearing fact about the torch.compile baseline: whether the
conservative training step (which contains a true double backward) can run under
torch.compile at all, and at which layer it stops.

Backend ladder, cheapest-first, so the failure is *located* rather than merely reported:

    backend="eager"      dynamo capture only; never enters AOTAutograd
    backend="aot_eager"  AOTAutograd, no inductor codegen
    inductor (default)   the full stack

If `eager` succeeds while `aot_eager` and inductor fail, AOTAutograd is precisely the
blocker, and `backend="eager"` is the honest "best achievable torch.compile hybrid".

Kept separate from b2_compile.py because the latter's `torch._dynamo.explain` +
max-autotune paths are slow and, on this contended login node, deadlocked inductor's
parallel compile-worker pool (parent blocked in do_wait, zero output).
"""

from __future__ import annotations

import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from baselines.common import load_block_and_batch, make_step, precision_context
from bench.harness import Measurement, assert_step_is_live, time_training_step

LADDER = [
    ("eager", None),
    ("compile_backend_eager", "eager"),
    ("compile_aot_eager", "aot_eager"),
    ("compile_inductor", "inductor"),
]


def run(fixture: str, precision: str, name: str, backend: str | None) -> Measurement:
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    with precision_context(precision):
        block, batch, jd, stats = load_block_and_batch(fixture, precision)
        target = block if backend is None else torch.compile(block, backend=backend)
        step, zero, _ = make_step(target, batch, jd, precision)
        m = time_training_step(
            step, zero, label=f"B2 {name}", fixture=fixture, precision=precision,
            atoms=stats["atoms"], edges=stats["edges"],
            liveness_fn=lambda: assert_step_is_live(block, batch["pos"]),
        )
    del block, batch, jd
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    return m


def main():
    fixtures = sys.argv[1:] or ["si_small"]
    results = []
    for fixture in fixtures:
        for name, backend in LADDER:
            try:
                m = run(fixture, "fp32", name, backend)
            except Exception as exc:
                m = Measurement(f"B2 {name}", fixture, "fp32", float("nan"), float("nan"),
                                float("nan"), float("nan"), float("nan"), 0, 0, 0,
                                error=f"{type(exc).__name__}: {str(exc)[:150]}")
                torch._dynamo.reset()
                torch.cuda.empty_cache()
            results.append(m)
            status = m.error or (f"{m.median_ms:8.2f} ms  IQR {m.iqr_ms:6.2f}  "
                                 f"peak {m.peak_mem_gib:6.2f} GiB")
            print(f"{m.label:24s} {fixture:10s}  {status}", flush=True)

    out = pathlib.Path("bench/results/b2_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([vars(m) for m in results], indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
