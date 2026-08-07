"""S1c performance: fused forward vs eager, at the measured boundary.

**This is the sanity anchor, not the bet.** The store-elision lever D23 identifies lives in force
and double-backward, where the intermediates are 3x and 9x the forward's. A forward-only number
warrants neither despair nor a victory lap, and REPORT says so where it reports it.

Measured boundary, identical for both parties: inputs already on device, neighbour list already
built, energy tensor produced. Neighbour-list construction is excluded for everyone. Timing is
CUDA events around that region; peak memory is `max_memory_allocated` reset per configuration;
launch count comes from the compiled program for the fused side and from a profiler pass for
eager, and the two are *not* claimed comparable without saying how each was obtained.

Run under the pinned protocol (slurm/verdict.sbatch): N=5 independent allocations, median of
medians, full range reported as the error bar. A speedup claim must exceed the measured spread
for its configuration -- ~2 % at the medium fixtures, ~4-5 % at the small ones.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import socket
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from blocks.eso2_ir import build_forward                                    # noqa: E402
from blocks.eso2_ref import BlockConfig, ESO2RefBlock                       # noqa: E402
from blocks.ir_bind import bind                                             # noqa: E402
from codegen.compose import (DEFAULT_MAX_VOLUME, allocate,                  # noqa: E402
                             compile_program, run_program)
from fixtures.load import load_batch                                        # noqa: E402


def time_ms(fn, warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        samples.append(a.elapsed_time(b))
    samples.sort()
    n = len(samples)
    return {"median_ms": samples[n // 2], "min_ms": samples[0], "max_ms": samples[-1],
            "iqr_ms": samples[int(n * 0.75)] - samples[int(n * 0.25)], "iters": n}


def peak_mib(fn) -> float:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2 ** 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="si_small")
    ap.add_argument("--dtype", default="f32", choices=["f32", "f64"])
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--max-volume", type=int, default=DEFAULT_MAX_VOLUME)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dt = torch.float32 if args.dtype == "f32" else torch.float64
    cfg = BlockConfig()
    torch.manual_seed(0)
    jd = [j.to(dt) for j in torch.load("blocks/Jd.pt", weights_only=False)]
    block = ESO2RefBlock(cfg).to("cuda", dt).eval()
    batch = load_batch(args.fixture, "cuda", dt, cfg, requires_grad=False)

    prog, meta = build_forward(cfg, gauss_coeff=block.gauss_coeff)
    cpu_block = ESO2RefBlock(cfg).to("cpu", dt)
    cpu_block.load_state_dict(block.state_dict())
    inputs, sizes = bind(cpu_block, {k: (v.cpu() if torch.is_tensor(v) else v)
                                     for k, v in batch.items()},
                         [j.cpu() for j in jd], cfg)

    cp = compile_program(prog, sizes, f"bench_{args.fixture}", dtype=args.dtype,
                         max_volume=args.max_volume)
    env = allocate(cp, inputs, dtype=dt)
    run_program(cp, env)                       # compiles on first launch, outside the timed region

    def fused():
        run_program(cp, env, compile_kernels=False)

    def eager():
        with torch.no_grad():
            block(batch["pos"], batch["atomic_numbers"], batch["x_node"], batch["edge_index"],
                  batch["shifts"], batch["cos_gamma_k"], batch["sin_gamma_k"], jd)

    result = {
        "fixture": args.fixture, "dtype": args.dtype, "sizes": sizes,
        "host": socket.gethostname(), "slurm_job": None,
        "launches_fused": cp.n_launches,
        "templates": {t: sum(1 for g in cp.groups if g.template == t)
                      for t in ("T1", "T2", "T3")},
        "fused": time_ms(fused, args.warmup, args.iters),
        "eager": time_ms(eager, args.warmup, args.iters),
        "peak_mib_fused": peak_mib(fused),
        "peak_mib_eager": peak_mib(eager),
    }
    import os
    result["slurm_job"] = os.environ.get("SLURM_JOB_ID")
    f, e = result["fused"]["median_ms"], result["eager"]["median_ms"]
    result["speedup"] = e / f
    result["peak_ratio"] = result["peak_mib_eager"] / result["peak_mib_fused"]

    print(f"{args.fixture} {args.dtype}  edges={sizes['edge']}  host={result['host']}")
    print(f"  fused  {f:8.3f} ms  (IQR {result['fused']['iqr_ms']:.3f})  "
          f"{cp.n_launches} launches  peak {result['peak_mib_fused']:.1f} MiB")
    print(f"  eager  {e:8.3f} ms  (IQR {result['eager']['iqr_ms']:.3f})  "
          f"peak {result['peak_mib_eager']:.1f} MiB")
    print(f"  speedup {result['speedup']:.3f}x   peak ratio {result['peak_ratio']:.3f}x")

    out = pathlib.Path(args.out or
                       f"bench/results/s1c_{args.fixture}_{args.dtype}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
