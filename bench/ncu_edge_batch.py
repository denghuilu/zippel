"""Profile one edge-batch arm of `conv1_90` under Nsight Compute. One arm per process, by design.

D81 fired the **plateau** cell: demand bytes fell 3.92× and time fell 1.33×, with the realised
fraction dropping from 61.5 % to 33.8 % as `E_c` rose. The bytes law (D59) held to 1.9 % across
the transpose arms and 0.5 % out-of-sample, so its failure here is a fact about *this*
transformation and this pass exists to say which fact.

## The units fault this run was built to expose

D59's law was validated on **DRAM** bytes — `dram__bytes.sum.per_second` × duration. D69's
edge-batch predictions were **demand** bytes — what the kernel *requests*, `1.3107/E_c + 0.0087`
MB/edge. **I divided a demand-byte prediction into a DRAM-validated law.** Those are different
quantities separated by the whole cache hierarchy, and the gap between them is exactly what an
L2 with a 57 % hit rate does for a living. The plateau may be nothing more than that error.

Henceforth every byte prediction names its kind, and may only be divided into a law validated on
the same kind. This run measures **both** so the two can finally be compared on one kernel.

## Pre-registered split

* **(a)** DRAM bytes track demand (fall ~`1/E_c`) **and** achieved bandwidth falls with occupancy
  → the bytes law reigns in the form `t = bytes / BW(occ)`, the BW(occupancy) curve gains three
  points, and D72's out-of-sample check against `baseline`/`B_smem` becomes runnable.
* **(b)** DRAM bytes do **not** track demand → the demand→DRAM amplification governs, D64's
  **3.6× residual becomes load-bearing**, and the parked probes (strided-read, partial-sector
  store; D71) unpark.
* **(c)** both, apportioned — report the split rather than choosing.

## Why one arm per process

Serial compile of the three arms is 78 minutes (286 + 1 047 + 3 355 s). Three processes on three
GPUs cost the longest arm alone. **Counters are unaffected by CPU contention; wall-clock is not** —
so durations from this run are NOT used as timings. D81's separately-measured wall-clock stands,
and the pinning law (compile workers never share cores with a live measurement) is honoured by
never taking a timing from here.

    uenv run prgenv-gnu/25.6:v2 --view=default -- bash bench/run_ncu_edge_batch.sh 4 0
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from blocks.eso2_ir import build_forward                                    # noqa: E402
from blocks.eso2_ref import BlockConfig, ESO2RefBlock                       # noqa: E402
from blocks.ir_bind import bind                                             # noqa: E402
from codegen.compose import route                                           # noqa: E402
from codegen.emit import build_kernel                                       # noqa: E402
from codegen.emit_tile import emit_tile_source                              # noqa: E402
from codegen.schedule import analyze_group                                  # noqa: E402
from fixtures.load import load_batch                                        # noqa: E402
from zippel.simplify import fusion_groups, simplify                         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge-batch", type=int, required=True)
    ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--fixture", default="si_medium")
    ap.add_argument("--dtype", default="f32")
    args = ap.parse_args()

    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32
    from cutlass.cute.runtime import from_dlpack

    dt = torch.float32 if args.dtype == "f32" else torch.float64
    cfg = BlockConfig()
    torch.manual_seed(0)
    jd = [j.to(torch.float64) for j in torch.load("blocks/Jd.pt", weights_only=False)]
    block = ESO2RefBlock(cfg).to("cpu", torch.float64)
    prog, _ = build_forward(cfg, gauss_coeff=block.gauss_coeff)
    simp = simplify(prog, keep=prog.outputs)

    batch_s = load_batch("si_small", "cpu", torch.float64, cfg, requires_grad=False)
    inputs_s, sizes_s = bind(block, batch_s, jd, cfg)
    batch_m = load_batch(args.fixture, "cpu", torch.float64, cfg, requires_grad=False)
    _im, sizes = bind(block, batch_m, jd, cfg)

    groups = fusion_groups(simp, max_volume=10_000)
    gi = next(i for i, g in enumerate(groups) if "conv1_90" in g)
    spec = analyze_group(simp, groups[gi], name="conv1_90")
    _t, sched, _ = route(simp, spec)

    from zippel.interp import run
    small = run(simp, inputs_s, sizes_s)

    def at_scale(buf, v):
        t = simp.type_of(buf)
        if t.segment == "none":
            return v
        n = sizes[t.segment]
        return v.repeat((-(-n // v.shape[0]),) + (1,) * (v.dim() - 1))[:n].contiguous()

    needed = set(spec.live_in) | set(spec.live_out) | set(getattr(spec, "internal", ()))
    ref = {b: at_scale(b, small[b]) for b in needed if b in small}
    del small

    E = args.edge_batch
    src = emit_tile_source(simp, sched, dtype=args.dtype, edge_batch=E, chunk=args.chunk)
    name = f"nb_E{E}_c{args.chunk}_{args.dtype}"
    Kernel, order = build_kernel(src, name, sched=sched)
    module = sys.modules[f"zippel_generated.{name}"]
    eff = getattr(module, "TRANSPOSE", {})

    tensors = {}
    for b in order:
        v = (ref[b] if b in ref else torch.zeros(1, dtype=torch.float64)).to("cuda", dt)
        if b in eff:
            v = v.permute((0,) + tuple(k + 1 for k in eff[b]))
        tensors[b] = v.contiguous()
    for b in spec.live_out:
        tensors[b] = torch.zeros_like(ref[b].to("cuda", dt))
    stream = cutlass.cuda.default_stream()
    call = tuple(from_dlpack(tensors[b], assumed_align=16) for b in order) + (
        Int32(sizes[spec.segment]), stream)

    print(f"E_c={E} chunk={args.chunk}: compiling…", flush=True)
    fn = cute.compile(Kernel(), *call)
    fn(*call)
    torch.cuda.synchronize()
    print(f"E_c={E}: warm launch done, entering profiled region", flush=True)

    torch.cuda.profiler.start()
    fn(*call)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print(f"E_c={E}: profiled region ends", flush=True)


if __name__ == "__main__":
    main()
