"""Profile `conv1_90` under Nsight Compute: baseline, A_transpose, B_smem. si_medium, fp32.

Two static models have now produced numbers that looked like measurements and were properties of
my bookkeeping: D42's traffic model (refuted in D53 by its own baseline, which it says should take
7.9x longer than it does) and the issue-bound estimate (57x below the winner, so not binding
either). `ncu` is the only remaining instrument, and this is its run.

**The adjudication table below is fixed before any counter exists.** Every row names the metric,
the reading, and the consequence. Numbers land into a written frame, not the other way around.

Nsight Compute 2025.2.0, from the `prgenv-gnu/25.6:v2` uenv image, which ships its own CUPTI.
`RmProfilingAdminOnly: 0` on this node -- counter access is open, no permission blocker.

    uenv run prgenv-gnu/25.6:v2 --view=default -- bash bench/run_ncu.sh

## The three arms, and why exactly these

  baseline      714.819 ms   the kernel as emitted
  A_transpose   582.023 ms   1.228x -- the ratified winner, thread axis innermost
  B_smem      1 852.103 ms   0.386x -- the struck arm, whose *loss* has to be attributed

`A_matched` and `AB_both` are omitted: they are controls whose job was to make B readable, and
they add replay time without addressing any row of the table.

## Adjudication table, pre-registered

**Row 1 -- was the 32-line fan ever real?**
  metric: `l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio`
  At fp32 a fully coalesced warp load is 128 B = **4 sectors**; the uncoalesced fan D42 described
  is one line per lane = **32 sectors**.
    * baseline ~32, A_transpose ~4 -> the fan was **real**, and since removing it bought only
      1.228x, coalescing was real **but not binding**. D42 is upheld as a description of the
      access pattern and refuted as an account of the cost -- which is what D53 already concluded
      from timing alone, now confirmed or denied at the hardware.
    * baseline already ~4 -> **the layout reading was wrong too**, the 1.228x came from something
      other than coalescing (address arithmetic, register pressure), and D42 fails completely
      rather than partially.

**Row 2 -- why does B lose? This row alone decides input-row staging.**
  metrics: `sm__warps_active.avg.pct_of_peak_sustained_active` (achieved occupancy),
           `smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio`
  Per the standing ruling there are exactly two outcomes and **no third**:
    * **capacity-driven occupancy collapse -> input-row staging REVIVES.**
    * **barrier / double-touch -> input-row staging DIES.**
  The discriminator is quantitative and fixed here, so that it cannot be chosen afterwards. A pure
  occupancy explanation predicts, for a latency-bound kernel,
        t_B_predicted  =  714.819 ms  x  (occupancy_baseline / occupancy_B)
  Occupancy **revives** staging iff that prediction lands within **1.5x** of the measured
  1 852.103 ms. Otherwise the residual is barrier and double-touch, and staging **dies**.
  This is the D55 model self-check applied at the moment of use: the occupancy account must
  *reproduce the number*, not merely point in the right direction. If it cannot, it does not get
  to claim the outcome just because its sign is right -- which is precisely how D42 survived.

**Row 3 -- what is the winner actually waiting on?**
  metrics: the four stall reasons, per issue-active warp --
    `long_scoreboard` (global/L2 latency) · `mio_throttle` (LSU/smem pipe saturation) ·
    `barrier` · `no_instruction` (front end starved)
  The largest share on `A_transpose` names the bottleneck for the remaining 582 ms. No consequence
  is pre-assigned to rows 3-5 beyond "this is what the next intervention targets", because
  pre-committing an intervention to an unmeasured bottleneck is the error this whole sequence
  exists to stop repeating.

**Row 4 -- is the L2-residency claim right?**
  metrics: `lts__t_sector_hit_rate.pct`, `dram__bytes.sum.per_second`, `lts__t_bytes.sum.per_second`
  D50 and D53 blame the traffic model's 7.9x over-prediction on the four weights being 1.25 MB in
  a 60 MiB L2 -- L2 hits charged at DRAM bandwidth. A high L2 hit rate with low DRAM throughput
  confirms it. A **low** hit rate would mean the traffic model failed for a reason I have not yet
  identified, and D53's stated mechanism would need withdrawing even though its refutation stands.

**Row 5 -- instruction-fetch bound (hypothesis #5)?**
  metrics: `no_instruction` stall share; i-cache miss counters if sm_90 exposes them.
  ~10 246 straight-line instructions per thread against a small L1 instruction cache is a
  mechanism that would explain the baseline, the winner, *and* why neither memory model fits --
  and that unifying elegance is exactly why it is being sent to the profiler rather than into the
  emitter. It is confirmed only if `no_instruction` is the dominant stall. **A hypothesis that
  explains everything and is checked against nothing is the D42 failure mode with a new name.**

## What is NOT being decided here

Nothing about the SP-IR bet, nothing about the S1 hill, and no emitter change. This run
attributes cost on one kernel. The layout requirement is already ratified on the timing evidence
and does not depend on any of these counters.
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
from codegen.emit_tile import SMEM_CAP_BYTES, emit_tile_source              # noqa: E402
from codegen.emit_tile import staged_layout                                 # noqa: E402
from codegen.schedule import analyze_group                                  # noqa: E402
from codegen.tile import Ch                                                 # noqa: E402
from fixtures.load import load_batch                                        # noqa: E402
from zippel.simplify import fusion_groups, simplify                         # noqa: E402


def thread_axis(sched, buf):
    for a in sched.assigns:
        for t in a.terms:
            for f in t.factors:
                if f[0] == buf:
                    for k, i in enumerate(f[1]):
                        if isinstance(i, Ch):
                            return k
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="si_medium")
    ap.add_argument("--dtype", default="f32")
    args = ap.parse_args()

    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32
    from cutlass.cute.runtime import from_dlpack

    dt = torch.float32 if args.dtype == "f32" else torch.float64
    itemsize = dt.itemsize
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
    template, sched, _ = route(simp, spec)
    assert template == "T2"

    plan = {}
    for buf in spec.live_in:
        t = simp.type_of(buf)
        pos = thread_axis(sched, buf)
        if pos is None or t.segment != "none" or len(t.sizes) < 2 or pos == len(t.sizes) - 1:
            continue
        rank = len(t.sizes)
        lay = staged_layout(simp, sched, buf)
        plan[buf] = {"perm": tuple([k for k in range(rank) if k != pos] + [pos]),
                     "kib": lay["extent"] * lay["T"] * itemsize / 1024}
    transpose = {b: p["perm"] for b, p in plan.items()}
    stage, used = [], 0.0
    for b, p in sorted(plan.items(), key=lambda kv: kv[1]["kib"]):
        if (used + p["kib"]) * 1024 <= SMEM_CAP_BYTES:
            stage.append(b); used += p["kib"]
    stage = tuple(stage)

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

    stream = cutlass.cuda.default_stream()
    arms = {"baseline": ({}, ()), "A_transpose": (transpose, ()), "B_smem": ({}, stage)}

    # Everything -- compilation, allocation, permutation -- happens before the profiler starts, so
    # ncu sees exactly three kernels and no torch setup work.
    ready = []
    for name, (tr, st) in arms.items():
        src = emit_tile_source(simp, sched, dtype=args.dtype, transpose=tr, stage_shared=st)
        Kernel, order = build_kernel(src, f"ncu_{name}_{args.dtype}", sched=sched)
        module = sys.modules[f"zippel_generated.ncu_{name}_{args.dtype}"]
        eff = getattr(module, "TRANSPOSE", {})
        tensors = {}
        for b in order:
            v = (ref[b] if b in ref else torch.zeros(1, dtype=torch.float64)).to("cuda", dt)
            if b in eff:
                v = v.permute((0,) + tuple(k + 1 for k in eff[b]))
            tensors[b] = v.contiguous()
        for b in spec.live_out:
            tensors[b] = torch.zeros_like(ref[b].to("cuda", dt))
        call = tuple(from_dlpack(tensors[b], assumed_align=16) for b in order) + (
            Int32(sizes[spec.segment]), stream)
        fn = cute.compile(Kernel(), *call)
        fn(*call)                                   # warm up outside the profiled region
        torch.cuda.synchronize()
        ready.append((name, fn, call, tensors))
        print(f"prepared {name}", flush=True)

    print("--- profiled region begins: one launch per arm ---", flush=True)
    torch.cuda.profiler.start()
    for name, fn, call, _t in ready:
        fn(*call)
        torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print("--- profiled region ends ---", flush=True)


if __name__ == "__main__":
    main()
