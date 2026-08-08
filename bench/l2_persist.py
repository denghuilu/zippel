"""Track 1 / lever (b): pin the 1.31 MB weight region in L2 and see whether the leak collapses.

D85 measured the thing this tests. `conv1_90` demands **1.3194 MB/edge** and DRAM delivers
**7.1683 MB/edge** — a **5.43× amplification** that *rises* to 16.2× as edge batching cuts demand.
The arithmetic that fits: 132 SMs × 16 blocks = **2 112 concurrent CTAs**, each streaming the full
1.31 MB weight footprint, giving a **2.8 GB concurrent working set against a 60 MiB L2**. Every CTA
evicts every other CTA's weights.

If that is right, pinning the weights should stop the eviction dead. The region is **1.31 MB
against a 37.5 MiB persisting-L2 capacity** — 3.5 % of it — so this is not a close-run capacity
argument, which is what makes it a clean test.

## Single variable, deliberately

Runs at **`E_c` = 1**. Edge batching is orthogonal and moves occupancy, registers and demand at
once; changing one thing is the whole point. The layout requirement (D54) stays on because it is
already ratified and is part of the baseline this is measured against.

## Pre-registered outcomes

* **hypothesis true** — the weight leak collapses, DRAM/edge falls toward the compulsory 13.3 KB,
  and the kernel very likely **leaves the DRAM-bound regime**. Consequence: the next binding
  resource is **measured, not assumed** — the same discipline that killed the occupancy story.
  Lever (c) CTA-scheduling becomes redundant to the extent persistence already fixed it.
* **hypothesis false** — the leak persists. The **eviction story is retired**, and **lever (c)
  CTA-scheduling moves up**: if the working set is not the problem, the *order* CTAs touch it in
  might be.
* Partial — report the fraction recovered rather than choosing a side.

## Mechanics

The four weights are packed into **one contiguous allocation** so a single access-policy window
covers them; CUDA allows one window per stream, and four separate tensors cannot be covered by
one. They are packed *already permuted* into their D54 layout, so the window and the layout
requirement compose rather than fight.

`cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize)` reserves the carve-out;
`cudaStreamSetAttribute(cudaStreamAttributeAccessPolicyWindow)` marks the range persisting with
`hitRatio = 1.0`. Both go through `ctypes` — torch exposes neither.

    python bench/l2_persist.py
"""

from __future__ import annotations

import argparse
import ctypes
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from blocks.eso2_ir import build_forward                                    # noqa: E402
from blocks.eso2_ref import BlockConfig, ESO2RefBlock                       # noqa: E402
from blocks.ir_bind import bind                                             # noqa: E402
from codegen.bounds import ordering_bound                                   # noqa: E402
from codegen.compose import route                                           # noqa: E402
from codegen.emit import build_kernel                                       # noqa: E402
from codegen.emit_tile import emit_tile_source                              # noqa: E402
from codegen.schedule import analyze_group                                  # noqa: E402
from fixtures.load import load_batch                                        # noqa: E402
from zippel.simplify import fusion_groups, simplify                         # noqa: E402

#: cudaLimitPersistingL2CacheSize. Enum 5 is MaxL2FetchGranularity -- checked, not assumed.
LIMIT_PERSISTING_L2 = 6
#: cudaStreamAttributeAccessPolicyWindow
STREAM_ATTR_ACCESS_POLICY = 1
ACCESS_NORMAL, ACCESS_STREAMING, ACCESS_PERSISTING = 0, 1, 2


class AccessPolicyWindow(ctypes.Structure):
    _fields_ = [("base_ptr", ctypes.c_void_p), ("num_bytes", ctypes.c_size_t),
                ("hitRatio", ctypes.c_float), ("hitProp", ctypes.c_int),
                ("missProp", ctypes.c_int)]


class StreamAttrValue(ctypes.Union):
    _fields_ = [("accessPolicyWindow", AccessPolicyWindow), ("syncPolicy", ctypes.c_int)]


def cudart():
    for lib in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
        try:
            return ctypes.CDLL(lib)
        except OSError:
            continue
    raise SystemExit("libcudart not loadable")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="si_medium")
    ap.add_argument("--dtype", default="f32")
    ap.add_argument("--carveout-mib", type=int, default=32)
    ap.add_argument("--configs", default="",
                    help="semicolon list of <carveout_mib>:<window 0|1>; overrides the two-arm "
                         "default. The control that matters is carve-out WITHOUT a window: it "
                         "separates 'persistence did not help' from 'my carve-out starved the "
                         "streaming data', which a single oversized arm cannot.")
    ap.add_argument("--ncu", action="store_true",
                    help="profiled mode: one launch per config inside profiler start/stop, no "
                         "timing. Settles whether pinning changes DRAM BYTES, which the wall-clock "
                         "ladder can only infer.")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", default="bench/results/l2_persist.json")
    args = ap.parse_args()

    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32
    from cutlass.cute.runtime import from_dlpack

    rt = cudart()
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

    src = emit_tile_source(simp, sched, dtype=args.dtype)
    Kernel, order = build_kernel(src, f"l2p_{args.dtype}", sched=sched)
    module = sys.modules[f"zippel_generated.l2p_{args.dtype}"]
    eff = getattr(module, "TRANSPOSE", {})

    # ---- pack the weights (already permuted) into ONE contiguous allocation -------------
    weights = [b for b in order if b in ref and simp.type_of(b).segment == "none"
               and ref[b].numel() > 1]
    shaped = {}
    for b in weights:
        v = ref[b].to("cuda", dt)
        if b in eff:
            v = v.permute((0,) + tuple(k + 1 for k in eff[b]))
        shaped[b] = v.contiguous()
    total = sum(v.numel() for v in shaped.values())
    pool = torch.empty(total, dtype=dt, device="cuda")
    tensors, off = {}, 0
    for b in weights:
        v = shaped[b]
        pool[off:off + v.numel()].copy_(v.reshape(-1))
        tensors[b] = pool[off:off + v.numel()].view(v.shape)
        off += v.numel()
    pool_bytes = pool.numel() * pool.element_size()
    print(f"weight pool: {len(weights)} buffers, {pool_bytes/2**20:.3f} MiB contiguous at "
          f"0x{pool.data_ptr():x}  ({', '.join(weights)})", flush=True)

    for b in order:
        if b in tensors:
            continue
        v = (ref[b] if b in ref else torch.zeros(1, dtype=torch.float64)).to("cuda", dt)
        if b in eff:
            v = v.permute((0,) + tuple(k + 1 for k in eff[b]))
        tensors[b] = v.contiguous()
    for b in spec.live_out:
        tensors[b] = torch.zeros_like(ref[b].to("cuda", dt))

    stream = cutlass.cuda.default_stream()
    call = tuple(from_dlpack(tensors[b], assumed_align=16) for b in order) + (
        Int32(sizes[spec.segment]), stream)
    print("compiling…", flush=True)
    fn = cute.compile(Kernel(), *call)
    fn(*call)
    torch.cuda.synchronize()

    eps_ratio = torch.finfo(dt).eps / torch.finfo(torch.float64).eps
    bound = ordering_bound(sched, {k: v.to("cuda") for k, v in ref.items()}) * eps_ratio

    # the raw CUDA stream torch is using, for the attribute call
    raw = torch.cuda.current_stream().cuda_stream

    def set_window(on: bool) -> int:
        w = AccessPolicyWindow()
        val = StreamAttrValue()
        if on:
            w.base_ptr = ctypes.c_void_p(pool.data_ptr())
            w.num_bytes = ctypes.c_size_t(pool_bytes)
            w.hitRatio = ctypes.c_float(1.0)
            w.hitProp = ACCESS_PERSISTING
            w.missProp = ACCESS_NORMAL
        else:
            w.base_ptr = ctypes.c_void_p(0)
            w.num_bytes = ctypes.c_size_t(0)
            w.hitRatio = ctypes.c_float(0.0)
            w.hitProp = ACCESS_NORMAL
            w.missProp = ACCESS_NORMAL
        val.accessPolicyWindow = w
        return rt.cudaStreamSetAttribute(ctypes.c_void_p(raw), STREAM_ATTR_ACCESS_POLICY,
                                         ctypes.byref(val))

    if args.configs:
        plan = []
        for spec_s in args.configs.split(";"):
            co, w = spec_s.split(":")
            plan.append((f"co{co}MiB_win{w}", int(co), w == "1"))
    else:
        plan = [("persist_off", 0, False), ("persist_on", args.carveout_mib, True)]

    results, base_out = {}, None
    for name, co_mib, on in plan:
        rc = rt.cudaDeviceSetLimit(LIMIT_PERSISTING_L2, ctypes.c_size_t(co_mib * 1024 * 1024))
        got = ctypes.c_size_t(0)
        rt.cudaDeviceGetLimit(ctypes.byref(got), LIMIT_PERSISTING_L2)
        print(f"\n[{name}] carve-out setLimit({co_mib} MiB) rc={rc}, reserved "
              f"{got.value/2**20:.2f} MiB; window={'on' if on else 'off'}", flush=True)
        rc = set_window(on)
        if rc != 0:
            print(f"{name}: cudaStreamSetAttribute rc={rc} -- NOT APPLIED, reporting as blocked",
                  flush=True)
            results[name] = {"status": "attribute_failed", "rc": rc}
            continue
        for b in spec.live_out:
            tensors[b].zero_()
        fn(*call)
        torch.cuda.synchronize()
        out = {b: tensors[b].clone() for b in spec.live_out}
        err = max(float((out[b].double() - ref[b].to("cuda")).abs().max()) for b in spec.live_out)
        if base_out is None:
            base_out, bitwise = out, True
        else:
            bitwise = all(torch.equal(out[b], base_out[b]) for b in spec.live_out)

        if args.ncu:
            torch.cuda.profiler.start()
            fn(*call)
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            results[name] = {"status": "profiled", "err": err, "bitwise_equal": bitwise}
            print(f"{name}: profiled one launch (no timing taken)", flush=True)
            continue
        for _ in range(args.warmup):
            fn(*call)
        torch.cuda.synchronize()
        s = []
        for _ in range(args.iters):
            a, bb = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            a.record(); fn(*call); bb.record(); torch.cuda.synchronize()
            s.append(a.elapsed_time(bb))
        s.sort()
        ms = s[len(s) // 2]
        results[name] = {"status": "ok" if err <= bound else "WRONG", "ms": ms, "err": err,
                         "bound": bound, "bitwise_equal": bitwise,
                         "spread_pct": (s[-1] - s[0]) / ms * 100}
        print(f"{name}: {ms:9.3f} ms  err {err:.3e} (bound {bound:.3e})  "
              f"{'bit-equal' if bitwise else 'DIFFERS'}  spread {results[name]['spread_pct']:.2f}%",
              flush=True)

    a, b = results.get("persist_off", {}), results.get("persist_on", {})
    if "ms" in a and "ms" in b:
        print(f"\npersistence effect: {a['ms']:.3f} -> {b['ms']:.3f} ms  = {a['ms']/b['ms']:.4f}x")
        print("  DRAM bytes are the real verdict and need the ncu pass; this is the wall-clock "
              "signal only.")
    pathlib.Path(args.out).write_text(json.dumps(
        {"fixture": args.fixture, "dtype": args.dtype, "pool_bytes": pool_bytes,
         "carveout_mib": args.carveout_mib, "arms": results}, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
