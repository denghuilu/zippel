"""Lever (a): edge-batched CTAs on `conv1_90`. Bit-equality first, then the route probe.

Design and predictions are pre-registered in `docs/edge_batch_design.md` (D69), corrected by D76
(the reuse is intra-thread, so a register is the vehicle and smem would stage for an audience of
one), D78 (registers are `chunk + 9·E_c`, so the refusal moves to `E_c`=16 at `CHUNK`=48) and D80
(A ≈ B is predicted; the live contrast is `{A, B}` vs `C`).

**Order matters and is not negotiable: bit-equality against `E_c`=1 before any timing is
reported.** Tiling and interleaving preserve the summation order per (assignment, edge) by
construction, so the bar is bit-equality and not a tolerance. An arm that is fast and wrong is not
a result — the standing rule, and the reason it is checked first here rather than alongside.

## Compile time is split, because it is not one quantity

    parse    `build_kernel` — writing and importing the generated Python module
    compile  `cute.compile` — DSL trace plus MLIR/NVRTC/ptxas

Route A pays parse on `5 123 × E_c` terms of source; route B would pay it on `5 123`. D80 predicts
that difference is negligible because parse tracks *statements*, which the chunk-interleaved
structure holds nearly flat (141 → 5 392 → 5 644 for `E_c` = 1 → 2 → 4). Reported separately so the
prediction is falsifiable rather than assumed.

## Registers and the hoist are NOT read from these numbers

Per D78: `launch__registers_per_thread` comes from `ncu`, and whether the weight load sits hoisted
outside the unrolled copies comes from **disassembly**. Nothing about either is inferred from
compile time or runtime here. This script measures time and correctness; attribution is a separate
run with a separate instrument.

## Sweep shape (D79)

Frontier walk, not grid: climb `E_c` at `CHUNK`=48 until the register guard refuses, drop `CHUNK`
only where forced. **Refusals print as data rows with their arithmetic**, never as omissions.

    python bench/s1c_edge_batch.py --arms 1,2,4
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from blocks.eso2_ir import build_forward                                    # noqa: E402
from blocks.eso2_ref import BlockConfig, ESO2RefBlock                       # noqa: E402
from blocks.ir_bind import bind                                             # noqa: E402
from codegen.bounds import ordering_bound                                   # noqa: E402
from codegen.compose import route                                           # noqa: E402
from codegen.emit import build_kernel                                       # noqa: E402
from codegen.emit_common import REGISTER_BUDGET                             # noqa: E402
from codegen.emit_tile import emit_tile_source                              # noqa: E402
from codegen.schedule import analyze_group                                  # noqa: E402
from fixtures.load import load_batch                                        # noqa: E402
from zippel.simplify import fusion_groups, simplify                         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="si_medium")
    ap.add_argument("--dtype", default="f32")
    ap.add_argument("--arms", default="1,2,4", help="E_c ladder")
    ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", default="bench/results/s1c_edge_batch.json")
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

    # si_medium shapes, si_small values tiled up (D49): the interpreter materialises every
    # intermediate at once and is terabyte-scale at si_medium. Timing is input-independent here.
    batch_s = load_batch("si_small", "cpu", torch.float64, cfg, requires_grad=False)
    inputs_s, sizes_s = bind(block, batch_s, jd, cfg)
    batch_m = load_batch(args.fixture, "cpu", torch.float64, cfg, requires_grad=False)
    _im, sizes = bind(block, batch_m, jd, cfg)

    groups = fusion_groups(simp, max_volume=10_000)
    gi = next(i for i, g in enumerate(groups) if "conv1_90" in g)
    spec = analyze_group(simp, groups[gi], name="conv1_90")
    template, sched, _ = route(simp, spec)
    assert template == "T2"
    print(f"conv1_90 = group {gi}, {sched.n_terms:,} terms, channel extent {sched.extent}, "
          f"{args.dtype}, chunk={args.chunk}, budget={REGISTER_BUDGET}", flush=True)

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
    eps_ratio = torch.finfo(dt).eps / torch.finfo(torch.float64).eps
    bound = ordering_bound(sched, {k: v.to("cuda") for k, v in ref.items()}) * eps_ratio

    results, base_out = {}, None
    for E in [int(x) for x in args.arms.split(",")]:
        name = f"E{E}"
        t0 = time.perf_counter()
        try:
            src = emit_tile_source(simp, sched, dtype=args.dtype,
                                   edge_batch=E, chunk=args.chunk)
        except ValueError as exc:
            # A refusal is a data row, with its arithmetic, never an omission (D79).
            print(f"\n{name}: REFUSED -- {exc}", flush=True)
            results[name] = {"status": "refused", "reason": str(exc),
                             "predicted_registers": args.chunk + 9 * E}
            continue
        t_emit = time.perf_counter() - t0
        stmts = sum(1 for ln in src.splitlines() if "=" in ln)

        t0 = time.perf_counter()
        Kernel, order = build_kernel(src, f"eb_{name}_c{args.chunk}_{args.dtype}", sched=sched)
        t_parse = time.perf_counter() - t0

        module = sys.modules[f"zippel_generated.eb_{name}_c{args.chunk}_{args.dtype}"]
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

        t0 = time.perf_counter()
        try:
            fn = cute.compile(Kernel(), *call)
        except Exception as exc:                                    # noqa: BLE001
            print(f"\n{name}: COMPILE FAILED -- {type(exc).__name__}: {str(exc)[:140]}", flush=True)
            results[name] = {"status": "compile_failed", "error": str(exc)[:300],
                             "emit_s": t_emit, "parse_s": t_parse, "statements": stmts}
            continue
        t_compile = time.perf_counter() - t0
        fn(*call)
        torch.cuda.synchronize()

        out = {b: tensors[b].clone() for b in spec.live_out}
        err = max(float((out[b].double() - ref[b].to("cuda")).abs().max()) for b in spec.live_out)
        if base_out is None:
            base_out, bitwise, dbase = out, True, 0.0
        else:
            bitwise = all(torch.equal(out[b], base_out[b]) for b in spec.live_out)
            dbase = max(float((out[b] - base_out[b]).abs().max()) for b in spec.live_out)
        ok = err <= bound and dbase <= bound

        for _ in range(args.warmup):
            fn(*call)
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.iters):
            a, bb = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            a.record(); fn(*call); bb.record(); torch.cuda.synchronize()
            samples.append(a.elapsed_time(bb))
        samples.sort()
        ms = samples[len(samples) // 2]
        results[name] = {"status": "ok" if ok else "WRONG", "E_c": E, "chunk": args.chunk,
                         "ms": ms, "err": err, "bound": bound,
                         "bitwise_equal_to_E1": bitwise, "max_diff_vs_E1": dbase,
                         "emit_s": t_emit, "parse_s": t_parse, "compile_s": t_compile,
                         "statements": stmts,
                         "predicted_registers": args.chunk + 9 * E,
                         "spread_pct": (samples[-1] - samples[0]) / ms * 100}
        print(f"\n{name}: {ms:9.3f} ms | emit {t_emit:6.2f}s parse {t_parse:6.2f}s "
              f"compile {t_compile:7.1f}s | {stmts:,} stmts | "
              f"vs E1 {'BIT-EQUAL' if bitwise else f'{dbase:.3e}'} | "
              f"{'ok' if ok else 'WRONG -- not a result'}", flush=True)

    base = results.get("E1", {}).get("ms")
    print(f"\n{'arm':>6} {'ms':>10} {'speedup':>9} {'stmts':>9} {'parse s':>9} {'compile s':>10} "
          f"{'pred regs':>10} {'bit-eq':>8}  status")
    for n, r in results.items():
        if r.get("status") == "refused":
            print(f"{n:>6} {'--':>10} {'--':>9} {'--':>9} {'--':>9} {'--':>10} "
                  f"{r['predicted_registers']:>10} {'--':>8}  REFUSED (register guard)")
            continue
        if "ms" not in r:
            print(f"{n:>6} {'--':>10} {'--':>9} {r.get('statements','--'):>9} "
                  f"{'--':>9} {'--':>10} {'--':>10} {'--':>8}  {r['status']}")
            continue
        sp = base / r["ms"] if base else float("nan")
        print(f"{n:>6} {r['ms']:>10.3f} {sp:>8.3f}x {r['statements']:>9,} {r['parse_s']:>9.2f} "
              f"{r['compile_s']:>10.1f} {r['predicted_registers']:>10} "
              f"{str(r['bitwise_equal_to_E1']):>8}  {r['status']}")

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"fixture": args.fixture, "dtype": args.dtype,
                                    "chunk": args.chunk, "group": gi,
                                    "budget": REGISTER_BUDGET, "arms": results}, indent=2))
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
