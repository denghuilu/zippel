"""Three-arm factorial on `conv1_90`: {transpose, smem staging, both}. si_medium only.

`conv1_90` is 714.8 ms of the forward's 1392 ms (51.3 %). D42 attributes that to uncoalesced
weight access: `c1_w1a` is `none[j:2, o:128, k:2, c:256]`, T2 maps the thread index onto `o`, so
consecutive threads read 2 048 B apart and every warp load touches 32 distinct cache lines.
Static arithmetic predicts 651 ms against 714.8 measured, with no rival hypothesis within an
order of magnitude. This tests it, and attributes the effect.

## The arms

  A  transpose   permute the weight's trailing axes so the thread-mapped axis is innermost, and
                 hand in the correspondingly permuted tensor. Pure layout: same values, same
                 arithmetic, contiguous addresses across a warp. No capacity constraint, so it
                 applies to `c1_w1a`/`c1_w1b` (512 KiB) which cannot enter smem untiled at all.
  B  smem        cooperative, coalesced load of a weight into shared memory once per block, then
                 per-thread reads from smem.
  AB both        together.

**What B is and is not.** Each thread reads a *disjoint* slice of the weight — thread `c` touches
only `o = c` — so there is **no intra-block reuse to capture**. The 259 474x sharing measured in
3(a) is entirely cross-CTA, and shared memory cannot capture cross-CTA reuse; that is L2's job.
So B does not amortise re-reads here. It fixes *coalescing by a different route*: the cooperative
load is contiguous, and the scattered per-thread access is paid in smem rather than in global.
Stated before measuring, because it changes what a null result from B would mean.

## Interpretation rules, fixed in advance

Let `dA`, `dB`, `dAB` be the speedups of each arm over the unmodified kernel.

  **coalescing-dominant**  `dA` large, `dB` ~ `dA`, `dAB` ~ `dA` (no stacking).
      Both arms attack the same bottleneck by different routes, and it is the one D42 named.
      Consequence: **T2's default emission rule gains a layout requirement** — a group's
      thread-mapped axis must be innermost in every operand it reads, enforced at emission by
      permuting the operand and its handle together. Transpose is preferred over smem: no
      capacity limit, no barrier, and it applies to operands too large to stage.

  **sharing-dominant**  `dB` >> `dA`.
      Would contradict the disjoint-slice analysis above, so it is evidence that the sharing is
      *not* purely cross-CTA and that something is re-reading within a block. Consequence: the
      access-pattern model in 3(a) is wrong and must be rebuilt before it guides anything else;
      no emission rule changes until it is.

  **superadditive**  `dAB` >> `dA * dB`.
      The two fixes are not redundant: transpose makes the cooperative load itself coalesced,
      so staging becomes cheap only once the layout is right. Consequence: both enter the
      default rule, ordered — layout first, staging second, staging conditional on capacity.

  **all-null**  none of the three moves the kernel beyond measurement spread.
      D42's mechanism is refuted despite predicting the magnitude. Consequence: the 651-vs-715
      agreement was coincidence, the diagnosis returns to open, and the next instrument is a real
      profiler (uenv `ncu`, currently parked in REPORT 8.9) rather than more static analysis.
      **This is the outcome that costs the most to accept and is therefore written down first.**

Correctness is checked against the FP64 interpreter on every arm, using the bound the emitter
ships. An arm that is fast and wrong is not a result.

si_medium only. The si_small regime check belongs with the post-adoption composition re-measure,
not inside the factorial, so that the arms are compared at one problem size.

    python bench/s1c_factorial.py
"""

from __future__ import annotations

import argparse
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
from codegen.tile import Ch                                                 # noqa: E402
from fixtures.load import load_batch                                        # noqa: E402
from zippel.simplify import fusion_groups, simplify                         # noqa: E402

#: ~228 KiB of shared memory per SM on Hopper; a block staging more than this cannot run.
SMEM_BUDGET_KIB = 200


def channel_axis_of(prog, sched, buf) -> int | None:
    """Which trailing axis of `buf` the thread index lands on, or None if it is not indexed by it."""
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
    ap.add_argument("--dtype", default="f32", choices=["f32", "f64"])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--out", default="bench/results/s1c_factorial.json")
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
    batch = load_batch(args.fixture, "cpu", torch.float64, cfg, requires_grad=False)
    prog, _ = build_forward(cfg, gauss_coeff=block.gauss_coeff)
    simp = simplify(prog, keep=prog.outputs)
    inputs, sizes = bind(block, batch, jd, cfg)

    groups = fusion_groups(simp, max_volume=10_000)
    gi = next(i for i, g in enumerate(groups) if "conv1_90" in g)
    spec = analyze_group(simp, groups[gi], name="conv1_90")
    template, sched, _ = route(simp, spec)
    assert template == "T2", f"expected T2, got {template}"
    print(f"conv1_90 = group {gi}, {template}, {sched.n_terms:,} terms, "
          f"channel extent {sched.extent}", flush=True)

    # which live-ins carry the thread index, and where
    plan = {}
    for buf in spec.live_in:
        t = simp.type_of(buf)
        pos = channel_axis_of(simp, sched, buf)
        if pos is None or t.segment != "none" or len(t.sizes) < 2:
            continue
        rank = len(t.sizes)
        if pos == rank - 1:
            continue                                   # already innermost
        perm = tuple([k for k in range(rank) if k != pos] + [pos])
        elems = 1
        for x in t.sizes:
            elems *= x
        plan[buf] = {"axis": pos, "rank": rank, "perm": perm,
                     "kib": elems * (4 if dt is torch.float32 else 8) / 1024}
    print("\noperands carrying the thread index on a non-innermost axis:")
    for b, p in plan.items():
        print(f"  {b:>10} {simp.type_of(b)}  axis {p['axis']}/{p['rank']-1} -> perm {p['perm']}  "
              f"{p['kib']:.0f} KiB  {'stageable' if p['kib'] <= SMEM_BUDGET_KIB else 'too large for smem'}",
              flush=True)

    transpose = {b: p["perm"] for b, p in plan.items()}
    stageable = tuple(b for b, p in plan.items() if p["kib"] <= SMEM_BUDGET_KIB)
    arms = {"baseline": ({}, ()),
            "A_transpose": (transpose, ()),
            "B_smem": ({}, stageable),
            "AB_both": (transpose, stageable)}

    # reference values from the interpreter, once
    from zippel.interp import run
    ref = run(simp, inputs, sizes)
    stream = cutlass.cuda.default_stream()

    results = {}
    for name, (tr, st) in arms.items():
        try:
            src = emit_tile_source(simp, sched, dtype="f64", transpose=tr, stage_shared=st)
            Kernel, order = build_kernel(src, f"fact_{name}_f64", sched=sched)
        except Exception as exc:                                     # noqa: BLE001
            print(f"\n{name}: NOT EMITTED -- {type(exc).__name__}: {str(exc)[:90]}", flush=True)
            results[name] = {"status": "not_emitted", "error": str(exc)[:200]}
            continue

        tensors = {}
        for b in order:
            v = ref[b].to("cuda", torch.float64) if b in ref else None
            if v is None:
                v = torch.zeros(1, dtype=torch.float64, device="cuda")
            if b in tr:
                v = v.permute((0,) + tuple(k + 1 for k in tr[b])).contiguous()
            tensors[b] = v.contiguous()
        for b in spec.live_out:
            tensors[b] = torch.zeros_like(ref[b].to("cuda", torch.float64))

        call = tuple(from_dlpack(tensors[b], assumed_align=16) for b in order) + (
            Int32(sizes[spec.segment]), stream)
        try:
            fn = cute.compile(Kernel(), *call)
            fn(*call)
            torch.cuda.synchronize()
        except Exception as exc:                                     # noqa: BLE001
            print(f"\n{name}: FAILED TO RUN -- {type(exc).__name__}: {str(exc)[:90]}", flush=True)
            results[name] = {"status": "failed", "error": str(exc)[:200]}
            continue

        bound = ordering_bound(sched, {k: v.to("cuda") for k, v in ref.items()})
        err = max(float((tensors[b] - ref[b].to("cuda")).abs().max()) for b in spec.live_out)
        ok = err <= bound

        for _ in range(args.warmup):
            fn(*call)
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.iters):
            a, bb = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            a.record()
            fn(*call)
            bb.record()
            torch.cuda.synchronize()
            samples.append(a.elapsed_time(bb))
        samples.sort()
        ms = samples[len(samples) // 2]
        results[name] = {"status": "ok" if ok else "WRONG", "ms": ms,
                         "err": err, "bound": bound,
                         "spread_pct": (samples[-1] - samples[0]) / ms * 100}
        print(f"\n{name:>12}: {ms:9.3f} ms   err {err:.2e} vs bound {bound:.2e}   "
              f"{'ok' if ok else 'WRONG -- not a result'}", flush=True)

    base = results.get("baseline", {}).get("ms")
    print(f"\n{'arm':>12} {'ms':>10} {'speedup':>9} {'correct':>8}")
    for name, r in results.items():
        if r.get("status") in ("not_emitted", "failed"):
            print(f"{name:>12} {'--':>10} {'--':>9}   {r['status']}")
            continue
        sp = base / r["ms"] if base and r.get("ms") else float("nan")
        print(f"{name:>12} {r['ms']:>10.3f} {sp:>8.3f}x {r['status']:>8}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"fixture": args.fixture, "dtype": "f64", "group": gi,
                               "plan": {b: {k: (list(v) if isinstance(v, tuple) else v)
                                            for k, v in p.items()} for b, p in plan.items()},
                               "arms": results}, indent=2))
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
