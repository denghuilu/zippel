"""Factorial on `conv1_90`: {weight-layout transpose, padded smem staging, both}. si_medium only.

`conv1_90` is 714.8 ms of the forward's 1392 ms (51.3 %). D42 attributes that to uncoalesced
weight access: `c1_w1a` is `none[j:2, o:128, k:2, c:256]`, T2 maps the thread index onto `o`, so
consecutive threads read 2 048 B apart and every warp load touches 32 distinct cache lines.
Static arithmetic predicts 651 ms against 714.8 measured, with no rival hypothesis within an
order of magnitude. This tests it, and attributes the effect.

Run at **fp32**, the dtype D42 was measured and predicted in.

## The arms

  A  transpose   permute the weight's trailing axes so the thread-mapped axis is innermost, and
                 hand in the correspondingly permuted tensor. Pure layout: same values, same
                 arithmetic, contiguous addresses across a warp. No capacity constraint, so it
                 reaches all four offending operands including the 512 KiB pair.
  B  smem        cooperative, coalesced load of a weight into shared memory once per block, then
                 per-thread reads from smem.
  A_matched      A restricted to exactly the operands B can stage. **The control that makes B
                 readable**: B is capacity-limited (below), so B < A could otherwise mean nothing
                 more than that A covered more operands.
  AB both        A on every operand it reaches, B on the ones it can stage.

**What B is and is not.** Each thread reads a *disjoint* slice of the weight -- thread `c` touches
only `o = c` -- so there is **no intra-block reuse to capture**. The 259 474x sharing measured in
3(a) is entirely cross-CTA, and shared memory cannot capture cross-CTA reuse; that is L2's job.
So B does not amortise re-reads here. It fixes *coalescing by a different route*: the cooperative
load is contiguous, and the scattered per-thread access is paid in smem rather than in global.
Stated before measuring, because it changes what a null result from B would mean.

**B's smem layout is padded, and the padding is load-bearing.** Slab-major, thread `o` owning
`sh[o*T + rest]`. The direct copy `T = S` puts every lane of a warp on one bank -- `S` is 256 or
512 here, both multiples of 32, so `(o*S + rest) % 32` does not depend on `o` -- which is a
**32-way bank conflict**: the same factor of 32 the arm exists to remove, relocated from HBM into
smem. `T = S + 1` for even `S` makes the stride odd, hence a unit mod 32, hence a bijection of
lanes onto banks. Costs 0.4 % of the footprint. Without it no B reading is interpretable, so it
is fixed here rather than discovered afterwards. (`codegen/emit_tile.py:staged_layout`.)

**B is capacity-limited and A is not.** At fp32 the two `c1_w1*` weights are 512 KiB each -- past
any block's shared memory -- so B cannot touch the two largest offenders at all. Even among the
128 KiB pair, two padded slabs are 257 KiB against a measured 224 KiB per-block ceiling, so B
stages **one** operand. That asymmetry is structural, not incidental, and it is why `A_matched`
exists. It is also already an argument for transpose as the default rule, independent of timing.

## Interpretation rules, fixed in advance

Let `dA`, `dB`, `dAB` be speedups over the unmodified kernel; compare `dB` against `dA_matched`,
never against the unrestricted `dA`.

  **coalescing-dominant**  `dA` large, `dB` ~ `dA_matched`, `dAB` ~ `dA` (no stacking).
      Both arms attack the same bottleneck by different routes, and it is the one D42 named.
      Consequence: **T2's default emission rule gains a layout requirement** -- a group's
      thread-mapped axis must be innermost in every operand it reads, enforced at emission by
      permuting the operand and its handle together. Transpose is preferred over smem: no
      capacity limit, no barrier, and it applies to operands too large to stage.

  **sharing-dominant**  `dB` >> `dA_matched`.
      Would contradict the disjoint-slice analysis above, so it is evidence that the sharing is
      *not* purely cross-CTA and that something is re-reading within a block. Consequence: the
      access-pattern model in 3(a) is wrong and must be rebuilt before it guides anything else;
      no emission rule changes until it is.

  **B << A_matched, padding confirmed**  staging loses on matched operands.
      Points at **staging overhead** -- the barrier, and touching every element twice -- and not
      at any statement about sharing, which the disjoint-slice analysis has already settled.
      Consequence: same layout requirement as coalescing-dominant, and staging is struck from the
      T2 rule rather than made conditional.

  **superadditive**  `dAB` >> `dA * dB`.
      The two fixes are not redundant: transpose makes the cooperative load itself coalesced,
      so staging becomes cheap only once the layout is right. Consequence: both enter the
      default rule, ordered -- layout first, staging second, staging conditional on capacity.

  **all-null**  none of the arms moves the kernel beyond measurement spread.
      D42's mechanism is refuted despite predicting the magnitude. Consequence: the 651-vs-715
      agreement was coincidence, the diagnosis returns to open, and the next instrument is a real
      profiler (uenv `ncu`, currently parked in REPORT 8.9) rather than more static analysis.
      **This is the outcome that costs the most to accept and is therefore written down first.**

## Correctness

Neither arm reorders arithmetic. A transpose is a pure layout change and staging is a pure
memory-path change; both evaluate the identical expression tree over the identical values in the
identical order. So the bar is not a tolerance but **bit-equality against the baseline arm** --
the sharpest check available here, and one that needs no bound at all. A wrong permutation, a
wrong smem index or a missing barrier all move the result by O(1) and cannot survive it.

Reported alongside: every arm against the FP64 interpreter within the ordering bound at fp32 unit
roundoff, which catches an error common to all arms that bit-equality between them cannot see.

An arm that is fast and wrong is not a result.

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
from codegen.emit_tile import (SMEM_CAP_BYTES, emit_tile_source,            # noqa: E402
                               staged_layout)
from codegen.schedule import analyze_group                                  # noqa: E402
from codegen.tile import Ch                                                 # noqa: E402
from fixtures.load import load_batch                                        # noqa: E402
from zippel.simplify import fusion_groups, simplify                         # noqa: E402


def thread_axis(sched, buf) -> int | None:
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
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--out", default="bench/results/s1c_factorial.json")
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
          f"channel extent {sched.extent}, dtype {args.dtype}", flush=True)

    # which live-ins carry the thread index on a non-innermost axis
    plan = {}
    for buf in spec.live_in:
        t = simp.type_of(buf)
        pos = thread_axis(sched, buf)
        if pos is None or t.segment != "none" or len(t.sizes) < 2 or pos == len(t.sizes) - 1:
            continue
        rank = len(t.sizes)
        lay = staged_layout(simp, sched, buf)
        plan[buf] = {"axis": pos, "rank": rank,
                     "perm": tuple([k for k in range(rank) if k != pos] + [pos]),
                     "kib": lay["extent"] * lay["T"] * itemsize / 1024}

    print("\noperands carrying the thread index on a non-innermost axis:")
    for b, p in sorted(plan.items(), key=lambda kv: kv[1]["kib"]):
        print(f"  {b:>10} {simp.type_of(b)}  axis {p['axis']}/{p['rank']-1} -> "
              f"perm {p['perm']}  padded slab {p['kib']:.1f} KiB  "
              f"{'stageable' if p['kib'] * 1024 <= SMEM_CAP_BYTES else 'exceeds the smem cap'}",
              flush=True)

    # B stages the largest subset that fits under the measured per-block ceiling, smallest first.
    stage, used = [], 0.0
    for b, p in sorted(plan.items(), key=lambda kv: kv[1]["kib"]):
        if (used + p["kib"]) * 1024 <= SMEM_CAP_BYTES:
            stage.append(b)
            used += p["kib"]
    stage = tuple(stage)
    transpose = {b: p["perm"] for b, p in plan.items()}
    matched = {b: transpose[b] for b in stage}
    print(f"\nB stages {', '.join(stage) or '(nothing)'} -- {used:.1f} KiB of the "
          f"{SMEM_CAP_BYTES/1024:.0f} KiB cap. A reaches all {len(transpose)}.", flush=True)
    print(f"A_matched transposes exactly {', '.join(matched) or '(nothing)'}.", flush=True)

    arms = {"baseline": ({}, ()),
            "A_transpose": (transpose, ()),
            "A_matched": (matched, ()),
            "B_smem": ({}, stage),
            "AB_both": (transpose, stage)}

    from zippel.interp import run
    ref = run(simp, inputs, sizes)
    stream = cutlass.cuda.default_stream()
    # fp32 unit roundoff against an ordering bound derived at fp64; the schedule is identical, so
    # only the roundoff scales. Not a loosened tolerance -- the same bound at the right epsilon.
    eps_ratio = torch.finfo(dt).eps / torch.finfo(torch.float64).eps
    bound = ordering_bound(sched, {k: v.to("cuda") for k, v in ref.items()}) * eps_ratio

    results, base_out = {}, None
    for name, (tr, st) in arms.items():
        try:
            src = emit_tile_source(simp, sched, dtype=args.dtype, transpose=tr, stage_shared=st)
            Kernel, order = build_kernel(src, f"fact_{name}_{args.dtype}", sched=sched)
        except Exception as exc:                                     # noqa: BLE001
            print(f"\n{name}: NOT EMITTED -- {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            results[name] = {"status": "not_emitted", "error": str(exc)[:300]}
            continue

        tensors = {}
        for b in order:
            v = (ref[b] if b in ref else torch.zeros(1, dtype=torch.float64)).to("cuda", dt)
            if b in tr:
                v = v.permute((0,) + tuple(k + 1 for k in tr[b]))
            tensors[b] = v.contiguous()
        for b in spec.live_out:
            tensors[b] = torch.zeros_like(ref[b].to("cuda", dt))

        call = tuple(from_dlpack(tensors[b], assumed_align=16) for b in order) + (
            Int32(sizes[spec.segment]), stream)
        try:
            fn = cute.compile(Kernel(), *call)
            fn(*call)
            torch.cuda.synchronize()
        except Exception as exc:                                     # noqa: BLE001
            print(f"\n{name}: FAILED TO RUN -- {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            results[name] = {"status": "failed", "error": str(exc)[:300]}
            continue

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
            a.record()
            fn(*call)
            bb.record()
            torch.cuda.synchronize()
            samples.append(a.elapsed_time(bb))
        samples.sort()
        ms = samples[len(samples) // 2]
        results[name] = {"status": "ok" if ok else "WRONG", "ms": ms, "err": err,
                         "bound": bound, "bitwise_equal_to_baseline": bitwise,
                         "max_diff_vs_baseline": dbase,
                         "spread_pct": (samples[-1] - samples[0]) / ms * 100}
        print(f"\n{name:>12}: {ms:9.3f} ms   vs interp {err:.3e} (bound {bound:.3e})   "
              f"vs baseline {'BIT-EQUAL' if bitwise else f'{dbase:.3e}'}   "
              f"{'ok' if ok else 'WRONG -- not a result'}", flush=True)

    base = results.get("baseline", {}).get("ms")
    print(f"\n{'arm':>12} {'ms':>10} {'speedup':>9} {'spread':>8} {'bit-eq':>8} {'status':>8}")
    for name, r in results.items():
        if r.get("status") in ("not_emitted", "failed"):
            print(f"{name:>12} {'--':>10} {'--':>9} {'--':>8} {'--':>8} {r['status']:>8}")
            continue
        sp = base / r["ms"] if base and r.get("ms") else float("nan")
        print(f"{name:>12} {r['ms']:>10.3f} {sp:>8.3f}x {r['spread_pct']:>7.2f}% "
              f"{str(r['bitwise_equal_to_baseline']):>8} {r['status']:>8}", flush=True)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"fixture": args.fixture, "dtype": args.dtype, "group": gi,
         "staged": list(stage), "smem_kib": used, "smem_cap_kib": SMEM_CAP_BYTES / 1024,
         "plan": {b: {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()}
                  for b, p in plan.items()},
         "arms": results}, indent=2))
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
