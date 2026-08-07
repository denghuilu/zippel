"""Per-kernel breakdown of the composed forward: which of the 55 dominate?

The S1c deficit is 0.036-0.107x with a *constant* per-edge cost, so it is per-thread work rather
than parallelism (D39 refuted occupancy). This locates the work.

Each kernel is timed individually with CUDA events, in the same composed program and on the same
buffers it runs on normally, so nothing about the measurement changes what is measured. Reported
alongside each kernel's static census -- terms, live-in element reads, template -- so the timing
can be compared against what the schedule says the kernel should cost.

**[measurement]** for the times; **[static analysis]** for the census. The two are reported side
by side precisely so that a discrepancy between them is visible rather than assumed away.

    python bench/s1c_kernel_profile.py --fixture si_medium --dtype f32
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
from codegen.compose import (DEFAULT_MAX_VOLUME, allocate,                  # noqa: E402
                             compile_program)
from codegen.schedule import analyze_group, build_schedule                 # noqa: E402
from codegen.tile import Ch                                                 # noqa: E402
from fixtures.load import load_batch                                        # noqa: E402


#: GH200 issue capacity. 132 SMs x 4 warp schedulers, each retiring at most one warp
#: instruction per cycle at 1.98 GHz; x32 lanes gives thread-instructions per second.
SMS, SCHEDULERS_PER_SM, CLOCK_HZ, LANES = 132, 4, 1.98e9, 32
PEAK_THREAD_INSTR_PER_S = SMS * SCHEDULERS_PER_SM * CLOCK_HZ * LANES


def static_census(prog, sched, template: str, sizes: dict) -> dict:
    """Per-thread work and cross-thread sharing, from the schedule alone. **[static analysis]**

    `intra_thread_reuse` is `factor_reads / distinct_reads` and is **NOT a cost**. For T1 it is
    reuse already banked: `codegen/emit.py:144-154` builds `wanted` as a `dict[str, set]` and
    emits exactly one load per distinct `(buffer, index)` into a register symbol, so a factor
    appearing five times costs one load. For T2 (`emit_tile.py:51-57`) and T3
    (`emit_reduce.py:158`) live-ins are inlined at each use and the repetition is textual;
    NVRTC common-subexpression-eliminates within a basic block, though predication chains may
    block that. Either way it measures reuse the symbol table or the backend already captures,
    which is why "hoist and register-cache" is not an intervention here -- it is the semantics.

    `sharing` is the quantity that *does* indicate a cost: for each live-in buffer,

        sharing[b] = (threads x distinct elements each thread reads of b) / globally unique
                     elements of b touched

    A weight read identically by every segment element comes out near the segment count; per-edge
    data read by one thread comes out near 1; per-edge data read by all channel-threads of its CTA
    comes out near the channel count. Anything with `sharing >> 1` and a footprint inside the
    shared-memory budget is a candidate for staging once per block instead of re-reading per
    thread.
    """
    live_in = set(sched.spec.live_in)
    ch = getattr(sched, "extent", 1)
    seg = sched.spec.segment
    n_seg = 1 if seg == "none" else sizes.get(seg, 1)

    # threads the kernel launches
    if template == "T2":
        threads = n_seg * ch
    else:
        threads = n_seg

    reads = 0
    per_buf_distinct: dict[str, set] = {}
    instr = 0                      # per-thread: one issue per factor load, one per multiply-add
    for a in sched.assigns:
        for t in a.terms:
            instr += max(len(t.factors), 1)          # (k-1) multiplies + 1 accumulate ~ k
            for f in t.factors:
                if f[0] in live_in:
                    n = ch if any(isinstance(i, Ch) for i in f[1]) else 1
                    reads += n
                    per_buf_distinct.setdefault(f[0], set()).add(f[1])
        if a.source is not None:
            instr += 4                                # transcendental/scalar-map, charged coarsely
            if a.source[0] in live_in:
                reads += 1
                per_buf_distinct.setdefault(a.source[0], set()).add(a.source[1])

    distinct = sum(len(v) for v in per_buf_distinct.values())

    sharing = {}
    for buf, idxs in per_buf_distinct.items():
        t = prog.type_of(buf)
        bseg = getattr(t, "segment", "none")
        bsizes = getattr(t, "sizes", ()) or (1,)
        elems_per_seg = 1
        for x in bsizes:
            elems_per_seg *= x
        n_bseg = 1 if bseg == "none" else sizes.get(bseg, 1)
        global_unique = n_bseg * elems_per_seg
        # elements one thread reads: a CH component spans the channel extent
        per_thread = 0
        for idx in idxs:
            per_thread += ch if any(isinstance(i, Ch) for i in idx) else 1
        total_reads = threads * per_thread
        itemsize = 4
        sharing[buf] = {"sharing": total_reads / max(global_unique, 1),
                        "footprint_kib": global_unique * itemsize / 1024,
                        "segment": bseg}

    # issue-bound floor: total thread-instructions / peak issue rate, with an efficiency interval
    total_instr = instr * threads
    lo = total_instr / PEAK_THREAD_INSTR_PER_S * 1e3            # 100 % issue efficiency
    hi = lo * 4.0                                                # 25 % issue efficiency
    return {"terms": sched.n_terms, "factor_reads": reads, "distinct_reads": distinct,
            "intra_thread_reuse": reads / max(distinct, 1),
            "instr_per_thread": instr, "threads": threads,
            "issue_bound_ms_lo": lo, "issue_bound_ms_hi": hi,
            "sharing": sharing}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="si_medium")
    ap.add_argument("--dtype", default="f32", choices=["f32", "f64"])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--max-volume", type=int, default=DEFAULT_MAX_VOLUME)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32
    from cutlass.cute.runtime import from_dlpack

    dt = torch.float32 if args.dtype == "f32" else torch.float64
    cfg = BlockConfig()
    torch.manual_seed(0)
    jd = [j.to(dt) for j in torch.load("blocks/Jd.pt", weights_only=False)]
    block = ESO2RefBlock(cfg).to("cpu", dt)
    batch = load_batch(args.fixture, "cpu", dt, cfg, requires_grad=False)
    prog, _ = build_forward(cfg, gauss_coeff=block.gauss_coeff)
    inputs, sizes = bind(block, batch, jd, cfg)

    from zippel.simplify import fusion_groups, simplify
    from codegen.compose import route
    simp = simplify(prog, keep=prog.outputs)
    groups = fusion_groups(simp, max_volume=args.max_volume)

    cp = compile_program(prog, sizes, f"prof_{args.fixture}", dtype=args.dtype,
                         max_volume=args.max_volume)
    env = allocate(cp, inputs, dtype=dt)
    stream = cutlass.cuda.default_stream()
    edges = sizes["edge"]

    # compile every kernel once, outside the timed region
    calls = []
    for g in cp.groups:
        a = tuple(from_dlpack(env[b], assumed_align=16) for b in g.order) + (
            Int32(sizes[g.driving_segment]), stream)
        calls.append((g, cute.compile(g.launch, *a), a))
    torch.cuda.synchronize()
    print(f"{args.fixture} {args.dtype}: {len(calls)} kernels, {edges} edges", flush=True)

    rows = []
    for g, fn, a in calls:
        for _ in range(args.warmup):
            fn(*a)
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.iters):
            s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            s.record()
            fn(*a)
            e.record()
            torch.cuda.synchronize()
            samples.append(s.elapsed_time(e))
        samples.sort()
        spec = analyze_group(simp, groups[g.index], name=g.name)
        _t, sched, _e = route(simp, spec)
        cen = static_census(simp, sched, g.template, sizes)
        rows.append({"group": g.index, "name": g.name, "template": g.template,
                     "ops": list(g.ops), "ms": samples[len(samples) // 2],
                     "terms": g.terms, **cen})

    total = sum(r["ms"] for r in rows)
    rows.sort(key=lambda r: -r["ms"])
    print(f"\nsum of individual kernels: {total:.2f} ms\n")
    print(f"{'rank':>4} {'grp':>4} {'tmpl':>5} {'ms':>9} {'share':>7} {'cum':>7} "
          f"{'instr/thr':>10} {'issue-bound ms':>18} {'measured/bound':>15}  ops")
    cum = 0.0
    for i, r in enumerate(rows[:12], 1):
        cum += r["ms"]
        lo, hi = r["issue_bound_ms_lo"], r["issue_bound_ms_hi"]
        print(f"{i:>4} {r['group']:>4} {r['template']:>5} {r['ms']:>9.3f} "
              f"{r['ms']/total:>6.1%} {cum/total:>6.1%} {r['instr_per_thread']:>10,} "
              f"{lo:>8.3f}-{hi:<9.3f} {r['ms']/max(lo,1e-9):>14.1f}x  "
              f"{','.join(r['ops'][:2])}", flush=True)

    print(f"\n=== cross-thread sharing: buffers worth staging in smem ===")
    print(f"{'grp':>4} {'buffer':>16} {'seg':>6} {'sharing':>10} {'footprint':>11}  verdict")
    seen = set()
    for r in rows[:8]:
        for buf, info in sorted(r["sharing"].items(), key=lambda kv: -kv[1]["sharing"]):
            if info["sharing"] < 2 or (r["group"], buf) in seen:
                continue
            seen.add((r["group"], buf))
            fits = info["footprint_kib"] <= 200          # ~228 KiB smem/SM on Hopper
            print(f"{r['group']:>4} {buf:>16} {info['segment']:>6} "
                  f"{info['sharing']:>9.1f}x {info['footprint_kib']:>9.1f}K  "
                  f"{'SMEM CANDIDATE' if fits else 'too large for smem'}", flush=True)

    by_t = {}
    for r in rows:
        by_t.setdefault(r["template"], [0.0, 0])
        by_t[r["template"]][0] += r["ms"]
        by_t[r["template"]][1] += 1
    print(f"\n{'template':>9} {'kernels':>8} {'ms':>10} {'share':>7}")
    for t in sorted(by_t):
        ms, n = by_t[t]
        print(f"{t:>9} {n:>8} {ms:>10.2f} {ms/total:>6.1%}")

    out = pathlib.Path(args.out or
                       f"bench/results/s1c_kernels_{args.fixture}_{args.dtype}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"fixture": args.fixture, "dtype": args.dtype, "edges": edges,
                               "total_ms": total, "kernels": rows}, indent=2))
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
