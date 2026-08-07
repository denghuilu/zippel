"""How does schedule construction scale with term count? (S3 entry criterion)

The dbwd scale preflight stalled: 320 acyclic groups, and after fifteen minutes it had not
emitted a single row. The AST-chunking fix was never the bottleneck -- **building the schedule**
is. That matters more than the emission question it was meant to answer, because every Phase 2
stage runs the constructor over every group, and S3's programs are 9x the forward's.

So: measure the exponent. `t ~ n^k` for term count `n`; near-linear (k <= 1.2) means S3 is a
matter of wall-clock, and anything approaching quadratic means the constructor needs work before
S3 rather than after.

Group size is predicted *analytically* first -- summing each path's index-space volume, which is
O(paths) arithmetic -- so a spread of sizes can be chosen without paying construction to discover
them. That was the preflight's mistake: it sorted candidates by `n_terms`, which requires
building every schedule to learn which are big.

    python bench/schedule_scaling.py --budget 90
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import math
import pathlib
import pstats
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from blocks.eso2_ir import build_dbwd, build_force, build_forward       # noqa: E402
from blocks.eso2_ref import BlockConfig                                 # noqa: E402
from codegen.schedule import (analyze_group, build_schedule,            # noqa: E402
                              index_maps_used)
from codegen.tile import build_tile_schedule, channel_axis              # noqa: E402
from zippel.simplify import fusion_groups, simplify                     # noqa: E402


def predicted_terms(prog, spec) -> int:
    """Index-space volume of a group, in O(paths) arithmetic -- no schedule built.

    Upper bound on the emitted term count: it ignores structural sparsity, which only removes
    terms. Good enough to rank groups by size, which is all it is for.
    """
    total = 0
    for name in spec.ops:
        op = prog.ops[name]
        if op.kind == "scalar_map":
            total += math.prod(op.out_type.sizes or (1,))
            continue
        for p in op.paths:
            specs, out_spec = p.parse()
            extent: dict[str, int] = {}
            for pos, j in enumerate(p.operands):
                t = prog.type_of(op.inputs[j])
                sl = p.slices_for(pos)
                sizes = (tuple(len(range(*s.indices(f))) for s, f in zip(sl, t.sizes))
                         if sl else t.sizes)
                for ch, size in zip(specs[pos], sizes):
                    extent[ch] = size
            total += math.prod(extent.values()) if extent else 1
    return total


def build_timed(prog, spec):
    """Build a group's schedule under whichever template applies, returning (seconds, terms)."""
    t0 = time.perf_counter()
    sched = build_schedule(prog, spec)
    if sched.peak_live_values() > 168:
        axis = channel_axis(prog, spec)
        if axis is None:
            return None
        sched = build_tile_schedule(prog, spec, *axis)
    return time.perf_counter() - t0, sched.n_terms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=90.0,
                    help="seconds; stop sampling groups once exceeded")
    ap.add_argument("--out", default="bench/results/schedule_scaling.json")
    args = ap.parse_args()

    payload = {"programs": {}}
    for label, build in (("fwd", build_forward), ("force", build_force), ("dbwd", build_dbwd)):
        prog, _ = build(BlockConfig())
        simp = simplify(prog, keep=prog.outputs)
        groups = fusion_groups(simp)

        specs = []
        for gi, g in enumerate(groups):
            spec = analyze_group(simp, g, name=f"g{gi}")
            if index_maps_used(simp, spec):
                continue
            specs.append((predicted_terms(simp, spec), gi, spec))
        specs.sort()
        print(f"\n=== {label}: {len(groups)} groups, {len(specs)} schedulable ===", flush=True)
        print(f"predicted index-space volume: min {specs[0][0]:,} "
              f"median {specs[len(specs) // 2][0]:,} max {specs[-1][0]:,}", flush=True)

        # sample across the size range, largest last so the budget cuts the expensive tail
        picks, step = [], max(len(specs) // 12, 1)
        for i in range(0, len(specs), step):
            picks.append(specs[i])
        if specs[-1] not in picks:
            picks.append(specs[-1])

        rows, spent = [], 0.0
        print(f"{'group':>7} {'predicted':>12} {'terms':>12} {'build s':>9}", flush=True)
        for pred, gi, spec in picks:
            if spent > args.budget:
                print(f"  budget exhausted; {len(picks) - len(rows)} larger groups not sampled",
                      flush=True)
                break
            got = build_timed(simp, spec)
            if got is None:
                continue
            dt, terms = got
            spent += dt
            rows.append({"group": gi, "predicted": pred, "terms": terms, "seconds": dt})
            print(f"{gi:>7} {pred:>12,} {terms:>12,} {dt:>9.3f}", flush=True)

        usable = [r for r in rows if r["terms"] > 0 and r["seconds"] > 1e-4]

        def fit(key):
            """log-log slope and R^2 of seconds against `key`."""
            xs = [math.log(r[key]) for r in usable if r[key] > 0]
            ys = [math.log(r["seconds"]) for r in usable if r[key] > 0]
            n = len(xs)
            if n < 3:
                return None, None
            mx, my = sum(xs) / n, sum(ys) / n
            sxx = sum((x - mx) ** 2 for x in xs)
            if not sxx:
                return None, None
            k = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
            ss_tot = sum((y - my) ** 2 for y in ys)
            ss_res = sum((y - (my + k * (x - mx))) ** 2 for x, y in zip(xs, ys))
            return k, (1 - ss_res / ss_tot if ss_tot else float("nan"))

        k_terms, r2_terms = fit("terms")
        k_vol, r2_vol = fit("predicted")
        if k_vol is not None:
            # Both are reported because which one FITS is the finding: cost is driven by the
            # dense index space the constructor walks, not by the sparse terms it emits.
            print(f"  t ~ terms^{k_terms:.2f} (R^2 {r2_terms:.2f})   "
                  f"t ~ volume^{k_vol:.2f} (R^2 {r2_vol:.2f})", flush=True)
        payload["programs"][label] = {
            "rows": rows,
            "exponent_vs_terms": k_terms, "r2_vs_terms": r2_terms,
            "exponent_vs_volume": k_vol, "r2_vs_volume": r2_vol,
            "max_terms_built": max((r["terms"] for r in usable), default=0),
            "max_volume_built": max((r["predicted"] for r in usable), default=0)}

    # profile the largest group actually built, for dbwd
    dbwd = payload["programs"].get("dbwd", {})
    if dbwd.get("rows"):
        prog, _ = build_dbwd(BlockConfig())
        simp = simplify(prog, keep=prog.outputs)
        groups = fusion_groups(simp)
        biggest = max(dbwd["rows"], key=lambda r: r["terms"])
        spec = analyze_group(simp, groups[biggest["group"]], name="profile")
        print(f"\n=== cProfile: dbwd g{biggest['group']}, {biggest['terms']:,} terms ===",
              flush=True)
        pr = cProfile.Profile()
        pr.enable()
        build_timed(simp, spec)
        pr.disable()
        buf = io.StringIO()
        pstats.Stats(pr, stream=buf).sort_stats("cumulative").print_stats(12)
        text = buf.getvalue()
        print(text, flush=True)
        payload["profile"] = {"group": biggest["group"], "terms": biggest["terms"],
                              "text": text}

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
