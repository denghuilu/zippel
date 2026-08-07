"""Aggregate N independent verdict allocations: median-of-medians and full range.

The protocol (slurm/verdict.sbatch) runs N = 5 *separate* allocations rather than N repeats
inside one, because between-node and between-placement variance is the dominant term for
dispatch-bound configurations and is invisible to repeats within a single allocation.

Reported: median-of-medians as the central value, and the **full range** across allocations
as the error bar. A within-allocation IQR is reported alongside, but it is the smaller and
more flattering number and is never the headline.

    python bench/aggregate_verdict.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="verdict_rep*.json")
    ap.add_argument("--out", default="bench/results/verdict_summary.json")
    args = ap.parse_args()

    results = pathlib.Path("bench/results")
    reps = sorted(results.glob(args.glob))
    if not reps:
        print(f"no files matching {args.glob} in {results} -- run slurm/verdict.sbatch first")
        return

    by_key: dict[tuple, list[dict]] = {}
    for f in reps:
        for r in json.loads(f.read_text()):
            if r.get("error"):
                continue
            by_key.setdefault((r["fixture"], r["precision"]), []).append(r)

    print(f"{len(reps)} allocations: {', '.join(f.name for f in reps)}\n")
    print(f"{'fixture':10s} {'prec':5s} {'med-of-med':>11s} {'range':>18s} {'spread%':>8s} "
          f"{'hosts':>6s} {'max in-alloc IQR%':>18s}")
    rows = []
    for (fx, pr), rs in sorted(by_key.items()):
        meds = sorted(r["median_ms"] for r in rs)
        mom = statistics.median(meds)
        lo, hi = meds[0], meds[-1]
        spread = 100 * (hi - lo) / mom if mom else float("nan")
        iqr_pct = max(100 * r["iqr_ms"] / r["median_ms"] for r in rs)
        hosts = len({r.get("host", "?") for r in rs})
        print(f"{fx:10s} {pr:5s} {mom:11.2f} {f'{lo:.2f}-{hi:.2f}':>18s} {spread:7.1f}% "
              f"{hosts:6d} {iqr_pct:17.1f}%")
        rows.append({"fixture": fx, "precision": pr, "n_allocations": len(rs),
                     "median_of_medians_ms": mom, "min_ms": lo, "max_ms": hi,
                     "spread_pct": spread, "max_in_alloc_iqr_pct": iqr_pct,
                     "distinct_hosts": hosts,
                     "hosts": sorted({r.get("host", "?") for r in rs})})

    print("\nspread% is the honest error bar: full range across allocations, over the")
    print("median-of-medians. Where it greatly exceeds the in-allocation IQR, the dominant")
    print("variance is between allocations (node, placement), not within a measurement.")

    out = pathlib.Path(args.out)
    out.write_text(json.dumps({"n_allocations": len(reps), "rows": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
