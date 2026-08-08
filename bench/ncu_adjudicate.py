"""Apply the pre-registered adjudication table to the ncu report. Decides nothing on its own.

Every rule here was fixed in `bench/ncu_profile.py`'s docstring before the profiler ran. This
script only evaluates them, so that the reading is arithmetic rather than judgement — which is the
whole point after two static models produced numbers that agreed with a measurement and were
wrong anyway (D53, D57).

    python bench/ncu_adjudicate.py

Kernels are identified by launch order, which the driver fixes: 0 = baseline, 1 = A_transpose,
2 = B_smem.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

#: Measured wall-clock from the factorial (D53), fp32 si_medium. Row 2's rule is stated against
#: these, so they are constants here rather than re-measured.
MEASURED_MS = {"baseline": 714.819, "A_transpose": 582.023, "B_smem": 1852.103}
ARM_BY_LAUNCH = ["baseline", "A_transpose", "B_smem"]

M_SECTORS = "l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio"
M_OCC = "sm__warps_active.avg.pct_of_peak_sustained_active"
M_STALL = {
    "long_scoreboard": "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "mio_throttle": "smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",
    "barrier": "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
    "no_instruction": "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio",
}
M_L2_HIT = "lts__t_sector_hit_rate.pct"
M_DRAM_BW = "dram__bytes.sum.per_second"
M_L2_BW = "lts__t_bytes.sum.per_second"


def load(path: pathlib.Path) -> dict[str, dict[str, float]]:
    """ncu `--csv --page raw` is long-format: one row per (launch, metric)."""
    per_launch: dict[int, dict[str, float]] = {}
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    idcol = next((c for c in rows[0] if c.strip().upper() == "ID"), None)
    namecol = next((c for c in rows[0] if "Metric Name" in c), None)
    valcol = next((c for c in rows[0] if "Metric Value" in c), None)
    if not (idcol and namecol and valcol):
        raise SystemExit(f"unexpected CSV columns: {list(rows[0])[:8]}")
    for r in rows:
        try:
            lid = int(r[idcol])
        except (ValueError, TypeError):
            continue
        try:
            v = float(str(r[valcol]).replace(",", ""))
        except ValueError:
            continue
        per_launch.setdefault(lid, {})[r[namecol].strip()] = v
    out = {}
    for lid in sorted(per_launch):
        if lid < len(ARM_BY_LAUNCH):
            out[ARM_BY_LAUNCH[lid]] = per_launch[lid]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="bench/results/ncu_conv1_90.csv")
    args = ap.parse_args()
    path = pathlib.Path(args.csv)
    if not path.exists():
        raise SystemExit(f"{path} not found -- run bench/run_ncu.sh inside the uenv first")
    d = load(path)
    print(f"arms found: {', '.join(d)}\n")
    verdicts = {}

    # ---- Row 1: was the 32-line fan ever real? -----------------------------------------
    print("=" * 78)
    print("ROW 1  was the 32-line fan ever real?   sectors/request, global loads")
    sec = {a: d[a].get(M_SECTORS) for a in d}
    for a, v in sec.items():
        print(f"  {a:>12}: {v if v is None else f'{v:.2f}'} sectors/request")
    b, at = sec.get("baseline"), sec.get("A_transpose")
    if b is None or at is None:
        verdicts["row1"] = "metric unavailable"
    elif b > 16 and at < 8:
        verdicts["row1"] = ("fan REAL (baseline ~%.0f -> A ~%.0f), but not binding: removing it "
                            "bought only 1.228x. D42 upheld as a description of the access "
                            "pattern, refuted as an account of the cost." % (b, at))
    elif b < 8:
        verdicts["row1"] = ("fan NEVER REAL (baseline already %.2f sectors/request). The layout "
                            "reading was wrong too, and the 1.228x came from something other than "
                            "coalescing. D42 fails completely, not partially." % b)
    else:
        verdicts["row1"] = f"neither branch: baseline {b:.2f}, A {at:.2f} -- report as-is, do not choose"
    print(f"  -> {verdicts['row1']}\n")

    # ---- Row 2: why does B lose? THIS ROW DECIDES INPUT-ROW STAGING --------------------
    print("=" * 78)
    print("ROW 2  why does B lose?   *** decides input-row staging ***")
    ob, oB = d.get("baseline", {}).get(M_OCC), d.get("B_smem", {}).get(M_OCC)
    print(f"  achieved occupancy: baseline {ob}%  B_smem {oB}%")
    for k, m in M_STALL.items():
        print(f"  stall {k:>16}: baseline {d.get('baseline',{}).get(m)}  "
              f"B_smem {d.get('B_smem',{}).get(m)}")
    if ob and oB:
        pred = MEASURED_MS["baseline"] * (ob / oB)
        ratio = pred / MEASURED_MS["B_smem"]
        print(f"\n  occupancy-only prediction: {MEASURED_MS['baseline']:.1f} x ({ob:.2f}/{oB:.2f})"
              f" = {pred:.1f} ms   vs measured {MEASURED_MS['B_smem']:.1f} ms   ratio {ratio:.2f}x")
        if 1 / 1.5 <= ratio <= 1.5:
            verdicts["row2"] = ("CAPACITY-DRIVEN OCCUPANCY COLLAPSE -- the occupancy account "
                                f"reproduces B's time to {ratio:.2f}x, inside the pre-registered "
                                "1.5x. **INPUT-ROW STAGING REVIVES.**")
        else:
            verdicts["row2"] = ("BARRIER / DOUBLE-TOUCH -- the occupancy account predicts "
                                f"{pred:.0f} ms against a measured {MEASURED_MS['B_smem']:.0f} ms "
                                f"({ratio:.2f}x, outside the pre-registered 1.5x), so it does not "
                                "get the outcome merely for having the right sign. "
                                "**INPUT-ROW STAGING DIES.**")
    else:
        verdicts["row2"] = "occupancy metric unavailable -- cannot adjudicate; report the blocker"
    print(f"  -> {verdicts['row2']}\n")

    # ---- Row 3: what is the winner waiting on? ----------------------------------------
    print("=" * 78)
    print("ROW 3  what is the winner (A_transpose) waiting on?")
    st = {k: d.get("A_transpose", {}).get(m) for k, m in M_STALL.items()}
    for k, v in sorted(st.items(), key=lambda kv: -(kv[1] or 0)):
        print(f"  {k:>16}: {v}")
    top = max((k for k in st if st[k] is not None), key=lambda k: st[k], default=None)
    verdicts["row3"] = f"dominant stall on the winner: {top} ({st.get(top)})" if top else "unavailable"
    print(f"  -> {verdicts['row3']}\n")

    # ---- Row 4: is the L2-residency claim right? --------------------------------------
    print("=" * 78)
    print("ROW 4  is the L2-residency claim (D50/D53) right?")
    for a in d:
        print(f"  {a:>12}: L2 hit {d[a].get(M_L2_HIT)}%  DRAM {d[a].get(M_DRAM_BW)} B/s  "
              f"L2 {d[a].get(M_L2_BW)} B/s")
    hit = d.get("baseline", {}).get(M_L2_HIT)
    if hit is None:
        verdicts["row4"] = "metric unavailable"
    elif hit > 80:
        verdicts["row4"] = (f"CONFIRMED: L2 hit rate {hit:.1f} %. The weights are cache-resident, "
                            "so charging them DRAM bandwidth was the error D53 named.")
    else:
        verdicts["row4"] = (f"NOT CONFIRMED: L2 hit rate only {hit:.1f} %. The traffic model's "
                            "refutation stands, but D53's stated *mechanism* for it must be "
                            "withdrawn -- being wrong about why it was wrong.")
    print(f"  -> {verdicts['row4']}\n")

    # ---- Row 5: instruction-fetch bound? ----------------------------------------------
    print("=" * 78)
    print("ROW 5  hypothesis #5 -- instruction-fetch bound?")
    ni = {a: d[a].get(M_STALL["no_instruction"]) for a in d}
    print(f"  no_instruction stall: {ni}")
    a_st = {k: v for k, v in st.items() if v is not None}
    if not a_st:
        verdicts["row5"] = "metrics unavailable"
    elif top == "no_instruction":
        verdicts["row5"] = ("CONFIRMED: `no_instruction` is the dominant stall on the winner. "
                            "~10 246 straight-line instructions per thread against the L1 "
                            "instruction cache is the operative bottleneck.")
    else:
        verdicts["row5"] = (f"NOT CONFIRMED: the dominant stall is {top}, not no_instruction. "
                            "The hypothesis explained everything and is not what the hardware "
                            "says; it does not enter the emitter.")
    print(f"  -> {verdicts['row5']}\n")

    print("=" * 78)
    print("SUMMARY")
    for k in sorted(verdicts):
        print(f"  {k}: {verdicts[k]}")
    out = pathlib.Path("bench/results/ncu_adjudication.json")
    out.write_text(json.dumps({"metrics": d, "verdicts": verdicts}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
