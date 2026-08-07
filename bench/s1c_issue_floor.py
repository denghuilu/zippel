"""Issue-floor re-check on the factorial's arms. Reads the factorial JSON; runs no kernels.

D42 tested four hypotheses for `conv1_90`'s 714.8 ms and kept one. Two of the refutations were
*conditional on the kernel being memory-bound*, and if an arm removes that bottleneck the
conditions no longer hold. This re-states both floors against every arm's measured time, so the
question after the factorial is answered with arithmetic rather than with the previous verdict:

  **issue floor**   `instr_per_thread x threads / peak thread-instruction rate`. D42 measured
                    18-70x above it, which is why issue rate was refuted as *the* explanation.
                    An arm that removes a 10x memory cost lands 10x closer, and the ratio is the
                    statement of whether the next bottleneck has arrived.

  **traffic floor** the D42 model itself: warp loads x lines touched x line size / HBM bandwidth.
                    The baseline touches 32 lines per warp load of a weight; a transposed or
                    staged operand touches `ceil(32 * itemsize / 128)`. Predicting each arm
                    *separately* is what makes D42 falsifiable rather than merely consistent --
                    it predicted 651 vs 714.8 measured for the baseline with no fitted parameters,
                    and it must now predict the arms with the same none.

**[static analysis]** throughout. The measured times come from the factorial, which is
**[intervention]**; the comparison of the two is what carries the evidential weight.

    python bench/s1c_issue_floor.py
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from bench.s1c_kernel_profile import PEAK_THREAD_INSTR_PER_S, static_census   # noqa: E402
from blocks.eso2_ir import build_forward                                      # noqa: E402
from blocks.eso2_ref import BlockConfig, ESO2RefBlock                         # noqa: E402
from blocks.ir_bind import bind                                               # noqa: E402
from codegen.compose import route                                             # noqa: E402
from codegen.schedule import analyze_group                                    # noqa: E402
from codegen.tile import Ch                                                   # noqa: E402
from fixtures.load import load_batch                                          # noqa: E402
from zippel.simplify import fusion_groups, simplify                           # noqa: E402

#: GH200 HBM3, 96 GB SKU. Vendor peak; the traffic model is quoted against peak and is therefore
#: a floor, not a prediction of achievable time.
HBM_BYTES_PER_S = 4.0e12
#: L2/DRAM sector granularity.
LINE_BYTES = 128
WARP = 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="bench/results/s1c_factorial.json")
    args = ap.parse_args()

    res = json.loads(pathlib.Path(args.results).read_text())
    dtype = res["dtype"]
    itemsize = 4 if dtype == "f32" else 8
    staged = set(res.get("staged", []))
    offenders = set(res["plan"])

    cfg = BlockConfig()
    torch.manual_seed(0)
    jd = [j.to(torch.float64) for j in torch.load("blocks/Jd.pt", weights_only=False)]
    block = ESO2RefBlock(cfg).to("cpu", torch.float64)
    batch = load_batch(res["fixture"], "cpu", torch.float64, cfg, requires_grad=False)
    prog, _ = build_forward(cfg, gauss_coeff=block.gauss_coeff)
    simp = simplify(prog, keep=prog.outputs)
    _inputs, sizes = bind(block, batch, jd, cfg)
    groups = fusion_groups(simp, max_volume=10_000)
    gi = next(i for i, g in enumerate(groups) if "conv1_90" in g)
    spec = analyze_group(simp, groups[gi], name="conv1_90")
    _t, sched, _e = route(simp, spec)
    cen = static_census(simp, sched, "T2", sizes)

    issue_lo = cen["issue_bound_ms_lo"]
    print(f"conv1_90 @ {res['fixture']} {dtype}: {cen['instr_per_thread']:,} instr/thread x "
          f"{cen['threads']:,} threads")
    print(f"issue floor (100 % issue efficiency): {issue_lo:.3f} ms\n")

    # Per-arm traffic: count each factor read of each live-in, and charge lines by access pattern.
    per_buf_reads: dict[str, int] = {}
    for a in sched.assigns:
        for t in a.terms:
            for f in t.factors:
                if f[0] in set(spec.live_in):
                    per_buf_reads[f[0]] = per_buf_reads.get(f[0], 0) + 1

    n_seg = sizes[spec.segment]
    ch = sched.extent

    def traffic_ms(fixed: set[str], staged_here: set[str]) -> float:
        """Bytes / peak bandwidth. `fixed` reads coalesced; `staged_here` is read once per block."""
        total = 0.0
        for buf, reads in per_buf_reads.items():
            t = simp.type_of(buf)
            if buf in staged_here:
                # loaded once per block, contiguously, then served from smem
                elems = 1
                for x in t.sizes:
                    elems *= x
                total += n_seg * elems * itemsize
                continue
            warp_loads = reads * n_seg * ch / WARP
            if buf in offenders and buf not in fixed:
                lines = WARP                      # stride >= line size: one line per lane
            else:
                lines = math.ceil(WARP * itemsize / LINE_BYTES)
            total += warp_loads * lines * LINE_BYTES
        return total / HBM_BYTES_PER_S * 1e3

    arm_cover = {"baseline": (set(), set()),
                 "A_transpose": (offenders, set()),
                 "A_matched": (staged, set()),
                 "B_smem": (set(), staged),
                 "AB_both": (offenders - staged, staged)}

    print(f"{'arm':>12} {'measured':>10} {'traffic':>10} {'meas/traf':>10} "
          f"{'meas/issue':>11} {'bit-eq':>7} {'status':>8}")
    rows = {}
    for name, r in res["arms"].items():
        if r.get("status") in ("not_emitted", "failed") or "ms" not in r:
            print(f"{name:>12} {'--':>10} {'--':>10} {'--':>10} {'--':>11} {'--':>7} "
                  f"{r.get('status', '?'):>8}")
            continue
        fixed, st = arm_cover.get(name, (set(), set()))
        tf = traffic_ms(fixed, st)
        rows[name] = {"measured_ms": r["ms"], "traffic_floor_ms": tf,
                      "issue_floor_ms": issue_lo,
                      "measured_over_traffic": r["ms"] / tf,
                      "measured_over_issue": r["ms"] / issue_lo}
        print(f"{name:>12} {r['ms']:>9.1f}  {tf:>9.1f}  {r['ms']/tf:>9.2f}x "
              f"{r['ms']/issue_lo:>10.1f}x {str(r.get('bitwise_equal_to_baseline')):>7} "
              f"{r['status']:>8}")

    # A floor the measurement beats is not a floor. Check the model against the one arm whose
    # answer is already known from two independent runs before letting it attribute anything.
    if "baseline" in rows and rows["baseline"]["measured_over_traffic"] < 1.0:
        b = rows["baseline"]
        implied = HBM_BYTES_PER_S / b["measured_over_traffic"] / 1e12
        print(f"\n*** TRAFFIC MODEL REFUTED BY ITS OWN BASELINE ***")
        print(f"  predicts {b['traffic_floor_ms']:.1f} ms for a kernel measured at "
              f"{b['measured_ms']:.1f} ms -- {1 / b['measured_over_traffic']:.1f}x over, and a "
              f"floor cannot be beaten.")
        print(f"  It implies an effective {implied:.1f} TB/s against {HBM_BYTES_PER_S/1e12:.1f} "
              f"TB/s of HBM. Two compounding errors, both mine:")
        print(f"    1. the four weights total 1.25 MB and are L2-resident in a 60 MiB L2, so they "
              f"are not HBM traffic at all after the first CTAs;")
        print(f"    2. `reads` counts textual factor occurrences, not issued loads -- NVRTC "
              f"common-subexpression-eliminates within a basic block (emit_tile.py:68-73).")
        print(f"  No per-arm traffic attribution is made. The measured times stand on their own; "
              f"the model does not, and is not used to interpret them.")
        print(f"\n{'arm':>12} {'measured':>10} {'meas/issue':>11}")
        for name, r in rows.items():
            print(f"{name:>12} {r['measured_ms']:>9.1f}  {r['measured_over_issue']:>10.1f}x")
        print(f"\nIssue floor {issue_lo:.1f} ms at 100 % efficiency; the fastest arm is "
              f"{min(r['measured_over_issue'] for r in rows.values()):.0f}x above it, and "
              f"{min(r['measured_over_issue'] for r in rows.values()) / 4:.0f}x above even a "
              f"25 %-efficiency reading. **Neither floor is within reach and one of the two is "
              f"refuted outright.** Next instrument is ncu, not more static analysis.")
        out = pathlib.Path("bench/results/s1c_issue_floor.json")
        out.write_text(json.dumps({"fixture": res["fixture"], "dtype": dtype,
                                   "traffic_model": "REFUTED by baseline; not used",
                                   "instr_per_thread": cen["instr_per_thread"],
                                   "threads": cen["threads"], "arms": rows}, indent=2))
        print(f"\nwrote {out}")
        return

    if rows:
        win = min(rows, key=lambda k: rows[k]["measured_ms"])
        w = rows[win]
        print(f"\nwinner: {win} at {w['measured_ms']:.1f} ms "
              f"({rows['baseline']['measured_ms'] / w['measured_ms']:.2f}x over baseline)"
              if "baseline" in rows else f"\nwinner: {win}")
        print(f"  measured/traffic {w['measured_over_traffic']:.2f}x   "
              f"measured/issue {w['measured_over_issue']:.1f}x")
        # The re-check's actual question, answered by whichever ratio is now near 1.
        if w["measured_over_issue"] < 3:
            print("  -> the kernel has arrived at the issue floor; instruction count is the next "
                  "lever, and D42's refutation of issue-bound no longer applies to this arm.")
        elif w["measured_over_traffic"] < 2:
            print("  -> still traffic-bound, but at the *coalesced* floor: the remaining cost is "
                  "bytes that must move, not wasted lines.")
        else:
            print("  -> neither floor is within reach; a third cost dominates, and static "
                  "analysis has now been wrong once here. Next instrument is ncu, not more of "
                  "this.")

    out = pathlib.Path("bench/results/s1c_issue_floor.json")
    out.write_text(json.dumps({"fixture": res["fixture"], "dtype": dtype,
                               "instr_per_thread": cen["instr_per_thread"],
                               "threads": cen["threads"], "arms": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
