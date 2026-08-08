"""Compulsory vs measured DRAM bytes for the top-3 T2 kernels. Static, and deliberately generous.

ncu says `conv1_90` runs at **78 % of HBM peak** and that its 1.228x speedup is *exactly* its 17 %
byte reduction (1.205 vs 1.228, 1.9 % apart, D59). The kernel is bandwidth-bound, so from here
**every intervention passes or fails one test: does it cut DRAM bytes?** This table says how much
room that test has.

**Compulsory** is the traffic a *correct implementation with an infinite cache* could not avoid:
each distinct byte of each operand read once, each output byte written once. Weights are
`none`-segment and shared by every edge, so they count **once for the whole launch**, not once per
CTA — that is the whole point, and it is what makes the ratio meaningful rather than tautological.

The arithmetic is deliberately **safe** — it over-counts compulsory traffic wherever it is unsure,
so the reported ratio is a **lower bound on the waste**:

  * every live-in is charged in full even if the kernel reads only part of it;
  * outputs are charged as a full write, with no write-allocate read;
  * no credit is taken for values that never leave registers.

Measured DRAM bytes come from ncu for `conv1_90` (`dram__bytes.sum.per_second` x kernel time).
For the other two kernels ncu has not been run, so their measured column is **derived under a
stated assumption** — that they sustain the same 3.13 TB/s the profiled sibling does — and is
labelled as such rather than presented as a measurement. They are the same template, the same
block shape and the same access structure, which makes the assumption reasonable and still an
assumption.

    python bench/compulsory_bytes.py
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
from codegen.compose import route                                           # noqa: E402
from codegen.schedule import analyze_group                                  # noqa: E402
from fixtures.load import load_batch                                        # noqa: E402
from zippel.simplify import fusion_groups, simplify                         # noqa: E402

#: **[measurement]** ncu, si_medium fp32, D59.
MEASURED_DRAM_TBPS = {"conv1_90": 3.13}
#: **[assumption]** the unprofiled siblings sustain the profiled one's throughput.
ASSUMED_DRAM_TBPS = 3.13
#: **[measurement]** per-kernel times, si_medium fp32 (REPORT 8.5g).
KERNEL_MS = {"conv1_90": 714.819, "conv2_95": 344.7, "conv1_m0_86": 190.5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="si_medium")
    ap.add_argument("--dtype", default="f32")
    args = ap.parse_args()
    itemsize = 4 if args.dtype == "f32" else 8

    cfg = BlockConfig()
    torch.manual_seed(0)
    jd = [j.to(torch.float64) for j in torch.load("blocks/Jd.pt", weights_only=False)]
    block = ESO2RefBlock(cfg).to("cpu", torch.float64)
    batch = load_batch(args.fixture, "cpu", torch.float64, cfg, requires_grad=False)
    prog, _ = build_forward(cfg, gauss_coeff=block.gauss_coeff)
    simp = simplify(prog, keep=prog.outputs)
    _inp, sizes = bind(block, batch, jd, cfg)
    groups = fusion_groups(simp, max_volume=10_000)

    def buf_bytes(b: str) -> tuple[int, str]:
        t = simp.type_of(b)
        elems = 1
        for x in getattr(t, "sizes", ()) or ():
            elems *= x
        seg = getattr(t, "segment", "none")
        n = 1 if seg == "none" else sizes[seg]
        return n * elems * itemsize, seg

    rows = []
    print(f"{args.fixture} {args.dtype}, {sizes['edge']:,} edges\n")
    for name in KERNEL_MS:
        gi = next((i for i, g in enumerate(groups) if name in g), None)
        if gi is None:
            continue
        spec = analyze_group(simp, groups[gi], name=name)
        _t, sched, _e = route(simp, spec)
        comp, detail = 0, []
        for b in sorted(set(spec.live_in) | set(spec.live_out)):
            nb, seg = buf_bytes(b)
            comp += nb
            detail.append((b, seg, nb))
        ms = KERNEL_MS[name]
        tbps = MEASURED_DRAM_TBPS.get(name, ASSUMED_DRAM_TBPS)
        measured = tbps * 1e12 * ms / 1e3
        kind = "[measured]" if name in MEASURED_DRAM_TBPS else "[assumed 3.13 TB/s]"
        rows.append({"kernel": name, "ms": ms, "compulsory_bytes": comp,
                     "measured_dram_bytes": measured, "ratio": measured / comp,
                     "provenance": kind,
                     "operands": [{"buf": b, "segment": s, "bytes": n} for b, s, n in detail]})
        print(f"=== {name}  ({ms:.1f} ms, {sched.n_terms:,} terms) ===")
        for b, seg, nb in sorted(detail, key=lambda x: -x[2]):
            print(f"    {b:<18} {seg:>5}  {nb/2**20:12,.2f} MiB")
        print(f"    {'COMPULSORY':<18} {'':>5}  {comp/2**20:12,.2f} MiB")
        print(f"    {'MEASURED DRAM':<18} {'':>5}  {measured/2**30:12,.2f} GiB   {kind}")
        print(f"    {'RATIO':<18} {'':>5}  {measured/comp:12,.0f}x\n")

    tot_c = sum(r["compulsory_bytes"] for r in rows)
    tot_m = sum(r["measured_dram_bytes"] for r in rows)
    print("=" * 72)
    print(f"top-3 kernels: compulsory {tot_c/2**20:,.1f} MiB   vs   DRAM {tot_m/2**30:,.2f} GiB"
          f"   =  {tot_m/tot_c:,.0f}x")
    print("\nThe ratio is a LOWER bound on the waste: every rule above rounds in favour of the")
    print("compulsory column. It is the size of the prize for any byte-cutting intervention, and")
    print("it is why 'bandwidth-bound at 78 % of peak' is not the same as 'nothing left to win'.")

    out = pathlib.Path("bench/results/compulsory_bytes.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"fixture": args.fixture, "dtype": args.dtype,
                               "edges": sizes["edge"], "kernels": rows,
                               "total_compulsory": tot_c, "total_measured": tot_m,
                               "total_ratio": tot_m / tot_c}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
