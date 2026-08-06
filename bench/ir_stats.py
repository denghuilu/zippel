"""The Gate 1 complexity table: how big do fwd / force / dbwd get, and how many kernels?

Two numbers decide whether Phase 2's hand scheduling is tractable:

  * **distinct contraction signatures** -- the kernel-count proxy. Ops sharing a signature
    differ only in which buffers they point at, so one generated kernel serves all of them.
    This is an upper bound on how many kernels have to be written by hand.
  * **peak live bytes** under a naive topological order, with no rematerialization and no
    fusion. This is the *unscheduled* memory baseline that Phase 2's joint scheduling has to
    beat, so it is deliberately measured without any of the tricks Phase 2 will use.

Neither is a timing. The interpreter is an oracle, not a contender, and no number here
belongs in a performance table.

    python bench/ir_stats.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from blocks.eso2_ir import build_dbwd, build_force, build_forward
from blocks.eso2_ref import ANCHOR_CONFIG_LMAX4, BlockConfig
from fixtures.load import fixture_stats
from zippel.interp import peak_live_bytes
from zippel.simplify import (contraction_signatures, kernel_families, op_counts,
                             signatures, simplify)
from zippel.vjp import assert_closed

GIB = 1024 ** 3
BUILDERS = {"fwd": build_forward, "force": build_force, "dbwd": build_dbwd}


def collect(cfg: BlockConfig, shapes: dict[str, dict]) -> list[dict]:
    rows = []
    for name, build in BUILDERS.items():
        prog, meta = build(cfg)
        assert_closed(prog)
        pre = op_counts(prog)
        simp = simplify(prog, keep=prog.outputs)
        assert_closed(simp)
        post = op_counts(simp)
        row = {
            "program": name,
            "ops_pre_cse": pre["total"],
            "ops_post_cse": post["total"],
            "contractions": post["segmented_contraction"],
            "scalar_maps": post["scalar_map"],
            "paths": post["paths"],
            "contraction_signatures": len(contraction_signatures(simp)),
            "all_signatures": len(signatures(simp)),
            "kernel_families": len(kernel_families(simp)),
            "scalar_fns": sorted({o.fn for o in simp.ops.values() if o.kind == "scalar_map"}),
            "peak_live_gib": {},
        }
        for shape, sizes in shapes.items():
            row["peak_live_gib"][shape] = peak_live_bytes(simp, sizes, itemsize=4) / GIB
        rows.append(row)
    return rows


def markdown(rows: list[dict], shapes: dict[str, dict]) -> str:
    heads = " | ".join(f"peak live GiB ({s})" for s in shapes)
    out = [f"| program | ops pre-CSE | ops post-CSE | paths | distinct contraction signatures | kernel families | {heads} |",
           "|---|---|---|---|---|---|" + "---|" * len(shapes)]
    for r in rows:
        peaks = " | ".join(f"{r['peak_live_gib'][s]:.2f}" for s in shapes)
        out.append(f"| {r['program']} | {r['ops_pre_cse']} | {r['ops_post_cse']} | "
                   f"{r['paths']} | {r['contraction_signatures']} | {r['kernel_families']} | {peaks} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", action="store_true",
                    help="also report the lmax=4 anchor shapes (forward only; DECISIONS.md D12)")
    ap.add_argument("--out", default="bench/results/ir_stats.json")
    args = ap.parse_args()

    shapes = {}
    for fx in ("si_small", "si_medium"):
        st = fixture_stats(fx)
        shapes[fx] = {"node": st["atoms"], "edge": st["edges"], "graph": 1}

    rows = collect(BlockConfig(), shapes)
    print(markdown(rows, shapes))
    print()
    for r in rows:
        print(f"{r['program']:6s} contractions {r['contractions']:5d}  scalar_maps "
              f"{r['scalar_maps']:4d}  fns {r['scalar_fns']}")

    payload = {"config": "K4L2 (eSEN-sm)", "shapes": shapes, "rows": rows}

    if args.anchor:
        # Forward only: the anchor exists for the Phase 2 S1 comparison, never for the
        # verdict table (DECISIONS.md D12).
        prog, _ = build_forward(ANCHOR_CONFIG_LMAX4)
        simp = simplify(prog, keep=prog.outputs)
        payload["anchor_lmax4_fwd"] = {
            "ops_post_cse": op_counts(simp)["total"],
            "contraction_signatures": len(contraction_signatures(simp)),
        }
        print(f"\nlmax=4 anchor (fwd only): {payload['anchor_lmax4_fwd']}")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
