"""S3 scale preflight: can the emitter produce dbwd-sized kernels at all?

Structure only -- source is emitted and parsed, nothing is compiled or validated. The question
is narrow: the 48-term chunking added at S1b exists because CPython's AST recursion limit is hit
while the DSL parses a generated module, and dbwd groups are an order of magnitude larger than
the forward's. If the largest dbwd group cannot even be parsed, S3 needs a different emission
strategy and it is much cheaper to learn that now than at S3.

    python bench/dbwd_scale_preflight.py
"""

from __future__ import annotations

import ast
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from blocks.eso2_ir import build_dbwd                                        # noqa: E402
from blocks.eso2_ref import BlockConfig                                      # noqa: E402
from codegen.emit import emit_source                                        # noqa: E402
from codegen.emit_tile import emit_tile_source                              # noqa: E402
from codegen.schedule import analyze_group, build_schedule, index_maps_used  # noqa: E402
from codegen.tile import build_tile_schedule, channel_axis                   # noqa: E402
from zippel.simplify import fusion_groups, simplify                          # noqa: E402


def main():
    prog, _ = build_dbwd(BlockConfig())
    simp = simplify(prog, keep=prog.outputs)
    groups = fusion_groups(simp)
    print(f"dbwd: {len(simp.ops)} ops -> {len(groups)} acyclic groups", flush=True)

    cands = []
    for gi, g in enumerate(groups):
        spec = analyze_group(simp, g, name=f"g{gi}")
        if index_maps_used(simp, spec):
            continue
        sched = build_schedule(simp, spec)
        if sched.peak_live_values() <= 168:
            cands.append((sched.n_terms, gi, spec, "T1", sched))
            continue
        axis = channel_axis(simp, spec)
        if axis is None:
            continue
        try:
            tile = build_tile_schedule(simp, spec, *axis)
            cands.append((tile.n_terms, gi, spec, "T2", tile))
        except Exception as exc:                                   # noqa: BLE001
            print(f"  g{gi}: schedule failed {type(exc).__name__}", flush=True)
    cands.sort(reverse=True)

    print(f"\n{'#':>4} {'tmpl':>5} {'ops':>4} {'terms':>9} {'lines':>8} {'MiB':>7} "
          f"{'emit s':>7} {'parse s':>8}  status", flush=True)
    rows, worst = [], 0
    for terms, gi, spec, tmpl, sched in cands[:5]:
        try:
            t0 = time.time()
            src = (emit_source(simp, sched, dtype="f32") if tmpl == "T1"
                   else emit_tile_source(simp, sched, dtype="f32"))
            t1 = time.time()
            ast.parse(src)
            t2 = time.time()
            worst = max(worst, terms)
            print(f"{gi:>4} {tmpl:>5} {len(spec.ops):>4} {terms:>9} {len(src.splitlines()):>8} "
                  f"{len(src)/2**20:>7.2f} {t1-t0:>7.2f} {t2-t1:>8.2f}  parsed OK", flush=True)
            rows.append({"group": gi, "template": tmpl, "ops": len(spec.ops), "terms": terms,
                         "lines": len(src.splitlines()), "bytes": len(src),
                         "emit_s": t1 - t0, "parse_s": t2 - t1, "ok": True})
        except Exception as exc:                                   # noqa: BLE001
            print(f"{gi:>4} {tmpl:>5} {len(spec.ops):>4} {terms:>9}  "
                  f"{type(exc).__name__}: {str(exc)[:60]}", flush=True)
            rows.append({"group": gi, "template": tmpl, "terms": terms, "ok": False,
                         "error": f"{type(exc).__name__}: {exc}"})

    print(f"\nmax terms emitted and parsed: {worst}", flush=True)
    out = pathlib.Path("bench/results/dbwd_scale_preflight.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"max_terms_parsed": worst, "rows": rows}, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
