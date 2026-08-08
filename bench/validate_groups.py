"""Emit, compile and validate every fusion group of a program. Records the cost ledger.

Replaces the throwaway per-template scripts used through S1a/S1b/S1c. Those lived in a scratch
directory that a session teardown cleared, taking a 40-minute run with them; this is in the repo,
writes its results under `bench/results/`, and is reproducible.

Each group is routed by the selection rule (docs/templates.md 2), emitted, compiled, launched,
and checked against the FP64 interpreter using **the bound its own emitter shipped** (D25). Every
phase is timed into `codegen.costs`, so the compile-time column exists as a by-product rather
than as a separate exercise.

Groups run in increasing size order. That is not cosmetic: one forward group is 23 040 terms --
4.5x the next largest -- and putting it first means a single outlier blocks every result behind
it. `--max-terms` skips such a group and says so, rather than silently timing out.

    python bench/validate_groups.py --program fwd --fixture si_small
    python bench/validate_groups.py --program fwd --max-terms 10000   # skip the outlier
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from blocks.eso2_ir import build_dbwd, build_force, build_forward          # noqa: E402
from blocks.eso2_ref import BlockConfig, ESO2RefBlock                     # noqa: E402
from blocks.ir_bind import bind                                           # noqa: E402
from codegen import costs                                                 # noqa: E402
from codegen.bounds import ordering_bound                                 # noqa: E402
from codegen.emit import build_kernel, emit_source                        # noqa: E402
from codegen.emit_reduce import emit_reduce_source, gather_maps, scatter_map  # noqa: E402
from codegen.emit_tile import emit_tile_source                            # noqa: E402
from codegen.schedule import (analyze_group, build_schedule,              # noqa: E402
                              index_maps_used)
from codegen.tile import build_tile_schedule, channel_axis                # noqa: E402
from fixtures.load import load_batch                                      # noqa: E402
from zippel.interp import run                                             # noqa: E402
from zippel.simplify import fusion_groups, simplify                       # noqa: E402

BUILDERS = {"fwd": build_forward, "force": build_force, "dbwd": build_dbwd}
DT = torch.float64


def route(prog, spec):
    """Apply the selection rule. Returns `(template, schedule, emit_fn)`."""
    if index_maps_used(prog, spec):
        sched = build_schedule(prog, spec)
        return "T3", sched, emit_reduce_source
    sched = build_schedule(prog, spec)
    if sched.peak_live_values() <= 168:
        return "T1", sched, emit_source
    axis = channel_axis(prog, spec)
    if axis is None:
        # rank-0 or extent-1 output channel axis is the signature of a reduction
        return "T3", sched, emit_reduce_source
    return "T2", build_tile_schedule(prog, spec, *axis), emit_tile_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--program", default="fwd", choices=sorted(BUILDERS))
    ap.add_argument("--fixture", default="si_small")
    ap.add_argument("--max-terms", type=int, default=0,
                    help="skip groups larger than this (0 = no limit)")
    ap.add_argument("--min-terms", type=int, default=0,
                    help="skip groups smaller than this; lets an outlier be run on its own")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32
    from cutlass.cute.runtime import from_dlpack

    cfg = BlockConfig()
    torch.manual_seed(0)
    jd = [j.to(DT) for j in torch.load("blocks/Jd.pt", weights_only=False)]
    block = ESO2RefBlock(cfg).to("cpu", DT)
    batch = load_batch(args.fixture, "cpu", DT, cfg)
    prog, _ = BUILDERS[args.program](cfg, gauss_coeff=block.gauss_coeff) \
        if args.program == "fwd" else BUILDERS[args.program](cfg)
    simp = simplify(prog, keep=prog.outputs)
    inp, sizes = bind(block, batch, jd, cfg)
    env = run(simp, inp, sizes)
    groups = fusion_groups(simp)
    stream = cutlass.cuda.default_stream()
    print(f"{args.program} @ {args.fixture}: {len(simp.ops)} ops, {len(groups)} groups, "
          f"{sizes}", flush=True)

    # size first, so one 23k-term outlier cannot block every result behind it
    plan = []
    for gi, g in enumerate(groups):
        spec = analyze_group(simp, g, name=f"g{gi}")
        with costs.phase(spec.name, "schedule"):
            template, sched, emit = route(simp, spec)
        plan.append((sched.n_terms, gi, spec, template, sched, emit))
    plan.sort()

    print(f"\n{'#':>4} {'tmpl':>5} {'terms':>8} {'emit s':>7} {'compile s':>10} "
          f"{'bound':>10} {'measured':>10}  verdict", flush=True)
    rows, ok, bad, skipped = [], 0, 0, 0
    for n_terms, gi, spec, template, sched, emit in plan:
        if args.min_terms and n_terms < args.min_terms:
            skipped += 1
            continue
        if args.max_terms and n_terms > args.max_terms:
            skipped += 1
            print(f"{gi:>4} {template:>5} {n_terms:>8,} "
                  f"{'--':>7} {'--':>10} {'--':>10} {'--':>10}  SKIPPED (over --max-terms)",
                  flush=True)
            rows.append({"group": gi, "template": template, "terms": n_terms,
                         "status": "skipped"})
            continue
        try:
            with costs.phase(spec.name, "emit", template=template, terms=n_terms):
                source = emit(simp, sched, dtype="f64")
            name = f"{args.program}_g{gi}_f64"
            with costs.phase(spec.name, "guard"):
                Kernel, order = build_kernel(source, name, sched=sched)
            module = sys.modules[f"zippel_generated.{name}"]
            n_seg = sizes[getattr(module, "DRIVING_SEGMENT", module.SEGMENT)]

            outs = {b: torch.zeros_like(env[b]) for b in spec.live_out}
            # The layout requirement (D54) is now T2's default, so an operand may need permuting
            # before launch. Read the permutation back from the module the kernel was built from,
            # never recompute it (D52). `env` itself is left alone: it is the reference the result
            # is checked against, and several groups share it.
            _tr = getattr(module, "TRANSPOSE", {})

            def _prep(b, _tr=_tr):
                v = env[b]
                if b in _tr:
                    v = v.permute((0,) + tuple(k + 1 for k in _tr[b]))
                return v.contiguous()

            tensors = {b: (outs[b] if b in outs else _prep(b)) for b in order}
            call = tuple(from_dlpack(tensors[b], assumed_align=16) for b in order) + (
                Int32(n_seg), stream)
            t0 = time.perf_counter()
            compiled = cute.compile(Kernel(), *call)
            torch.cuda.synchronize()
            costs.record(spec.name, compile_s=time.perf_counter() - t0)
            compiled(*call)
            torch.cuda.synchronize()

            with costs.phase(spec.name, "guard"):
                scatter = scatter_map(simp, spec)
                bound = ordering_bound(sched, env, gathers=gather_maps(simp, spec),
                                       scatter=scatter)
                err = max(float((outs[b] - env[b]).abs().max()) for b in spec.live_out)
            good = err <= bound
            ok += good
            bad += not good
            c = costs.costs()[spec.name]
            print(f"{gi:>4} {template:>5} {n_terms:>8,} {c.get('emit_s', 0):>7.2f} "
                  f"{c.get('compile_s', 0):>10.2f} {bound:>10.2e} {err:>10.2e}  "
                  f"{'ok' if good else 'EXCEEDS BOUND'}", flush=True)
            rows.append({"group": gi, "template": template, "terms": n_terms,
                         "bound": bound, "measured": err,
                         "status": "ok" if good else "exceeds", **c})
        except Exception as exc:                                    # noqa: BLE001
            bad += 1
            print(f"{gi:>4} {template:>5} {n_terms:>8,} {'--':>7} {'--':>10} {'--':>10} "
                  f"{'--':>10}  {type(exc).__name__}: {str(exc)[:60]}", flush=True)
            rows.append({"group": gi, "template": template, "terms": n_terms,
                         "status": "error", "error": f"{type(exc).__name__}: {exc}"})

    print(f"\n{ok} correct, {bad} failed, {skipped} skipped of {len(plan)} groups", flush=True)
    summary = costs.summary()
    print(f"cost ledger: " + "  ".join(f"{k}={v:.1f}s" for k, v in summary["totals_s"].items())
          + f"   guard={summary['guard_fraction']:.1%} of build", flush=True)

    out = pathlib.Path(args.out or
                       f"bench/results/validate_{args.program}_{args.fixture}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"program": args.program, "fixture": args.fixture,
                               "sizes": sizes, "rows": rows, "costs": summary}, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
