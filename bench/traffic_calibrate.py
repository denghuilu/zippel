"""Calibrate the traffic model against measured DRAM traffic (D27).

The work order asks for ncu-measured DRAM traffic. **`ncu` is not installed on this system, and
neither is CUPTI** -- see REPORT.md. The substitute is DCGM's `dram_active` hardware counter
(field 1005), the fraction of cycles the memory interface is transferring, which yields

    bytes  ~=  dram_active  x  peak_bandwidth  x  elapsed

That is coarser than ncu's exact byte counters, so the method is itself calibrated first, against
a workload whose traffic is known by construction. Two stages:

  stage 1  instrument calibration. `torch.Tensor.copy_` of N bytes moves exactly 2N (one read,
           one write). Run it at several sizes and *fit* the effective bandwidth, because the
           raw counter against nameplate peak reads a consistent ~15 % low -- a systematic bias,
           which is what a calibration constant is for. The residual after the fit is how much
           the instrument can be trusted.
  stage 2  model check. Run each emitted kernel back-to-back, derive its DRAM bytes with the
           fitted bandwidth, and compare against `codegen.traffic.estimate`.

Stage 1's residual is reported alongside stage 2's error, because a model error smaller than the
instrument residual is not a measurement of the model.

    python bench/traffic_calibrate.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

#: GH200 HBM3e theoretical peak. `dram_active` is a duty cycle against this, not against the
#: achievable copy rate, so the theoretical figure is the right multiplier.
PEAK_BW_BYTES = 4.0e12


def dcgm_sample(fn, gpu: int = 0, seconds: float = 3.0, interval_ms: int = 100,
                batch: int = 32):
    """Run `fn` back-to-back for `seconds` while sampling dram_active.

    Launches are queued `batch` at a time between synchronisations so the GPU stays busy: a
    launch-bound loop leaves the memory interface idle and the counter would then be measuring
    Python, not the kernel.
    """
    proc = subprocess.Popen(
        ["dcgmi", "dmon", "-e", "1005", "-i", str(gpu), "-d", str(interval_ms)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    try:
        time.sleep(0.6)                      # let the sampler start
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        iters = 0
        while time.perf_counter() - t0 < seconds:
            for _ in range(batch):
                fn()
            iters += batch
            torch.cuda.synchronize()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    finally:
        proc.terminate()
        out, _ = proc.communicate(timeout=10)

    vals = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "GPU":
            try:
                vals.append(float(parts[2]))
            except ValueError:
                pass
    if len(vals) > 4:
        vals = vals[2:-1]                    # drop start-up and tear-down samples
    if not vals:
        return None, iters, elapsed
    return sum(vals) / len(vals), iters, elapsed


def measured_bytes(active: float, elapsed: float, iters: int) -> float:
    return active * PEAK_BW_BYTES * elapsed / max(iters, 1)


def stage1_fit_bandwidth(rows: list, gpu: int) -> float | None:
    """Fit the effective bandwidth that makes known traffic come out right."""
    print("=== stage 1: calibrate the instrument against known traffic ===")
    print(f"{'MiB moved':>10} {'known B/iter':>14} {'raw B/iter':>14} {'raw err':>9} "
          f"{'implied BW TB/s':>16}")
    fits = []
    for mib in (256, 512, 1024):
        n = mib * 1024 * 1024 // 4
        src = torch.randn(n, device="cuda", dtype=torch.float32)
        dst = torch.empty_like(src)
        known = 2 * n * 4                    # one read + one write
        active, iters, elapsed = dcgm_sample(lambda: dst.copy_(src), gpu=gpu)
        if active is None:
            print(f"{mib:>10} -- no dcgm samples --")
            del src, dst
            continue
        raw = measured_bytes(active, elapsed, iters)
        implied = known * iters / (active * elapsed)
        fits.append(implied)
        print(f"{mib:>10} {known:>14,} {raw:>14,.0f} {(raw-known)/known:>8.1%} "
              f"{implied/1e12:>15.3f}")
        rows.append({"stage": "instrument", "case": f"copy_{mib}MiB", "known_bytes": known,
                     "raw_bytes": raw, "implied_bw": implied, "dram_active": active,
                     "iters": iters, "elapsed_s": elapsed})
        del src, dst
        torch.cuda.empty_cache()

    if not fits:
        return None
    fits.sort()
    bw = fits[len(fits) // 2]
    print(f"\nfitted effective bandwidth: {bw/1e12:.3f} TB/s "
          f"(nameplate {PEAK_BW_BYTES/1e12:.1f})")
    print(f"{'case':>16} {'residual after fit':>20}")
    for r in rows:
        if r["stage"] != "instrument":
            continue
        corrected = r["raw_bytes"] * bw / PEAK_BW_BYTES
        r["rel_error"] = (corrected - r["known_bytes"]) / r["known_bytes"]
        print(f"{r['case']:>16} {r['rel_error']:>19.1%}")
    return bw


def _build_forward_env(fixture: str):
    """The emitted kernels, compiled against synthetic inputs of the right shapes.

    Inputs are synthesized from the IR buffer types rather than produced by running the FP64
    interpreter. Two reasons, and the first is the honest one: the full FP64 forward at
    si_medium materialises ~40 GiB of intermediates and was killed by the OOM killer. The
    second is that it is unnecessary -- **DRAM traffic depends on shapes and access patterns,
    not on values**, and none of these kernels branch on data. Correctness is established
    separately, on si_small, against the real interpreter (tests/test_codegen.py).
    """
    from blocks.eso2_ir import build_forward
    from blocks.eso2_ref import BlockConfig
    from codegen.emit import build_kernel, emit_source
    from codegen.emit_tile import emit_tile_source
    from codegen.schedule import analyze_group, build_schedule
    from codegen.tile import build_tile_schedule, channel_axis
    from fixtures.load import fixture_stats
    from zippel.ir import IndexType
    from zippel.simplify import fusion_groups, simplify

    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32
    from cutlass.cute.runtime import from_dlpack

    CFG = BlockConfig()
    torch.manual_seed(0)
    st = fixture_stats(fixture)
    sizes = {"node": st["atoms"], "edge": st["edges"], "graph": 1, "none": 1}
    prog, _ = build_forward(CFG)
    simp = simplify(prog, keep=prog.outputs)
    groups = fusion_groups(simp)
    stream = cutlass.cuda.default_stream()

    def alloc(buf):
        t = simp.type_of(buf)
        n = 1 if t.segment == "none" else sizes[t.segment]
        if isinstance(t, IndexType):
            return torch.zeros(n, dtype=torch.int64, device="cuda")
        return torch.randn(n, *t.sizes, dtype=torch.float64, device="cuda")

    cases = []
    picks = [("wigner_chain", lambda g: any(n.startswith("rot_") for n in g), "T1"),
             ("radial_lin0", lambda g: g == ["rl0_8"], "T2"),
             ("radial_stage2", lambda g: "rs0_16" in g and "rl1_17" in g, "T2")]
    for name, sel, tmpl in picks:
        group = next(g for g in groups if sel(g))
        spec = analyze_group(simp, group, name=name)
        if tmpl == "T1":
            sched = build_schedule(simp, spec)
            src = emit_source(simp, sched, dtype="f64")
        else:
            sched = build_tile_schedule(simp, spec, *channel_axis(simp, spec))
            src = emit_tile_source(simp, sched, dtype="f64")
        Kernel, order = build_kernel(src, f"{name}_traffic_f64")
        tensors = {b: alloc(b) for b in order}
        args = tuple(from_dlpack(tensors[b], assumed_align=16) for b in order) + (
            Int32(sizes["edge"]), stream)
        compiled = cute.compile(Kernel(), *args)
        compiled(*args)
        torch.cuda.synchronize()
        cases.append({"name": name, "template": tmpl, "spec": spec, "sched": sched,
                      "hold": tensors, "launch": (lambda c=compiled, a=args: c(*a))})
    return simp, sizes, cases


def stage2_model_check(rows: list, gpu: int, bw: float, payload: dict):
    """Measure each emitted kernel's DRAM traffic and compare against the model."""
    from codegen.traffic import estimate, read_fraction

    print("\n=== stage 2: does the traffic model predict what the kernels move? ===")
    try:
        simp, sizes, cases = _build_forward_env(payload.get("fixture", "si_medium"))
    except Exception as exc:                      # noqa: BLE001 - reported, not swallowed
        payload["blocked"] = f"stage 2 could not build the kernels: {exc!r}"
        print("BLOCKED:", payload["blocked"])
        return

    print(f"{'kernel':16s} {'tmpl':5s} {'model MiB':>11} {'measured MiB':>13} {'rel err':>9}")
    for case in cases:
        est = estimate(simp, case["spec"], sizes, itemsize=8,
                       reads=read_fraction(simp, case["sched"], 8))  # FP64 kernels
        active, iters, elapsed = dcgm_sample(case["launch"], gpu=gpu)
        if active is None:
            print(f"{case['name']:16s} -- no dcgm samples --")
            continue
        got = active * bw * elapsed / max(iters, 1)
        err = (got - est.total) / est.total if est.total else None
        print(f"{case['name']:16s} {case['template']:5s} {est.total/2**20:>11.2f} "
              f"{got/2**20:>13.2f} {err:>8.1%}")
        rows.append({"stage": "model", "case": case["name"], "template": case["template"],
                     "model_bytes": est.total, "measured_bytes": got, "rel_error": err,
                     "live_in_bytes": est.live_in_bytes, "live_out_bytes": est.live_out_bytes,
                     "avoided_bytes": est.internal_bytes_avoided,
                     "dram_active": active, "iters": iters, "elapsed_s": elapsed})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--fixture", default="si_medium")
    ap.add_argument("--out", default="bench/results/traffic_calibration.json")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if subprocess.run(["which", "dcgmi"], capture_output=True).returncode != 0:
        out.write_text(json.dumps({
            "blocked": "neither ncu, CUPTI, nor dcgmi is available on this system",
            "rows": []}, indent=2))
        print("BLOCKED: no traffic instrument available; recorded in", out)
        return

    rows: list = []
    bw = stage1_fit_bandwidth(rows, args.gpu)

    payload = {"instrument": "dcgm dram_active (field 1005); ncu and CUPTI unavailable",
               "nameplate_bw_bytes": PEAK_BW_BYTES, "fitted_bw_bytes": bw,
               "fixture": args.fixture, "rows": rows}
    if bw is None:
        payload["blocked"] = "dcgmi produced no usable samples"
        out.write_text(json.dumps(payload, indent=2))
        print("BLOCKED:", payload["blocked"])
        return

    resid = [abs(r["rel_error"]) for r in rows if r["stage"] == "instrument"]
    payload["instrument_residual"] = max(resid)
    print(f"\ninstrument residual after fit: {max(resid):.1%}")

    stage2_model_check(rows, args.gpu, bw, payload)

    by_template: dict[str, float] = {}
    for r in rows:
        if r["stage"] == "model" and r.get("rel_error") is not None:
            t = r["template"]
            by_template[t] = max(by_template.get(t, 0.0), abs(r["rel_error"]))
    payload["worst_rel_error_by_template"] = by_template
    if not by_template:
        payload["blocked_instrument"] = "stage 2 produced no comparable rows"

    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")
    print("\n=== D27 gate, per template ===")
    for t, err in sorted(by_template.items()):
        state = "OPEN " if err <= 0.20 else "CLOSED"
        print(f"  {t}: {state}  worst {err:.1%}")
    print("A CLOSED template may not have this model drive its fusion or template decisions.")


if __name__ == "__main__":
    main()
