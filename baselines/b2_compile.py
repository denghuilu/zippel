"""B2 -- best achievable `torch.compile` hybrid, plus the fallback inventory.

Two deliverables, and the second matters as much as the first:

  1. the fastest honest torch.compile number for the conservative training step;
  2. a precise record of *what compiles and where the double backward forces eager*.

Variants tried, fastest-wins:

  eager            control, same code path as B1
  compile_default  torch.compile(block) -- graph breaks allowed
  compile_fullgraph torch.compile(block, fullgraph=True) -- fails loudly if it cannot
  compile_reduce_overhead  mode="reduce-overhead" (CUDA graphs) -- expected to be
                   incompatible with create_graph double backward; recorded either way
  compile_backward torch._dynamo.config.compiled_autograd, if available

Note on scope: the block compiled here is our reference block, which uses the *rational*
Wigner construction, so it does not contain fairchem's `Safeatan2` -- whose `backward` is
decorated `@torch.compiler.disable` and is therefore an unconditional graph break for any
fairchem user. That fairchem-specific break is measured separately by
`--fairchem-break-probe` so the inventory reflects what a real fairchem stack hits.

    python baselines/b2_compile.py --fixtures si_medium
"""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import sys
import contextlib

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from baselines.common import PRECISIONS, load_block_and_batch, make_step, precision_context
from bench.harness import Measurement, time_training_step


def _reset_dynamo():
    torch._dynamo.reset()
    torch.cuda.empty_cache()


def explain_graph_breaks(block, batch, jd) -> dict:
    """Record dynamo's own account of what it could and could not capture."""
    from blocks.eso2_ref import BlockConfig  # noqa: F401

    def fwd(pos):
        return block(pos, batch["atomic_numbers"], batch["x_node"], batch["edge_index"],
                     batch["shifts"], batch["cos_gamma_k"], batch["sin_gamma_k"], jd)

    _reset_dynamo()
    try:
        exp = torch._dynamo.explain(fwd)(batch["pos"])
        return {
            "graph_count": getattr(exp, "graph_count", None),
            "graph_break_count": getattr(exp, "graph_break_count", None),
            "op_count": getattr(exp, "op_count", None),
            "break_reasons": [str(getattr(r, "reason", r))[:200]
                              for r in (getattr(exp, "break_reasons", []) or [])][:20],
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {str(exc)[:200]}"}
    finally:
        _reset_dynamo()


def fairchem_compile_probe() -> dict:
    """Does fairchem's own rotation survive torch.compile, and its double backward?

    This is the fallback evidence for a real fairchem stack, independent of our block.
    """
    out: dict = {}
    try:
        import fairchem.core.models.uma.common.rotation as R

        dev = "cuda"
        jd = [j.to(dev).double() for j in
              torch.load("blocks/Jd.pt", weights_only=False)]
        pos = torch.randn(64, 3, device=dev, dtype=torch.float64, requires_grad=True)
        u = torch.randn(9, device=dev, dtype=torch.float64)
        v = torch.randn(9, device=dev, dtype=torch.float64)

        def energy(p):
            torch.manual_seed(0)
            w = R.eulers_to_wigner(R.init_edge_rot_euler_angles(p), 0, 2, jd)
            return torch.einsum("i,eij,j->", u, w, v)

        _reset_dynamo()
        exp = torch._dynamo.explain(energy)(pos)
        out["graph_count"] = getattr(exp, "graph_count", None)
        out["graph_break_count"] = getattr(exp, "graph_break_count", None)
        out["break_reasons"] = [str(getattr(r, "reason", r))[:200]
                                for r in (getattr(exp, "break_reasons", []) or [])][:20]
        _reset_dynamo()

        compiled = torch.compile(energy)
        e = compiled(pos)
        (f,) = torch.autograd.grad(e, pos, create_graph=True)
        try:
            torch.autograd.grad(f.square().sum(), pos)
            out["double_backward_under_compile"] = "ok"
        except Exception as exc:
            out["double_backward_under_compile"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    finally:
        _reset_dynamo()
    return out


VARIANTS = {
    "eager": lambda b: b,
    "compile_default": lambda b: torch.compile(b),
    "compile_fullgraph": lambda b: torch.compile(b, fullgraph=True),
    "compile_reduce_overhead": lambda b: torch.compile(b, mode="reduce-overhead"),
    "compile_max_autotune": lambda b: torch.compile(b, mode="max-autotune-no-cudagraphs"),
    # Backend ladder, to locate *which* layer rejects the double backward rather than
    # just reporting that "torch.compile fails". `backend="eager"` keeps dynamo's graph
    # capture but runs eagerly and never enters AOTAutograd, so if it succeeds while
    # aot_eager/inductor fail, AOTAutograd is precisely the blocker -- and this variant
    # is then the honest "best achievable torch.compile hybrid" for this workload.
    "compile_backend_eager": lambda b: torch.compile(b, backend="eager"),
    "compile_aot_eager": lambda b: torch.compile(b, backend="aot_eager"),
}


def run_variant(fixture: str, precision: str, variant: str) -> Measurement:
    _reset_dynamo()
    with precision_context(precision):
        block, batch, jd, stats = load_block_and_batch(fixture, precision)
        wrapped = VARIANTS[variant](block)
        step, zero, live = make_step(wrapped, batch, jd, precision)
        # keep the liveness check pointed at the real parameter set
        from bench.harness import assert_step_is_live
        live = lambda: assert_step_is_live(block, batch["pos"])  # noqa: E731
        m = time_training_step(
            step, zero, label=f"B2 {variant}", fixture=fixture, precision=precision,
            atoms=stats["atoms"], edges=stats["edges"], liveness_fn=live,
        )
    del block, batch, jd
    _reset_dynamo()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", nargs="*", default=["si_medium"])
    ap.add_argument("--precisions", nargs="*", default=list(PRECISIONS))
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--out", default="bench/results/b2_compile.json")
    args = ap.parse_args()

    inventory: dict = {}
    block, batch, jd, _ = load_block_and_batch(args.fixtures[0], "fp32")
    inventory["ours_rational_block"] = explain_graph_breaks(block, batch, jd)
    del block, batch, jd
    torch.cuda.empty_cache()
    inventory["fairchem_rotation"] = fairchem_compile_probe()
    print("fallback inventory:", json.dumps(inventory, indent=2)[:1500], flush=True)

    results = []
    for fixture in args.fixtures:
        for precision in args.precisions:
            for variant in args.variants:
                try:
                    m = run_variant(fixture, precision, variant)
                except Exception as exc:
                    m = Measurement(f"B2 {variant}", fixture, precision, float("nan"),
                                    float("nan"), float("nan"), float("nan"), float("nan"),
                                    0, 0, 0, error=f"{type(exc).__name__}: {str(exc)[:160]}")
                    _reset_dynamo()
                results.append(m)
                status = m.error or (f"{m.median_ms:8.2f} ms  IQR {m.iqr_ms:6.2f}  "
                                     f"peak {m.peak_mem_gib:6.2f} GiB")
                print(f"{m.label:26s} {fixture:10s} {precision:5s}  {status}", flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"inventory": inventory, "measurements": [vars(m) for m in results]}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
