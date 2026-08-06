"""Secondary metric: largest batch of replicated medium cells inside a memory budget.

Units are **GiB** throughout (1 GiB = 2**30 bytes). The work order's "80 GB" against a
95.6 GiB card is ambiguous about base-10 vs base-2; resolved at Gate 0 review as
DECISIONS.md D13:

  primary    largest replication factor whose measured peak allocation is <= 80.0 GiB
  secondary  the same at the full-card budget, 95.6 GiB

The number reported is a *measured* `torch.cuda.max_memory_allocated`, not a capacity
estimate, and it is found by binary search over the replication factor: a config counts as
fitting only if a complete conservative training step runs and its peak stays under budget.
OOM counts as not fitting.

Replication builds a block-diagonal graph: k disjoint copies of the medium cell, node
indices offset per copy. That keeps degree and edge statistics identical to the fixture
while scaling the work, which is what "max batch" is supposed to vary.

    python bench/max_batch.py --fixture si_medium --precisions fp32 bf16
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

GIB = 1024 ** 3
BUDGETS_GIB = {"primary_80GiB": 80.0, "secondary_full_card_95.6GiB": 95.6}


def replicate(batch: dict, k: int) -> dict:
    """k disjoint copies of the graph, node indices offset per copy."""
    if k == 1:
        return batch
    n = batch["pos"].shape[0]
    dev = batch["pos"].device
    offsets = (torch.arange(k, device=dev) * n).repeat_interleave(batch["edge_index"].shape[1])
    out = {
        "pos": batch["pos"].detach().repeat(k, 1).requires_grad_(True),
        "atomic_numbers": batch["atomic_numbers"].repeat(k),
        "x_node": batch["x_node"].repeat(k, 1, 1),
        "edge_index": batch["edge_index"].repeat(1, k) + offsets.unsqueeze(0),
        "shifts": batch["shifts"].repeat(k, 1),
        "cos_gamma_k": batch["cos_gamma_k"].repeat(k, 1),
        "sin_gamma_k": batch["sin_gamma_k"].repeat(k, 1),
        "e_ref": batch["e_ref"],
        "f_ref": batch["f_ref"].repeat(k, 1),
    }
    return out


def peak_gib_for(fixture: str, precision: str, k: int) -> float | None:
    """Run one full training step at replication k; return peak GiB, or None on OOM."""
    from baselines.common import load_block_and_batch, make_step, precision_context

    try:
        with precision_context(precision):
            block, batch, jd, _ = load_block_and_batch(fixture, precision)
            big = replicate(batch, k)
            step, zero, _ = make_step(block, big, jd, precision)
            zero()
            step()  # warm, and proves the step actually completes
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            zero()
            step()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / GIB
    except torch.cuda.OutOfMemoryError:
        peak = None
    finally:
        for name in ("block", "batch", "big", "jd"):
            if name in dir():
                pass
        torch.cuda.empty_cache()
    return peak


def largest_fitting(fixture: str, precision: str, budget_gib: float, k_max: int = 64) -> dict:
    """Binary search the largest k whose measured peak allocation fits the budget."""
    def fits(k: int) -> tuple[bool, float | None]:
        peak = peak_gib_for(fixture, precision, k)
        return (peak is not None and peak <= budget_gib), peak

    ok1, peak1 = fits(1)
    if not ok1:
        return {"k": 0, "peak_gib": peak1, "note": "a single cell already exceeds the budget"}

    # exponential probe up, then bisect
    lo, lo_peak, hi = 1, peak1, 2
    while hi <= k_max:
        ok, _ = fits(hi)
        if not ok:
            break
        lo, hi = hi, hi * 2
    if hi > k_max:
        return {"k": lo, "peak_gib": lo_peak, "note": f"hit the k_max={k_max} cap"}

    while hi - lo > 1:
        mid = (lo + hi) // 2
        ok, peak = fits(mid)
        if ok:
            lo, lo_peak = mid, peak
        else:
            hi = mid
    return {"k": lo, "peak_gib": lo_peak, "note": ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default="si_medium")
    ap.add_argument("--precisions", nargs="*", default=["fp32", "bf16"])
    ap.add_argument("--out", default="bench/results/max_batch.json")
    args = ap.parse_args()

    from fixtures.load import fixture_stats

    stats = fixture_stats(args.fixture)
    results = []
    for precision in args.precisions:
        for label, budget in BUDGETS_GIB.items():
            r = largest_fitting(args.fixture, precision, budget)
            row = {"fixture": args.fixture, "precision": precision, "budget": label,
                   "budget_gib": budget, "atoms": stats["atoms"] * max(r["k"], 1),
                   "edges": stats["edges"] * max(r["k"], 1), **r}
            results.append(row)
            print(f"{args.fixture:10s} {precision:5s} {label:28s} "
                  f"k={r['k']:3d}  peak={r['peak_gib'] if r['peak_gib'] else float('nan'):6.2f} GiB"
                  f"  {r['note']}", flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
