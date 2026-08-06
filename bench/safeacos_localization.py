"""Localize fairchem's `Safeacos` second-derivative defect: interior, or clamp-band only?

This matters for how the upstream issue should be framed. `Safeacos.forward` clamps its
input to (-1+EPS, 1-EPS) and saves the *clamped* value; if the defect were confined to the
narrow band where that clamp is actually active (|x| > 1-1e-7), it would be a numerical
edge case. If it is present in the interior too, it is a structural graph bug affecting
every edge in every conservative training step.

Ground truth is analytic:

    d/dx   acos(x) = -1 / sqrt(1 - x^2)
    d2/dx2 acos(x) = -x / (1 - x^2)^(3/2)

    python bench/safeacos_localization.py
"""

from __future__ import annotations

import json
import pathlib

import torch

EPS = 1e-7  # fairchem's clamp epsilon


def analytic_second_derivative(x: torch.Tensor) -> torch.Tensor:
    return -x / (1 - x * x) ** 1.5


def measured_second_derivative(x: torch.Tensor, fn) -> torch.Tensor | None:
    """d2/dx2 of sum(fn(x)) via double autograd. None if autograd refuses (no graph)."""
    x = x.detach().clone().requires_grad_(True)
    try:
        (g,) = torch.autograd.grad(fn(x).sum(), x, create_graph=True)
        # d/dx of the elementwise first derivative -> the diagonal of the Hessian
        (gg,) = torch.autograd.grad(g.sum(), x)
        return gg
    except RuntimeError:
        return None


def main():
    from fairchem.core.models.uma.common.rotation import Safeacos

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float64

    def safe_acos(x):
        return Safeacos.apply(x)

    def true_acos(x):
        return torch.acos(x.clamp(-1 + EPS, 1 - EPS))

    bands = {
        "deep interior |x| <= 0.5": torch.linspace(-0.5, 0.5, 4096, device=dev, dtype=dt),
        "mid |x| in [0.5, 0.9]": torch.linspace(0.5, 0.9, 4096, device=dev, dtype=dt),
        "near-edge |x| in [0.9, 1-1e-3]": torch.linspace(0.9, 1 - 1e-3, 4096, device=dev, dtype=dt),
        "clamp band |x| > 1-1e-7": torch.linspace(1 - EPS, 1 - 1e-12, 4096, device=dev, dtype=dt),
    }

    rows = []
    print(f"{'band':34s} {'Safeacos d2':>14s} {'analytic d2':>14s} {'rel err':>10s}")
    print("-" * 78)
    for name, x in bands.items():
        want = analytic_second_derivative(x)
        got_safe = measured_second_derivative(x, safe_acos)
        got_true = measured_second_derivative(x, true_acos)

        if got_safe is None:
            safe_norm, rel = 0.0, 1.0  # autograd refused: derivative structurally absent
            verdict = "ABSENT (no graph)"
        else:
            safe_norm = got_safe.norm().item()
            rel = ((got_safe - want).abs().max() / want.abs().max()).item()
            verdict = "wrong" if rel > 1e-6 else "ok"
        true_rel = (
            ((got_true - want).abs().max() / want.abs().max()).item()
            if got_true is not None else float("nan")
        )
        print(f"{name:34s} {safe_norm:14.4e} {want.norm().item():14.4e} {rel:10.3e}  {verdict}"
              f"   (torch.acos ref rel {true_rel:.1e})")
        rows.append({"band": name, "safeacos_norm": safe_norm,
                     "analytic_norm": want.norm().item(), "rel_err": rel,
                     "torch_acos_rel_err": true_rel, "verdict": verdict})

    interior = rows[0]
    print()
    if interior["rel_err"] > 1e-6:
        print("LOCALIZATION: the defect is NOT confined to the clamp band -- the second "
              "derivative is wrong in the deep interior too, where the clamp is inactive.")
        print("It is a graph-structure bug (the clamped value is saved from inside forward, "
              "so it carries no grad_fn), not a numerical edge case.")
    else:
        print("LOCALIZATION: the defect appears only in the clamp band.")

    out = pathlib.Path("bench/results/safeacos_localization.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
