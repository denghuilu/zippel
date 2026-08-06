"""The single reader for M1 fixtures. `make_fixtures.py` is the single writer.

Deriving `x_node` and `gamma` from the stored seed (rather than storing them) keeps the
.npz files small while guaranteeing every implementation -- reference, SP-IR interpreter,
generated kernels, and every baseline -- sees bit-identical inputs. That identity is one
of the binding anti-gaming rules for the Phase 3 benchmark.

The schema is versioned: a fixture written by a different writer version is rejected loudly
rather than silently misread.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import torch

from fixtures.make_fixtures import SCHEMA_VERSION

FIXTURE_DIR = pathlib.Path(__file__).resolve().parent


def _open(name: str):
    path = FIXTURE_DIR / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing -- run `python fixtures/make_fixtures.py --only {name}`"
        )
    raw = np.load(path, allow_pickle=False)
    version = int(raw["schema_version"])
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{path} is schema v{version}, this loader expects v{SCHEMA_VERSION}. "
            "Regenerate the fixtures (and re-run any table measured on the old ones)."
        )
    return raw, json.loads(str(raw["meta"]))


def fixture_meta(name: str) -> dict:
    _, meta = _open(name)
    return meta


def fixture_stats(name: str) -> dict:
    meta = fixture_meta(name)
    return {"atoms": meta["n_atoms"], "edges": meta["n_edges"],
            "avg_degree": meta["avg_degree"], "cutoff": meta["cutoff"]}


def load_batch(name, device, dtype=torch.float32, cfg=None, requires_grad=True):
    """Returns the batch dict consumed by `blocks.eso2_ref.conservative_training_step`."""
    from blocks.eso2_ref import BlockConfig
    from blocks.wigner import gamma_harmonics

    cfg = cfg or BlockConfig()
    raw, meta = _open(name)

    pos = torch.as_tensor(raw["pos"], device=device, dtype=dtype)
    edge_index = torch.as_tensor(raw["edge_index"], device=device)
    n, e = pos.shape[0], edge_index.shape[1]

    # Generated on CPU from the stored seed so the values do not depend on device, dtype,
    # or RNG-stream ordering elsewhere in the process.
    g = torch.Generator(device="cpu").manual_seed(int(meta["seed"]) + 1)
    x_node = torch.randn(
        n, cfg.num_coeffs, cfg.sphere_channels, generator=g, dtype=torch.float64
    )
    gamma = torch.rand(e, generator=g, dtype=torch.float64) * 2 * np.pi
    cos_g, sin_g = gamma_harmonics(gamma.to(device=device, dtype=dtype), cfg.lmax)

    return {
        "pos": pos.clone().requires_grad_(requires_grad),
        "cell": torch.as_tensor(raw["cell"], device=device, dtype=dtype),
        "atomic_numbers": torch.as_tensor(raw["atomic_numbers"], device=device),
        "x_node": x_node.to(device=device, dtype=dtype),
        "edge_index": edge_index,
        "shifts": torch.as_tensor(raw["shifts"], device=device, dtype=dtype),
        "cos_gamma_k": cos_g,
        "sin_gamma_k": sin_g,
        # Regression targets for the conservative loss. Zeros keep the loss well-defined
        # without inventing physics: M1 measures the cost of the step, not model quality.
        "e_ref": torch.zeros((), device=device, dtype=dtype),
        "f_ref": torch.zeros(n, 3, device=device, dtype=dtype),
        "meta": meta,
    }


def available():
    return sorted(p.stem for p in FIXTURE_DIR.glob("*.npz"))
