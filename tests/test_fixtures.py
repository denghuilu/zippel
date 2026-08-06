"""Fixture schema, manifest integrity, and write->load round-trip.

The fixtures define the measured unit's inputs for every implementation, so a silent
change to them would invalidate every number in REPORT.md. These tests make that loud.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from fixtures import make_fixtures as mk
from fixtures.load import fixture_meta, fixture_stats, load_batch
from tests.conftest import requires_cuda

NAMES = [s[0] for s in mk.SPECS]


@pytest.fixture(scope="module")
def manifest():
    assert mk.MANIFEST.exists(), "fixtures/manifest.json missing -- run make_fixtures.py"
    return json.loads(mk.MANIFEST.read_text())


def test_manifest_covers_every_fixture(manifest):
    assert manifest["schema_version"] == mk.SCHEMA_VERSION
    assert manifest["cutoff"] == mk.CUTOFF
    assert sorted(e["name"] for e in manifest["fixtures"]) == sorted(NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_content_hash_matches_manifest(name, manifest):
    """The .npz on disk is the one the manifest pins.

    `sha256_content` hashes the array bytes in fixed key order, so it is reproducible on
    any machine and with either neighbour-list backend -- unlike the file hash, which
    np.savez_compressed perturbs via zip timestamps.
    """
    entry = next(e for e in manifest["fixtures"] if e["name"] == name)
    with np.load(mk.HERE / f"{name}.npz", allow_pickle=False) as z:
        got = mk.content_hash({k: z[k] for k in mk.CONTENT_KEYS})
    assert got == entry["sha256_content"], f"{name} content differs from manifest"


@pytest.mark.parametrize("name", NAMES)
def test_schema_version_is_stored_and_current(name):
    with np.load(mk.HERE / f"{name}.npz", allow_pickle=False) as z:
        assert int(z["schema_version"]) == mk.SCHEMA_VERSION
    assert fixture_meta(name)["schema_version"] == mk.SCHEMA_VERSION


def test_loader_rejects_a_wrong_schema_version(tmp_path, monkeypatch):
    """A stale fixture must fail loudly, not be silently misread."""
    name = "si_small"
    with np.load(mk.HERE / f"{name}.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in mk.CONTENT_KEYS}
        meta = json.loads(str(z["meta"]))
    bogus = tmp_path / f"{name}.npz"
    np.savez_compressed(bogus, schema_version=np.array(mk.SCHEMA_VERSION + 1),
                        meta=json.dumps(meta), **arrays)
    monkeypatch.setattr("fixtures.load.FIXTURE_DIR", tmp_path)
    with pytest.raises(ValueError, match="schema v"):
        fixture_meta(name)


@pytest.mark.parametrize("name", NAMES)
def test_edges_are_canonically_sorted(name):
    """Canonical ordering is what makes the content hash backend-independent."""
    with np.load(mk.HERE / f"{name}.npz", allow_pickle=False) as z:
        ei, shifts = z["edge_index"], z["shifts"]
    resorted, _ = mk.canonicalise(ei, shifts)
    assert np.array_equal(ei, resorted), f"{name} edges are not in canonical order"


@pytest.mark.parametrize("name", ["si_small", "cu_small"])
def test_write_load_round_trip(name, tmp_path, monkeypatch):
    """Rebuild from the recorded spec and confirm the loader sees identical content."""
    spec = next(s for s in mk.SPECS if s[0] == name)
    meta = fixture_meta(name)
    monkeypatch.setattr(mk, "HERE", tmp_path)
    entry = mk.make(*spec, seed=meta["seed"])

    assert entry["sha256_content"] == meta["sha256_content"], (
        f"{name} is not reproducible: rebuilding from the same spec gave a different "
        "content hash"
    )
    assert (entry["n_atoms"], entry["n_edges"]) == (meta["n_atoms"], meta["n_edges"])


@requires_cuda
@pytest.mark.parametrize("name", ["si_small", "cu_small"])
def test_loaded_batch_is_self_consistent(name, device):
    """Every field the measured unit consumes is present, finite, and correctly shaped."""
    from blocks.eso2_ref import BlockConfig

    cfg = BlockConfig()
    b = load_batch(name, device, torch.float64, cfg)
    stats = fixture_stats(name)
    n, e = stats["atoms"], stats["edges"]

    assert b["pos"].shape == (n, 3) and b["pos"].requires_grad
    assert b["edge_index"].shape == (2, e)
    assert b["shifts"].shape == (e, 3)
    assert b["x_node"].shape == (n, cfg.num_coeffs, cfg.sphere_channels)
    assert b["cos_gamma_k"].shape == (e, cfg.lmax + 1)
    assert b["atomic_numbers"].shape == (n,)
    for k, v in b.items():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            assert torch.isfinite(v).all(), f"{k} has non-finite entries"

    # No zero-length edges: the edge frame is undefined there.
    vec = b["pos"][b["edge_index"][1]] - b["pos"][b["edge_index"][0]] + b["shifts"]
    d = vec.norm(dim=-1)
    assert d.min() > 1e-6
    assert d.max() <= stats["cutoff"] + 1e-9
    # cos^2 + sin^2 == 1 for the roll harmonics
    assert torch.allclose(b["cos_gamma_k"] ** 2 + b["sin_gamma_k"] ** 2,
                          torch.ones_like(b["cos_gamma_k"]), atol=1e-12)


@requires_cuda
def test_loader_is_deterministic_across_calls(device):
    """Two loads of the same fixture must be bit-identical -- the anti-gaming rule that
    every implementation sees the same inputs depends on it."""
    a = load_batch("si_small", device, torch.float64)
    b = load_batch("si_small", device, torch.float64)
    for key in ("pos", "x_node", "shifts", "cos_gamma_k", "sin_gamma_k"):
        assert torch.equal(a[key], b[key]), f"{key} differs between loads"
