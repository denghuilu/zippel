"""Canonical writer for the M1 benchmark fixtures: perturbed bulk Si (diamond), Cu (fcc).

Three sizes each, per the work order: small ~200 atoms, medium ~5k (the primary fixture),
large ~50k. Ragged neighbour lists at the eSEN K4L2 cutoff of 6.0 A, with PBC.

This module is the **only** writer. `fixtures/load.py` is the **only** reader. The on-disk
schema is versioned (`schema_version`) and pinned by `fixtures/manifest.json`.

Only geometry and connectivity are stored. The node irrep features `x_node` and the Wigner
roll angles `gamma` are regenerated deterministically from the stored seed at load time --
storing x_node for the large fixture would be ~430 MB in FP64 for no benefit.

The published eSEN `max_neighbors=30` cap is deliberately NOT applied: it is a data-loader
detail whose tie-break among equidistant neighbours is nondeterministic, and it is not part
of the block's math (DECISIONS.md D6). Average degree is recorded instead.

**Determinism.** Edges are sorted into a canonical order (src, dst, then shift), so the
content hash does not depend on whether vesin or ASE produced the list. Two hashes are
recorded per fixture:
  * `sha256_content` -- over the array bytes in fixed key order. Reproducible on any
    machine with any neighbour-list backend; this is what the round-trip test checks.
  * `sha256_file`    -- over the .npz bytes as committed. NOT reproducible across runs,
    because np.savez_compressed embeds zip timestamps; recorded for artifact integrity
    only.

    python fixtures/make_fixtures.py            # all six
    python fixtures/make_fixtures.py --only si_medium
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import time

import numpy as np

SCHEMA_VERSION = 1
CUTOFF = 6.0  # A, from the K4L2 / eSEN-sm config
RATTLE = 0.05  # A, gaussian displacement std -- perturbed but still physical

# (name, element, structure, lattice constant A, supercell repeats, atoms per cubic cell)
SPECS = [
    ("si_small", "Si", "diamond", 5.431, 3, 8),
    ("si_medium", "Si", "diamond", 5.431, 9, 8),
    ("si_large", "Si", "diamond", 5.431, 18, 8),
    ("cu_small", "Cu", "fcc", 3.615, 4, 4),
    ("cu_medium", "Cu", "fcc", 3.615, 11, 4),
    ("cu_large", "Cu", "fcc", 3.615, 23, 4),
]

HERE = pathlib.Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"

# Array keys, in the fixed order used to compute sha256_content.
CONTENT_KEYS = ("pos", "cell", "atomic_numbers", "edge_index", "shifts")


def build_atoms(element: str, structure: str, a: float, reps: int, seed: int):
    from ase.build import bulk

    atoms = bulk(element, structure, a=a, cubic=True) * (reps, reps, reps)
    rng = np.random.default_rng(seed)
    atoms.positions += rng.normal(0.0, RATTLE, atoms.positions.shape)
    return atoms


def neighbor_list(atoms, cutoff: float):
    """Ragged neighbour list -> (edge_index [2,E], shifts [E,3] cartesian, backend).

    Prefers vesin (much faster at 50k atoms); falls back to ASE, which the work order also
    permits. The two disagree on edge *order*, so the caller canonicalises.
    """
    pos = np.ascontiguousarray(atoms.positions, dtype=np.float64)
    cell = np.ascontiguousarray(atoms.cell.array, dtype=np.float64)
    try:
        from vesin import NeighborList

        nl = NeighborList(cutoff=cutoff, full_list=True)
        src, dst, offsets = nl.compute(points=pos, box=cell, periodic=True, quantities="ijS")
        backend = "vesin"
    except ImportError:
        from ase.neighborlist import neighbor_list as ase_nl

        src, dst, offsets = ase_nl("ijS", atoms, cutoff)
        backend = "ase"

    offsets = np.asarray(offsets, dtype=np.float64)
    edge_index = np.stack([np.asarray(src), np.asarray(dst)]).astype(np.int64)
    return edge_index, offsets, backend


def canonicalise(edge_index: np.ndarray, offsets: np.ndarray):
    """Sort edges by (src, dst, shift) so the fixture is backend-independent."""
    order = np.lexsort(
        (offsets[:, 2], offsets[:, 1], offsets[:, 0], edge_index[1], edge_index[0])
    )
    return edge_index[:, order], offsets[order]


def content_hash(arrays: dict) -> str:
    """SHA256 over array bytes in fixed key order -- independent of zip metadata."""
    h = hashlib.sha256()
    for key in CONTENT_KEYS:
        a = np.ascontiguousarray(arrays[key])
        h.update(key.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def file_hash(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make(name, element, structure, a, reps, per_cell, seed=0):
    t0 = time.time()
    atoms = build_atoms(element, structure, a, reps, seed)
    edge_index, offsets, backend = neighbor_list(atoms, CUTOFF)
    edge_index, offsets = canonicalise(edge_index, offsets)

    cell = np.ascontiguousarray(atoms.cell.array, dtype=np.float64)
    shifts = offsets @ cell
    n, e = len(atoms), edge_index.shape[1]
    assert n == per_cell * reps**3, (n, per_cell, reps)

    # A neighbour list must never contain a self-loop with a zero-length vector: the edge
    # frame is undefined there (both 1/|v| and sin(beta) collapse), and it silently breaks
    # equivariance and the finite-difference checks.
    vec = atoms.positions[edge_index[1]] - atoms.positions[edge_index[0]] + shifts
    dist = np.linalg.norm(vec, axis=1)
    assert dist.min() > 1e-6, "zero-length edge in neighbour list"
    assert dist.max() <= CUTOFF + 1e-9, "edge beyond cutoff"

    arrays = {
        "pos": np.ascontiguousarray(atoms.positions, dtype=np.float64),
        "cell": cell,
        "atomic_numbers": np.ascontiguousarray(atoms.numbers, dtype=np.int64),
        "edge_index": np.ascontiguousarray(edge_index, dtype=np.int64),
        "shifts": np.ascontiguousarray(shifts, dtype=np.float64),
    }
    meta = {
        "schema_version": SCHEMA_VERSION,
        "name": name, "element": element, "structure": structure,
        "n_atoms": int(n), "n_edges": int(e), "avg_degree": round(e / n, 4),
        "cutoff": CUTOFF, "rattle_std": RATTLE, "reps": reps, "lattice_a": a,
        "seed": seed, "backend": backend,
        "min_edge_len": float(dist.min()), "max_edge_len": float(dist.max()),
    }
    sha_content = content_hash(arrays)
    meta["sha256_content"] = sha_content

    out = HERE / f"{name}.npz"
    np.savez_compressed(
        out, schema_version=np.array(SCHEMA_VERSION), meta=json.dumps(meta), **arrays
    )
    entry = {**meta, "sha256_file": file_hash(out), "bytes": out.stat().st_size}
    print(f"  {name:11s} N={n:6d} E={e:9d} deg={e/n:6.2f} [{backend}] "
          f"{time.time() - t0:5.1f}s  content={sha_content[:16]}")
    return entry


def verify(names=None) -> bool:
    """Re-hash the .npz on disk against manifest.json. Returns True if all match."""
    if not MANIFEST.exists():
        print("no manifest.json -- run make_fixtures.py first")
        return False
    entries = {e["name"]: e for e in json.loads(MANIFEST.read_text())["fixtures"]}
    ok = True
    for name, entry in entries.items():
        if names and name not in names:
            continue
        path = HERE / f"{name}.npz"
        if not path.exists():
            print(f"  {name:11s} MISSING")
            ok = False
            continue
        with np.load(path, allow_pickle=False) as z:
            got = content_hash({k: z[k] for k in CONTENT_KEYS})
        match = got == entry["sha256_content"]
        print(f"  {name:11s} content {'OK' if match else 'MISMATCH'}  {got[:16]}")
        ok &= match
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="build a single fixture by name")
    ap.add_argument("--verify", action="store_true", help="check .npz against manifest.json")
    args = ap.parse_args()

    if args.verify:
        raise SystemExit(0 if verify([args.only] if args.only else None) else 1)

    specs = [s for s in SPECS if args.only is None or s[0] == args.only]
    print(f"building {len(specs)} fixture(s), schema v{SCHEMA_VERSION}, cutoff {CUTOFF} A")
    entries = [make(*s) for s in specs]

    if args.only and MANIFEST.exists():  # keep the other entries intact
        old = {e["name"]: e for e in json.loads(MANIFEST.read_text())["fixtures"]}
        old.update({e["name"]: e for e in entries})
        entries = [old[n] for n, *_ in SPECS if n in old]
    MANIFEST.write_text(
        json.dumps({"schema_version": SCHEMA_VERSION, "cutoff": CUTOFF,
                    "fixtures": entries}, indent=2) + "\n"
    )
    print(f"wrote {MANIFEST}")
