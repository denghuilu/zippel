"""Collate bench/results/*.json into the markdown tables used in REPORT.md.

Reads whatever result files exist; missing baselines are reported as missing rather than
silently dropped, so a partial run cannot look like a complete one.

    python bench/summarize.py
"""

from __future__ import annotations

import json
import pathlib

RESULTS = pathlib.Path(__file__).resolve().parent / "results"
FIXTURE_ORDER = ["si_small", "cu_small", "si_medium", "cu_medium", "si_large", "cu_large"]
PRECISIONS = ["fp32", "tf32", "bf16"]


def _load(name: str) -> list[dict]:
    path = RESULTS / name
    if not path.exists():
        return []
    raw = json.loads(path.read_text())
    return raw.get("measurements", raw) if isinstance(raw, dict) else raw


def main():
    files = {
        "B1 eager": "b1_eager.json",
        "B2 torch.compile (backend ladder)": "b2_probe.json",
        "B2 torch.compile (full inventory)": "b2_compile.json",
        "B2 torch.compile (max-autotune)": "b2_maxautotune.json",
        "B3 cuEquivariance": "b3_cueq.json",
    }
    rows: list[dict] = []
    for tag, fname in files.items():
        got = _load(fname)
        if not got:
            print(f"MISSING: {tag} ({fname}) — not run")
        rows.extend(got)

    if not rows:
        print("no results found")
        return

    labels = sorted({r["label"] for r in rows})
    print(f"\n{'label':26s} {'fixture':11s} {'prec':5s} {'median ms':>11s} "
          f"{'IQR':>8s} {'peak GiB':>9s}  notes")
    print("-" * 96)
    for label in labels:
        for fixture in FIXTURE_ORDER:
            for precision in PRECISIONS:
                m = next((r for r in rows if r["label"] == label
                          and r["fixture"] == fixture and r["precision"] == precision), None)
                if m is None:
                    continue
                if m.get("error"):
                    print(f"{label:26s} {fixture:11s} {precision:5s} {'—':>11s} "
                          f"{'—':>8s} {'—':>9s}  {m['error'][:40]}")
                else:
                    print(f"{label:26s} {fixture:11s} {precision:5s} "
                          f"{m['median_ms']:11.2f} {m['iqr_ms']:8.2f} "
                          f"{m['peak_mem_gib']:9.2f}  {m.get('notes','')[:34]}")

    # speedup vs the strongest baseline, per (fixture, precision)
    print("\nbest baseline per (fixture, precision):")
    for fixture in FIXTURE_ORDER:
        for precision in PRECISIONS:
            cands = [r for r in rows if r["fixture"] == fixture
                     and r["precision"] == precision and not r.get("error")]
            if not cands:
                continue
            best = min(cands, key=lambda r: r["median_ms"])
            print(f"  {fixture:11s} {precision:5s}  {best['label']:26s} "
                  f"{best['median_ms']:9.2f} ms  {best['peak_mem_gib']:6.2f} GiB")


if __name__ == "__main__":
    main()
