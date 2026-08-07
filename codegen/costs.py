"""Per-kernel build-cost metadata: schedule, emit, compile.

Recorded from now on with no analysis attached. Phase 2's reviewers will ask what the compiler
costs to run, and that column cannot be reconstructed later -- it has to be collected as kernels
are built. Three phases are timed separately because they have different causes and different
fixes:

  schedule   building the scalar schedule from the IR. Driven by the *dense* index-space volume
             rather than the emitted term count (bench/schedule_scaling.py), so it is the phase
             that scales worst into S3.
  emit       rendering the schedule as CuTe DSL source. Roughly linear in emitted terms.
  compile    `cute.compile` -- NVRTC/NVVM through the DSL. Charged once per distinct kernel and
             cached in CUTE_DSL_CACHE_DIR, so a warm run reports near-zero and a cold one does
             not. Which of those a number came from is recorded alongside it.

Nothing here feeds a decision. It is a ledger.
"""

from __future__ import annotations

import json
import pathlib
import time
from contextlib import contextmanager

_COSTS: dict[str, dict] = {}


@contextmanager
def phase(name: str, stage: str, **extra):
    """Time one build phase of one kernel.

    >>> with phase("wigner_chain", "emit", terms=356):
    ...     source = emit_source(prog, sched)
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        entry = _COSTS.setdefault(name, {})
        entry[f"{stage}_s"] = time.perf_counter() - start
        entry.update(extra)


def record(name: str, **fields) -> None:
    _COSTS.setdefault(name, {}).update(fields)


def costs() -> dict[str, dict]:
    return {k: dict(v) for k, v in _COSTS.items()}


def total(stage: str) -> float:
    return sum(v.get(f"{stage}_s", 0.0) for v in _COSTS.values())


def dump(path: str | pathlib.Path = "bench/results/kernel_costs.json") -> pathlib.Path:
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"kernels": costs(),
         "totals": {s: total(s) for s in ("schedule", "emit", "compile")}}, indent=2))
    return out


def reset() -> None:
    _COSTS.clear()


__all__ = ["phase", "record", "costs", "total", "dump", "reset"]
