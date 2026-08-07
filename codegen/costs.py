"""Per-kernel build-cost metadata: schedule, emit, compile.

Recorded from now on with no analysis attached. Phase 2's reviewers will ask what the compiler
costs to run, and that column cannot be reconstructed later -- it has to be collected as kernels
are built. Three phases are timed separately because they have different causes and different
fixes:

  schedule   building the scalar schedule from the IR. Linear in the dense index-space volume
             it enumerates (k = 0.96-1.01, R^2 >= 0.97) since D31 removed a quadratic liveness
             scan; ~41 % of it is the double walk that D33 leaves on the backlog.
  emit       rendering the schedule as CuTe DSL source. Roughly linear in emitted terms.
  compile    `cute.compile` -- NVRTC/NVVM through the DSL. Charged once per distinct kernel and
             cached in CUTE_DSL_CACHE_DIR, so a warm run reports near-zero and a cold one does
             not. Which of those a number came from is recorded alongside it.
  guard      the verification infrastructure itself: the register-budget check, the Kahn
             acyclicity sort, metadata validation at load, closure assertions.

**Guards are timed separately and by name, not folded into the phases they protect.** They are
part of what this compiler costs to run, and a verification budget that is invisible inside the
compile budget is a verification budget nobody can argue with. It is also not hypothetical: the
register-budget guard was 97 % of schedule construction until it was profiled (D31), and it was
sitting inside a phase that reported it as "schedule" time.

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


#: The phases a kernel's build time is attributed to. `guard` is verification, kept distinct.
STAGES = ("schedule", "emit", "compile", "guard")


def total(stage: str) -> float:
    return sum(v.get(f"{stage}_s", 0.0) for v in _COSTS.values())


def summary() -> dict:
    """Totals per phase, plus guard time as a share of the whole."""
    totals = {s: total(s) for s in STAGES}
    whole = sum(totals.values())
    return {"totals_s": totals, "total_s": whole,
            "guard_fraction": (totals["guard"] / whole) if whole else 0.0,
            "kernels": len(_COSTS)}


def dump(path: str | pathlib.Path = "bench/results/kernel_costs.json") -> pathlib.Path:
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"kernels": costs(), **summary()}, indent=2))
    return out


def reset() -> None:
    _COSTS.clear()


__all__ = ["phase", "record", "costs", "total", "summary", "dump", "reset", "STAGES"]
