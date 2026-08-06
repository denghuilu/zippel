"""Shared measurement harness. Every implementation is timed through this and only this.

The measured unit (work order section 3) is one **conservative training step**:

    E  = block(pos, ...)
    F  = -dE/dpos                       with create_graph=True
    L  = w_E * MSE(E) + w_F * MSE(F)
    L.backward()                        -> all parameter grads and pos grad

Neighbour-list construction is outside the boundary for every implementation equally;
fixtures arrive already on device. Timing is CUDA-event based around exactly that
boundary, so host-side Python setup that precedes the first kernel is excluded the same
way for everyone.

Binding anti-gaming rules this file enforces mechanically:
  * identical measured boundary for all parties (there is one timing function);
  * identical inputs (the fixture loader is the only source of batches);
  * grads are zeroed *outside* the timed region, so no implementation gets to skip it;
  * a liveness assertion on the first iteration -- if an implementation silently fails to
    produce parameter or position gradients, it cannot post a fast time.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

import torch

GIB = 1024 ** 3  # units are GiB everywhere (DECISIONS.md D13)


@dataclass
class Measurement:
    label: str
    fixture: str
    precision: str
    median_ms: float
    iqr_ms: float
    p05_ms: float
    p95_ms: float
    peak_mem_gib: float
    iters: int
    atoms: int
    edges: int
    clocks_mhz: float | None = None
    notes: str = ""
    error: str | None = None
    extra: dict = field(default_factory=dict)

    def row(self) -> str:
        if self.error:
            return f"| {self.label} | {self.fixture} | {self.precision} | — | — | {self.error} |"
        return (f"| {self.label} | {self.fixture} | {self.precision} | "
                f"{self.median_ms:.2f} | {self.iqr_ms:.2f} | {self.peak_mem_gib:.2f} |")


def gpu_clocks_mhz() -> float | None:
    """Current SM clock, recorded alongside every timing so throttling is visible."""
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
        return float(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
    except Exception:
        return None


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    pos = q * (len(sorted_vals) - 1)
    lo, hi = int(pos), min(int(pos) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def iters_for(n_atoms: int) -> tuple[int, int]:
    """Work order: >=20 warmup and >=100 measured iters, >=30 at the 50k fixture."""
    return (20, 30) if n_atoms >= 20_000 else (20, 100)


def time_training_step(
    step_fn,
    zero_grads_fn,
    label: str,
    fixture: str,
    precision: str,
    atoms: int,
    edges: int,
    warmup: int | None = None,
    iters: int | None = None,
    liveness_fn=None,
    notes: str = "",
) -> Measurement:
    """Time `step_fn` (one full conservative training step) with CUDA events.

    `zero_grads_fn` runs outside the timed region. `liveness_fn` is called once after the
    first warmup step and must raise if gradients are missing or non-finite.
    """
    w, n = warmup or iters_for(atoms)[0], iters or iters_for(atoms)[1]

    try:
        for k in range(w):
            zero_grads_fn()
            step_fn()
            if k == 0 and liveness_fn is not None:
                liveness_fn()
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        samples: list[float] = []
        for _ in range(n):
            zero_grads_fn()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            step_fn()
            end.record()
            torch.cuda.synchronize()
            samples.append(start.elapsed_time(end))
        peak = torch.cuda.max_memory_allocated() / GIB
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return Measurement(label, fixture, precision, float("nan"), float("nan"),
                           float("nan"), float("nan"), float("nan"), 0, atoms, edges,
                           notes=notes, error="OOM")

    samples.sort()
    return Measurement(
        label=label, fixture=fixture, precision=precision,
        median_ms=statistics.median(samples),
        iqr_ms=_percentile(samples, 0.75) - _percentile(samples, 0.25),
        p05_ms=_percentile(samples, 0.05), p95_ms=_percentile(samples, 0.95),
        peak_mem_gib=peak, iters=n, atoms=atoms, edges=edges,
        clocks_mhz=gpu_clocks_mhz(), notes=notes,
    )


def assert_step_is_live(module: torch.nn.Module, pos: torch.Tensor) -> None:
    """Refuse to time a step that did not actually produce the gradients it claims to.

    Guards against an implementation posting a fast number because part of the backward
    silently did nothing.
    """
    dead = [n for n, p in module.named_parameters()
            if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0]
    if dead:
        raise RuntimeError(f"parameters with missing/zero/non-finite grad: {dead[:6]}")
    if pos.grad is None or not torch.isfinite(pos.grad).all() or pos.grad.abs().sum() == 0:
        raise RuntimeError("position gradient missing, zero, or non-finite")
