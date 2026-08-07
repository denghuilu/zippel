"""Per-group DRAM traffic model (DECISIONS.md D27).

D24 established that the FLOPs a fused kernel saves are not the mechanism -- the loads and stores
it avoids are. So the objective function for grouping and rematerialisation has to be **bytes**,
not term counts, and this module is that objective.

The model is *compulsory* traffic: each segment-varying buffer a group reads or writes crosses
DRAM once, and each static (`none`-segment) operand crosses once per launch regardless of how
many threads read it, because it is small and L2-resident. Under that model a fused group's
traffic is

    live_in_bytes + live_out_bytes + smem_spill

and the group's *internal* buffers contribute nothing at all -- which is precisely the quantity
Phase 2 is trying to maximise, expressed in the units the hardware charges for.

Two honest caveats, both of which the calibration exists to expose:

* It is a **lower** bound on real traffic, not an upper one. Poor locality (a gather with no
  reuse, a write pattern that misses) makes actual DRAM traffic exceed compulsory traffic, and
  nothing here predicts by how much. Per D26 an estimator that gates decisions must be an upper
  bound *or* carry a falsification test; this one takes the second route, and
  `bench/traffic_calibrate.py` is that test.
* The L2-residency assumption for static operands is a claim about size, not a proof. A weight
  table larger than L2 would be re-fetched per CTA and the model would understate badly. The
  estimator reports which operands it assumed resident so the assumption is visible rather than
  buried.

**Until calibrated to within ±20 %, this must not drive a fusion or template decision** (D27).
`calibrated()` is the gate, and it reads the recorded calibration rather than trusting the model.
"""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass, field

from zippel.ir import IndexType, Program

#: GH200 L2 is 60 MB; an operand comfortably under that is assumed to stay resident across the
#: launch. Deliberately conservative -- half the cache, not all of it.
L2_RESIDENT_BYTES = 30 * 1024 * 1024

CALIBRATION_PATH = pathlib.Path(__file__).resolve().parent.parent / \
    "bench/results/traffic_calibration.json"


@dataclass
class TrafficEstimate:
    group: str
    live_in_bytes: int = 0
    live_out_bytes: int = 0
    smem_bytes: int = 0
    internal_bytes_avoided: int = 0
    resident_operands: list[str] = field(default_factory=list)
    streamed_operands: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.live_in_bytes + self.live_out_bytes

    def __str__(self) -> str:
        return (f"{self.group}: in {self.live_in_bytes/2**20:.2f} MiB + "
                f"out {self.live_out_bytes/2**20:.2f} MiB = {self.total/2**20:.2f} MiB "
                f"(avoided {self.internal_bytes_avoided/2**20:.2f} MiB, "
                f"smem {self.smem_bytes} B)")


#: L2 cache-line size: the granularity at which L2 fetches from DRAM. Not the 32-byte L1<->L2
#: sector -- a strided read that touches one element still pulls a whole 128-byte line from DRAM.
SECTOR_BYTES = 128


def read_fraction(prog: Program, sched, itemsize: int) -> dict[str, float]:
    """Fraction of each live-in that a schedule's reads actually pull from DRAM.

    Two refinements over "all of it", both forced by measurement:

    1. The emitter loads only the elements its terms reference, so a block-diagonal operand does
       not cost its whole buffer. Charging the full buffer over-predicted the Wigner chain by
       28 %.
    2. But it does not cost the element fraction either, because **DRAM moves 32-byte sectors**.
       Reading 35 scattered elements of a row-major 9x9 FP64 block touches 16 of its 21 sectors,
       so the real cost is 76 % of the buffer, not 35/81 = 43 %. Charging the element fraction
       under-predicted the same kernel by 90 %.

    Sector occupancy is the physics between those two errors, and it is computable from the
    layout and the index set alone.
    """
    from codegen.tile import Ch

    seen: dict[str, set] = {}
    live_in = set(sched.spec.live_in)
    for a in sched.assigns:
        for term in a.terms:
            for factor in term.factors:
                if factor[0] in live_in:
                    seen.setdefault(factor[0], set()).add(factor[1])
        if a.source is not None and a.source[0] in live_in:
            seen.setdefault(a.source[0], set()).add(a.source[1])

    per_sector = max(SECTOR_BYTES // itemsize, 1)
    channel_extent = getattr(sched, "extent", 1)
    out: dict[str, float] = {}

    for buf, idxs in seen.items():
        t = prog.type_of(buf)
        if isinstance(t, IndexType):
            continue
        shape = t.sizes or (1,)
        strides = []
        acc = 1
        for size in reversed(shape):
            strides.append(acc)
            acc *= size
        strides.reverse()

        offsets: set[int] = set()
        for idx in idxs:
            base = [0]
            for pos, i in enumerate(idx):
                if isinstance(i, Ch):
                    base = [b + c * strides[pos] for b in base for c in range(channel_extent)]
                else:
                    base = [b + i * strides[pos] for b in base]
            offsets.update(base)

        sectors = {o // per_sector for o in offsets}
        total_sectors = math.ceil(acc / per_sector)
        out[buf] = min(1.0, len(sectors) / total_sectors) if total_sectors else 1.0
    return out


def _bytes_of(t, sizes: dict[str, int], itemsize: int) -> int:
    if isinstance(t, IndexType):
        return sizes.get(t.segment, 1) * 8
    n = sizes.get(t.segment, 1) if t.segment != "none" else 1
    return n * math.prod(t.sizes or (1,)) * itemsize


def estimate(prog: Program, spec, sizes: dict[str, int], itemsize: int = 4,
             smem_bytes: int = 0, reads: dict[str, int] | None = None) -> TrafficEstimate:
    """Compulsory DRAM traffic for one fused group.

    `reads` maps a live-in to the fraction of its bytes the kernel actually pulls, at DRAM
    sector granularity (see `read_fraction`). Without it every live-in is charged in full, which
    over-predicts any group whose schedule exploits structural sparsity.
    """
    est = TrafficEstimate(group=spec.name, smem_bytes=smem_bytes)

    for buf in spec.live_in:
        t = prog.type_of(buf)
        nbytes = _bytes_of(t, sizes, itemsize)
        if reads is not None and buf in reads and not isinstance(t, IndexType):
            nbytes = int(nbytes * reads[buf])
        est.live_in_bytes += nbytes
        (est.resident_operands if t.segment == "none" and nbytes <= L2_RESIDENT_BYTES
         else est.streamed_operands).append(buf)

    # Live-outs are charged in full: the emitter writes structural zeros as literals rather
    # than leaving them stale, so every element of an output buffer is stored.
    for buf in spec.live_out:
        est.live_out_bytes += _bytes_of(prog.type_of(buf), sizes, itemsize)

    for buf in spec.internal:
        est.internal_bytes_avoided += 2 * _bytes_of(prog.type_of(buf), sizes, itemsize)

    return est


def program_traffic(prog: Program, groups, sizes: dict[str, int], itemsize: int = 4) -> dict:
    """Whole-program traffic under a given partition, and the unfused counterfactual."""
    from codegen.schedule import analyze_group

    fused = 0
    avoided = 0
    per_group = []
    for i, g in enumerate(groups):
        spec = analyze_group(prog, g, name=f"g{i}")
        e = estimate(prog, spec, sizes, itemsize)
        per_group.append(e)
        fused += e.total
        avoided += e.internal_bytes_avoided
    return {"fused_bytes": fused, "avoided_bytes": avoided,
            "unfused_bytes": fused + avoided, "groups": per_group}


def calibrated(template: str | None = None, tolerance: float = 0.20) -> tuple[bool, str]:
    """Has the model been checked against measured traffic for `template`, and did it pass?

    The D27 gate, and it is **per template** rather than global, because the measurement came
    out that way: the model predicts dense-access (T2) groups to within ~5 % and sparse
    strided-read (T1) groups to +30 %. A single global verdict would either forbid a use that is
    well supported or permit one that is not.

    Pass `template=None` to ask whether *everything* is calibrated. Callers that would let the
    model decide something must consult this; a model that has never been measured is not
    calibrated no matter how reasonable it looks.
    """
    if not CALIBRATION_PATH.exists():
        return False, f"no calibration recorded at {CALIBRATION_PATH}"
    data = json.loads(CALIBRATION_PATH.read_text())
    if data.get("blocked_instrument"):
        return False, f"instrument unusable: {data['blocked_instrument']}"

    rows = [r for r in data.get("rows", [])
            if r.get("stage") == "model" and r.get("rel_error") is not None
            and (template is None or r.get("template") == template)]
    if not rows:
        return False, (f"no measured rows for template {template!r}" if template
                       else "calibration file has no comparable model rows")
    worst = max(abs(r["rel_error"]) for r in rows)
    label = template or "all templates"
    if worst > tolerance:
        return False, (f"{label}: worst relative error {worst:.1%} exceeds +-{tolerance:.0%}; "
                       f"recalibrate before this drives a fusion or template decision")
    return True, f"{label}: worst relative error {worst:.1%} within +-{tolerance:.0%}"


__all__ = ["TrafficEstimate", "estimate", "read_fraction", "program_traffic", "calibrated",
           "SECTOR_BYTES",
           "L2_RESIDENT_BYTES", "CALIBRATION_PATH"]
