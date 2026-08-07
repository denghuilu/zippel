"""Drive a whole program through emitted kernels: S1c.

Individually-correct kernels are not a correct program. This module chains them -- allocating
every buffer once, launching each group's kernel in dependence order, and handing each kernel the
buffers its predecessors wrote -- so that what is validated is the composition rather than a
collection.

Three things it has to get right that per-group validation never exercises:

  ordering        groups run in an order that respects the group DAG. The Kahn guard proves one
                  exists; this is where one is actually used.
  aliasing        a buffer written by one group and read by three others is one allocation, not
                  four. Getting this wrong shows up as a correct-looking kernel reading stale
                  memory.
  scatter zeroing an atomic scatter-add accumulates into whatever is already there, so its output
                  must be zeroed before the launch and only then. A buffer zeroed too late loses
                  the contributions already made; too early and it is fine but the test is weaker
                  than it looks.

`max_volume` is passed to the fusion pass: fusing a gather into a channel-heavy op forces it out
of T2 into T3's fully-unrolled form, 23 040 terms instead of 5 123, and 114 minutes of compile
instead of 9 (D36).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from codegen import costs
from codegen.emit import build_kernel, emit_source
from codegen.emit_reduce import emit_reduce_source, scatter_map
from codegen.emit_tile import emit_tile_source
from codegen.schedule import analyze_group, build_schedule, index_maps_used
from codegen.tile import build_tile_schedule, channel_axis
from zippel.ir import IndexType, Program
from zippel.simplify import fusion_groups, simplify

#: Index-space volume above which a group is split rather than fused (D36).
DEFAULT_MAX_VOLUME = 10_000


@dataclass
class CompiledGroup:
    index: int
    name: str
    template: str
    ops: tuple[str, ...]
    order: tuple[str, ...]
    live_out: tuple[str, ...]
    driving_segment: str
    scatters: bool
    launch: object = None
    terms: int = 0


@dataclass
class CompiledProgram:
    prog: Program
    groups: list[CompiledGroup] = field(default_factory=list)
    sizes: dict[str, int] = field(default_factory=dict)

    @property
    def n_launches(self) -> int:
        return len(self.groups)


def route(prog: Program, spec):
    """The selection rule (docs/templates.md 2). Returns `(template, schedule, emit_fn)`."""
    if index_maps_used(prog, spec):
        return "T3", build_schedule(prog, spec), emit_reduce_source
    sched = build_schedule(prog, spec)
    if sched.peak_live_values() <= 168:
        return "T1", sched, emit_source
    axis = channel_axis(prog, spec)
    if axis is None:
        return "T3", sched, emit_reduce_source
    return "T2", build_tile_schedule(prog, spec, *axis), emit_tile_source


def topological_groups(prog: Program, groups: list[list[str]]) -> list[int]:
    """A launch order for the group DAG. Raises if none exists -- see the Kahn guard."""
    where = {n: i for i, g in enumerate(groups) for n in g}
    succ: dict[int, set[int]] = {i: set() for i in range(len(groups))}
    indeg = dict.fromkeys(succ, 0)
    for name, op in prog.ops.items():
        for src in op.inputs:
            if src in where and where[src] != where[name]:
                if where[name] not in succ[where[src]]:
                    succ[where[src]].add(where[name])
                    indeg[where[name]] += 1

    ready = sorted(i for i, d in indeg.items() if d == 0)
    order: list[int] = []
    while ready:
        i = ready.pop(0)
        order.append(i)
        for j in sorted(succ[i]):
            indeg[j] -= 1
            if indeg[j] == 0:
                ready.append(j)
    if len(order) != len(groups):
        raise ValueError(f"group DAG has a cycle: only {len(order)} of {len(groups)} orderable")
    return order


def compile_program(prog: Program, sizes: dict[str, int], label: str,
                    dtype: str = "f64", max_volume: int | None = DEFAULT_MAX_VOLUME,
                    ) -> CompiledProgram:
    """Emit and compile every group, in launch order."""
    import cutlass
    import cutlass.cute as cute
    import sys

    simp = simplify(prog, keep=prog.outputs)
    groups = fusion_groups(simp, max_volume=max_volume)
    order = topological_groups(simp, groups)
    out = CompiledProgram(prog=simp, sizes=dict(sizes))

    for gi in order:
        spec = analyze_group(simp, groups[gi], name=f"{label}_g{gi}")
        with costs.phase(spec.name, "schedule"):
            template, sched, emit = route(simp, spec)
        with costs.phase(spec.name, "emit", template=template, terms=sched.n_terms):
            source = emit(simp, sched, dtype=dtype)
        kname = f"{label}_g{gi}_{dtype}"
        with costs.phase(spec.name, "guard"):
            Kernel, tensor_order = build_kernel(source, kname, sched=sched)
        module = sys.modules[f"zippel_generated.{kname}"]
        out.groups.append(CompiledGroup(
            index=gi, name=spec.name, template=template, ops=tuple(groups[gi]),
            order=tuple(tensor_order), live_out=tuple(spec.live_out),
            driving_segment=getattr(module, "DRIVING_SEGMENT", module.SEGMENT),
            scatters=getattr(module, "SCATTERS", False),
            launch=Kernel(), terms=sched.n_terms))
    return out


def allocate(cp: CompiledProgram, inputs: dict[str, torch.Tensor],
             dtype: torch.dtype = torch.float64) -> dict[str, torch.Tensor]:
    """One tensor per buffer, program inputs bound, everything else zeroed.

    Zeroed rather than left uninitialised on purpose: a scatter-add accumulates into what is
    already there, and an uninitialised output would make the result depend on the allocator.
    """
    env: dict[str, torch.Tensor] = {}
    for name, t in cp.prog.inputs.items():
        a = inputs[name]
        if isinstance(t, IndexType):
            env[name] = a.to(device="cuda", dtype=torch.int64).contiguous()
        else:
            env[name] = a.to(device="cuda", dtype=dtype).contiguous()

    for name, op in cp.prog.ops.items():
        t = op.out_type
        n = 1 if t.segment == "none" else cp.sizes[t.segment]
        env[name] = torch.zeros(n, *t.sizes, device="cuda", dtype=dtype)
    return env


def run_program(cp: CompiledProgram, env: dict[str, torch.Tensor], compile_kernels: bool = True):
    """Launch every group in order. Returns wall-clock seconds for the launch sequence alone."""
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32
    from cutlass.cute.runtime import from_dlpack

    stream = cutlass.cuda.default_stream()
    for g in cp.groups:
        args = tuple(from_dlpack(env[b], assumed_align=16) for b in g.order) + (
            Int32(cp.sizes[g.driving_segment]), stream)
        if compile_kernels and not hasattr(g, "_compiled"):
            t0 = time.perf_counter()
            g._compiled = cute.compile(g.launch, *args)
            torch.cuda.synchronize()
            costs.record(g.name, compile_s=time.perf_counter() - t0)
        # a scatter accumulates, so its target starts at zero every launch
        if g.scatters:
            for b in g.live_out:
                env[b].zero_()
        g._compiled(*args)
    torch.cuda.synchronize()


__all__ = ["CompiledGroup", "CompiledProgram", "compile_program", "allocate", "run_program",
           "route", "topological_groups", "DEFAULT_MAX_VOLUME"]
