"""Render a scalar schedule as CuTe DSL source.

The emitter generates Python source text for a `@cute.kernel`, writes it to `_generated/`, and
imports it. Generating source rather than calling a builder API is deliberate: the emitted kernel
is a readable artifact that can be diffed, pasted into a bug report, and audited against the IR
it came from. It is also forced -- CuTe DSL reads a decorated function's source with
`inspect.getsourcelines`, so a kernel that exists only as a string cannot compile. `emit_source()` is the whole compiler backend for this bucket -- the thing a
human writes, as opposed to the ~350 kernels it writes (REPORT.md section on archetypes).

Bucket A: register-resident groups. One thread owns one segment element and holds every trailing
value of every live buffer in registers, so the group's internal buffers never reach memory.
Applicable when `Schedule.peak_live_values()` fits the register file; `codegen/schedule.py`
measures that, and `emit_source` refuses rather than silently spilling.

Everything is a runtime tensor. No coefficient is folded into the emitted code beyond the path
coefficients that are part of the IR itself -- weights, Wigner inputs and rotation matrices are
all read from memory, per the anti-gaming rule in the work order.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import textwrap

from zippel.ir import IndexType, Program
from codegen.schedule import Schedule, all_indices

#: Registers per thread we are willing to ask for. Above this the schedule spills to local
#: memory and the whole point of fusing the group is lost, so refuse and report instead.
REGISTER_BUDGET = 168

DTYPE = {"f64": "Float64", "f32": "Float32"}


def _sym(buf: str, idx: tuple[int, ...]) -> str:
    return f"v_{buf}" + ("".join(f"_{i}" for i in idx) if idx else "")


def _ref(prog: Program, buf: str, idx: tuple[int, ...], seg: str) -> str:
    """A gmem reference `m_<buf>[...]`.

    Every buffer carries a leading segment coordinate, matching the interpreter's convention
    that a `none`-segment buffer is stored with a length-1 segment axis
    (`zippel/interp.py:segment_length`). So a static [9,9] operand is `m_jd[0, i, j]`, and the
    two conventions cannot drift apart silently.
    """
    t = prog.type_of(buf)
    lead = "e" if t.segment != "none" else "0"
    return f"m_{buf}[{', '.join([lead] + [str(i) for i in idx])}]"


def _fn_expr(fn: str, order: int, arg: str, dt: str) -> str:
    """Scalar vocabulary in CuTe DSL. Each must match zippel/interp.py exactly."""
    if fn == "exp":
        return f"cute.math.exp({arg})"
    if fn == "sigmoid":
        return f"({dt}(1.0) / ({dt}(1.0) + cute.math.exp(-{arg})))"
    if fn == "silu":
        return f"({arg} / ({dt}(1.0) + cute.math.exp(-{arg})))"
    if fn == "rsqrt":
        return f"cute.math.rsqrt({arg})"
    if fn == "reciprocal":
        return f"({dt}(1.0) / {arg})"
    if fn == "sin":
        return f"cute.math.sin({arg})"
    if fn == "cos":
        return f"cute.math.cos({arg})"
    if fn == "poly_envelope":
        raise NotImplementedError("poly_envelope needs its order-specific polynomial (D16)")
    raise NotImplementedError(f"no CuTe DSL lowering for scalar_map {fn!r}")


def _term_expr(term, dt: str) -> str:
    factors = " * ".join(_sym(b, i) for b, i in term.factors)
    if term.coeff == 1.0:
        return factors
    if term.coeff == -1.0:
        return f"-({factors})"
    return f"{dt}({term.coeff!r}) * {factors}"


#: Terms per emitted statement; see codegen/emit_tile.py. Chunking is left-to-right, so the
#: summation order -- and therefore the ordering bound -- is unchanged.
CHUNK = 48


def _chunked_sum(target: str, parts: list[str], uid: int) -> list[str]:
    if len(parts) <= CHUNK:
        return [f"{target} = " + " + ".join(parts)]
    lines, acc = [], None
    for k in range(0, len(parts), CHUNK):
        piece = parts[k:k + CHUNK]
        name = f"_s{uid}_{k // CHUNK}"
        lines.append(f"{name} = " + " + ".join(([acc] if acc else []) + piece))
        acc = name
    lines.append(f"{target} = {acc}")
    return lines


def emit_source(prog: Program, sched: Schedule, block: int = 128,
                dtype: str = "f32", budget: int = REGISTER_BUDGET) -> str:
    """Emit a complete CuTe DSL module for one register-resident fusion group."""
    spec = sched.spec
    peak = sched.peak_live_values()
    if peak > budget:
        raise ValueError(
            f"group {spec.name} needs {peak} live scalars per thread, over the {budget} register "
            f"budget -- it is not a register-resident group. Use a channel-parallel mapping "
            f"(bucket B) or split the group.")
    dt = DTYPE[dtype]
    depth = max((len(a.terms) for a in sched.assigns), default=1)

    # tensors the kernel takes, in a stable order
    tensors = [b for b in spec.live_in if not isinstance(prog.type_of(b), IndexType)]
    tensors += list(spec.live_out)
    params = ", ".join(f"m_{b}: cute.Tensor" for b in tensors)

    body: list[str] = []

    # Load exactly the live-in elements the schedule reads -- nothing more. A dense load of
    # every trailing element would read the structural zeros this pass exists to skip.
    wanted: dict[str, set[tuple[int, ...]]] = {}
    for a in sched.assigns:
        for t in a.terms:
            for buf, idx in t.factors:
                if buf in spec.live_in:
                    wanted.setdefault(buf, set()).add(idx)
        if a.source is not None and a.source[0] in spec.live_in:
            wanted.setdefault(a.source[0], set()).add(a.source[1])
    for buf in spec.live_in:
        for idx in sorted(wanted.get(buf, ())):
            body.append(f"{_sym(buf, idx)} = {_ref(prog, buf, idx, spec.segment)}")

    for i, a in enumerate(sched.assigns):
        if a.fn is not None:
            body.append(f"{_sym(a.target, a.index)} = "
                        f"{_fn_expr(a.fn, a.order, _sym(*a.source), dt)}")
        elif not a.terms:
            body.append(f"{_sym(a.target, a.index)} = {dt}(0.0)")
        else:
            parts = [_term_expr(t, dt) for t in a.terms]
            body.extend(_chunked_sum(_sym(a.target, a.index), parts, i))

    for buf in spec.live_out:
        for idx in sorted(sched.masks[buf]):
            body.append(f"{_ref(prog, buf, idx, spec.segment)} = {_sym(buf, idx)}")
        # Structural zeros still have to be *written*, or the consumer reads stale memory.
        # They are stored as a literal, never computed.
        for idx in sorted(all_indices(prog.type_of(buf).sizes) - sched.masks[buf]):
            body.append(f"{_ref(prog, buf, idx, spec.segment)} = {dt}(0.0)")

    indented = textwrap.indent("\n".join(body), " " * 12)
    return f'''"""Generated by codegen/emit.py from fusion group {spec.name}. Do not edit.

{spec}
  {sched.n_values} values, {sched.n_terms} terms, {peak} live scalars/thread.
  Internal buffers never stored: {", ".join(spec.internal) or "(none)"}
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Float64, Int32, const_expr
from cutlass.cute.runtime import from_dlpack

BLOCK = {block}
TENSOR_ORDER = {tensors!r}

#: Correctness contract for this kernel (DECISIONS.md D25). REDUCTION_DEPTH is the most terms
#: any single output element sums; the harness turns it into a numeric bound against real
#: inputs and asserts measured <= bound. EXACT additionally demands bit-equality.
#: Which segment axis this kernel iterates. The caller must pass that segment's length as
#: `n_seg`; passing another segment's length indexes past the end of every buffer. A node-rooted
#: group launched with the edge count segfaults, which is how this came to be declared.
SEGMENT = "{spec.segment}"
TEMPLATE = "T1"
REDUCTION_DEPTH = {depth}
EXACT = True


class Kernel:
    """One thread per {spec.segment}; the group's {len(spec.internal)} internal buffers stay in registers."""

    @cute.jit
    def __call__(self, {params}, n_seg: Int32, stream):
        self.kernel({", ".join(f"m_{b}" for b in tensors)}, n_seg).launch(
            grid=[(n_seg + BLOCK - 1) // BLOCK, 1, 1], block=[BLOCK, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, {params}, n_seg: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        e = bidx * BLOCK + tidx
        if e < n_seg:
{indented}
'''


#: Where emitted kernels land. They are real files on purpose, not exec'd strings: CuTe DSL
#: reads a decorated function's source with `inspect.getsourcelines`, so a kernel that exists
#: only in memory fails to compile ("DSL does not support REPL mode"). Writing them out also
#: makes every generated kernel an inspectable, diffable artifact.
GENERATED_DIR = pathlib.Path(__file__).resolve().parent / "_generated"


def build_kernel(source: str, name: str, directory: pathlib.Path | None = None):
    """Write the emitted module, import it, and hand back its `Kernel` and tensor order."""
    directory = directory or GENERATED_DIR
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "__init__.py").touch(exist_ok=True)

    path = directory / f"{name}.py"
    if not path.exists() or path.read_text() != source:
        path.write_text(source)

    spec = importlib.util.spec_from_file_location(f"zippel_generated.{name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module          # inspect.getsource needs it importable by name
    spec.loader.exec_module(module)
    return module.Kernel, module.TENSOR_ORDER


__all__ = ["emit_source", "build_kernel", "REGISTER_BUDGET", "GENERATED_DIR"]
