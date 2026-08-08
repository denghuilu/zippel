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

from zippel.interp import _A, _B, _C
from zippel.ir import IndexType, Program
from codegen.schedule import Schedule, all_indices

#: Files whose content defines what an emitted kernel *is*. A generated artifact records the
#: hash of these, not a repo git SHA: a git SHA misses uncommitted edits, and during S1 the T3
#: emitter changed underneath an already-measured composition. Every kernel must trace to its
#: generator's exact content.
_EMITTER_SOURCES = ("emit_common.py", "emit.py", "emit_tile.py", "emit_reduce.py",
                    "schedule.py", "tile.py",
                    "bounds.py")


def emitter_sha() -> str:
    """Content hash of the emitter itself. Stable across runs, sensitive to any edit."""
    import hashlib

    h = hashlib.sha256()
    here = pathlib.Path(__file__).resolve().parent
    for name in _EMITTER_SOURCES:
        f = here / name
        h.update(name.encode())
        h.update(f.read_bytes() if f.exists() else b"")
    return h.hexdigest()[:16]


#: Registers per thread we are willing to ask for. Above this the schedule spills to local
#: memory and the whole point of fusing the group is lost, so refuse and report instead.

#: Re-exported from the substrate so existing importers keep working (D70a).
from codegen.emit_common import (CHUNK, DTYPE, REGISTER_BUDGET, chunked_sum,  # noqa: E402,F401
                                metadata_block, ref, sym)


def _sym(buf: str, idx: tuple[int, ...]) -> str:
    return sym(buf, idx)


def _ref(prog: Program, buf: str, idx: tuple[int, ...], seg: str) -> str:
    """A gmem reference `m_<buf>[...]`.

    Every buffer carries a leading segment coordinate, matching the interpreter's convention
    that a `none`-segment buffer is stored with a length-1 segment axis
    (`zippel/interp.py:segment_length`). So a static [9,9] operand is `m_jd[0, i, j]`, and the
    two conventions cannot drift apart silently.
    """
    return ref(prog, buf, idx)


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
        # p(d) = 1 + a d^5 + b d^6 + c d^7, and its derivatives, zero for d >= 1. Written in the
        # same Horner form as zippel/interp.py:_envelope so the two agree term for term rather
        # than merely mathematically -- a differently-associated polynomial would round
        # differently and the kernel would miss its own bound for no reason.
        d = arg
        if order == 0:
            val = f"({dt}(1.0) + ({d})**5 * ({dt}({_A!r}) + ({d}) * ({dt}({_B!r}) + {dt}({_C!r}) * ({d}))))"
        elif order == 1:
            val = (f"({dt}({5 * _A!r}) * ({d})**4 + {dt}({6 * _B!r}) * ({d})**5 + "
                   f"{dt}({7 * _C!r}) * ({d})**6)")
        elif order == 2:
            val = (f"({dt}({20 * _A!r}) * ({d})**3 + {dt}({30 * _B!r}) * ({d})**4 + "
                   f"{dt}({42 * _C!r}) * ({d})**5)")
        elif order == 3:
            val = (f"({dt}({60 * _A!r}) * ({d})**2 + {dt}({120 * _B!r}) * ({d})**3 + "
                   f"{dt}({210 * _C!r}) * ({d})**4)")
        else:
            raise NotImplementedError(f"poly_envelope order {order} (D16)")
        # the cutoff is a select, not a branch: threads in a warp straddle d = 1
        return f"({val} if ({d}) < {dt}(1.0) else {dt}(0.0))"
    raise NotImplementedError(f"no CuTe DSL lowering for scalar_map {fn!r}")


def _term_expr(term, dt: str) -> str:
    factors = " * ".join(_sym(b, i) for b, i in term.factors)
    if term.coeff == 1.0:
        return factors
    if term.coeff == -1.0:
        return f"-({factors})"
    return f"{dt}({term.coeff!r}) * {factors}"


def _chunked_sum(target: str, parts: list[str], uid: int) -> list[str]:
    return chunked_sum(target, parts, uid)


#: The prose belongs to the template; the field set, order and formatting belong to the substrate.
T1_NOTES = """#: Correctness contract for this kernel (DECISIONS.md D25). REDUCTION_DEPTH is the most terms
#: any single output element sums; the harness turns it into a numeric bound against real
#: inputs and asserts measured <= bound. EXACT additionally demands bit-equality.
#: Which segment axis this kernel iterates. The caller must pass that segment's length as
#: `n_seg`; passing another segment's length indexes past the end of every buffer. A node-rooted
#: group launched with the edge count segfaults, which is how this came to be declared."""


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
    _esha = emitter_sha()
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

{metadata_block(spec.segment, "T1", _esha, depth, True, notes=T1_NOTES)}


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


#: Metadata every generated module must declare. These are the kernel's *contract*, checked at
#: load rather than trusted: SEGMENT fixes the launch geometry, REDUCTION_DEPTH fixes the
#: numerical bound, TEMPLATE and EXACT fix which correctness tier applies.
REQUIRED_METADATA = ("TENSOR_ORDER", "SEGMENT", "TEMPLATE", "REDUCTION_DEPTH", "EXACT",
                     "EMITTER_SHA")


class MetadataMismatch(RuntimeError):
    """A generated module's declared contract disagrees with the schedule that produced it."""


def build_kernel(source: str, name: str, directory: pathlib.Path | None = None,
                 sched=None):
    """Write the emitted module, import it, validate its contract, and return it.

    Passing `sched` turns the module's metadata into a **checked** contract instead of a comment.
    `SEGMENT` in particular is launch geometry: a node-rooted group launched with the edge count
    reads past the end of every buffer and segfaults, so it belongs with TEMPLATE / DEPTH / EXACT
    as something verified at load rather than remembered by the caller.
    """
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

    missing = [k for k in REQUIRED_METADATA if not hasattr(module, k)]
    if missing:
        raise MetadataMismatch(f"{name} declares no {', '.join(missing)}")

    # A generated file on disk can outlive the emitter that wrote it. Refuse a stale artifact
    # rather than measure it: during S1 the T3 emitter changed under an already-measured
    # composition, and nothing would have noticed.
    current = emitter_sha()
    if module.EMITTER_SHA != current:
        raise MetadataMismatch(
            f"{name} was generated by emitter {module.EMITTER_SHA}, but the emitter is now "
            f"{current}. Regenerate; do not measure a kernel whose generator has changed.")

    if sched is not None:
        want_seg = sched.spec.segment
        if module.SEGMENT != want_seg:
            raise MetadataMismatch(
                f"{name} declares SEGMENT={module.SEGMENT!r} but its group is rooted on "
                f"{want_seg!r}; launching it would index a buffer with the wrong extent")
        driving = getattr(module, "DRIVING_SEGMENT", module.SEGMENT)
        if driving == want_seg and getattr(module, "SCATTERS", False):
            raise MetadataMismatch(
                f"{name} declares SCATTERS but its DRIVING_SEGMENT equals its SEGMENT; a scatter "
                f"iterates the segment it reads, not the one it writes")
        want_depth = max((len(a.terms) for a in sched.assigns), default=1)
        if module.REDUCTION_DEPTH != want_depth:
            raise MetadataMismatch(
                f"{name} declares REDUCTION_DEPTH={module.REDUCTION_DEPTH} but its schedule has "
                f"{want_depth}; the numerical bound would be computed for a different kernel")

    return module.Kernel, module.TENSOR_ORDER


def load_metadata(name: str) -> dict:
    """The declared contract of an already-built kernel."""
    module = sys.modules[f"zippel_generated.{name}"]
    return {k: getattr(module, k) for k in REQUIRED_METADATA}


__all__ = ["emit_source", "build_kernel", "load_metadata", "REQUIRED_METADATA",
           "MetadataMismatch", "REGISTER_BUDGET", "GENERATED_DIR"]
