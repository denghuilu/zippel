"""Render a T2 tile schedule as CuTe DSL source.

One CTA owns one segment element; one thread owns one channel. Non-channel trailing axes are
unrolled into per-thread registers exactly as in T1, so this shares T1's scalar-expression model
and differs only in that the channel coordinate is the symbolic `c` rather than a literal.

Correctness-first (Phase 2 discipline): sums over the contracted channel axis are emitted fully
expanded, so the arithmetic order matches the interpreter's `einsum` reduction order closely
enough to be checked, and nothing depends on loop-carried SSA in the DSL. Vectorized loads,
multi-edge CTAs and MMA atoms are performance variants that come after bit-exactness.
"""

from __future__ import annotations

import itertools
import textwrap

from codegen.emit import DTYPE, GENERATED_DIR, build_kernel
from codegen.tile import CH, Ch, TileSchedule
from zippel.ir import IndexType, Program


def _ch_expr(i: "Ch") -> str:
    """The channel coordinate this thread reads: `c`, or `c + k` / `c - k`."""
    if i.offset == 0:
        return "c"
    return f"c + {i.offset}" if i.offset > 0 else f"c - {-i.offset}"


def _sym(buf: str, idx: tuple) -> str:
    """Register name. A value is named by its *non-channel* index; the channel is this thread's.

    Two factors differing only in channel offset are different memory reads, not different
    registers, so the offset appears in `_ref`/`_factor` and never in a register name.
    """
    return f"v_{buf}" + "".join(("_c" if isinstance(i, Ch) else f"_{i}") for i in idx)


def _ref(prog: Program, buf: str, idx: tuple) -> str:
    t = prog.type_of(buf)
    lead = "e" if t.segment != "none" else "0"
    coords = [lead] + [(_ch_expr(i) if isinstance(i, Ch) else str(i)) for i in idx]
    return f"m_{buf}[{', '.join(coords)}]"


def _factor(prog: Program, buf: str, idx: tuple, from_smem: bool, live_in: set) -> str:
    if from_smem:
        # a cross-channel read has a literal channel coordinate; that is the smem index
        chan = [i for i in idx if not isinstance(i, Ch)]
        return f"s_{buf}[{', '.join(str(i) for i in chan)}]"
    if buf in live_in:
        # Inlined, not hoisted. A hoisted load sits outside the predicate that makes its index
        # valid: `xedge_7` concatenates three operands, and thread c<64 reading `emb_src[e,c-64]`
        # is an out-of-bounds negative index. Inlining puts every load inside the branch that
        # guarantees its range; NVRTC re-materialises common subexpressions within a block.
        return _ref(prog, buf, idx)
    return _sym(buf, idx)


def emit_tile_source(prog: Program, sched: TileSchedule, dtype: str = "f32") -> str:
    spec = sched.spec
    dt = DTYPE[dtype]
    C = sched.extent
    depth = max((len(a.terms) for a in sched.assigns), default=1)

    tensors = [b for b in spec.live_in if not isinstance(prog.type_of(b), IndexType)]
    tensors += list(spec.live_out)
    params = ", ".join(f"m_{b}: cute.Tensor" for b in tensors)

    live_in = set(spec.live_in)
    body: list[str] = []

    def rhs_lines(a, uid: int) -> list[str]:
        target = _sym(a.target, a.index)
        if a.fn is not None:
            src = a.source
            arg = _ref(prog, *src) if src[0] in live_in else _sym(*src)
            return [f"{target} = {_scalar(a.fn, arg, dt)}"]
        if not a.terms:
            return [f"{target} = {dt}(0.0)"]
        parts = []
        for t in a.terms:
            fs = " * ".join(_factor(prog, b, ix, sm, live_in) for b, ix, sm in t.factors)
            parts.append(fs if t.coeff == 1.0 else
                         (f"-({fs})" if t.coeff == -1.0 else f"{dt}({t.coeff!r}) * {fs}"))
        return _chunked_sum(target, parts, uid)

    # Group by the value being produced. Several disjoint channel ranges writing one value is a
    # concatenation, and must become ONE if/elif chain: separate `if` blocks leave the register
    # undefined on the paths that do not run, and would emit one store per branch.
    order_keys: list[tuple] = []
    by_value: dict[tuple, list] = {}
    for a in sched.assigns:
        key = (a.target, a.index)
        if key not in by_value:
            by_value[key] = []
            order_keys.append(key)
        by_value[key].append(a)

    emitted = 0
    for key in order_keys:
        group = by_value[key]
        idx_in_sched = sched.assigns.index(group[0])
        for buf in sched.stage_before.get(idx_in_sched, []):
            body.append(f"s_{buf}[c] = {_sym(buf, (CH,))}")
            body.append("cute.arch.barrier()")

        ranged = [a for a in group if a.ch_range is not None]
        if not ranged:
            for a in group:
                body.extend(rhs_lines(a, emitted)); emitted += 1
            continue

        ranged.sort(key=lambda a: a.ch_range[0])
        covered = ranged[0].ch_range[0] == 0 and ranged[-1].ch_range[1] == C and all(
            ranged[k].ch_range[1] == ranged[k + 1].ch_range[0] for k in range(len(ranged) - 1))
        # CuTe DSL requires a value to exist before dynamic control flow assigns it:
        # "Using variables defined in dynamic control flow is not supported. Please give an
        # initial value before control flow." The chain below is exhaustive when `covered`, so
        # this initialiser is never the value that survives -- it exists to satisfy the tracer.
        body.append(f"{_sym(key[0], key[1])} = {dt}(0.0)")
        for k, a in enumerate(ranged):
            lo, hi = a.ch_range
            kw = "if" if k == 0 else "elif"
            cond = f"c < {hi}" if k == 0 and lo == 0 else f"c >= {lo} and c < {hi}"
            body.append(f"{kw} {cond}:")
            body.extend("    " + ln for ln in rhs_lines(a, emitted))
            emitted += 1
        # When the ranges do not tile the axis, the initialiser above is the value uncovered
        # threads keep -- which is correct, since no path writes them.

    # Stores are unconditional: every thread owns exactly one element of each output, and the
    # if/elif chain above has already given that element a value on every path.
    for buf in spec.live_out:
        t = prog.type_of(buf)
        ranges = [(CH,) if k == sched.axis else range(s) for k, s in enumerate(t.sizes)]
        for idx in itertools.product(*ranges):
            idx = tuple(idx)
            body.append(f"{_ref(prog, buf, idx)} = {_sym(buf, idx)}")

    smem_decls = "\n".join(
        f"        s_{b} = smem.allocate_tensor({dt}, cute.make_layout({C}), 16)"
        for b in sched.staged)
    indented = textwrap.indent("\n".join(body), " " * 12)

    return f'''"""Generated by codegen/emit_tile.py from fusion group {spec.name} (template T2).

{spec}
  channel axis {sched.axis} of extent {C} on threads; {sched.n_values} values,
  {sched.n_terms} terms. Staged through smem: {", ".join(sched.staged) or "(none)"}.
  Internal buffers never stored: {", ".join(spec.internal) or "(none)"}
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import Float32, Float64, Int32, const_expr
from cutlass.cute.runtime import from_dlpack

CHANNELS = {C}
TENSOR_ORDER = {tensors!r}

#: Correctness contract for this kernel (DECISIONS.md D25). A channel contraction cannot be
#: bit-exact against a blocked einsum, so the bar is the ordering bound the harness derives
#: from REDUCTION_DEPTH and the real input magnitudes.
#: Which segment axis this kernel iterates. The caller must pass that segment's length as
#: `n_seg`; passing another segment's length indexes past the end of every buffer. A node-rooted
#: group launched with the edge count segfaults, which is how this came to be declared.
SEGMENT = "{spec.segment}"
TEMPLATE = "T2"
REDUCTION_DEPTH = {depth}
EXACT = False


class Kernel:
    """One CTA per {spec.segment}, one thread per channel."""

    @cute.jit
    def __call__(self, {params}, n_seg: Int32, stream):
        self.kernel({", ".join(f"m_{b}" for b in tensors)}, n_seg).launch(
            grid=[n_seg, 1, 1], block=[CHANNELS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, {params}, n_seg: Int32):
        c, _, _ = cute.arch.thread_idx()
        e, _, _ = cute.arch.block_idx()
        smem = cutlass_utils.SmemAllocator()
{smem_decls}
        if e < n_seg:
{indented}
'''


#: Terms per emitted statement. A single `a + b + ... + z` of thousands of terms overflows
#: CPython's AST recursion limit while the DSL parses the generated module (the SO(2) conv group
#: has 5 132). Chunking left-to-right preserves the summation order exactly, so the ordering
#: bound is unaffected.
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


def _scalar(fn: str, arg: str, dt: str) -> str:
    from codegen.emit import _fn_expr
    return _fn_expr(fn, 0, arg, dt)


__all__ = ["emit_tile_source", "build_kernel", "GENERATED_DIR"]
