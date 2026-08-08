"""T3 — reduction / segment-rooted kernels: gather, scatter-add, intra-feature reduction.

T1 and T2 both require the group's ops to share one segment axis with no index maps, because both
map one loop nest over that axis onto the grid. T3 is what happens when that assumption breaks,
and the forward needs it in three shapes:

  gather                an operand is read at `index_map[e]` instead of `e` -- `pos[src]`,
                        `x_node[dst]`. Structurally this is the smallest change of the three: the
                        segment coordinate of a live-in read becomes an indirection.
  scatter-add           the result accumulates into `out[out_index_map[e]]` -- the edge→node
                        message sum, and the readout's reduction to a graph scalar. Several
                        segment elements target the same output, so the store is an atomic add
                        and the output must be zeroed first.
  intra-feature         the output's channel axis is rank-0 or extent 1, so the contraction sums
  reduction             *within* one segment element -- LayerNorm's mean and variance, and the
                        final 128→1 readout linear. One thread owns one segment element and sums
                        sequentially.

They share a template because they share a parallel axis (the **output** segment) and a
register discipline: live-in reads are inlined at use rather than hoisted, so a 128-term
reduction costs O(1) registers instead of O(128). That is also why the reduction shape does not
need a warp-cooperative sum to be correct -- it is the performance variant, not the first one.

Correctness first, per the Phase 2 discipline established at S1a/S1b.
"""

from __future__ import annotations

import itertools
import textwrap

from codegen.bounds import inlined_live_upper_bound
from codegen.emit import GENERATED_DIR, build_kernel, emitter_sha
from codegen.emit_common import (CHUNK, DTYPE, REGISTER_BUDGET, chunked_sum,
                                 metadata_block, ref, sym)
from codegen.schedule import Schedule, all_indices
from zippel.ir import IndexType, Program

#: Terms per emitted statement (see codegen/emit.py). Left-to-right, so summation order is
#: unchanged and the ordering bound still applies.
T3_DRIVING_NOTE = ("#: The segment the grid iterates. Differs from SEGMENT exactly when this "
                   "kernel scatters.")


def _sym(buf: str, idx: tuple) -> str:
    return sym(buf, idx)


def gather_maps(prog: Program, spec) -> dict[str, str]:
    """`{live-in buffer -> index buffer}`, for reporting only.

    **Lossy by construction, and must not drive codegen.** One buffer can be read through several
    index maps in a single op -- `evec_0`'s inputs are `(pos, pos, shifts)` with maps
    `(dst, src, None)` -- so this dict keeps whichever it saw last. The emitter uses the
    schedule's *per-factor* gathers instead (`Term.gathers`); this is kept for kernel headers and
    diagnostics, where "which buffers are gathered at all" is the question being asked.
    """
    maps: dict[str, str] = {}
    for name in spec.ops:
        op = prog.ops[name]
        for k, src in enumerate(op.inputs):
            if k < len(op.index_maps) and op.index_maps[k] is not None:
                maps[src] = op.index_maps[k]
    return maps


def index_buffers(prog: Program, spec) -> list[str]:
    """Every index buffer the group needs, from the ops themselves.

    Derived from `op.index_maps` rather than from `gather_maps`, because that dict collapses a
    buffer read through two maps into one entry -- which silently dropped `dst` from `evec_0`'s
    parameter list while the kernel body still referenced it.
    """
    found: set[str] = set()
    for name in spec.ops:
        op = prog.ops[name]
        found.update(m for m in op.index_maps if m is not None)
        if op.out_index_map is not None:
            found.add(op.out_index_map)
    return sorted(found)


def scatter_map(prog: Program, spec) -> str | None:
    """The index buffer this group scatters its result through, if any."""
    for name in spec.ops:
        imap = prog.ops[name].out_index_map
        if imap is not None:
            return imap
    return None


def _ref(prog: Program, buf: str, idx: tuple, gather: str | None = None) -> str:
    """A memory reference. `gather` is *this read's* index map, not the buffer's."""
    t = prog.type_of(buf)
    if t.segment == "none":
        lead = "0"
    elif gather is not None:
        lead = f"m_{gather}[e]"                  # the gather: one indirection, nothing more
    else:
        lead = "e"
    return ref(prog, buf, idx, lead=lead)


def _chunked_sum(target: str, parts: list[str], uid: int) -> list[str]:
    return chunked_sum(target, parts, uid)


def _store_line(prog: Program, buf: str, idx: tuple, value: str, scatter: str | None,
                dt: str) -> str:
    """One output store: plain, or an atomic accumulate when the group scatters."""
    if scatter is None:
        return f"{_ref(prog, buf, idx)} = {value}"
    # Several segment elements target the same output row, so the store accumulates. The flat
    # offset comes from the buffer's own layout: scatter target row, plus this element's offset.
    coords = ", ".join([f"m_{scatter}[e]"] + [str(i) for i in idx])
    return f"cute.arch.atomic_add(m_{buf}.iterator + m_{buf}.layout(({coords})), {value})"


def emit_reduce_source(prog: Program, sched: Schedule, block: int = 128,
                       dtype: str = "f32", budget: int = REGISTER_BUDGET) -> str:
    """Emit a T3 kernel: one thread per segment element, gathers inlined, stores atomic."""
    spec = sched.spec
    dt = DTYPE[dtype]
    _esha = emitter_sha()

    # Same precondition as T2, same reason. T3 inlines its live-ins too, so T1's
    # `peak_live_values` is the wrong bound here -- it counts hoisted loads and would refuse
    # `cat_83`, which reads 4 608 live-in elements and holds almost none of them.
    live = inlined_live_upper_bound(sched, interleaved_stores=True)
    if live > budget:
        raise ValueError(
            f"group {spec.name} needs up to {live} live scalars per thread under T3, over the "
            f"{budget} register budget -- it would spill to local memory.")
    gathers = gather_maps(prog, spec)
    scatter = scatter_map(prog, spec)

    # A scattering group's OUTPUT segment is not the segment it iterates. `scatter_100` produces
    # node[m,c] by accumulating over edges, so SEGMENT (what it writes) is "node" while the grid
    # must cover edges. Both are metadata: a caller that guesses gets a kernel that runs the
    # wrong number of times, silently, and for `E_105` -- graph[] scattered from nodes -- would
    # launch once.
    driving = spec.segment
    if scatter is not None:
        for b in spec.live_in:
            t = prog.type_of(b)
            if not isinstance(t, IndexType) and t.segment not in ("none", spec.segment):
                driving = t.segment
                break
        else:
            driving = prog.type_of(scatter).segment

    value_ins = [b for b in spec.live_in if not isinstance(prog.type_of(b), IndexType)]
    index_ins = index_buffers(prog, spec)
    tensors = value_ins + index_ins + list(spec.live_out)
    params = ", ".join(f"m_{b}: cute.Tensor" for b in tensors)
    live_in = set(spec.live_in)

    # Stores are interleaved with the assignments that produce them, not appended after all of
    # them. Emitting every assignment first makes every produced value live at once: a T3 thread
    # writing a [9,256] output held 2 304 scalars and spilled to local memory. A value that is a
    # live-out and is never read again can be stored the instant it exists, and then dies.
    later_use: dict[tuple, int] = {}
    for i, a in enumerate(sched.assigns):
        for t in a.terms:
            for f in t.factors:
                later_use[(f[0], f[1])] = i
        if a.source is not None:
            later_use[a.source] = i

    out_bufs = set(spec.live_out)
    stored: set = set()
    body: list[str] = []
    for i, a in enumerate(sched.assigns):
        target = _sym(a.target, a.index)
        if a.fn is not None:
            src = a.source
            arg = (_ref(prog, src[0], src[1], gathers.get(src[0]))
                   if src[0] in live_in else _sym(*src))
            from codegen.emit import _fn_expr
            body.append(f"{target} = {_fn_expr(a.fn, a.order, arg, dt)}")
        elif not a.terms:
            body.append(f"{target} = {dt}(0.0)")
        else:
            parts = []
            for term in a.terms:
                gs = term.gathers or (None,) * len(term.factors)
                factors = " * ".join(
                    _ref(prog, b, ix, g) if b in live_in else _sym(b, ix)
                    for (b, ix), g in zip(term.factors, gs))
                parts.append(factors if term.coeff == 1.0 else
                             (f"-({factors})" if term.coeff == -1.0
                              else f"{dt}({term.coeff!r}) * {factors}"))
            body.extend(_chunked_sum(target, parts, i))

        key = (a.target, a.index)
        if a.target in out_bufs and later_use.get(key, -1) <= i:
            body.append(_store_line(prog, a.target, a.index, target, scatter, dt))
            stored.add(key)

    # anything not stored inline: structural zeros, and values still read later
    for buf in spec.live_out:
        t = prog.type_of(buf)
        for idx in sorted(all_indices(t.sizes)):
            if (buf, idx) in stored:
                continue
            value = (_sym(buf, idx) if idx in sched.masks.get(buf, set()) else f"{dt}(0.0)")
            body.append(_store_line(prog, buf, idx, value, scatter, dt))

    indented = textwrap.indent("\n".join(body), " " * 12)
    _meta = metadata_block(
        spec.segment, "T3", _esha,
        max((len(a.terms) for a in sched.assigns), default=1), False,
        after_segment=T3_DRIVING_NOTE + f'\nDRIVING_SEGMENT = "{driving}"')

    return f'''"""Generated by codegen/emit_reduce.py from fusion group {spec.name} (template T3).

{spec}
  one thread per {driving}; writes {spec.segment}; gathers {gathers or "(none)"};
  scatter {scatter or "(none)"}.
  {sched.n_values} values, {sched.n_terms} terms.
  Internal buffers never stored: {", ".join(spec.internal) or "(none)"}
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Float64, Int32, const_expr
from cutlass.cute.runtime import from_dlpack

BLOCK = {block}
TENSOR_ORDER = {tensors!r}

{_meta}
SCATTERS = {scatter is not None!r}


class Kernel:
    """T3: parallel over the driving segment; the output segment is reached by index map."""

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


__all__ = ["emit_reduce_source", "gather_maps", "index_buffers", "scatter_map",
           "build_kernel", "GENERATED_DIR"]
