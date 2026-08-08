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

from codegen.bounds import inlined_live_upper_bound
from codegen.emit import GENERATED_DIR, build_kernel, emitter_sha
from codegen.emit_common import (CHUNK, DTYPE, REGISTER_BUDGET, chunked_sum,
                                metadata_block, ref, sym)
from codegen.tile import CH, Ch, TileSchedule
from zippel.ir import IndexType, Program


def _ch_expr(i: "Ch") -> str:
    """The channel coordinate this thread reads: `c`, or `c + k` / `c - k`."""
    if i.offset == 0:
        return "c"
    return f"c + {i.offset}" if i.offset > 0 else f"c - {-i.offset}"


def _name_part(i) -> str:
    """T2's register-name hook: the channel is this thread's, so it names itself `c`."""
    return "c" if isinstance(i, Ch) else str(i)


def _index_part(i) -> str:
    """T2's reference hook: a channel component resolves to this thread's coordinate."""
    return _ch_expr(i) if isinstance(i, Ch) else str(i)


def _sym(buf: str, idx: tuple) -> str:
    """Register name. A value is named by its *non-channel* index; the channel is this thread's.

    Two factors differing only in channel offset are different memory reads, not different
    registers, so the offset appears in `_ref`/`_factor` and never in a register name.
    """
    return sym(buf, idx, render=_name_part)


def _ref(prog: Program, buf: str, idx: tuple, transpose: dict | None = None) -> str:
    """A memory reference. `transpose[buf]` permutes the trailing axes of the reference.

    The permutation is a *layout* change, applied identically to the emitted index order and to
    the tensor handed in at launch, so the kernel computes the same thing from the same values.
    Its point is stride: with the thread-mapped axis buried, consecutive threads read addresses
    2 048 B apart and every warp load touches 32 cache lines (D42). Moving that axis last makes
    the same reads contiguous.
    """
    return ref(prog, buf, idx, render=_index_part, perm=(transpose or {}).get(buf))


#: Shared memory is 32 banks of 4 B. A stride that is a multiple of 32 words maps every lane of a
#: warp onto one bank.
SMEM_BANKS = 32
#: Largest dynamic shared-memory allocation per block. Hopper's architectural ceiling is 227 KiB;
#: 224 KiB is what was probed to compile, launch and round-trip correctly on this GH200.
SMEM_CAP_BYTES = 224 * 1024


def staged_layout(prog: Program, sched: TileSchedule, buf: str) -> dict:
    """Padded shared-memory layout for a staged operand. **The padding is not an optimisation.**

    A staged operand is laid out slab-major: thread `o` owns the `S` elements of its own slab at
    `sh[o * T + rest]`. Take the direct copy, `T = S`. Every weight here has `S` a multiple of 32
    (256 or 512), so lane `o` at step `rest` lands on bank `(o*S + rest) % 32 == rest % 32` -- the
    *same* bank for all 32 lanes of the warp. That is a **32-way bank conflict**, serialising the
    read into 32 phases: numerically the same factor of 32 as the uncoalesced global read arm B
    exists to remove, merely moved from HBM into smem. Under it, any B reading -- null, weak, or
    otherwise -- would be uninterpretable, because B would be paying the very cost it is testing.

    So `T = S + 1` whenever `S` is even, making `T` odd. Odd numbers are units mod 32, so
    `o -> o*T mod 32` is a bijection on the 32 lanes and every lane hits a distinct bank. For
    64-bit elements the access splits into two 16-lane phases and `2*T mod 32` with `T` odd is
    likewise a bijection of 16 lanes onto the 16 even banks. One rule covers both widths, which
    is why padding is preferred here over a swizzle: a swizzle would need a width-dependent
    XOR schedule to say the same thing.

    Costs one extra element per thread -- 128 elements of 32 768, 0.4 %.
    """
    t = prog.type_of(buf)
    p = _thread_axis(sched, buf)
    sizes = list(t.sizes)
    others = [k for k in range(len(sizes)) if k != p]
    S = 1
    for k in others:
        S *= sizes[k]
    T = S + 1 if S % 2 == 0 else S
    # row-major strides *within a slab*, in the original axis order
    slab_stride = {}
    acc = 1
    for k in reversed(others):
        slab_stride[k] = acc
        acc *= sizes[k]
    return {"axis": p, "sizes": sizes, "others": others, "S": S, "T": T,
            "slab_stride": slab_stride, "extent": sizes[p]}


def required_transpose(prog: Program, sched: TileSchedule) -> dict[str, tuple]:
    """T2's layout requirement: the thread-mapped axis must be innermost in every operand (D54).

    T2 puts one channel on each thread. An operand whose thread-mapped axis is *not* its innermost
    is read by consecutive threads at a stride of its trailing extents -- 2 048 B for `c1_w1a` --
    so a warp load touches 32 distinct cache lines where a coalesced one touches 4 sectors.
    Permuting that axis last makes the same reads contiguous. **Measured 1.228x on `conv1_90`**
    (D53), bit-exact: a permutation changes neither the values nor the order they are summed in.

    Applies to **10 of the 24 T2 groups** in the forward program. Every operand it names is a
    *program input* -- a weight -- so the permutation is applied once when the environment is
    allocated and costs nothing per launch.

    **Scoped to program inputs on purpose.** A *produced* buffer with the same defect would need
    its producing kernel to write the permuted layout, which is a different and larger change; it
    is left alone rather than silently mishandled. No such buffer exists in the forward program
    today, so the restriction costs nothing measurable -- but it will not fail quietly if one
    appears, because `compose.transpose_inputs` raises on anything it is asked to permute that it
    did not allocate.

    Published by the generated module as `TRANSPOSE` so the launch side reads it back rather than
    recomputing it -- the D52 bug, which cost an illegal memory access.
    """
    out: dict[str, tuple] = {}
    for b in sched.spec.live_in:
        if b not in prog.inputs:
            continue
        t = prog.type_of(b)
        sizes = getattr(t, "sizes", ())
        if len(sizes) < 2:
            continue
        p = _thread_axis_or_none(sched, b)
        if p is None or p == len(sizes) - 1:
            continue
        out[b] = tuple([k for k in range(len(sizes)) if k != p] + [p])
    return out


def _thread_axis_or_none(sched: TileSchedule, buf: str) -> int | None:
    for a in sched.assigns:
        for t in a.terms:
            for f in t.factors:
                if f[0] == buf:
                    for k, i in enumerate(f[1]):
                        if isinstance(i, Ch):
                            return k
    return None


def _thread_axis(sched: TileSchedule, buf: str) -> int:
    for a in sched.assigns:
        for t in a.terms:
            for f in t.factors:
                if f[0] == buf:
                    for k, i in enumerate(f[1]):
                        if isinstance(i, Ch):
                            return k
    raise ValueError(f"{buf} is not indexed by the channel and cannot be staged slab-major")


def _staged_ref(buf: str, idx: tuple, lay: dict) -> str:
    """This thread's element of a staged operand: `sh[(c+off) * T + rest]`."""
    slab = _ch_expr(idx[lay["axis"]])
    rest = sum(int(idx[k]) * lay["slab_stride"][k] for k in lay["others"])
    off = f" + {rest}" if rest else ""
    return f"sh_{buf}[({slab}) * {lay['T']}{off}]"


def _factor(prog: Program, buf: str, idx: tuple, from_smem: bool, live_in: set,
            transpose: dict | None = None, staged: dict | None = None) -> str:
    if staged and buf in staged:
        return _staged_ref(buf, idx, staged[buf])
    if from_smem:
        # a cross-channel read has a literal channel coordinate; that is the smem index
        chan = [i for i in idx if not isinstance(i, Ch)]
        return f"s_{buf}[{', '.join(str(i) for i in chan)}]"
    if buf in live_in:
        # Inlined, not hoisted. A hoisted load sits outside the predicate that makes its index
        # valid: `xedge_7` concatenates three operands, and thread c<64 reading `emb_src[e,c-64]`
        # is an out-of-bounds negative index. Inlining puts every load inside the branch that
        # guarantees its range; NVRTC re-materialises common subexpressions within a block.
        return _ref(prog, buf, idx, transpose)
    return _sym(buf, idx)


def emit_tile_source(prog: Program, sched: TileSchedule, dtype: str = "f32",
                     budget: int = REGISTER_BUDGET,
                     transpose: dict | None = None,
                     stage_shared: tuple = ()) -> str:
    spec = sched.spec
    dt = DTYPE[dtype]
    C = sched.extent
    _esha = emitter_sha()
    depth = max((len(a.terms) for a in sched.assigns), default=1)

    # Register precondition. T1 has had one since S1a; T2 was written without, so a group that
    # spilled would have done so silently -- luck, not a guard. The bound is the inlined-load
    # form (codegen/bounds.py), an upper bound by construction per D26.
    live = inlined_live_upper_bound(sched)
    if live > budget:
        raise ValueError(
            f"group {spec.name} needs up to {live} live scalars per thread under T2, over the "
            f"{budget} register budget -- it would spill to local memory. Split the group, or "
            f"stage its operands rather than holding them.")

    tensors = [b for b in spec.live_in if not isinstance(prog.type_of(b), IndexType)]
    tensors += list(spec.live_out)
    params = ", ".join(f"m_{b}: cute.Tensor" for b in tensors)

    # `transpose=None` means "apply T2's layout requirement", which is the default and the
    # ratified rule. An explicit dict overrides it -- including `{}` for a deliberately unfixed
    # kernel, which is how the factorial's baseline arm is built. `or {}` would have conflated
    # those two, so the sentinel is `is None` and not falsiness.
    if transpose is None:
        transpose = required_transpose(prog, sched)
    # Staging subsumes transposition: every read of a staged operand goes through smem, and the
    # cooperative load reads the tensor in its original layout. Permuting it as well would be a
    # no-op on the arithmetic and a second, invisible difference between arms.
    staged = {b: staged_layout(prog, sched, b) for b in stage_shared}
    transpose = {b: p for b, p in transpose.items() if b not in staged}
    itemsize = 8 if dtype == "f64" else 4
    smem_bytes = sum(lay["extent"] * lay["T"] * itemsize for lay in staged.values())
    if smem_bytes > SMEM_CAP_BYTES:
        raise ValueError(
            f"staging {', '.join(staged)} needs {smem_bytes / 1024:.1f} KiB of shared memory, over "
            f"the {SMEM_CAP_BYTES / 1024:.0f} KiB per-block cap -- the launch would fail. Stage "
            f"fewer operands, or tile the staging.")

    live_in = set(spec.live_in)
    body: list[str] = []

    # Cooperative load, once per block, before anything reads it. Outer loop walks slabs, inner
    # loop walks the contiguous axis with consecutive lanes on consecutive elements: the global
    # read is coalesced (which is the entire point of the arm) and the smem write lands on
    # consecutive words, so the write is conflict-free as well as the read.
    for b, lay in staged.items():
        lead = [k for k in lay["others"][:-1]]
        last = lay["others"][-1] if lay["others"] else None
        body.append(f"o_{b} = Int32(0)")
        body.append(f"while o_{b} < {lay['extent']}:")
        for combo in itertools.product(*[range(lay["sizes"][k]) for k in lead]):
            base = sum(v * lay["slab_stride"][k] for k, v in zip(lead, combo))
            coords = {k: str(v) for k, v in zip(lead, combo)}
            coords[lay["axis"]] = f"o_{b}"
            coords[last] = f"r_{b}"
            src = ", ".join(["0"] + [coords[k] for k in range(len(lay["sizes"]))])
            dst = f"o_{b} * {lay['T']}" + (f" + {base}" if base else "") + f" + r_{b}"
            body.append(f"    r_{b} = c")
            body.append(f"    while r_{b} < {lay['sizes'][last]}:")
            body.append(f"        sh_{b}[{dst}] = m_{b}[{src}]")
            body.append(f"        r_{b} += {C}")
        body.append(f"    o_{b} += 1")
    if staged:
        body.append("cute.arch.barrier()")

    def rhs_lines(a, uid: int) -> list[str]:
        target = _sym(a.target, a.index)
        if a.fn is not None:
            src = a.source
            arg = (_ref(prog, src[0], src[1], transpose) if src[0] in live_in
                   else _sym(*src))
            return [f"{target} = {_scalar(a.fn, arg, dt)}"]
        if not a.terms:
            return [f"{target} = {dt}(0.0)"]
        parts = []
        for t in a.terms:
            fs = " * ".join(_factor(prog, b, ix, sm, live_in, transpose, staged)
                            for b, ix, sm in t.factors)
            parts.append(fs if t.coeff == 1.0 else
                         (f"-({fs})" if t.coeff == -1.0 else f"{dt}({t.coeff!r}) * {fs}"))
        return _chunked_sum(target, parts, uid)

    # Group by the value being produced. Several disjoint channel ranges writing one value is a
    # concatenation, and must become ONE if/elif chain: separate `if` blocks leave the register
    # undefined on the paths that do not run, and would emit one store per branch.
    # Carry each assignment's position, rather than searching for it later. `list.index()` on a
    # dataclass compares by VALUE, so two assignments agreeing on every field resolve to the
    # first -- and this index selects where a barrier is emitted. Same bug class as D21 and the
    # T3 gather map (findings/keyed-by-identity.md); found by the standing `.index(` audit that
    # finding instituted, which is the first thing that audit caught.
    order_keys: list[tuple] = []
    by_value: dict[tuple, list] = {}
    for pos, a in enumerate(sched.assigns):
        key = (a.target, a.index)
        if key not in by_value:
            by_value[key] = []
            order_keys.append(key)
        by_value[key].append((pos, a))

    emitted = 0
    for key in order_keys:
        group = by_value[key]
        idx_in_sched = group[0][0]
        for buf in sched.stage_before.get(idx_in_sched, []):
            body.append(f"s_{buf}[c] = {_sym(buf, (CH,))}")
            body.append("cute.arch.barrier()")

        ranged = [a for _pos, a in group if a.ch_range is not None]
        if not ranged:
            for _pos, a in group:
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
            body.append(f"{_ref(prog, buf, idx)} = {_sym(buf, idx)}")  # outputs never transposed

    smem_decls = "\n".join(
        [f"        s_{b} = smem.allocate_tensor({dt}, cute.make_layout({C}), 16)"
         for b in sched.staged]
        + [f"        sh_{b} = smem.allocate_tensor({dt}, "
           f"cute.make_layout({lay['extent'] * lay['T']}), 16)"
           for b, lay in staged.items()])
    indented = textwrap.indent("\n".join(body), " " * 12)

    _after_sha = "\n" + T2_TRANSPOSE_NOTE + (
        f"\nTRANSPOSE = {transpose!r}\nSTAGED = {list(staged)!r}")
    _meta = metadata_block(spec.segment, "T2", _esha, depth, False,
                           notes=T2_NOTES, after_sha=_after_sha)

    return f'''"""Generated by codegen/emit_tile.py from fusion group {spec.name} (template T2).

{spec}
  channel axis {sched.axis} of extent {C} on threads; {sched.n_values} values,
  {sched.n_terms} terms. Cross-channel values staged: {", ".join(sched.staged) or "(none)"}.
  Operands transposed: {", ".join(f"{b}{tuple(p)}" for b, p in transpose.items()) or "(none)"}
  Operands staged in smem: {", ".join(f"{b}[{staged[b]['extent']}x{staged[b]['T']}]" for b in staged) or "(none)"}
    ({smem_bytes / 1024:.1f} KiB, slab stride padded to an odd word count so the 32 lanes of a
     warp land on 32 distinct banks rather than one)
  Internal buffers never stored: {", ".join(spec.internal) or "(none)"}
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass import Float32, Float64, Int32, const_expr
from cutlass.cute.runtime import from_dlpack

CHANNELS = {C}
TENSOR_ORDER = {tensors!r}

{_meta}


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


def _chunked_sum(target: str, parts: list[str], uid: int) -> list[str]:
    return chunked_sum(target, parts, uid)


T2_TRANSPOSE_NOTE = """#: The permutation each operand's tensor MUST be handed in under, and the operands served from
#: shared memory instead. **The caller must read these back rather than recompute them.** They are
#: the emitter's decision, not the caller's request: staging subsumes transposition, so a buffer
#: asked for in both is emitted staged and *not* permuted. A caller that permuted its own copy of
#: the request would hand in a tensor whose axes disagree with the emitted index order -- which is
#: an illegal memory access, not a wrong answer, because the permuted extents no longer bound the
#: emitted coordinates. That is how this constant came to exist."""

T2_NOTES = """#: Correctness contract for this kernel (DECISIONS.md D25). A channel contraction cannot be
#: bit-exact against a blocked einsum, so the bar is the ordering bound the harness derives
#: from REDUCTION_DEPTH and the real input magnitudes.
#: Which segment axis this kernel iterates. The caller must pass that segment's length as
#: `n_seg`; passing another segment's length indexes past the end of every buffer. A node-rooted
#: group launched with the edge count segfaults, which is how this came to be declared."""


def _scalar(fn: str, arg: str, dt: str) -> str:
    from codegen.emit import _fn_expr
    return _fn_expr(fn, 0, arg, dt)


__all__ = ["emit_tile_source", "build_kernel", "GENERATED_DIR"]
