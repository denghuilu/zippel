"""Phase 2: the emitted CuTe DSL kernel must equal the FP64 interpreter exactly.

`rot_82` is the end of the Wigner chain -- five chained 9x9 matmuls whose six intermediates the
fused kernel never stores. Bit-exactness rather than a tolerance is the right bar here: the
emitter reorders nothing, so any difference is a codegen bug, not rounding.
"""

from __future__ import annotations

import pytest
import torch

from blocks.eso2_ir import build_forward
from blocks.eso2_ref import BlockConfig, ESO2RefBlock
from blocks.ir_bind import bind
from codegen.bounds import assert_within_bound, ordering_bound, reduction_depth
from codegen.emit import build_kernel, emit_source
from codegen.schedule import analyze_group, build_schedule, dense_term_count
from fixtures.load import load_batch
from zippel.interp import run
from zippel.simplify import fusion_groups, simplify

DT, CFG = torch.float64, BlockConfig()

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(scope="module")
def wigner_group():
    torch.manual_seed(0)
    jd = [j.to(DT) for j in torch.load("blocks/Jd.pt", weights_only=False)]
    block = ESO2RefBlock(CFG).to("cpu", DT)
    batch = load_batch("si_small", "cpu", DT, CFG)
    prog, _ = build_forward(CFG, gauss_coeff=block.gauss_coeff)
    simp = simplify(prog, keep=prog.outputs)
    inp, sizes = bind(block, batch, jd, CFG)
    env = run(simp, inp, sizes)
    # Select by content, never by position: the fusion partition is a compiler output and
    # reordering it must not silently point this test at a different kernel.
    groups = fusion_groups(simp)
    wanted = next(g for g in groups if any(n.startswith("rot_") for n in g))
    spec = analyze_group(simp, wanted, name="wigner_chain")
    return simp, spec, build_schedule(simp, spec), env, sizes


def test_sparsity_elides_the_block_diagonal_zeros(wigner_group):
    """The claim D22 rests on: the block-diagonal costs nothing in an unrolled schedule."""
    simp, spec, sched, _, _ = wigner_group
    dense = dense_term_count(simp, spec)
    assert sched.n_terms < 0.5 * dense, (
        f"only {1 - sched.n_terms / dense:.1%} of terms elided; the sparsity pass is not "
        f"finding the block structure ({sched.n_terms} of {dense})")
    assert sched.peak_live_values() <= 168


def test_emitted_kernel_is_deterministic(wigner_group):
    """Same IR in, byte-identical source out -- otherwise the artifact is not reviewable."""
    simp, _, sched, _, _ = wigner_group
    assert emit_source(simp, sched, dtype="f64") == emit_source(simp, sched, dtype="f64")


@cuda
def test_emitted_wigner_chain_matches_interpreter_exactly(wigner_group):
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32
    from cutlass.cute.runtime import from_dlpack

    simp, spec, sched, env, sizes = wigner_group
    Kernel, order = build_kernel(emit_source(simp, sched, dtype="f64"), "wigner_chain_f64")
    import sys as _sys
    mod = _sys.modules["zippel_generated.wigner_chain_f64"]

    out = spec.live_out[0]
    ref = env[out]
    got = torch.zeros_like(ref)
    tensors = {b: env[b].contiguous() for b in order if b != out} | {out: got}
    stream = cutlass.cuda.default_stream()
    args = tuple(from_dlpack(tensors[b], assumed_align=16) for b in order) + (
        Int32(sizes["edge"]), stream)
    cute.compile(Kernel(), *args)(*args)
    torch.cuda.synchronize()

    # Same mechanism as T2: the kernel shipped its own contract. T1 declares EXACT, so this
    # asserts bit-equality *and* the bound.
    assert mod.TEMPLATE == "T1" and mod.EXACT is True
    assert mod.REDUCTION_DEPTH == reduction_depth(sched)
    assert_within_bound("wigner_chain", got, ref, ordering_bound(sched, env), exact=mod.EXACT)


# ---------------------------------------------------------------------------------------
# T2 -- cooperative tile: channels on threads
# ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def forward_env():
    torch.manual_seed(0)
    jd = [j.to(DT) for j in torch.load("blocks/Jd.pt", weights_only=False)]
    block = ESO2RefBlock(CFG).to("cpu", DT)
    batch = load_batch("si_small", "cpu", DT, CFG)
    prog, _ = build_forward(CFG, gauss_coeff=block.gauss_coeff)
    simp = simplify(prog, keep=prog.outputs)
    inp, sizes = bind(block, batch, jd, CFG)
    return simp, run(simp, inp, sizes), sizes, fusion_groups(simp)


def _run_tile(simp, env, sizes, group, name):
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32
    from cutlass.cute.runtime import from_dlpack

    from codegen.emit_tile import emit_tile_source
    from codegen.tile import build_tile_schedule, channel_axis

    spec = analyze_group(simp, group, name=name)
    axis = channel_axis(simp, spec)
    assert axis is not None, f"{name} has no channel axis; it is not a T2 candidate"
    sched = build_tile_schedule(simp, spec, *axis)
    module_src = emit_tile_source(simp, sched, dtype="f64")
    Kernel, order = build_kernel(module_src, f"{name}_f64")
    import sys as _sys
    mod = _sys.modules[f"zippel_generated.{name}_f64"]
    meta = {"TEMPLATE": mod.TEMPLATE, "REDUCTION_DEPTH": mod.REDUCTION_DEPTH,
            "EXACT": mod.EXACT}

    outs = {b: torch.zeros_like(env[b]) for b in spec.live_out}
    tensors = {b: env[b].contiguous() for b in order if b not in outs} | outs
    stream = cutlass.cuda.default_stream()
    args = tuple(from_dlpack(tensors[b], assumed_align=16) for b in order) + (
        Int32(sizes["edge"]), stream)
    cute.compile(Kernel(), *args)(*args)
    torch.cuda.synchronize()
    return spec, sched, outs, meta


@cuda
def test_t2_channel_contraction_is_within_its_shipped_bound(forward_env):
    """A per-edge 128->128 Linear, channels on threads, contracted through gmem.

    The bound is not written here. The emitter shipped `REDUCTION_DEPTH` with the kernel, and
    `codegen.bounds` turns it into a number against the real inputs (D25).
    """
    simp, env, sizes, groups = forward_env
    group = next(g for g in groups if g == ["rl0_8"])
    spec, sched, outs, meta = _run_tile(simp, env, sizes, group, "radial_lin0")

    bound = ordering_bound(sched, env)
    err = assert_within_bound("radial_lin0", outs["rl0_8"], env["rl0_8"], bound,
                              exact=meta["EXACT"])
    assert meta["REDUCTION_DEPTH"] == reduction_depth(sched), \
        "the shipped depth no longer matches the schedule that produced it"
    assert meta["TEMPLATE"] == "T2" and meta["EXACT"] is False
    assert err > 0.0, "exactly equal to a blocked einsum would mean the test is not exercising it"


@cuda
def test_t2_stages_an_in_group_value_through_smem(forward_env):
    """The barrier path: `rs0_16` lives in one thread's registers and is read by all of them.

    `rl1_17` contracts over channels of `rs0_16`, which is produced *inside* the group, so its
    values sit in other threads' registers and must travel through shared memory. Getting the
    barrier wrong here does not perturb the result slightly -- it reads uninitialised smem.
    """
    simp, env, sizes, groups = forward_env
    group = next(g for g in groups if "rs0_16" in g and "rl1_17" in g)
    spec, sched, outs, meta = _run_tile(simp, env, sizes, group, "radial_stage2")

    assert "rs0_16" in sched.staged, "the in-group contracted value was not staged to smem"
    bound = ordering_bound(sched, env)
    assert_within_bound("radial_stage2", outs["rl1_17"], env["rl1_17"], bound,
                        exact=meta["EXACT"])


def test_t1_refuses_a_channel_group(forward_env):
    """The register budget is a precondition, not a suggestion (docs/templates.md 2)."""
    from codegen.schedule import build_schedule

    simp, env, sizes, groups = forward_env
    group = next(g for g in groups if g == ["rl0_8"])
    spec = analyze_group(simp, group, name="radial_lin0")
    with pytest.raises(ValueError, match="register budget"):
        emit_source(simp, build_schedule(simp, spec))
