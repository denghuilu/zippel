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

    out = spec.live_out[0]
    ref = env[out]
    got = torch.zeros_like(ref)
    tensors = {b: env[b].contiguous() for b in order if b != out} | {out: got}
    stream = cutlass.cuda.default_stream()
    args = tuple(from_dlpack(tensors[b], assumed_align=16) for b in order) + (
        Int32(sizes["edge"]), stream)
    cute.compile(Kernel(), *args)(*args)
    torch.cuda.synchronize()

    assert torch.equal(got, ref), f"max abs {(got - ref).abs().max().item():.3e}"
