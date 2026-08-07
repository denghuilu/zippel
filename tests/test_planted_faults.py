"""Planted-fault battery: prove the bound *rejects* broken kernels, not just that it accepts good ones.

`tests/test_codegen.py` shows every emitted kernel landing inside the bound its emitter shipped.
That is only half a claim. A check that never fires is indistinguishable from a check that cannot
fire, and the D25 bound is the only thing standing between a compiled-and-running kernel and a
silently wrong number — `invar_101` was O(1) wrong, ran clean, and nothing else in the suite
noticed (`findings/compiled-ran-clean-wrong.md`).

So each test here deliberately breaks a kernel and asserts the failure is caught. The faults are
planted in the **schedule**, not in the generated text, so the real emitter produces them and
they are genuinely compilable: this exercises the same path a real bug would take.

Fault classes, chosen to match defects that actually occurred in this project:

| fault | real instance |
|---|---|
| dropped term | D21 — `operands.index(k)` halved a derivative by dropping occurrences |
| transposed index | the `_z_rot_mat` write-order bug, and the D/R convention error |
| off-by-one channel slice | `invar_101` / `conv2_95` — 4.47e+00 and 4.03e-01 |
| missing barrier | T2 staging: reads another thread's registers before they are written |
| wrong SEGMENT | node-rooted group launched with the edge count — segfault |

`SEGMENT` is caught at *load*, by metadata validation, and never reaches the bound. That is the
intended division: launch geometry is a contract violation, not a numerical one.

**What this battery certifies, and what it does not.** It certifies **emitter faithfulness given
a correct schedule**: that the CuTe DSL text produced from a schedule computes what that schedule
says, and that a discrepancy is detected. It does *not* certify that the schedule itself is the
right schedule. A fault introduced during schedule *construction* -- a mis-partitioned group, a
mislabelled segment, a wrong template choice -- is invisible here, because both the kernel and
the bound would be derived from the same wrong schedule and would agree with each other.

That layer is covered separately, by the structural contracts: the Kahn acyclicity guard on the
partition (`tests/test_ir_core.py`), the IR type checker, `assert_closed`, the register-budget
precondition, and `MetadataMismatch` at load. Those are exact, pre-run checks; this battery is
the post-run numerical one. Neither subsumes the other, and saying so is the point -- the
`invar_101` class lives in the emitter layer, and the 107-launch class lived in the structural
one.
"""

from __future__ import annotations

import copy

import pytest
import torch

from blocks.eso2_ir import build_forward
from blocks.eso2_ref import BlockConfig, ESO2RefBlock
from blocks.ir_bind import bind
from codegen.bounds import assert_within_bound, ordering_bound, term_magnitudes
from codegen.emit import MetadataMismatch, build_kernel, emit_source
from codegen.emit_tile import emit_tile_source
from codegen.schedule import analyze_group, build_schedule
from codegen.tile import Ch, build_tile_schedule, channel_axis
from fixtures.load import load_batch
from zippel.interp import run
from zippel.simplify import fusion_groups, simplify

DT, CFG = torch.float64, BlockConfig()

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(scope="module")
def forward():
    torch.manual_seed(0)
    jd = [j.to(DT) for j in torch.load("blocks/Jd.pt", weights_only=False)]
    block = ESO2RefBlock(CFG).to("cpu", DT)
    batch = load_batch("si_small", "cpu", DT, CFG)
    prog, _ = build_forward(CFG, gauss_coeff=block.gauss_coeff)
    simp = simplify(prog, keep=prog.outputs)
    inp, sizes = bind(block, batch, jd, CFG)
    return simp, run(simp, inp, sizes), sizes, fusion_groups(simp)


def _pick(groups, predicate):
    return next(g for g in groups if predicate(g))


def _run(simp, env, sizes, sched, source, name):
    """Compile and launch, returning the outputs. Raises whatever the pipeline raises."""
    import cutlass
    import cutlass.cute as cute
    from cutlass import Int32
    from cutlass.cute.runtime import from_dlpack

    Kernel, order = build_kernel(source, name, sched=sched)
    spec = sched.spec
    outs = {b: torch.zeros_like(env[b]) for b in spec.live_out}
    tensors = {b: env[b].contiguous() for b in order if b not in outs} | outs
    stream = cutlass.cuda.default_stream()
    args = tuple(from_dlpack(tensors[b], assumed_align=16) for b in order) + (
        Int32(sizes[spec.segment]), stream)
    cute.compile(Kernel(), *args)(*args)
    torch.cuda.synchronize()
    return outs


def _assert_fault_is_caught(simp, env, sizes, pristine, faulted, source, name):
    """Run the faulted kernel, check it against the bound of the *intended* schedule.

    The bound comes from `pristine`, not from `faulted`, and that is the faithful arrangement:
    every defect of this class in the real project lived in the **emitter** while the schedule
    was correct. `invar_101` had the right channel offsets in its schedule and an emitter that
    ignored them, so the bound was computed for the kernel we meant to write and the kernel we
    actually wrote missed it by 4.47e+00. Deriving the bound from the faulted schedule instead
    would let a fault quietly widen its own tolerance.
    """
    outs = _run(simp, env, sizes, faulted, source, name)
    bound = ordering_bound(pristine, env)
    with pytest.raises(AssertionError):
        for buf in pristine.spec.live_out:
            assert_within_bound(name, outs[buf], env[buf], bound)


# ------------------------------------------------------------------------------------------
# T1 faults
# ------------------------------------------------------------------------------------------


@cuda
def test_dropped_term_is_caught(forward):
    """D21's failure mode: a term silently missing from a sum."""
    simp, env, sizes, groups = forward
    spec = analyze_group(simp, _pick(groups, lambda g: any(n.startswith("rot_") for n in g)),
                         name="wigner_chain")
    pristine = build_schedule(simp, spec)
    sched = copy.deepcopy(pristine)

    # Pick the largest contribution anywhere in the schedule. Choosing the *first* multi-term
    # assignment instead plants the fault in `-sin(0.gamma)`, which is exactly zero, and the
    # test then passes while proving nothing.
    _, ai, ti = next((m, a, t) for m, a, t in term_magnitudes(pristine, env)
                     if len(sched.assigns[a].terms) > 1)
    victim = sched.assigns[ai]
    victim.terms = victim.terms[:ti] + victim.terms[ti + 1:]

    _assert_fault_is_caught(simp, env, sizes, pristine, sched,
                            emit_source(simp, sched, dtype="f64"), "fault_dropped_term")


@cuda
def test_transposed_index_is_caught(forward):
    """The `_z_rot_mat` class: right values, wrong place."""
    simp, env, sizes, groups = forward
    spec = analyze_group(simp, _pick(groups, lambda g: any(n.startswith("rot_") for n in g)),
                         name="wigner_chain")
    pristine = build_schedule(simp, spec)
    sched = copy.deepcopy(pristine)

    # again by magnitude, and only where transposing is observable (a non-symmetric index)
    def transposable(term):
        return any(len(f[1]) == 2 and f[1][0] != f[1][1] for f in term.factors)

    _, ai, ti = next((m, a, t) for m, a, t in term_magnitudes(pristine, env)
                     if transposable(sched.assigns[a].terms[t]))
    victim, term = sched.assigns[ai], sched.assigns[ai].terms[ti]
    factors = tuple((b, (i[1], i[0])) if len(i) == 2 and i[0] != i[1] else (b, i)
                    for b, i in term.factors)
    victim.terms = (victim.terms[:ti] + (type(term)(coeff=term.coeff, factors=factors),)
                    + victim.terms[ti + 1:])

    _assert_fault_is_caught(simp, env, sizes, pristine, sched,
                            emit_source(simp, sched, dtype="f64"), "fault_transposed_index")


# ------------------------------------------------------------------------------------------
# T2 faults
# ------------------------------------------------------------------------------------------


@cuda
def test_off_by_one_channel_slice_is_caught(forward):
    """The `invar_101` class: a channel-slice offset wrong by one.

    This is the fault that actually shipped and measured 4.47e+00. It is worth planting rather
    than only remembering, because it is invisible to compilation and to every structural test.
    """
    simp, env, sizes, groups = forward
    group = _pick(groups, lambda g: g == ["invar_101"])
    spec = analyze_group(simp, group, name="invar")
    pristine = build_tile_schedule(simp, spec, *channel_axis(simp, spec))
    sched = copy.deepcopy(pristine)

    # Reproduce the original defect exactly: the channel offset is dropped, so a path that
    # should read `c - 128` reads `c`. That is what the pre-Ch(offset) emitter did.
    def has_offset(term):
        return any(isinstance(i, Ch) and i.offset != 0 for f in term.factors for i in f[1])

    ai, ti = next((a, t) for _, a, t in term_magnitudes(pristine, env)
                  if has_offset(sched.assigns[a].terms[t]))
    victim, term = sched.assigns[ai], sched.assigns[ai].terms[ti]
    factors = tuple((b, tuple(Ch(0) if isinstance(i, Ch) else i for i in idx), sm)
                    for b, idx, sm in term.factors)
    victim.terms = (victim.terms[:ti] + (type(term)(coeff=term.coeff, factors=factors),)
                    + victim.terms[ti + 1:])

    _assert_fault_is_caught(simp, env, sizes, pristine, sched,
                            emit_tile_source(simp, sched, dtype="f64"), "fault_channel_slice")


@cuda
def test_missing_barrier_is_caught(forward):
    """A T2 group reading another thread's registers before they reach shared memory.

    Removing the staging point does not remove the smem *reads* -- the kernel then reads whatever
    the allocator left there. This is the one fault whose result is not deterministic, so the
    test asserts only that it is rejected, never what it computes.
    """
    simp, env, sizes, groups = forward
    group = _pick(groups, lambda g: "rs0_16" in g and "rl1_17" in g)
    spec = analyze_group(simp, group, name="radial_stage2")
    pristine = build_tile_schedule(simp, spec, *channel_axis(simp, spec))
    sched = copy.deepcopy(pristine)
    assert pristine.stage_before, "this group is supposed to stage a value through smem"

    sched.stage_before = {}                                # the barrier and the store with it

    _assert_fault_is_caught(simp, env, sizes, pristine, sched,
                            emit_tile_source(simp, sched, dtype="f64"), "fault_missing_barrier")


# ------------------------------------------------------------------------------------------
# contract faults -- caught at load, before anything runs
# ------------------------------------------------------------------------------------------


def test_wrong_segment_is_caught_at_load(forward):
    """Launch geometry is a contract violation, not a numerical one, and is rejected earlier."""
    simp, env, sizes, groups = forward
    spec = analyze_group(simp, _pick(groups, lambda g: g == ["invar_101"]), name="invar")
    sched = build_tile_schedule(simp, spec, *channel_axis(simp, spec))
    assert spec.segment == "node", "invar_101 is node-rooted; the fault below depends on it"

    source = emit_tile_source(simp, sched, dtype="f64").replace(
        'SEGMENT = "node"', 'SEGMENT = "edge"')
    with pytest.raises(MetadataMismatch, match="SEGMENT"):
        build_kernel(source, "fault_wrong_segment", sched=sched)


def test_wrong_reduction_depth_is_caught_at_load(forward):
    """A bound computed for a different kernel is not a bound."""
    simp, env, sizes, groups = forward
    spec = analyze_group(simp, _pick(groups, lambda g: any(n.startswith("rot_") for n in g)),
                         name="wigner_chain")
    sched = build_schedule(simp, spec)

    source = emit_source(simp, sched, dtype="f64")
    depth = max(len(a.terms) for a in sched.assigns)
    source = source.replace(f"REDUCTION_DEPTH = {depth}", f"REDUCTION_DEPTH = {depth + 7}")
    with pytest.raises(MetadataMismatch, match="REDUCTION_DEPTH"):
        build_kernel(source, "fault_wrong_depth", sched=sched)
