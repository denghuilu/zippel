"""Validation ladder for the block expressed in the IR (work order section 4).

Tiny synthetic graphs plus ONE real fixture (si_small), per the runtime budget: FP64 oracles
are deliberately not run on the medium or large fixtures inside pytest.
"""

from __future__ import annotations

import pytest
import torch

from blocks.eso2_ir import build_dbwd, build_force, build_forward
from blocks.eso2_ref import BlockConfig, ESO2RefBlock, conservative_training_step
from blocks.ir_bind import bind
from tests.test_ref_block import make_batch, random_rotation
from zippel.interp import run
from zippel.simplify import simplify
from zippel.vjp import assert_closed

DT = torch.float64
CFG = BlockConfig()


def rel(a, b):
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


@pytest.fixture(scope="module")
def jd():
    return [j.to(DT) for j in torch.load("blocks/Jd.pt", weights_only=False)]


@pytest.fixture(scope="module")
def block():
    torch.manual_seed(0)
    return ESO2RefBlock(CFG).to("cpu", DT)


def dbwd_inputs(block, batch, jd, meta):
    inp, sizes = bind(block, batch, jd, CFG)
    n = batch["pos"].shape[0]
    inp |= {
        "seed_E": torch.ones(1, dtype=DT), "seed_L": torch.ones(1, dtype=DT),
        "e_ref": torch.zeros(1, dtype=DT), "f_ref": torch.zeros(n, 3, dtype=DT),
        "inv_nf": torch.full((1,), 1.0 / (3 * n), dtype=DT),
    }
    return inp, sizes


# -- 4.1 interpreter fwd == reference ------------------------------------------------------


def test_interpreter_forward_matches_reference_on_si_small(block, jd):
    """The one real-fixture oracle run. Work order tolerance: rel err <= 1e-12."""
    from fixtures.load import load_batch

    batch = load_batch("si_small", "cpu", DT, CFG)
    prog, meta = build_forward(CFG, gauss_coeff=block.gauss_coeff)
    inp, sizes = bind(block, batch, jd, CFG)
    got = run(prog, inp, sizes)[meta["energy"]].squeeze()
    want = block(batch["pos"], batch["atomic_numbers"], batch["x_node"], batch["edge_index"],
                 batch["shifts"], batch["cos_gamma_k"], batch["sin_gamma_k"], jd)
    assert abs(got.item() - want.item()) / abs(want.item()) < 1e-12


# -- 4.2 IR bwd == autograd ----------------------------------------------------------------


def test_ir_force_matches_autograd(block, jd):
    batch = make_batch("cpu", n=12, e=80, cfg=CFG)
    prog, meta = build_force(CFG, gauss_coeff=block.gauss_coeff)
    inp, sizes = bind(block, batch, jd, CFG)
    inp["seed_E"] = torch.ones(1, dtype=DT)
    env = run(prog, inp, sizes)

    pos = batch["pos"].detach().clone().requires_grad_(True)
    e = block(pos, batch["atomic_numbers"], batch["x_node"], batch["edge_index"],
              batch["shifts"], batch["cos_gamma_k"], batch["sin_gamma_k"], jd)
    (g,) = torch.autograd.grad(e, pos, create_graph=True)
    assert rel(env[meta["energy"]].squeeze(), e) < 1e-12
    assert rel(env[meta["force"]], -g) < 1e-11


@pytest.mark.parametrize("param", ["c1_w0", "ro_w1", "rad_w0", "c2_w0"])
def test_ir_parameter_gradients_match_autograd(block, jd, param):
    batch = make_batch("cpu", n=12, e=80, cfg=CFG)
    prog, meta = build_dbwd(CFG, gauss_coeff=block.gauss_coeff)
    inp, sizes = dbwd_inputs(block, batch, jd, meta)
    env = run(prog, inp, sizes)

    block.zero_grad(set_to_none=True)
    b2 = dict(batch)
    b2["pos"] = batch["pos"].detach().clone().requires_grad_(True)
    _, _, loss = conservative_training_step(block, b2, jd)
    loss.backward()

    ref = {"c1_w0": block.c1_m0.weight, "ro_w1": block.readout[2].weight,
           "rad_w0": block.rad_func[0].weight, "c2_w0": block.c2_m0.weight}[param]
    got = env[meta["grads"][param]].reshape(ref.grad.shape)
    assert rel(got, ref.grad) < 1e-11


# -- 4.3 dbwd: three-way agreement ---------------------------------------------------------


def test_dbwd_matches_double_autograd_and_finite_differences(block, jd):
    """Three legs, not two: IR, torch double-autograd, and central differences of the loss.

    Two legs agreeing can mean one shared misconception -- both IR and autograd descend from
    the same op ordering. Finite differences are the independent check.
    """
    batch = make_batch("cpu", n=8, e=40, cfg=CFG)
    prog, meta = build_dbwd(CFG, gauss_coeff=block.gauss_coeff)
    inp, sizes = dbwd_inputs(block, batch, jd, meta)
    env = run(prog, inp, sizes)

    block.zero_grad(set_to_none=True)
    b2 = dict(batch)
    b2["pos"] = batch["pos"].detach().clone().requires_grad_(True)
    _, _, loss = conservative_training_step(block, b2, jd)
    loss.backward()

    assert rel(env[meta["loss"]].squeeze(), loss) < 1e-12
    assert rel(env[meta["grads"]["pos"]], b2["pos"].grad) < 1e-11

    # leg 3: central differences of L along a random direction
    def loss_at(p):
        b3 = dict(batch)
        b3["pos"] = p.detach().clone().requires_grad_(True)
        return conservative_training_step(block, b3, jd)[2].item()

    d = torch.randn_like(batch["pos"])
    h = 1e-6
    base = batch["pos"].detach()
    fd = (loss_at(base + h * d) - loss_at(base - h * d)) / (2 * h)
    ir = (env[meta["grads"]["pos"]] * d).sum().item()
    assert abs(ir - fd) / max(abs(fd), 1e-12) < 1e-6, f"IR {ir} vs finite differences {fd}"


# -- 4.4 symmetries, at the IR level -------------------------------------------------------


def test_ir_energy_is_translation_invariant(block, jd):
    batch = make_batch("cpu", n=12, e=80, cfg=CFG)
    prog, meta = build_forward(CFG, gauss_coeff=block.gauss_coeff)
    inp, sizes = bind(block, batch, jd, CFG)
    e0 = run(prog, inp, sizes)[meta["energy"]].item()
    inp2 = dict(inp)
    inp2["pos"] = inp["pos"] + torch.tensor([1.3, -2.7, 0.4], dtype=DT)
    e1 = run(prog, inp2, sizes)[meta["energy"]].item()
    assert abs(e1 - e0) / abs(e0) < 1e-12


def test_ir_forces_sum_to_zero(block, jd):
    """Sum F = 0 to machine precision -- valid under PBC.

    Net *torque* is deliberately not checked: it is not a valid invariant on a periodic
    cell, so a torque test here would be testing the fixture, not the block.
    """
    batch = make_batch("cpu", n=12, e=80, cfg=CFG)
    prog, meta = build_force(CFG, gauss_coeff=block.gauss_coeff)
    inp, sizes = bind(block, batch, jd, CFG)
    inp["seed_E"] = torch.ones(1, dtype=DT)
    f = run(prog, inp, sizes)[meta["force"]]
    assert f.sum(dim=0).abs().max() < 1e-10 * f.abs().max()


def test_ir_energy_is_rotation_invariant(block, jd):
    batch = make_batch("cpu", n=12, e=80, cfg=CFG)
    r_xyz, d = random_rotation("cpu", CFG.lmax, jd)
    prog, meta = build_forward(CFG, gauss_coeff=block.gauss_coeff)

    inp, sizes = bind(block, batch, jd, CFG)
    e0 = run(prog, inp, sizes)[meta["energy"]].item()

    b2 = dict(batch)
    b2["pos"] = batch["pos"].detach() @ r_xyz.T
    b2["x_node"] = torch.einsum("ij,njc->nic", d, batch["x_node"])
    inp2, _ = bind(block, b2, jd, CFG)
    e1 = run(prog, inp2, sizes)[meta["energy"]].item()
    assert abs(e1 - e0) / abs(e0) < 1e-11


def test_ir_energy_is_permutation_invariant(block, jd):
    batch = make_batch("cpu", n=12, e=80, cfg=CFG)
    prog, meta = build_forward(CFG, gauss_coeff=block.gauss_coeff)
    inp, sizes = bind(block, batch, jd, CFG)
    e0 = run(prog, inp, sizes)[meta["energy"]].item()

    n = batch["pos"].shape[0]
    perm = torch.randperm(n)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n)
    b2 = dict(batch)
    b2["pos"] = batch["pos"].detach()[perm]
    b2["x_node"] = batch["x_node"][perm]
    b2["atomic_numbers"] = batch["atomic_numbers"][perm]
    b2["edge_index"] = inv[batch["edge_index"]]
    inp2, _ = bind(block, b2, jd, CFG)
    e1 = run(prog, inp2, sizes)[meta["energy"]].item()
    assert abs(e1 - e0) / abs(e0) < 1e-12


# -- closure and the vocabulary shrink -----------------------------------------------------


@pytest.mark.parametrize("name", ["fwd", "force", "dbwd"])
def test_derived_programs_are_closed_and_never_use_sin_or_cos(name):
    """Vocabulary accounting: the rational Wigner path leaves sin and cos unused everywhere.

    v1.1 declares eight scalar functions. The assembled programs use six -- and crucially
    neither sin nor cos appears in fwd, force or dbwd, because no derivative rule introduces
    them and the rotation is built from Chebyshev/de Moivre polynomials over rsqrt.
    """
    build = {"fwd": build_forward, "force": build_force, "dbwd": build_dbwd}[name]
    prog, _ = build(CFG)
    assert_closed(prog)
    simp = simplify(prog, keep=prog.outputs)
    assert_closed(simp)
    fns = {op.fn for op in simp.ops.values() if op.kind == "scalar_map"}
    assert "sin" not in fns and "cos" not in fns, f"{name} uses {fns}"
    assert fns <= {"exp", "sigmoid", "silu", "rsqrt", "reciprocal", "poly_envelope"}


def test_repeated_operand_in_one_path_gets_the_product_rule():
    """Regression: `x*x` is one path reading operand 0 twice; d/dx must be 2x, not x.

    Taking `operands.index(k)` finds only the first occurrence and silently halves the
    derivative. The buffer-level diamond test cannot catch this -- it is a *path*-level
    accumulation site (DECISIONS.md D21).
    """
    from zippel.ir import BufferType, ContractionPath, IndexType, Program

    t = BufferType("edge", (("c", 3),))
    sl = (slice(None),)
    p = Program()
    p.add_input("x", t)
    p.add_input("ze", IndexType("edge"))
    sq = p.contract(["x"], t,
                    [ContractionPath(1.0, "c,c->c", (0, 0), (sl, sl), sl)], hint="sq")
    tot = p.contract([sq, sq], BufferType("graph", ()),
                     [ContractionPath(1.0, "c,c->", (0, 1), (sl, sl), ())],
                     out_index_map="ze", hint="sum")
    p.outputs = (tot,)
    seed = p.add_input("seed", BufferType("graph", ()))
    from zippel.vjp import vjp
    grads = vjp(p, tot, ["x"], seed=seed, zero_index={"edge": "ze"})

    xv = torch.randn(5, 3, dtype=DT)
    env = run(p, {"x": xv, "ze": torch.zeros(5, dtype=torch.long),
                  "seed": torch.ones(1, dtype=DT)}, {"edge": 5, "graph": 1})
    xt = xv.clone().requires_grad_(True)
    (want,) = torch.autograd.grad(((xt * xt) ** 2).sum(), xt)
    assert rel(env[grads["x"]], want) < 1e-13
