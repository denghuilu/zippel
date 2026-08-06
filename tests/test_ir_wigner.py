"""The rational Wigner-D construction, expressed in the IR and checked against the reference.

This is the most intricate part of the block -- Chebyshev and de Moivre recursions feeding a
block-diagonal assembly -- and the part where slice bookkeeping is easiest to get wrong, so it
is validated on its own before the full block is assembled on top of it.

CPU/FP64, E = 16-32, so the whole file runs in about a second.
"""

from __future__ import annotations

import pytest
import torch

from blocks.eso2_ir import build_wigner
from blocks.wigner import gamma_harmonics, wigner_from_edge_vec
from zippel.interp import run
from zippel.ir import BufferType, ContractionPath, IndexType, Program
from zippel.simplify import op_counts, simplify
from zippel.vjp import assert_closed, vjp

DT = torch.float64
S = slice(None)
LMAX = 2
K = (LMAX + 1) ** 2


def packed_jd():
    """Jd stored in the same block-diagonal [1, K, K] layout the IR contracts against."""
    jd = [j.to(DT) for j in torch.load("blocks/Jd.pt", weights_only=False)]
    out = torch.zeros(1, K, K, dtype=DT)
    for l in range(LMAX + 1):
        o, n = l * l, 2 * l + 1
        out[0, o:o + n, o:o + n] = jd[l]
    return out, jd


def wigner_program(with_energy=False):
    p = Program()
    for name, t in [
        ("edge_vec", BufferType("edge", (("x", 3),))),
        ("cos_g", BufferType("edge", (("k", LMAX + 1),))),
        ("sin_g", BufferType("edge", (("k", LMAX + 1),))),
        ("jd", BufferType("none", (("i", K), ("j", K)))),
        ("ones", BufferType("edge", ())),
        ("unit", BufferType("none", (("x", 1),))),
        ("unit_mat", BufferType("none", (("i", 1), ("j", 1)))),
        ("uu", BufferType("none", (("i", K),))),
        ("vv", BufferType("none", (("j", K),))),
    ]:
        p.add_input(name, t)
    p.add_input("ze", IndexType("edge"))
    w = build_wigner(p, "edge_vec", "cos_g", "sin_g", LMAX, "jd", "ones", "unit", "unit_mat")
    if not with_energy:
        p.outputs = (w,)
        return p, w, None
    # u^T W v summed over edges: direction-dependent, unlike ||Wv||^2 or sum(W^2), which are
    # rotation-invariant and would make the gradient check vacuous.
    e = p.contract(
        [w, "uu", "vv"], BufferType("graph", ()),
        [ContractionPath(1.0, "ij,i,j->", (0, 1, 2), ((S, S), (S,), (S,)), ())],
        out_index_map="ze", hint="E")
    p.outputs = (e,)
    return p, w, e


def make_inputs(e=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    jdp, jd_list = packed_jd()
    vec = torch.randn(e, 3, generator=g, dtype=DT)
    gamma = torch.rand(e, generator=g, dtype=DT) * 2 * torch.pi
    cos_g, sin_g = gamma_harmonics(gamma, LMAX)
    return {
        "edge_vec": vec, "cos_g": cos_g, "sin_g": sin_g, "jd": jdp,
        "ones": torch.ones(e, dtype=DT),
        "unit": torch.ones(1, 1, dtype=DT),
        "unit_mat": torch.ones(1, 1, 1, dtype=DT),
        "uu": torch.randn(1, K, generator=g, dtype=DT),
        "vv": torch.randn(1, K, generator=g, dtype=DT),
        "ze": torch.zeros(e, dtype=torch.long),
    }, jd_list


def test_ir_wigner_matches_the_reference_construction():
    e = 32
    prog, w, _ = wigner_program()
    inp, jd_list = make_inputs(e)
    env = run(prog, inp, {"edge": e, "node": 1, "graph": 1})
    want = wigner_from_edge_vec(inp["edge_vec"], inp["cos_g"], inp["sin_g"], LMAX, jd_list)
    assert (env[w] - want).abs().max() < 1e-12


def test_ir_wigner_is_orthogonal():
    """A property of the result, independent of the reference: Wigner-D blocks are orthogonal."""
    e = 32
    prog, w, _ = wigner_program()
    inp, _ = make_inputs(e)
    got = run(prog, inp, {"edge": e, "node": 1, "graph": 1})[w]
    eye = torch.eye(K, dtype=DT).expand(e, K, K)
    assert (got @ got.transpose(1, 2) - eye).abs().max() < 1e-12


def test_vjp_through_the_wigner_construction_matches_autograd():
    """The whole rational path -- rsqrt, Chebyshev, de Moivre, block assembly -- differentiated.

    This is the load-bearing check that the vocabulary is closed *in practice* and not only on
    toy programs: the derived program contains scatter-adds, broadcasts, unit-operand
    selections and nested matmuls, and still uses only the two ops.
    """
    e = 16
    prog, _, energy = wigner_program(with_energy=True)
    seed = prog.add_input("seed", BufferType("graph", ()))
    grads = vjp(prog, energy, ["edge_vec"], seed=seed, ones="ones", zero_index={"edge": "ze"})
    assert_closed(prog)

    inp, jd_list = make_inputs(e)
    inp["seed"] = torch.ones(1, dtype=DT)
    env = run(prog, inp, {"edge": e, "node": 1, "graph": 1})

    vt = inp["edge_vec"].clone().requires_grad_(True)
    want = torch.einsum("i,eij,j->", inp["uu"][0],
                        wigner_from_edge_vec(vt, inp["cos_g"], inp["sin_g"], LMAX, jd_list),
                        inp["vv"][0])
    (gv,) = torch.autograd.grad(want, vt)

    assert abs(env[energy].item() - want.item()) < 1e-12
    assert (env[grads["edge_vec"]] - gv).abs().max() / gv.abs().max() < 1e-12


def test_simplify_is_exact_on_the_derived_wigner_program():
    """CSE/DCE must not perturb a single bit, and should find real redundancy here."""
    e = 16
    prog, _, energy = wigner_program(with_energy=True)
    seed = prog.add_input("seed", BufferType("graph", ()))
    grads = vjp(prog, energy, ["edge_vec"], seed=seed, ones="ones", zero_index={"edge": "ze"})
    keep = (energy, grads["edge_vec"])

    inp, _ = make_inputs(e)
    inp["seed"] = torch.ones(1, dtype=DT)
    sizes = {"edge": e, "node": 1, "graph": 1}
    before = run(prog, inp, sizes)
    simplified = simplify(prog, keep=keep)
    after = run(simplified, inp, sizes)

    for name in keep:
        assert torch.equal(before[name], after[name]), "CSE/DCE perturbed a value"
    assert len(simplified.ops) < len(prog.ops), "no redundancy found where some is expected"


def test_wigner_forward_stays_inside_the_vocabulary():
    prog, _, _ = wigner_program()
    assert_closed(prog)
    counts = op_counts(prog)
    assert counts["total"] == counts["segmented_contraction"] + counts["scalar_map"]
    # the only transcendentals on the position path are rsqrt: no sin/cos/acos/atan2
    fns = {op.fn for op in prog.ops.values() if op.kind == "scalar_map"}
    assert fns == {"rsqrt"}, f"unexpected transcendentals on the position path: {fns}"
