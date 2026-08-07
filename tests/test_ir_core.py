"""IR core, VJP, and simplifier tests.

All CPU/FP64 and tiny, so the whole file runs in seconds — the Phase 1 budget is < 5 min for
the entire suite, and FP64 oracles are deliberately not run on medium/large fixtures.
"""

from __future__ import annotations

import math

import pytest
import torch

from zippel.ir import BufferType, ContractionPath, IndexType, Program
from zippel.interp import apply_scalar, peak_live_bytes, run
from zippel.simplify import cse, dce, op_counts, signatures, simplify
from zippel.vjp import ClosureError, assert_closed, vjp

DT = torch.float64


# ----------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------


def tiny_graph(n=6, e=11, c=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "n": n, "e": e, "c": c,
        "src": torch.randint(0, n, (e,), generator=g),
        "dst": torch.randint(0, n, (e,), generator=g),
        "x": torch.randn(n, c, generator=g, dtype=DT),
        "w": torch.randn(1, c, generator=g, dtype=DT),
    }


def build_chain(gd):
    """gather -> *w -> silu -> scatter-add -> sum of squares. Returns (prog, energy_name)."""
    c = gd["c"]
    node_t = BufferType("node", (("c", c),))
    edge_t = BufferType("edge", (("c", c),))
    one_t = BufferType("none", (("c", c),))
    graph_t = BufferType("graph", ())
    sl = (slice(None),)

    p = Program()
    p.add_input("x", node_t)
    p.add_input("w", one_t)
    p.add_input("ones", one_t)
    p.add_input("src", IndexType("edge"))
    p.add_input("dst", IndexType("edge"))
    p.add_input("zn", IndexType("node"))
    p.add_input("ze", IndexType("edge"))

    g = p.contract(["x"], edge_t, [ContractionPath(1.0, "c->c", (0,), (sl,), sl)],
                   index_maps=["src"], hint="gather")
    m = p.mul(g, "w")
    a = p.scalar(m, "silu")
    s = p.contract([a], node_t, [ContractionPath(1.0, "c->c", (0,), (sl,), sl)],
                   out_index_map="dst", hint="scatter")
    e = p.contract([s, s], graph_t,
                   [ContractionPath(1.0, "c,c->", (0, 1), (sl, sl), ())],
                   out_index_map="zn", hint="energy")
    p.outputs = (e,)
    return p, e


def chain_inputs(gd):
    return {
        "x": gd["x"], "w": gd["w"], "ones": torch.ones(1, gd["c"], dtype=DT),
        "src": gd["src"], "dst": gd["dst"],
        "zn": torch.zeros(gd["n"], dtype=torch.long),
        "ze": torch.zeros(gd["e"], dtype=torch.long),
    }


def chain_sizes(gd):
    return {"node": gd["n"], "edge": gd["e"], "graph": 1}


def torch_chain(gd):
    x = gd["x"].clone().requires_grad_(True)
    w = gd["w"].clone().requires_grad_(True)
    a = torch.nn.functional.silu(x[gd["src"]] * w)
    s = torch.zeros(gd["n"], gd["c"], dtype=DT).index_add_(0, gd["dst"], a)
    return (s * s).sum(), x, w


# ----------------------------------------------------------------------------------------
# construction-time type checking
# ----------------------------------------------------------------------------------------


def test_shape_errors_are_caught_at_construction():
    p = Program()
    p.add_input("a", BufferType("edge", (("c", 4),)))
    p.add_input("b", BufferType("edge", (("c", 8),)))
    sl = (slice(None),)
    with pytest.raises(ValueError, match="inconsistent extents"):
        p.contract(["a", "b"], BufferType("edge", (("c", 4),)),
                   [ContractionPath(1.0, "c,c->c", (0, 1), (sl, sl), sl)])


def test_single_operand_reduction_is_rejected():
    """docs/ir.md 2.1: its transpose would need a broadcast, which einsum cannot express."""
    p = Program()
    p.add_input("a", BufferType("edge", (("m", 3), ("c", 4))))
    with pytest.raises(ValueError, match="appearing in only one operand"):
        p.contract(["a"], BufferType("edge", (("c", 4),)),
                   [ContractionPath(1.0, "mc->c", (0,), ((slice(None), slice(None)),),
                                    (slice(None),))])


def test_scalar_map_rejects_functions_outside_the_vocabulary():
    p = Program()
    p.add_input("a", BufferType("edge", (("c", 4),)))
    with pytest.raises(ValueError, match="outside vocabulary"):
        p.scalar("a", "tanh")


def test_only_poly_envelope_carries_an_order():
    p = Program()
    p.add_input("a", BufferType("edge", (("c", 4),)))
    with pytest.raises(ValueError, match="derivative order"):
        p.scalar("a", "exp", order=1)


# ----------------------------------------------------------------------------------------
# interpreter and VJP against autograd
# ----------------------------------------------------------------------------------------


def test_interpreter_matches_torch_forward():
    gd = tiny_graph()
    prog, e = build_chain(gd)
    env = run(prog, chain_inputs(gd), chain_sizes(gd))
    want, _, _ = torch_chain(gd)
    assert torch.allclose(env[e].squeeze(), want, rtol=0, atol=1e-14)


def test_vjp_matches_autograd_for_positions_and_parameters():
    gd = tiny_graph()
    prog, e = build_chain(gd)
    seed = prog.add_input("seed", BufferType("graph", ()))
    grads = vjp(prog, e, ["x", "w"], seed=seed, ones="ones",
                zero_index={"node": "zn", "edge": "ze"})

    inp = chain_inputs(gd) | {"seed": torch.ones(1, dtype=DT)}
    env = run(prog, inp, chain_sizes(gd))
    want, x, w = torch_chain(gd)
    gx, gw = torch.autograd.grad(want, [x, w])

    assert (env[grads["x"]] - gx).abs().max() / gx.abs().max() < 1e-13
    assert (env[grads["w"]] - gw).abs().max() / gw.abs().max() < 1e-13


def test_cotangent_accumulation_on_a_diamond():
    """One buffer, two consumers, merged: the classic AD bug site.

    A transform that overwrote instead of accumulating would produce exactly one of the two
    branch gradients here, which is why this is tested on its own rather than incidentally.
    """
    c = 4
    t = BufferType("edge", (("c", c),))
    sl = (slice(None),)
    g = torch.Generator().manual_seed(3)

    p = Program()
    p.add_input("x", t)
    p.add_input("ones", t)
    p.add_input("ze", IndexType("edge"))

    a = p.scalar("x", "sigmoid")          # branch 1
    b = p.mul("x", "x", hint="sq")        # branch 2
    merged = p.add(a, b, hint="merge")    # diamond join
    total = p.contract([merged, "ones"], BufferType("graph", ()),
                       [ContractionPath(1.0, "c,c->", (0, 1), (sl, sl), ())],
                       out_index_map="ze", hint="sum")
    p.outputs = (total,)

    seed = p.add_input("seed", BufferType("graph", ()))
    grads = vjp(p, total, ["x"], seed=seed, ones="ones", zero_index={"edge": "ze"})

    xv = torch.randn(5, c, generator=g, dtype=DT)
    inp = {"x": xv, "ones": torch.ones(5, c, dtype=DT),
           "ze": torch.zeros(5, dtype=torch.long), "seed": torch.ones(1, dtype=DT)}
    env = run(p, inp, {"edge": 5, "graph": 1})

    xt = xv.clone().requires_grad_(True)
    want = (torch.sigmoid(xt) + xt * xt).sum()
    (gx,) = torch.autograd.grad(want, xt)

    assert torch.allclose(env[total].squeeze(), want, rtol=0, atol=1e-14)
    got = env[grads["x"]]
    assert (got - gx).abs().max() / gx.abs().max() < 1e-13
    # and it is genuinely the sum: neither branch alone would match
    only_sigmoid = torch.sigmoid(xt) * (1 - torch.sigmoid(xt))
    assert not torch.allclose(got, only_sigmoid, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("fn", ["exp", "sigmoid", "silu", "rsqrt", "reciprocal", "sin", "cos"])
def test_scalar_map_derivatives_match_autograd(fn):
    """Every f' in the VJP table, checked against autograd on its own."""
    t = BufferType("edge", (("c", 3),))
    sl = (slice(None),)
    p = Program()
    p.add_input("x", t)
    p.add_input("ones", t)
    p.add_input("ze", IndexType("edge"))
    y = p.scalar("x", fn)
    total = p.contract([y, "ones"], BufferType("graph", ()),
                       [ContractionPath(1.0, "c,c->", (0, 1), (sl, sl), ())],
                       out_index_map="ze", hint="sum")
    p.outputs = (total,)
    seed = p.add_input("seed", BufferType("graph", ()))
    grads = vjp(p, total, ["x"], seed=seed, ones="ones", zero_index={"edge": "ze"})

    g = torch.Generator().manual_seed(11)
    base = torch.rand(7, 3, generator=g, dtype=DT) + 0.5  # > 0 for rsqrt/reciprocal
    inp = {"x": base, "ones": torch.ones(7, 3, dtype=DT),
           "ze": torch.zeros(7, dtype=torch.long), "seed": torch.ones(1, dtype=DT)}
    env = run(p, inp, {"edge": 7, "graph": 1})

    xt = base.clone().requires_grad_(True)
    want = apply_scalar(fn, xt).sum()
    (gx,) = torch.autograd.grad(want, xt)
    assert (env[grads["x"]] - gx).abs().max() / gx.abs().max().clamp_min(1e-30) < 1e-12


# ----------------------------------------------------------------------------------------
# closure
# ----------------------------------------------------------------------------------------


def test_derived_programs_stay_inside_the_vocabulary():
    gd = tiny_graph()
    prog, e = build_chain(gd)
    seed = prog.add_input("seed", BufferType("graph", ()))
    vjp(prog, e, ["x", "w"], seed=seed, ones="ones", zero_index={"node": "zn", "edge": "ze"})
    assert_closed(prog)   # vjp() already asserts, but state it as the load-bearing check
    kinds = {op.kind for op in prog.ops.values()}
    assert kinds <= {"segmented_contraction", "scalar_map"}


def test_closure_assertion_actually_fires():
    """A falsification check: the closure test must be able to detect an escape."""
    from dataclasses import replace as dc_replace

    p = Program()
    p.add_input("a", BufferType("edge", (("c", 4),)))
    y = p.scalar("a", "exp")
    p.ops[y] = dc_replace(p.ops[y], fn="tanh")   # smuggle in an out-of-vocabulary function
    with pytest.raises(ClosureError, match="tanh"):
        assert_closed(p)


# ----------------------------------------------------------------------------------------
# envelope smoothness
# ----------------------------------------------------------------------------------------


def test_poly_envelope_is_c2_across_the_cutoff():
    """Value, first and second derivative continuous at d = 1.

    This is what makes the double backward well-defined at the cutoff; a C^1-only envelope
    would give a finite but wrong second derivative there.
    """
    # Approaching d = 1 from inside, p^(k)(1-eps) -> 0 for k = 0,1,2. Testing against a fixed
    # threshold would be arbitrary -- p''(1-eps) ~ 210*eps, so any threshold just encodes a
    # choice of eps. Test the *rate* instead: halving eps must halve the residual, which is
    # what continuity at the boundary actually means.
    for order in (0, 1, 2):
        vals = [apply_scalar("poly_envelope", torch.tensor([1 - eps], dtype=DT), order).abs().item()
                for eps in (1e-6, 5e-7)]
        assert vals[0] < 1e-3, f"order {order} does not approach 0 at d=1 (got {vals[0]})"
        if vals[0] > 1e-15:
            ratio = vals[0] / max(vals[1], 1e-300)
            assert 1.5 < ratio < 4.5, (
                f"order {order} residual scales as {ratio:.2f}x when eps halves; "
                "continuous vanishing should give ~2x (linear) or ~4x (quadratic)"
            )
        # outside the support it is identically zero, not merely small
        right = apply_scalar("poly_envelope", torch.tensor([1 + 1e-9], dtype=DT), order)
        assert right.abs().item() == 0.0

    # and the derivative chain is consistent with finite differences inside the support
    d = torch.tensor([0.3, 0.6, 0.9], dtype=DT)
    h = 1e-6
    for order in (0, 1):
        fd = (apply_scalar("poly_envelope", d + h, order)
              - apply_scalar("poly_envelope", d - h, order)) / (2 * h)
        analytic = apply_scalar("poly_envelope", d, order + 1)
        assert (fd - analytic).abs().max() < 1e-6


# ----------------------------------------------------------------------------------------
# simplifier
# ----------------------------------------------------------------------------------------


def test_cse_is_exact_and_removes_duplicates():
    gd = tiny_graph()
    prog, e = build_chain(gd)
    # duplicate the whole chain: CSE must collapse it back
    dup, e2 = build_chain(gd)
    before = run(prog, chain_inputs(gd), chain_sizes(gd))[e]

    p2 = Program(inputs=dict(prog.inputs), outputs=prog.outputs, _counter=prog._counter)
    p2.ops = dict(prog.ops)
    n_before = len(p2.ops)
    for name, op in dup.ops.items():
        if name not in p2.ops:
            p2.ops[name] = op
    simplified = simplify(p2, keep=(e,))
    after = run(simplified, chain_inputs(gd), chain_sizes(gd))[e]

    assert torch.equal(before, after), "CSE/DCE must be bit-exact"
    assert len(simplified.ops) <= n_before


def test_dce_drops_unreachable_ops():
    gd = tiny_graph()
    prog, e = build_chain(gd)
    prog.scalar("x", "exp", hint="unused")       # nothing consumes it
    assert any(n.startswith("unused") for n in prog.ops)
    pruned = dce(prog, keep=(e,))
    assert not any(n.startswith("unused") for n in pruned.ops)


def test_signature_count_is_at_most_op_count():
    gd = tiny_graph()
    prog, e = build_chain(gd)
    counts = op_counts(prog)
    assert counts["total"] == counts["segmented_contraction"] + counts["scalar_map"]
    assert len(signatures(prog)) <= counts["total"]


def test_peak_live_bytes_is_monotone_in_graph_size():
    small = tiny_graph(n=6, e=11)
    big = tiny_graph(n=60, e=110)
    prog, _ = build_chain(small)
    assert peak_live_bytes(prog, chain_sizes(big)) > peak_live_bytes(prog, chain_sizes(small))


def test_repeated_operand_with_differing_slices_is_checked_per_position():
    """A path naming one operand twice must be extent-checked at *each* position.

    `_check_contraction` resolved an operand's slice with `p.operands.index(j)`, which finds
    only the first position that operand appears at. For `x[0:3] * x[3:8]` -- operands (0, 0)
    with different slices -- both groups were then checked against the extent of the first
    slice, so a spec demanding equal extents passed against unequal ones. The type checker's
    guarantee ("an ill-typed program cannot be built", docs/ir.md) had a hole exactly where the
    D21 product-rule bug lived, and Phase 2's emitter relies on that guarantee.

    Latent when found: all four repeated-operand paths in fwd/force/dbwd use identical slices
    at both positions, so no computed value was ever wrong.
    """
    prog = Program()
    x = prog.add_input("x", BufferType("edge", (("a", 8),)))

    # "a,a->a" demands both operands have the same extent; 3 != 5, so this must be rejected.
    with pytest.raises(ValueError, match="inconsistent extents"):
        prog.contract(
            [x], BufferType("edge", (("a", 3),)),
            [ContractionPath(1.0, "a,a->a", (0, 0),
                             ((slice(0, 3),), (slice(3, 8),)), (slice(0, 3),))])

    # The legal form -- two equal-extent slices of the same buffer -- still builds.
    y = prog.contract(
        [x], BufferType("edge", (("a", 3),)),
        [ContractionPath(1.0, "a,a->a", (0, 0),
                         ((slice(0, 3),), (slice(5, 8),)), (slice(0, 3),))])
    assert prog.type_of(y).sizes == (3,)


def test_fusion_groups_are_schedulable():
    """The group graph must be acyclic, or the partition is not a launch count.

    The first version of `fusion_groups` merged an op into any group holding a fusable producer
    without checking the group graph. LayerNorm breaks that: `x - mean(x)` fuses with `x` while
    `mean(x)` reduces `x` into its own group, so the two groups each depend on the other. 101 of
    107 dbwd groups were in such cycles, and the reported "107 launches" was unachievable rather
    than merely optimistic.
    """
    from blocks.eso2_ir import build_dbwd, build_force, build_forward
    from blocks.eso2_ref import BlockConfig
    from zippel.simplify import fusion_groups

    for build in (build_forward, build_force, build_dbwd):
        prog, _ = build(BlockConfig())
        simp = simplify(prog, keep=prog.outputs)
        groups = fusion_groups(simp)

        where = {n: i for i, g in enumerate(groups) for n in g}
        succ = {i: set() for i in range(len(groups))}
        for name, op in simp.ops.items():
            for src in op.inputs:
                if src in where and where[src] != where[name]:
                    succ[where[src]].add(where[name])

        indeg = dict.fromkeys(succ, 0)
        for outs in succ.values():
            for j in outs:
                indeg[j] += 1
        queue = [i for i, d in indeg.items() if d == 0]
        visited = 0
        while queue:
            i = queue.pop()
            visited += 1
            for j in succ[i]:
                indeg[j] -= 1
                if indeg[j] == 0:
                    queue.append(j)
        assert visited == len(groups), (
            f"{len(groups) - visited} of {len(groups)} groups are in a dependence cycle; "
            "the partition cannot be scheduled as kernel launches")

        # every op lands in exactly one group
        assert sum(len(g) for g in groups) == len(simp.ops)
