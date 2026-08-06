"""Schwartz-Zippel identity checks on the multilinear core (work order section 4.6).

The rewrite under test is nontrivial and real: the per-m SO(2) contraction is a complex
product, so it can be written either as the four-term real form the block uses
(`out_r = W1 x_r - W2 x_i`, `out_i = W1 x_i + W2 x_r`) or by the three-multiplication
Karatsuba-style factorisation. Those are the same polynomial; PIT should say so.

Its limitation is documented in findings/pit-exactness.md and is not solved here.
"""

from __future__ import annotations

import torch

from zippel.interp import run
from zippel.ir import BufferType, ContractionPath, IndexType, Program
from zippel.pit import pit_equal

DT = torch.float64
S = slice(None)
E, K, C = 12, 4, 3


def _prog(karatsuba: bool):
    """Complex product (W1 + i W2)(x_r + i x_i), two algebraically equal formulations."""
    p = Program()
    xt = BufferType("edge", (("k", 2 * K), ("c", C)))
    wt = BufferType("none", (("k", K), ("c", C)))
    out_t = BufferType("edge", (("k", 2 * K), ("c", C)))
    p.add_input("x", xt)
    p.add_input("w1", wt)
    p.add_input("w2", wt)
    re, im = slice(0, K), slice(K, 2 * K)

    if not karatsuba:
        return p, p.contract(
            ["x", "w1", "w2"], out_t,
            [ContractionPath(1.0, "kc,kc->kc", (0, 1), ((re, S), (S, S)), (re, S)),
             ContractionPath(-1.0, "kc,kc->kc", (0, 2), ((im, S), (S, S)), (re, S)),
             ContractionPath(1.0, "kc,kc->kc", (0, 1), ((im, S), (S, S)), (im, S)),
             ContractionPath(1.0, "kc,kc->kc", (0, 2), ((re, S), (S, S)), (im, S))],
            hint="direct")

    # Karatsuba: a = W1 x_r, b = W2 x_i, c = (W1 + W2)(x_r + x_i)
    #   out_r = a - b ;  out_i = c - a - b
    half = BufferType("edge", (("k", K), ("c", C)))
    a = p.contract(["x", "w1"], half,
                   [ContractionPath(1.0, "kc,kc->kc", (0, 1), ((re, S), (S, S)), (S, S))],
                   hint="a")
    b = p.contract(["x", "w2"], half,
                   [ContractionPath(1.0, "kc,kc->kc", (0, 1), ((im, S), (S, S)), (S, S))],
                   hint="b")
    xs = p.contract(["x"], half,
                    [ContractionPath(1.0, "kc->kc", (0,), ((re, S),), (S, S)),
                     ContractionPath(1.0, "kc->kc", (0,), ((im, S),), (S, S))], hint="xs")
    ws = p.contract(["w1", "w2"], wt,
                    [ContractionPath(1.0, "kc->kc", (0,), ((S, S),), (S, S)),
                     ContractionPath(1.0, "kc->kc", (1,), ((S, S),), (S, S))], hint="ws")
    cc = p.contract([xs, ws], half,
                    [ContractionPath(1.0, "kc,kc->kc", (0, 1), ((S, S), (S, S)), (S, S))],
                    hint="c")
    return p, p.contract(
        [a, b, cc], out_t,
        [ContractionPath(1.0, "kc->kc", (0,), ((S, S),), (re, S)),
         ContractionPath(-1.0, "kc->kc", (1,), ((S, S),), (re, S)),
         ContractionPath(1.0, "kc->kc", (2,), ((S, S),), (im, S)),
         ContractionPath(-1.0, "kc->kc", (0,), ((S, S),), (im, S)),
         ContractionPath(-1.0, "kc->kc", (1,), ((S, S),), (im, S))],
        hint="karatsuba")


def _inputs(seed: int):
    g = torch.Generator().manual_seed(seed)
    return {"x": torch.randn(E, 2 * K, C, generator=g, dtype=DT),
            "w1": torch.randn(1, K, C, generator=g, dtype=DT),
            "w2": torch.randn(1, K, C, generator=g, dtype=DT)}


def test_pit_accepts_a_genuine_rewrite_of_the_complex_product():
    pa, oa = _prog(False)
    pb, ob = _prog(True)
    ok, worst = pit_equal(pa, oa, pb, ob, _inputs, {"edge": E, "graph": 1}, trials=8)
    assert ok, f"PIT rejected an identity it should accept (worst {worst:.3e})"
    assert worst < 1e-11


def test_pit_rejects_a_planted_sign_flip():
    """Falsification: PIT must detect a wrong rewrite, or accepting one proves nothing."""
    pa, oa = _prog(False)
    pb, _ = _prog(True)
    # flip one sign in the Karatsuba assembly
    bad = pb.ops[list(pb.ops)[-1]]
    flipped = tuple(
        ContractionPath(-pth.coeff, pth.subscripts, pth.operands, pth.in_slices, pth.out_slice)
        if i == 1 else pth for i, pth in enumerate(bad.paths))
    from dataclasses import replace
    pb.ops[bad.name] = replace(bad, paths=flipped)
    ok, worst = pit_equal(pa, oa, pb, bad.name, _inputs, {"edge": E, "graph": 1}, trials=4)
    assert not ok, "PIT accepted a program with a planted sign flip"
    assert worst > 1e-3
