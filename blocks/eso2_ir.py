"""The eSEN SO(2) block expressed in the segmented-polynomial IR.

Mirrors `blocks/eso2_ref.py` exactly, including the rational Wigner-D construction, so the
interpreter can be checked against the reference op for op.

Built bottom-up with small helpers, because the Wigner assembly is where slice bookkeeping is
easy to get wrong: `build_wigner` is validated on its own against `blocks/wigner.py` before
the full block is assembled.
"""

from __future__ import annotations

import torch

from blocks.eso2_ref import BlockConfig, Layout
from zippel.ir import BufferType, ContractionPath, IndexType, Program

S = slice(None)


# ----------------------------------------------------------------------------------------
# small builders
# ----------------------------------------------------------------------------------------


def _scalar_t(seg="edge") -> BufferType:
    """A per-segment scalar: no trailing axes."""
    return BufferType(seg, ())


def component(p: Program, vec: str, i: int, unit: str, hint: str) -> str:
    """Extract component i of a vector-valued buffer as a rank-0 scalar buffer.

    Not a slice-and-reduce: `"x->"` would sum an index appearing in only one operand, which
    the vocabulary rejects because its transpose needs a broadcast (docs/ir.md 2.1). Instead
    it is a contraction against a static unit operand, so `x` appears twice and the transpose
    is the ordinary `",x->x"` scatter of the cotangent back into slot i (DECISIONS.md D19).

    `unit` is a `none`-segment buffer of ones with a single unit axis.
    """
    t = p.type_of(vec)
    return p.contract(
        [vec, unit], _scalar_t(t.segment),
        [ContractionPath(1.0, "x,x->", (0, 1), ((slice(i, i + 1),), (S,)), ())],
        hint=hint,
    )


def smul(p: Program, a: str, b: str, coeff: float = 1.0, hint="sm") -> str:
    """Product of two scalar buffers."""
    t = p.type_of(a)
    return p.contract(
        [a, b], t,
        [ContractionPath(coeff, ",->", (0, 1), ((), ()), ())],
        hint=hint,
    )


def sadd(p: Program, a: str, b: str, ca=1.0, cb=1.0, hint="sa") -> str:
    t = p.type_of(a)
    return p.contract(
        [a, b], t,
        [ContractionPath(ca, "->", (0,), ((),), ()),
         ContractionPath(cb, "->", (1,), ((),), ())],
        hint=hint,
    )


def sscale(p: Program, a: str, coeff: float, hint="ss") -> str:
    t = p.type_of(a)
    return p.contract([a], t, [ContractionPath(coeff, "->", (0,), ((),), ())], hint=hint)


def saffine(p: Program, a: str, coeff: float, const: float, ones: str, hint="af") -> str:
    """coeff*a + const, using a `none`-segment scalar `ones` buffer for the constant."""
    t = p.type_of(a)
    return p.contract(
        [a, ones], t,
        [ContractionPath(coeff, "->", (0,), ((),), ()),
         ContractionPath(const, "->", (1,), ((),), ())],
        hint=hint,
    )


# ----------------------------------------------------------------------------------------
# rational Wigner-D
# ----------------------------------------------------------------------------------------


def build_wigner(p: Program, edge_vec: str, cos_g: str, sin_g: str, lmax: int,
                 jd: str, ones: str, unit: str, unit_mat: str) -> str:
    """Block-diagonal Wigner-D as an IR buffer [E, K, K], K = (lmax+1)^2.

    Follows blocks/wigner.py: no acos/atan2/sin/cos on the position path. cos(k*beta) is a
    Chebyshev polynomial in y_hat, sin(k*beta) = r * U_{k-1}(y_hat), and cos/sin of k*alpha
    follow by de Moivre from z_hat/r and x_hat/r. rsqrt is the only non-polynomial primitive.
    """
    k_axis = (lmax + 1) ** 2
    e_scalar = _scalar_t("edge")

    # --- normalise -------------------------------------------------------------------
    r2 = p.contract(
        [edge_vec, edge_vec], e_scalar,
        [ContractionPath(1.0, "x,x->", (0, 1), ((S,), (S,)), ())], hint="r2")
    inv_norm = p.scalar(r2, "rsqrt", hint="invnorm")
    xyz = p.contract(
        [edge_vec, inv_norm], p.type_of(edge_vec),
        [ContractionPath(1.0, "x,->x", (0, 1), ((S,), ()), (S,))], hint="xyz")

    x_h = component(p, xyz, 0, unit, "xh")
    y_h = component(p, xyz, 1, unit, "yh")
    z_h = component(p, xyz, 2, unit, "zh")

    # s2 = x^2 + z^2 = sin^2(beta); r = sqrt(s2) = s2 * rsqrt(s2)
    s2 = p.contract(
        [xyz, xyz], e_scalar,
        [ContractionPath(1.0, "x,x->", (0, 1), ((slice(0, 1),), (slice(0, 1),)), ()),
         ContractionPath(1.0, "x,x->", (0, 1), ((slice(2, 3),), (slice(2, 3),)), ())],
        hint="s2")
    inv_r = p.scalar(s2, "rsqrt", hint="invr")
    r = smul(p, s2, inv_r, hint="r")

    # --- beta: cos(k b) = T_k(y), sin(k b) = r * U_{k-1}(y) ---------------------------
    cos_b = [ones, y_h]
    u = [ones, sscale(p, y_h, 2.0, hint="u1")]
    for k in range(2, lmax + 1):
        cos_b.append(sadd(p, smul(p, y_h, cos_b[k - 1], 2.0, hint="tc"), cos_b[k - 2],
                          1.0, -1.0, hint=f"T{k}"))
        u.append(sadd(p, smul(p, y_h, u[k - 1], 2.0, hint="uc"), u[k - 2],
                      1.0, -1.0, hint=f"U{k}"))
    zero = sscale(p, y_h, 0.0, hint="zero")
    sin_b = [zero] + [smul(p, r, u[k - 1], hint=f"sb{k}") for k in range(1, lmax + 1)]

    # --- alpha: cos = z/r, sin = x/r, then de Moivre ----------------------------------
    ca1 = smul(p, z_h, inv_r, hint="ca1")
    sa1 = smul(p, x_h, inv_r, hint="sa1")
    cos_a, sin_a = [ones, ca1], [zero, sa1]
    for k in range(2, lmax + 1):
        cos_a.append(sadd(p, smul(p, ca1, cos_a[k - 1], hint="cc"),
                          smul(p, sa1, sin_a[k - 1], hint="sss"), 1.0, -1.0, hint=f"ca{k}"))
        sin_a.append(sadd(p, smul(p, sa1, cos_a[k - 1], hint="sc"),
                          smul(p, ca1, sin_a[k - 1], hint="cs"), 1.0, 1.0, hint=f"sa{k}"))

    cos_gk = [component(p, cos_g, k, unit, f"cg{k}") for k in range(lmax + 1)]
    sin_gk = [component(p, sin_g, k, unit, f"sg{k}") for k in range(lmax + 1)]

    # --- assemble the block-diagonal rotation ------------------------------------------
    # Per l: Xa(-gamma) @ J_l @ Xb(-beta) @ J_l @ Xc(-alpha), placed on the diagonal.
    mat_t = BufferType("edge", (("i", k_axis), ("j", k_axis)))
    parts = []
    for lv in range(lmax + 1):
        n = 2 * lv + 1
        off = lv * lv
        xa = _zrot(p, cos_gk, sin_gk, lv, -1.0, k_axis, off, unit_mat, hint=f"Xa{lv}")
        xb = _zrot(p, cos_b, sin_b, lv, -1.0, k_axis, off, unit_mat, hint=f"Xb{lv}")
        xc = _zrot(p, cos_a, sin_a, lv, -1.0, k_axis, off, unit_mat, hint=f"Xc{lv}")
        blk = slice(off, off + n)
        # Xa J Xb J Xc, left to right, each a dense (n x n) matmul on the l-block
        acc = _matmul_block(p, xa, jd, blk, blk, mat_t, jd_slice=lv, hint=f"aj{lv}")
        acc = _matmul_block(p, acc, xb, blk, blk, mat_t, hint=f"ajb{lv}")
        acc = _matmul_block(p, acc, jd, blk, blk, mat_t, jd_slice=lv, hint=f"ajbj{lv}")
        acc = _matmul_block(p, acc, xc, blk, blk, mat_t, hint=f"w{lv}")
        parts.append((acc, blk))

    # sum the per-l blocks into one block-diagonal buffer
    return p.contract(
        [name for name, _ in parts], mat_t,
        [ContractionPath(1.0, "ij->ij", (k,), ((blk, blk),), (blk, blk))
         for k, (_, blk) in enumerate(parts)],
        hint="wigner",
    )


def _zrot(p: Program, cos_k, sin_k, lv: int, sign: float, k_axis: int, off: int,
          unit_mat: str, hint: str) -> str:
    """`_z_rot_mat(sign*theta, lv)` placed at [off:off+n, off:off+n] of a K x K buffer.

    Write order matters on the middle row (frequency 0), where the diagonal and the
    anti-diagonal are the same element and must end up holding cos(0) = 1 -- so cos paths are
    listed after sin paths, and the interpreter accumulates in path order.
    """
    n = 2 * lv + 1
    mat_t = BufferType("edge", (("i", k_axis), ("j", k_axis)))
    inputs, paths = [unit_mat], []

    def operand(buf: str) -> int:
        if buf not in inputs:
            inputs.append(buf)
        return inputs.index(buf)

    # Placing a rank-0 scalar into a 1x1 slot needs an operand to supply i and j: a bare
    # "->ij" produces indices nothing provides, which the type check rejects. `unit_mat` is a
    # none-segment [1,1] buffer of ones, so the path is ",ij->ij" -- the same static-operand
    # trick as `component` (DECISIONS.md D19).
    u = 0
    usl = (S, S)

    # sin on the anti-diagonal first...
    for i in range(n):
        f = lv - i
        sgn = sign * (1.0 if f >= 0 else -1.0)
        k = operand(sin_k[abs(f)])
        paths.append(ContractionPath(
            sgn, ",ij->ij", (k, u), ((), usl),
            (slice(off + i, off + i + 1), slice(off + n - 1 - i, off + n - i))))
    # ...then cos on the diagonal, so the middle element ends as cos(0) = 1
    for i in range(n):
        k = operand(cos_k[abs(lv - i)])
        paths.append(ContractionPath(
            1.0, ",ij->ij", (k, u), ((), usl),
            (slice(off + i, off + i + 1), slice(off + i, off + i + 1))))

    return p.contract(inputs, mat_t, paths, hint=hint)


def _matmul_block(p: Program, a: str, b: str, blk_a: slice, blk_b: slice,
                  mat_t: BufferType, jd_slice: int | None = None, hint="mm") -> str:
    """(a @ b) restricted to one diagonal block.

    `jd_slice` selects the l-block of the `none`-segment Jd buffer, which is stored in the
    same block-diagonal layout.
    """
    b_slice = (blk_b, blk_b)
    return p.contract(
        [a, b], mat_t,
        [ContractionPath(1.0, "ik,kj->ij", (0, 1), ((blk_a, blk_b), b_slice), (blk_a, blk_b))],
        hint=hint,
    )


__all__ = ["build_wigner", "component", "smul", "sadd", "sscale", "saffine"]
