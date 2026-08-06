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


# ----------------------------------------------------------------------------------------
# dense layers
# ----------------------------------------------------------------------------------------


def linear(p: Program, x: str, w: str, b: str | None, n_out: int, out_name: str,
           hint="lin") -> str:
    """y = W x + b.  x: [seg, n_in]; W: none [n_out, n_in]; b: none [n_out]."""
    t = p.type_of(x)
    out_t = BufferType(t.segment, ((out_name, n_out),))
    paths = [ContractionPath(1.0, "i,oi->o", (0, 1), ((S,), (S, S)), (S,))]
    inputs = [x, w]
    if b is not None:
        inputs.append(b)
        paths.append(ContractionPath(1.0, "o->o", (2,), ((S,),), (S,)))
    return p.contract(inputs, out_t, paths, hint=hint)


def layernorm(p: Program, x: str, gamma: str, beta: str, ones_c: str, eps: float,
              hint="ln") -> str:
    """LayerNorm over the (single) trailing axis.

    The mean is a contraction against a `none` vector of ones rather than a bare `"c->"`
    reduction, for the reason in DECISIONS.md D19: a single-operand reduction has no
    expressible transpose.
    """
    t = p.type_of(x)
    n = t.sizes[0]
    sc = BufferType(t.segment, ())

    mean = p.contract([x, ones_c], sc,
                      [ContractionPath(1.0 / n, "c,c->", (0, 1), ((S,), (S,)), ())], hint="mean")
    centered = p.contract([x, mean, ones_c], t,
                          [ContractionPath(1.0, "c->c", (0,), ((S,),), (S,)),
                           ContractionPath(-1.0, ",c->c", (1, 2), ((), (S,)), (S,))], hint="cen")
    var = p.contract([centered, centered], sc,
                     [ContractionPath(1.0 / n, "c,c->", (0, 1), ((S,), (S,)), ())], hint="var")
    # var + eps, then rsqrt. The eps term rides on a ones-contraction because a bare
    # constant needs some operand to carry the segment axis.
    shifted = p.contract([var, ones_c], sc,
                         [ContractionPath(1.0, "->", (0,), ((),), ()),
                          ContractionPath(eps / n, "c,c->", (1, 1), ((S,), (S,)), ())],
                         hint="vareps")
    inv = p.scalar(shifted, "rsqrt", hint="invstd")
    normed = p.contract([centered, inv], t,
                        [ContractionPath(1.0, "c,->c", (0, 1), ((S,), ()), (S,))], hint="nrm")
    return p.contract([normed, gamma, beta], t,
                      [ContractionPath(1.0, "c,c->c", (0, 1), ((S,), (S,)), (S,)),
                       ContractionPath(1.0, "c->c", (2,), ((S,),), (S,))], hint=hint)


# ----------------------------------------------------------------------------------------
# the full forward block
# ----------------------------------------------------------------------------------------


def build_forward(cfg: BlockConfig = BlockConfig(),
                  gauss_coeff: float | None = None) -> tuple[Program, dict]:
    """The eSEN SO(2) block, position -> E, entirely in the two-op vocabulary.

    Returns (program, meta) where meta names the buffers a caller needs: the energy, the
    positions, and the parameter buffers to differentiate with respect to.

    Scope note: the element-embedding *table lookup* is not in the program. `emb_src` and
    `emb_dst` arrive as edge-segment inputs. A table lookup is a gather along a non-segment
    axis, which the v1.1 index-map model does not carry, and it is position-independent -- so
    it lies outside the position->E path that forces and the double backward flow along, which
    is what Phase 1 is testing. Its VJP is the same scatter-add lemma already exercised
    (DECISIONS.md D20).
    """
    L, C, H = cfg.lmax, cfg.sphere_channels, cfg.hidden_channels
    K, mmax = cfg.num_coeffs, cfg.mmax
    layout = Layout.make(L, mmax)
    m_split, m_size = layout.m_split, layout.m_size
    EC, NB = cfg.edge_channels, cfg.num_distance_basis
    C2 = 2 * C
    p = Program()

    def inp(name, t):
        return p.add_input(name, t)

    # ---- inputs ---------------------------------------------------------------------
    inp("pos", BufferType("node", (("x", 3),)))
    inp("x_node", BufferType("node", (("m", K), ("c", C))))
    inp("shifts", BufferType("edge", (("x", 3),)))
    inp("cos_g", BufferType("edge", (("k", L + 1),)))
    inp("sin_g", BufferType("edge", (("k", L + 1),)))
    inp("emb_src", BufferType("edge", (("c", EC),)))
    inp("emb_dst", BufferType("edge", (("c", EC),)))
    for n in ("src", "dst"):
        inp(n, IndexType("edge"))
    inp("zn", IndexType("node"))
    inp("ze", IndexType("edge"))
    inp("jd", BufferType("none", (("i", K), ("j", K))))
    inp("to_m", BufferType("none", (("m", K), ("k", K))))
    inp("gauss_offset", BufferType("none", (("g", NB),)))
    inp("ones_g", BufferType("none", (("g", NB),)))
    inp("ones", BufferType("edge", ()))
    inp("unit", BufferType("none", (("x", 1),)))
    inp("unit_mat", BufferType("none", (("i", 1), ("j", 1))))
    inp("unit_m", BufferType("none", (("m", 1),)))

    params: list[str] = []

    def par(name, t):
        params.append(inp(name, t))
        return name

    # radial MLP: [320 -> 128] LN SiLU [128 -> 128] LN SiLU [128 -> per-m blocks]
    w_in = NB + 2 * EC
    par("rad_w0", BufferType("none", (("o", EC), ("i", w_in))))
    par("rad_b0", BufferType("none", (("o", EC),)))
    par("rad_g0", BufferType("none", (("c", EC),)))
    par("rad_be0", BufferType("none", (("c", EC),)))
    par("rad_w1", BufferType("none", (("o", EC), ("i", EC))))
    par("rad_b1", BufferType("none", (("o", EC),)))
    par("rad_g1", BufferType("none", (("c", EC),)))
    par("rad_be1", BufferType("none", (("c", EC),)))
    inp("ones_ec", BufferType("none", (("c", EC),)))
    # final radial layer emits one buffer per m, already shaped (k, c): the reference splits
    # a single 1536-wide output immediately, and emitting per-block avoids a reshape, which
    # einsum cannot express.
    for m in range(mmax + 1):
        par(f"rad_wm{m}", BufferType("none", (("k", m_size[m]), ("c", C2), ("h", EC))))
        par(f"rad_bm{m}", BufferType("none", (("k", m_size[m]), ("c", C2))))

    # conv1 (radial-modulated, emits gate scalars) and conv2 (internal weights)
    par("c1_w0", BufferType("none", (("o", H * (L + 1) + L * H), ("m", m_size[0]), ("c", C2))))
    par("c1_b0", BufferType("none", (("o", H * (L + 1) + L * H),)))
    par("c2_w0", BufferType("none", (("o", C * (L + 1)), ("m", m_size[0]), ("c", H))))
    par("c2_b0", BufferType("none", (("o", C * (L + 1)),)))
    for m in range(1, mmax + 1):
        km = m_size[m]
        for tag, cin, cout in (("c1", C2, H), ("c2", H, C)):
            par(f"{tag}_w{m}a", BufferType("none", (("j", km), ("o", cout), ("k", km), ("c", cin))))
            par(f"{tag}_w{m}b", BufferType("none", (("j", km), ("o", cout), ("k", km), ("c", cin))))

    par("ro_w0", BufferType("none", (("o", C), ("i", C * (L + 1)))))
    par("ro_b0", BufferType("none", (("o", C),)))
    par("ro_w1", BufferType("none", (("o", 1), ("i", C))))
    par("ro_b1", BufferType("none", (("o", 1),)))

    # ---- geometry -------------------------------------------------------------------
    evec_t = BufferType("edge", (("x", 3),))
    edge_vec = p.contract(
        ["pos", "pos", "shifts"], evec_t,
        [ContractionPath(1.0, "x->x", (0,), ((S,),), (S,)),
         ContractionPath(-1.0, "x->x", (1,), ((S,),), (S,)),
         ContractionPath(1.0, "x->x", (2,), ((S,),), (S,))],
        index_maps=["dst", "src", None], hint="evec")

    sc = BufferType("edge", ())
    r2 = p.contract([edge_vec, edge_vec], sc,
                    [ContractionPath(1.0, "x,x->", (0, 1), ((S,), (S,)), ())], hint="r2")
    dist = smul(p, r2, p.scalar(r2, "rsqrt", hint="invd"), hint="dist")

    # gaussian distance basis: exp(coeff * (d - offset)^2)
    gb_t = BufferType("edge", (("g", NB),))
    # dist must be broadcast across the basis axis by a *ones* vector: using gauss_offset as
    # the index-supplying operand would multiply by the offsets instead of broadcasting.
    diff = p.contract([dist, "ones_g", "gauss_offset"], gb_t,
                      [ContractionPath(1.0, ",g->g", (0, 1), ((), (S,)), (S,)),
                       ContractionPath(-1.0, "g->g", (2,), ((S,),), (S,))], hint="gdiff")
    # The reference builds its offsets with torch.linspace at the default (float32) dtype and
    # reads the spacing back with .item(), so float32 rounding is baked into its coefficient --
    # faithfully, since fairchem's GaussianSmearing does the same. Recomputing it exactly in
    # float64 leaves a ~9e-12 relative gap in E, so the caller may bind the block's own value.
    coeff = (-0.5 / ((cfg.cutoff / (NB - 1)) ** 2) if gauss_coeff is None else gauss_coeff)
    sq = p.contract([diff, diff], gb_t,
                    [ContractionPath(coeff, "g,g->g", (0, 1), ((S,), (S,)), (S,))], hint="gsq")
    gauss = p.scalar(sq, "exp", hint="gauss")

    xe_t = BufferType("edge", (("i", w_in),))
    x_edge = p.contract(
        [gauss, "emb_src", "emb_dst"], xe_t,
        # subscript letters are local to a path: reusing 'i' makes each of these a copy into
        # a distinct output slice. Writing "g->i" instead would read as summing g and
        # producing i from nowhere, which the checker rejects.
        [ContractionPath(1.0, "i->i", (0,), ((S,),), (slice(0, NB),)),
         ContractionPath(1.0, "i->i", (1,), ((S,),), (slice(NB, NB + EC),)),
         ContractionPath(1.0, "i->i", (2,), ((S,),), (slice(NB + EC, w_in),))],
        hint="xedge")

    h = linear(p, x_edge, "rad_w0", "rad_b0", EC, "c", hint="rl0")
    h = p.scalar(layernorm(p, h, "rad_g0", "rad_be0", "ones_ec", 1e-5, hint="rln0"),
                 "silu", hint="rs0")
    h = linear(p, h, "rad_w1", "rad_b1", EC, "c", hint="rl1")
    h = p.scalar(layernorm(p, h, "rad_g1", "rad_be1", "ones_ec", 1e-5, hint="rln1"),
                 "silu", hint="rs1")
    radial = {}
    for m in range(mmax + 1):
        rt = BufferType("edge", (("k", m_size[m]), ("c", C2)))
        radial[m] = p.contract(
            [h, f"rad_wm{m}", f"rad_bm{m}"], rt,
            [ContractionPath(1.0, "h,kch->kc", (0, 1), ((S,), (S, S, S)), (S, S)),
             ContractionPath(1.0, "kc->kc", (2,), ((S, S),), (S, S))], hint=f"radm{m}")

    # ---- rotation into the edge frame, fused with the l -> m' reordering --------------
    wig = build_wigner(p, edge_vec, "cos_g", "sin_g", L, "jd", "ones", "unit", "unit_mat")
    rot_t = BufferType("edge", (("m", K), ("j", K)))
    rot = p.contract(["to_m", wig], rot_t,
                     [ContractionPath(1.0, "mk,kj->mj", (0, 1), ((S, S), (S, S)), (S, S))],
                     hint="rot")

    msg_t = BufferType("edge", (("m", K), ("c", C2)))
    gathered = p.contract(
        ["x_node", "x_node"], msg_t,
        [ContractionPath(1.0, "mc->mc", (0,), ((S, S),), (S, slice(0, C))),
         ContractionPath(1.0, "mc->mc", (1,), ((S, S),), (S, slice(C, C2)))],
        index_maps=["src", "dst"], hint="cat")
    msg = p.contract([rot, gathered], msg_t,
                     [ContractionPath(1.0, "mj,jc->mc", (0, 1), ((S, S), (S, S)), (S, S))],
                     hint="rotin")

    # ---- conv1 -> gate -> conv2 -------------------------------------------------------
    conv1, gate = _so2_conv(p, msg, "c1", C2, H, layout, radial, L * H, hint="conv1")
    gated = _gate(p, gate, conv1, layout, H, L, "unit_m")
    conv2, _ = _so2_conv(p, gated, "c2", H, C, layout, None, 0, hint="conv2")

    # ---- envelope, rotate back, scatter ------------------------------------------------
    env_s = p.scalar(sscale(p, dist, 1.0 / cfg.cutoff, hint="dsc"), "poly_envelope", hint="env")
    out_t = BufferType("edge", (("m", K), ("c", C)))
    damped = p.contract([conv2, env_s], out_t,
                        [ContractionPath(1.0, "mc,->mc", (0, 1), ((S, S), ()), (S, S))],
                        hint="damp")
    back = p.contract([rot, damped], out_t,
                      [ContractionPath(1.0, "jm,jc->mc", (0, 1), ((S, S), (S, S)), (S, S))],
                      hint="rotback")
    node_out = p.contract([back], BufferType("node", (("m", K), ("c", C))),
                          [ContractionPath(1.0, "mc->mc", (0,), ((S, S),), (S, S))],
                          out_index_map="dst", hint="scatter")

    # ---- invariant readout --------------------------------------------------------------
    invar_t = BufferType("node", (("i", C * (L + 1)),))
    paths = [ContractionPath(1.0, "mi,m->i", (0, 1), ((slice(0, 1), S), (S,)), (slice(0, C),))]
    for l in range(1, L + 1):
        blk = slice(l * l, (l + 1) ** 2)
        paths.append(ContractionPath(
            1.0, "mi,mi->i", (0, 0), ((blk, S), (blk, S)), (slice(l * C, (l + 1) * C),)))
    invar = p.contract([node_out, "unit_m"], invar_t, paths, hint="invar")

    r = linear(p, invar, "ro_w0", "ro_b0", C, "c", hint="rol0")
    r = p.scalar(r, "silu", hint="rosl")
    r = linear(p, r, "ro_w1", "ro_b1", 1, "c", hint="rol1")
    energy = p.contract(
        [r, "unit_m"], BufferType("graph", ()),
        [ContractionPath(1.0, "c,c->", (0, 1), ((S,), (S,)), ())],
        out_index_map="zn", hint="E")
    p.outputs = (energy,)
    return p, {"energy": energy, "pos": "pos", "params": params, "cfg": cfg, "layout": layout}


def _so2_conv(p, x, tag, c_in, c_out, layout, radial, extra_m0, hint):
    """Per-m block contractions. Returns (output, gate_scalars_or_None)."""
    L, mmax = layout.lmax, layout.mmax
    K = (L + 1) ** 2
    m_size, m_split = layout.m_size, layout.m_split
    out_t = BufferType("edge", (("m", K), ("c", c_out)))

    # m = 0: rows [0, m_size[0]), a dense linear over (m, c) jointly
    n0 = m_size[0]
    m0_out = c_out * (L + 1) + extra_m0
    ins = [x, f"{tag}_w0", f"{tag}_b0"]
    sl0 = (slice(0, n0), S)
    if radial is not None:
        ins.append(radial[0])
        modulated = p.contract([x, radial[0]], BufferType("edge", (("k", n0), ("c", c_in))),
                               [ContractionPath(1.0, "kc,kc->kc", (0, 1), (sl0, (S, S)), (S, S))],
                               hint=f"{hint}_mod0")
        src0, sl_src = modulated, (S, S)
    else:
        src0, sl_src = x, sl0
    flat = p.contract([src0, f"{tag}_w0", f"{tag}_b0"],
                      BufferType("edge", (("o", m0_out),)),
                      [ContractionPath(1.0, "mc,omc->o", (0, 1), (sl_src, (S, S, S)), (S,)),
                       ContractionPath(1.0, "o->o", (2,), ((S,),), (S,))],
                      hint=f"{hint}_m0")

    gate = None
    if extra_m0:
        gate = p.contract([flat], BufferType("edge", (("o", extra_m0),)),
                          [ContractionPath(1.0, "o->o", (0,), ((slice(0, extra_m0),),), (S,))],
                          hint=f"{hint}_gate")

    # The m=0 head is laid out flat as (l, channel); placing each l-row into the m'-major
    # output is a reshape, which einsum cannot express -- so `unit_m` supplies the m index.
    paths = [ContractionPath(
        1.0, "c,m->mc", (0, 1),
        ((slice(extra_m0 + i * c_out, extra_m0 + (i + 1) * c_out),), (S,)),
        (slice(i, i + 1), S)) for i in range(L + 1)]
    inputs, all_paths = [flat, "unit_m"], list(paths)
    off = n0
    for m in range(1, mmax + 1):
        km = m_size[m]
        re_in, im_in = slice(off, off + km), slice(off + km, off + 2 * km)
        if radial is not None:
            xm = p.contract([x, radial[m]], BufferType("edge", (("k", 2 * km), ("c", c_in))),
                            [ContractionPath(1.0, "kc,kc->kc", (0, 1),
                                             ((slice(off, off + km), S), (S, S)),
                                             (slice(0, km), S)),
                             ContractionPath(1.0, "kc,kc->kc", (0, 1),
                                             ((slice(off + km, off + 2 * km), S), (S, S)),
                                             (slice(km, 2 * km), S))],
                            hint=f"{hint}_mod{m}")
            src, re_s, im_s = xm, slice(0, km), slice(km, 2 * km)
        else:
            src, re_s, im_s = x, re_in, im_in
        wa, wb = f"{tag}_w{m}a", f"{tag}_w{m}b"
        i0 = len(inputs); inputs += [src, wa, wb]
        # out_r = W1 x_re - W2 x_im ;  out_i = W1 x_im + W2 x_re
        all_paths += [
            ContractionPath(1.0, "kc,jokc->jo", (i0, i0 + 1), ((re_s, S), (S, S, S, S)),
                            (re_in, S)),
            ContractionPath(-1.0, "kc,jokc->jo", (i0, i0 + 2), ((im_s, S), (S, S, S, S)),
                            (re_in, S)),
            ContractionPath(1.0, "kc,jokc->jo", (i0, i0 + 1), ((im_s, S), (S, S, S, S)),
                            (im_in, S)),
            ContractionPath(1.0, "kc,jokc->jo", (i0, i0 + 2), ((re_s, S), (S, S, S, S)),
                            (im_in, S)),
        ]
        off += 2 * km
    return p.contract(inputs, out_t, all_paths, hint=hint), gate


def _gate(p, gate, x, layout, H, L, unit_m):
    """SiLU on the l=0 scalar row; sigmoid(gate) multiplies every other row."""
    K = (L + 1) ** 2
    t = BufferType("edge", (("m", K), ("c", H)))
    sig = p.scalar(gate, "sigmoid", hint="gsig")
    scal = p.scalar(x, "silu", hint="gsilu")
    idx = layout.gate_expand_index("cpu").tolist()
    paths = [ContractionPath(1.0, "mc->mc", (0,), ((slice(0, 1), S),), (slice(0, 1), S))]
    for row, g in enumerate(idx, start=1):
        paths.append(ContractionPath(
            1.0, "mc,c->mc", (1, 2),
            ((slice(row, row + 1), S), (slice(g * H, (g + 1) * H),)),
            (slice(row, row + 1), S)))
    return p.contract([scal, x, sig], t, paths, hint="gate")
