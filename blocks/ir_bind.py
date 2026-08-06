"""Bind an `ESO2RefBlock`'s parameters and a fixture batch to the IR program's inputs.

Keeping this separate from `eso2_ir.py` makes the correspondence auditable: every reshape
here is a claim about how the reference lays out a weight, and a wrong claim shows up
immediately as a mismatch against the reference forward.
"""

from __future__ import annotations

import torch

from blocks.eso2_ref import BlockConfig, Layout

DT = torch.float64


def pack_jd(jd_list, lmax: int) -> torch.Tensor:
    k = (lmax + 1) ** 2
    out = torch.zeros(1, k, k, dtype=DT)
    for l in range(lmax + 1):
        o, n = l * l, 2 * l + 1
        out[0, o:o + n, o:o + n] = jd_list[l].to(DT)
    return out


def bind(block, batch, jd_list, cfg: BlockConfig | None = None) -> tuple[dict, dict]:
    """Returns (inputs, sizes) for `zippel.interp.run`."""
    cfg = cfg or block.cfg
    L, C, H = cfg.lmax, cfg.sphere_channels, cfg.hidden_channels
    mmax, EC, NB = cfg.mmax, cfg.edge_channels, cfg.num_distance_basis
    C2, layout = 2 * C, Layout.make(L, mmax)
    n = batch["pos"].shape[0]
    e = batch["edge_index"].shape[1]
    src, dst = batch["edge_index"][0].cpu(), batch["edge_index"][1].cpu()
    z = batch["atomic_numbers"].cpu()

    def t(x):
        return x.detach().to("cpu", DT)

    p = {n_: v for n_, v in block.named_parameters()}
    inputs = {
        "pos": t(batch["pos"]),
        "x_node": t(batch["x_node"]),
        "shifts": t(batch["shifts"]),
        "cos_g": t(batch["cos_gamma_k"]),
        "sin_g": t(batch["sin_gamma_k"]),
        "emb_src": t(block.elem_emb(z[src].to(block.elem_emb.weight.device))),
        "emb_dst": t(block.elem_emb(z[dst].to(block.elem_emb.weight.device))),
        "src": src, "dst": dst,
        "zn": torch.zeros(n, dtype=torch.long),
        "ze": torch.zeros(e, dtype=torch.long),
        "jd": pack_jd(jd_list, L),
        "to_m": layout.to_m("cpu", DT).unsqueeze(0),
        "gauss_offset": t(block.gauss_offset).unsqueeze(0),
        "ones_g": torch.ones(1, NB, dtype=DT),
        "ones": torch.ones(e, dtype=DT),
        "unit": torch.ones(1, 1, dtype=DT),
        "unit_mat": torch.ones(1, 1, 1, dtype=DT),
        "unit_m": torch.ones(1, 1, dtype=DT),
        "ones_ec": torch.ones(1, EC, dtype=DT),
    }

    rad = block.rad_func
    inputs |= {
        "rad_w0": t(rad[0].weight).unsqueeze(0), "rad_b0": t(rad[0].bias).unsqueeze(0),
        "rad_g0": t(rad[1].weight).unsqueeze(0), "rad_be0": t(rad[1].bias).unsqueeze(0),
        "rad_w1": t(rad[3].weight).unsqueeze(0), "rad_b1": t(rad[3].bias).unsqueeze(0),
        "rad_g1": t(rad[4].weight).unsqueeze(0), "rad_be1": t(rad[4].bias).unsqueeze(0),
    }
    # final radial layer, split per m-block and viewed as (k, c, h) -- the reference splits
    # the same 1536-wide output immediately after computing it
    fw, fb, off = t(rad[6].weight), t(rad[6].bias), 0
    for m in range(mmax + 1):
        km = layout.m_size[m]
        w = km * C2
        inputs[f"rad_wm{m}"] = fw[off:off + w].reshape(km, C2, EC).unsqueeze(0)
        inputs[f"rad_bm{m}"] = fb[off:off + w].reshape(km, C2).unsqueeze(0)
        off += w

    extra = L * H
    inputs["c1_w0"] = t(block.c1_m0.weight).reshape(-1, layout.m_size[0], C2).unsqueeze(0)
    inputs["c1_b0"] = t(block.c1_m0.bias).unsqueeze(0)
    inputs["c2_w0"] = t(block.c2_m0.weight).reshape(-1, layout.m_size[0], H).unsqueeze(0)
    inputs["c2_b0"] = t(block.c2_m0.bias).unsqueeze(0)

    for m in range(1, mmax + 1):
        km = layout.m_size[m]
        for tag, mod, cin, cout in (("c1", block.c1_m[m - 1], C2, H),
                                    ("c2", block.c2_m[m - 1], H, C)):
            w = t(mod.weight)
            half = w.shape[0] // 2
            # rows are (j, o) row-major; columns are (k, c) row-major
            inputs[f"{tag}_w{m}a"] = w[:half].reshape(km, cout, km, cin).unsqueeze(0)
            inputs[f"{tag}_w{m}b"] = w[half:].reshape(km, cout, km, cin).unsqueeze(0)

    inputs |= {
        "ro_w0": t(block.readout[0].weight).unsqueeze(0),
        "ro_b0": t(block.readout[0].bias).unsqueeze(0),
        "ro_w1": t(block.readout[2].weight).unsqueeze(0),
        "ro_b1": t(block.readout[2].bias).unsqueeze(0),
    }
    del extra
    return inputs, {"node": n, "edge": e, "graph": 1}
