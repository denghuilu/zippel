"""The lmax=4 anchor bucket: forward-only correctness at a second shape.

Scope is deliberately narrow (DECISIONS.md D12). This bucket exists only because FlashSO2
enforces `SUPPORTED_LMAXES = (4, 6, 8)` with `mmax == lmax` and so cannot run the M1 config
(lmax = 2), which would leave Phase 2's stage S1 with no fused-kernel forward to be "within
noise of".

**Forward only.** No backward, no double backward, no training-step test lives here, and
nothing measured at lmax = 4 may enter the Gate 3 verdict table. Those tests belong to the
lmax = 2 config and stay there.
"""

from __future__ import annotations

import pytest
import torch

from blocks.eso2_ref import ANCHOR_CONFIG_LMAX4, Layout
from tests.conftest import requires_cuda
from tests.test_ref_block import energy, make_batch, make_block, random_rotation

DT = torch.float64


def test_anchor_is_flashso2_compatible_and_not_the_headline_config():
    """lmax == mmax and lmax in FlashSO2's supported set; distinct from the M1 config."""
    from blocks.eso2_ref import BlockConfig

    cfg = ANCHOR_CONFIG_LMAX4
    assert cfg.lmax == cfg.mmax == 4, "FlashSO2 requires mmax == lmax"
    assert cfg.lmax in (4, 6, 8), "must be in FlashSO2's SUPPORTED_LMAXES"
    assert (cfg.lmax, cfg.mmax) != (BlockConfig().lmax, BlockConfig().mmax)
    assert cfg.num_coeffs == 25
    # channels unchanged, so only the irrep shape differs from the headline config
    assert cfg.sphere_channels == BlockConfig().sphere_channels


def test_anchor_layout_tables_are_consistent():
    cfg = ANCHOR_CONFIG_LMAX4
    layout = Layout.make(cfg.lmax, cfg.mmax)
    assert layout.m_size == (5, 4, 3, 2, 1)
    assert layout.m_split == (5, 8, 6, 4, 2)
    assert sum(layout.m_split) == cfg.num_coeffs
    assert len(layout.gate_expand_index("cpu")) == cfg.num_coeffs - 1


@requires_cuda
def test_anchor_forward_runs_and_is_finite(device, jd64):
    cfg = ANCHOR_CONFIG_LMAX4
    block = make_block(device, cfg=cfg)
    batch = make_batch(device, n=32, e=400, cfg=cfg)
    e = energy(block, batch, jd64)
    assert e.shape == () and torch.isfinite(e)


@requires_cuda
def test_anchor_forward_is_rotation_invariant(device, jd64):
    """The symmetry that makes the forward meaningful must hold at this shape too."""
    cfg = ANCHOR_CONFIG_LMAX4
    block = make_block(device, cfg=cfg)
    r_xyz, d = random_rotation(device, cfg.lmax, jd64)

    b0 = make_batch(device, n=32, e=400, cfg=cfg)
    e0 = energy(block, b0, jd64)

    b1 = make_batch(device, n=32, e=400, cfg=cfg)
    b1["pos"] = (b0["pos"].detach() @ r_xyz.T).requires_grad_(True)
    b1["x_node"] = torch.einsum("ij,njc->nic", d, b0["x_node"])
    e1 = energy(block, b1, jd64)

    assert torch.allclose(e0, e1, rtol=0, atol=1e-9), f"E: {e0.item()} vs {e1.item()}"


@requires_cuda
def test_anchor_forward_is_independent_of_the_random_roll(device, jd64):
    """Same structural check as the headline config, at the anchor shape."""
    from blocks.wigner import gamma_harmonics

    cfg = ANCHOR_CONFIG_LMAX4
    block = make_block(device, cfg=cfg)
    batch = make_batch(device, n=32, e=400, cfg=cfg)
    n_edges = batch["edge_index"].shape[1]

    energies = []
    for seed in range(3):
        g = torch.Generator(device="cpu").manual_seed(200 + seed)
        gamma = (torch.rand(n_edges, generator=g, dtype=DT) * 2 * torch.pi).to(device)
        cos_g, sin_g = gamma_harmonics(gamma, cfg.lmax)
        energies.append(
            energy(block, {**batch, "cos_gamma_k": cos_g, "sin_gamma_k": sin_g}, jd64).item()
        )
    assert max(energies) - min(energies) < 1e-12


@requires_cuda
def test_anchor_matches_fairchem_so2_conv_at_lmax4(device):
    """The per-m contraction must agree with fairchem at this shape as well as at lmax=2."""
    fairchem = pytest.importorskip("fairchem.core.models.uma.nn.so2_layers")
    so3 = pytest.importorskip("fairchem.core.models.uma.common.so3")

    from blocks.eso2_ref import ESO2RefBlock

    cfg = ANCHOR_CONFIG_LMAX4
    torch.manual_seed(0)
    c_in, h = 2 * cfg.sphere_channels, cfg.hidden_channels
    mapping = so3.CoefficientMapping(cfg.lmax, cfg.mmax).to(device)

    fc = fairchem.SO2_Convolution(
        c_in, h, cfg.lmax, cfg.mmax, mapping, internal_weights=False,
        edge_channels_list=list(cfg.edge_channels_list),
        extra_m0_output_channels=cfg.lmax * h,
    ).to(device, DT)
    ref = ESO2RefBlock(cfg).to(device, DT)

    with torch.no_grad():
        for dst, src in ((ref.c1_m0, fc.fc_m0), (ref.rad_func, fc.rad_func)):
            for p, q in zip(dst.parameters(), src.parameters()):
                p.copy_(q)
        for mine, theirs in zip(ref.c1_m, fc.so2_m_conv):
            for p, q in zip(mine.parameters(), theirs.parameters()):
                p.copy_(q)

    e = 256
    x = torch.randn(e, cfg.num_coeffs, c_in, device=device, dtype=DT)
    x_edge = torch.randn(e, cfg.edge_channels_list[0], device=device, dtype=DT)

    want_out, want_gate = fc(x, x_edge)
    got_out, got_gate = ref._so2_conv(
        x, ref.c1_m0, ref.c1_m, h, ref.rad_func(x_edge), ref.extra_m0
    )
    rel = lambda a, b: ((a - b).abs().max() / b.abs().max()).item()  # noqa: E731
    assert rel(got_out, want_out) <= 1e-10, f"anchor conv rel {rel(got_out, want_out):.2e}"
    assert rel(got_gate, want_gate) <= 1e-10
