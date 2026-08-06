"""Gate 0: validate `blocks/eso2_ref.py` against the actual fairchem module.

Target is fairchem-core 2.11.0's UMA `SO2_Convolution` / `GateActivation`, instantiated
at the smallest published eSEN config (K4L2). The standalone `esen/` package exists only
in fairchem-core 2.0.0, which declares Requires-Python >=3.9,<3.13 and cannot be
installed here; UMA is its maintained descendant with the same math (DECISIONS.md D3).

Tolerances from the work order: FP32 <= 1e-5, FP64 <= 1e-10 max relative error.
"""

from __future__ import annotations

import pytest
import torch

from blocks.eso2_ref import BlockConfig, ESO2RefBlock, Layout
from tests.conftest import requires_cuda

fairchem = pytest.importorskip("fairchem.core.models.uma.nn.so2_layers")
_so3 = pytest.importorskip("fairchem.core.models.uma.common.so3")
_act = pytest.importorskip("fairchem.core.models.uma.nn.activation")

TOL = {torch.float64: 1e-10, torch.float32: 1e-5}


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).abs().max() / b.abs().max().clamp_min(1e-30)).item()


def _copy_(dst: torch.nn.Module | torch.Tensor, src) -> None:
    with torch.no_grad():
        if isinstance(dst, torch.Tensor):
            dst.copy_(src)
        else:
            for p, q in zip(dst.parameters(), src.parameters()):
                p.copy_(q)


@pytest.fixture(scope="module")
def cfg():
    return BlockConfig()


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@requires_cuda
def test_so2_conv1_matches_fairchem(device, cfg, dtype):
    """conv1: radial-modulated, with the extra m=0 head that emits the gate scalars."""
    torch.manual_seed(0)
    c_in, h = 2 * cfg.sphere_channels, cfg.hidden_channels
    mapping = _so3.CoefficientMapping(cfg.lmax, cfg.mmax).to(device)

    fc = fairchem.SO2_Convolution(
        c_in, h, cfg.lmax, cfg.mmax, mapping,
        internal_weights=False,
        edge_channels_list=list(cfg.edge_channels_list),
        extra_m0_output_channels=cfg.lmax * h,
    ).to(device, dtype)
    ref = ESO2RefBlock(cfg).to(device, dtype)

    _copy_(ref.c1_m0, fc.fc_m0)
    for mine, theirs in zip(ref.c1_m, fc.so2_m_conv):
        _copy_(mine, theirs)
    _copy_(ref.rad_func, fc.rad_func)

    e = 512
    x = torch.randn(e, cfg.num_coeffs, c_in, device=device, dtype=dtype)
    x_edge = torch.randn(e, cfg.edge_channels_list[0], device=device, dtype=dtype)

    want_out, want_gate = fc(x, x_edge)
    got_out, got_gate = ref._so2_conv(
        x, ref.c1_m0, ref.c1_m, h, ref.rad_func(x_edge), ref.extra_m0
    )
    assert got_out.shape == want_out.shape
    assert _rel(got_out, want_out) <= TOL[dtype], f"conv1 out rel {_rel(got_out, want_out):.2e}"
    assert _rel(got_gate, want_gate) <= TOL[dtype], f"gate rel {_rel(got_gate, want_gate):.2e}"


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@requires_cuda
def test_so2_conv2_matches_fairchem(device, cfg, dtype):
    """conv2: internal weights, no radial modulation, no extra m=0 head."""
    torch.manual_seed(1)
    h, c_out = cfg.hidden_channels, cfg.sphere_channels
    mapping = _so3.CoefficientMapping(cfg.lmax, cfg.mmax).to(device)

    fc = fairchem.SO2_Convolution(
        h, c_out, cfg.lmax, cfg.mmax, mapping,
        internal_weights=True, edge_channels_list=None, extra_m0_output_channels=None,
    ).to(device, dtype)
    ref = ESO2RefBlock(cfg).to(device, dtype)

    _copy_(ref.c2_m0, fc.fc_m0)
    for mine, theirs in zip(ref.c2_m, fc.so2_m_conv):
        _copy_(mine, theirs)

    e = 512
    x = torch.randn(e, cfg.num_coeffs, h, device=device, dtype=dtype)
    want = fc(x, torch.empty(e, 0, device=device, dtype=dtype))
    got, gate = ref._so2_conv(x, ref.c2_m0, ref.c2_m, c_out, None, 0)

    assert gate is None
    assert got.shape == want.shape
    assert _rel(got, want) <= TOL[dtype], f"conv2 rel {_rel(got, want):.2e}"


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@requires_cuda
def test_gate_activation_matches_fairchem(device, cfg, dtype):
    torch.manual_seed(2)
    h = cfg.hidden_channels
    fc = _act.GateActivation(cfg.lmax, cfg.mmax, h, m_prime=True).to(device, dtype)
    ref = ESO2RefBlock(cfg).to(device, dtype)

    e = 512
    gate = torch.randn(e, cfg.lmax * h, device=device, dtype=dtype)
    x = torch.randn(e, cfg.num_coeffs, h, device=device, dtype=dtype)

    got, want = ref.gate_activation(gate, x), fc(gate, x)
    assert _rel(got, want) <= TOL[dtype], f"gate act rel {_rel(got, want):.2e}"


@requires_cuda
def test_layout_matches_fairchem_coefficient_mapping(device, cfg):
    """The l->m' permutation and per-m sizes must agree with fairchem exactly."""
    mapping = _so3.CoefficientMapping(cfg.lmax, cfg.mmax).to(device)
    layout = Layout.make(cfg.lmax, cfg.mmax)

    assert list(layout.m_size) == list(mapping.m_size)
    assert torch.equal(layout.to_m(device, torch.float64), mapping.to_m.to(device, torch.float64))
    assert torch.equal(
        layout.gate_expand_index(device),
        _act.GateActivation(cfg.lmax, cfg.mmax, cfg.hidden_channels, m_prime=True)
        .to(device).expand_index,
    )


@requires_cuda
def test_k4l2_derived_shapes_are_as_published(cfg):
    """Pin the K4L2 numbers so a config drift cannot pass silently.

    Source: fairchem_core-2.0.0 configs/puma/training_release/backbone/K4L2.yaml
    (lmax/mmax 2, sphere_channels 128, hidden_channels 128, edge_channels 128,
    num_distance_basis 64, cutoff 6.0), cross-checked against arXiv:2502.12147 App. A.1.
    """
    assert (cfg.lmax, cfg.mmax) == (2, 2)
    assert (cfg.sphere_channels, cfg.hidden_channels, cfg.edge_channels) == (128, 128, 128)
    assert (cfg.num_distance_basis, cfg.cutoff) == (64, 6.0)
    assert cfg.num_coeffs == 9
    assert cfg.edge_channels_list == [320, 128, 128]

    block = ESO2RefBlock(cfg)
    assert block.num_channels_rad == 1536
    assert block.edge_split == [768, 512, 256]
    assert tuple(block.layout.m_split) == (3, 4, 2)
    assert block.extra_m0 == 256
