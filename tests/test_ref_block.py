"""Property tests for the reference block: symmetries, liveness, and the double backward.

All in FP64. These are the invariants the SP-IR interpreter and the generated kernels
must also satisfy later, so they are written against a `make_batch` helper that any
implementation can be swapped into.
"""

from __future__ import annotations

import pytest
import torch

from blocks.eso2_ref import BlockConfig, ESO2RefBlock, conservative_training_step
from blocks.wigner import gamma_harmonics, wigner_blockdiag_from_angles
from tests.conftest import requires_cuda

DT = torch.float64


def make_batch(device, n=48, e=600, seed=0, cfg=BlockConfig()):
    g = torch.Generator(device="cpu").manual_seed(seed)
    gamma = torch.rand(e, generator=g, dtype=DT).to(device) * 2 * torch.pi
    cos_g, sin_g = gamma_harmonics(gamma, cfg.lmax)
    # No self-loops: src == dst gives a zero-length edge vector, where the edge frame is
    # genuinely undefined (both the 1/|v| normalisation and s2 = sin^2(beta) collapse).
    # Real neighbour lists never contain them; plain randint produces ~e/n of them and
    # they silently break equivariance and the finite-difference checks.
    src = torch.randint(0, n, (e,), generator=g)
    dst = (src + 1 + torch.randint(0, n - 1, (e,), generator=g)) % n
    return {
        "pos": (torch.rand(n, 3, generator=g, dtype=DT).to(device) * 9.0).requires_grad_(True),
        "atomic_numbers": torch.randint(1, 30, (n,), generator=g).to(device),
        "x_node": torch.randn(n, cfg.num_coeffs, cfg.sphere_channels, generator=g, dtype=DT).to(device),
        "edge_index": torch.stack([src, dst]).to(device),
        "shifts": torch.zeros(e, 3, device=device, dtype=DT),
        "cos_gamma_k": cos_g,
        "sin_gamma_k": sin_g,
        "e_ref": torch.zeros((), device=device, dtype=DT),
        "f_ref": torch.zeros(n, 3, device=device, dtype=DT),
    }


def make_block(device, seed=0, cfg=BlockConfig()):
    torch.manual_seed(seed)
    return ESO2RefBlock(cfg).to(device, DT)


def energy(block, batch, jd):
    return block(
        batch["pos"], batch["atomic_numbers"], batch["x_node"], batch["edge_index"],
        batch["shifts"], batch["cos_gamma_k"], batch["sin_gamma_k"], jd,
    )


def random_rotation(device, lmax, jd, seed=7):
    """Returns (R [3,3], D [K,K]) for one random rotation, consistently derived.

    D is built from explicit Euler angles and R is read straight off its l=1 block, so
    the two cannot drift out of convention with each other.

    The l=1 irrep components are ordered (x, y, z) -- plain cartesian, no permutation.
    Verified two independent ways: `e3nn.o3.spherical_harmonics(1, v, normalize=True,
    normalization="norm")` returns v_hat exactly, and the edge frame satisfies
    `W(v) @ v_hat == (0, 1, 0)` to 1.1e-16 under this ordering (and under no other).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    a, b, c = (torch.rand(3, generator=g, dtype=DT) * torch.tensor([6.28, 3.14, 6.28])).tolist()
    ang = lambda v: torch.tensor([v], device=device, dtype=DT)  # noqa: E731
    d = wigner_blockdiag_from_angles(ang(a), ang(b), ang(c), lmax, jd)[0]
    return d[1:4, 1:4], d


# ---------------------------------------------------------------------------------------


@requires_cuda
def test_all_parameters_receive_gradient(device, jd64):
    """Every parameter must be live.

    A readout reading only the l=0 row leaves all four m>0 convolution weights with
    exactly zero gradient -- the SO(2) machinery under test goes dead. This test is the
    guard for that regression (DECISIONS.md D9).
    """
    block, batch = make_block(device), make_batch(device)
    _, _, loss = conservative_training_step(block, batch, jd64)
    loss.backward()

    dead = [n for n, p in block.named_parameters()
            if p.grad is None or p.grad.abs().sum().item() == 0.0]
    assert not dead, f"parameters with zero gradient: {dead}"
    assert batch["pos"].grad is not None and batch["pos"].grad.abs().sum() > 0


@requires_cuda
def test_energy_is_rotation_invariant_and_force_is_equivariant(device, jd64):
    """E(R.pos, D(R).x) == E(pos, x) and F(R.pos, D(R).x) == R.F(pos, x).

    gamma is held fixed across the two calls: the SO(2) convolution commutes with a roll
    about the edge axis, so equivariance must hold for any gamma choice.
    """
    cfg = BlockConfig()
    block = make_block(device)
    r_xyz, d = random_rotation(device, cfg.lmax, jd64)

    b0 = make_batch(device, cfg=cfg)
    e0 = energy(block, b0, jd64)
    (f0,) = torch.autograd.grad(e0, b0["pos"], create_graph=True)

    b1 = make_batch(device, cfg=cfg)
    b1["pos"] = (b0["pos"].detach() @ r_xyz.T).requires_grad_(True)
    b1["x_node"] = torch.einsum("ij,njc->nic", d, b0["x_node"])
    e1 = energy(block, b1, jd64)
    (f1,) = torch.autograd.grad(e1, b1["pos"], create_graph=True)

    assert torch.allclose(e0, e1, rtol=0, atol=1e-9), f"E: {e0.item()} vs {e1.item()}"
    assert torch.allclose(f1, f0 @ r_xyz.T, rtol=0, atol=1e-9)


@requires_cuda
@pytest.mark.parametrize("fixture", ["si_small", "cu_small"])
def test_every_parameter_has_nonzero_grad_norm_on_real_fixtures(device, jd64, fixture):
    """Every parameter must be live on the *actual* fixtures, not just synthetic inputs.

    This is the guard for the readout defect: a linear `l = 0` head leaves all four m > 0
    convolution weights with exactly zero gradient, because the l = 0 output row is fed
    only by the m = 0 branch (DECISIONS.md D9). Asserting a nonzero *norm* per parameter
    catches that, and catches any future change that silently strands part of the block.
    """
    from blocks.eso2_ref import conservative_training_step
    from fixtures.load import load_batch

    cfg = BlockConfig()
    block = make_block(device, cfg=cfg).to(DT)
    batch = load_batch(fixture, device, DT, cfg)
    _, _, loss = conservative_training_step(block, batch, jd64)
    loss.backward()

    norms = {n: (0.0 if p.grad is None else p.grad.norm().item())
             for n, p in block.named_parameters()}
    dead = sorted(n for n, v in norms.items() if not v > 0.0)
    assert not dead, (
        f"{fixture}: {len(dead)}/{len(norms)} parameters have zero grad norm: {dead}"
    )
    assert all(torch.isfinite(p.grad).all() for p in block.parameters())
    assert len(norms) == 23, f"expected 23 parameter tensors, found {len(norms)}"

    # The measured unit is "backward L to all parameter grads AND position grads", so
    # pos.grad must also be populated: the force term makes the loss depend on pos twice.
    pos_grad = batch["pos"].grad
    assert pos_grad is not None and torch.isfinite(pos_grad).all()
    assert pos_grad.abs().sum() > 0


@requires_cuda
def test_energy_is_independent_of_the_random_roll(device, jd64):
    """E must not depend on gamma, the random roll about the edge axis.

    fairchem draws gamma freshly on every forward (`torch.rand_like`), so the model would
    be non-deterministic if this did not hold. It is a sharp structural check on the whole
    assembly: it passes only if the per-m complex contraction, the gate's real/imaginary
    pairing, and the l->m' permutation all commute with a roll. A mis-assembled block
    fails this even when every individual layer matches fairchem.
    """
    cfg = BlockConfig()
    block, batch = make_block(device), make_batch(device, cfg=cfg)
    n_edges = batch["edge_index"].shape[1]

    energies = []
    for seed in range(4):
        g = torch.Generator(device="cpu").manual_seed(100 + seed)
        gamma = (torch.rand(n_edges, generator=g, dtype=DT) * 2 * torch.pi).to(device)
        cos_g, sin_g = gamma_harmonics(gamma, cfg.lmax)
        energies.append(energy(block, {**batch, "cos_gamma_k": cos_g, "sin_gamma_k": sin_g}, jd64))

    spread = max(e.item() for e in energies) - min(e.item() for e in energies)
    assert spread < 1e-12, f"E depends on the roll angle: spread {spread:.3e}"


@requires_cuda
def test_energy_is_translation_invariant(device, jd64):
    block = make_block(device)
    b0 = make_batch(device)
    e0 = energy(block, b0, jd64)

    b1 = make_batch(device)
    b1["pos"] = (b0["pos"].detach() + torch.tensor([1.3, -2.7, 0.4], device=device, dtype=DT)
                 ).requires_grad_(True)
    assert torch.allclose(e0, energy(block, b1, jd64), rtol=0, atol=1e-9)


@requires_cuda
def test_energy_is_permutation_invariant(device, jd64):
    """Relabelling atoms (and remapping edges accordingly) must not change E."""
    block = make_block(device)
    b0 = make_batch(device)
    e0 = energy(block, b0, jd64)

    n = b0["pos"].shape[0]
    perm = torch.randperm(n, device=device)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n, device=device)

    b1 = make_batch(device)
    b1["pos"] = b0["pos"].detach()[perm].requires_grad_(True)
    b1["atomic_numbers"] = b0["atomic_numbers"][perm]
    b1["x_node"] = b0["x_node"][perm]
    b1["edge_index"] = inv[b0["edge_index"]]
    assert torch.allclose(e0, energy(block, b1, jd64), rtol=0, atol=1e-9)


@requires_cuda
def test_force_matches_finite_differences(device, jd64):
    """F = -dE/dpos against central differences -- an oracle independent of autograd."""
    block, batch = make_block(device), make_batch(device, n=16, e=120)
    e = energy(block, batch, jd64)
    (grad,) = torch.autograd.grad(e, batch["pos"], create_graph=True)

    d = torch.randn_like(batch["pos"])
    h = 1e-6
    base = batch["pos"].detach()
    fwd = energy(block, {**batch, "pos": base + h * d}, jd64)
    bwd = energy(block, {**batch, "pos": base - h * d}, jd64)
    fd = ((fwd - bwd) / (2 * h)).item()
    ad = (grad * d).sum().item()
    assert abs(ad - fd) <= 1e-6 * max(1.0, abs(fd)), f"autograd {ad} vs fd {fd}"


@requires_cuda
def test_double_backward_matches_finite_differences(device, jd64):
    """d(F.d)/dpos . d against central differences of the force -- the dbwd oracle.

    fairchem's own autograd is disqualified here: its Safeacos drops the second
    derivative through the beta Euler angle (DECISIONS.md D5), so finite differences and
    the FP64 interpreter are the ground truth for anything second-order.
    """
    block, batch = make_block(device), make_batch(device, n=16, e=120)
    base = batch["pos"].detach()
    d = torch.randn_like(base)
    h = 1e-6

    def force_dot_d(p):
        p = p.detach().clone().requires_grad_(True)
        (grad,) = torch.autograd.grad(energy(block, {**batch, "pos": p}, jd64), p, create_graph=True)
        return grad, p

    grad, p = force_dot_d(base)
    (hvp,) = torch.autograd.grad((grad * d).sum(), p)

    gp, _ = force_dot_d(base + h * d)
    gm, _ = force_dot_d(base - h * d)
    fd = (gp - gm) / (2 * h)

    rel = (hvp - fd).abs().max() / fd.abs().max()
    assert rel < 1e-6, f"dbwd vs finite differences: rel err {rel.item():.3e}"
