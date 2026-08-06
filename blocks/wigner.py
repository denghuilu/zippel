"""Wigner-D construction for the eSEN SO(2) block, in *rational* form.

fairchem builds the edge frame as Euler angles and then takes sines/cosines of integer
multiples of them:

    y_hat, x_hat, z_hat = normalize(edge_vec)
    beta  = acos(y_hat)                 <- Safeacos, a custom autograd.Function
    alpha = atan2(x_hat, z_hat)         <- Safeatan2
    gamma = rand()                      <- random roll; SO(2) conv is exact in it
    eulers = (-gamma, -beta, -alpha)
    wigner = blockdiag_l( Xa(-gamma) @ J_l @ Xb(-beta) @ J_l @ Xc(-alpha) )

where `Xz(theta)` carries cos(k*theta) on the diagonal and sin(k*theta) on the
anti-diagonal for k = l, l-1, ..., -l.

Two problems with that form.

1. `acos`/`atan2` are outside the SP-IR's declared transcendental set
   {exp, sigmoid/SiLU, rsqrt, polynomial envelope}.
2. fairchem's `Safeacos` is *silently wrong under double differentiation*: its
   `forward` saves `x.clamp(...)`, which is evaluated under no-grad, so the saved
   tensor carries no graph and the first derivative's dependence on x is invisible to
   a second differentiation. Measured in FP64 on GH200 against u^T W(pos) v: energy
   exact, force to 8.9e-16, but the grad-of-grad is off by 5.5% relative. See
   DECISIONS.md D5.

Both go away if we never form the angles. Write s2 = x_hat^2 + z_hat^2. Because the
vector is normalised, s2 == 1 - y_hat^2 exactly, so `r = sqrt(s2)` is *both* sin(beta)
and the atan2 radius, and one guard covers both degeneracies (edge along +-y).

    cos(beta)   = y_hat
    sin(beta)   = r                       = s2 * rsqrt(s2)
    cos(k*beta) = T_k(y_hat)                          <- Chebyshev 1st kind, polynomial
    sin(k*beta) = r * U_{k-1}(y_hat)                  <- Chebyshev 2nd kind, polynomial
    cos(alpha)  = z_hat * rsqrt(s2)
    sin(alpha)  = x_hat * rsqrt(s2)
    cos/sin(k*alpha)  by de Moivre recursion from those two   <- polynomial

Every Wigner entry is therefore a polynomial in (x_hat, y_hat, z_hat, rsqrt(s2)) — the
only non-polynomial primitive is `rsqrt`, which is already in the declared set. The set
is not widened and differentiation closure is preserved (DECISIONS.md D4).

gamma is position-independent, so cos(k*gamma)/sin(k*gamma) are per-edge constants;
they are generated once per fixture from a fixed seed and passed in (DECISIONS.md D7).

A fully angle-free Wigner recursion (straight from the unit vector, no Euler
parameterisation at all) is out of M1 scope; see DECISIONS.md D4 future work.
"""

from __future__ import annotations

import torch

# Guard on s2 = sin^2(beta). Hit only for edges within ~1e-6 rad of the +-y poles,
# where the Euler frame is genuinely singular; fairchem guards the same degeneracy with
# its YTOL/rot_clip and EPS clamps.
EPS_S2 = 1e-12


def unit_edge_vector(edge_vec: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Normalise edge vectors. Returns (x_hat, y_hat, z_hat, s2, inv_r).

    s2 = x_hat^2 + z_hat^2 = sin^2(beta), clamped away from 0.
    inv_r = rsqrt(s2). The only non-polynomial primitive in the whole construction.
    """
    inv_norm = torch.rsqrt(edge_vec.pow(2).sum(dim=-1, keepdim=True).clamp_min(EPS_S2))
    xyz = edge_vec * inv_norm
    x_hat, y_hat, z_hat = xyz.unbind(dim=-1)
    s2 = (x_hat * x_hat + z_hat * z_hat).clamp_min(EPS_S2)
    return x_hat, y_hat, z_hat, s2, torch.rsqrt(s2)


def _chebyshev_cos_sin(
    cos_t: torch.Tensor, sin_t: torch.Tensor, kmax: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """cos(k*t), sin(k*t) for k = 0..kmax by the de Moivre recursion.

    c_k = c_1*c_{k-1} - s_1*s_{k-1},  s_k = s_1*c_{k-1} + c_1*s_{k-1}.
    Pure multiply-add: a polynomial in (cos_t, sin_t), hence a segmented contraction.
    """
    cos_k = [torch.ones_like(cos_t), cos_t]
    sin_k = [torch.zeros_like(sin_t), sin_t]
    for k in range(2, kmax + 1):
        cos_k.append(cos_t * cos_k[k - 1] - sin_t * sin_k[k - 1])
        sin_k.append(sin_t * cos_k[k - 1] + cos_t * sin_k[k - 1])
    return cos_k[: kmax + 1], sin_k[: kmax + 1]


def _z_rot_mat_from_cos_sin(
    cos_k: list[torch.Tensor], sin_k: list[torch.Tensor], lv: int, sign: float
) -> torch.Tensor:
    """Rebuild fairchem's `_z_rot_mat(sign * theta, lv)` from precomputed cos/sin of k*theta.

    Frequencies run f = lv, lv-1, ..., -lv down the rows; entry (i, i) is cos(f*theta)
    and entry (i, 2*lv - i) is sin(f*theta). cos is even in f and in `sign`, sin is odd
    in both, so negative frequencies need no extra work.

    Write order matters: on the middle row (i == lv, frequency 0) the diagonal and the
    anti-diagonal are the *same* element, and it must end up holding cos(0) = 1, not
    sin(0) = 0. fairchem's `_z_rot_mat` writes sin first then cos for exactly this
    reason; we match that order.
    """
    ref = cos_k[0]
    m = ref.new_zeros((*ref.shape, 2 * lv + 1, 2 * lv + 1))
    for i in range(2 * lv + 1):
        f = lv - i
        k = abs(f)
        s = sign * (1.0 if f >= 0 else -1.0)
        m[..., i, 2 * lv - i] = s * sin_k[k]
    for i in range(2 * lv + 1):
        m[..., i, i] = cos_k[abs(lv - i)]
    return m


def wigner_from_edge_vec(
    edge_vec: torch.Tensor,
    cos_gamma_k: torch.Tensor,
    sin_gamma_k: torch.Tensor,
    lmax: int,
    jd: list[torch.Tensor],
) -> torch.Tensor:
    """Block-diagonal Wigner-D, [E, (lmax+1)^2, (lmax+1)^2], with no acos/atan2/sin/cos.

    Matches fairchem `eulers_to_wigner(init_edge_rot_euler_angles(edge_vec), 0, lmax, Jd)`
    when `cos_gamma_k[:, k] == cos(k*gamma)` and `sin_gamma_k[:, k] == sin(k*gamma)` for
    the same per-edge gamma.

    fairchem passes eulers = (-gamma, -beta, -alpha) as wigner_D's (alpha, beta, gamma),
    so the three z-rotations carry angles -gamma, -beta, -alpha respectively.
    """
    x_hat, y_hat, z_hat, s2, inv_r = unit_edge_vector(edge_vec)
    r = s2 * inv_r  # sqrt(s2) == sin(beta), via rsqrt only

    # beta: cos(k*beta) = T_k(y_hat), sin(k*beta) = r * U_{k-1}(y_hat).
    cos_b = [torch.ones_like(y_hat), y_hat]
    u = [torch.ones_like(y_hat), 2.0 * y_hat]  # U_0, U_1
    for k in range(2, lmax + 1):
        cos_b.append(2.0 * y_hat * cos_b[k - 1] - cos_b[k - 2])
        u.append(2.0 * y_hat * u[k - 1] - u[k - 2])
    sin_b = [torch.zeros_like(y_hat)] + [r * u[k - 1] for k in range(1, lmax + 1)]

    # alpha = atan2(x_hat, z_hat): cos = z_hat/r, sin = x_hat/r.
    cos_a, sin_a = _chebyshev_cos_sin(z_hat * inv_r, x_hat * inv_r, lmax)

    cos_g = [cos_gamma_k[:, k] for k in range(lmax + 1)]
    sin_g = [sin_gamma_k[:, k] for k in range(lmax + 1)]

    size = (lmax + 1) ** 2
    wigner = edge_vec.new_zeros(edge_vec.shape[0], size, size)
    start = 0
    for lv in range(lmax + 1):
        j = jd[lv].to(dtype=edge_vec.dtype, device=edge_vec.device)
        xa = _z_rot_mat_from_cos_sin(cos_g, sin_g, lv, -1.0)  # angle -gamma
        xb = _z_rot_mat_from_cos_sin(cos_b, sin_b, lv, -1.0)  # angle -beta
        xc = _z_rot_mat_from_cos_sin(cos_a, sin_a, lv, -1.0)  # angle -alpha
        block = xa @ j @ xb @ j @ xc
        end = start + 2 * lv + 1
        wigner[:, start:end, start:end] = block
        start = end
    return wigner


def wigner_blockdiag_from_angles(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    lmax: int,
    jd: list[torch.Tensor],
) -> torch.Tensor:
    """Block-diagonal Wigner-D from explicit ZYZ Euler angles, [B, (lmax+1)^2, ...].

    FOR TESTS ONLY -- this uses `torch.sin`/`torch.cos` directly and is therefore *not*
    part of the SP-IR path (the block itself never forms angles; see
    `wigner_from_edge_vec`). It exists so property tests can build D(R) for an arbitrary
    global rotation R without pulling in e3nn.

    Convention note: the l=1 block IS the plain cartesian 3x3 rotation matrix, in (x, y,
    z) order -- no permutation. Verified against e3nn's spherical_harmonics and by
    W(v) @ v_hat == (0, 1, 0) to 1.1e-16. The edge frame maps the edge onto +y, which is
    also the m = 0 axis.
    """
    ks = torch.arange(lmax + 1, device=alpha.device, dtype=alpha.dtype)
    trig = {}
    for name, ang in (("a", alpha), ("b", beta), ("c", gamma)):
        ka = ang[:, None] * ks[None, :]
        trig[name] = ([torch.cos(ka[:, k]) for k in range(lmax + 1)],
                      [torch.sin(ka[:, k]) for k in range(lmax + 1)])

    size = (lmax + 1) ** 2
    out = alpha.new_zeros(alpha.shape[0], size, size)
    start = 0
    for lv in range(lmax + 1):
        j = jd[lv].to(dtype=alpha.dtype, device=alpha.device)
        xa = _z_rot_mat_from_cos_sin(*trig["a"], lv, 1.0)
        xb = _z_rot_mat_from_cos_sin(*trig["b"], lv, 1.0)
        xc = _z_rot_mat_from_cos_sin(*trig["c"], lv, 1.0)
        end = start + 2 * lv + 1
        out[:, start:end, start:end] = xa @ j @ xb @ j @ xc
        start = end
    return out


def gamma_harmonics(
    gamma: torch.Tensor, lmax: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """cos(k*gamma), sin(k*gamma) for k = 0..lmax, as [E, lmax+1] each.

    gamma is the random roll angle. It carries no position dependence, so this is
    evaluated once per fixture and passed in as a constant per-edge input.
    """
    k = torch.arange(lmax + 1, device=gamma.device, dtype=gamma.dtype)
    kg = gamma[:, None] * k[None, :]
    return torch.cos(kg), torch.sin(kg)
