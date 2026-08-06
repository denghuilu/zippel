"""Standalone reference for one eSEN-style SO(2) interaction block.

Dependency-light (torch only), readable, FP64-capable. This is the semantic ground
truth that the SP-IR interpreter is validated against; it is itself validated against
fairchem 2.11's UMA `SO2_Convolution` / `Edgewise` (see tests/test_ref_vs_fairchem.py).

Block config is the smallest *published* eSEN config, K4L2 / eSEN-sm:
  fairchem_core-2.0.0 configs/puma/training_release/backbone/K4L2.yaml
  cutoff/max_neighbors from the OMol25 eSEN-sm top-level config
  cross-checked against arXiv:2502.12147 App. A.1 ("Lmax=2, Mmax=2", 6 A)

Data flow (mirrors fairchem UMA `Edgewise.forward_chunk`):

    gather src/dst node irreps  ->  cat to 2C channels
      -> rotate into the edge frame, fused with the l->m' reordering   (to_m @ W)
      -> so2_conv_1  (radial-MLP-modulated, emits gate scalars)
      -> gate activation
      -> so2_conv_2  (internal weights, no radial)
      -> polynomial cutoff envelope
      -> rotate back                                                    (to_m @ W)^T
      -> index_add_ scatter to target nodes
      -> per-atom scalar readout head  ->  E

Two deliberate departures from fairchem, both to make the computation well-defined
rather than to make it faster (DECISIONS.md D4, D5, D7):
  * Wigner-D is built in rational form (blocks/wigner.py) -- no acos/atan2/sin/cos, and
    correct under double differentiation.
  * gamma (the random Wigner roll) is a seeded per-edge input, not drawn inside forward.

Node features `x_node` are an *input*, not a function of position: this is one block in
the middle of a network, so position-dependence enters only through the geometry
(Wigner-D, radial MLP, envelope) -- which is exactly the path forces flow along.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from blocks.wigner import wigner_from_edge_vec


# --------------------------------------------------------------------------------------
# config + layout
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BlockConfig:
    """Smallest published eSEN config (K4L2 / eSEN-sm). Do not invent values here."""

    lmax: int = 2
    mmax: int = 2
    sphere_channels: int = 128
    hidden_channels: int = 128
    edge_channels: int = 128
    num_distance_basis: int = 64
    cutoff: float = 6.0
    max_num_elements: int = 100
    envelope_exponent: int = 5

    @property
    def num_coeffs(self) -> int:
        return (self.lmax + 1) ** 2

    @property
    def edge_channels_list(self) -> list[int]:
        # [num_distance_basis + 2*edge_channels, edge_channels, edge_channels]
        return [
            self.num_distance_basis + 2 * self.edge_channels,
            self.edge_channels,
            self.edge_channels,
        ]


#: Secondary shape bucket, used **only** as the Phase-2 S1 forward anchor (DECISIONS.md D12).
#:
#: FlashSO2 enforces ``SUPPORTED_LMAXES = (4, 6, 8)`` with ``mmax == lmax`` and cannot run the
#: M1 config (lmax = 2), which would leave stage S1 with no fused-kernel forward to be "within
#: noise of". This bucket exists so that comparison is possible at all.
#:
#: Scope, deliberately narrow:
#:   * forward only -- no backward, no double backward, no training path;
#:   * used for (i) correctness of our generated forward against the FP64 interpreter at a
#:     second shape, and (ii) a wall-clock comparison against FlashSO2's forward in its own env;
#:   * **excluded from the Gate 3 verdict table** and from every speedup / peak-memory claim.
#:
#: Note this is *not* a published eSEN config: the nearest one (K10L4) is lmax 4 with mmax 2,
#: but FlashSO2 requires mmax == lmax, so the anchor uses lmax = mmax = 4. It is a shape bucket
#: for a sanity comparison, not a model claim.
ANCHOR_CONFIG_LMAX4 = BlockConfig(lmax=4, mmax=4)


@dataclass(frozen=True)
class Layout:
    """Every table derived from (lmax, mmax) -- never hand-tabulated."""

    lmax: int
    mmax: int
    m_size: tuple[int, ...]  # number of l's carrying each m
    m_split: tuple[int, ...]  # row counts per m-block in m'-order (m=0 real; m>0 re+im)

    @staticmethod
    def make(lmax: int, mmax: int) -> Layout:
        m_size = tuple(lmax - m + 1 for m in range(mmax + 1))
        m_split = (m_size[0],) + tuple(2 * s for s in m_size[1:])
        return Layout(lmax=lmax, mmax=mmax, m_size=m_size, m_split=m_split)

    def to_m(self, device, dtype) -> torch.Tensor:
        """Permutation [(lmax+1)^2, (lmax+1)^2] taking l-major order to m'-major order.

        l-major (e3nn/fairchem natural): for l = 0..lmax, m_c = -l..+l.
        m'-major: [m=0 for all l] then for |m| = 1..mmax [real for all l, imag for all l].
        Reproduces fairchem `CoefficientMapping.to_m` for the mmax == lmax case.
        """
        l_harm, m_complex = [], []
        for l in range(self.lmax + 1):
            mm = min(self.mmax, l)
            for m in range(-mm, mm + 1):
                l_harm.append(l)
                m_complex.append(m)
        n = len(l_harm)
        mat = torch.zeros(n, n, device=device, dtype=dtype)
        row = 0
        for m in range(self.mmax + 1):
            # fairchem's complex_idx: real part is +m (and m=0), imaginary part is -m
            for sign in ((1,) if m == 0 else (1, -1)):
                for idx, (lh, mc) in enumerate(zip(l_harm, m_complex)):
                    del lh
                    if mc == sign * m:
                        mat[row, idx] = 1.0
                        row += 1
        assert row == n, (row, n)
        return mat

    def gate_expand_index(self, device) -> torch.Tensor:
        """fairchem GateActivation(m_prime=True) expand_index, for lmax/mmax."""
        idx = []
        idx.extend(range(self.lmax))  # m = 0 rows, l = 1..lmax
        for m in range(1, self.mmax + 1):
            half = list(range(m - 1, self.lmax))
            idx.extend(half + half)  # real then imaginary
        return torch.tensor(idx, dtype=torch.long, device=device)


# --------------------------------------------------------------------------------------
# the block
# --------------------------------------------------------------------------------------


class ESO2RefBlock(nn.Module):
    """One SO(2) interaction block + per-atom energy head. Reference semantics."""

    def __init__(self, cfg: BlockConfig = BlockConfig()):
        super().__init__()
        self.cfg = cfg
        self.layout = Layout.make(cfg.lmax, cfg.mmax)
        L, C, H = cfg.lmax, cfg.sphere_channels, cfg.hidden_channels

        # --- element embedding + gaussian distance basis (edge invariants) -------------
        self.elem_emb = nn.Embedding(cfg.max_num_elements, cfg.edge_channels)
        offset = torch.linspace(0.0, cfg.cutoff, cfg.num_distance_basis)
        self.register_buffer("gauss_offset", offset, persistent=False)
        self.gauss_coeff = -0.5 / (offset[1] - offset[0]).item() ** 2

        # --- radial MLP: Linear/LN/SiLU stack, final Linear widens to num_channels_rad --
        # conv1 sees 2C channels (source (+) target), so its per-m input widths are:
        in_m0_1 = (L + 1) * (2 * C)
        in_m_1 = [self.layout.m_size[m] * (2 * C) for m in range(1, cfg.mmax + 1)]
        self.num_channels_rad = in_m0_1 + sum(in_m_1)
        widths = [*cfg.edge_channels_list, self.num_channels_rad]
        layers: list[nn.Module] = []
        for i in range(1, len(widths)):
            layers.append(nn.Linear(widths[i - 1], widths[i], bias=True))
            if i != len(widths) - 1:
                layers.append(nn.LayerNorm(widths[i]))
                layers.append(nn.SiLU())
        self.rad_func = nn.Sequential(*layers)
        self.edge_split = [in_m0_1, *in_m_1]

        # --- conv1: radial-modulated, emits the gate scalars in its m=0 head ------------
        self.extra_m0 = L * H  # act_type="gate"
        self.c1_m0 = nn.Linear(in_m0_1, H * (L + 1) + self.extra_m0, bias=True)
        self.c1_m = nn.ModuleList(
            [nn.Linear(in_m_1[m - 1], 2 * H * self.layout.m_size[m], bias=False)
             for m in range(1, cfg.mmax + 1)]
        )
        for lin in self.c1_m:
            lin.weight.data.mul_(1 / math.sqrt(2))

        # --- conv2: internal weights, no radial ----------------------------------------
        in_m0_2 = (L + 1) * H
        self.c2_m0 = nn.Linear(in_m0_2, C * (L + 1), bias=True)
        self.c2_m = nn.ModuleList(
            [nn.Linear(self.layout.m_size[m] * H, 2 * C * self.layout.m_size[m], bias=False)
             for m in range(1, cfg.mmax + 1)]
        )
        for lin in self.c2_m:
            lin.weight.data.mul_(1 / math.sqrt(2))

        # --- gate + envelope + readout --------------------------------------------------
        self.register_buffer(
            "expand_index", self.layout.gate_expand_index("cpu"), persistent=False
        )
        # `to_m` is a constant permutation. Building it per forward (a Python loop of
        # single-element assignments) costs one kernel launch per row every step, which
        # would make B1 look slower than eager actually is -- the anti-gaming rule that
        # baselines are never crippled cuts this way too. fairchem also keeps it as a
        # buffer.
        self.register_buffer(
            "to_m_matrix", self.layout.to_m("cpu", torch.get_default_dtype()),
            persistent=False,
        )
        p = float(cfg.envelope_exponent)
        self.env_p = p
        self.env_a = -(p + 1) * (p + 2) / 2
        self.env_b = p * (p + 2)
        self.env_c = -p * (p + 1) / 2

        # Rotation-invariant per-atom readout over ALL output rows.
        #
        # The obvious head -- a linear map on the l=0 row -- is what eSEN uses, but eSEN
        # applies it after a *stack* of blocks. On a single block it is degenerate: the
        # l=0 output row is fed only by the m=0 branch, so all four m>0 convolution
        # weight tensors receive exactly zero gradient and the SO(2) machinery under
        # test goes dead. Verified: c1_m.{0,1}.weight and c2_m.{0,1}.weight all had
        # |grad| == 0 with that head.
        #
        # Instead: l=0 enters linearly (already invariant) and each l>0 block enters
        # through its squared norm sum_m x[l,m,c]^2, which is invariant because the
        # Wigner-D blocks are orthogonal. Every output row is live, every parameter gets
        # a gradient, and E is still exactly rotation-invariant. See DECISIONS.md D9.
        self.readout = nn.Sequential(
            nn.Linear(C * (L + 1), C, bias=True), nn.SiLU(), nn.Linear(C, 1, bias=True)
        )

    # -- pieces ------------------------------------------------------------------------

    def envelope(self, d_scaled: torch.Tensor) -> torch.Tensor:
        """Polynomial cutoff envelope (exponent 5). Smooth to 0 at d == cutoff."""
        p = self.env_p
        val = 1 + (d_scaled**p) * (self.env_a + d_scaled * (self.env_b + self.env_c * d_scaled))
        return torch.where(d_scaled < 1, val, torch.zeros_like(val))

    def gaussian_basis(self, dist: torch.Tensor) -> torch.Tensor:
        d = dist.view(-1, 1) - self.gauss_offset.view(1, -1)
        return torch.exp(self.gauss_coeff * d * d)

    def _so2_conv(
        self,
        x: torch.Tensor,
        lin_m0: nn.Linear,
        lin_m: nn.ModuleList,
        c_out: int,
        radial: torch.Tensor | None,
        extra_m0: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Per-m block contractions in m'-major order.

        `radial` is the per-edge diagonal gain (conv1 only); it scales the *inputs*, and
        the same real gain is broadcast over the (real, imaginary) pair. So each m block
        is `out_m = W_m @ diag(r_m(e)) @ x_m(e)` -- trilinear in (weights, radial, feats),
        which is what makes the double backward non-trivial.
        """
        e = x.shape[0]
        by_m = x.split(self.layout.m_split, dim=1)
        rad_m = radial.split(self.edge_split, dim=1) if radial is not None else None

        x0 = by_m[0].reshape(e, -1)
        if rad_m is not None:
            x0 = x0 * rad_m[0]
        y0 = lin_m0(x0)

        gate = None
        if extra_m0:
            gate, y0 = y0.split((extra_m0, y0.shape[-1] - extra_m0), dim=-1)
        out = [y0.view(e, -1, c_out)]

        for m in range(1, self.cfg.mmax + 1):
            xm = by_m[m].reshape(e, 2, -1)
            if rad_m is not None:
                xm = xm * rad_m[m].unsqueeze(1)
            ym = lin_m[m - 1](xm)  # [E, 2, 2*half]
            half = ym.shape[-1] // 2
            r0, i0, r1, i1 = ym.reshape(e, 4, half).split(1, dim=1)
            # complex product: (W1 + i W2)(x0 + i x1)
            out.append((r0 - i1).view(e, -1, c_out))
            out.append((r1 + i0).view(e, -1, c_out))

        return torch.cat(out, dim=1), gate

    def gate_activation(self, gate: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """SiLU on the l=0 scalar row; sigmoid(gate) multiplies every other row."""
        e = gate.shape[0]
        g = torch.sigmoid(gate).view(e, self.cfg.lmax, self.cfg.hidden_channels)
        g = torch.index_select(g, dim=1, index=self.expand_index)
        scalars, vectors = x.split((1, x.shape[1] - 1), dim=1)
        return torch.cat([torch.nn.functional.silu(scalars), vectors * g], dim=1)

    # -- forward -----------------------------------------------------------------------

    def forward(
        self,
        pos: torch.Tensor,          # [N, 3]  (differentiated w.r.t.)
        atomic_numbers: torch.Tensor,  # [N]
        x_node: torch.Tensor,       # [N, K, C]  incoming node irreps
        edge_index: torch.Tensor,   # [2, E]  (row 0 = source, row 1 = target)
        shifts: torch.Tensor,       # [E, 3]  PBC image offsets, already in cartesian
        cos_gamma_k: torch.Tensor,  # [E, lmax+1]  seeded roll-angle harmonics
        sin_gamma_k: torch.Tensor,  # [E, lmax+1]
        jd: list[torch.Tensor],
    ) -> torch.Tensor:
        cfg = self.cfg
        src, dst = edge_index[0], edge_index[1]

        # geometry -> the only path position-dependence enters
        edge_vec = pos[dst] - pos[src] + shifts
        dist = edge_vec.pow(2).sum(-1).clamp_min(1e-24).sqrt()

        x_edge_in = torch.cat(
            [self.gaussian_basis(dist), self.elem_emb(atomic_numbers[src]),
             self.elem_emb(atomic_numbers[dst])], dim=-1
        )
        radial = self.rad_func(x_edge_in)

        wigner = wigner_from_edge_vec(edge_vec, cos_gamma_k, sin_gamma_k, cfg.lmax, jd)
        to_m = self.to_m_matrix.to(dtype=wigner.dtype)
        rot = torch.einsum("mk,ekj->emj", to_m, wigner)  # fused l->m' reorder + rotation

        # message
        msg = torch.cat([x_node[src], x_node[dst]], dim=2)  # [E, K, 2C]
        msg = torch.bmm(rot, msg)
        msg, gate = self._so2_conv(
            msg, self.c1_m0, self.c1_m, cfg.hidden_channels, radial, self.extra_m0
        )
        msg = self.gate_activation(gate, msg)
        msg, _ = self._so2_conv(
            msg, self.c2_m0, self.c2_m, cfg.sphere_channels, None, 0
        )
        msg = msg * self.envelope(dist / cfg.cutoff).view(-1, 1, 1)
        msg = torch.bmm(rot.transpose(1, 2), msg)  # rotate back

        node_out = torch.zeros(
            (x_node.shape[0], *msg.shape[1:]), dtype=msg.dtype, device=msg.device
        )
        node_out.index_add_(0, dst, msg)

        return self.readout(self.node_invariants(node_out)).sum()

    def node_invariants(self, node_out: torch.Tensor) -> torch.Tensor:
        """Per-l rotation invariants, [N, (lmax+1)*C].

        l = 0 passes through (already invariant); l > 0 contributes its squared norm
        over m, which is invariant because each Wigner-D block is orthogonal.
        """
        feats = [node_out[:, 0, :]]
        for l in range(1, self.cfg.lmax + 1):
            feats.append(node_out[:, l * l : (l + 1) ** 2, :].pow(2).sum(dim=1))
        return torch.cat(feats, dim=-1)


def conservative_training_step(
    block: ESO2RefBlock,
    batch: dict,
    jd: list[torch.Tensor],
    w_e: float = 1.0,
    w_f: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The measured unit: E, F = -dE/dpos (create_graph=True), then loss backward.

    Returns (E, F, L). The caller runs L.backward() to populate parameter grads; that
    second backward through F is the true double backward this milestone is about.
    """
    pos = batch["pos"]
    e = block(
        pos, batch["atomic_numbers"], batch["x_node"], batch["edge_index"],
        batch["shifts"], batch["cos_gamma_k"], batch["sin_gamma_k"], jd,
    )
    (f,) = torch.autograd.grad(e, pos, create_graph=True)
    f = -f
    loss = w_e * (e - batch["e_ref"]).pow(2).mean() + w_f * (f - batch["f_ref"]).pow(2).mean()
    return e, f, loss
