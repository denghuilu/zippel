"""B3 -- cuEquivariance fused segmented-polynomial baseline.

Survey result (work order section 4.3): **yes, the API can express this block.**
`cuequivariance.group_theory.experimental.escn.escn_tp_compact` builds a
`SegmentedPolynomial` whose paths are exactly eSEN's per-m complex contraction. Verified
against the descriptor itself for lmax = mmax = 2:

    operand 1 (input)  segments  m = -2,-1,0,+1,+2  sizes (C, 2C, 3C, 2C, C)
    operand 2 (output) segments  same ordering
    operand 0 (weights) blocks   [m=0], [|m|=1 W1], [|m|=1 W2], [|m|=2 W1], [|m|=2 W2]
    paths   (0, m0,  m0 )  c=+1/sqrt(u)
            (W1, -m, -m )  c=+1/sqrt(2u)     (W1, +m, +m)  c=+1/sqrt(2u)
            (W2, +m, -m )  c=+1/sqrt(2u)     (W2, -m, +m)  c=-1/sqrt(2u)

which is `out_r = W1 x_r - W2 x_i`, `out_i = W1 x_i + W2 x_r` with cuEq's `+m` segment
holding the real part and `-m` the imaginary part. The `1/sqrt(...)` factors come from
`normalize_paths_for_operand(2)`; they are known exactly, so our weights map onto cuEq's
by an exact rescale and B3 can be *numerically validated*, not merely benchmarked.

**Outcome: a coverage gap. No backend is both correct and scalable for eSEN's weight
sharing**, so B3 contributes no valid wall-clock number. Measured, not assumed
(si_small, 9 620 edges, peak allocation for one forward of the conv1-shaped descriptor;
relative error against the plain-torch contraction built from the same weights):

    backend                        peak       rel err
    fused_tp + input_indices      24.18 GB    1.63e-15   exact, but densifies
    fused_tp + expanded           24.18 GB    1.63e-15
    indexed_linear + indices       0.25 GB    0.665      scales, but WRONG
    indexed_linear + expanded      --         KeyError (index map required, by design)
    uniform_1d                     --         cannot run: rejects the 2-D `uv` operand

The cause is architectural. `escn_tp_compact` descends from **eSCN**, where the radial
network emits a *per-edge* weight matrix, so operand 0 is a batched input and `fused_tp`
densifies it to [E, 622592] however it is called. **eSEN instead shares W across edges**
and varies only a per-edge diagonal gain.

`indexed_linear` is the backend built for a shared weight table, and it does scale --
but it does not reproduce `escn_tp_compact`'s semantics. The obvious explanation, a
weight-block memory-order convention, was tested and ruled out: sweeping both packings
gives 0.712 ((u,v)) and 1.210 ((v,u)) for `indexed_linear`, while `fused_tp` gives
6.13e-07 for (u,v) -- confirming our packing is the right one. This does not prove the
backend can never serve this workload; it may need a differently *constructed* descriptor
rather than a differently *packed* operand. That needs cuEquivariance internals and is
time-boxed out of M1.

One platform constraint: the `cuequivariance-ops-torch-cu13==0.11.0` extension (newest
published) does not load against torch 2.13; it needs **torch <= 2.11.0**. Without it
every method silently degrades to `naive`. B3 therefore runs in its own interpreter, and
this script also measures an eager control in the *same* interpreter so the B3/eager
ratio is internally consistent.

Scope: cuEquivariance covers the two per-m contractions. The Wigner rotation, the gate,
the envelope, the gather/scatter and the readout have no cuEq equivalent and stay eager --
that *is* the "cuEquivariance + autograd per-operator stack" the bet compares against.

    $SPIR_CUEQ_PY baselines/b3_cueq.py --validate
    $SPIR_CUEQ_PY baselines/b3_cueq.py --fixtures si_medium
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import torch
import torch.nn as nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def _irreps(channels: int, lmax: int):
    import cuequivariance as cue

    return cue.Irreps("SO3", "+".join(f"{channels}x{l}" for l in range(lmax + 1)))


class CueqSO2Conv(nn.Module):
    """eSEN's per-m SO(2) contraction, executed by a cuEquivariance fused polynomial.

    Holds weights in *our* layout (`[out, in]` per m, W1/W2 stacked) and converts to
    cuEq's flat weight operand on the fly, so the parameters a training step updates are
    the same tensors in both baselines.
    """

    def __init__(self, c_in: int, c_out: int, lmax: int, mmax: int,
                 method: str = "indexed_linear", math_dtype=torch.float32):
        super().__init__()
        import cuequivariance_torch as cuet
        from cuequivariance.group_theory.experimental.escn import escn_tp_compact

        self.lmax, self.mmax = lmax, mmax
        self.c_in, self.c_out = c_in, c_out
        self.poly_desc = escn_tp_compact(_irreps(c_in, lmax), _irreps(c_out, lmax), m_max=mmax)
        self.poly = cuet.SegmentedPolynomial(
            self.poly_desc, method=method, math_dtype=math_dtype
        )
        self.method = method

        # our-layout parameters, one per m
        self.w_m0 = nn.Parameter(torch.empty(c_out * (lmax + 1), c_in * (lmax + 1)))
        self.w_m = nn.ParameterList([
            nn.Parameter(torch.empty(2 * c_out * (lmax - m + 1), c_in * (lmax - m + 1)))
            for m in range(1, mmax + 1)
        ])
        for p in (self.w_m0, *self.w_m):
            nn.init.normal_(p, std=(1.0 / math.sqrt(p.shape[1])))

        self._path_coeffs = self._extract_path_coeffs()
        self._w_index: torch.Tensor | None = None

    def _extract_path_coeffs(self) -> list[float]:
        """One coefficient per weight block, read off the descriptor (never assumed)."""
        coeffs: dict[int, float] = {}
        for _op, stp in self.poly_desc.operations:
            for path in stp.paths:
                w_idx = int(path.indices[0])
                coeffs.setdefault(w_idx, abs(float(path.coefficients)))
        return [coeffs[i] for i in range(len(coeffs))]

    def _cueq_weights(self) -> torch.Tensor:
        """Pack our [out, in] weights into cuEq's flat operand, undoing normalisation.

        cuEq computes `out[v] = c * sum_u weight[u, v] * x[u]`; we want
        `out[v] = sum_u W[v, u] * x[u]`, so `weight = W.T / c`.
        """
        c = self._path_coeffs
        blocks = [(self.w_m0 / c[0]).T.reshape(-1)]
        k = 1
        for m_idx, w in enumerate(self.w_m):
            half = w.shape[0] // 2
            for part in (w[:half], w[half:]):  # W1 then W2
                blocks.append((part / c[k]).T.reshape(-1))
                k += 1
        return torch.cat(blocks)

    # -- layout conversion: our m'-major [E, K, C] <-> cuEq flat segments -------------

    def _to_cueq(self, x: torch.Tensor) -> torch.Tensor:
        """m'-major [m0 | re1 | im1 | re2 | im2] -> cuEq [m=-2, -1, 0, +1, +2]."""
        e = x.shape[0]
        n0 = self.lmax + 1
        m0 = x[:, :n0, :].reshape(e, -1)
        re, im = [], []
        off = n0
        for m in range(1, self.mmax + 1):
            k = self.lmax - m + 1
            re.append(x[:, off:off + k, :].reshape(e, -1))
            im.append(x[:, off + k:off + 2 * k, :].reshape(e, -1))
            off += 2 * k
        return torch.cat([*reversed(im), m0, *re], dim=1)

    def _from_cueq(self, y: torch.Tensor) -> torch.Tensor:
        """Inverse of `_to_cueq`, back to m'-major [E, K, C_out]."""
        e = y.shape[0]
        sizes = ([self.c_out * (self.lmax - m + 1) for m in range(self.mmax, 0, -1)]
                 + [self.c_out * (self.lmax + 1)]
                 + [self.c_out * (self.lmax - m + 1) for m in range(1, self.mmax + 1)])
        parts = list(y.split(sizes, dim=1))
        n_m = self.mmax
        im = list(reversed(parts[:n_m]))
        m0 = parts[n_m]
        re = parts[n_m + 1:]
        out = [m0.reshape(e, -1, self.c_out)]
        for m in range(1, self.mmax + 1):
            out.append(re[m - 1].reshape(e, -1, self.c_out))
            out.append(im[m - 1].reshape(e, -1, self.c_out))
        return torch.cat(out, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = x.shape[0]
        # The weights are *shared* across edges (eSEN's per-edge variation is the radial
        # diagonal gain, already folded into x), so they are passed as a one-row table
        # plus an index map. The *method* matters as much as the call shape -- measured at
        # si_small (9 620 edges), peak allocation for one forward:
        #
        #     indexed_linear + input_indices   0.25 GB  (but 0.665 rel err -- WRONG)
        #     indexed_linear + expanded        KeyError (index map required, by design)
        #     fused_tp       + input_indices  24.18 GB
        #     fused_tp       + expanded       24.18 GB
        #
        # `fused_tp` densifies the weight operand to [E, 622592] whichever way it is
        # called, because escn_tp_compact descends from eSCN where the radial net emits a
        # *per-edge* weight matrix. `indexed_linear` is 97x smaller but does not reproduce
        # the descriptor's semantics (see the module docstring), so neither backend is both
        # correct and scalable. The default stays `indexed_linear` only so the shape of the
        # call is exercised; its numbers are marked NUMERICALLY INVALID in the results.
        w = self._cueq_weights().unsqueeze(0)
        if self._w_index is None or self._w_index.shape[0] != e or self._w_index.device != x.device:
            self._w_index = torch.zeros(e, dtype=torch.long, device=x.device)
        y = self.poly([w, self._to_cueq(x)], input_indices={0: self._w_index})[0]
        return self._from_cueq(y)


def reference_so2(x: torch.Tensor, conv: CueqSO2Conv) -> torch.Tensor:
    """The same contraction in plain torch, using the identical parameters."""
    e = x.shape[0]
    n0 = conv.lmax + 1
    out = [(x[:, :n0, :].reshape(e, -1) @ conv.w_m0.T).reshape(e, -1, conv.c_out)]
    off = n0
    for m_idx, w in enumerate(conv.w_m):
        k = conv.lmax - (m_idx + 1) + 1
        xr = x[:, off:off + k, :].reshape(e, -1)
        xi = x[:, off + k:off + 2 * k, :].reshape(e, -1)
        half = w.shape[0] // 2
        w1, w2 = w[:half], w[half:]
        out.append((xr @ w1.T - xi @ w2.T).reshape(e, -1, conv.c_out))
        out.append((xi @ w1.T + xr @ w2.T).reshape(e, -1, conv.c_out))
        off += 2 * k
    return torch.cat(out, dim=1)


def validate(device="cuda", dtype=torch.float64, method="indexed_linear") -> dict:
    """cuEq fused output vs the plain-torch contraction with identical weights."""
    torch.manual_seed(0)
    cfg_c_in, cfg_c_out, lmax, mmax = 256, 128, 2, 2
    conv = CueqSO2Conv(cfg_c_in, cfg_c_out, lmax, mmax,
                       method=method, math_dtype=dtype).to(device, dtype)
    x = torch.randn(64, (lmax + 1) ** 2, cfg_c_in, device=device, dtype=dtype)

    got = conv(x)
    want = reference_so2(x, conv)
    rel = ((got - want).abs().max() / want.abs().max()).item()

    # double backward through the fused op
    x.requires_grad_(True)
    y = conv(x)
    (g,) = torch.autograd.grad(y.square().sum(), x, create_graph=True)
    (gg,) = torch.autograd.grad(g.square().sum(), conv.w_m0)
    return {"shape_match": tuple(got.shape) == tuple(want.shape),
            "max_rel_err": rel, "path_coeffs": conv._path_coeffs,
            "dbwd_norm": float(gg.norm()), "method": conv.method}


def build_cueq_block(cfg, device, dtype, method="fused_tp"):
    """The reference block with both per-m contractions replaced by cuEq fused polynomials.

    Everything cuEquivariance does not cover -- Wigner rotation, gate, envelope,
    gather/scatter, radial MLP, readout -- stays eager. That mixture *is* the
    "cuEquivariance + autograd" per-operator stack under test.

    Two pieces of conv1 fall outside the escn descriptor and are kept as eager tensors:
    the m=0 bias, and the extra m=0 head that emits the gate scalars.
    """
    from blocks.eso2_ref import ESO2RefBlock

    block = ESO2RefBlock(cfg).to(device, dtype)
    c_in = 2 * cfg.sphere_channels

    conv1 = CueqSO2Conv(c_in, cfg.hidden_channels, cfg.lmax, cfg.mmax,
                        method=method, math_dtype=dtype).to(device, dtype)
    conv2 = CueqSO2Conv(cfg.hidden_channels, cfg.sphere_channels, cfg.lmax, cfg.mmax,
                        method=method, math_dtype=dtype).to(device, dtype)
    gate_head = nn.Linear(c_in * (cfg.lmax + 1), cfg.lmax * cfg.hidden_channels).to(device, dtype)
    m0_bias1 = nn.Parameter(torch.zeros(cfg.hidden_channels * (cfg.lmax + 1),
                                        device=device, dtype=dtype))
    m0_bias2 = nn.Parameter(torch.zeros(cfg.sphere_channels * (cfg.lmax + 1),
                                        device=device, dtype=dtype))

    def so2_cueq(x, conv, bias, radial, edge_split, want_gate):
        e = x.shape[0]
        if radial is not None:
            # eSEN's radial modulation is a per-edge diagonal gain on the *inputs*; fold
            # it into x so cuEq can use a single shared weight tensor per m.
            by_m = list(x.split(block.layout.m_split, dim=1))
            rad_m = radial.split(edge_split, dim=1)
            by_m[0] = by_m[0] * rad_m[0].reshape(e, -1, x.shape[-1])
            off = 1
            for m in range(1, cfg.mmax + 1):
                k = cfg.lmax - m + 1
                g = rad_m[m].reshape(e, 1, k, x.shape[-1])
                pair = by_m[off].reshape(e, 2, k, x.shape[-1]) * g
                by_m[off] = pair.reshape(e, 2 * k, x.shape[-1])
                off += 1
            x = torch.cat(by_m, dim=1)
        gate = gate_head(x[:, : cfg.lmax + 1, :].reshape(e, -1)) if want_gate else None
        y = conv(x)
        n0 = cfg.lmax + 1
        y = torch.cat([y[:, :n0, :] + bias.view(1, n0, -1), y[:, n0:, :]], dim=1)
        return y, gate

    block._cueq_parts = nn.ModuleList([conv1, conv2, gate_head])
    block._cueq_biases = nn.ParameterList([m0_bias1, m0_bias2])
    # The eager convolution weights are now dead -- cuEq owns those contractions. Drop
    # them, or the harness liveness check would (correctly) flag parameters that receive
    # no gradient, and `parameters()` would report a block that is not the one being run.
    for attr in ("c1_m0", "c1_m", "c2_m0", "c2_m"):
        delattr(block, attr)

    def forward(pos, atomic_numbers, x_node, edge_index, shifts, cos_g, sin_g, jd):
        from blocks.wigner import wigner_from_edge_vec

        src, dst = edge_index[0], edge_index[1]
        edge_vec = pos[dst] - pos[src] + shifts
        dist = edge_vec.pow(2).sum(-1).clamp_min(1e-24).sqrt()
        x_edge_in = torch.cat([block.gaussian_basis(dist), block.elem_emb(atomic_numbers[src]),
                               block.elem_emb(atomic_numbers[dst])], dim=-1)
        radial = block.rad_func(x_edge_in)

        wigner = wigner_from_edge_vec(edge_vec, cos_g, sin_g, cfg.lmax, jd)
        rot = torch.einsum("mk,ekj->emj", block.to_m_matrix.to(wigner.dtype), wigner)

        msg = torch.bmm(rot, torch.cat([x_node[src], x_node[dst]], dim=2))
        msg, gate = so2_cueq(msg, conv1, m0_bias1, radial, block.edge_split, True)
        msg = block.gate_activation(gate, msg)
        msg, _ = so2_cueq(msg, conv2, m0_bias2, None, None, False)
        msg = msg * block.envelope(dist / cfg.cutoff).view(-1, 1, 1)
        msg = torch.bmm(rot.transpose(1, 2), msg)

        node_out = torch.zeros((x_node.shape[0], *msg.shape[1:]),
                               dtype=msg.dtype, device=msg.device)
        node_out.index_add_(0, dst, msg)
        return block.readout(block.node_invariants(node_out)).sum()

    block.forward = forward
    return block


def run_timings(fixtures, precisions, method="fused_tp"):
    from baselines.common import make_step, precision_context
    from bench.harness import Measurement, time_training_step, assert_step_is_live
    from blocks.eso2_ref import BlockConfig, ESO2RefBlock
    from fixtures.load import fixture_stats, load_batch

    cfg = BlockConfig()
    results = []
    for fixture in fixtures:
        for precision in precisions:
            for label, builder in (
                (f"B3 cueq[{method}]", lambda: build_cueq_block(cfg, "cuda", torch.float32, method)),
                ("B3-control eager", lambda: ESO2RefBlock(cfg).to("cuda", torch.float32)),
            ):
                try:
                    with precision_context(precision):
                        torch.manual_seed(0)
                        blk = builder()
                        batch = load_batch(fixture, "cuda", torch.float32, cfg)
                        jd = [j.to("cuda", torch.float32)
                              for j in torch.load("blocks/Jd.pt", weights_only=False)]
                        step, zero, _ = make_step(blk, batch, jd, precision)
                        stats = fixture_stats(fixture)
                        m = time_training_step(
                            step, zero, label=label, fixture=fixture, precision=precision,
                            atoms=stats["atoms"], edges=stats["edges"],
                            liveness_fn=lambda: assert_step_is_live(blk, batch["pos"]),
                            notes=(f"torch {torch.__version__}" + (
                                " | NUMERICALLY INVALID: indexed_linear does not reproduce "
                                "escn_tp_compact semantics (0.665 rel err)"
                                if method == "indexed_linear" and "cueq" in label else
                                " | exact but densifies shared weights to [E, 622592]"
                                if "cueq" in label else "")),
                        )
                    del blk, batch, jd
                    torch.cuda.empty_cache()
                except Exception as exc:
                    m = Measurement(label, fixture, precision, float("nan"), float("nan"),
                                    float("nan"), float("nan"), float("nan"), 0, 0, 0,
                                    error=f"{type(exc).__name__}: {str(exc)[:160]}")
                    torch.cuda.empty_cache()
                results.append(m)
                status = m.error or (f"{m.median_ms:8.2f} ms  IQR {m.iqr_ms:6.2f}  "
                                     f"peak {m.peak_mem_gib:6.2f} GiB")
                print(f"{m.label:20s} {fixture:10s} {precision:5s}  {status}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--fixtures", nargs="*", default=["si_medium"])
    ap.add_argument("--precisions", nargs="*", default=["fp32"])
    ap.add_argument("--method", default="indexed_linear")
    ap.add_argument("--out", default="bench/results/b3_cueq.json")
    args = ap.parse_args()

    if args.validate:
        for method in ("indexed_linear", "fused_tp"):
            for dt in (torch.float64, torch.float32):
                try:
                    r = json.dumps(validate(dtype=dt, method=method), default=str)
                except Exception as exc:
                    r = f"FAIL {type(exc).__name__}: {str(exc)[:120]}"
                print(f"{method:16s} {str(dt):20s} {r}", flush=True)
        return

    results = run_timings(args.fixtures, args.precisions, args.method)
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([vars(m) for m in results], indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
