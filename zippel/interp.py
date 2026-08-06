"""Reference interpreter: defines the IR's semantics.

**CPU, FP64, deterministic.** Deliberately not GPU: `index_add_` on CUDA is nondeterministic
(atomics reorder), and TF32 can leak into matmuls. Both would make this a fuzzy oracle, and
the whole point of an oracle is that it is not fuzzy.

This is an oracle, not a contender -- its timings never appear in a benchmark table.
"""

from __future__ import annotations

import math

import torch

from zippel.ir import BufferType, IndexType, Op, Program

DTYPE = torch.float64
DEVICE = "cpu"


# ----------------------------------------------------------------------------------------
# scalar_map
# ----------------------------------------------------------------------------------------

# Degree-5 polynomial cutoff envelope, matching blocks/eso2_ref.py exactly.
_P = 5.0
_A = -(_P + 1) * (_P + 2) / 2
_B = _P * (_P + 2)
_C = -_P * (_P + 1) / 2


def _envelope(d: torch.Tensor, order: int) -> torch.Tensor:
    """p(d) and its derivatives, zero for d >= 1.

    p(d)   = 1 + a d^5 + b d^6 + c d^7
    p'(d)  = 5a d^4 + 6b d^5 + 7c d^6
    p''(d) = 20a d^3 + 30b d^4 + 42c d^5

    C^2 at d = 1 by construction: p(1) = p'(1) = p''(1) = 0, which is what makes the double
    backward well-defined at the cutoff (tested in tests/test_ir_validation.py).
    """
    if order == 0:
        val = 1 + (d**5) * (_A + d * (_B + _C * d))
    elif order == 1:
        val = 5 * _A * d**4 + 6 * _B * d**5 + 7 * _C * d**6
    elif order == 2:
        val = 20 * _A * d**3 + 30 * _B * d**4 + 42 * _C * d**5
    elif order == 3:
        val = 60 * _A * d**2 + 120 * _B * d**3 + 210 * _C * d**4
    else:
        raise NotImplementedError(f"poly_envelope order {order} not implemented")
    return torch.where(d < 1, val, torch.zeros_like(val))


def apply_scalar(fn: str, x: torch.Tensor, order: int = 0) -> torch.Tensor:
    if fn == "exp":
        return torch.exp(x)
    if fn == "sigmoid":
        return torch.sigmoid(x)
    if fn == "silu":
        return torch.nn.functional.silu(x)
    if fn == "rsqrt":
        return torch.rsqrt(x)
    if fn == "reciprocal":
        return torch.reciprocal(x)
    if fn == "sin":
        return torch.sin(x)
    if fn == "cos":
        return torch.cos(x)
    if fn == "poly_envelope":
        return _envelope(x, order)
    raise ValueError(f"{fn!r} is outside vocabulary v1.1")


# ----------------------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------------------


def segment_length(seg: str, sizes: dict[str, int]) -> int:
    return 1 if seg == "none" else sizes[seg]


def eval_op(op: Op, env: dict[str, torch.Tensor], sizes: dict[str, int]) -> torch.Tensor:
    if op.kind == "scalar_map":
        return apply_scalar(op.fn, env[op.inputs[0]], op.order)

    if op.kind != "segmented_contraction":
        raise ValueError(f"unknown op kind {op.kind!r}")

    t = op.out_type
    n_out = segment_length(t.segment, sizes)
    out = torch.zeros((n_out, *t.sizes), dtype=DTYPE, device=DEVICE)

    # The segment length each path produces: the scatter target's length if scattering,
    # else the result's own length.
    for path in op.paths:
        operands = []
        for j, k in enumerate(path.operands):
            a = env[op.inputs[k]]
            imap = op.index_maps[k]
            if imap is not None:
                a = a[env[imap]]
            slices = path.slices_for(j)
            if slices:
                a = a[(slice(None), *slices)]
            operands.append(a)

        # broadcast any length-1 ('none'-segment) operand to the contribution length
        lengths = [a.shape[0] for a in operands if a.shape[0] != 1]
        n_c = lengths[0] if lengths else (n_out if op.out_index_map is None else 1)
        if any(l != n_c for l in lengths):
            raise ValueError(f"operand segment lengths disagree: {lengths}")
        operands = [a.expand(n_c, *a.shape[1:]) if a.shape[0] == 1 and n_c != 1 else a
                    for a in operands]

        # 'Z' batches the segment axis; it is reserved and never appears in a user subscript.
        specs, out_spec = path.parse()
        sub = ",".join("Z" + s for s in specs) + "->Z" + out_spec
        contrib = path.coeff * torch.einsum(sub, *operands)

        if op.out_index_map is not None:
            idx = env[op.out_index_map]
            target = out[(slice(None), *path.out_slice)] if path.out_slice else out
            target.index_add_(0, idx, contrib)
        else:
            target = out[(slice(None), *path.out_slice)] if path.out_slice else out
            if contrib.shape[0] == 1 and target.shape[0] != 1:
                contrib = contrib.expand_as(target)
            target += contrib
    return out


def run(prog: Program, inputs: dict[str, torch.Tensor],
        sizes: dict[str, int]) -> dict[str, torch.Tensor]:
    """Evaluate a program. `sizes` gives the dynamic segment lengths (node/edge/graph)."""
    env: dict[str, torch.Tensor] = {}
    for name, t in prog.inputs.items():
        if name not in inputs:
            raise KeyError(f"missing input {name!r}")
        a = inputs[name]
        if isinstance(t, IndexType):
            env[name] = a.to(device=DEVICE, dtype=torch.long)
            continue
        a = a.to(device=DEVICE, dtype=DTYPE)
        want = (segment_length(t.segment, sizes), *t.sizes)
        if tuple(a.shape) != want:
            raise ValueError(f"input {name!r} has shape {tuple(a.shape)}, expected {want}")
        env[name] = a

    for name in prog.topo():
        env[name] = eval_op(prog.ops[name], env, sizes)
    return env


def peak_live_bytes(prog: Program, sizes: dict[str, int], itemsize: int = 4) -> int:
    """Peak simultaneously-live buffer bytes under a naive topological order.

    No rematerialization, no fusion: a buffer is live from its definition to its last use.
    This is the memory-pressure baseline that Phase 2's joint scheduling has to beat, so it
    is deliberately the *unscheduled* number.
    """
    last_use: dict[str, int] = {}
    order = prog.topo()
    for i, name in enumerate(order):
        for src in prog.ops[name].inputs:
            last_use[src] = i
        for m in (*prog.ops[name].index_maps, prog.ops[name].out_index_map):
            if m:
                last_use[m] = i
    for out in prog.outputs:
        last_use[out] = len(order)

    def nbytes(t) -> int:
        if isinstance(t, IndexType):
            return segment_length(t.segment, sizes) * 8
        return segment_length(t.segment, sizes) * math.prod(t.sizes or (1,)) * itemsize

    live = {n: nbytes(t) for n, t in prog.inputs.items()}
    peak = sum(live.values())
    for i, name in enumerate(order):
        live[name] = nbytes(prog.ops[name].out_type)
        peak = max(peak, sum(live.values()))
        for buf, idx in list(last_use.items()):
            if idx == i and buf in live:
                del live[buf]
    return peak


__all__ = ["run", "eval_op", "apply_scalar", "peak_live_bytes", "segment_length", "DTYPE"]
