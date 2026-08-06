"""Shared plumbing for the B1-B4 baseline runners.

Precision policy is defined here once so every implementation in a given table row runs
under identical settings (a binding anti-gaming rule):

  fp32   strict fp32 -- TF32 explicitly OFF. The validation row.
  tf32   fp32 storage, TF32 matmuls allowed. A perf row.
  bf16   bf16 autocast + TF32 allowed, i.e. how fairchem would actually train. A perf row.

Baselines run at their recommended fast settings and are never crippled; where a setting
has to differ (e.g. an implementation cannot support a precision at all) that is recorded
as a `notes`/`error` string on the Measurement rather than silently worked around.
"""

from __future__ import annotations

import contextlib

import torch

PRECISIONS = ("fp32", "tf32", "bf16")


@contextlib.contextmanager
def precision_context(precision: str):
    """Set the global matmul policy for the duration of a measurement, then restore it."""
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}; expected one of {PRECISIONS}")
    prev = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = precision in ("tf32", "bf16")
    torch.backends.cudnn.allow_tf32 = precision in ("tf32", "bf16")
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
        torch.backends.cudnn.allow_tf32 = prev_cudnn


def autocast_for(precision: str):
    if precision == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def make_step(block, batch, jd, precision: str, w_e: float = 1.0, w_f: float = 1.0):
    """Build (step_fn, zero_grads_fn, liveness_fn) for one conservative training step.

    Everything that is not the measured computation -- grad zeroing, loss weights, target
    tensors -- is hoisted out of `step_fn` so the timed region is the same for everyone.
    """
    from bench.harness import assert_step_is_live

    pos = batch["pos"]
    params = [p for p in block.parameters() if p.requires_grad]

    def zero_grads():
        for p in params:
            p.grad = None
        pos.grad = None

    def step():
        with autocast_for(precision):
            e = block(
                pos, batch["atomic_numbers"], batch["x_node"], batch["edge_index"],
                batch["shifts"], batch["cos_gamma_k"], batch["sin_gamma_k"], jd,
            )
            (f,) = torch.autograd.grad(e, pos, create_graph=True)
            f = -f
            loss = (w_e * (e - batch["e_ref"]).pow(2).mean()
                    + w_f * (f - batch["f_ref"]).pow(2).mean())
        loss.backward()

    return step, zero_grads, lambda: assert_step_is_live(block, pos)


def load_block_and_batch(fixture: str, precision: str, device="cuda", seed=0):
    """Fixture + reference block at the dtype implied by `precision`.

    bf16 runs keep master weights and inputs in fp32 and rely on autocast, which is how
    AMP training actually works; only the fp32/tf32 rows change the storage dtype.
    """
    from blocks.eso2_ref import BlockConfig, ESO2RefBlock
    from fixtures.load import fixture_stats, load_batch

    cfg = BlockConfig()
    torch.manual_seed(seed)
    block = ESO2RefBlock(cfg).to(device, torch.float32)
    batch = load_batch(fixture, device, torch.float32, cfg)
    jd = [j.to(device=device, dtype=torch.float32)
          for j in torch.load("blocks/Jd.pt", weights_only=False)]
    return block, batch, jd, fixture_stats(fixture)
