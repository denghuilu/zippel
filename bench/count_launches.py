"""Count *actual* CUDA kernel launches in one conservative training step.

The naive count -- summing `count` over key_averages() entries with device time -- double
counts, because an `aten::mm` row and the `sm90_xmma_gemm...` row it launched both carry
device time. Here we count only leaf kernel events reported by CUPTI.
"""

import sys

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, "/iopsstor/scratch/cscs/dlu/iclr/zippel")

from baselines.common import load_block_and_batch, make_step, precision_context

FIXTURE = sys.argv[1] if len(sys.argv) > 1 else "si_small"

with precision_context("fp32"):
    block, batch, jd, stats = load_block_and_batch(FIXTURE, "fp32")
    step, zero, _ = make_step(block, batch, jd, "fp32")
    for _ in range(5):
        zero()
        step()
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        zero()
        step()
        torch.cuda.synchronize()

# Leaf CUDA kernels: every FunctionEvent carries the kernels it launched.
n_kernels, kernel_time_us = 0, 0.0
for ev in prof.events():
    for k in getattr(ev, "kernels", []) or []:
        n_kernels += 1
        kernel_time_us += k.duration

# Cross-check: aten-level op count (the host-side dispatch count)
aten = [e for e in prof.key_averages() if e.key.startswith("aten::")]
n_aten = sum(e.count for e in aten)

print(f"fixture           : {FIXTURE}  ({stats['atoms']} atoms, {stats['edges']} edges)")
print(f"CUDA kernels      : {n_kernels}")
print(f"kernel time       : {kernel_time_us / 1e3:.2f} ms")
print(f"aten:: op calls   : {n_aten}")
print("\ntop aten ops by call count:")
for e in sorted(aten, key=lambda e: -e.count)[:10]:
    print(f"   {e.key:34s} n={e.count:5d}  dev={e.self_device_time_total / 1e3:7.2f} ms")
