"""What does `lts__t_sectors_op_read` actually count? A calibration kernel with known traffic.

D64 flagged an unreconciled 5.4×: `lts__t_sectors_op_read.sum` reported 93.37e9 sectors (2.99 TB)
on `conv1_90` while global-load L1 misses accounted for only 17.36e9 (555 GB). Until that is
understood, **no total-traffic budget built on these counters is trustworthy**, which is why
levers (b) L2-persistence and (c) CTA-scheduling are held behind this experiment: both would be
*evaluated* by exactly the counters whose semantics are in question.

Timeboxed and deliberately minimal. Three torch kernels whose traffic is known by construction:

  read_only    `x.sum()`          reads N bytes, writes 4
  read_write   `y.copy_(x)`       reads N, writes N  -- NOT MEASURED: this is a DtoD memcpy,
                                    not a kernel, so ncu never sees it (D68)
  write_only   `y.fill_(1.0)`     reads 0, writes N   (isolates write-allocate)

For each, the ratio `measured_sectors x 32 B / known_bytes` is the metric's multiplier. The
question is answered by three numbers:

  * **read_only ≈ 1.0** → `lts__t_sectors_op_read` counts read sectors at face value, and
    `conv1_90`'s 5.4× is **real re-reading through L2** that the L1-miss counter does not capture.
    That would make the 2.99 TB a genuine quantity and worth budgeting against.
  * **read_only ≈ some fixed k > 1** → the metric counts per-slice or per-subrequest and must be
    divided by k before use. The 5.4× shrinks accordingly.
  * **write_only shows nonzero reads** → write-allocate is real here and part of `conv1_90`'s
    excess is the output buffer, not input re-reads.

Whatever the answer, it is recorded before it is used, and the D64 flag is resolved or kept
rather than quietly dropped. **[intervention]** — known traffic is varied and the response read.

    uenv run prgenv-gnu/25.6:v2 --view=default -- bash bench/run_counter_semantics.sh
"""

from __future__ import annotations

import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mib", type=int, default=512,
                    help="buffer size; must exceed L2 (60 MiB) by enough that reuse cannot "
                         "confound the reading -- 512 MiB is 8.5x L2")
    args = ap.parse_args()

    n = args.mib * 1024 * 1024 // 4
    x = torch.ones(n, dtype=torch.float32, device="cuda")
    y = torch.zeros(n, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    print(f"buffer {args.mib} MiB = {n:,} f32 elements; known bytes per pass:", flush=True)
    print(f"  read_only  read {args.mib} MiB, write ~0", flush=True)
    print(f"  read_write read {args.mib} MiB, write {args.mib} MiB", flush=True)
    print(f"  write_only read 0,             write {args.mib} MiB", flush=True)

    # warm up outside the profiled region so no compilation or allocation is counted
    x.sum(); y.copy_(x); y.fill_(1.0)
    torch.cuda.synchronize()

    print("--- profiled region begins ---", flush=True)
    torch.cuda.profiler.start()
    x.sum()
    torch.cuda.synchronize()
    y.copy_(x)
    torch.cuda.synchronize()
    y.fill_(1.0)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print("--- profiled region ends ---", flush=True)
    print("launch order: 0=read_only(sum) 1=read_write(copy_) 2=write_only(fill_)", flush=True)


if __name__ == "__main__":
    main()
