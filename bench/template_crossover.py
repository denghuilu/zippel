"""Where does a dense MMA tile beat exploiting block-diagonal structure?

D22 rejected a dense-WGMMA design for lmax=2 using FlashSO2's measured curve plus a density
argument. That decision was right, but it was arithmetic applied at one point. The template
selection rule in `docs/templates.md` needs the *curve*: at what structural density does paying
for the zeros become cheaper than working around them, on this hardware?

The measured operation is the one that matters -- rotating per-edge coefficients by a
block-diagonal Wigner matrix, `[E, nc, nc] @ [E, nc, C]` -- under three strategies:

  dense-pad   one bmm on the tile padded to an MMA-friendly extent. What a dense MMA tile does:
              every structural zero is multiplied and accumulated.
  dense-exact one bmm at exactly nc, no padding. Isolates padding waste from block-diagonal
              waste; a real MMA tile cannot always do this, so it is a lower bound on dense.
  block       one bmm per degree block. Exploits the block-diagonal exactly, at the cost of
              lmax+1 smaller launches with worse per-call efficiency.

`block / dense-pad` is the speedup from exploiting structure. Above 1.0 structure wins and the
emitter should not use a dense tile; below 1.0 the dense tile is genuinely cheaper and the
selection rule should say so.

This measures library GEMMs, not our emitted kernels, deliberately: it isolates the *structural*
question from our codegen quality, so the crossover it reports is a property of the hardware and
the shape rather than of how good our emitter happens to be.

    python bench/template_crossover.py --edges 65536 --channels 128
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def pad_to(v: int, m: int) -> int:
    return ((v + m - 1) // m) * m


def block_sizes(lmax: int) -> list[int]:
    return [2 * ell + 1 for ell in range(lmax + 1)]


def time_ms(fn, warmup: int = 10, iters: int = 50) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


def measure(lmax: int, edges: int, channels: int, dtype: torch.dtype, mma_granularity: int):
    nc = (lmax + 1) ** 2
    nnz = sum(n * n for n in block_sizes(lmax))
    pad = pad_to(nc, mma_granularity)

    x = torch.randn(edges, nc, channels, device="cuda", dtype=dtype)
    w_dense = torch.zeros(edges, nc, nc, device="cuda", dtype=dtype)
    blocks, row = [], 0
    for n in block_sizes(lmax):
        blk = torch.randn(edges, n, n, device="cuda", dtype=dtype)
        w_dense[:, row:row + n, row:row + n] = blk
        blocks.append((row, n, blk.contiguous()))
        row += n

    w_pad = torch.zeros(edges, pad, pad, device="cuda", dtype=dtype)
    w_pad[:, :nc, :nc] = w_dense
    x_pad = torch.zeros(edges, pad, channels, device="cuda", dtype=dtype)
    x_pad[:, :nc] = x

    out_pad = torch.empty(edges, pad, channels, device="cuda", dtype=dtype)
    out = torch.empty(edges, nc, channels, device="cuda", dtype=dtype)

    def dense_pad():
        torch.bmm(w_pad, x_pad, out=out_pad)

    def dense_exact():
        torch.bmm(w_dense, x, out=out)

    def block_wise():
        for r, n, blk in blocks:
            out[:, r:r + n] = torch.bmm(blk, x[:, r:r + n])

    t_pad = time_ms(dense_pad)
    t_exact = time_ms(dense_exact)
    t_block = time_ms(block_wise)

    return {
        "lmax": lmax, "nc": nc, "nnz": nnz, "pad": pad,
        "density_vs_pad": nnz / (pad * pad),
        "density_vs_exact": nnz / (nc * nc),
        "ms_dense_pad": t_pad, "ms_dense_exact": t_exact, "ms_block": t_block,
        "block_over_dense_pad": t_pad / t_block,
        "block_over_dense_exact": t_exact / t_block,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", type=int, default=65536)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--lmax", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8])
    ap.add_argument("--granularity", type=int, default=16,
                    help="MMA N/K granularity the dense tile must pad to (16 is generous to "
                         "the dense side; FlashSO2's WGMMA path used 32)")
    ap.add_argument("--out", default="bench/results/template_crossover.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a GPU")

    rows = []
    for name, dt in (("bf16", torch.bfloat16), ("fp32", torch.float32)):
        print(f"\n=== {name}, E={args.edges}, C={args.channels}, "
              f"dense tile padded to a multiple of {args.granularity} ===")
        print(f"{'lmax':>4} {'nc':>4} {'nnz':>5} {'pad':>4} {'density':>8} "
              f"{'dense-pad':>10} {'dense-exact':>12} {'block':>9} {'block wins by':>14}")
        for lmax in args.lmax:
            r = measure(lmax, args.edges, args.channels, dt, args.granularity)
            r["dtype"] = name
            rows.append(r)
            print(f"{r['lmax']:>4} {r['nc']:>4} {r['nnz']:>5} {r['pad']:>4} "
                  f"{r['density_vs_pad']:>7.1%} {r['ms_dense_pad']:>9.3f}ms "
                  f"{r['ms_dense_exact']:>11.3f}ms {r['ms_block']:>8.3f}ms "
                  f"{r['block_over_dense_pad']:>13.2f}x")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"edges": args.edges, "channels": args.channels,
                               "granularity": args.granularity, "rows": rows}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
