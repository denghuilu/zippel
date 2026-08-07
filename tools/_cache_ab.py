"""Trivial-kernel cache A/B: does CuTe DSL reuse a compiled kernel across processes?

One canonical kernel in a stable file, compiled twice in two separate processes. If the second
process is much faster, the cache works and any failure to reuse OUR kernels is our fault -- an
emitted-source fingerprint that varies per run (paths, module names, a nonce). If the second is
equally slow, the cache does not reuse across processes for this workload at all.

    python tools/_cache_ab.py            # one compile, prints seconds
"""
from __future__ import annotations
import sys, time
import torch, cutlass, cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack


class Canonical:
    @cute.jit
    def __call__(self, mX: cute.Tensor, mO: cute.Tensor, n: Int32, stream):
        self.kernel(mX, mO, n).launch(grid=[8, 1, 1], block=[256, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mO: cute.Tensor, n: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * 256 + tidx
        if i < n:
            a = mX[i]
            # enough work that compilation is measurable but the kernel stays canonical
            for _ in range(1):
                a = a * a + mX[i]
            mO[i] = a


def main() -> int:
    x = torch.randn(2048, device="cuda", dtype=torch.float32)
    o = torch.zeros_like(x)
    st = cutlass.cuda.default_stream()
    args = (from_dlpack(x, assumed_align=16), from_dlpack(o, assumed_align=16), Int32(2048), st)
    t0 = time.perf_counter()
    compiled = cute.compile(Canonical(), *args)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    compiled(*args)
    torch.cuda.synchronize()
    print(f"compile_seconds={dt:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
