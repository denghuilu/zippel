"""CuTe DSL smoke test for `tools/switch_env.sh`. Must be a FILE, not a heredoc.

CuTe DSL reads a decorated function's source with `inspect.getsourcelines`, so a kernel piped in
on stdin fails with "DSL does not support REPL mode, save the function to a file instead" -- the
same constraint that forces `codegen/emit.py` to write generated kernels to disk. The verify
script originally inlined this as a heredoc and hit it a second time.

Exits non-zero on any disagreement, so `set -e` in the caller does the right thing.
"""

from __future__ import annotations

import sys

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack


class Square:
    @cute.jit
    def __call__(self, mX: cute.Tensor, mO: cute.Tensor, n: Int32, stream):
        self.kernel(mX, mO, n).launch(grid=[4, 1, 1], block=[256, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, mX: cute.Tensor, mO: cute.Tensor, n: Int32):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        i = bidx * 256 + tidx
        if i < n:
            mO[i] = mX[i] * mX[i]


def main() -> int:
    if not torch.cuda.is_available():
        print("  FAIL: no CUDA device visible")
        return 1
    torch.manual_seed(0)
    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    o = torch.zeros_like(x)
    stream = cutlass.cuda.default_stream()
    args = (from_dlpack(x, assumed_align=16), from_dlpack(o, assumed_align=16),
            Int32(1024), stream)
    cute.compile(Square(), *args)(*args)
    torch.cuda.synchronize()

    err = (o - x * x).abs().max().item()
    print(f"  device={torch.cuda.get_device_name(0)}  torch={torch.__version__}  "
          f"max abs err={err:.3e}")
    if err != 0.0:
        print("  FAIL: CuTe DSL result disagrees with torch")
        return 1
    print("  CuTe DSL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
