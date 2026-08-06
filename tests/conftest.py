"""Pins every JIT cache project-local *at import time*.

This runs before any kernel library is imported, so a bare `pytest` is as
cache-persistent as a run under `env.sh`.

Two failure modes this closes:
  * the defaults land in $TMPDIR / /tmp, which Alps purges between jobs -> silent re-JIT
    that poisons both test time and benchmark numbers (the FlashSO2 failure mode);
  * a *shared* cache root is not self-contained. `~/.bashrc` exports
    TRITON_CACHE_DIR=/iopsstor/scratch/cscs/dlu/.cache/triton, which is on /iopsstor but
    shared with other projects; "reproducible from a clean clone" has to mean this repo's
    own cache.

So these are set **unconditionally**, not via `setdefault` -- an inherited value would
otherwise silently win. Point ZIPPEL_CACHE_ROOT elsewhere if a shared cache is wanted.

/capstor/scratch/cscs/dlu is over its inode quota with grace expired, so nothing may be
written there (DECISIONS.md D1). CUDA_CACHE_PATH is deliberately left alone -- see env.sh.
"""

from __future__ import annotations

import os
import pathlib

import pytest
import torch

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_CACHE = pathlib.Path(os.environ.get("ZIPPEL_CACHE_ROOT", _ROOT / ".jit-cache"))

for _var, _sub in (
    ("CUTE_DSL_CACHE_DIR", "cute_dsl"),
    ("TRITON_CACHE_DIR", "triton"),
    ("QUACK_CACHE_DIR", "quack"),
    ("TORCHINDUCTOR_CACHE_DIR", "inductor"),
    ("PYTHONPYCACHEPREFIX", "pycache"),
    ("TMPDIR", "tmp"),
):
    _path = _CACHE / _sub
    _path.mkdir(parents=True, exist_ok=True)
    os.environ[_var] = str(_path)

os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

# The FP64 CPU interpreter runs many small einsums. On a 288-core node torch's default
# thread pool thrashes badly on them: the block ladder took 2m27s wall for 158 minutes of
# user time, versus 13s wall when capped. Cap unless the caller has an opinion.
os.environ.setdefault("OMP_NUM_THREADS", "8")
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)


@pytest.fixture(scope="session")
def jd64():
    """fairchem's J_d change-of-basis tensors, FP64 on device."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    raw = torch.load(_ROOT / "blocks" / "Jd.pt", weights_only=False)
    return [j.to(device=dev, dtype=torch.float64) for j in raw]


@pytest.fixture(scope="session")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"
