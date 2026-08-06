"""Environment invariants that silently corrupt measurements if they drift.

These are cheap and have no GPU dependency, so they run everywhere.
"""

from __future__ import annotations

import os
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Every compiler/JIT cache that must be project-local and on persistent scratch.
CACHE_VARS = [
    "CUTE_DSL_CACHE_DIR",
    "TRITON_CACHE_DIR",
    "QUACK_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "PYTHONPYCACHEPREFIX",
    "TMPDIR",
]


@pytest.mark.parametrize("var", CACHE_VARS)
def test_jit_cache_is_on_persistent_scratch(var):
    """No cache may resolve to node-local storage or to the over-quota filesystem.

    `/tmp` and `/dev/shm` are purged between jobs -> silent re-JIT, which poisons both
    test time and benchmark numbers. `/capstor/scratch/cscs/dlu` is at 306.9% of its
    inode quota with grace expired, so writes there fail with EDQUOT (DECISIONS.md D1).
    """
    value = os.environ.get(var)
    assert value, f"{var} is unset; it would default to node-local storage"
    resolved = str(pathlib.Path(value).resolve())
    assert resolved.startswith("/iopsstor/"), f"{var}={resolved} is not on /iopsstor"
    assert not resolved.startswith("/capstor/"), f"{var}={resolved} is on the over-quota fs"
    for bad in ("/tmp/", "/dev/shm/", "/var/tmp/"):
        assert not resolved.startswith(bad), f"{var}={resolved} is node-local ({bad})"


@pytest.mark.parametrize("var", CACHE_VARS)
def test_jit_cache_is_project_local(var):
    """Caches live under this repo, so a clean clone reproduces without shared state.

    `~/.bashrc` exports a shared TRITON_CACHE_DIR; conftest overrides it unconditionally
    for exactly this reason. Honours ZIPPEL_CACHE_ROOT if set deliberately.
    """
    root = pathlib.Path(os.environ.get("ZIPPEL_CACHE_ROOT", ROOT / ".jit-cache")).resolve()
    resolved = pathlib.Path(os.environ[var]).resolve()
    assert str(resolved).startswith(str(root)), f"{var}={resolved} is outside {root}"


def test_repo_is_writable():
    """The whole point of the /capstor -> /iopsstor move: creating files must work."""
    probe = ROOT / ".jit-cache" / ".write_probe"
    probe.write_text("ok")
    assert probe.read_text() == "ok"
    probe.unlink()
