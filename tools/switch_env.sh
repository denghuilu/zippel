#!/usr/bin/env bash
# Verify the cloned `zippel` conda env reproduces the `spir` stack, then switch to it.
#
# Why a clone and not a rename: the env was created with `conda create -p <prefix>`, and a prefix
# env bakes its absolute path into shebangs and activation scripts under `bin/`. `mv` leaves a
# subtly broken env (pip, f2py, and every console-script entry point point at the old path), so
# the safe operation is clone-verify-switch.
#
# Why verify at all: D11 declined this rename partly because "renaming risks breaking the working
# torch 2.13 / CuTe DSL stack". That risk is real and this script is the answer to it -- the old
# env is not removed until the new one has run the full suite.
#
#   bash tools/switch_env.sh verify    # check the clone, change nothing
#   bash tools/switch_env.sh switch    # verify, then repoint env.sh and the docs
#   bash tools/switch_env.sh cleanup   # remove the old env, only after switch + a green run

set -euo pipefail

OLD=/iopsstor/scratch/cscs/dlu/envs/spir
NEW=/iopsstor/scratch/cscs/dlu/envs/zippel
ROOT=/iopsstor/scratch/cscs/dlu/iclr/zippel

verify() {
    echo "=== verifying $NEW against $OLD ==="
    [ -x "$NEW/bin/python" ] || { echo "FAIL: $NEW/bin/python missing (clone incomplete?)"; exit 1; }

    # 1. the stack the project actually depends on, version for version
    echo "--- versions ---"
    for env in "$OLD" "$NEW"; do
        PYTHONNOUSERSITE=1 "$env/bin/python" - <<'PY'
import importlib.metadata as m, sys, torch
pkgs = ["torch", "nvidia-cutlass-dsl", "fairchem-core", "e3nn", "ase", "vesin",
        "cuequivariance", "pytest"]
got = []
for p in pkgs:
    try:
        got.append(f"{p}={m.version(p)}")
    except Exception:
        got.append(f"{p}=absent")
print(f"  py{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} " + " ".join(got))
PY
    done

    # 2. shebangs must point at the NEW prefix, or the clone is cosmetic
    echo "--- shebang audit ---"
    # grep exits 1 when it finds nothing, and under `set -euo pipefail` that killed this script
    # precisely when the audit PASSED -- a verification that failed on success. The braces
    # matter: `grep || true | wc -l` parses as `grep || (true | wc -l)`, so wc never sees grep's
    # output when grep succeeds. Group the fallback with the grep, then pipe.
    stale=$({ grep -rl "^#\!$OLD" "$NEW/bin" 2>/dev/null || true; } | wc -l | tr -d '[:space:]')
    echo "  scripts still pointing at the old prefix: $stale"
    [ "$stale" -eq 0 ] || { echo "FAIL: clone did not rewrite shebangs"; exit 1; }

    # 3. the GPU stack actually loads and runs, which is what D11 was worried about.
    #    Runs a FILE, not a heredoc: CuTe DSL reads decorated functions with
    #    inspect.getsourcelines and rejects anything on stdin ("DSL does not support REPL mode").
    echo "--- CuTe DSL smoke ---"
    CUTE_DSL_CACHE_DIR="$ROOT/.jit-cache/cute_dsl_envcheck" \
    PYTHONNOUSERSITE=1 PYTHONPATH="$ROOT" "$NEW/bin/python" "$ROOT/tools/_env_smoke.py"

    # 4. the whole suite, in the new env
    echo "--- pytest in the new env ---"
    ( cd "$ROOT" && ulimit -c 0 && \
      CUTE_DSL_CACHE_DIR="$ROOT/.jit-cache/cute_dsl" PYTHONNOUSERSITE=1 PYTHONPATH="$ROOT" \
      "$NEW/bin/python" -m pytest tests/ -q -p no:warnings 2>&1 | tail -3 )
    echo "=== verified ==="
}

switch() {
    verify
    echo "=== repointing env.sh and docs ==="
    sed -i "s|$OLD|$NEW|g" "$ROOT/env.sh"
    grep -n "ZIPPEL_ENV" "$ROOT/env.sh"
    echo "Docs (REPORT.md, DECISIONS.md, PLAN.md) are edited by hand, not sed: PLAN.md records"
    echo "the env as it was CREATED and must keep saying so, with the rename noted separately."
}

cleanup() {
    [ -x "$NEW/bin/python" ] || { echo "refusing: new env missing"; exit 1; }
    grep -q "envs/zippel" "$ROOT/env.sh" || { echo "refusing: env.sh still points at the old env"; exit 1; }
    echo "removing $OLD"
    conda env remove -p "$OLD" -y
}

case "${1:-verify}" in
    verify)  verify ;;
    switch)  switch ;;
    cleanup) cleanup ;;
    *) echo "usage: $0 {verify|switch|cleanup}"; exit 2 ;;
esac
