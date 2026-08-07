# Source this before running anything in zippel.
#   source /iopsstor/scratch/cscs/dlu/iclr/zippel/env.sh
#
# EVERY compiler/JIT cache is pinned project-local, under the repo on /iopsstor.
#
# Why /iopsstor and not /capstor: /capstor/scratch/cscs/dlu is at 306.9% of its
# 1,000,000-inode quota with the grace period expired, so file *creation* there fails with
# EDQUOT (reproduced). Reads are unaffected, so the conda base, FlashSO2 and fairchem stay
# readable in place. See DECISIONS.md D1 and REPORT.md section 1.
#
# Why project-local and not just "somewhere on /iopsstor": the defaults land in $TMPDIR or
# /tmp, which Alps purges between jobs -> silent re-JIT that poisons both test time and
# benchmark numbers (the FlashSO2 failure mode). And a *shared* cache root (e.g. the
# ~/.bashrc TRITON_CACHE_DIR=/iopsstor/scratch/cscs/dlu/.cache/triton) is not
# self-contained: "reproducible from a clean clone" has to mean this repo's own cache.
#
# Override the root with ZIPPEL_CACHE_ROOT if you really want a shared cache.

export ZIPPEL_ROOT=/iopsstor/scratch/cscs/dlu/iclr/zippel
export ZIPPEL_ENV=/iopsstor/scratch/cscs/dlu/envs/zippel
export ZIPPEL_CACHE_ROOT="${ZIPPEL_CACHE_ROOT:-$ZIPPEL_ROOT/.jit-cache}"

# CUTE_DSL_CACHE_DIR is REAL and honoured. env_manager builds its option names by f-string
# interpolation (f"{prefix}_CACHE_DIR"), so the literal never appears in the source and a grep
# for it finds nothing -- which is how this line got labelled a no-op for a while. The cache does
# not populate for us because `cute.compile()` sets no_cache=True by design; that is our choice
# of API, not a broken variable. See findings/cute-dsl-cache-dir-is-a-noop.md.
export CUTE_DSL_CACHE_DIR="$ZIPPEL_CACHE_ROOT/cute_dsl"
export TRITON_CACHE_DIR="$ZIPPEL_CACHE_ROOT/triton"
export QUACK_CACHE_DIR="$ZIPPEL_CACHE_ROOT/quack"
export TORCHINDUCTOR_CACHE_DIR="$ZIPPEL_CACHE_ROOT/inductor"
export PYTHONPYCACHEPREFIX="$ZIPPEL_CACHE_ROOT/pycache"
# TMPDIR is the DEFAULT root for the DSL's generated-IR cache when CUTE_DSL_CACHE_DIR is unset,
# and is used by everything else that respects it. Keeping it project-local is right regardless;
# it is not, as an earlier comment here claimed, what makes the JIT cache work.
export TMPDIR="$ZIPPEL_CACHE_ROOT/tmp"
mkdir -p "$CUTE_DSL_CACHE_DIR" "$TRITON_CACHE_DIR" "$QUACK_CACHE_DIR" \
         "$TORCHINDUCTOR_CACHE_DIR" "$PYTHONPYCACHEPREFIX" "$TMPDIR"

# Inductor's parallel compile-worker pool deadlocks on this contended login node (parent
# blocked in do_wait, zero output, GPU idle). Single-threaded compilation is slower but
# deterministic, and compile time is outside the measured region regardless.
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"

# CUDA_CACHE_PATH is deliberately NOT redirected. Alps points it at /dev/shm (node-local
# RAM). It caches driver-level PTX->SASS JIT, which our kernels do not trigger: CuTe DSL
# emits sm_90a cubin through NVRTC directly. It is the one cache not on /iopsstor; noted
# here rather than changed, because moving it onto (currently degraded) Lustre would cost
# real time for no reproducibility gain. Flagged in REPORT.md section 1.

# Never let ~/.local/lib site-packages shadow the env (Alps /users is a separate quota).
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ZIPPEL_ROOT${PYTHONPATH:+:$PYTHONPATH}"

source /capstor/scratch/cscs/dlu/miniforge3/etc/profile.d/conda.sh
conda activate "$ZIPPEL_ENV"
