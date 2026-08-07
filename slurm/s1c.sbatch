#!/bin/bash -l
#SBATCH --job-name=zippel-s1c
#SBATCH --account=lp16
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --array=1-5
#SBATCH --output=slurm/logs/s1c_%A_%a.out
#
# S1c performance: fused forward vs eager, N = 5 INDEPENDENT allocations.
#
# THE SANITY ANCHOR, NOT THE BET. The store-elision lever D23 identifies lives in force and
# double-backward, where the intermediates are 3x and 9x the forward's. A forward-only number
# warrants neither despair nor a victory lap, and REPORT says so where it reports it.
#
# Protocol (unchanged from slurm/verdict.sbatch): five separate allocations rather than five
# repeats inside one, because repeats inside one allocation cannot see between-node and
# between-placement variance -- the variance that made si_small differ by 80 % across two clean
# runs. Reported statistic is median-of-medians across allocations plus the FULL RANGE; the
# range is the honest error bar. A speedup claim must exceed the measured spread for its
# configuration: ~2 % at the medium fixtures, ~4-5 % at the small ones.
#
# TIME LIMIT IS SIZED FOR COLD COMPILES. The CuTe DSL JIT cache does not work here (one 4 KB
# entry after three full passes, CUTE_DSL_CACHE_DIR set and visible to the child -- R8, under
# diagnosis). Whole-forward cold compile measured at 523.7 s, and this job compiles four
# configurations, so ~35 min of the budget is compile before a single timing sample is taken.
# 2 h leaves headroom without being so generous that a hang looks like progress.
#
# UNIFORM CONFIGURATION ACROSS ALL FIVE ALLOCATIONS: the term-minimal split, max_volume=10000
# (D36). Recorded into each result's provenance, because five allocations that silently
# disagreed about the partition would produce a spread that looks like hardware variance and is
# not.

set -euo pipefail
cd "${ZIPPEL_ROOT:-/iopsstor/scratch/cscs/dlu/iclr/zippel}"
source env.sh

REP="${SLURM_ARRAY_TASK_ID:-1}"
MAXVOL=10000                       # term-minimal split, identical in every allocation (D36)

echo "=== rep $REP  node $(hostname)  job ${SLURM_JOB_ID}  max_volume=$MAXVOL ==="
nvidia-smi --query-gpu=index,name,clocks.max.sm --format=csv,noheader | head -1
echo "pinning: numactl --cpunodebind=0 --membind=0"

# core dumps: a segfaulting kernel drops 4.6 GB in the repo root and filled the disk once
ulimit -c 0

for FIX in si_small si_medium; do
  for DT in f32 f64; do
    OUT="bench/results/s1c_rep${REP}_${SLURM_JOB_ID}_${FIX}_${DT}.json"
    echo "--- $FIX $DT -> $OUT ---"
    numactl --cpunodebind=0 --membind=0 \
      env CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 \
      python -u bench/s1c_bench.py \
        --fixture "$FIX" --dtype "$DT" --max-volume "$MAXVOL" --out "$OUT" \
      || echo "  FAILED: $FIX $DT (recorded as a gap, not retried)"
  done
done

echo "=== rep $REP done ==="
