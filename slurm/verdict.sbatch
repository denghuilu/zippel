#!/bin/bash -l
#SBATCH --job-name=zippel-verdict
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
#SBATCH --output=slurm/logs/verdict_%A_%a.out
#
# Verdict-table protocol (REPORT.md 5.x): N = 5 INDEPENDENT allocations, not 5 repeats inside
# one. Repeats inside one allocation cannot see between-node and between-placement variance,
# which is exactly the variance that made si_small differ by 80% across two clean runs.
#
# Reported statistic: median-of-medians across the 5 allocations, plus the FULL RANGE. The
# range is the honest error bar -- an IQR within one allocation understates it.
#
# CPU/NUMA pinning: GH200 nodes present 4 GPUs across 4 NUMA domains. An unpinned process can
# land on a socket that is not local to the GPU it drives, and for a dispatch-bound workload
# every launch then crosses the interconnect. numactl binds us to the domain owning GPU 0.

set -euo pipefail
cd "${ZIPPEL_ROOT:-/iopsstor/scratch/cscs/dlu/iclr/zippel}"
mkdir -p slurm/logs bench/results

source env.sh
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16

REP="${SLURM_ARRAY_TASK_ID:-1}"
OUT="bench/results/verdict_rep${REP}_${SLURM_JOB_ID}.json"

echo "=== rep $REP  node $(hostname)  job ${SLURM_JOB_ID} ==="
numactl --hardware | head -3 || true
nvidia-smi --query-gpu=index,name,clocks.max.sm,power.limit --format=csv,noheader

# Bind to the NUMA domain local to GPU 0. `numactl -N 0 -m 0` is the fallback when the
# topology query is unavailable; both are recorded in the log so placement is auditable.
PIN=""
if command -v numactl >/dev/null 2>&1; then
  PIN="numactl --cpunodebind=0 --membind=0"
  echo "pinning: $PIN"
else
  echo "pinning: numactl unavailable -- placement NOT controlled for this rep"
fi

$PIN python baselines/b1_eager.py \
    --fixtures si_small si_medium cu_medium \
    --precisions fp32 bf16 --out "$OUT"

echo "=== rep $REP done -> $OUT ==="
