#!/bin/bash -l
#SBATCH --job-name=zippel-bench
#SBATCH --account=lp16
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --output=slurm/logs/bench_%j.out
#
# Every *reported* number comes from here, on an exclusive node -- never from the shared
# login node, whose 4 GH200s are fine for pytest and development but are shared with
# other users and would put unattributable variance into the medians.
#
# Single GPU by design: M1 has no multi-GPU scope. The other three are left idle rather
# than packed, so nothing else on the node perturbs the measurement.

set -euo pipefail
cd "${ZIPPEL_ROOT:-/iopsstor/scratch/cscs/dlu/iclr/zippel}"
mkdir -p slurm/logs bench/results

source env.sh
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=16

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID:-none} ==="
nvidia-smi --query-gpu=index,name,driver_version,clocks.max.sm,power.limit \
           --format=csv,noheader
python -c "import torch;print('torch',torch.__version__,'| cc',torch.cuda.get_device_capability(0))"

bash bench/run_all.sh

echo "=== done ==="
