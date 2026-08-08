#!/bin/bash
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/iopsstor/scratch/cscs/dlu/envs/zippel/bin/python
OUT=bench/results/ncu_l2_persist
mkdir -p bench/results logs
CUDA_VISIBLE_DEVICES=0 ncu --profile-from-start off --replay-mode kernel \
  --metrics dram__bytes.sum,dram__bytes_read.sum,dram__bytes_write.sum,\
lts__t_sector_hit_rate.pct,sm__warps_active.avg.pct_of_peak_sustained_active,\
gpu__time_duration.sum,dram__bytes.sum.per_second \
  --export "$OUT" --force-overwrite \
  "$PY" -u bench/l2_persist.py --ncu --configs "0:0;8:1" --out bench/results/l2_persist_ncu.json
echo "ncu exit=$?"
[ -f "$OUT.ncu-rep" ] && ncu --import "$OUT.ncu-rep" --page details --csv > "$OUT.csv" 2>/dev/null && echo "wrote $OUT.csv"
