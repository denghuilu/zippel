#!/bin/bash
# Is the 539x amplification register spilling? Local-memory traffic is invisible to every
# demand-side model, unaffected by weight pinning, and produces DRAM WRITES -- which are 42% of
# this kernel's traffic against a 1.196 GB output.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/iopsstor/scratch/cscs/dlu/envs/zippel/bin/python
OUT=bench/results/ncu_local_spill
mkdir -p bench/results logs
CUDA_VISIBLE_DEVICES=0 ncu --profile-from-start off --replay-mode kernel \
  --metrics \
l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_local_op_st.sum,\
l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,\
smsp__inst_executed_op_local_ld.sum,\
smsp__inst_executed_op_local_st.sum,\
launch__registers_per_thread,\
dram__bytes.sum,dram__bytes_read.sum,dram__bytes_write.sum,\
gpu__time_duration.sum \
  --export "$OUT" --force-overwrite \
  "$PY" -u bench/l2_persist.py --ncu --configs "0:0" --out bench/results/l2_local.json
echo "ncu exit=$?"
[ -f "$OUT.ncu-rep" ] && ncu --import "$OUT.ncu-rep" --page details --csv > "$OUT.csv" 2>/dev/null && echo "wrote $OUT.csv"
