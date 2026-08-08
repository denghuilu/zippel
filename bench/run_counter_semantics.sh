#!/bin/bash
# Timeboxed counter-semantics calibration. See bench/counter_semantics.py for what it decides.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/iopsstor/scratch/cscs/dlu/envs/zippel/bin/python
OUT=bench/results/counter_semantics
mkdir -p bench/results logs
ncu --profile-from-start off --replay-mode kernel \
    --metrics \
lts__t_sectors_op_read.sum,\
lts__t_sectors_op_write.sum,\
lts__t_sectors_op_read_lookup_miss.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
gpu__time_duration.sum \
    --export "$OUT" --force-overwrite "$PY" -u bench/counter_semantics.py
rc=$?
echo "ncu exit=$rc"
[ -f "$OUT.ncu-rep" ] && ncu --import "$OUT.ncu-rep" --page details --csv > "$OUT.csv" 2>/dev/null \
  && echo "wrote $OUT.csv ($(wc -l < "$OUT.csv") rows)"
exit $rc
