#!/bin/bash
# One edge-batch arm, one GPU, one process.  usage: run_ncu_edge_batch.sh <E_c> <gpu>
# Counters only -- durations from this run are NOT timings (compile of sibling arms overlaps).
set -uo pipefail
cd "$(dirname "$0")/.."
E=$1; GPU=$2
PY=/iopsstor/scratch/cscs/dlu/envs/zippel/bin/python
OUT=bench/results/ncu_eb_E${E}
mkdir -p bench/results logs
CUDA_VISIBLE_DEVICES=$GPU ncu --profile-from-start off --replay-mode kernel \
    --section SpeedOfLight --section MemoryWorkloadAnalysis --section Occupancy \
    --section LaunchStats --section WarpStateStats --section SourceCounters \
    --metrics \
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,\
launch__occupancy_limit_registers,\
launch__occupancy_limit_shared_mem,\
launch__occupancy_per_register_count,\
dram__bytes.sum,\
dram__bytes.sum.per_second,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
lts__t_sector_hit_rate.pct,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
memory_l2_theoretical_sectors_global,\
memory_l2_theoretical_sectors_global_ideal,\
gpu__time_duration.sum \
    --export "$OUT" --force-overwrite "$PY" -u bench/ncu_edge_batch.py --edge-batch "$E"
rc=$?
echo "ncu exit=$rc for E_c=$E"
if [ -f "$OUT.ncu-rep" ]; then
  ncu --import "$OUT.ncu-rep" --page details --csv > "$OUT.csv" 2>/dev/null && echo "wrote $OUT.csv"
  # SASS hoist check (D78): is the weight load outside the unrolled copies or repeated inside?
  ncu --import "$OUT.ncu-rep" --page source --print-source sass --csv > "$OUT.sass.csv" 2>/dev/null
  echo "sass rows: $(wc -l < "$OUT.sass.csv" 2>/dev/null || echo 0)"
fi
exit $rc
