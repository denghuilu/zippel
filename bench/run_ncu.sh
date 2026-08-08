#!/bin/bash
# Nsight Compute run for conv1_90. Must be invoked INSIDE the uenv that carries ncu:
#   uenv run prgenv-gnu/25.6:v2 --view=default -- bash bench/run_ncu.sh
# The image ships ncu 2025.2.0 and its own CUPTI; the conda env on /iopsstor stays visible and
# torch 2.13.0+cu130 initialises CUDA normally under the mount (verified before this was written).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/iopsstor/scratch/cscs/dlu/envs/zippel/bin/python
OUT=bench/results/ncu_conv1_90
mkdir -p bench/results logs

# --profile-from-start off pairs with torch.cuda.profiler.start() in the driver, so ncu profiles
# exactly the three kernel launches and none of the setup.
ncu --profile-from-start off \
    --replay-mode kernel \
    --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section MemoryWorkloadAnalysis_Chart \
    --section WarpStateStats \
    --section Occupancy \
    --section LaunchStats \
    --metrics \
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,\
smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio,\
lts__t_sector_hit_rate.pct,\
lts__t_bytes.sum.per_second,\
dram__bytes.sum.per_second,\
launch__occupancy_limit_shared_mem,\
launch__occupancy_limit_registers,\
launch__registers_per_thread,\
launch__shared_mem_per_block_dynamic \
    --export "$OUT" --force-overwrite \
    "$PY" -u bench/ncu_profile.py "$@"
rc=$?
echo "ncu exit=$rc"
[ -f "$OUT.ncu-rep" ] && ncu --import "$OUT.ncu-rep" --csv --page raw > "$OUT.csv" 2>/dev/null \
  && echo "wrote $OUT.csv ($(wc -l < "$OUT.csv") rows)"
exit $rc
