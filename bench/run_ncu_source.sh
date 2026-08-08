#!/bin/bash
# Source-attributed ncu on conv1_90, baseline + A_transpose. Names the 1.25 MB-leaks-terabytes
# anomaly before the weight-tile arm is allowed to fire.
#
#   uenv run prgenv-gnu/25.6:v2 --view=default -- bash bench/run_ncu_source.sh
#
# THE QUESTION. D60's arithmetic says the traffic is not weight re-reads *once per CTA* (that is
# 1.31 MB x 259 474 = 340 GB, only 15 % of the 2.24 TB measured) and is not per-edge data (8.5 KB
# per edge x 5x sector amplification is ~11 GB). What fits is weights re-read *per term*: ~5 000
# weight reads per thread x 128 threads x 20.31/4 sector amplification is ~13 MB of L1->L2 per
# CTA, i.e. ~3.4 TB against a measured L2 total of 4.48 TB, of which the 57.4 % hit rate leaves
# ~1.9 TB to DRAM against 2.24 measured. Every step of that is arithmetic on measured quantities,
# and none of it is an attribution. This run decides it.
#
# So the one thing to read off the output: **do the L2 misses come from the weight loads or from
# the per-edge loads?** The two have completely different fixes, and picking wrong is exactly the
# D42 failure this sequence exists to stop repeating.
#
# CAVEAT, recorded before the run. The kernel is NVRTC-compiled from generated Python through the
# CuTe DSL; if the toolchain emits no line information, ncu can correlate to SASS but not to
# anything human-readable. That is a real possible outcome and it will be reported as a blocker
# rather than worked around with a guess. SASS-level attribution is still usable here: the weight
# loads and the per-edge loads differ in their address arithmetic, and their sector counts differ
# by construction.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/iopsstor/scratch/cscs/dlu/envs/zippel/bin/python
OUT=bench/results/ncu_conv1_90_source
mkdir -p bench/results logs

ncu --profile-from-start off \
    --replay-mode kernel \
    --section SourceCounters \
    --section MemoryWorkloadAnalysis \
    --section MemoryWorkloadAnalysis_Tables \
    --import-source yes \
    --metrics \
memory_l2_theoretical_sectors_global,\
memory_l2_theoretical_sectors_global_ideal,\
memory_l1_wavefronts_shared,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum,\
lts__t_sectors_op_read.sum,\
lts__t_sectors_op_read_lookup_miss.sum,\
lts__t_sector_hit_rate.pct,\
dram__sectors_read.sum,\
dram__bytes_read.sum \
    --export "$OUT" --force-overwrite \
    "$PY" -u bench/ncu_profile.py --arms baseline,A_transpose
rc=$?
echo "ncu exit=$rc"
if [ -f "$OUT.ncu-rep" ]; then
  ncu --import "$OUT.ncu-rep" --page details --csv > "$OUT.csv" 2>/dev/null \
    && echo "wrote $OUT.csv ($(wc -l < "$OUT.csv") rows)"
  # The source page is the point of the run; if it is empty, that IS the finding.
  ncu --import "$OUT.ncu-rep" --page source --csv > "$OUT.source.csv" 2>/dev/null
  n=$(wc -l < "$OUT.source.csv" 2>/dev/null || echo 0)
  echo "source page: $n rows"
  [ "$n" -lt 5 ] && echo "SOURCE CORRELATION UNAVAILABLE -- report as a blocker, do not guess"
fi
exit $rc
