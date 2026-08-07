#!/usr/bin/env bash
# S1c measurement on the local node's 4 GPUs, one configuration per GPU.
#
# NOT THE PINNED PROTOCOL. slurm/s1c.sbatch runs N=5 INDEPENDENT allocations because repeats
# inside one allocation cannot see between-node and between-placement variance -- the variance
# that made si_small differ by 80 % across two clean runs. This runs 5 repeats on ONE node, on a
# shared login node, so its error bar covers run-to-run noise only and NOT placement.
#
# Numbers from here are DEVELOPMENT numbers. They are labelled `provenance: local-login-node` in
# the JSON so they cannot be mistaken later for verdict-table numbers, and REPORT must say so
# wherever it quotes them.
#
# Sources env.sh, which is what points TMPDIR at the repo and therefore what makes CuTe DSL's
# compile cache work (findings/cute-dsl-cache-dir-is-a-noop.md). The first repeat of each
# configuration is a cold compile; the rest should be warm.

set -uo pipefail
cd "${ZIPPEL_ROOT:-/iopsstor/scratch/cscs/dlu/iclr/zippel}"
source env.sh

# EXCLUSIVE LOCK ON THE RESULTS DIRECTORY. Without it a second runner -- on another host, in
# another session, started before a `cd` that moved us to a compute node -- writes the same
# filenames on the same shared filesystem and the two interleave. That is not hypothetical: it
# happened at Gate 0 (job 4376123 overwritten by a login-node run, REPORT 8.7) and it happened
# again here, with a login-node sweep and a compute-node sweep both writing
# bench/results/s1c_local_rep*.json. `bench/run_all.sh` grew this lock after the first time; this
# script did not inherit it, which is the whole lesson.
exec 9> bench/results/.s1c_local.lock
if ! flock -n 9; then
    echo "REFUSING: another s1c_local run holds the results lock." >&2
    echo "  held by: $(cat bench/results/.s1c_local.owner 2>/dev/null || echo unknown)" >&2
    exit 1
fi
echo "$(hostname) pid=$$ $(date -Is)" > bench/results/.s1c_local.owner

REPS="${REPS:-5}"
MAXVOL=10000                       # term-minimal split, identical everywhere (D36)
mkdir -p logs bench/results

run_config() {
    local gpu="$1" fix="$2" dt="$3"
    for r in $(seq 1 "$REPS"); do
        # hostname in the filename: even under the lock, results from different hosts must not
        # be able to occupy the same path -- a lock prevents concurrency, not confusion
        local out="bench/results/s1c_local_$(hostname)_rep${r}_${fix}_${dt}.json"
        CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=8 \
        numactl --cpunodebind=0 --membind=0 \
            python -u bench/s1c_bench.py --fixture "$fix" --dtype "$dt" \
                --max-volume "$MAXVOL" --out "$out" \
            >> "logs/s1c_local_${fix}_${dt}.log" 2>&1 \
            || echo "  rep $r FAILED: $fix $dt" >> "logs/s1c_local_${fix}_${dt}.log"
    done
    echo "done: $fix $dt (gpu $gpu)" >> logs/s1c_local_progress.log
}

: > logs/s1c_local_progress.log
run_config 0 si_small  f32 &
run_config 1 si_small  f64 &
run_config 2 si_medium f32 &
run_config 3 si_medium f64 &
wait
echo "ALL CONFIGS DONE" >> logs/s1c_local_progress.log
