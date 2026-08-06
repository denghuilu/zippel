#!/usr/bin/env bash
# Reproduce every number in REPORT.md from a clean clone.
#
#   source env.sh && bash bench/run_all.sh
#
# or, for the numbers actually reported (exclusive node):
#
#   sbatch slurm/bench.sbatch
#
# Fixtures are regenerated from fixed seeds if absent, so the .npz files need not be
# committed. Baselines run at their recommended fast settings; the measured boundary and
# the inputs are identical for every implementation (bench/harness.py is the only timing
# path, fixtures/load.py the only input path).

set -euo pipefail
cd "$(dirname "$0")/.."

# Two concurrent runs sharing bench/results/ silently overwrite each other's JSON: a
# login-node run once clobbered an exclusive-node B1 result, leaving contended numbers on
# disk while the authoritative ones survived only in the SLURM log. Take an exclusive lock
# so that cannot recur; every Measurement also records host/slurm_job so provenance is
# checkable after the fact.
mkdir -p bench/results
exec 9>bench/results/.run.lock
if ! flock -n 9; then
  echo "another benchmark run holds bench/results/.run.lock -- refusing to overwrite" >&2
  exit 1
fi

FIXTURES="${FIXTURES:-si_small cu_small si_medium cu_medium si_large cu_large}"
PRECISIONS="${PRECISIONS:-fp32 tf32 bf16}"
mkdir -p bench/results

echo "############ 0. fixtures ############"
# Regenerate if absent, then ALWAYS verify against the committed manifest. The fixtures
# define the inputs every implementation sees, so a silent change to them would invalidate
# every number in REPORT.md; `--verify` re-hashes the array contents (backend-independent)
# against fixtures/manifest.json and exits non-zero on any mismatch.
missing=0
for f in $FIXTURES; do [ -f "fixtures/$f.npz" ] || missing=1; done
[ "$missing" = 1 ] && python fixtures/make_fixtures.py || echo "all fixtures present"
python fixtures/make_fixtures.py --verify

echo "############ 1. correctness gate ############"
# No benchmark number is reported without a green test run (work order, section 0).
python -m pytest tests/ -q -p no:warnings

echo "############ 2. B1 eager ############"
python baselines/b1_eager.py --fixtures $FIXTURES --precisions $PRECISIONS \
    --out bench/results/b1_eager.json

echo "############ 3. B2 torch.compile ############"
# b2_probe is the backend ladder (eager / backend=eager / aot_eager / inductor): it
# establishes where the double backward is rejected in seconds. b2_compile.py carries the
# fuller inventory but autotunes forwards that then fail at the backward, so it is not on
# the reproduction path.
python baselines/b2_probe.py $FIXTURES

echo "############ 4. B3 cuEquivariance ############"
# cuEquivariance's fused ops require torch <= 2.11.0 on this platform (its 0.11.0
# extension does not load against the torch 2.13 the rest of the stack needs -- see
# REPORT.md). B3 therefore runs in its own interpreter, together with an eager control
# measured in the same interpreter so the comparison is internally consistent.
if [ -x "${ZIPPEL_CUEQ_PY:-}" ]; then
  "$ZIPPEL_CUEQ_PY" baselines/b3_cueq.py --fixtures $FIXTURES --precisions $PRECISIONS \
      --out bench/results/b3_cueq.json
else
  echo "ZIPPEL_CUEQ_PY not set -> B3 skipped; see REPORT.md section on the cuEquivariance ABI"
fi

echo "############ 5. max batch (GiB budgets) ############"
# Primary 80 GiB, secondary full-card 95.6 GiB, both by binary search on the replication
# factor with a measured peak allocation (DECISIONS.md D13).
python bench/max_batch.py --fixture si_medium --precisions fp32 bf16 \
    --out bench/results/max_batch.json

echo "############ 6. summary ############"
python bench/summarize.py
