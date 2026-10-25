#!/usr/bin/env bash
# One job, one GB200 node: every GENCO run for the entry. Launch with
#   bash scripts/slurm_gpu.sh bash scripts/run_all_genco.sh [scenarios.npz]
# Without an npz argument only the case118.m (synthetic load walk) runs are done.
set -uo pipefail
SCEN=${1:-}
cd "$(dirname "$0")/.."
mkdir -p results logs

run() { echo; echo "### $*"; "$@"; }

if [ "${SKIP_CASE118:-0}" != 1 ]; then
  # 1. acceptance + timing (both slack variants)
  run python scripts/test_genco_oracle.py

  # 2. case118.m operating points (out-of-distribution for the pretrained PF checkpoint)
  run python scripts/run_experiment.py --oracle genco --n_traj 60 --T 10 --attack_at 5 --seed 1
  run python scripts/run_experiment.py --oracle genco --n_traj 60 --T 12 --attack_at 5 --ramp 5 --seed 1
fi

# 3. datakit operating points (in-distribution): GENCO and the NR reference on the SAME scenarios
if [ -n "$SCEN" ]; then
  for o in genco nr; do
    run python scripts/run_experiment.py --oracle $o --scenarios "$SCEN" --n_traj 60 --T 10 --attack_at 5 --seed 1
    run python scripts/run_experiment.py --oracle $o --scenarios "$SCEN" --n_traj 60 --T 12 --attack_at 5 --ramp 5 --seed 1
  done
fi
echo; echo "ALL DONE"; ls -la results
