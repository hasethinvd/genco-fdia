#!/usr/bin/env bash
# One GPU job: fine-tune GENCO-PF, then run every experiment with the fine-tuned oracle.
#   TIME=01:10:00 bash scripts/slurm_gpu.sh bash scripts/run_ft.sh
# FT_MINUTES (default 25) is the training wall-clock budget.
set -uo pipefail
cd "$(dirname "$0")/.."
W=${W:?set W to a writable work directory that holds venv/, ckpt/ and hf/}
TAG=${FT_TAG:-}                      # e.g. _mix ; keeps runs from overwriting each other
LABEL=${FT_LABEL:-GENCO-PF (fine-tuned)}
FT=$W/ckpt_ft$TAG
mkdir -p results

run() { echo; echo "### $*"; "$@"; }

TRAIN=""
[ -f data/scenarios_train.npz ] && TRAIN="--datakit_train data/scenarios_train.npz"
run python scripts/finetune_genco.py $TRAIN --minutes "${FT_MINUTES:-25}" --out_dir "$FT" --log results/finetune_log$TAG.json

# datakit test scenarios (the headline comparison; NR reference already exists for these)
run python scripts/run_experiment.py --oracle genco --genco_ckpt "$FT" --tag _ft$TAG --label "$LABEL" --scenarios data/scenarios_118.npz --n_traj 60 --T 10 --attack_at 5 --seed 1
run python scripts/run_experiment.py --oracle genco --genco_ckpt "$FT" --tag _ft$TAG --label "$LABEL" --scenarios data/scenarios_118.npz --n_traj 60 --T 12 --attack_at 5 --ramp 5 --seed 1
# case118.m operating points (the off-distribution case)
run python scripts/run_experiment.py --oracle genco --genco_ckpt "$FT" --tag _ft$TAG --label "$LABEL" --n_traj 60 --T 10 --attack_at 5 --seed 1
run python scripts/run_experiment.py --oracle genco --genco_ckpt "$FT" --tag _ft$TAG --label "$LABEL" --n_traj 60 --T 12 --attack_at 5 --ramp 5 --seed 1
echo; echo "ALL DONE"; ls -la results
