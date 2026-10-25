#!/usr/bin/env bash
# Run a command on a CPU node inside the genco venv (datakit/Julia need more memory
# than the login node allows). Same env as slurm_gpu.sh but no GPU.
#   bash scripts/slurm_cpu.sh python scripts/make_datakit_scenarios.py --n_scen 720 ...
# Override if the scheduler rejects a value:
#   ACCOUNT=<your_account> PARTITION=cpu QOS=cpu-short TIME=01:00:00 CPUS=16 MEM=64G
set -euo pipefail
W=${W:?set W to a writable work directory that holds venv/, ckpt/ and hf/}
REPO=${REPO:-$HOME/genco-fdia}
ACCOUNT=${ACCOUNT:?set your Slurm account}
PARTITION=${PARTITION:-cpu}
QOS=${QOS:-cpu-short}
TIME=${TIME:-01:00:00}
CPUS=${CPUS:-16}
MEM=${MEM:-64G}

cd "$REPO"
srun -A "$ACCOUNT" -p "$PARTITION" -q "$QOS" -N1 -n1 -c "$CPUS" --mem="$MEM" -t "$TIME" \
  --export=ALL,JULIA_DEPOT_PATH="$W/julia",HF_HOME="$W/hf",GENCO_CKPT_DIR="$W/ckpt/case118_ieee/base/seed0" \
  bash -c 'source '"$W"'/venv/bin/activate && cd '"$REPO"' && "$@"' _ "$@"
