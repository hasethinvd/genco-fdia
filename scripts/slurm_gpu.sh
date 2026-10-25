#!/usr/bin/env bash
# Run a command on one GB200 node inside the genco venv (example for a Slurm cluster).
#   bash scripts/slurm_gpu.sh python scripts/test_genco_oracle.py
# On the cluster we used every QoS required gres/gpu=4 per job and the batch partition was
# node-exclusive, so a job was one whole 4-GPU node; the code uses GPU 0. Adapt to your site.
# `interactive` has the highest priority; `short` (<=2h) is the fallback.
# Override with env vars if the scheduler rejects a value:
#   ACCOUNT=<your_account> PARTITION=batch QOS=interactive TIME=00:20:00
set -euo pipefail
W=${W:?set W to a writable work directory that holds venv/, ckpt/ and hf/}
REPO=${REPO:-$HOME/genco-fdia}
ACCOUNT=${ACCOUNT:?set your Slurm account}
PARTITION=${PARTITION:-batch}
QOS=${QOS:-interactive}
TIME=${TIME:-00:20:00}

cd "$REPO"
srun -A "$ACCOUNT" -p "$PARTITION" -q "$QOS" -N1 -n1 --gres=gpu:4 -t "$TIME" \
  --export=ALL,GENCO_CKPT_DIR="$W/ckpt/case118_ieee/base/seed0",HF_HOME="$W/hf" \
  bash -c 'source '"$W"'/venv/bin/activate && cd '"$REPO"' && nvidia-smi -L | head -1 && "$@"' _ "$@"
