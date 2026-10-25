#!/usr/bin/env bash
# Fresh-clone reproduction of the GENCO rows of the write-up (needs a GPU and a venv with
# gridfm-graphkit, see setup_env.sh). Clones the public repo, downloads the released base
# checkpoint and the fine-tuned checkpoint from Hugging Face, reruns the step-attack
# experiments on the shipped datakit scenarios, and compares with the committed results.
#   W=<work dir holding venv/> bash scripts/reproduce_genco.sh [parent dir for the clone]
set -euo pipefail
W=${W:?set W to the work dir that holds venv/}
P=${1:-$(mktemp -d)}
D=$P/genco-fdia-repro
export HF_HOME=${HF_HOME:-$W/hf}

# Two phases when compute nodes have no internet: FETCH_ONLY=1 on a login node (clone +
# downloads), then SKIP_FETCH=1 inside the GPU job, both with the same parent dir argument.
if [ -z "${SKIP_FETCH:-}" ]; then
  if [ -d "$D/.git" ]; then git -C "$D" pull -q; else git clone -q https://github.com/hasethinvd/genco-fdia "$D"; fi
  cd "$D"
  echo "clone at $(git rev-parse --short HEAD) in $D"
  hf download gridfm/genco-pf-datakit --include "case118_ieee/base/seed0/*" --local-dir ckpt_base >/dev/null
  hf download hasethi/genco-fdia-finetuned --include "mixed/*" --local-dir ckpt_hf >/dev/null
  [ -n "${FETCH_ONLY:-}" ] && { echo "FETCH DONE"; exit 0; }
fi
cd "$D"
export GENCO_CKPT_DIR=$D/ckpt_base/case118_ieee/base/seed0

mkdir -p reference
cp results/results_genco_ramp1_datakit.json results/results_genco_ft_mix_ramp1_datakit.json reference/

echo; echo "### adapter acceptance + timing"
python scripts/test_genco_oracle.py | grep -E "===|BASE CASE|ACCEPTANCE|ms/solve"
echo; echo "### released checkpoint, datakit scenarios, step attack"
python scripts/run_experiment.py --oracle genco --scenarios data/scenarios_118.npz --n_traj 60 --T 10 --attack_at 5 --seed 1 | grep -E "clean:|^k=|saved"
echo; echo "### fine-tuned (mixed) checkpoint from Hugging Face, same scenarios"
python scripts/run_experiment.py --oracle genco --genco_ckpt ckpt_hf/mixed --tag _ft_mix --label "GENCO-PF (fine-tuned, datakit + case file)" \
       --scenarios data/scenarios_118.npz --n_traj 60 --T 10 --attack_at 5 --seed 1 | grep -E "clean:|^k=|saved"

echo; echo "### comparison with the committed results (GPU tolerance: det 0.02, threshold 2%)"
python scripts/compare_results.py reference/results_genco_ramp1_datakit.json results/results_genco_ramp1_datakit.json --tol_det 0.02 --tol_thr 0.02
python scripts/compare_results.py reference/results_genco_ft_mix_ramp1_datakit.json results/results_genco_ft_mix_ramp1_datakit.json --tol_det 0.02 --tol_thr 0.02
echo "REPRODUCTION DONE"
