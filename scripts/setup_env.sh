#!/usr/bin/env bash
# Environment setup on the HSG login node (GB200, aarch64, sm_100, driver CUDA 12.8).
# Needs internet, so run it on the login node, not a compute node. Safe to re-run.
#   bash scripts/setup_env.sh [install dir]
set -uo pipefail

W=${1:?usage: setup_env.sh <install dir>}
REPO=${REPO:-$HOME/genco-fdia}
cd "$W" || exit 1
export UV_CACHE_DIR=$W/.uv HF_HOME=$W/hf JULIA_DEPOT_PATH=$W/julia
UV=$HOME/.local/bin/uv
echo "arch=$(uname -m) host=$(hostname) W=$W"

[ -x venv/bin/python ] || $UV venv --python 3.12 venv || exit 1
. venv/bin/activate || exit 1

# 1. torch trio from the cu128 index. The node driver is CUDA 12.8, so the cu130
#    build that gridfm-graphkit's PyPI release drags in cannot run here. Range
#    matches the paper-repro branch pin (torch>=2.7.1,<2.9).
$UV pip install --reinstall "torch>=2.7.1,<2.9" torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128 || { echo "TORCH INSTALL FAILED"; exit 1; }
TORCH_BEFORE=$(python -c "import torch; print(torch.__version__)")

# 2. no compiled torch_scatter on aarch64: swap in the pure-torch stand-in
$UV pip uninstall torch-scatter gridfm-graphkit 2>/dev/null
$UV pip install "$REPO/scripts/torch_scatter_shim" || exit 1

# 3. graphkit from the paper-repro branch source (editable) + helpers
[ -d src-graphkit ] || git clone -b genco-paper-repro https://github.com/gridfm/gridfm-graphkit src-graphkit
$UV pip install -e src-graphkit
$UV pip install numpy scipy matplotlib pandas pyarrow pyyaml huggingface_hub

TORCH_AFTER=$(python -c "import torch; print(torch.__version__)")
if [ "$TORCH_BEFORE" != "$TORCH_AFTER" ]; then
  echo "TORCH CHANGED $TORCH_BEFORE -> $TORCH_AFTER during graphkit install; fix before continuing"; exit 1
fi

# 4. case118 Base checkpoint + its dataset (current CLI is `hf`, not `huggingface-cli`)
hf download gridfm/genco-pf-datakit --include "case118_ieee/base/seed0/*" --local-dir ckpt
hf download gridfm/pf_small_case118_ieee --repo-type dataset --local-dir data

python - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda build", torch.version.cuda, "| arch list", torch.cuda.get_arch_list())
import torch_scatter; print("torch_scatter", torch_scatter.__version__)
import gridfm_graphkit; print("gridfm_graphkit imports OK")
EOF
find ckpt data -maxdepth 4 | head -40
du -sh "$W"
echo "SETUP DONE in $W"
