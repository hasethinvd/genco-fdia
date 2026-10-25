#!/usr/bin/env bash
# Publish the two fine-tuned checkpoints to a public Hugging Face model repo.
# Run where the checkpoints are (the cluster work dir), after `hf auth login`:
#   HF_USER=<your_hf_username> W=<work dir> bash scripts/upload_checkpoints_hf.sh
set -euo pipefail
HF_USER=${HF_USER:?set HF_USER to your Hugging Face username}
W=${W:?set W to the work dir that holds ckpt_ft/ and ckpt_ft_mix/}
REPO=$HF_USER/genco-fdia-finetuned
cd "$(dirname "$0")/.."
hf repo create "$REPO" --type model 2>/dev/null || echo "repo exists, continuing"
hf upload "$REPO" submission/hf_model_card.md README.md
hf upload "$REPO" "$W/ckpt_ft_mix" mixed --include "*.pt" --include "*.yaml"
hf upload "$REPO" "$W/ckpt_ft"     case_file_only --include "*.pt" --include "*.yaml"
echo "published: https://huggingface.co/$REPO"
