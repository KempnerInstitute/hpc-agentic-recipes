#!/usr/bin/env bash
# Download model weights from Hugging Face into the shared models dir.
# Usage: download_model.sh <hf_repo> [local_name]
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/../lib/repo_root.sh"
source "$S/../defaults.sh"
export PATH="$HOME/.local/bin:$PATH"
REPO="${1:?usage: download_model.sh <hf_repo> [local_name]}"
NAME="${2:-$(basename "$REPO")}"
DEST="$MODELS_DIR/$NAME"
# Both accelerated transfer backends are off: the plain path resumes cleanly after an interruption,
# which matters more than peak speed for a several-hundred-gigabyte repository.
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
hf download "$REPO" --local-dir "$DEST"
echo "downloaded to $DEST"
