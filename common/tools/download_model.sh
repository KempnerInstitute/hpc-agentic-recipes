#!/usr/bin/env bash
# Download model weights from Hugging Face into a directory you choose.
#
#   download_model.sh [--dry-run] <hf_repo> <dest_parent_dir> [local_name]
#
# The destination is required rather than defaulting to MODELS_DIR, which points at the shared model
# repository. That is read-only for almost everyone, so defaulting there would send most users into a
# permission error several minutes into a large transfer, and would let anyone who does have write access
# fill a shared directory by accident.
#
# It reports no free space. df would show the whole filesystem, while what actually limits a download is
# the group quota, read with lfs quota -g on Lustre or quota on netscratch. Confirm your own space first.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

usage() {
  echo "usage: download_model.sh [--dry-run] <hf_repo> <dest_parent_dir> [local_name]" >&2
  exit 2
}

DRY=0
[ "${1:-}" = "--dry-run" ] && { DRY=1; shift; }
[ $# -ge 2 ] || usage
REPO="$1"
PARENT="$2"
NAME="${3:-$(basename "$REPO")}"
DEST="$PARENT/$NAME"

command -v hf >/dev/null 2>&1 || { echo "download_model.sh: hf is not on PATH" >&2; exit 1; }
[ -d "$PARENT" ] || { echo "download_model.sh: $PARENT does not exist" >&2; exit 1; }
[ -w "$PARENT" ] || { echo "download_model.sh: $PARENT is not writable by $USER" >&2; exit 1; }

echo "repo:        $REPO"
echo "destination: $DEST"
[ "$DRY" = 1 ] && { echo "dry run, nothing transferred"; exit 0; }

# Both accelerated transfer backends are off: the plain path resumes cleanly after an interruption,
# which matters more than peak speed for a several-hundred-gigabyte repository.
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
hf download "$REPO" --local-dir "$DEST"
echo "downloaded to $DEST"
echo "serve it with MODELS_DIR=$PARENT, or MODEL=$DEST"
