#!/usr/bin/env bash
# Make the SGLang container available for this recipe. Idempotent; --force re-pulls.
#
# This recipe has no virtual environment. The engine is a container, so "building the environment"
# means having the image, and the image is already staged beside the weights in the shared repository.
# The normal path therefore costs nothing: it verifies the staged copy and stops.
#
# The pull exists only for a site that has no staged copy. It took 2 hours 6 minutes when it was done
# here, and it resolves a moving tag, so a rebuilt image is not guaranteed to match the one these
# numbers were measured on. Prefer the staged copy.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/../../../../common/lib/repo_root.sh"
source "$REPO_ROOT/common/defaults.sh"
source "$S/env.sh"   # resolves SIF, preferring MODELS_DIR and falling back to the shared repository

IMAGE="${IMAGE:-docker://lmsysorg/sglang:kimi-k3-cu12}"
FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

if [ -f "$SIF" ] && [ "$FORCE" = 0 ]; then
  echo "container present: $SIF"
  echo "  $(stat -c %s "$SIF") bytes"
  command -v singularity >/dev/null 2>&1 \
    && singularity exec "$SIF" python3 -c "import sglang; print('  sglang', sglang.__version__)" \
    || echo "  singularity not on PATH here, so the version was not read"
  exit 0
fi

command -v singularity >/dev/null 2>&1 || { echo "singularity is required to pull the image" >&2; exit 1; }
DEST="$(dirname "$SIF")"
[ -d "$DEST" ] || { echo "destination does not exist: $DEST" >&2; exit 1; }
[ -w "$DEST" ] || { echo "destination is not writable by $USER: $DEST" >&2; exit 1; }

# Keep the conversion scratch beside the target rather than in the home directory, which is too small
# for a 16 GB image and its intermediate layers.
export SINGULARITY_CACHEDIR="$DEST/.singularity-cache"
export SINGULARITY_TMPDIR="$DEST/.singularity-tmp"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"

echo "pulling $IMAGE to $SIF (this took 2 h 6 min here)"
singularity pull --force "$SIF" "$IMAGE"
rm -rf "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"
echo "done: $(stat -c %s "$SIF") bytes"
