#!/usr/bin/env bash
# Verify the staged Ripple container and report its SGLang version. Pulls only if the image is missing.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/../../../../common/lib/repo_root.sh"
source "$REPO_ROOT/common/defaults.sh"
source "$S/env.sh"   # resolves SIF, preferring MODELS_DIR and falling back to the shared repository

IMAGE="${IMAGE:-docker://lmsysorg/sglang:v0.5.19-cu130}"
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
mkdir -p "$DEST"
[ -w "$DEST" ] || { echo "destination is not writable by $USER: $DEST" >&2; exit 1; }

export SINGULARITY_CACHEDIR="$DEST/.singularity-cache"
export SINGULARITY_TMPDIR="$DEST/.singularity-tmp"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"

# A venv is not an option: sglang 0.5.19 ships only manylinux_2_34 wheels, needing glibc 2.34, and
# the nodes here are Rocky 8 with glibc 2.28.
echo "pulling $IMAGE to $SIF (this took 1 h 20 min here)"
singularity pull --force "$SIF" "$IMAGE"
rm -rf "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"
echo "done: $(stat -c %s "$SIF") bytes"
