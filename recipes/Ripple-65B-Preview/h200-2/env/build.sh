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
  if command -v singularity >/dev/null 2>&1; then
    singularity exec "$SIF" python3 -c "import sglang; print('  sglang', sglang.__version__)" || {
      echo "  the image is present but sglang does not import inside it: treat it as unusable" >&2
      exit 1
    }
  else
    echo "  singularity is not on PATH here, so the image could not be verified" >&2
    echo "  run this on a node that has it, or set SIF and verify by hand" >&2
    exit 1
  fi
  exit 0
fi

command -v singularity >/dev/null 2>&1 || { echo "singularity is required to pull the image" >&2; exit 1; }

# Never pull into the shared repository. MODELS_DIR points there by default, so a missing image would
# otherwise write 12.7 GiB into a directory every cluster user reads. A Kempner user finds the image
# already staged; anyone else points MODELS_DIR at their own space first.
# env.sh falls back to the shared copy when the image is missing, which would otherwise leave a user
# who set MODELS_DIR unable to build one at all. Retarget the pull into their own space instead.
_shared_prefix=/n/holylfs06/LABS/kempner_shared/
_resolved="$(readlink -m "$SIF" 2>/dev/null || echo "$SIF")"
case "$_resolved" in
  "$_shared_prefix"*)
    _own="$MODELS_DIR/Ripple-65B-Preview/container/sglang-0.5.19.sif"
    case "$(readlink -m "$_own" 2>/dev/null || echo "$_own")" in
      "$_shared_prefix"*) : ;;
      *) echo "the shared copy is missing, so building into your own space: $_own"
         SIF="$_own"; _resolved="$(readlink -m "$SIF" 2>/dev/null || echo "$SIF")" ;;
    esac ;;
esac

case "$_resolved" in
  "$_shared_prefix"*)
    echo "refusing to pull into the shared repository: $SIF" >&2
    echo "The staged copy is missing there, and MODELS_DIR also points inside it. Set MODELS_DIR to" >&2
    echo "your own space, or SIF to a writable path, and run this again." >&2
    exit 1 ;;
esac
DEST="$(dirname "$SIF")"
mkdir -p "$DEST"
[ -w "$DEST" ] || { echo "destination is not writable by $USER: $DEST" >&2; exit 1; }

# Scoped to this process: a fixed directory let concurrent builds delete each other's work.
_work="$(mktemp -d "$DEST/.singularity-build.XXXXXX")" || { echo "cannot create a work directory under $DEST" >&2; exit 1; }
trap 'rm -rf "$_work" 2>/dev/null || true' EXIT
export SINGULARITY_CACHEDIR="$_work/cache"
export SINGULARITY_TMPDIR="$_work/tmp"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"

# A venv is not an option: sglang 0.5.19 ships only manylinux_2_34 wheels, needing glibc 2.34, and
# the nodes here are Rocky 8 with glibc 2.28.
echo "pulling $IMAGE to $SIF (this took 1 h 20 min here)"
singularity pull --force "$SIF.partial" "$IMAGE"
# Verify before publishing, so a bad or mismatched IMAGE override cannot replace a staged image
# with something that does not run.
singularity exec "$SIF.partial" python3 -c "import sglang; print('  sglang', sglang.__version__)" || {
  echo "the pulled image does not import sglang; leaving the staged image untouched" >&2
  rm -f "$SIF.partial"
  exit 1
}
mv -f "$SIF.partial" "$SIF"
echo "done: $(stat -c %s "$SIF") bytes"
