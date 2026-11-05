#!/usr/bin/env bash
# Verify the staged Kimi-K3.1-Pro container and report its SGLang version. Pulls only if missing.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/../../../../common/lib/repo_root.sh"
source "$REPO_ROOT/common/defaults.sh"
source "$S/env.sh"

IMAGE="${IMAGE:-docker://lmsysorg/sglang:dev-kimi-k31-pro}"
# The tag moves, so record what the recipe's numbers were measured against and warn on a mismatch.
MEASURED_SGLANG="0.0.0.dev1+g41208373c"
FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

if [ -f "$SIF" ] && [ "$FORCE" = 0 ]; then
  echo "container present: $SIF"
  echo "  $(stat -c %s "$SIF") bytes"
  if command -v singularity >/dev/null 2>&1; then
    _v="$(singularity exec "$SIF" python3 -c "import sglang,torch; print(sglang.__version__, torch.__version__)" 2>/dev/null)" || {
      echo "  the image is present but sglang does not import inside it: treat it as unusable" >&2
      exit 1
    }
    echo "  sglang ${_v% *}, torch ${_v#* }"
    case "${_v% *}" in
      "$MEASURED_SGLANG"*) : ;;
      *) echo "  note: measured against sglang $MEASURED_SGLANG; this image reports ${_v% *}" ;;
    esac
  else
    echo "  singularity not on PATH here, so the version was not read"
  fi
  exit 0
fi

command -v singularity >/dev/null 2>&1 || { echo "singularity is required to pull the image" >&2; exit 1; }

# The layers MUST be unpacked on node-local disk. This image carries a .wh.rust whiteout whose
# removal walks rust/target/release/build recursively, and on a parallel filesystem the rapid
# recursive unlink leaves .nfs* placeholders behind, so umoci fails with "directory not empty".
LOCAL="${LOCAL_SCRATCH:-${SLURM_TMPDIR:-${TMPDIR:-/tmp}}}/k31pull.$$"
mkdir -p "$LOCAL" || { echo "cannot create local scratch at $LOCAL" >&2; exit 1; }
FREE=$(df -BG --output=avail "$LOCAL" 2>/dev/null | tail -1 | tr -dc '0-9')
[ "${FREE:-0}" -ge 45 ] || { echo "need >=45G local scratch at $LOCAL, have ${FREE:-0}G" >&2; rmdir "$LOCAL"; exit 1; }
trap 'rm -rf "$LOCAL" 2>/dev/null || true' EXIT

export SINGULARITY_CACHEDIR="$LOCAL/cache"
export SINGULARITY_TMPDIR="$LOCAL/tmp"
mkdir -p "$SINGULARITY_CACHEDIR" "$SINGULARITY_TMPDIR"

DEST="$(dirname "$SIF")"
mkdir -p "$DEST"
echo "pulling $IMAGE, unpacking on $LOCAL (about 15 minutes)"
singularity pull --force "$SIF.partial" "$IMAGE"
singularity exec "$SIF.partial" python3 -c "import sglang,torch; print('  sglang',sglang.__version__,'torch',torch.__version__)"
mv -f "$SIF.partial" "$SIF"
echo "done: $(stat -c %s "$SIF") bytes"
