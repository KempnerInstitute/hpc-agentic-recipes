#!/usr/bin/env bash
# Verify the staged Muse-Spark NVFP4 container and report its SGLang version. There is no venv for
# this recipe: the engine is a prebuilt image carrying a Muse fork at /sgl-workspace/sglang-muse-spark,
# which no upstream release provides, so a staged image cannot be reconstructed from a version number.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/../../../../common/lib/repo_root.sh"
source "$REPO_ROOT/common/defaults.sh"
source "$S/env.sh"

if [ -f "$SIF" ]; then
  echo "container present: $SIF"
  echo "  $(stat -c %s "$SIF") bytes"
  if command -v singularity >/dev/null 2>&1; then
    _v="$(singularity exec "$SIF" python3 -c "import sglang; print(sglang.__version__)" 2>/dev/null)" || {
      echo "  the image is present but sglang does not import inside it: treat it as unusable" >&2
      exit 1
    }
    echo "  sglang $_v"
  else
    echo "  singularity not on PATH here, so the version was not read"
  fi
  exit 0
fi

echo "container not found: $SIF" >&2
echo "This recipe does not pull an image. The engine carries a Muse fork that no upstream SGLang" >&2
echo "release provides, so point SIF at a staged copy, or copy the container/ subdirectory from" >&2
echo "the shared testbed checkpoint next to your own weights." >&2
exit 1
