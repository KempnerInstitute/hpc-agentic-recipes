#!/usr/bin/env bash
# Verify the staged Mistral-Large-4-1T-A52B-NVFP4 container and report its version. There is no venv for this recipe.
# No vLLM release or nightly wheel carries Mistral Large 4, so the image cannot be rebuilt from a version.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/../../../../common/lib/repo_root.sh"
source "$REPO_ROOT/common/defaults.sh"
source "$S/env.sh"

if [ -f "$SIF" ]; then
  echo "container present: $SIF"
  echo "  $(stat -c %s "$SIF") bytes"
  if command -v singularity >/dev/null 2>&1; then
    _v="$(singularity exec "$SIF" python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null)" || {
      echo "  the image is present but vllm does not import inside it: treat it as unusable" >&2
      exit 1
    }
    echo "  vllm $_v"
  else
    echo "  singularity not on PATH here, so the version was not read"
  fi
  exit 0
fi

echo "container not found: $SIF" >&2
echo "This recipe does not pull an image. Point SIF at a staged copy, or copy the container/" >&2
echo "subdirectory from the shared testbed checkpoint next to your own weights." >&2
exit 1
