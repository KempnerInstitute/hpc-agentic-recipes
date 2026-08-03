#!/usr/bin/env bash
# Scaffold a new recipe by copying the nearest existing one, so it passes the audit by construction.
#
#   new_recipe.sh <Checkpoint-Name> <hardware> --from <Checkpoint-Name>/<hardware>
#
# Hardware names are <gpu-type>-<gpus-per-node>[-nodes<N>][-<engine>], for example rtx-8, h200-4,
# h100-1, h200-4-nodes2.
#
# Copy by toolchain, not by model similarity: an RTX source brings torch cu130, the conda CUDA 13
# toolkit and FlashInfer 0.6.15, while a Hopper source brings the cu129 release wheel. Getting this
# wrong means rewriting env/build.sh from scratch.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/../lib/repo_root.sh"
source "$REPO_ROOT/common/defaults.sh"

NAME="${1:-}"; HW="${2:-}"; FROM=""
shift 2 2>/dev/null || true
while [ $# -gt 0 ]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$NAME" ] && [ -n "$HW" ] && [ -n "$FROM" ] || {
  echo "usage: $(basename "$0") <Checkpoint-Name> <hardware> --from <Checkpoint-Name>/<hardware>" >&2
  echo "existing recipes to copy from:" >&2
  (cd "$REPO_ROOT/recipes" && for d in */*/; do echo "  ${d%/}"; done) >&2
  exit 2
}

SRC="$REPO_ROOT/recipes/$FROM"
DST="$REPO_ROOT/recipes/$NAME/$HW"
[ -d "$SRC" ] || { echo "source recipe not found: recipes/$FROM" >&2; exit 1; }
[ -e "$DST" ] && { echo "destination already exists: recipes/$NAME/$HW" >&2; exit 1; }

if [ ! -d "$MODELS_DIR/$NAME" ]; then
  echo "note: no checkpoint directory named $NAME under $MODELS_DIR."
  echo "      Recipe directory names must match the checkpoint directory exactly."
fi

mkdir -p "$(dirname "$DST")"
cp -r "$SRC" "$DST"
# The copied recipe still describes its source. Blank the parts that must not be inherited: a status
# claiming validation it has not earned, and measured numbers belonging to different hardware.
python3 - "$DST/README.md" "$FROM" "$NAME" "$HW" <<'PY'
import re, sys
p, src, name, hw = sys.argv[1:5]
s = open(p).read()
s = re.sub(r"^Status:.*$", "Status: Untested - scaffolded from %s, not yet run end to end" % src, s, count=1, flags=re.M)
s = re.sub(r"^# .*$", "# %s on %s" % (name, hw), s, count=1, flags=re.M)
open(p, "w").write(s)
PY
FROM_NAME="$(dirname "$FROM")"
FROM_HW="$(basename "$FROM")"
STALE=("$FROM_NAME")
[ "$FROM_HW" != "$HW" ] && STALE+=("$FROM_HW")
for pat in "${STALE[@]}"; do
  grep -rl -- "$pat" "$DST" 2>/dev/null \
    | sed "s|^$REPO_ROOT/||;s|^|  still says $pat: |"
done

cat <<MSG

created recipes/$NAME/$HW from $FROM

Next, in order:
  1. Edit env/build.sh and env/env.sh. Every VLLM_, NCCL_, TORCH_NCCL_, FLASHINFER_ or CUDA_HOME export
     needs a provenance comment within the 3 lines above it, or the audit rejects it.
  2. Edit serve.sh for this model: served name, parsers, TP and PP, context length.
  3. Update client.env so ANTHROPIC_MODEL equals serve.sh's --served-model-name.
  4. Leave the <!-- issue:<slug> --> marker pairs empty. A maintainer fills them from common/issues.
  5. Rewrite every section of README.md. Inherited prose from $FROM is wrong until you change it.
  6. Add one row to the model table in README.md.
MSG
