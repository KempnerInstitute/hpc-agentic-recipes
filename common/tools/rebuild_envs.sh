#!/usr/bin/env bash
# Rebuild recipe environments from their own build scripts.
#
#   rebuild_envs.sh                 rebuild every recipe that has no complete environment
#   rebuild_envs.sh --force         rebuild everything from scratch
#   rebuild_envs.sh <recipe> ...    rebuild only the named recipes, for example GLM-4.6-FP8/h200-4
#
# ENV_ROOT points at scratch, which has a 90-day retention policy, so environments are expected to
# disappear periodically. That is fine: they are disposable, and every recipe can rebuild its own.
#
# Note that not every environment is reproducible from pins alone. Recipes that install from a nightly
# wheel index record the exact artifact URLs and hashes in env/WHEELS, because nightly wheels are
# rotated and deleted upstream. If a rebuild fails on a missing wheel, that file is where to look.
set -uo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/../lib/repo_root.sh"
source "$S/../defaults.sh"

FORCE=""
targets=()
for a in "$@"; do
  case "$a" in
    --force) FORCE="--force" ;;
    *) targets+=("$a") ;;
  esac
done

if [ "${#targets[@]}" -eq 0 ]; then
  while read -r d; do targets+=("$d"); done < <(
    cd "$REPO_ROOT/recipes" && for m in */; do
      for h in "$m"*/; do [ -f "$h/env/build.sh" ] && printf '%s\n' "${h%/}"; done
    done
  )
fi

echo "ENV_ROOT: $ENV_ROOT"
built=0; skipped=0; failed=0
for r in "${targets[@]}"; do
  b="$REPO_ROOT/recipes/$r/env/build.sh"
  if [ ! -f "$b" ]; then
    echo "SKIP  $r has no env/build.sh (documentation-only recipe?)"
    skipped=$((skipped + 1)); continue
  fi
  echo "== $r"
  if bash "$b" $FORCE 2>&1 | sed 's/^/  /'; then
    built=$((built + 1))
  else
    echo "  FAILED"
    failed=$((failed + 1))
  fi
done
echo
echo "built or verified $built, skipped $skipped, failed $failed"
exit $((failed > 0))
