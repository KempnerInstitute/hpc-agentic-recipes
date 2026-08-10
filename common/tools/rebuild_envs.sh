#!/usr/bin/env bash
# Rebuild recipe environments from their own build scripts.
#
# WHY THIS EXISTS. Retention-based storage. ENV_ROOT defaults to scratch, which expires after 90 days,
# so environments there vanish periodically and every recipe has to be rebuilt from its own build.sh.
# The script itself is not scratch specific and will rebuild any ENV_ROOT.
#
# WHEN YOU NEED THIS. Routinely, if ENV_ROOT points at scratch. Scratch has a 90-day retention policy, so
# environments there vanish periodically and every recipe has to be rebuilt from its own build.sh.
#
# WHEN YOU DO NOT. ENV_ROOT is yours to choose. Point it at lab or project space with no retention
# policy and environments persist indefinitely, so this script is only useful after an engine version
# bump or to repair a corrupted environment. Set it in common/site.conf:
#
#     ENV_ROOT=/n/<your-lab-space>/<you>/agentic-coding-envs
#
# The tradeoff is speed, not correctness: scratch is measurably faster to import from, which is why it
# is the default. See the note in common/defaults.sh.
#
#   rebuild_envs.sh                 rebuild every recipe that has no complete environment
#   rebuild_envs.sh --force         rebuild everything from scratch
#   rebuild_envs.sh <recipe> ...    rebuild only the named recipes, for example GLM-4.6-FP8/h200-4
#
# ENV_ROOT points at scratch, which has a 90-day retention policy, so environments are expected to
# disappear periodically. That is fine: they are disposable, and every recipe can rebuild its own.
#
# Note that not every environment is reproducible from pins alone. A recipe that installs a wheel by URL
# records that URL in env/WHEELS, which is where to look if a rebuild fails on a missing wheel. The RTX
# recipes install from a nightly index that rotates its builds, so a rebuild there resolves to a
# different wheel than the one the rates were measured on, and cannot be pinned back.
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
case "$ENV_ROOT" in
  */netscratch/*|*/scratch/*)
    echo "  ENV_ROOT is on scratch, which expires. Rebuilding here is expected to be routine." ;;
  *)
    echo "  ENV_ROOT is not on scratch, so environments should persist. If one is missing, something"
    echo "  removed it; a rebuild is fine but is not the routine maintenance this script assumes." ;;
esac
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
