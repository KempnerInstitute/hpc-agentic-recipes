#!/usr/bin/env bash
# Verify that every recipe is standalone, complete, and internally consistent.
#
# Recipes deliberately duplicate text: an issue that affects nine recipes is written out in all nine,
# because a user reading one recipe must not have to open another file. Hand-maintained duplication
# rots, so it is generated from a single source and verified here.
#
#   audit_recipes.sh          verify, exit non-zero on any failure
#   audit_recipes.sh --fix    inject the shared issue text into marked blocks, then verify
#
# Blocks are delimited in recipe READMEs by:
#   <!-- issue:<slug> begin -->  ...  <!-- issue:<slug> end -->
set -uo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/../lib/repo_root.sh"
ISSUES="$REPO_ROOT/common/issues"
RECIPES="$REPO_ROOT/recipes"
FIX=0
[ "${1:-}" = "--fix" ] && FIX=1

fail=0
note () { printf '  %s\n' "$*"; }
bad () { printf 'FAIL  %s\n' "$*"; fail=$((fail + 1)); }

REQUIRED_FILES="README.md env/build.sh env/env.sh serve.sh serve.sbatch serve_ssh.sh client.env"
REQUIRED_HEADINGS="Configure once|Status|What this is|Hardware|Environment build|Launch|Verify|Connect a client|Tunable inputs|Web search|Measured performance|Parallelism and quantization|Gotchas|Stop the endpoint|Expected startup time"

# All recipe variant directories, as <Checkpoint-Name>/<hardware>.
all_recipes () {
  [ -d "$RECIPES" ] || return 0
  for d in "$RECIPES"/*/*/; do
    [ -d "$d" ] || continue
    printf '%s\n' "${d#"$RECIPES"/}" | sed 's:/$::'
  done
}

readme_only () {
  local r="$1"
  [ -f "$ISSUES/readme-only.txt" ] || return 1
  grep -qxF "$r" "$ISSUES/readme-only.txt"
}

# Expand a comma separated glob list against the recipes that exist.
expand () {
  local patterns="$1" out=""
  [ "$patterns" = "-" ] && return 0
  local IFS=,
  for p in $patterns; do
    while read -r r; do
      # shellcheck disable=SC2254
      case "$r" in $p) out+="$r"$'\n' ;; esac
    done < <(all_recipes)
  done
  printf '%s' "$out" | sed '/^$/d' | sort -u
}

echo "== recipes found"
mapfile -t RECIPE_LIST < <(all_recipes)
[ "${#RECIPE_LIST[@]}" -gt 0 ] || { echo "no recipes under $RECIPES"; exit 1; }
note "${#RECIPE_LIST[@]} recipe variants"

echo "== structure and required sections"
for r in "${RECIPE_LIST[@]}"; do
  d="$RECIPES/$r"
  if readme_only "$r"; then
    [ -f "$d/README.md" ] || bad "$r: documentation-only recipe has no README.md"
    continue
  fi
  for f in $REQUIRED_FILES; do
    [ -f "$d/$f" ] || bad "$r: missing $f"
  done
  [ -f "$d/README.md" ] || continue
  # Headings must all be present and in the mandated order.
  got="$(grep -oE "^## ($REQUIRED_HEADINGS)$" "$d/README.md" | sed 's/^## //')"
  want="$(printf '%s' "$REQUIRED_HEADINGS" | tr '|' '\n')"
  missing="$(comm -23 <(printf '%s\n' "$want" | sort) <(printf '%s\n' "$got" | sort))"
  [ -n "$missing" ] && bad "$r: README missing sections: $(printf '%s' "$missing" | paste -sd', ')"
  if [ -z "$missing" ] && [ "$got" != "$want" ]; then
    bad "$r: README sections are out of the mandated order"
  fi
done

echo "== status lines"
for r in "${RECIPE_LIST[@]}"; do
  f="$RECIPES/$r/README.md"
  [ -f "$f" ] || continue
  line="$(grep -m1 '^Status:' "$f" || true)"
  [ -n "$line" ] || { bad "$r: no Status: line"; continue; }
  case "$line" in
    "Status: Blocked"*) ;;
    "Status: Untested"*) ;;
    "Status: Validated"*)
      case "$line" in
        *", protocol: "*) ;;
        *) bad "$r: Validated status must name a measurement protocol: $line" ;;
      esac
      grep -qE '^Status: Validated - .*[0-9]' "$f" \
        || bad "$r: Validated status must name the engine version it was measured with: $line"
      ;;
    *) bad "$r: unrecognized status: $line" ;;
  esac
done

echo "== issue propagation"
python3 "$S/issue_blocks.py" "$REPO_ROOT" ${FIX:+$([ "$FIX" = 1 ] && echo --fix)} || fail=$((fail + 1))

echo "== standalone rule: no cross-references for required information"
for r in "${RECIPE_LIST[@]}"; do
  f="$RECIPES/$r/README.md"
  [ -f "$f" ] || continue
  hits="$(grep -nEi 'see (the )?(top-level |root )?README|see docs/|described in docs/|refer to docs/' "$f" | grep -viE 'additional context|further reading' || true)"
  [ -n "$hits" ] && bad "$r: README points elsewhere for information: $(printf '%s' "$hits" | head -1 | cut -c1-90)"
done

echo "== flag provenance in env/env.sh"
for r in "${RECIPE_LIST[@]}"; do
  f="$RECIPES/$r/env/env.sh"
  [ -f "$f" ] || continue
  # Only the flags that encode a decision need provenance. PATH, cache dirs and compiler plumbing are
  # self-evident and are covered by the group comment above them.
  while IFS= read -r ln; do
    n="${ln%%:*}"; var="$(printf '%s' "${ln#*:}" | sed 's/^export \([A-Z_]*\)=.*/\1/')"
    case "$var" in
      VLLM_*|NCCL_*|TORCH_NCCL_*|FLASHINFER_*|CUDA_HOME|VLLM_ATTENTION_BACKEND) ;;
      *) continue ;;
    esac
    ctx="$(sed -n "$((n > 3 ? n - 3 : 1)),$((n - 1))p" "$f")"
    case "$ctx" in
      *verified:*|*inherited*|*required:*|*required*|*measured*) ;;
      *) bad "$r: env.sh line $n sets $var with no provenance comment (verified/inherited/required)" ;;
    esac
  done < <(grep -n '^export [A-Z_]*=' "$f" || true)
done

echo "== mandatory flags"
for r in "${RECIPE_LIST[@]}"; do
  f="$RECIPES/$r/env/env.sh"
  [ -f "$f" ] || continue
  case "$r" in
    */rtx-8*) grep -q 'NCCL_P2P_DISABLE=1' "$f" || bad "$r: RTX multi-GPU recipe must set NCCL_P2P_DISABLE=1 or NCCL init hangs" ;;
  esac
  case "$r" in
    */rtx-8*|*/h200-4*|*/h100-4*) grep -q 'TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC' "$f" || bad "$r: multi-rank recipe must set TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC" ;;
  esac
  case "$r" in
    GLM-5.2-FP8/h200*|GLM-4.6-FP8/h200*) grep -q 'VLLM_USE_DEEP_GEMM=0' "$f" || bad "$r: must set VLLM_USE_DEEP_GEMM=0 on H200" ;;
  esac
done

echo "== sbatch hygiene"
for r in "${RECIPE_LIST[@]}"; do
  f="$RECIPES/$r/serve.sbatch"
  [ -f "$f" ] || continue
  grep -q '^#SBATCH --account=' "$f" && bad "$r: serve.sbatch hardcodes an account; pass it at submit time"
  grep -q '^#SBATCH --reservation=' "$f" && bad "$r: serve.sbatch names a cluster-local node arrangement"  # check_prose: allow, the guard must name the flag it forbids
  grep -qE '^#SBATCH --output=logs/' "$f" && bad "$r: serve.sbatch uses a relative logs/ path, which fails unless submitted from the repo root"
done

echo "== client configuration"
for r in "${RECIPE_LIST[@]}"; do
  c="$RECIPES/$r/client.env"; sv="$RECIPES/$r/serve.sh"
  [ -f "$c" ] || continue
  grep -qE '^[[:space:]]*(export[[:space:]]+)?ANTHROPIC_API_KEY=' "$c" && bad "$r: client.env sets ANTHROPIC_API_KEY, which makes the client send x-api-key and get 401"
  case "$r" in
    *-sglang) ;;
    *)
      grep -q 'ANTHROPIC_AUTH_TOKEN' "$c" || bad "$r: client.env must set ANTHROPIC_AUTH_TOKEN"
      grep -q 'ANTHROPIC_SMALL_FAST_MODEL' "$c" || bad "$r: client.env must set ANTHROPIC_SMALL_FAST_MODEL"
      if [ -f "$sv" ]; then
        served="$(grep -oE -- '--served-model-name [A-Za-z0-9._-]+' "$sv" | head -1 | awk '{print $2}')"
        cm="$(grep -oE 'ANTHROPIC_MODEL=[A-Za-z0-9._-]+' "$c" | head -1 | cut -d= -f2)"
        [ -n "$served" ] && [ -n "$cm" ] && [ "$served" != "$cm" ] \
          && bad "$r: client.env ANTHROPIC_MODEL=$cm does not match serve.sh --served-model-name $served"
      fi
      ;;
  esac
done

echo "== pins"
for r in "${RECIPE_LIST[@]}"; do
  for f in "$RECIPES/$r/env/requirements.lock"; do
    [ -f "$f" ] || continue
    grep -qE '^[A-Za-z0-9._-]+>=' "$f" && bad "$r: $(basename "$f") contains a >= pin; pins must be exact"
  done
done

echo "== comments inside continued commands"
# A comment on a continued line silently terminates the command, dropping every remaining argument.
# bash -n accepts it because it is valid syntax, so only a lint catches it. This shipped once: a
# --kv-cache-dtype flag placed after an explanatory comment never reached the engine, and the model
# failed at init with a message about the value it was supposed to have been given.
while read -r f; do
  python3 - "$f" <<'PYLINT'
import sys, re
path = sys.argv[1]
lines = open(path).read().split("\n")
for i in range(len(lines) - 1):
    if lines[i].rstrip().endswith("\\") and lines[i + 1].lstrip().startswith("#"):
        print("FAIL  %s:%d comment on a continued line silently ends the command" % (path, i + 2))
PYLINT
done < <(find "$REPO_ROOT/recipes" "$REPO_ROOT/common" -name '*.sh' -o -name '*.sbatch' 2>/dev/null) > /tmp/contlint.$$ 2>&1
if [ -s /tmp/contlint.$$ ]; then
  while read -r l; do bad "${l#FAIL  }"; done < /tmp/contlint.$$
fi
rm -f /tmp/contlint.$$

echo "== no client configuration tracked"
# Claude Code writes settings and session state into .claude when run inside a checkout. The shared
# skill lives in common/skills, so nothing under .claude should ever be tracked; committing it would
# leak local state into a public repo.
if git -C "$REPO_ROOT" ls-files --error-unmatch .claude >/dev/null 2>&1; then
  bad "tracked files exist under .claude: $(git -C "$REPO_ROOT" ls-files .claude | paste -sd' ')"
fi

echo "== index coverage"
# Both directions. A recipe missing from the model table is invisible to users, and a table row pointing
# at a directory that does not exist is a broken promise. The second is how two multi-node recipes were
# advertised in the README before they were written.
for p in $(grep -oE '\(recipes/[A-Za-z0-9._/-]+\)' "$REPO_ROOT/README.md" | tr -d '()' | sort -u); do
  [ -e "$REPO_ROOT/$p" ] || bad "README links recipes/$(basename "$p") but $p does not exist"
done
for r in "${RECIPE_LIST[@]}"; do
  model="${r%%/*}"
  grep -qF "recipes/$r" "$REPO_ROOT/README.md" || grep -qF "recipes/$model)" "$REPO_ROOT/README.md" \
    || bad "$r is not referenced from the model table in README.md"
done

echo "== sourcing shared config under set -e"
# A sourced file returns the status of its last command. If common/defaults.sh or a recipe env.sh ends
# on a failed test for an optional file, every caller running under set -e aborts silently with no
# output, which is extremely hard to diagnose. Prove it does not happen.
( set -euo pipefail; source "$REPO_ROOT/common/defaults.sh" ) >/dev/null 2>&1 \
  || bad "common/defaults.sh returns non-zero when sourced under set -e"
for r in "${RECIPE_LIST[@]}"; do
  f="$RECIPES/$r/env/env.sh"
  [ -f "$f" ] || continue
  out="$( set +e; ( set -euo pipefail; source "$f" ) 2>&1 )"
  case "$out" in
    *"no environment at"*) ;;
    *) [ -n "$out" ] && note "$r: env.sh said: $(printf '%s' "$out" | head -1 | cut -c1-70)" ;;
  esac
done

echo "== no dates or internal vocabulary in recipe prose"
# Delegated to check_prose.py because it has to be fence-aware: dates inside code blocks are verbatim
# engine output pasted as evidence, and editing a quoted log line to drop its timestamp would falsify it.
if ! out="$(python3 "$S/check_prose.py" 2>&1)"; then
  # A while-read pipeline runs in a subshell, so incrementing the failure count inside one is lost
  # and the audit prints every finding and then reports PASS.
  while read -r line; do bad "$line"; done < <(echo "$out" | grep -E '^ +(DATE|JARGON)')
fi

echo "== syntax and hygiene"
while read -r f; do
  bash -n "$f" 2>/dev/null || bad "$f: shell syntax error"
done < <(find "$REPO_ROOT/recipes" "$REPO_ROOT/common" -name '*.sh' -o -name '*.sbatch' 2>/dev/null)

leak="$(grep -rlE 'holygpu[0-9]+[a-z]?[0-9]*|/n/home[0-9]+/' "$REPO_ROOT/recipes" "$REPO_ROOT/common" "$REPO_ROOT/docs" 2>/dev/null || true)"
[ -n "$leak" ] && bad "hostnames or home paths in tracked files: $(printf '%s' "$leak" | head -3 | paste -sd' ')"
EMDASH="$(printf '\u2014')"
dash="$(grep -rl "$EMDASH" "$REPO_ROOT/recipes" "$REPO_ROOT/common" "$REPO_ROOT/docs" "$REPO_ROOT/README.md" 2>/dev/null || true)"
[ -n "$dash" ] && bad "em dashes present: $(printf '%s' "$dash" | head -3 | paste -sd' ')"

echo
if [ "$fail" = 0 ]; then
  echo "PASS  ${#RECIPE_LIST[@]} recipes, no findings"
else
  echo "$fail finding(s)"
fi
exit $((fail > 0))
