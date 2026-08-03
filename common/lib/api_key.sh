# Load the endpoint API key into VLLM_API_KEY. Source this, do not execute.
#
# The key reaches the engine through the environment, never as an argument, so it stays out of ps output
# and out of /proc/<pid>/cmdline, which any user on the node can read. vLLM reads VLLM_API_KEY itself.
# SGLang accepts the key only as --api-key, so that recipe goes through common/tools/sglang_launch.py,
# which supplies it after exec. The file is gitignored, so no key is ever committed.
#
# Where the key comes from, in order. An exported VLLM_API_KEY wins, so a shell that has already sourced a
# recipe's client.env keeps that recipe's key when it runs the tools. Then an explicit KEY_FILE. Then the
# key belonging to this recipe, named by KEY_NAME, so one endpoint's key does not open the others. The
# shared file is the last resort, so a setup that predates per-recipe keys keeps working untouched.
#
# Scoping is per recipe rather than per served model because a recipe is the unit someone takes and runs:
# two recipes serving one model on different hardware are separate endpoints and get separate keys.
#
# If no key is found this warns rather than failing silently. A silent empty key makes every request
# return HTTP 401, and that symptom reads as an auth bug rather than as a missing file.
[ -n "${REPO_ROOT:-}" ] || source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/repo_root.sh"

_ak_shared="$REPO_ROOT/secrets/vllm_api_key"
_ak_scoped="${KEY_NAME:+$REPO_ROOT/secrets/$KEY_NAME.key}"

# The scoped file is chosen when it exists, and also when neither exists, so that the path reported and
# the path the warning tells you to create are the same one.
if [ -z "${KEY_FILE:-}" ]; then
  if [ -n "$_ak_scoped" ] && { [ -f "$_ak_scoped" ] || [ ! -f "$_ak_shared" ]; }; then
    KEY_FILE="$_ak_scoped"
  else
    KEY_FILE="$_ak_shared"
  fi
fi

# Last resort before giving up: a clone with exactly one key in secrets/ has no ambiguity to resolve, and
# the documented tools take a host and a served name rather than a recipe path, so they cannot know which
# recipe's key to ask for. With several keys present this stays quiet and the warning below lists them.
if [ ! -f "$KEY_FILE" ] && [ -z "${VLLM_API_KEY:-}" ]; then
  _ak_only=""
  for _ak_f in "$REPO_ROOT"/secrets/*.key; do
    [ -f "$_ak_f" ] || continue
    [ -n "$_ak_only" ] && { _ak_only=""; break; }
    _ak_only="$_ak_f"
  done
  [ -n "$_ak_only" ] && KEY_FILE="$_ak_only"
  unset _ak_f _ak_only
fi

if [ -n "${VLLM_API_KEY:-}" ]; then
  export VLLM_API_KEY
elif [ -f "$KEY_FILE" ]; then
  VLLM_API_KEY="$(tr -d '\n\r' < "$KEY_FILE")"
  export VLLM_API_KEY
else
  echo "warning: no API key at $KEY_FILE and VLLM_API_KEY is unset." >&2
  _ak_n=0
  for _ak_f in "$REPO_ROOT"/secrets/*.key; do [ -f "$_ak_f" ] && _ak_n=$((_ak_n + 1)); done
  if [ "$_ak_n" -gt 1 ]; then
    echo "         secrets/ holds several keys, so pick one with KEY_NAME=<recipe>:" >&2
    for _ak_f in "$REPO_ROOT"/secrets/*.key; do
      [ -f "$_ak_f" ] && echo "           $(basename "$_ak_f" .key)" >&2
    done
  fi
  unset _ak_f _ak_n
  echo "         Otherwise the endpoint will be UNGATED. Create a key with:" >&2
  echo "           mkdir -p '$REPO_ROOT/secrets'" >&2
  # The suggested command must survive being copied. Single quotes around the substitution would
  # suppress it, writing the literal text as the key, which looks random and is not.
  echo "           printf '%s' \"sk-local-\$(openssl rand -hex 24)\" > '$KEY_FILE'" >&2
  echo "           chmod 600 '$KEY_FILE'" >&2
fi

unset _ak_shared _ak_scoped
