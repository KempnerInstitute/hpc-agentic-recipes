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

if [ -n "${VLLM_API_KEY:-}" ]; then
  export VLLM_API_KEY
elif [ -f "$KEY_FILE" ]; then
  VLLM_API_KEY="$(tr -d '\n\r' < "$KEY_FILE")"
  export VLLM_API_KEY
else
  echo "warning: no API key at $KEY_FILE and VLLM_API_KEY is unset." >&2
  echo "         The endpoint will be UNGATED. Create one with:" >&2
  echo "           mkdir -p '$REPO_ROOT/secrets'" >&2
  # The suggested command must survive being copied. Single quotes around the substitution would
  # suppress it, writing the literal text as the key, which looks random and is not.
  echo "           printf '%s' \"sk-local-\$(openssl rand -hex 24)\" > '$KEY_FILE'" >&2
  echo "           chmod 600 '$KEY_FILE'" >&2
fi

unset _ak_shared _ak_scoped
