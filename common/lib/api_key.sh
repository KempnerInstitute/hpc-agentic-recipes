# Load the endpoint API key into VLLM_API_KEY. Source this, do not execute.
#
# The key is passed to the engine through the environment, never on the command line, so it does not
# appear in ps output. It is read from a gitignored file so no key is ever committed.
#
# If the file is missing this prints a warning rather than failing silently. A silent empty key makes
# every request return HTTP 401, and the symptom looks like an auth bug instead of a missing file.
[ -n "${REPO_ROOT:-}" ] || source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/repo_root.sh"
KEY_FILE="${KEY_FILE:-$REPO_ROOT/secrets/vllm_api_key}"
if [ -f "$KEY_FILE" ]; then
  VLLM_API_KEY="$(tr -d '\n\r' < "$KEY_FILE")"
  export VLLM_API_KEY
elif [ -z "${VLLM_API_KEY:-}" ]; then
  echo "warning: no API key at $KEY_FILE and VLLM_API_KEY is unset." >&2
  echo "         The endpoint will be UNGATED. Create one with:" >&2
  echo "           mkdir -p '$REPO_ROOT/secrets'" >&2
  echo "           printf '%s' 'sk-local-\$(openssl rand -hex 24)' > '$KEY_FILE'" >&2
  echo "           chmod 600 '$KEY_FILE'" >&2
fi
