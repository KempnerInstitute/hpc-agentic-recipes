#!/usr/bin/env bash
# Check a served endpoint: list models and run one chat completion (sends the API key).
# Usage: smoke_test.sh [host] [port] [model]
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/../lib/repo_root.sh"
source "$S/../defaults.sh"
source "$S/../lib/api_key.sh"
HOST="${1:?usage: smoke_test.sh <host> [port] [model]}"
PORT="${2:-$API_PORT}"
MODEL="${3:-glm-5.2}"
BASE="http://$HOST:$PORT/v1"
KEY="${VLLM_API_KEY:-}"
AUTH=(); [ -n "$KEY" ] && AUTH=(-H "Authorization: Bearer $KEY")
echo "== models =="
curl -s "${AUTH[@]}" "$BASE/models" | python3 -m json.tool
echo "== chat =="
curl -s "${AUTH[@]}" "$BASE/chat/completions" -H 'Content-Type: application/json' -d "{
  \"model\": \"$MODEL\",
  \"messages\": [{\"role\": \"user\", \"content\": \"Write a Python function returning the nth Fibonacci number, then explain its time complexity.\"}],
  \"max_tokens\": 300,
  \"temperature\": 0.2
}" | python3 -m json.tool
