#!/usr/bin/env bash
# Send one chat message to the served model and print the reply.
# Usage: chat.sh "prompt" [host] [port] [model]
# Sends the API key (VLLM_API_KEY env, else secrets/vllm_api_key) so it works against gated endpoints.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"; REPO_DIR="$(dirname "$S")"
PROMPT="${1:?usage: chat.sh <prompt> [host] [port] [model]}"
HOST="${2:-holygpu8a10101}"
PORT="${3:-8000}"
MODEL="${4:-glm-5.2}"
KEY="${VLLM_API_KEY:-}"
[ -z "$KEY" ] && [ -f "$REPO_DIR/secrets/vllm_api_key" ] && KEY="$(tr -d '\n\r' < "$REPO_DIR/secrets/vllm_api_key")"
CHAT_KEY="$KEY" python3 - "$PROMPT" "$HOST" "$PORT" "$MODEL" <<'PY'
import json, sys, urllib.request, os
prompt, host, port, model = sys.argv[1:5]
hdr = {"Content-Type": "application/json"}
key = os.environ.get("CHAT_KEY")
if key:
    hdr["Authorization"] = f"Bearer {key}"
req = urllib.request.Request(
    f"http://{host}:{port}/v1/chat/completions",
    data=json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                     "max_tokens": 512, "temperature": 0.3}).encode(),
    headers=hdr)
resp = json.load(urllib.request.urlopen(req, timeout=600))
print(resp["choices"][0]["message"]["content"])
PY
