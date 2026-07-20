#!/usr/bin/env bash
# Check the vLLM endpoint: list models and run one chat completion.
# Usage: smoke_test.sh [host] [port]
set -euo pipefail
HOST="${1:-holygpu8a10101}"
PORT="${2:-8000}"
BASE="http://$HOST:$PORT/v1"
echo "== models =="
curl -s "$BASE/models" | python3 -m json.tool
echo "== chat =="
curl -s "$BASE/chat/completions" -H 'Content-Type: application/json' -d '{
  "model": "glm-5.2",
  "messages": [{"role": "user", "content": "Write a Python function returning the nth Fibonacci number, then explain its time complexity."}],
  "max_tokens": 300,
  "temperature": 0.2
}' | python3 -m json.tool
