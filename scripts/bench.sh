#!/usr/bin/env bash
# Rough throughput probe: send N chat requests and report completion tokens/sec.
# Usage: bench.sh [host] [port] [max_tokens] [n] [model]
# Sends the API key (VLLM_API_KEY env, else secrets/vllm_api_key) so it works against gated endpoints.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"; REPO_DIR="$(dirname "$S")"
source "$S/config.sh"
HOST="${1:-$RTX_NODE}"
PORT="${2:-8000}"
MAXTOK="${3:-256}"
N="${4:-3}"
MODEL="${5:-glm-5.2}"
KEY="${VLLM_API_KEY:-}"
[ -z "$KEY" ] && [ -f "$REPO_DIR/secrets/vllm_api_key" ] && KEY="$(tr -d '\n\r' < "$REPO_DIR/secrets/vllm_api_key")"
BENCH_KEY="$KEY" python3 - "$HOST" "$PORT" "$MAXTOK" "$N" "$MODEL" <<'PY'
import json, sys, time, urllib.request, os
host, port, maxtok, n, model = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
prompt = "Write a detailed Python implementation of an LRU cache class with get and put, including docstrings and a short usage example."
hdr = {"content-type": "application/json"}
key = os.environ.get("BENCH_KEY")
if key:
    hdr["Authorization"] = f"Bearer {key}"
def one():
    body = {"model": model, "max_tokens": maxtok, "temperature": 0.2,
            "messages": [{"role": "user", "content": prompt}]}
    t = time.time()
    r = json.load(urllib.request.urlopen(urllib.request.Request(
        f"http://{host}:{port}/v1/chat/completions", data=json.dumps(body).encode(),
        headers=hdr), timeout=600))
    dt = time.time() - t
    u = r["usage"]
    return u["prompt_tokens"], u["completion_tokens"], dt
for i in range(n):
    p, c, dt = one()
    print(f"run {i+1}: prompt={p} completion={c} time={dt:.1f}s  tok/s={c/dt:.1f}")
PY
