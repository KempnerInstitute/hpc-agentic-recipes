#!/usr/bin/env bash
# Stop vLLM and Ray on both nodes.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
for n in "${HEAD:-holygpu8a10101}" "${WORKER:-holygpu8a10102}"; do
  ssh -o BatchMode=yes "$n" "source '$S/lib_env.sh' 2>/dev/null || true; ray stop 2>/dev/null || true; pkill -f 'vllm serve' 2>/dev/null || true; echo stopped $n"
done
