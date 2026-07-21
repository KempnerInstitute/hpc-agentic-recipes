#!/usr/bin/env bash
# Stop vLLM and Ray on both nodes.
set -euo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/config.sh"
for n in "${HEAD:-$GLM52_HEAD}" "${WORKER:-$GLM52_WORKER}"; do
  ssh -o BatchMode=yes "$n" "source '$S/lib_env.sh' 2>/dev/null || true; ray stop 2>/dev/null || true; pkill -f 'vllm serve' 2>/dev/null || true; echo stopped $n"
done
