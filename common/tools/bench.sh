#!/usr/bin/env bash
# Measure sustained decode rate with the slope method.
#
# A single timed generation conflates prefill, scheduling, and detokenization with decode, and
# understates the sustained rate by up to 40 percent. Timing two lengths and dividing the difference
# cancels every fixed cost:
#
#   rate = (long_tokens - short_tokens) / (long_seconds - short_seconds)
#
# Usage:
#   bench.sh --host <host> --model <served-name> [--port 8000] [--repeats 3]
#   bench.sh --host <host> --model <served-name> --single    # old single-generation probe
#
# Prints the median rate and the protocol string to paste into a recipe README.
set -uo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/../lib/repo_root.sh"
source "$S/../lib/api_key.sh"

HOST=""; PORT=8000; MODEL=""; REPEATS=3; SINGLE=0
SHORT_N=128; LONG_N=1152
PROMPT="Explain memory hierarchies in detail."

while [ $# -gt 0 ]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --short) SHORT_N="$2"; shift 2 ;;
    --long) LONG_N="$2"; shift 2 ;;
    --single) SINGLE=1; shift ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$HOST" ] || { echo "--host is required" >&2; exit 2; }
[ -n "$MODEL" ] || { echo "--model is required" >&2; exit 2; }

URL="http://$HOST:$PORT/v1/chat/completions"
AUTH=()
[ -n "${VLLM_API_KEY:-}" ] && AUTH=(-H "Authorization: Bearer $VLLM_API_KEY")

# ignore_eos and temperature 0 make both requests generate exactly max_tokens deterministically,
# which is what lets the two timings differ only by the extra decode steps.
ask () {
  curl -s -m 1800 "${AUTH[@]}" -H 'Content-Type: application/json' "$URL" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$1,\"ignore_eos\":true,\"temperature\":0}" \
    -o /dev/null -w '%{time_total}'
}

if [ "$SINGLE" = 1 ]; then
  echo "WARNING: --single reports a single-generation rate, which understates sustained decode by"
  echo "         up to 40 percent because prefill and fixed costs are counted as decode time."
  t=$(ask "$LONG_N")
  awk -v t="$t" -v n="$LONG_N" 'BEGIN{printf "  %d tokens in %.3fs = %.1f tok/s  protocol: single-generation(%d)\n", n, t, n/t, n}'
  exit 0
fi

command -v python3 >/dev/null || { echo "python3 is required" >&2; exit 1; }
echo "warming up ($HOST:$PORT, model $MODEL)"
ask 64 > /dev/null

rates=()
for i in $(seq 1 "$REPEATS"); do
  s=$(ask "$SHORT_N"); l=$(ask "$LONG_N")
  ok=$(awk -v s="$s" -v l="$l" 'BEGIN{print (l>s)?1:0}')
  if [ "$ok" != 1 ]; then
    echo "  run $i: long request ($l s) was not slower than short ($s s); endpoint may be loaded, skipping" >&2
    continue
  fi
  r=$(awk -v s="$s" -v l="$l" -v a="$SHORT_N" -v b="$LONG_N" 'BEGIN{printf "%.2f", (b-a)/(l-s)}')
  printf '  run %d: %s tok in %.3fs, %s tok in %.3fs -> %s tok/s\n' "$i" "$SHORT_N" "$s" "$LONG_N" "$l" "$r"
  rates+=("$r")
done

[ "${#rates[@]}" -gt 0 ] || { echo "no usable runs" >&2; exit 1; }
printf '%s\n' "${rates[@]}" | python3 -c "
import sys, statistics
v = sorted(float(x) for x in sys.stdin.read().split())
m = statistics.median(v)
print()
print(f'  SUSTAINED DECODE: {m:.1f} tok/s   ({1000/m:.2f} ms/token)')
print(f'  runs: {len(v)}  min {min(v):.1f}  max {max(v):.1f}')
print(f'  protocol: slope($SHORT_N,$LONG_N)')
print()
print(f'  paste into the recipe README: {m:.1f} tok/s, protocol: slope($SHORT_N,$LONG_N)')
"
