#!/usr/bin/env bash
# Measure decode rate. Thin wrapper around bench.py, which does the timing and the concurrency.
#
# Two different numbers, and it matters which one you quote:
#
#   Single stream (the default)   how fast one interactive session feels. The right number for
#                                 agentic coding, where one person waits on one response.
#   Saturated (--concurrency N)   total tokens per second across N simultaneous requests. Higher,
#                                 because continuous batching decodes many sequences per forward
#                                 pass. The right number when serving several users at once.
#
# Usage:
#   bench.sh --host <host> --model <served-name>                    single stream
#   bench.sh --host <host> --model <name> --concurrency 16          saturated at 16
#   bench.sh --host <host> --model <name> --sweep 1,4,16,32,64      find where throughput plateaus
#   bench.sh --host <host> --model <name> --prompt-tokens 8000      decode at a realistic context
#   bench.sh --host <host> --model <name> --single                  old biased probe, for comparison
#
# Both modes use the slope method: time the same request at 128 and 1152 output tokens and divide the
# difference, which cancels prefill, queueing and detokenization because none of those scale with the
# number of output tokens. A single timed generation counts all of it as decode and understates the
# rate by up to 40 percent, worst on the fastest models.
#
# Prompt length is a separate axis. It does not disturb the slope, but a long context slows every
# decode step, because attention reads a larger KV cache per token. Measure at the context you
# actually use before quoting a number for long-context work.
set -uo pipefail
S="$(cd "$(dirname "$0")" && pwd)"
source "$S/../lib/repo_root.sh"
source "$S/../lib/api_key.sh"
exec python3 "$S/bench.py" --key "${VLLM_API_KEY:-}" "$@"
