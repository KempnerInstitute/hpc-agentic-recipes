#!/usr/bin/env python3
"""Measure decode rate, either single stream or saturated with concurrent requests.

Two different questions, two different numbers:

  Single stream (--concurrency 1, the default)
      How fast does one interactive session feel. This is the number that matters for agentic
      coding, where one person is waiting on one response.

  Saturated (--concurrency N)
      Total tokens per second the endpoint delivers across N simultaneous requests. Higher than
      single stream, because continuous batching decodes many sequences per forward pass, and it
      keeps rising until compute, memory bandwidth, or the KV cache runs out. This is the number
      that matters when serving several users from one endpoint.

Both use the slope method: time the same request at two output lengths and divide the difference,

    rate = concurrency * (long_tokens - short_tokens) / (t_long - t_short)

which cancels prefill, scheduling, queueing, and detokenization, since those do not scale with the
number of output tokens. Timing a single generation instead counts all of that as decode time and
understates the rate by up to 40 percent, with the error growing as the model gets faster.

Prompt length matters separately. Long context does not change prefill's contribution to the slope,
because that cancels, but it does slow every decode step, since attention reads a larger KV cache per
token. Use --prompt-tokens to measure decode at a realistic context length rather than an empty one.
"""
import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

FILLER = (
    "The memory hierarchy of a modern accelerator spans registers, shared memory, L2 cache, and "
    "high bandwidth memory, and each level trades capacity against latency in a way that shapes "
    "how a kernel should be written. "
)


def build_prompt(target_tokens):
    """Roughly target_tokens of filler. The exact count is read back from the response."""
    if target_tokens <= 0:
        return "Explain memory hierarchies in detail."
    words = max(1, int(target_tokens * 0.75))
    text = (FILLER * (words // len(FILLER.split()) + 2)).split()
    return " ".join(text[:words]) + "\n\nSummarize the passage above in detail."


def one_request(url, key, model, prompt, max_tokens, timeout):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        # ignore_eos forces exactly max_tokens so both lengths measure the same kind of work, and
        # temperature 0 keeps runs comparable.
        "ignore_eos": True,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", "Bearer " + key)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.load(resp)
    usage = payload.get("usage") or {}
    return usage.get("completion_tokens"), usage.get("prompt_tokens")


def timed_batch(url, key, model, prompt, max_tokens, concurrency, timeout):
    """Wall time for `concurrency` simultaneous requests to all finish."""
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        t0 = time.monotonic()
        futures = [pool.submit(one_request, url, key, model, prompt, max_tokens, timeout)
                   for _ in range(concurrency)]
        results = [f.result() for f in futures]
        elapsed = time.monotonic() - t0
    completed = [c for c, _ in results if c]
    prompt_tokens = next((p for _, p in results if p), None)
    if len(completed) != concurrency:
        raise RuntimeError("only %d of %d requests returned usage" % (len(completed), concurrency))
    short = min(completed)
    if short != max_tokens:
        # ignore_eos should make this exact. If the server clamped it, the slope denominator is wrong.
        print("  warning: asked for %d output tokens, got %d; rate may be off"
              % (max_tokens, short), file=sys.stderr)
    return elapsed, prompt_tokens, short


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", required=True)
    ap.add_argument("--key", default="")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="simultaneous requests; 1 measures interactive latency, higher saturates")
    ap.add_argument("--prompt-tokens", type=int, default=0,
                    help="pad the prompt to about this many tokens to measure decode at context")
    ap.add_argument("--short", type=int, default=128)
    ap.add_argument("--long", type=int, default=1152)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--sweep", default="",
                    help="comma separated concurrency levels, for example 1,4,16,32")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--single", action="store_true",
                    help="one timed generation instead of the slope; biased, kept for comparison")
    args = ap.parse_args()

    url = "http://%s:%d/v1/chat/completions" % (args.host, args.port)
    prompt = build_prompt(args.prompt_tokens)

    print("warming up (%s:%d, model %s)" % (args.host, args.port, args.model))
    try:
        one_request(url, args.key, args.model, prompt, 64, args.timeout)
    except urllib.error.HTTPError as e:
        print("  request failed: HTTP %s %s" % (e.code, e.read()[:200].decode(errors="replace")))
        return 1
    except Exception as e:  # noqa: BLE001
        print("  request failed: %s" % str(e)[:200])
        return 1

    if args.single:
        print("WARNING: --single times one generation, which counts prefill and fixed per-request")
        print("         cost as decode and understates the sustained rate by up to 40 percent.")
        el, ptok, got = timed_batch(url, args.key, args.model, prompt, args.long, 1, args.timeout)
        print("  %d tokens in %.3fs = %.1f tok/s   protocol: single-generation(%d)"
              % (got, el, got / el, args.long))
        return 0

    levels = [int(x) for x in args.sweep.split(",")] if args.sweep else [args.concurrency]
    rows = []
    for c in levels:
        rates = []
        for i in range(args.repeats):
            try:
                t_s, ptok, _ = timed_batch(url, args.key, args.model, prompt, args.short, c, args.timeout)
                t_l, _, _ = timed_batch(url, args.key, args.model, prompt, args.long, c, args.timeout)
            except Exception as e:  # noqa: BLE001
                print("  concurrency %d run %d failed: %s" % (c, i + 1, str(e)[:140]))
                continue
            if t_l <= t_s:
                print("  concurrency %d run %d: long batch was not slower than short, skipping" % (c, i + 1))
                continue
            rate = c * (args.long - args.short) / (t_l - t_s)
            rates.append(rate)
            print("  c=%-3d run %d: short %.2fs, long %.2fs -> %.1f tok/s aggregate"
                  % (c, i + 1, t_s, t_l, rate))
        if not rates:
            print("  concurrency %d: no usable runs" % c)
            continue
        med = statistics.median(rates)
        rows.append((c, med, ptok))

    if not rows:
        return 1

    print()
    label = "prompt about %d tokens" % rows[0][2] if rows[0][2] else "short prompt"
    for c, med, _ in rows:
        per = med / c
        kind = "SUSTAINED DECODE" if c == 1 else "AGGREGATE THROUGHPUT"
        print("  %s at concurrency %d: %.1f tok/s   (%.2f ms/token, %.1f tok/s per stream)"
              % (kind, c, med, 1000 / med, per))
    print("  protocol: slope(%d,%d) c=%s %s"
          % (args.short, args.long, ",".join(str(c) for c, _, _ in rows), label))
    print()
    best = max(rows, key=lambda r: r[1])
    if len(rows) > 1:
        print("  peak aggregate %.1f tok/s at concurrency %d" % (best[1], best[0]))
        print("  paste into the recipe README: %.1f tok/s single stream, %.1f tok/s aggregate at c=%d,"
              % (rows[0][1], best[1], best[0]))
        print("  protocol: slope(%d,%d)" % (args.short, args.long))
    else:
        c, med, _ = rows[0]
        tag = "single stream" if c == 1 else "aggregate at c=%d" % c
        print("  paste into the recipe README: %.1f tok/s %s, protocol: slope(%d,%d)"
              % (med, tag, args.short, args.long))
    return 0


if __name__ == "__main__":
    sys.exit(main())
