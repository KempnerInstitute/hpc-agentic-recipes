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
understates the rate, by more the shorter the generation. Measured on H200 with a warm endpoint and a
short prompt the gap is under 2 percent, but it is unbounded on a cold endpoint or a long prompt, which is
why the slope is the default.

Prompt length matters separately. Long context does not change prefill's contribution to the slope,
because that cancels, but it does slow every decode step, since attention reads a larger KV cache per
token. Use --prompt-tokens to measure decode at a realistic context length rather than an empty one.
"""
import argparse
import json
import os
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


def is_first_token(delta):
    """Whether a streaming delta carries generated text the user would see.

    A thinking model with a reasoning parser installed emits its first tokens under
    reasoning_content rather than content, and vLLM and SGLang differ on the field name. Watching only
    content on such a model either reports no first token at all or times the first post-reasoning token,
    which can be thousands of tokens late. Role-only preambles and tool-call deltas are not text.
    """
    return any(delta.get(k) for k in ("content", "reasoning_content", "reasoning"))


def slope_rate(concurrency, short_tokens, long_tokens, t_short, t_long):
    """Output tokens per second across all streams, by the slope method.

    Timing two output lengths and dividing the difference cancels prefill and every other cost that does
    not scale with output tokens. Raises if the two timings do not separate, because a non-positive
    denominator means the measurement failed rather than that the model is infinitely fast.
    """
    if long_tokens <= short_tokens:
        raise ValueError("long length %d must exceed short length %d" % (long_tokens, short_tokens))
    if t_long <= t_short:
        raise ValueError("long run (%.3fs) must take longer than short run (%.3fs); "
                         "the slope denominator would be non-positive" % (t_long, t_short))
    return concurrency * (long_tokens - short_tokens) / (t_long - t_short)


def ttft(url, key, model, prompt, timeout):
    """Time to first token, measured on a streaming request. Standard practice reports this next to
    throughput, because it is what determines whether an interactive session feels responsive."""
    body = json.dumps({
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 64, "temperature": 0, "stream": True,
    }).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    if key:
        req.add_header("Authorization", "Bearer " + key)
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            try:
                chunk = json.loads(line[5:].strip())
            except ValueError:
                continue
            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
            if is_first_token(delta):
                return time.monotonic() - t0
    return None


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
        # ignore_eos should make this exact. A clamped count makes the slope denominator wrong, which
        # would silently misreport the rate, so fail instead of warning.
        raise RuntimeError("asked for %d output tokens, server returned %d; slope would be wrong"
                           % (max_tokens, short))
    return elapsed, prompt_tokens, short


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", required=True)
    # Defaults from the environment so the key never reaches a command line, where any user on the
    # node could read it out of /proc.
    ap.add_argument("--key", default=os.environ.get("VLLM_API_KEY", ""))
    ap.add_argument("--concurrency", type=int, default=1,
                    help="simultaneous requests; 1 measures interactive latency, higher saturates")
    ap.add_argument("--prompt-tokens", type=int, default=0,
                    help="pad the prompt to about this many tokens to measure decode at context")
    ap.add_argument("--short", type=int, default=128)
    ap.add_argument("--long", type=int, default=1152)
    ap.add_argument("--repeats", type=int, default=3,
                    help="samples per concurrency level; 3 or more for a publishable number")
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
        print("         cost as decode and understates the sustained rate.")
        el, ptok, got = timed_batch(url, args.key, args.model, prompt, args.long, 1, args.timeout)
        print("  %d tokens in %.3fs = %.1f tok/s   protocol: single-generation(%d)"
              % (got, el, got / el, args.long))
        return 0

    levels = [int(x) for x in args.sweep.split(",")] if args.sweep else [args.concurrency]
    if args.repeats < 2:
        print("NOTE: --repeats %d gives a single sample per level, so there is no median and no way to"
              % args.repeats)
        print("      see run to run spread. Use 3 or more for a number you intend to publish.")
    rows = []
    for c in levels:
        rates = []
        # Warm this exact shape before timing it. Kernel selection and graph capture depend on batch
        # size, so measuring a cold shape charges one-time setup to the first run.
        try:
            timed_batch(url, args.key, args.model, prompt, args.short, c, args.timeout)
        except Exception as e:  # noqa: BLE001
            print("  concurrency %d: warmup failed, %s" % (c, str(e)[:120]))
            continue
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
            rate = slope_rate(c, args.short, args.long, t_s, t_l)
            rates.append(rate)
            print("  c=%-3d run %d: short %.2fs, long %.2fs -> %.1f tok/s aggregate"
                  % (c, i + 1, t_s, t_l, rate))
        if not rates:
            print("  concurrency %d: no usable runs" % c)
            continue
        med = statistics.median(rates)
        first = []
        for _ in range(min(3, max(1, args.repeats))):
            try:
                with ThreadPoolExecutor(max_workers=c) as pool:
                    futs = [pool.submit(ttft, url, args.key, args.model, prompt, args.timeout)
                            for _ in range(c)]
                    vals = [f.result() for f in futs]
                first.extend(v for v in vals if v)
            except Exception:  # noqa: BLE001
                pass
        ttft_med = statistics.median(first) if first else None
        ttft_p90 = sorted(first)[int(len(first) * 0.9)] if len(first) >= 4 else None
        rows.append((c, med, ptok, ttft_med, ttft_p90, min(rates), max(rates), len(rates)))

    if not rows:
        return 1

    print()
    isl = rows[0][2] or 0
    for c, med, _, tmed, tp90, lo, hi, n in rows:
        kind = "SUSTAINED DECODE" if c == 1 else "AGGREGATE THROUGHPUT"
        spread = "" if n < 2 else "  [n=%d, %.1f to %.1f]" % (n, lo, hi)
        print("  %s at concurrency %-3d %8.1f tok/s   (%.2f ms/token, %.1f per stream)%s"
              % (kind, c, med, 1000 / med, med / c, spread))
        if tmed is not None:
            extra = "" if tp90 is None else ", p90 %.0f ms" % (tp90 * 1000)
            print("      time to first token: median %.0f ms%s" % (tmed * 1000, extra))

    print()
    print("  DISCLOSE ALL OF THIS with any rate you quote. A tokens per second figure without the")
    print("  input length, the output length and the concurrency cannot be compared against anything.")
    print("    ISL (input tokens)   %d%s" % (isl, "   <-- very short, best case for decode" if isl < 128 else ""))
    print("    OSL (output tokens)  %d, measured as the slope between %d and %d"
          % (args.long, args.short, args.long))
    print("    concurrency          %s" % ",".join(str(r[0]) for r in rows))
    print("    repeats per level    %d" % args.repeats)
    print("    counted              output tokens only, never input plus output")
    print("    protocol             slope(%d,%d)" % (args.short, args.long))
    if isl < 128:
        print()
        print("  WARNING: a %d token prompt is not a realistic workload. Decode may be faster here than" % isl)
        print("           at a working context. Re-run with --prompt-tokens set before publishing.")

    print()
    single = next((r[1] for r in rows if r[0] == 1), None)
    best = max(rows, key=lambda r: r[1])
    if len(rows) > 1 and single is not None:
        print("  paste into the recipe README:")
        print("    %.1f tok/s single stream, %.1f tok/s aggregate at concurrency %d,"
              % (single, best[1], best[0]))
        print("    ISL %d, OSL %d, protocol slope(%d,%d)" % (isl, args.long, args.short, args.long))
    else:
        c, med = rows[0][0], rows[0][1]
        tag = "single stream" if c == 1 else "aggregate at concurrency %d" % c
        print("  paste into the recipe README:")
        print("    %.1f tok/s %s, ISL %d, OSL %d, protocol slope(%d,%d)"
              % (med, tag, isl, args.long, args.short, args.long))
    return 0


if __name__ == "__main__":
    sys.exit(main())
