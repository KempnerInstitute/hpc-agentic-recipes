#!/usr/bin/env python3
"""Offline tests for bench.py. No GPU, no endpoint, no network.

Run: python3 common/tools/test_bench.py

Covers the two pieces that decide whether a published number is right: the slope arithmetic, and whether
a streaming delta counts as the first token. Both have been wrong before. The first-token check missed
reasoning_content, so a thinking model reported either no TTFT or the first post-reasoning token.
"""
import importlib.util
import pathlib
import sys

spec = importlib.util.spec_from_file_location("bench", pathlib.Path(__file__).with_name("bench.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

FAILURES = []


def check(name, got, want):
    ok = got == want
    if not ok:
        FAILURES.append("%s: got %r, wanted %r" % (name, got, want))
    print("  %s %-52s %r" % ("ok  " if ok else "FAIL", name, got))


def check_raises(name, fn, exc=Exception):
    try:
        fn()
    except exc:
        print("  ok   %-52s raised" % name)
        return
    FAILURES.append("%s: should have raised" % name)
    print("  FAIL %-52s did not raise" % name)


print("is_first_token, across the shapes different engines actually send")
check("plain content, vLLM", bench.is_first_token({"content": "hi"}), True)
check("reasoning_content, SGLang thinking model", bench.is_first_token({"reasoning_content": "let me"}), True)
check("reasoning, older field name", bench.is_first_token({"reasoning": "let me"}), True)
check("role-only preamble is not a token", bench.is_first_token({"role": "assistant"}), False)
check("empty content string is not a token", bench.is_first_token({"content": ""}), False)
check("tool call alone is not text", bench.is_first_token({"tool_calls": [{"id": "x"}]}), False)
check("empty delta", bench.is_first_token({}), False)

print("\nslope_rate, the formula every published number comes from")
# 1024 extra tokens in 10 extra seconds is 102.4 tok/s on one stream.
check("single stream", round(bench.slope_rate(1, 128, 1152, 5.0, 15.0), 1), 102.4)
# Concurrency multiplies: the same per-stream slope across 32 streams.
check("concurrency multiplies aggregate", round(bench.slope_rate(32, 128, 1152, 5.0, 15.0), 1), 3276.8)
# Prefill cancels: adding the same fixed cost to both timings must not change the rate.
check("fixed cost cancels", round(bench.slope_rate(1, 128, 1152, 7.0, 17.0), 1), 102.4)
check("large fixed cost still cancels", round(bench.slope_rate(1, 128, 1152, 100.0, 110.0), 1), 102.4)

print("\nslope_rate guards, so a failed measurement cannot masquerade as a fast one")
check_raises("long run not slower than short", lambda: bench.slope_rate(1, 128, 1152, 10.0, 10.0), ValueError)
check_raises("long run faster than short", lambda: bench.slope_rate(1, 128, 1152, 10.0, 9.0), ValueError)
check_raises("long length not above short", lambda: bench.slope_rate(1, 1152, 1152, 5.0, 15.0), ValueError)
check_raises("long length below short", lambda: bench.slope_rate(1, 1152, 128, 5.0, 15.0), ValueError)

print()
if FAILURES:
    print("%d FAILURE(S):" % len(FAILURES))
    for f in FAILURES:
        print("  " + f)
    sys.exit(1)
print("all tests pass")
