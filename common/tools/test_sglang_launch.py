#!/usr/bin/env python3
"""Offline tests for sglang_launch.py. No GPU, no container, no endpoint.

Run: python3 common/tools/test_sglang_launch.py

The point of the launcher is that a key reaches the engine without reaching /proc/<pid>/cmdline. These
cover the argument rewriting; the cmdline property itself is checked by the last test, which starts a real
process and reads its cmdline back.
"""
import importlib.util
import os
import pathlib
import subprocess
import sys
import time

spec = importlib.util.spec_from_file_location(
    "sglang_launch", pathlib.Path(__file__).with_name("sglang_launch.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

FAILURES = []


def check(name, got, want):
    if got == want:
        print("  ok    %s" % name)
        return
    FAILURES.append("%s: got %r, want %r" % (name, got, want))
    print("  FAIL  %s: got %r, want %r" % (name, got, want))


BASE = ["launcher", "--model-path", "/models/Kimi-K3", "--tp-size", "16"]

check(
    "key appended from VLLM_API_KEY",
    mod.build_argv(BASE, {"VLLM_API_KEY": "sk-a"}),
    BASE + ["--api-key", "sk-a"],
)
check(
    "SGLANG_API_KEY also accepted",
    mod.build_argv(BASE, {"SGLANG_API_KEY": "sk-b"}),
    BASE + ["--api-key", "sk-b"],
)
check(
    "VLLM_API_KEY wins over SGLANG_API_KEY",
    mod.build_argv(BASE, {"VLLM_API_KEY": "sk-a", "SGLANG_API_KEY": "sk-b"}),
    BASE + ["--api-key", "sk-a"],
)
check("no key in the environment leaves the arguments alone", mod.build_argv(BASE, {}), BASE)
check(
    "an empty key is treated as absent, not as an empty key",
    mod.build_argv(BASE, {"VLLM_API_KEY": ""}),
    BASE,
)
check(
    "whitespace only is treated as absent",
    mod.build_argv(BASE, {"VLLM_API_KEY": "  \n"}),
    BASE,
)
check(
    "surrounding whitespace is stripped",
    mod.build_argv(BASE, {"VLLM_API_KEY": " sk-c\n"}),
    BASE + ["--api-key", "sk-c"],
)
check(
    "an explicit --api-key is not overridden",
    mod.build_argv(BASE + ["--api-key", "sk-cli"], {"VLLM_API_KEY": "sk-env"}),
    BASE + ["--api-key", "sk-cli"],
)
check(
    "an explicit --api-key= is not overridden",
    mod.build_argv(BASE + ["--api-key=sk-cli"], {"VLLM_API_KEY": "sk-env"}),
    BASE + ["--api-key=sk-cli"],
)
check("the caller's list is not mutated", (BASE, mod.build_argv(BASE, {"VLLM_API_KEY": "sk-a"}) is BASE),
      (BASE, False))

# The property the launcher exists for: a key injected after exec is absent from the process cmdline that
# any user on the node can read. Uses the same technique on a process that only sleeps.
SECRET = "sk-cmdline-probe-2f9c"
probe = pathlib.Path(__file__).with_name(".probe_sglang_launch.py")
probe.write_text(
    "import os, sys, time\n"
    "sys.argv += ['--api-key', os.environ['PROBE_KEY']]\n"
    "print(os.getpid())\n"
    "sys.stdout.flush()\n"
    "time.sleep(8)\n"
)
try:
    proc = subprocess.Popen(
        [sys.executable, str(probe), "--model-path", "/x"],
        stdout=subprocess.PIPE,
        env=dict(os.environ, PROBE_KEY=SECRET),
    )
    pid = proc.stdout.readline().decode().strip()
    time.sleep(1)
    cmdline = pathlib.Path("/proc/%s/cmdline" % pid).read_bytes().decode(errors="replace")
    check("key injected after exec is absent from /proc/pid/cmdline", SECRET in cmdline, False)
    check("the arguments that were passed are still in cmdline", "--model-path" in cmdline, True)
    proc.kill()
    proc.wait()
finally:
    if probe.exists():
        probe.unlink()

if FAILURES:
    print("%d FAILURE(S):" % len(FAILURES))
    for f in FAILURES:
        print("  " + f)
    sys.exit(1)
print("all tests pass")
