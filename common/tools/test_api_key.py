#!/usr/bin/env python3
"""Offline tests for common/lib/api_key.sh. No GPU, no endpoint, no network.

Run: python3 common/tools/test_api_key.py

This resolves which secret an endpoint is gated with, so a mistake here does not merely misconfigure
something, it can leave an endpoint open. Each case builds a throwaway repo root, sources the library the
way serve.sh and client.env do, and reports which file was chosen and whether the key was exported.
"""
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

LIB = pathlib.Path(__file__).resolve().parents[1] / "lib"

FAILURES = []


def check(name, got, want):
    if got == want:
        print("  ok    %s" % name)
        return
    FAILURES.append("%s: got %r, want %r" % (name, got, want))
    print("  FAIL  %s: got %r, want %r" % (name, got, want))


def run(files, key_name=None, key_file=None, preset_key=None, strict=True):
    """Source the library under a temp repo root and report (key, chosen file, warned)."""
    root = pathlib.Path(tempfile.mkdtemp())
    try:
        (root / "common").mkdir()
        shutil.copytree(str(LIB), str(root / "common" / "lib"))
        (root / "secrets").mkdir()
        for name, value in files.items():
            (root / "secrets" / name).write_text(value)
        script = [
            "set -uo pipefail" if strict else "true",
            "REPO_ROOT=%s" % root,
            'source "$REPO_ROOT/common/lib/api_key.sh"',
            'printf "KEY=%s\\n" "${VLLM_API_KEY:-}"',
            'printf "FILE=%s\\n" "${KEY_FILE:-}"',
        ]
        env = dict(os.environ)
        env.pop("VLLM_API_KEY", None)
        env.pop("KEY_NAME", None)
        env.pop("KEY_FILE", None)
        if key_name is not None:
            env["KEY_NAME"] = key_name
        if key_file is not None:
            env["KEY_FILE"] = str(root / key_file)
        if preset_key is not None:
            env["VLLM_API_KEY"] = preset_key
        p = subprocess.Popen(
            ["bash", "-c", "\n".join(script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        out, err = p.communicate()
        out, err = out.decode(), err.decode()
        key = file = ""
        for line in out.splitlines():
            if line.startswith("KEY="):
                key = line[4:]
            elif line.startswith("FILE="):
                file = line[5:]
        chosen = pathlib.Path(file).name if file else ""
        return key, chosen, ("UNGATED" in err)
    finally:
        shutil.rmtree(str(root), ignore_errors=True)


SHARED, SCOPED = "shared-secret", "scoped-secret"
RECIPE = "GLM-5.2-NVFP4-rtx-8"

check(
    "scoped key is preferred when it exists",
    run({"vllm_api_key": SHARED, (RECIPE + ".key"): SCOPED}, key_name=RECIPE),
    (SCOPED, (RECIPE + ".key"), False),
)
check(
    "falls back to the shared key when no scoped key exists",
    run({"vllm_api_key": SHARED}, key_name=RECIPE),
    (SHARED, "vllm_api_key", False),
)
check(
    "another recipe's scoped key is not picked up",
    run({"vllm_api_key": SHARED, "Kimi-K3-h200-4-nodes4-sglang.key": SCOPED}, key_name=RECIPE),
    (SHARED, "vllm_api_key", False),
)
check(
    "with no KEY_NAME the shared key is used, as before",
    run({"vllm_api_key": SHARED}),
    (SHARED, "vllm_api_key", False),
)
check(
    "a scoped key alone is enough, no shared file needed",
    run({(RECIPE + ".key"): SCOPED}, key_name=RECIPE),
    (SCOPED, (RECIPE + ".key"), False),
)
check(
    "an explicit KEY_FILE overrides both",
    run(
        {"vllm_api_key": SHARED, (RECIPE + ".key"): SCOPED, "chosen.key": "explicit-secret"},
        key_name=RECIPE,
        key_file="secrets/chosen.key",
    ),
    ("explicit-secret", "chosen.key", False),
)
check(
    "no key anywhere warns that the endpoint would be ungated",
    run({}, key_name=RECIPE),
    ("", (RECIPE + ".key"), True),
)
check(
    "a key already in the environment suppresses the warning",
    run({}, key_name=RECIPE, preset_key="from-environment"),
    ("from-environment", (RECIPE + ".key"), False),
)
check(
    "an exported key wins over the recipe file, so the tools follow a sourced client.env",
    run({"vllm_api_key": SHARED, RECIPE + ".key": SCOPED}, key_name=RECIPE, preset_key="from-environment"),
    ("from-environment", (RECIPE + ".key"), False),
)
check(
    "an exported key wins over the shared file too",
    run({"vllm_api_key": SHARED}, preset_key="from-environment"),
    ("from-environment", "vllm_api_key", False),
)
check(
    "two recipes serving one model get different keys",
    (
        run({"GLM-5.2-NVFP4-rtx-8.key": "rtx-secret",
             "GLM-5.2-FP8-h200-4-nodes2.key": "h200-secret"}, key_name="GLM-5.2-NVFP4-rtx-8")[0],
        run({"GLM-5.2-NVFP4-rtx-8.key": "rtx-secret",
             "GLM-5.2-FP8-h200-4-nodes2.key": "h200-secret"}, key_name="GLM-5.2-FP8-h200-4-nodes2")[0],
    ),
    ("rtx-secret", "h200-secret"),
)
check(
    "a trailing newline in the file is stripped",
    run({"vllm_api_key": SHARED + "\n"}),
    (SHARED, "vllm_api_key", False),
)
check(
    "a served name containing a dot resolves",
    run({"glm-5.2.key": SCOPED}, key_name="glm-5.2"),
    (SCOPED, "glm-5.2.key", False),
)

# serve.sh runs under set -u, so an unset KEY_NAME or KEY_FILE must not abort the caller.
key, chosen, _ = run({"vllm_api_key": SHARED}, strict=True)
check("sourcing under set -u does not abort on unset variables", (key, chosen), (SHARED, "vllm_api_key"))

if FAILURES:
    print("%d FAILURE(S):" % len(FAILURES))
    for f in FAILURES:
        print("  " + f)
    sys.exit(1)
print("all tests pass")
