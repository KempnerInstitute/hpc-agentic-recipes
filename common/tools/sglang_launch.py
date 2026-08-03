#!/usr/bin/env python3
"""Start SGLang's server with the API key taken from the environment.

SGLang accepts the key only as --api-key. A command line lands in /proc/<pid>/cmdline, which any user on
the node can read, while /proc/<pid>/environ is mode 400 and readable only by its owner and root. vLLM
reads VLLM_API_KEY from the environment for this reason and the recipes here pass it no key flag; this
gives SGLang the same property.

The list Python exposes as sys.argv is a copy. The kernel fills /proc/<pid>/cmdline at exec time from the
original argument array, so appending here never reaches it.

Every other argument is passed through untouched. With no key in the environment the flag is omitted, so
the endpoint is ungated exactly as the engine's own default would leave it. A key already present in the
arguments wins, so an explicit --api-key still behaves as it always did.
"""
import os
import runpy
import sys

KEY_VARS = ("VLLM_API_KEY", "SGLANG_API_KEY")


def key_from_env(env):
    for name in KEY_VARS:
        value = (env.get(name) or "").strip()
        if value:
            return value
    return None


def build_argv(argv, env):
    if "--api-key" in argv or any(a.startswith("--api-key=") for a in argv):
        return list(argv)
    key = key_from_env(env)
    return list(argv) if key is None else list(argv) + ["--api-key", key]


if __name__ == "__main__":
    sys.argv = build_argv(sys.argv, os.environ)
    runpy.run_module("sglang.launch_server", run_name="__main__")
