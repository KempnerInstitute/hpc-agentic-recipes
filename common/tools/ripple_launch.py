#!/usr/bin/env python3
"""Start SGLang for the Ripple checkpoint, with its model registration and the key from the environment.

The Ripple release ships its SGLang model implementation inside the checkpoint, under
inference/server: model.py defines RicursiveForCausalLM and ripple_format.py defines the ripple tool
and reasoning parsers. Neither is in any SGLang release, so both have to be registered before the
server starts. common/tools/sglang_launch.py cannot be used here, because it starts
sglang.launch_server without that registration and the architecture would be unknown.

Two details decide the shape of this file.

The registration runs at module scope, not under __main__. SGLang spawns its scheduler and detokenizer
in separate processes, and a spawned child re-imports the main module as __mp_main__, which skips
anything guarded by `if __name__ == "__main__"`. Registering there leaves the children without the
model or the parsers, and the scheduler fails with `Unsupported model type: ripple`. The checkpoint's
own serve.py registers at top level for the same reason and says so in a comment.

The checkpoint directory comes from RIPPLE_SERVER_DIR rather than argv, because a spawned child
inherits the environment but is not guaranteed the parent's argv.

This wrapper owns the argument list instead of running the checkpoint's serve.py, which bakes a full
DFlash speculative configuration into its DEFAULT_ARGS and appends the caller's arguments after it. A
caller can override --speculative-algorithm but cannot remove --speculative-num-draft-tokens,
--speculative-draft-window-size or --speculative-draft-attention-backend, so serving without
speculation through serve.py leaves the tokenizer manager expecting speculative metrics that never
arrive and it fails on the first request inside _calculate_spec_decoding_metrics.

SGLang accepts the key only as --api-key, and a command line lands in /proc/<pid>/cmdline, which any
user on the node can read, while /proc/<pid>/environ is mode 400. Taking it from the environment keeps
it out of the command line, as common/tools/sglang_launch.py does for the other SGLang recipes.

Usage:
    RIPPLE_SERVER_DIR=<checkpoint>/inference/server ripple_launch.py [SGLang flags...]
"""
import os
import sys

KEY_VARS = ("VLLM_API_KEY", "SGLANG_API_KEY")

# serve.py sets this before importing SGLang: the draft's positional cache is sized for the full
# target context even when draft attention is windowed. Harmless when serving without speculation.
os.environ.setdefault("SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN", "1")

_SERVER_DIR = os.environ.get("RIPPLE_SERVER_DIR", "").strip()
if not _SERVER_DIR:
    print("ripple_launch.py: set RIPPLE_SERVER_DIR to the checkpoint's inference/server directory",
          file=sys.stderr)
    raise SystemExit(2)
if not os.path.isfile(os.path.join(_SERVER_DIR, "model.py")):
    print(f"ripple_launch.py: no model.py in {_SERVER_DIR}", file=sys.stderr)
    raise SystemExit(1)

# The checkpoint's modules import each other by bare name, so its directory has to come first.
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

from sglang.srt.models.registry import ModelRegistry  # noqa: E402

from model import RicursiveForCausalLM  # noqa: E402
import ripple_format  # noqa: E402

ModelRegistry.models["RicursiveForCausalLM"] = RicursiveForCausalLM
ripple_format.register()


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
    from sglang.srt.entrypoints.http_server import launch_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    args = prepare_server_args(build_argv(sys.argv[1:], os.environ))
    try:
        launch_server(args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
