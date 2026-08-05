# gemma-4-26B-A4B-it on one H100 GPU

Status: Validated - vLLM 0.25.1+cu129, protocol: slope(128,1152) at concurrency 1, 64, 256, 640 and 1024

Serves `gemma-4-26b` on one H100 GPU at 256K context. One GPU, no NCCL, no multi-node coordination.

- Checkpoint: `gemma-4-26B-A4B-it`, Hugging Face `google/gemma-4-26B-A4B-it`
- 51.6 GB on disk, bf16, `Gemma4ForConditionalGeneration`, 262144 context

## 1. Create the API key

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/gemma-4-26B-A4B-it-h100-1.key
chmod 600 secrets/gemma-4-26B-A4B-it-h100-1.key
```

The endpoint refuses requests without it. `secrets/vllm_api_key` is read when this file is absent.

## 2. Build the environment

Run this on a compute node, not a login node. About 13 GB, built under `ENV_ROOT` on scratch.

```
bash recipes/gemma-4-26B-A4B-it/h100-1/env/build.sh
```

<!-- issue:hopper-cu129-wheel begin -->
**Hopper nodes need the cu129 wheel, not vLLM's default.** These nodes run NVIDIA driver 575
(CUDA 12.9), which cannot run vLLM's default CUDA 13 PyPI wheel. The recipe installs the cu129
release wheel from the vLLM GitHub release with `--torch-backend=cu129`.
<!-- issue:hopper-cu129-wheel end -->

Scratch has a 90-day retention policy, so rebuild with the same command when it expires.

## 3. Launch

Slurm, from the repo root:

```
sbatch --account=<your-account> recipes/gemma-4-26B-A4B-it/h100-1/serve.sbatch
squeue --me                       # NODELIST gives the host
tail -f gemma26-h100-1-<jobid>.log
```

Direct, on a node you already hold:

```
bash recipes/gemma-4-26B-A4B-it/h100-1/serve_ssh.sh <node>
GPU=1 bash recipes/gemma-4-26B-A4B-it/h100-1/serve_ssh.sh <node>    # pin one device of a shared node
```

On a shared node, pass `GPU=` the device Slurm gave you. Read it with
`scontrol show job -d <jobid>`, in the `GRES=...(IDX:n)` field.

Startup takes 3 to 7 minutes, most of it weight loading.

<!-- issue:engine-ready-timeout begin -->
**Startup exceeds vLLM's default readiness timeout.** Weight load plus torch.compile plus CUDA graph
capture routinely takes longer than the 600 second default, so `env/env.sh` sets
`VLLM_ENGINE_READY_TIMEOUT_S=3600`. A first launch that looks hung is usually still loading; check
the log before killing it.
<!-- issue:engine-ready-timeout end -->

## 4. Verify

```
KEY=$(cat secrets/gemma-4-26B-A4B-it-h100-1.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the node serving it>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models          # must print 401

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"gemma-4-26b","messages":[{"role":"user","content":"What is 2+2? Answer briefly."}],"max_tokens":400}'
```

A keyless request returning 401 is correct.

<!-- issue:thinking-model-max-tokens begin -->
**Give thinking models room, or `content` comes back empty.** This model emits reasoning before its
answer, and vLLM returns that in a separate `reasoning` field, not `reasoning_content`. With a small
budget the whole allowance is spent reasoning, `finish_reason` is `length`, and `content` is empty,
which looks like a broken endpoint but is not. Use at least 400 output tokens for a smoke test, and 800
or more for a model that reasons at length. If `content` is empty, raise the budget before suspecting
the endpoint.
<!-- issue:thinking-model-max-tokens end -->

## 5. Connect a client

```
export NODE=<the node serving it>
source recipes/gemma-4-26B-A4B-it/h100-1/client.env
claude
```

No output-token cap is needed at this context.

<!-- issue:anthropic-auth-token begin -->
**Use `ANTHROPIC_AUTH_TOKEN`, never `ANTHROPIC_API_KEY`.** Both engines accept only
`Authorization: Bearer <key>`. Setting `ANTHROPIC_API_KEY` makes Claude Code send an `x-api-key`
header instead, which the engine ignores, and every request returns HTTP 401. Also set
`ANTHROPIC_SMALL_FAST_MODEL` to this same served model, or the client reaches for a hosted Haiku that
this endpoint does not serve.
<!-- issue:anthropic-auth-token end -->

For an OpenAI-compatible client such as Codex, use base URL `http://<node>:8000/v1`, the same key, and
model name `gemma-4-26b`. See [docs/clients.md](../../../docs/clients.md).

## 6. Stop it

With Slurm, `scancel <jobid>` is enough. For the direct path:

```
bash common/tools/stop.sh <node>
```

On a shared node, confirm only your own device reads 0 MiB; another job may hold the others.

## Tunable inputs

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/gemma-4-26B-A4B-it` | Serve a different copy of the checkpoint |
| `API_PORT` | 8000 | Listening port, and part of the SSH log file name |
| `MAX_MODEL_LEN` | 262144 | Context window, the checkpoint maximum |
| `GPU_UTIL` | 0.90 | Fraction of VRAM for weights plus KV cache |
| `TP` | 1 | Tensor parallel size |
| `QUANT` | unset, meaning bf16 | `fp8` quantizes weights on load; measured no change in rate |
| `KV_FP8` | unset | `--kv-cache-dtype fp8`; halves KV bytes, measured no change in rate |
| `ENFORCE_EAGER` | unset | Skip torch.compile and CUDA graph capture, for debugging a startup failure |
| `SPEC_DRAFT` | unset | Must stay unset on vLLM 0.25.1 and 0.26.0; see Known limits |
| `EXTRA_ARGS` | unset | Extra flags appended to the `vllm serve` command line |
| `TOOL_PARSER` | `gemma4` | Tool call parser |
| `REASONING_PARSER` | `gemma4` | Reasoning parser |
| `GPU` | unset | SSH path only: pin one device of a shared node, for example `GPU=1` |
| `LOG_DIR` | `/tmp/$USER/vllm` | Where the SSH path writes the server log |
| `VENV_DIR` | under `ENV_ROOT` | Use an environment built elsewhere |

`VLLM_CACHE_ROOT` and `MODELS_DIR` come from `common/defaults.sh`; override them there or in
`common/site.conf`.

## Benchmarking

Measured on one H100, vLLM 0.25.1+cu129, at `MAX_MODEL_LEN=262144`, protocol `slope(128,1152)`, three
repeats per level, median reported, against a warm endpoint. The other three GPUs on the node were running
unrelated jobs, and the serving device held its full 1980 MHz clock throughout.

| Concurrency | Aggregate | Per stream |
| --- | --- | --- |
| 1 | 214.7 tok/s | 214.7 tok/s |
| 64 | 5245.6 tok/s | 82.0 tok/s |
| 256 | 6358.4 tok/s | 24.8 tok/s |
| 640 | 7048.6 tok/s | 11.0 tok/s |
| 1024 | 7070.7 tok/s | 6.9 tok/s |

Aggregate varies by 0.3 percent from 640 to 1024, so 7070.7 tok/s is a real ceiling rather than a point on
a rising curve. Quote 214.7 tok/s for one person coding, 7070.7 tok/s for a shared endpoint.

Warm the endpoint before measuring. Taken minutes after startup, concurrency 1 read 204.5 tok/s, 5 percent
low, while concurrency 640 was unaffected: fixed per-request cost dominates at low concurrency and is what
warms up.

The KV cache holds 661,098 tokens, giving 2.52 full-length requests at once. A 240,022-token prompt was
served in 14 seconds.

Reproduce with:

```
KEY_NAME=gemma-4-26B-A4B-it-h100-1 bash common/tools/bench.sh --host <node> --model gemma-4-26b
KEY_NAME=gemma-4-26B-A4B-it-h100-1 bash common/tools/bench.sh --host <node> --model gemma-4-26b \
  --sweep 1,64,256,640,1024
```

`KEY_NAME` is required whenever `secrets/` holds both this recipe's key and `vllm_api_key`: the server
uses the recipe key, while `bench.sh` on its own resolves the shared one and every request returns 401.

## Known limits

<!-- issue:gemma4-mtp-broken begin -->
**Speculative decoding is unavailable for Gemma 4 on this engine.** The drafter checkpoints
(`*-it-assistant`) are downloaded and wired through `SPEC_DRAFT`, but the `gemma4_mtp` method is
broken in both vLLM 0.25.1 and 0.26.0, failing with a shape mismatch:

```
a and b must have same reduction dim, but got [s47, 3840] X [5632, 1024]
```

The drafter expects twice 2816 rather than 5632. The fix (`_maybe_share_embeddings`) exists only on
vLLM's `main` branch, so this needs a build newer than 0.26.0. Do not enable `SPEC_DRAFT` on 0.25.1
or 0.26.0; it fails at the first request.
<!-- issue:gemma4-mtp-broken end -->

<!-- issue:node-local-logs begin -->
**Logs are written to node-local `/tmp`, not to the repo.** Every rank writes stderr for the life of
the endpoint, so a log on a network filesystem puts a blocking write on the critical path. During a
filesystem stall that write hangs, which freezes the server. `LOG_DIR` defaults to
`/tmp/$USER/vllm`, so read logs over SSH on the node that runs the server.
<!-- issue:node-local-logs end -->

## Web search

<!-- issue:anthropic-hosted-tools-400 begin -->
**Anthropic's hosted tools do not work against a local endpoint.** Server-side tools such as
`web_search_20250305`, `web_fetch_20250910` and `code_execution_20250522` are executed by Anthropic's own
API rather than by the model, so no endpoint here can run them. What you see differs by engine, measured
on both.

vLLM rejects all three with HTTP 400, because their definitions carry no `input_schema`:

```
1 validation error:
  {'type': 'missing', 'loc': ('body', 'tools', 0, 'input_schema'), 'msg': 'Field required',
   'input': {'type': 'web_search_20250305', 'name': 'web_search'}}
```

SGLang is harder to diagnose. It accepts `web_search_20250305` with HTTP 200 and drops the tool, logging
that it has no native support, so the model answers without searching and nothing in the reply says why.
It rejects `web_fetch_20250910` and `code_execution_20250522` with HTTP 400.

Client-side tools (file edits, shell, and anything you define) work normally on both. For web access,
install the repo's keyless search tool and skill, from the repo root:

```
mkdir -p ~/.local/bin ~/.claude/skills
ln -sf "$PWD/common/tools/search.sh" ~/.local/bin/search.sh
cp -r common/skills/local-search ~/.claude/skills/
```

Check it with `search.sh wiki "tensor parallelism" 1`, and add `~/.local/bin` to your `PATH` if the
command is not found. The model then searches through `search.sh` (web, arxiv, crossref, pubmed,
openalex, wiki, fetch) instead of the hosted tool. Full details in
[docs/web-search.md](../../docs/web-search.md).
<!-- issue:anthropic-hosted-tools-400 end -->
