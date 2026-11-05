# Ripple-65B-Preview on two H200 GPUs

Status: Validated - SGLang 0.5.19 in a container, protocol: slope(128,1152) at concurrency 1, 8 and 16

| | |
| --- | --- |
| Served name | `ripple` |
| Checkpoint | `Ripple-65B-Preview`, Hugging Face `ricursiveAI/Ripple-65B-Preview` |
| On disk | 121 GiB across 16 shards, BF16, `RicursiveForCausalLM` |
| Served precision | BF16 weights, BF16 KV cache; the launcher refuses anything else |
| Context | 262144. The checkpoint declares 2097152 |
| Hardware | 2 H200 GPUs of one node, 143771 MiB each, TP2 |
| Engine | SGLang 0.5.19 in Singularity, `fa3` attention, no speculative decoding |
| Allocation for the measurement | 2 GPUs of one node, 32 cores and 720 GB, `kempner_h200` |

Ripple-65B is a dense 65B transformer from Ricursive Intelligence, pretrained on 12T tokens, with a
hybrid sliding-window and full-attention backbone at a 1024-token window and 80 layers.

**Two settings here differ from what the checkpoint's own launcher uses, and both matter for
correctness rather than speed.** See Known limits before changing `ATTN` or `SPEC_MODE`.

## 1. Create the API key

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/Ripple-65B-Preview-h200-2.key
chmod 600 secrets/Ripple-65B-Preview-h200-2.key
```

- `secrets/vllm_api_key` is read when this file is absent.
- Requests without the key receive HTTP 401.
- The key reaches SGLang through `common/tools/ripple_launch.py`, which reads it from the environment
  and appends `--api-key` inside the process, keeping it out of `/proc/<pid>/cmdline`.

## 2. Build the environment

A venv is not an option: sglang 0.5.19 publishes only `manylinux_2_34` wheels, which need glibc 2.34,
and the nodes here are Rocky 8 with glibc 2.28. The engine is therefore the published image
`lmsysorg/sglang:v0.5.19-cu130`, the version the checkpoint's own launcher pins.

```
bash recipes/Ripple-65B-Preview/h200-2/env/build.sh
```

That verifies a staged image and reports its SGLang version, and pulls it only when the image is
missing. `env/env.sh` resolves it in order: an exported `SIF`, then
`$MODELS_DIR/Ripple-65B-Preview/container/sglang-0.5.19.sif`, then the shared testbed copy.

A copy of this checkpoint needs the `container/` subdirectory alongside the weights, 12.7 GiB on top
of the 121 GiB. There is no venv to build.

## 3. Launch

```
sbatch --account=<your-account> recipes/Ripple-65B-Preview/h200-2/serve.sbatch
```

On a node you already hold, naming the two GPUs:

```
bash recipes/Ripple-65B-Preview/h200-2/serve_ssh.sh <node> 0,1
```

Startup is about 80 seconds with a warm page cache and a few minutes cold. The log is
`$RIPPLE_LOG_DIR/ripple-sglang.log`, default under `/tmp/$USER/ripple`.

## 4. Verify

```
KEY=$(cat secrets/Ripple-65B-Preview-h200-2.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the node>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/completions \
  -d '{"model":"ripple","prompt":"The capital of France is","max_tokens":14,"temperature":0}'

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"ripple","messages":[{"role":"user","content":"Write a Python function that returns the n-th Fibonacci number iteratively."}],"max_tokens":600,"temperature":0.6}'
```

| Command | Expected |
| --- | --- |
| `/v1/models` with the key | `"id": "ripple"` |
| `/v1/models` without a key | `401` |
| raw completion | a coherent continuation, for example ` Paris. The capital of Germany is Berlin.` |
| chat completion | a correct Fibonacci function, `finish_reason: stop` |

**Run both generation probes, not just a short one.** A wrong attention backend on this hardware
serves happily, answers `/v1/models`, and returns `4` for "what is 2+2", while every longer generation
degenerates. The raw completion and the multi-line chat answer are what expose it. See Known limits.

## 5. Connect a client

```
export NODE=<the node>
source recipes/Ripple-65B-Preview/h200-2/client.env
claude
```

`client.env` exports the base URL, the key, the served model name, and
`CLAUDE_CODE_MAX_CONTEXT_TOKENS=262144`, without which Claude Code assumes and enforces a 200k window.
Use `ANTHROPIC_AUTH_TOKEN`, not `ANTHROPIC_API_KEY`, which sends a header the engine ignores and
yields 401 on every request.

## 6. Stop it

```
scancel <job id>
```

For an ssh launch, kill the `ripple_launch.py` process on the node. Kill by that name rather than by
GPU: the other two GPUs on the node may belong to someone else.

## Tunable inputs

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/Ripple-65B-Preview` | Serve a different copy of the checkpoint |
| `DRAFT` | `$MODELS_DIR/Ripple-65B-Preview-DFlash2` | DFlash2 draft, used only when `SPEC_MODE=dflash` |
| `SIF` | resolved as in step 2 | Container image |
| `API_PORT` | 8000 | Listening port |
| `MAX_MODEL_LEN` | 262144 | Context window, fixed at init |
| `MEM_FRACTION` | 0.80 | Static VRAM fraction for weights plus KV |
| `TP` | 2 | Tensor parallel size. One GPU cannot hold this model, see Known limits |
| `ATTN` | `fa3` | Attention backend. Do not change without reading Known limits |
| `PAGE_SIZE` | 64 | Tokens per KV page |
| `MAX_RUNNING` | 16 | Concurrent request cap, which also caps the sweep |
| `SPEC_MODE` | `none` | Set to `dflash` to try the shipped draft; it does not work here |
| `SPEC_TOKENS` | 16 | DFlash draft tokens, only with `SPEC_MODE=dflash` |
| `SPEC_WINDOW` | 16384 | DFlash draft window, only with `SPEC_MODE=dflash` |
| `EXTRA_ARGS` | unset | Extra flags appended to the launch command |
| `KEY_NAME`, `KEY_FILE`, `VLLM_API_KEY` | this recipe's key | Which key `common/lib/api_key.sh` resolves |

## Benchmarking

Conditions:

| | |
| --- | --- |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| Input length | ISL 23 tokens. Rates at a long input are not measured; use `--prompt-tokens` |
| Output length | OSL 1152 tokens, output only, `ignore_eos` |
| Context | `MAX_MODEL_LEN=262144`, `MEM_FRACTION=0.80`, `ATTN=fa3`, no speculation |
| Allocation for the measurement | 2 GPUs of one node, 32 cores and 720 GB, `kempner_h200` |
| Sequence cap | `max_running_requests` 16, which equals the top sweep level |
| Endpoint | idle, and the benchmark client ran on the serving node |

Results:

| Concurrency | Aggregate | Per stream | Spread over 3 runs |
| --- | --- | --- | --- |
| 1 | 40.5 tok/s | 40.5 tok/s | 40.5 to 40.5 |
| 8 | 307.3 tok/s | 38.4 tok/s | 307.3 to 307.4 |
| 16 | 587.7 tok/s | 36.7 tok/s | 587.6 to 590.5 |

| | |
| --- | --- |
| Engine facts | KV cache 392,960 tokens, 1.50 full-length requests at once; 26.38 GB free per GPU after sizing |
| Label | capped. The sweep stops at 16 because the engine admits no more, not because throughput turned over |

Per-stream throughput falls only from 40.5 to 36.7 tok/s between one caller and sixteen, so two GPUs
serve a small group with little latency cost to each caller.

## Known limits

- **The attention backend must be `fa3`, and the wrong one fails silently.** The checkpoint's own
  launcher defaults to `--attention-backend trtllm_mha`, which its README measured on two B200s. On
  H200 that backend serves, answers `/v1/models`, survives inference and returns `'4'` for
  "what is 2+2", but every longer generation degenerates:
  `"Reply with exactly one word: apple"` returned
  `'apple\napple\napple\n<think>\n<think>\nWeigh\nWeigh...'`, an iterative Fibonacci request returned
  `'def fib(lst) -> 2'` repeating and degrading to `'defb -> 2'`, and a raw completion of
  "The capital of France is" returned `' Paris* the capital of France is Paris*'`. With `fa3` the same
  prompts return correct code and `' Paris. The capital of Germany is Berlin.'`. Nothing in the logs
  marks the difference, so only a multi-token generation reveals it.
- **DFlash speculative decoding does not work here**, which is why `SPEC_MODE` defaults to `none`
  although the 7.5 GiB draft is downloaded. With the draft enabled the engine reports
  `DFLASH fused KV materialization disabled: non-neox RoPE is not supported for fused KV path:
  layer=4, rope_is_neox_style=False` and then dies with
  `AssertionError: Mask is required for speculative decoding`.
- A partial speculative configuration is worse than either extreme. The checkpoint's `serve.py` bakes
  the whole DFlash setup into its defaults and appends caller flags after them, so passing
  `--speculative-algorithm NONE` through it leaves the other draft flags set and the first request
  fails with `IndexError` in `_calculate_spec_decoding_metrics`. `common/tools/ripple_launch.py`
  therefore owns the argument list instead of running `serve.py`, and sets all five draft flags
  together or none.
- `common/tools/ripple_launch.py` registers the model and parsers at module scope, not under
  `if __name__ == "__main__"`. SGLang spawns its scheduler as a separate process that re-imports the
  module as `__mp_main__` and skips a `__main__` block, which leaves the children without the
  registration and fails with `ValueError: Unsupported model type: ripple`.
- One GPU is not enough. The 121 GiB of weights against about 129 GiB usable leaves no room for the
  KV cache, and at the 0.80 memory fraction the budget is smaller than the weights.
- Throughput above concurrency 16 is unmeasured, because the engine admits no more.
- Decode rate at a long input is not measured. Every figure here uses a 23-token input.
- Context is served at 262144 rather than the checkpoint's 2097152, which is untested here.
