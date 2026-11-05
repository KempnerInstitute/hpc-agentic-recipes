# Ripple-65B-Preview on one H200 GPU

Status: Validated - SGLang 0.5.19 in a container, protocol: slope(128,1152) at concurrency 1, 4 and 8

| | |
| --- | --- |
| Served name | `ripple` |
| Checkpoint | `Ripple-65B-Preview`, Hugging Face `ricursiveAI/Ripple-65B-Preview` |
| On disk | 121 GiB across 16 shards, BF16, `RicursiveForCausalLM` |
| Served precision | BF16 weights, BF16 KV cache; the launcher rejects any other `--dtype` or `--kv-cache-dtype` |
| Context | 32768. The checkpoint declares 2097152 |
| Hardware | 1 H200 GPU, 143771 MiB, TP1 |
| Engine | SGLang 0.5.19 in Singularity, `fa3` attention, no speculative decoding |
| Allocation for the measurement | 1 GPU of one node, 16 cores and 360 GB, `kempner_h200` |

This is the cheap shape: one GPU, leaving the other three of an H200 node free. The cost is context
and throughput, and the context limit is a hard one. Use
[`h200-2`](../h200-2/README.md) when you need a long context or more concurrent callers.

| | one GPU | two GPUs |
| --- | --- | --- |
| Usable context | 32768 | 262144 |
| KV pool | 33,728 tokens | 392,960 tokens |
| Concurrent requests | 8 | 16 |
| Single stream | 25.1 tok/s | 40.5 tok/s |
| Aggregate at its cap | 188.8 tok/s at 8 | 587.7 tok/s at 16 |
| `MEM_FRACTION` | 0.93 | 0.80 |

## 1. Create the API key

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/Ripple-65B-Preview-h200-1.key
chmod 600 secrets/Ripple-65B-Preview-h200-1.key
```

- `secrets/vllm_api_key` is read when this file is absent.
- Requests without the key receive HTTP 401.
- The key reaches SGLang through `common/tools/ripple_launch.py`, which reads it from the environment
  and appends `--api-key` inside the process, keeping it out of `/proc/<pid>/cmdline`.
- SGLang then logs the parsed `server_args`, the key among them, so `serve.sh` creates
  `$RIPPLE_LOG_DIR` mode 700 and the log mode 600. Without that the key sits in a world readable file
  on a shared node. Check the mode before trusting an older log, and rotate the key if it was exposed.

## 2. Build the environment

A venv is not an option: sglang 0.5.19 publishes only `manylinux_2_34` wheels, which need glibc 2.34,
and the nodes here are Rocky 8 with glibc 2.28. The engine is therefore the published image
`lmsysorg/sglang:v0.5.19-cu130`, the version the checkpoint's own launcher pins.

```
bash recipes/Ripple-65B-Preview/h200-1/env/build.sh
```

That verifies a staged image and reports its SGLang version, failing if `sglang` does not import
inside it, and pulls the image only when it is missing. `env/env.sh` resolves it in order: an exported
`SIF`, then `$MODELS_DIR/Ripple-65B-Preview/container/sglang-0.5.19.sif`, then the shared testbed copy.

A copy of this checkpoint needs the `container/` subdirectory alongside the weights, 12.7 GiB on top
of the 121 GiB. There is no venv to build.

## 3. Launch

```
sbatch --account=<your-account> recipes/Ripple-65B-Preview/h200-1/serve.sbatch
```

On a node you already hold, naming the GPU:

```
bash recipes/Ripple-65B-Preview/h200-1/serve_ssh.sh <node> 0
```

It refuses to start if that GPU already has a compute process, since the rest of the node may belong
to someone else. Startup is about 100 seconds with a warm page cache. The log is
`$RIPPLE_LOG_DIR/ripple-sglang.log`, default under `/tmp/$USER/ripple`.

## 4. Verify

```
KEY=$(cat secrets/Ripple-65B-Preview-h200-1.key 2>/dev/null || cat secrets/vllm_api_key)
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

Those `curl -H` lines put the key in curl's own argv, which other users on the node can see in
`ps`. That is acceptable for a one-off check on a node you hold alone; use `curl -K <file>` with a
mode 600 file otherwise.

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
source recipes/Ripple-65B-Preview/h200-1/client.env
claude
```

`client.env` exports the base URL, the key, the served model name, and
`CLAUDE_CODE_MAX_CONTEXT_TOKENS=32768`, matching what this shape can actually serve. Use
`ANTHROPIC_AUTH_TOKEN`, not `ANTHROPIC_API_KEY`, which sends a header the engine ignores and yields
401 on every request.

A 32768 token window is small for agentic coding. Expect compaction on long sessions, and prefer
`h200-2` for work that carries a large context.

## 6. Stop it

```
scancel <job id>
```

For an ssh launch, kill the `ripple_launch.py` process on the node. Kill by that name rather than by
GPU: the other GPUs on the node may belong to someone else.

## Tunable inputs

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/Ripple-65B-Preview` | Serve a different copy of the checkpoint |
| `DRAFT` | `$MODELS_DIR/Ripple-65B-Preview-DFlash2` | DFlash2 draft. Only read, and only bound into the container, when `SPEC_MODE=dflash` |
| `SIF` | resolved as in step 2 | Container image |
| `API_PORT` | 8000 | Listening port |
| `MAX_MODEL_LEN` | 32768 | Context window, fixed at init. Raising it does not enlarge the KV pool, see Known limits |
| `MEM_FRACTION` | 0.93 | Static VRAM fraction for weights plus KV. Below about 0.87 the model cannot load |
| `TP` | 1 | Tensor parallel size |
| `ATTN` | `fa3` | Attention backend. Do not change without reading Known limits |
| `PAGE_SIZE` | 64 | Tokens per KV page |
| `MAX_RUNNING` | 8 | Concurrent request cap, which also caps the sweep |
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
| Input length | ISL 17 tokens. Rates at a long input are not measured; use `--prompt-tokens` |
| Output length | OSL 1152 tokens, output only, `ignore_eos` |
| Context | `MAX_MODEL_LEN=32768`, `MEM_FRACTION=0.93`, `ATTN=fa3`, no speculation |
| Allocation for the measurement | 1 GPU of one node, 16 cores and 360 GB, `kempner_h200` |
| Sequence cap | `max_running_requests` 8, which equals the top sweep level |
| Endpoint | idle, and the benchmark client ran on the serving node |

Results:

| Concurrency | Aggregate | Per stream | Spread over 3 runs |
| --- | --- | --- | --- |
| 1 | 25.1 tok/s | 25.1 tok/s | 25.0 to 25.1 |
| 4 | 95.9 tok/s | 24.0 tok/s | 95.8 to 96.0 |
| 8 | 188.8 tok/s | 23.6 tok/s | 188.4 to 189.0 |

| | |
| --- | --- |
| Engine facts | KV cache 33,728 tokens, 1.03 full-length requests at once; 9.37 GB free after sizing |
| Label | capped. The sweep stops at 8 because the engine admits no more, not because throughput turned over |

Per-stream throughput falls only from 25.1 to 23.6 tok/s between one caller and eight, so a single
card serves a small group with little latency cost to each caller. It is 62 percent of the two GPU
single stream rate.

## Known limits

- **The context cannot be raised past about 33,000 tokens on one GPU, whatever you set.** SGLang sizes
  the KV pool from memory left after the weights, not from `--context-length`, so a larger context is
  nominal. Launching with `MAX_MODEL_LEN=262144` starts and reports `context_len=262144`, but the pool
  is still 33,280 tokens and a longer request is rejected:
  `Input length (70001 tokens) exceeds the maximum allowed length (33274 tokens)`, HTTP 400. Use
  `h200-2` for a real long context.
- `MEM_FRACTION` has little headroom here. The weights are 121 GiB against 140.4 GiB on the card, so
  the default 0.80 of the two GPU recipe gives a 112.3 GiB budget and cannot load the model at all.
  0.93 leaves 9.37 GB free.
- **The attention backend must be `fa3`, and the wrong one fails silently.** The checkpoint's own
  launcher defaults to `--attention-backend trtllm_mha`, which its README measured on two B200s. On
  H200 that backend serves, answers `/v1/models`, survives inference and returns `'4'` for
  "what is 2+2", but every longer generation degenerates into repetition. Nothing in the logs marks
  the difference, so only a multi-token generation reveals it.
- **DFlash speculative decoding does not work here**, which is why `SPEC_MODE` defaults to `none`
  although the 7.5 GiB draft is downloaded. With the draft enabled the engine reports that the fused
  KV path does not support non-neox RoPE and then dies with
  `AssertionError: Mask is required for speculative decoding`.
- Throughput above concurrency 8 is unmeasured, because the engine admits no more.
- Decode rate at a long input is not measured. Every figure here uses a 17-token input.
- The model fits because of its hybrid attention, not because 121 GiB leaves room. The engine reports
  a sliding window memory pool with `full_layer_tokens=33728` and `swa_layer_tokens=26944`, so the
  1024-token window layers cost far less KV than full attention layers would.
