# Ripple-65B-Preview NVFP4 on one H200 GPU

Status: Validated - SGLang 0.5.19 in a container, protocol: slope(128,1152) at concurrency 1, 8 and 16

| | |
| --- | --- |
| Served name | `ripple` |
| Checkpoint | `Ripple-65B-Preview-NVFP4`, Hugging Face `ricursiveAI/Ripple-65B-Preview-NVFP4` |
| On disk | 53.3 GiB across 6 shards, `RicursiveForCausalLM` |
| Served precision | NVFP4 MLP with BF16 attention, via ModelOpt; BF16 KV cache |
| Context | 262144. The checkpoint declares 2097152 |
| Hardware | 1 H200 GPU, 143771 MiB, TP1 |
| Engine | SGLang 0.5.19 in Singularity, `fa3` attention, no speculative decoding |
| Allocation for the measurement | 1 GPU of one node, 16 cores and 360 GB, `kempner_h200` |

**This is the recipe to reach for.** One GPU serves the full 262144 context at 89 percent of the two
GPU BF16 rate, because quantizing the MLP frees about 67 GiB for the KV cache.

| | [BF16, 1 GPU](../../Ripple-65B-Preview/h200-1/README.md) | [BF16, 2 GPUs](../../Ripple-65B-Preview/h200-2/README.md) | NVFP4, 1 GPU |
| --- | --- | --- | --- |
| Usable context | ~33,000 | 262,144 | **262,144** |
| KV pool | 33,728 tokens | 392,960 tokens | 278,720 tokens |
| Concurrent requests | 8 | 16 | 16 |
| Single stream | 25.1 tok/s | 40.5 tok/s | **36.2 tok/s** |
| Aggregate at its cap | 188.8 at c=8 | 587.7 at c=16 | 517.9 at c=16 |
| Weights | 120.7 GiB | 120.7 GiB | **53.71 GB** |

Only the MLP is quantized; the model's own docstring calls this an "NVFP4-MLP/BF16-attention"
checkpoint. So the KV cache costs the same per token as the BF16 build, and the context win comes
entirely from the weights that quantization frees, not from cheaper attention.

## 1. Create the API key

```
mkdir -p secrets
( umask 077; printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/Ripple-65B-Preview-NVFP4-h200-1.key )
```

- `secrets/vllm_api_key` is read when this file is absent.
- Requests without the key receive HTTP 401, and `serve.sh` refuses to start without one.
- SGLang logs the parsed `server_args`, the key among them, so `serve.sh` creates `$RIPPLE_LOG_DIR`
  mode 700 and the log mode 600, and aborts if it cannot.

## 2. Build the environment

A venv is not an option: sglang 0.5.19 publishes only `manylinux_2_34` wheels, needing glibc 2.34,
and these nodes are Rocky 8 with glibc 2.28.

```
bash recipes/Ripple-65B-Preview-NVFP4/h200-1/env/build.sh
```

**One image serves both builds.** The NVFP4 checkpoint imports the same SGLang layers as the BF16 one
and pins no other version, so `env/env.sh` resolves in order: an exported `SIF`, then this
checkpoint's own `container/`, then the **BF16 sibling's** copy, then the shared testbed. That avoids
a second 12.7 GiB image for the same engine, the way the Mistral recipes share one.

## 3. Launch

```
sbatch --account=<your-account> recipes/Ripple-65B-Preview-NVFP4/h200-1/serve.sbatch
```

On a node you already hold, naming the GPU:

```
bash recipes/Ripple-65B-Preview-NVFP4/h200-1/serve_ssh.sh <node> 0
```

It refuses to start if that GPU already has a compute process. Startup is about 6 minutes, of which
weight loading is 114 seconds; the engine reports `Using ModelOptModelLoader due to ModelOpt
quantization config` and then `Model is already quantized, loading directly`. The log is
`$RIPPLE_LOG_DIR/ripple-sglang.log`, default under `/tmp/$USER/ripple-nvfp4`, kept separate from the
BF16 recipes' directories.

## 4. Verify

```
KEY=$(cat secrets/Ripple-65B-Preview-NVFP4-h200-1.key 2>/dev/null || cat secrets/vllm_api_key)
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

Those `curl -H` lines put the key in curl's own argv, which other users on the node can see in `ps`.
That is acceptable for a one-off check on a node you hold alone; use `curl -K <file>` with a mode 600
file otherwise.

| Command | Expected |
| --- | --- |
| `/v1/models` with the key | `"id": "ripple"` |
| `/v1/models` without a key | `401` |
| raw completion | a coherent continuation, for example ` Paris. The capital of Germany is Berlin.` |
| chat completion | a correct Fibonacci function, `finish_reason: stop` |

**Run both generation probes, not just a short one.** A wrong attention backend on this hardware
serves happily, answers `/v1/models`, and returns `4` for "what is 2+2", while every longer generation
degenerates. Quantization gives the same reason to check: a damaged checkpoint answers short prompts
plausibly. See Known limits.

A long-input probe is worth running here, because it is the difference from the BF16 single GPU shape:

```
python3 -c 'import json; json.dump({"model":"ripple","prompt":"The quick brown fox jumps over the lazy dog. "*7000,"max_tokens":8,"temperature":0}, open("/tmp/long.json","w"))'
curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/completions -d @/tmp/long.json
```

That is about 70,000 tokens. It returns a completion here. The BF16 single GPU recipe rejects it with
`Input length (70001 tokens) exceeds the maximum allowed length (33274 tokens)`.

## 5. Connect a client

```
export NODE=<the node>
source recipes/Ripple-65B-Preview-NVFP4/h200-1/client.env
claude
```

`client.env` exports the base URL, the key, the served model name and
`CLAUDE_CODE_MAX_CONTEXT_TOKENS=262144`. Use `ANTHROPIC_AUTH_TOKEN`, not `ANTHROPIC_API_KEY`, which
sends a header the engine ignores and yields 401 on every request.

- OpenAI clients such as Codex: base URL `http://<node>:8000/v1`, same key, model `ripple`, and
  `wire_api = "responses"`, which recent versions require. See [docs/clients.md](../../../docs/clients.md).

## 6. Stop it

```
scancel <job id>
```

For an ssh launch, kill the `ripple_launch.py` process on the node. Kill by that name rather than by
GPU, since the other GPUs on the node may belong to someone else. All three Ripple recipes use that
same process name, so check the port or log directory if more than one is running.

## Tunable inputs

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/Ripple-65B-Preview-NVFP4` | Serve a different copy of the checkpoint |
| `RIPPLE_IMPL` | `inference/local` | Where the checkpoint's SGLang implementation lives. The BF16 build uses `inference/server` |
| `DRAFT` | `$MODELS_DIR/Ripple-65B-Preview-DFlash2` | DFlash2 draft. Only read, and only bound into the container, when `SPEC_MODE=dflash` |
| `SIF` | resolved as in step 2 | Container image |
| `API_PORT` | 8000 | Listening port |
| `MAX_MODEL_LEN` | 262144 | Context window, fixed at init |
| `MEM_FRACTION` | 0.90 | Static VRAM fraction for weights plus KV |
| `TP` | 1 | Tensor parallel size |
| `ATTN` | `fa3` | Attention backend. Do not change without reading Known limits |
| `PAGE_SIZE` | 64 | Tokens per KV page |
| `MAX_RUNNING` | 16 | Concurrent request cap, which also caps the sweep |
| `SPEC_MODE` | `none` | Set to `dflash` to try the shipped draft; it does not work here |
| `EXTRA_ARGS` | unset | Extra flags appended to the launch command |
| `KEY_NAME`, `KEY_FILE`, `VLLM_API_KEY` | this recipe's key | Which key `common/lib/api_key.sh` resolves |

## Benchmarking

Conditions:

| | |
| --- | --- |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| Input length | ISL 23 tokens. Rates at a long input are not measured; use `--prompt-tokens` |
| Output length | OSL 1152 tokens, output only, `ignore_eos` |
| Context | `MAX_MODEL_LEN=262144`, `MEM_FRACTION=0.90`, `ATTN=fa3`, no speculation |
| Allocation for the measurement | 1 GPU of one node, 16 cores and 360 GB, `kempner_h200` |
| Sequence cap | `max_running_requests` 16, which equals the top sweep level |
| Endpoint | idle, and the benchmark client ran on the serving node |

Results:

| Concurrency | Aggregate | Per stream | Spread over 3 runs |
| --- | --- | --- | --- |
| 1 | 36.2 tok/s | 36.2 tok/s | 36.2 to 36.2 |
| 8 | 277.7 tok/s | 34.7 tok/s | 277.7 to 277.7 |
| 16 | 517.9 tok/s | 32.4 tok/s | 517.7 to 517.9 |

| | |
| --- | --- |
| Engine facts | KV cache 278,720 tokens, 1.06 full-length requests at once; weights 53.71 GB, 13.62 GB free after sizing |
| Label | capped. The sweep stops at 16 because the engine admits no more, not because throughput turned over |

Per-stream throughput barely moves, from 36.2 to 32.4 tok/s between one caller and sixteen, which is
flatter than either BF16 shape. Against BF16 on one GPU this is 44 percent faster with eight times the
usable context; against BF16 on two GPUs it is 89 percent of the single stream rate and 88 percent of
the aggregate, on half the hardware.

## Known limits

- **NVFP4 runs on Hopper here, which is worth stating because it is easy to assume otherwise.** The
  format is Blackwell-native and the other NVFP4 recipes in this repo serve on RTX PRO 6000. SGLang's
  `modelopt_quant.py` sets `get_min_capability()` to 89, and an H200 is capability 90, so it loads and
  runs. What it does not get is the `use_sm120_gemv` path, which needs capability 12.x, so expect the
  generic kernels rather than the fastest ones.
- **The NVFP4 build ships its implementation under `inference/local`, not `inference/server`.** Its
  `model.py` is the same architecture with a `quant_config` threaded through every layer, and its
  `ripple_format.py` registers the same `ripple` parsers, so `common/tools/ripple_launch.py` is shared
  unchanged. Only `RIPPLE_IMPL` differs. The directory also carries a `spark.py` and a notebook rather
  than a `serve.py`, since the vendor targets DGX Spark and laptops for this build.
- **`reasoning_content` comes back empty.** `--reasoning-parser ripple` is passed and registered, and
  a deliberately reasoning-heavy prompt still produced no reasoning segment, with no `<think>` markers
  left in `content`. The parser is working and the model simply does not emit a reasoning block for
  these prompts. Whether the BF16 build differs is **untested**.
- **The attention backend must be `fa3`**, as with the BF16 recipes. The checkpoint's own launcher
  defaults to `trtllm_mha`, which on H200 serves and then degenerates on any longer generation, with
  nothing in the logs to mark it.
- DFlash speculative decoding is off. The draft is BF16 and the published DFlash path fails on H200
  for the BF16 build; it is untested against this checkpoint.
- Throughput above concurrency 16 is unmeasured, because the engine admits no more.
- Decode rate at a long input is not measured. Every figure here uses a 23-token input.
- There is **no FP8 build**. `ricursiveAI` publishes only the base, the BF16 instruct, the DFlash2
  draft and this NVFP4 checkpoint.
- Context is served at 262144 rather than the checkpoint's 2097152, which is untested here.
