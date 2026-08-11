# gemma-4-31B-it on one H100 GPU

Status: Validated - vLLM 0.25.1, protocol: slope(128,1152) at concurrency 1, 64, 128, 256, 512 and 1024

| | |
| --- | --- |
| Served name | `gemma-4-31b` |
| Checkpoint | `gemma-4-31B-it`, Hugging Face `google/gemma-4-31B-it` |
| On disk | 62.5 GB, BF16, `Gemma4ForConditionalGeneration` |
| Served precision | FP8, quantized on load, 31.73 GiB of weights |
| Context | 262144, the checkpoint maximum |
| Hardware | 1 H100, 81559 MiB, TP1 |
| Engine | vLLM 0.25.1 with torch cu129 |

## 1. Create the API key

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/gemma-4-31B-it-h100-1.key
chmod 600 secrets/gemma-4-31B-it-h100-1.key
```

- `secrets/vllm_api_key` is read when this file is absent.
- To rotate, replace the file and relaunch; the engine reads it once at launch.

## 2. Build the environment

Run on a compute node, not a login node.

```
bash recipes/gemma-4-31B-it/h100-1/env/build.sh
```

- About 13 GB, under `ENV_ROOT`, default scratch. Set `ENV_ROOT` in `common/site.conf` to move it.
- Installs the cu129 wheel. These nodes run driver 575, CUDA 12.9, and vLLM's default wheel is CUDA 13.

## 3. Launch

Slurm, from the repo root:

```
sbatch --account=<your-account> recipes/gemma-4-31B-it/h100-1/serve.sbatch
squeue --me                                  # NODELIST gives the host
tail -f gemma31-h100-1-<jobid>.log
```

Direct, on a node you already hold:

```
# whole node to yourself
bash recipes/gemma-4-31B-it/h100-1/serve_ssh.sh <node>

# shared node, the usual case: read the device Slurm gave you, then pin it
scontrol show job -d <jobid> | grep -oE 'IDX:[0-9]+' | cut -d: -f2
GPU=<n> bash recipes/gemma-4-31B-it/h100-1/serve_ssh.sh <node>
```

- Startup is about 3 to 4 minutes.
- On the direct path the server log is node-local at `/tmp/$USER/vllm/`. Under Slurm it lands in the submit directory.

## 4. Verify

```
KEY=$(cat secrets/gemma-4-31B-it-h100-1.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the node serving it>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"gemma-4-31b","messages":[{"role":"user","content":"What is 2+2? Answer briefly."}],"max_tokens":400}'
```

| Command | Expected |
| --- | --- |
| `/v1/models` with the key | `"id": "gemma-4-31b"`, `"max_model_len": 262144` |
| `/v1/models` without a key | `401` |
| chat completion | `content` holds the answer, `finish_reason: stop` |

- Use at least 400 output tokens. Reasoning arrives in a separate `reasoning` field, so a smaller budget
  can end with `finish_reason: length` and empty `content`.

## 5. Connect a client

```
export NODE=<the node serving it>
source recipes/gemma-4-31B-it/h100-1/client.env
claude
```

- No output cap is needed at 262144.
- `client.env` sets `CLAUDE_CODE_MAX_CONTEXT_TOKENS=262144`. Claude Code does not recognize this served name
  and assumes a 200k window, which client 2.1.223 enforces.
- Use `ANTHROPIC_AUTH_TOKEN`. `ANTHROPIC_API_KEY` sends `x-api-key` and returns 401.
- `client.env` also sets `ANTHROPIC_SMALL_FAST_MODEL`; without it the client reaches for a hosted Haiku.
- OpenAI clients: base URL `http://<node>:8000/v1`, same key, model `gemma-4-31b`. See
  [docs/clients.md](../../../docs/clients.md).

## 6. Stop it

```
scancel <jobid>                        # Slurm path
bash common/tools/stop.sh <node>       # direct path
```

## Tunable inputs

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/gemma-4-31B-it` | Serve a different copy of the checkpoint |
| `API_PORT` | 8000 | Listening port, and part of the SSH log file name |
| `MAX_MODEL_LEN` | 262144 | Context window, the checkpoint maximum |
| `GPU_UTIL` | 0.90 | Fraction of VRAM for weights plus KV cache |
| `TP` | 1 | Tensor parallel size |
| `QUANT` | `fp8` | Set but empty for BF16. BF16 weights are 58.25 GiB and will not start at 262144, since one full-length request needs 27.04 GiB of KV; cap `MAX_MODEL_LEN` near 52000 for BF16 |
| `KV_FP8` | unset | `--kv-cache-dtype fp8`; halves KV bytes |
| `ENFORCE_EAGER` | unset | Skip torch.compile and CUDA graph capture, to debug a startup failure |
| `SPEC_DRAFT` | unset | Path to the drafter checkpoint, `$MODELS_DIR/gemma-4-31B-it-assistant`. Works on this engine and measured 2.7 times faster; see Benchmarking |
| `SPEC_TOKENS` | 3 | Speculative tokens per step, read only when `SPEC_DRAFT` is set |
| `EXTRA_ARGS` | unset | Extra flags appended to the `vllm serve` command line |
| `TOOL_PARSER` | `gemma4` | Tool call parser |
| `REASONING_PARSER` | `gemma4` | Reasoning parser |
| `GPU` | unset | SSH path only: pin one device of a shared node |
| `LOG_DIR` | `/tmp/$USER/vllm` | Where the SSH path writes the server log |
| `VENV_DIR` | under `ENV_ROOT` | Use an environment built elsewhere |
| `VLLM_VERSION` | 0.25.1 | Wheel version `env/build.sh` installs |
| `TRANSFORMERS_VERSION` | 5.14.1 | The transformers version the rates were measured with |
| `UV_CACHE_DIR` | under `ENV_ROOT` | Keep it on the same filesystem as the venv, or uv copies about 13 GB instead of hardlinking |
| `KEY_NAME`, `KEY_FILE`, `VLLM_API_KEY` | this recipe's key | Which key `common/lib/api_key.sh` resolves; an exported `VLLM_API_KEY` wins |
| `NODE`, `GEMMA31_H100_NODE` | unset | The node to launch on or connect to; set either, or pass the node as an argument |
| `MODELS_DIR`, `ENV_ROOT`, `VLLM_CACHE_ROOT` | `common/defaults.sh` | Override there or in `common/site.conf` |

## Benchmarking

Conditions:

| | |
| --- | --- |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| Input length | ISL 19 tokens. Rates at a long input are not measured; use `--prompt-tokens` |
| Output length | OSL 1152 tokens, output only, `ignore_eos` |
| Context | `MAX_MODEL_LEN=262144` |
| Allocation for the measurement | 1 GPU, 24 cores, 360 GB |
| Sequence cap | `max_num_seqs` 1024, which equals the top sweep level |
| Endpoint | idle, no other traffic |
| Power cap | these H100s are capped to 550 W of a 700 W default. Under load the device sat at that limit for 1117 of 1200 samples, with clocks ranging 765 to 1980 MHz and a median of 1590. At concurrency 1 it peaked at 468 W, never reached the limit, and held 1980 MHz |
| Preemption | 110,565 preemptions accumulated across the sweep, so the KV cache saturates well before the sequence cap |

Results:

| Concurrency | Aggregate | Per stream | Spread over 3 runs |
| --- | --- | --- | --- |
| 1 | 68.7 tok/s | 68.7 tok/s | 68.7 to 68.7 |
| 64 | 1811.1 tok/s | 28.3 tok/s | 1798.2 to 1813.2 |
| 128 | 2159.4 tok/s | 16.9 tok/s | 2157.2 to 2163.2 |
| 256 | 2401.5 tok/s | 9.4 tok/s | 2399.7 to 2411.5 |
| 512 | 2471.4 tok/s | 4.8 tok/s | 2461.8 to 2480.3 |
| 1024 | 2425.1 tok/s | 2.4 tok/s | 2422.5 to 2438.9 |

| | |
| --- | --- |
| Label | saturated. It varies 1.9 percent from 512 to 1024, under the 4 percent the rule uses, and the highest value is at 512. An independent run measured 768 at 2465.6, between 512 and 1024, so the decline is monotonic rather than a dip |
| Quote for one caller | 68.7 tok/s, from nine runs with no spread. The c=1 stage inside a sweep reads 67.4, 1.9 percent lower |
| Quote for a shared endpoint | 2471.4 tok/s at concurrency 512 |
| KV cache | 365,231 tokens from 37.68 GiB, 1.39 full-length requests at once |
| Speculative decoding | `SPEC_DRAFT=$MODELS_DIR/gemma-4-31B-it-assistant` measured 187.3 tok/s single stream against 68.7, a 2.7 times gain. It costs 6.0 percent of the KV pool, which drops to 343,435 tokens |
| Cost of the larger context | 0.2 percent. Concurrency 512 measures 2476.1 tok/s at 32768 against 2471.4 at 262144 |
| Long prompt, cold | 30,048 tokens in 3.7 s, 120,049 in 26.6 s, 240,049 in 86.6 s |

Reproduce:

```
KEY_NAME=gemma-4-31B-it-h100-1 bash common/tools/bench.sh --host <node> --model gemma-4-31b
KEY_NAME=gemma-4-31B-it-h100-1 bash common/tools/bench.sh --host <node> --model gemma-4-31b \
  --sweep 1,64,128,256,512,1024
```

- `KEY_NAME` is required once `secrets/` holds both this recipe's key and `vllm_api_key`. The server uses
  the recipe key, `bench.sh` alone resolves the shared one, and every request returns 401.

## Known limits

- Keep `TRANSFORMERS_VERSION` at 5.14.1. From 5.15.0 `head_dim` is a per-layer attribute for this
  checkpoint and vLLM reads it globally, so the engine exits with
  `AmbiguousGlobalPerLayerAttributeError` before it loads any weights.
- The 26B drafter cannot be used here. It fails at engine startup with a shape mismatch, so only the 31B
  drafter named in the tunables table works.
- Anthropic's hosted tools return HTTP 400. Use [docs/web-search.md](../../../docs/web-search.md).
