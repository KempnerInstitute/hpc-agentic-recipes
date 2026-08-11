# gemma-4-31B-it on one H200 GPU

Status: Validated - vLLM 0.25.1+cu129, protocol: slope(128,1152) at concurrency 1, 64, 128, 256, 512, 768 and 1024

| | |
| --- | --- |
| Served name | `gemma-4-31b` |
| Checkpoint | `gemma-4-31B-it`, Hugging Face `google/gemma-4-31B-it` |
| On disk | 62.5 GB, BF16, `Gemma4ForConditionalGeneration` |
| Served precision | FP8, quantized on load, 31.7 GiB of weights |
| Context | 262144, the checkpoint maximum |
| Hardware | 1 H200, 143771 MiB, TP1 |
| Engine | vLLM 0.25.1+cu129 |

## 1. Create the API key

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/gemma-4-31B-it-h200-1.key
chmod 600 secrets/gemma-4-31B-it-h200-1.key
```

- `secrets/vllm_api_key` is read when this file is absent.
- To rotate, replace the file and relaunch; the engine reads it once at launch.

## 2. Build the environment

Run on a compute node, not a login node.

```
bash recipes/gemma-4-31B-it/h200-1/env/build.sh
```

- About 13 GB, under `ENV_ROOT`, default scratch. Set `ENV_ROOT` in `common/site.conf` to move it.
- Installs the cu129 wheel. These nodes run driver 575, CUDA 12.9, and vLLM's default wheel is CUDA 13.

## 3. Launch

Slurm, from the repo root:

```
sbatch --account=<your-account> recipes/gemma-4-31B-it/h200-1/serve.sbatch
squeue --me                                  # NODELIST gives the host
tail -f gemma31-h200-1-<jobid>.log
```

Direct, on a node you already hold:

```
# whole node to yourself
bash recipes/gemma-4-31B-it/h200-1/serve_ssh.sh <node>

# shared node, the usual case: read the device Slurm gave you, then pin it
scontrol show job -d <jobid> | grep -oE 'IDX:[0-9]+' | cut -d: -f2
GPU=<n> bash recipes/gemma-4-31B-it/h200-1/serve_ssh.sh <node>
```

- Startup is about 3 minutes.
- On the direct path the server log is node-local at `/tmp/$USER/vllm/`. Under Slurm it lands in the submit
  directory.

## 4. Verify

```
KEY=$(cat secrets/gemma-4-31B-it-h200-1.key 2>/dev/null || cat secrets/vllm_api_key)
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
source recipes/gemma-4-31B-it/h200-1/client.env
claude
```

- No output cap is needed at 262144.
- `client.env` sets `CLAUDE_CODE_MAX_CONTEXT_TOKENS=262144`. Claude Code does not recognize this served name
  and assumes a 200k window, which client 2.1.223 enforces.
- Use `ANTHROPIC_AUTH_TOKEN`. `ANTHROPIC_API_KEY` sends `x-api-key` and returns 401.
- `client.env` also sets `ANTHROPIC_SMALL_FAST_MODEL`; without it the client reaches for a hosted Haiku.
- OpenAI clients such as Codex: base URL `http://<node>:8000/v1`, same key, model `gemma-4-31b`. See
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
| `QUANT` | `fp8` | Set but empty for BF16. BF16 starts at 262144 on this card, with 629,860 tokens of KV and 2.40 full-length requests |
| `KV_FP8` | unset | `--kv-cache-dtype fp8`; halves KV bytes |
| `ENFORCE_EAGER` | unset | Skip torch.compile and CUDA graph capture, to debug a startup failure |
| `SPEC_DRAFT` | unset | Path to the drafter checkpoint, `$MODELS_DIR/gemma-4-31B-it-assistant`. Works on this build and measured 2.7 times faster; see Benchmarking |
| `SPEC_TOKENS` | 3 | Speculative tokens per step, read only when `SPEC_DRAFT` is set |
| `EXTRA_ARGS` | unset | Extra flags appended to the `vllm serve` command line |
| `TOOL_PARSER` | `gemma4` | Tool call parser |
| `REASONING_PARSER` | `gemma4` | Reasoning parser |
| `GPU` | unset | SSH path only: pin one device of a shared node |
| `LOG_DIR` | `/tmp/$USER/vllm` | Where the SSH path writes the server log |
| `VENV_DIR` | under `ENV_ROOT` | Use an environment built elsewhere |
| `VLLM_VERSION` | 0.25.1 | Wheel version `env/build.sh` installs |
| `TRANSFORMERS_VERSION` | 5.14.1 | The transformers version the rates were measured with |
| `VLLM_USE_DEEP_GEMM` | 0, set in `env/env.sh` | Must stay 0 on H200; see Known limits |
| `KEY_NAME`, `KEY_FILE`, `VLLM_API_KEY` | this recipe's key | Which key `common/lib/api_key.sh` resolves; an exported `VLLM_API_KEY` wins |
| `NODE`, `GEMMA31_H200_NODE` | unset | The node to launch on or connect to; set either, or pass the node as an argument |
| `MODELS_DIR`, `ENV_ROOT`, `VLLM_CACHE_ROOT`, `UV_CACHE_DIR` | `common/defaults.sh` | Override there or in `common/site.conf` |

## Benchmarking

Conditions:

| | |
| --- | --- |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| Input length | ISL 19 tokens. Rates at a long input are not measured; use `--prompt-tokens` |
| Output length | OSL 1152 tokens, output only, `ignore_eos` |
| Context | `MAX_MODEL_LEN=262144` |
| Allocation for the measurement | 1 GPU, 16 cores, 360 GB |
| Sequence cap | `max_num_seqs` 1024, which equals the top sweep level |
| Preemption | 124,318 across the sweep. Peak demand first exceeds the 894,418-token pool at concurrency 1024, which needs 1,199,104 slots, so the cache binds there rather than the sequence cap |
| Endpoint | idle, and the benchmark client ran on a separate CPU-only node |
| Power | 700 W enforced, the card default, so not capped |

Results:

| Concurrency | Aggregate | Per stream | Spread over 3 runs |
| --- | --- | --- | --- |
| 1 | 85.0 tok/s | 85.0 tok/s | 85.0 to 85.0 |
| 64 | 2649.6 tok/s | 41.4 tok/s | 2649.2 to 2654.9 |
| 128 | 2922.1 tok/s | 22.8 tok/s | 2921.5 to 2935.7 |
| 256 | 3054.2 tok/s | 11.9 tok/s | 3045.6 to 3055.5 |
| 512 | 3112.8 tok/s | 6.1 tok/s | 3111.0 to 3124.0 |
| 768 | 3154.3 tok/s | 4.1 tok/s | 3149.3 to 3154.4 |
| 1024 | 3153.3 tok/s | 3.1 tok/s | 3148.3 to 3156.9 |

| | |
| --- | --- |
| Label | saturated. It varies 1.32 percent across 512 to 1024, the window the rule uses, under its 4 percent threshold. The highest median is at 768, though the spreads at 768 and 1024 overlap |
| Quote for one caller | 85.0 tok/s |
| Quote for a shared endpoint | 3154.3 tok/s at concurrency 768 |
| KV cache | 894,418 tokens from 92.27 GiB, 3.41 full-length requests at once |
| Speculative decoding | `SPEC_DRAFT=$MODELS_DIR/gemma-4-31B-it-assistant` measured 233.1 tok/s single stream against 85.0, a 2.7 times gain. It costs 2.5 percent of the KV pool, which drops to 872,503 tokens. Aggregate throughput with it is not measured |
| Long prompt, cold | 30,047 tokens in 3.2 s, 120,048 in 23.4 s, 240,048 in 76.8 s |

Reproduce:

```
KEY_NAME=gemma-4-31B-it-h200-1 bash common/tools/bench.sh --host <node> --model gemma-4-31b
KEY_NAME=gemma-4-31B-it-h200-1 bash common/tools/bench.sh --host <node> --model gemma-4-31b \
  --sweep 1,64,128,256,512,768,1024
```

- `KEY_NAME` is required once `secrets/` holds both this recipe's key and `vllm_api_key`. The server uses
  the recipe key, `bench.sh` alone resolves the shared one, and every request returns 401.

## Known limits

- Keep `TRANSFORMERS_VERSION` at 5.14.1. From 5.15.0 `head_dim` is a per-layer attribute for this
  checkpoint and vLLM reads it globally, so the engine exits with
  `AmbiguousGlobalPerLayerAttributeError` before it loads any weights.
- Leave `VLLM_USE_DEEP_GEMM` at 0, which `env/env.sh` sets. The DeepGEMM MoE path takes an illegal memory
  access on H200, reproduced independently for two other models, so it is load bearing on this hardware.
- Anthropic's hosted tools return HTTP 400. Use [docs/web-search.md](../../../docs/web-search.md).
