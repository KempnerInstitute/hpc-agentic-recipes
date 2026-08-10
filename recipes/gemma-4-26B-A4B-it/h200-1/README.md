# gemma-4-26B-A4B-it on one H200 GPU

Status: Validated - vLLM 0.25.1+cu129, protocol: slope(128,1152) at concurrency 1, 64, 256, 640 and 1024

| | |
| --- | --- |
| Served name | `gemma-4-26b` |
| Checkpoint | `gemma-4-26B-A4B-it`, Hugging Face `google/gemma-4-26B-A4B-it` |
| On disk | 51.6 GB, BF16, `Gemma4ForConditionalGeneration` |
| Context | 262144, the checkpoint maximum |
| Hardware | 1 H200, 140 GB, TP1, no NCCL |
| Engine | vLLM 0.25.1+cu129 |

## 1. Create the API key

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/gemma-4-26B-A4B-it-h200-1.key
chmod 600 secrets/gemma-4-26B-A4B-it-h200-1.key
```

- `secrets/vllm_api_key` is read when this file is absent.
- To rotate, replace the file and relaunch; the engine reads it once at launch.

## 2. Build the environment

Run on a compute node, not a login node.

```
bash recipes/gemma-4-26B-A4B-it/h200-1/env/build.sh
```

- About 13 GB, under `ENV_ROOT`, default scratch. Set `ENV_ROOT` in `common/site.conf` to move it.
- Installs the cu129 wheel. These nodes run driver 575, CUDA 12.9, and vLLM's default wheel is CUDA 13.

## 3. Launch

Slurm, from the repo root:

```
sbatch --account=<your-account> recipes/gemma-4-26B-A4B-it/h200-1/serve.sbatch
squeue --me                                  # NODELIST gives the host
tail -f gemma26-h200-1-<jobid>.log
```

Direct, on a node you already hold:

```
# whole node to yourself
bash recipes/gemma-4-26B-A4B-it/h200-1/serve_ssh.sh <node>

# shared node, the usual case: read the device Slurm gave you, then pin it
scontrol show job -d <jobid> | grep -oE 'IDX:[0-9]+' | cut -d: -f2
GPU=<n> bash recipes/gemma-4-26B-A4B-it/h200-1/serve_ssh.sh <node>
```

- The server log is node-local at `/tmp/$USER/vllm/`.

## 4. Verify

```
KEY=$(cat secrets/gemma-4-26B-A4B-it-h200-1.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the node serving it>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"gemma-4-26b","messages":[{"role":"user","content":"What is 2+2? Answer briefly."}],"max_tokens":400}'
```

| Command | Expected |
| --- | --- |
| `/v1/models` with the key | `"id": "gemma-4-26b"`, `"max_model_len": 262144` |
| `/v1/models` without a key | `401` |
| chat completion | `content` holds the answer, `finish_reason: stop` |

- Use at least 400 output tokens. Reasoning arrives in a separate `reasoning` field, so a smaller budget
  can end with `finish_reason: length` and empty `content`.

## 5. Connect a client

```
export NODE=<the node serving it>
source recipes/gemma-4-26B-A4B-it/h200-1/client.env
claude
```

- No output cap is needed at 262144.
- `client.env` sets `CLAUDE_CODE_MAX_CONTEXT_TOKENS=262144`. Claude Code does not recognize this served name
  and assumes a 200k window, which client 2.1.223 enforces.
- Use `ANTHROPIC_AUTH_TOKEN`. `ANTHROPIC_API_KEY` sends `x-api-key` and returns 401.
- `client.env` also sets `ANTHROPIC_SMALL_FAST_MODEL`; without it the client reaches for a hosted Haiku.
- OpenAI clients: base URL `http://<node>:8000/v1`, same key, model `gemma-4-26b`. See
  [docs/clients.md](../../../docs/clients.md).

## 6. Stop it

```
scancel <jobid>                        # Slurm path
bash common/tools/stop.sh <node>       # direct path
```

## Tunable inputs

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/gemma-4-26B-A4B-it` | Serve a different copy of the checkpoint |
| `API_PORT` | 8000 | Listening port, and part of the SSH log file name |
| `MAX_MODEL_LEN` | 262144 | Context window, the checkpoint maximum |
| `GPU_UTIL` | 0.90 | Fraction of VRAM for weights plus KV cache |
| `TP` | 1 | Tensor parallel size |
| `QUANT` | unset, BF16 | `fp8` quantizes weights on load; measured no change in rate |
| `KV_FP8` | unset | `--kv-cache-dtype fp8`; halves KV bytes, measured no change in rate |
| `ENFORCE_EAGER` | unset | Skip torch.compile and CUDA graph capture, to debug a startup failure |
| `SPEC_DRAFT` | unset | Must stay unset on vLLM 0.25.1 and 0.26.0; see Known limits |
| `SPEC_TOKENS` | 3 | Speculative tokens per step, read only when `SPEC_DRAFT` is set |
| `EXTRA_ARGS` | unset | Extra flags appended to the `vllm serve` command line |
| `TOOL_PARSER` | `gemma4` | Tool call parser |
| `REASONING_PARSER` | `gemma4` | Reasoning parser |
| `GPU` | unset | SSH path only: pin one device of a shared node |
| `LOG_DIR` | `/tmp/$USER/vllm` | Where the SSH path writes the server log |
| `VENV_DIR` | under `ENV_ROOT` | Use an environment built elsewhere |
| `VLLM_VERSION` | 0.25.1 | Wheel version `env/build.sh` installs |
| `NODE`, `GEMMA26_H200_NODE` | unset | The node to launch on or connect to; set either, or pass the node as an argument |
| `MODELS_DIR`, `ENV_ROOT`, `VLLM_CACHE_ROOT` | `common/defaults.sh` | Override there or in `common/site.conf` |
| `VLLM_USE_DEEP_GEMM` | 0, set in `env/env.sh` | Must stay 0 on H200; see Known limits |

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
| Endpoint | idle, no other traffic |

Results:

| Concurrency | Aggregate | Per stream | Spread over 3 runs |
| --- | --- | --- | --- |
| 1 | 250.5 tok/s | 250.5 tok/s | 250.4 to 250.6 |
| 64 | 6259.8 tok/s | 97.8 tok/s | 6254.4 to 6311.3 |
| 256 | 10547.4 tok/s | 41.2 tok/s | 10517.1 to 10565.0 |
| 640 | 10506.2 tok/s | 16.4 tok/s | 10475.3 to 10524.2 |
| 1024 | 10904.7 tok/s | 10.6 tok/s | 10894.3 to 10914.3 |

| | |
| --- | --- |
| Label | saturated. Flat within 3.8 percent from 256 to 1024, and 1024 equals `max_num_seqs`, so the sweep cannot reach past it |
| Quote for one caller | 250.5 tok/s |
| Quote for a shared endpoint | 10904.7 tok/s at concurrency 1024 |
| KV cache | 2,776,615 tokens, 10.59 full-length requests at once |
| Long prompt | 240,050 tokens in 18.5 s cold |

Reproduce:

```
KEY_NAME=gemma-4-26B-A4B-it-h200-1 bash common/tools/bench.sh --host <node> --model gemma-4-26b
KEY_NAME=gemma-4-26B-A4B-it-h200-1 bash common/tools/bench.sh --host <node> --model gemma-4-26b \
  --sweep 1,64,256,640,1024
```

- `KEY_NAME` is required once `secrets/` holds both this recipe's key and `vllm_api_key`. The server uses
  the recipe key, `bench.sh` alone resolves the shared one, and every request returns 401.

## Known limits

- `SPEC_DRAFT` must stay unset. `gemma4_mtp` fails at engine startup on vLLM 0.25.1 and 0.26.0, so the
  server never comes up. Retested on 0.25.1 with the same shape mismatch.
- Leave `VLLM_USE_DEEP_GEMM` at 0, which `env/env.sh` sets. The DeepGEMM MoE path takes an illegal memory
  access on H200, reproduced independently for two other models, so it is load bearing on this hardware.
- Anthropic's hosted tools return HTTP 400. Use [docs/web-search.md](../../../docs/web-search.md).
