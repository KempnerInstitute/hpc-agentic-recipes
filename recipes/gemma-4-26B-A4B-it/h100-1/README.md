# gemma-4-26B-A4B-it on one H100 GPU

Status: Validated - vLLM 0.25.1+cu129, protocol: slope(128,1152) at concurrency 1, 64, 256, 640 and 1024

| | |
| --- | --- |
| Served name | `gemma-4-26b` |
| Checkpoint | `gemma-4-26B-A4B-it`, Hugging Face `google/gemma-4-26B-A4B-it` |
| On disk | 51.6 GB, bf16, `Gemma4ForConditionalGeneration` |
| Context | 262144, the checkpoint maximum |
| Hardware | 1 H100, 80 GB, TP1, no NCCL |
| Engine | vLLM 0.25.1+cu129 |

## 1. Create the API key

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/gemma-4-26B-A4B-it-h100-1.key
chmod 600 secrets/gemma-4-26B-A4B-it-h100-1.key
```

- `secrets/vllm_api_key` is read when this file is absent.
- To rotate, replace the file and relaunch; the engine reads it once at launch.

## 2. Build the environment

Run on a compute node, not a login node.

```
bash recipes/gemma-4-26B-A4B-it/h100-1/env/build.sh
```

- About 13 GB, under `ENV_ROOT`, default scratch. Set `ENV_ROOT` in `common/site.conf` to move it.
- Installs the cu129 wheel. These nodes run driver 575, CUDA 12.9, and vLLM's default wheel is CUDA 13.

## 3. Launch

Slurm, from the repo root:

```
sbatch --account=<your-account> recipes/gemma-4-26B-A4B-it/h100-1/serve.sbatch
squeue --me                                  # NODELIST gives the host
tail -f gemma26-h100-1-<jobid>.log
```

Direct, on a node you already hold:

```
# whole node to yourself
bash recipes/gemma-4-26B-A4B-it/h100-1/serve_ssh.sh <node>

# shared node, the usual case: read the device Slurm gave you, then pin it
scontrol show job -d <jobid> | grep -oE 'IDX:[0-9]+' | cut -d: -f2
GPU=<n> bash recipes/gemma-4-26B-A4B-it/h100-1/serve_ssh.sh <node>
```

- The server log is node-local at `/tmp/$USER/vllm/`.

## 4. Verify

```
KEY=$(cat secrets/gemma-4-26B-A4B-it-h100-1.key 2>/dev/null || cat secrets/vllm_api_key)
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
source recipes/gemma-4-26B-A4B-it/h100-1/client.env
claude
```

- No output cap is needed at 262144.
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
| `QUANT` | unset, bf16 | `fp8` quantizes weights on load; measured no change in rate |
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
| `NODE`, `GEMMA26_H100_NODE` | unset | The node to launch on or connect to; set either, or pass the node as an argument |
| `ACCOUNT` | unset | Slurm account, or pass `--account` at submit time |
| `MODELS_DIR`, `ENV_ROOT`, `VLLM_CACHE_ROOT` | `common/defaults.sh` | Override there or in `common/site.conf` |

## Benchmarking

Conditions:

| | |
| --- | --- |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| Input length | ISL 19 tokens. Rates at a long input are not measured; use `--prompt-tokens` |
| Output length | OSL 1152 tokens, output only, `ignore_eos` |
| Context | `MAX_MODEL_LEN=262144` |
| Allocation for the measurement | 1 GPU, 24 cores, 180 GB |
| Sequence cap | `max_num_seqs` 1024, which equals the top sweep level |
| Endpoint | idle, no other traffic |
| Power cap | these H100s are capped to 550 W of a 700 W default, unlike the H200 and RTX cards. At concurrency 640 the device sat at that limit for 132 of 150 samples with clocks between 1545 and 1980 MHz. At concurrency 1 it peaked at 345 W and never reached the limit |

Results:

| Concurrency | Aggregate | Per stream | Spread over 3 runs |
| --- | --- | --- | --- |
| 1 | 204.5 tok/s | 204.5 tok/s | 204.5 to 204.5 |
| 64 | 5171.2 tok/s | 80.8 tok/s | 5169.0 to 5174.6 |
| 256 | 6304.7 tok/s | 24.6 tok/s | 6293.6 to 6314.3 |
| 640 | 7164.7 tok/s | 11.2 tok/s | 7154.9 to 7188.8 |
| 1024 | 7146.9 tok/s | 7.0 tok/s | 7120.7 to 7148.3 |

| | |
| --- | --- |
| Label | peak. Throughput turns over at 640; 1024 is 0.25 percent lower |
| Cross-run spread | concurrency 640 measured 6998.8 to 7164.7 tok/s across four runs, 2.4 percent, with no cause established |
| Concurrency 1 across restarts | 204.5 to 213.8 tok/s. Both values are reproducible in triplicate within a launch and neither is power limited; the cause is not identified |
| Quote for one caller | 204.5 tok/s |
| Quote for a shared endpoint | 7164.7 tok/s at concurrency 640 |
| KV cache | 661,098 tokens, 2.52 full-length requests at once |
| Long prompt | 240,043 tokens in 20 s cold; 0.6 s when the prefix is already cached |

Reproduce:

```
KEY_NAME=gemma-4-26B-A4B-it-h100-1 bash common/tools/bench.sh --host <node> --model gemma-4-26b
KEY_NAME=gemma-4-26B-A4B-it-h100-1 bash common/tools/bench.sh --host <node> --model gemma-4-26b \
  --sweep 1,64,256,640,1024
```

- `KEY_NAME` is required once `secrets/` holds both this recipe's key and `vllm_api_key`. The server uses
  the recipe key, `bench.sh` alone resolves the shared one, and every request returns 401.

## Known limits

- `SPEC_DRAFT` must stay unset. `gemma4_mtp` fails at the first request on vLLM 0.25.1 and 0.26.0.
- Anthropic's hosted tools return HTTP 400. Use [docs/web-search.md](../../../docs/web-search.md).
