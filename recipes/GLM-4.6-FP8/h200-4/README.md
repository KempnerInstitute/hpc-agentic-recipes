# GLM-4.6-FP8 on one H200 node

Status: Validated - vLLM 0.25.1+cu129, protocol: slope(128,1152) at concurrency 1, 8, 32, 64, 128, 256, 512, 640, 768, 896 and 1024

| | |
| --- | --- |
| Served name | `glm-4.6` |
| Checkpoint | `GLM-4.6-FP8`, Hugging Face `zai-org/GLM-4.6-FP8` |
| On disk | 336.48 GiB, FP8, `Glm4MoeForCausalLM` |
| Served precision | FP8 from the checkpoint, 84.22 GiB of weights per GPU |
| Context | 202752, the checkpoint maximum |
| Hardware | 1 H200 node, 4 GPUs, 143771 MiB each, TP4 over NVLink |
| Engine | vLLM 0.25.1+cu129, eager, with MTP speculative decoding |

## 1. Create the API key

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/GLM-4.6-FP8-h200-4.key
chmod 600 secrets/GLM-4.6-FP8-h200-4.key
```

- `secrets/vllm_api_key` is read when this file is absent.
- To rotate, replace the file and relaunch; the engine reads it once at launch.

## 2. Build the environment

Run on a compute node, not a login node.

```
bash recipes/GLM-4.6-FP8/h200-4/env/build.sh
```

- About 13 GB and 215 packages, under `ENV_ROOT`, default scratch. Set `ENV_ROOT` in `common/site.conf` to
  move it.
- Installs the cu129 wheel. These nodes run driver 575, CUDA 12.9, and vLLM's default wheel is CUDA 13.
- `env/requirements.lock` records the tested resolution and `env/WHEELS` the non-PyPI wheel URL.

## 3. Launch

Slurm, from the repo root:

```
sbatch --account=<your-account> recipes/GLM-4.6-FP8/h200-4/serve.sbatch
squeue --me                                  # NODELIST gives the host
tail -f glm46-<jobid>.log
```

Direct, on a node you already hold. This recipe uses all four GPUs, so there is no device to pin:

```
bash recipes/GLM-4.6-FP8/h200-4/serve_ssh.sh <node>
```

- Startup is 8 minutes 21 seconds from the default checkpoint path on Lustre, and 5 minutes 56 seconds on a
  relaunch with a warm page cache. Weight load alone is 3 minutes 31 seconds.
- On the direct path the server log is node-local at `/tmp/$USER/vllm/`. Under Slurm it lands in the submit
  directory.

## 4. Verify

```
KEY=$(cat secrets/GLM-4.6-FP8-h200-4.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the node serving it>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"glm-4.6","messages":[{"role":"user","content":"What is 2+2? Answer briefly."}],"max_tokens":400,"chat_template_kwargs":{"enable_thinking":false}}'
```

| Command | Expected |
| --- | --- |
| `/v1/models` with the key | `"id": "glm-4.6"`, `"max_model_len": 202752` |
| `/v1/models` without a key | `401` |
| chat completion | `content` holds the answer, `finish_reason: stop`, about 3 output tokens |

- `enable_thinking: false` is what makes the smoke test deterministic. With reasoning on, this model spent
  between 435 and over 2000 tokens on this prompt, so a fixed budget returns empty `content` and
  `finish_reason: length` at unpredictable times.
- Leave reasoning on for real work. It arrives in a separate `reasoning` field.

## 5. Connect a client

```
export NODE=<the node serving it>
source recipes/GLM-4.6-FP8/h200-4/client.env
claude
```

- `client.env` sets `CLAUDE_CODE_MAX_CONTEXT_TOKENS=202752`. Claude Code assumes and enforces a 200k window
  for a served name it does not recognize, which is below what this endpoint serves.
- Use `ANTHROPIC_AUTH_TOKEN`. `ANTHROPIC_API_KEY` sends `x-api-key` and returns 401.
- `client.env` also sets `ANTHROPIC_SMALL_FAST_MODEL`; without it the client reaches for a hosted Haiku.
- OpenAI clients such as Codex: base URL `http://<node>:8000/v1`, same key, model `glm-4.6`. See
  [docs/clients.md](../../../docs/clients.md).

## 6. Stop it

```
scancel <jobid>                        # Slurm path
bash common/tools/stop.sh <node>       # direct path
```

## Tunable inputs

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/GLM-4.6-FP8` | Serve a different copy of the checkpoint |
| `API_PORT` | 8000 | Listening port |
| `MAX_MODEL_LEN` | 202752 | Context window, the checkpoint maximum |
| `GPU_UTIL` | 0.90 | Fraction of VRAM for weights plus KV cache |
| `TP` | 4 | Tensor parallel size, and the node's GPU count |
| `MTP_TOKENS` | 1 | Speculative tokens per step |
| `NO_MTP` | unset | Set to disable speculative decoding |
| `PERF` | unset | Retry CUDA graphs after an engine upgrade; crashes on 0.25.1 |
| `CUDAGRAPH_MODE` | `NONE` | Graph mode passed through when `PERF` is set |
| `EXTRA_ARGS` | unset | Extra flags appended to the `vllm serve` command line |
| `TOOL_PARSER` | `glm45` | Tool call parser |
| `REASONING_PARSER` | `glm45` | Reasoning parser |
| `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` | 3600 | Raised from 480 so a storage stall that resolves is survivable |
| `LOG_DIR` | `/tmp/$USER/vllm` | Where the SSH path writes the server log |
| `VENV_DIR` | under `ENV_ROOT` | Use an environment built elsewhere |
| `VLLM_VERSION` | 0.25.1 | Wheel version `env/build.sh` installs |
| `TRANSFORMERS_VERSION` | 5.14.1 | The transformers version the rates were measured with |
| `VLLM_CACHE_ROOT` | under `ENV_ROOT` | Where vLLM keeps compiled artifacts; the engine default is a small quota here |
| `KEY_NAME`, `KEY_FILE`, `VLLM_API_KEY` | this recipe's key | Which key `common/lib/api_key.sh` resolves; an exported `VLLM_API_KEY` wins |
| `NODE`, `GLM46_NODE` | unset | The node to launch on or connect to; set either, or pass the node as an argument |
| `MODELS_DIR`, `ENV_ROOT` | `common/defaults.sh` | Override there or in `common/site.conf` |

## Benchmarking

Conditions:

| | |
| --- | --- |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| Input length | ISL 14 tokens. Rates at a long input are not measured; use `--prompt-tokens` |
| Output length | OSL 1152 tokens, output only, `ignore_eos` |
| Context | `MAX_MODEL_LEN=202752` |
| Allocation for the measurement | 4 GPUs, 64 cores, 1440 GB, `kempner_h200` |
| Sequence cap | `max_num_seqs` 1024, the engine default, which equals the top sweep level |
| Preemption | 6,390 across the sweep, so the KV cache saturates before the sequence cap does |
| Endpoint | idle, and the benchmark client ran on a separate CPU-only node |
| Power | 700 W enforced, the card default, so not capped. Median 495 W over the sweep and a 704 W peak, with 47 of 4124 samples at or above 690 W |

Results:

| Concurrency | Aggregate | Per stream | Spread over 3 runs | TTFT median |
| --- | --- | --- | --- | --- |
| 1 | 19.1 tok/s | 19.1 tok/s | 19.1 to 19.1 | 398 ms |
| 8 | 151.7 tok/s | 19.0 tok/s | 151.1 to 151.9 | 401 ms |
| 32 | 596.3 tok/s | 18.6 tok/s | 596.1 to 596.7 | 391 ms |
| 64 | 1172.2 tok/s | 18.3 tok/s | 1172.0 to 1176.4 | 386 ms |
| 128 | 2315.9 tok/s | 18.1 tok/s | 2311.8 to 2329.9 | 418 ms |
| 256 | 4821.5 tok/s | 18.8 tok/s | 4814.5 to 4841.6 | 511 ms |
| 512 | 6308.8 tok/s | 12.3 tok/s | 6274.9 to 6321.7 | 644 ms |
| 640 | 6606.4 tok/s | 10.3 tok/s | 6596.7 to 6611.8 | 727 ms |
| 768 | 7114.8 tok/s | 9.3 tok/s | 6747.2 to 7134.7 | 810 ms |
| 896 | 7667.7 tok/s | 8.6 tok/s | 7649.9 to 7703.9 | 893 ms |
| 1024 | 8126.6 tok/s | 7.9 tok/s | 8123.3 to 8154.4 | 953 ms |

| | |
| --- | --- |
| Label | rising. The curve never turns over: 1024 is the highest level swept, 6.0 percent above 896 and 28.8 percent above 512, so this is a floor rather than a ceiling. Finding the peak needs `max_num_seqs` above 1024, a different serving configuration |
| Widest spread | concurrency 768 measured 6747.2 to 7134.7, 5.4 percent. Every other level held within 0.7 percent |
| Quote for one caller | 19.1 tok/s |
| Quote for a shared endpoint | 8126.6 tok/s at concurrency 1024 |
| KV cache | 395,392 tokens from 35.07 GiB per GPU, 1.95 full-length requests at once. The pool is the same at 131072, where it holds 3.02, so raising the context trades concurrency and not pool |
| Speculative decoding | MTP is on by default and accepted 89.2 percent of draft tokens across the sweep, 8,101,735 of 9,078,643 |
| Long prompt | 198,031 tokens in 51 s cold; 1.5 s when the prefix is already cached |

Reproduce:

```
KEY_NAME=GLM-4.6-FP8-h200-4 bash common/tools/bench.sh --host <node> --model glm-4.6
KEY_NAME=GLM-4.6-FP8-h200-4 bash common/tools/bench.sh --host <node> --model glm-4.6 \
  --sweep 1,8,32,64,128,256,512,640,768,896,1024
```

- `KEY_NAME` is required once `secrets/` holds more than one key. Without it `bench.sh` resolves the shared
  key, the server expects this recipe's key, and every request returns 401.

## Known limits

- Eager only. CUDA graph capture takes an illegal memory access on vLLM 0.25.1, so `serve.sh` passes
  `--enforce-eager`. `PERF=1` retries the compile path after an engine upgrade.
- Leave `VLLM_USE_DEEP_GEMM` at 0, which `env/env.sh` sets. The DeepGEMM MoE path takes an illegal memory
  access on H200, reproduced for two other models on this hardware.
- Anthropic's hosted tools return HTTP 400. Use [docs/web-search.md](../../../docs/web-search.md).
