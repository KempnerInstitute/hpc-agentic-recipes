# GLM-5.3-NVFP4 on one RTX node

Status: Validated - vLLM 0.32.0, protocol: slope(128,1152) at concurrency 1, 8, 32, 64, 128 and 256

| | |
| --- | --- |
| Served name | `glm-5.3` |
| Checkpoint | `GLM-5.3-NVFP4`, Hugging Face `Inferact/GLM-5.3-NVFP4` |
| On disk | 432.9 GiB across 88 shards, NVFP4, `GlmMoeDsaForCausalLM` |
| Served precision | NVFP4 via modelopt, group size 16, no excluded modules |
| Context | 217344 |
| Hardware | 1 RTX PRO 6000 Blackwell node, 8 GPUs, 97887 MiB each, TP8 |
| Engine | vLLM 0.32.0 with torch cu130 and FlashInfer 0.7.0.post1, CUDA graphs, no speculative decoding |
| Allocation for the measurement | 1 node, 8 GPUs, 128 cores and 1440 GB, `kempner_rtx` |

This is the four-bit build of GLM-5.3, re-quantized by Inferact. It fits one node, which the native FP8
checkpoint does not, and NVFP4 arithmetic needs Blackwell, so this recipe is RTX only.

## 1. Create the API key

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/GLM-5.3-NVFP4-rtx-8.key
chmod 600 secrets/GLM-5.3-NVFP4-rtx-8.key
```

- `secrets/vllm_api_key` is read when this file is absent.
- Requests without the key receive HTTP 401.

## 2. Build the environment

```
bash recipes/GLM-5.3-NVFP4/rtx-8/env/build.sh
```

Builds into `$ENV_ROOT/GLM-5.3-NVFP4/rtx-8/venv`. sm_120 needs the CUDA 13 toolchain, so this recipe
installs vLLM from the cu130 wheel index and adds FlashInfer separately. A CPU allocation is enough.

## 3. Launch

```
sbatch --account=<your-account> recipes/GLM-5.3-NVFP4/rtx-8/serve.sbatch
```

Single node, so no Ray bring-up. The endpoint is `http://<node>:8000/v1`.

Startup timings are not yet measured here.

## 4. Verify

```
KEY=$(cat secrets/GLM-5.3-NVFP4-rtx-8.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the node>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"glm-5.3","messages":[{"role":"user","content":"What is 2+2? Answer briefly."}],"max_tokens":800}'
```

| Command | Expected |
| --- | --- |
| `/v1/models` with the key | `"id": "glm-5.3"` |
| `/v1/models` without a key | `401` |
| chat completion | `content` holds the answer, `finish_reason: stop` |

Check the answers, not just the rate. A four-bit re-quantization is the case where fast but wrong output
is most likely, and this repo has seen a recipe do exactly that.

## 5. Connect a client

```
export NODE=<the node>
source recipes/GLM-5.3-NVFP4/rtx-8/client.env
claude
```

`client.env` exports the base URL, the key, the served model name, and
`CLAUDE_CODE_MAX_CONTEXT_TOKENS=217344`, without which Claude Code assumes and enforces a 200k window.

## 6. Stop it

```
scancel <job id>
```

Confirm the GPUs are released before relaunching, because workers that outlive the job hold VRAM.

## Tunable inputs

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/GLM-5.3-NVFP4` | Serve a different copy of the checkpoint |
| `API_PORT` | 8000 | Listening port |
| `MAX_MODEL_LEN` | 217344 | Context window; the ceiling this hardware allows is not yet measured |
| `GPU_UTIL` | 0.90 | Fraction of VRAM for weights plus KV cache |
| `TP` | 8 | Tensor parallel size, one node's GPU count |
| `MTP_TOKENS` | 3 | Draft tokens for MTP speculative decoding |
| `NO_MTP` | unset | Set to serve without speculative decoding |
| `TOOL_PARSER` | `glm47` | Tool call parser |
| `REASONING_PARSER` | `glm47` | Reasoning parser |
| `EXTRA_ARGS` | unset | Extra flags appended to the `vllm serve` command line |
| `KEY_NAME`, `KEY_FILE`, `VLLM_API_KEY` | this recipe's key | Which key `common/lib/api_key.sh` resolves; an exported `VLLM_API_KEY` wins |

## Benchmarking

Conditions:

| | |
| --- | --- |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| Input length | ISL 21 tokens. Rates at a long input are not measured; use `--prompt-tokens` |
| Output length | OSL 1152 tokens, output only, `ignore_eos` |
| Context | `MAX_MODEL_LEN=217344` |
| Allocation for the measurement | 1 node, 8 GPUs, 128 cores and 1440 GB, `kempner_rtx` |
| Sweep ceiling | 256. The engine dies above it; see Known limits |
| Endpoint | idle, and the benchmark client ran on the serving node |

Results:

| Concurrency | Aggregate | Per stream | Spread over 3 runs |
| --- | --- | --- | --- |
| 1 | 50.2 tok/s | 50.2 tok/s | 49.1 to 50.4 |
| 8 | 247.5 tok/s | 30.9 tok/s | 247.4 to 247.6 |
| 32 | 586.1 tok/s | 18.3 tok/s | 586.0 to 586.4 |
| 64 | 812.7 tok/s | 12.7 tok/s | 812.3 to 813.4 |
| 128 | 1060.2 tok/s | 8.3 tok/s | 1058.0 to 1060.3 |
| 256 | 1711.9 tok/s | 6.7 tok/s | 1711.8 to 1715.8 |

| | |
| --- | --- |
| Engine facts | KV cache 463,808 tokens, 2.13 full-length requests at once; graph capture 28 s and 1.25 GiB per GPU |
| Label | rising. The curve is still climbing at 256, which is the highest level this endpoint survives, so 1711.9 is a floor |

## Known limits

- The engine dies above concurrency 256. At 512 the first run returned HTTP 500 and the engine core then
  stopped answering, with `RPC call to sample_tokens timed out` and `EngineDeadError` in the log, after
  which every higher level refused connections. The sweep therefore stops at 256 and that figure is a
  floor, not a ceiling.
- Speculative decoding is unavailable. The checkpoint declares NVFP4 with an empty ignore list, but the
  MTP draft head's `parallel_lm_head` is stored in BF16, so the engine fails to load it with
  `NVFP4 weight_scale ... was never loaded (still NaN)`. Upstream's own command for this variant carries
  no speculative config either.
- FlashInfer autotuning must be off. With it on, a worker dies during kernel warmup on sm_120, so the
  scripts pass `--kernel-config` to disable it.
- NVFP4 arithmetic needs Blackwell. There is no H200 variant of this checkpoint and there cannot be one;
  on Hopper it would fall back to emulation and lose the point of four-bit weights.
- The quantization is third party, not from the model author, so published quality scores for GLM-5.3
  are an upper bound for this build.
