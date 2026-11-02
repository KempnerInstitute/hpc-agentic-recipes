# GLM-5.3 on two H200 nodes

Status: Untested - vLLM 0.32.0+cu129, not yet measured on this cluster

| | |
| --- | --- |
| Served name | `glm-5.3` |
| Checkpoint | `GLM-5.3`, Hugging Face `zai-org/GLM-5.3` |
| On disk | 703.5 GiB across 141 shards, FP8 e4m3, `GlmMoeDsaForCausalLM` |
| Served precision | FP8 with dynamic activations, `weight_block_size` [128, 128] |
| Context | Not yet fixed. The checkpoint declares 1048576; the ceiling this hardware allows is a KV budget question answered by the first bring-up |
| Hardware | 2 H200 nodes, 8 GPUs, 143771 MiB each, TP4 inside a node and PP2 between over Ray |
| Engine | vLLM 0.32.0+cu129, CUDA graphs at `FULL_AND_PIECEWISE` |
| Allocation for the measurement | 2 nodes, 4 GPUs each, 64 cores and 1440 GB per node, `kempner_h200` |

GLM-5.3 is a 743B-parameter mixture-of-experts with 39B active per token, 78 layers and 256 routed experts,
using the same hybrid sparse-attention backbone as GLM-5.2. It carries a 5-token MTP head and a 1M-token
native window. This is a text recipe.

## 1. Create the API key

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/GLM-5.3-h200-4-nodes2.key
chmod 600 secrets/GLM-5.3-h200-4-nodes2.key
```

- `secrets/vllm_api_key` is read when this file is absent.
- Requests without the key receive HTTP 401.

## 2. Build the environment

```
bash recipes/GLM-5.3/h200-4-nodes2/env/build.sh
```

Builds into `$ENV_ROOT/GLM-5.3/h200-4-nodes2/venv` and pins vLLM 0.32.0+cu129 with transformers 5.19.0.
The same path must be reachable from both nodes, which it is on shared scratch. Run it on a compute node:
a CPU allocation is enough, and it takes a few minutes.

## 3. Launch

```
sbatch --account=<your-account> recipes/GLM-5.3/h200-4-nodes2/serve.sbatch
```

The job starts a Ray head on the first node, joins the second, then serves from the head. The endpoint is
`http://<head>:8000/v1`, and the job log names the head node on its first line.

Startup timings are not yet measured here.

## 4. Verify

```
KEY=$(cat secrets/GLM-5.3-h200-4-nodes2.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the head node>

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

This model reasons before answering and returns that text in a separate field, so allow generous output
tokens when probing.

## 5. Connect a client

```
export NODE=<the head node>
source recipes/GLM-5.3/h200-4-nodes2/client.env
claude
```

`client.env` exports the base URL, the key, the served model name, and
`CLAUDE_CODE_MAX_CONTEXT_TOKENS=641664`, without which Claude Code assumes and enforces a 200k window. Use `ANTHROPIC_AUTH_TOKEN`, not
`ANTHROPIC_API_KEY`, which sends a header the engine ignores and yields 401 on every request.

## 6. Stop it

```
scancel <job id>
```

Ray and the engine exit with the job. Confirm the GPUs are released before launching again, because
workers that outlive the job hold VRAM and the next launch then fails to allocate its pool.

## Tunable inputs

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/GLM-5.3` | Serve a different copy of the checkpoint |
| `API_PORT` | 8000 | Listening port |
| `MAX_MODEL_LEN` | 641664 | Context window; the ceiling this hardware allows is not yet measured |
| `GPU_UTIL` | 0.94 | Fraction of VRAM for weights plus KV cache |
| `TP` | 4 | Tensor parallel size; 4 is one node's GPU count and must stay inside a node |
| `PP` | 2 | Pipeline parallel size; 2 is the node count |
| `KV_DTYPE` | `fp8_e4m3` | KV cache dtype |
| `MTP_TOKENS` | 0 | Draft tokens for MTP speculative decoding; 5 is what the checkpoint ships |
| `ENFORCE_EAGER` | unset | Set to fall back to eager instead of CUDA graphs |
| `CUDAGRAPH_MODE` | `FULL_AND_PIECEWISE` | Graph mode, ignored when `ENFORCE_EAGER` is set |
| `TOOL_PARSER` | `glm47` | Tool call parser |
| `REASONING_PARSER` | `glm47` | Reasoning parser |
| `EXTRA_ARGS` | unset | Extra flags appended to the `vllm serve` command line |
| `KEY_NAME`, `KEY_FILE`, `VLLM_API_KEY` | this recipe's key | Which key `common/lib/api_key.sh` resolves; an exported `VLLM_API_KEY` wins |

## Benchmarking

Nothing is measured yet. When it is, the rate is taken with `common/tools/bench.sh` against an endpoint
this recipe's own scripts launched, using `slope(128,1152)` so the figure is comparable with every other
recipe here.

## Known limits

- Not yet measured on this cluster, so no rate, context ceiling or startup time is published. The Status
  line stays `Untested` until it is.
- The checkpoint ships a 5-token MTP head. `MTP_TOKENS` defaults to 0 because speculative decoding under
  pipeline parallelism is not yet confirmed working on this engine version, and the sibling GLM-5.2-FP8
  recipe serves without it.
- NVFP4 is a separate checkpoint and a separate recipe; it needs Blackwell and will not run here.
