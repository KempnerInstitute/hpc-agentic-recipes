# GLM-5.3 on two H200 nodes

Status: Validated - vLLM 0.32.0+cu129, protocol: slope(128,1152) at concurrency 1, 8, 32, 64, 128, 256,
512, 640, 768, 896 and 1024

| | |
| --- | --- |
| Served name | `glm-5.3` |
| Checkpoint | `GLM-5.3`, Hugging Face `zai-org/GLM-5.3` |
| On disk | 703.5 GiB across 141 shards, FP8 e4m3, `GlmMoeDsaForCausalLM` |
| Served precision | FP8 with dynamic activations, `weight_block_size` [128, 128] |
| Context | 262144. The checkpoint declares 1048576; MTP needs about 14.7 GiB per GPU, which comes out of the KV pool |
| Hardware | 2 H200 nodes, 8 GPUs, 143771 MiB each, TP4 inside a node and PP2 between over Ray |
| Engine | vLLM 0.32.0+cu129, CUDA graphs at `FULL_AND_PIECEWISE`, MTP speculative decoding at 5 draft tokens |
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
`CLAUDE_CODE_MAX_CONTEXT_TOKENS=262144`, without which Claude Code assumes and enforces a 200k window. Use `ANTHROPIC_AUTH_TOKEN`, not
`ANTHROPIC_API_KEY`, which sends a header the engine ignores and yields 401 on every request.

- OpenAI clients such as Codex: base URL `http://<node>:8000/v1`, same key, model `glm-5.3`, and
  `wire_api = "responses"`, which recent versions require. See [docs/clients.md](../../../docs/clients.md).

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
| `MAX_MODEL_LEN` | 262144 | Context window. MTP takes about 14.7 GiB per GPU out of the KV pool, so the ceiling is lower than the FP8 sibling's |
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

Conditions:

| | |
| --- | --- |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| Input length | ISL 21 tokens. Rates at a long input are not measured; use `--prompt-tokens` |
| Output length | OSL 1152 tokens, output only, `ignore_eos` |
| Context | `MAX_MODEL_LEN=262144` |
| Allocation for the measurement | 2 nodes, 4 GPUs each, 64 cores and 1440 GB per node, `kempner_h200` |
| Sequence cap | `max_num_seqs` 1024, the engine default, which equals the top sweep level |
| Endpoint | idle, and the benchmark client ran on the serving node |

Results:

| Concurrency | Aggregate | Per stream | Spread over 3 runs |
| --- | --- | --- | --- |
| 1 | 142.3 tok/s | 142.3 tok/s | 142.3 to 142.3 |
| 8 | 349.3 tok/s | 43.7 tok/s | 344.1 to 349.3 |
| 32 | 991.9 tok/s | 31.0 tok/s | 857.8 to 996.2 |
| 64 | 1288.7 tok/s | 20.1 tok/s | 1285.9 to 1298.9 |
| 128 | 1890.1 tok/s | 14.8 tok/s | 1786.3 to 1933.3 |
| 256 | 2305.4 tok/s | 9.0 tok/s | 2227.3 to 2544.5 |
| 512 | 2783.4 tok/s | 5.4 tok/s | 2711.7 to 2843.6 |
| 640 | 2847.2 tok/s | 4.4 tok/s | 2759.4 to 3304.8 |
| 768 | 3404.5 tok/s | 4.4 tok/s | 2938.2 to 3511.0 |
| 896 | 3040.9 tok/s | 3.4 tok/s | 2981.8 to 3547.1 |
| 1024 | 3597.4 tok/s | 3.5 tok/s | 3061.3 to 3677.5 |

| | |
| --- | --- |
| Engine facts | KV cache 1,297,152 tokens, 4.95 full-length requests at once; graph capture 15 s and 1.60 GiB per GPU |
| Label | rising. The curve is still climbing at 1024, the engine's own sequence cap, so 3597.4 is a floor |

## Choosing MTP or plain decoding

MTP is on by default because it nearly doubles the rate a single caller sees. It costs aggregate
throughput above roughly concurrency 256, so set `MTP_TOKENS=0` for a busy shared endpoint. Both columns
were measured at the same 262144 context, so the difference is speculation alone.

| | MTP 5, the default | `MTP_TOKENS=0` |
| --- | --- | --- |
| One caller | 142.3 tok/s | 75.1 tok/s |
| Concurrency 8 | 349.3 tok/s | 195.7 tok/s |
| Concurrency 256 | 2305.4 tok/s | 2041.7 tok/s |
| Concurrency 1024 | 3597.4 tok/s | 4108.8 tok/s |
| KV cache | 1,297,152 tokens | 1,567,552 tokens |

## Known limits

- Aggregate throughput is noisy at the top of the sweep, varying up to 29 percent across the levels at or
  above 512, and the curve is not monotonic: 3404.5 at 768, 3040.9 at 896, 3597.4 at 1024. Treat any single
  high-concurrency figure here as approximate, and prefer the three-run spread column.
- MTP is served by default and does run under pipeline parallelism on this engine version, which the
  GLM-5.2-FP8 sibling could not do. It costs about 14.7 GiB per GPU and roughly 270,000 tokens of KV pool,
  which is why this recipe serves 262144 rather than the 641664 the FP8 sibling reaches.
- Weight load is dominated by the filesystem. Reading 703.5 GiB from scratch took 37 minutes cold at about
  34 seconds per shard, falling to 1.6 seconds per shard once the page cache was warm.
- NVFP4 is a separate checkpoint and a separate recipe; it needs Blackwell and will not run here.
