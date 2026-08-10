# GLM-5.2-FP8 on two H200 nodes

Status: Validated - vLLM 0.25.1+cu129, protocol: slope(128,1152) at concurrency 1, 8, 32, 64, 128, 256, 512, 640, 768, 896 and 1024

| | |
| --- | --- |
| Served name | `glm-5.2` |
| Checkpoint | `GLM-5.2-FP8`, Hugging Face `zai-org/GLM-5.2-FP8` |
| On disk | 704 GiB across 141 shards, FP8 e4m3, `GlmMoeDsaForCausalLM` |
| Served precision | FP8 with dynamic activations, `weight_block_size` [128, 128] |
| Context | 641664. The checkpoint declares 1048576, which does not fit; see Known limits |
| Hardware | 2 H200 nodes, 8 GPUs, 143771 MiB each, TP4 inside a node and PP2 between over Ray |
| Engine | vLLM 0.25.1+cu129, eager, no speculative decoding |

## 1. Create the API key

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/GLM-5.2-FP8-h200-4-nodes2.key
chmod 600 secrets/GLM-5.2-FP8-h200-4-nodes2.key
```

- `secrets/vllm_api_key` is read when this file is absent.
- To rotate, replace the file and relaunch; the engine reads it once at launch.

## 2. Build the environment

Run on a compute node, not a login node. Build once: `ENV_ROOT` is shared scratch, so both nodes activate
the same environment.

```
bash recipes/GLM-5.2-FP8/h200-4-nodes2/env/build.sh
```

- About 13 GB under `ENV_ROOT`, default scratch. Set `ENV_ROOT` in `common/site.conf` to move it.
- Installs the cu129 wheel. These nodes run driver 575, CUDA 12.9, and vLLM's default wheel is CUDA 13.
- Installs `ray[default]` alongside vLLM, because this recipe spans two nodes through the Ray executor.
- `ENV_ROOT` and `MODELS_DIR` must resolve to the same content on both nodes. The Ray worker imports vLLM
  and reads weights itself, so a path that exists only on the head fails after the cluster has formed,
  which reads as an engine fault rather than a path fault.

## 3. Launch

Slurm, from the repo root. The batch script takes both nodes, starts a Ray head on the first and a worker
on the second, then runs the engine on the head:

```
sbatch --account=<your-account> recipes/GLM-5.2-FP8/h200-4-nodes2/serve.sbatch
squeue --me                                  # NODELIST, the FIRST name serves
tail -f glm52-fp8-<jobid>.log
```

Direct, on two nodes you already hold:

```
bash recipes/GLM-5.2-FP8/h200-4-nodes2/serve_ssh.sh <head_node> <worker_node>
```

| Stage | Measured |
| --- | --- |
| Launcher to serving, first time on a node pair | 13 min 49 s |
| Launcher to serving, caches warm | 5 min 22 s |
| Weight load, 141 shards across 8 ranks | 342 s worker stage, 373 s head stage |

- The launcher prints `cluster GPUs: 8.0` before loading any weights. If it says 4, the worker did not
  join; stop there rather than discovering it minutes into a weight load.
- The endpoint is on the head node only. The worker holds half the layers and refuses HTTP.
- On the direct path the server log is node-local at `/tmp/$USER/vllm/`. Under Slurm it lands in the submit
  directory.

## 4. Verify

```
KEY=$(cat secrets/GLM-5.2-FP8-h200-4-nodes2.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the head node>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"What is 2+2? Answer briefly."}],"max_tokens":400}'
```

| Command | Expected |
| --- | --- |
| `/v1/models` with the key | `"id": "glm-5.2"`, `"max_model_len": 641664` |
| `/v1/models` without a key | `401` |
| chat completion | `content` holds the answer, `finish_reason: stop` |

- 400 output tokens is enough: three smoke tests used 86 to 88 tokens. Reasoning arrives in a separate
  `reasoning` field, so a smaller budget can end with empty `content`.

## 5. Connect a client

```
export NODE=<the head node>
source recipes/GLM-5.2-FP8/h200-4-nodes2/client.env
claude
```

- `client.env` sets `CLAUDE_CODE_MAX_CONTEXT_TOKENS=641664`. Claude Code assumes and enforces a 200k window
  for a served name it does not recognize, far below what this endpoint serves.
- Use `ANTHROPIC_AUTH_TOKEN`. `ANTHROPIC_API_KEY` sends `x-api-key` and returns 401.
- `client.env` also sets `ANTHROPIC_SMALL_FAST_MODEL`; without it the client reaches for a hosted Haiku.
- OpenAI clients such as Codex: base URL `http://<head-node>:8000/v1`, same key, model `glm-5.2`. See
  [docs/clients.md](../../../docs/clients.md).

## 6. Stop it

```
scancel <jobid>                                        # Slurm path, owns both nodes
bash common/tools/stop.sh <head_node> <worker_node>    # direct path, name both
```

- The direct path has no scheduler to clean up, so both nodes must be named. `stop.sh` stops Ray as well as
  the server.
- Do not run `ssh <node> ray stop` by hand. `ray` lives in the recipe venv and is not on `PATH` in a
  non-interactive shell, so it silently does nothing.
- Both nodes must be clean before a relaunch. A surviving Ray cluster is the most common cause of a launch
  that never becomes ready, and it fails on resources rather than memory, which reads like a different
  problem. Ray also holds host RAM while showing no GPU memory.

## Tunable inputs

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/GLM-5.2-FP8` | Serve a different copy of the checkpoint |
| `API_PORT` | 8000 | Listening port |
| `MAX_MODEL_LEN` | 641664 | Context window, the ceiling this hardware allows |
| `GPU_UTIL` | 0.90 | Fraction of VRAM for weights plus KV cache |
| `TP` | 4 | Tensor parallel size; 4 is one node's GPU count and must stay inside a node |
| `PP` | 2 | Pipeline parallel size; 2 is the node count |
| `PERF` | unset | Retry CUDA graph capture instead of eager; crashes on 0.25.1 |
| `CUDAGRAPH_MODE` | `NONE` | Graph mode passed through when `PERF` is set |
| `EXTRA_ARGS` | unset | Extra flags appended to the `vllm serve` command line |
| `TOOL_PARSER` | `glm45` | Tool call parser |
| `REASONING_PARSER` | `glm45` | Reasoning parser |
| `RAY_PORT` | 6379 | Ray head port |
| `RAY_HEAD_IP` | unset | Head address, when calling `serve.sh` directly |
| `GPUS_PER_NODE` | `SLURM_GPUS_ON_NODE`, then 4 | GPUs each Ray node advertises |
| `RAY_BLOCK` | unset on the SSH path, 1 under Slurm | Keep `ray start` in the foreground |
| `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` | 3600 | Raised from 480 so a storage stall that resolves is survivable |
| `NCCL_DEBUG` | `WARN` | NCCL log level; raise to `INFO` to debug a cross-node hang |
| `VLLM_VERSION` | 0.25.1 | Release wheel version `env/build.sh` installs |
| `LOG_DIR` | `/tmp/$USER/vllm` | Where the SSH path writes the server log |
| `VENV_DIR` | under `ENV_ROOT` | Use an environment built elsewhere |
| `VLLM_CACHE_ROOT` | under `ENV_ROOT` | Where vLLM keeps compiled artifacts; the engine default is a small quota here |
| `KEY_NAME`, `KEY_FILE`, `VLLM_API_KEY` | this recipe's key | Which key `common/lib/api_key.sh` resolves; an exported `VLLM_API_KEY` wins |
| `NODE`, `GLM52_HEAD`, `GLM52_WORKER` | unset | Nodes for the SSH path; pass them as arguments instead |
| `MODELS_DIR`, `ENV_ROOT` | `common/defaults.sh` | Override there or in `common/site.conf` |

## Benchmarking

Conditions:

| | |
| --- | --- |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| Input length | ISL 21 tokens. Rates at a long input are not measured; use `--prompt-tokens` |
| Output length | OSL 1152 tokens, output only, `ignore_eos` |
| Context | `MAX_MODEL_LEN=641664` |
| Allocation for the measurement | 2 nodes, 4 GPUs each, 64 cores and 1440 GB per node, `kempner_eng` |
| Sequence cap | `max_num_seqs` 1024, the engine default, which equals the top sweep level |
| Preemption | 3,500 across the sweep |
| Endpoint | idle, and the benchmark client ran on a separate CPU-only node |
| Power | 700 W enforced, the card default, so not capped. Median 352 and 367 W on the two nodes, peaks of 507 and 513 W, nothing at or above 690 |

Results:

| Concurrency | Aggregate | Per stream | Spread over 3 runs | TTFT median |
| --- | --- | --- | --- | --- |
| 1 | 12.9 tok/s | 12.9 tok/s | 12.9 to 12.9 | 236 ms |
| 8 | 100.9 tok/s | 12.6 tok/s | 99.6 to 101.6 | 240 ms |
| 32 | 396.7 tok/s | 12.4 tok/s | 392.1 to 396.8 | 242 ms |
| 64 | 789.4 tok/s | 12.3 tok/s | 788.6 to 790.9 | 429 ms |
| 128 | 1572.1 tok/s | 12.3 tok/s | 1566.6 to 1575.8 | 414 ms |
| 256 | 3066.3 tok/s | 12.0 tok/s | 3043.2 to 3097.8 | 596 ms |
| 512 | 5392.0 tok/s | 10.5 tok/s | 5379.9 to 5392.7 | 843 ms |
| 640 | 5118.7 tok/s | 8.0 tok/s | 5116.9 to 5123.6 | 980 ms |
| 768 | 4853.2 tok/s | 6.3 tok/s | 4852.5 to 4879.0 | 1110 ms |
| 896 | 4669.6 tok/s | 5.2 tok/s | 4613.8 to 4862.4 | 1257 ms |
| 1024 | 5162.8 tok/s | 5.0 tok/s | 5118.2 to 5168.8 | 1350 ms |

| | |
| --- | --- |
| Label | peak. Throughput turns over at 512 and varies 13.4 percent across 512 to 1024, well past the 4 percent that would make it saturated. The curve is not monotonic above the peak: 1024 recovers to 5162.8, above 768 and 896 |
| Quote for one caller | 12.9 tok/s |
| Quote for a shared endpoint | 5392.0 tok/s at concurrency 512 |
| KV cache | 659,584 tokens, 1.03 full-length requests at once. Split unevenly by stage: the head rank has 36.59 GiB and the worker rank less, and the worker is what binds |
| Cost of the context | measured against 131072 on the same nodes: pool falls 5.5 percent from 698,176, peak aggregate falls 0.6 percent from 5422.1, and the peak moves from concurrency 640 to 512. At 640 itself the larger window measures 5.6 percent lower |
| Long prompt | 612,672 tokens in 74 s cold; 3.5 s when the prefix is already cached |

Reproduce:

```
KEY_NAME=GLM-5.2-FP8-h200-4-nodes2 bash common/tools/bench.sh --host <head_node> --model glm-5.2
KEY_NAME=GLM-5.2-FP8-h200-4-nodes2 bash common/tools/bench.sh --host <head_node> --model glm-5.2 \
  --sweep 1,8,32,64,128,256,512,640,768,896,1024
```

- `KEY_NAME` is required once `secrets/` holds more than one key, otherwise `bench.sh` resolves the shared
  key and every request returns 401.

## Known limits

- The checkpoint declares 1048576 and the engine refuses it: one request that long needs 45.42 GiB of KV
  against 26.33 available on the binding stage. vLLM's first estimate, 798528, is computed on the head stage
  and is also refused; the fixed point is 641664. Context shrinks the pool as it grows, because the
  sparse-attention workspace scales with the declared window.
- No speculative decoding. The checkpoint ships an MTP head, `num_nextn_predict_layers` 1, but vLLM rejects
  a speculative config whenever pipeline parallelism is active, and 704 GiB of weights needs two nodes.
- Keep tensor parallelism inside a node. TP8 across two nodes is legal by the FP8 block constraint, since
  `moe_intermediate_size` 2048 shards to 256, but it hangs at NCCL initialization.
- Eager only. CUDA graph capture takes an illegal memory access on vLLM 0.25.1, so `serve.sh` passes
  `--enforce-eager`. Leave `VLLM_USE_DEEP_GEMM` at 0, which `env/env.sh` sets; this model is where that
  crash was first diagnosed.
- A Ray worker can die and the head log will not say why. One run served intermittently for about 21 hours
  and ended with `RayWorkerProc rank=[1] died unexpectedly`. Rank 1 is the worker stage, so look at
  `ssh <worker_node> 'ls -t /tmp/ray/session_latest/logs | head'` before theorizing.
- Anthropic's hosted tools return HTTP 400. Use [docs/web-search.md](../../../docs/web-search.md).
