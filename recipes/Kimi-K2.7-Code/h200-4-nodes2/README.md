# Kimi-K2.7-Code on two H200 nodes

Status: Validated - vLLM 0.25.1+cu129, protocol: slope(128,1152) at concurrency 1, 8, 32, 64, 128, 256, 512, 640, 768, 896 and 1024

| | |
| --- | --- |
| Served name | `kimi-k2.7-code` |
| Checkpoint | `Kimi-K2.7-Code`, Hugging Face `moonshotai/Kimi-K2.7-Code` |
| On disk | 554.3 GiB across 64 shards, INT4, `KimiK25ForConditionalGeneration`, multimodal |
| Served precision | INT4 pack-quantized at group size 32, attention and shared experts left unquantized |
| Context | 262144, the checkpoint maximum |
| Hardware | 2 H200 nodes, 8 GPUs, 143771 MiB each, TP4 inside a node and PP2 between over Ray |
| Engine | vLLM 0.25.1+cu129, CUDA graphs at `FULL_AND_PIECEWISE`, no speculative decoding |

## 1. Create the API key

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/Kimi-K2.7-Code-h200-4-nodes2.key
chmod 600 secrets/Kimi-K2.7-Code-h200-4-nodes2.key
```

- `secrets/vllm_api_key` is read when this file is absent.
- To rotate, replace the file and relaunch; the engine reads it once at launch.

## 2. Build the environment

Run on a compute node, not a login node. Build once: `ENV_ROOT` is shared scratch, so both nodes activate
the same environment.

```
bash recipes/Kimi-K2.7-Code/h200-4-nodes2/env/build.sh
```

- About 13 GB under `ENV_ROOT`, default scratch. Set `ENV_ROOT` in `common/site.conf` to move it.
- Installs the cu129 wheel. These nodes run driver 575, CUDA 12.9, and vLLM's default wheel is CUDA 13.
- Installs `ray[default]` alongside vLLM, because this recipe spans two nodes through the Ray executor.
- `ENV_ROOT` and `MODELS_DIR` must resolve to the same content on both nodes. The Ray worker imports vLLM
  and reads weights itself, so a path that exists only on the head fails after the cluster has formed.

## 3. Launch

Slurm, from the repo root. The batch script takes both nodes, starts a Ray head on the first and a worker
on the second, then runs the engine on the head:

```
sbatch --account=<your-account> recipes/Kimi-K2.7-Code/h200-4-nodes2/serve.sbatch
squeue --me                                  # NODELIST, the FIRST name serves
tail -f kimi-h200-<jobid>.log
```

Direct, on two nodes you already hold:

```
bash recipes/Kimi-K2.7-Code/h200-4-nodes2/serve_ssh.sh <head_node> <worker_node>
```

| Stage | Measured |
| --- | --- |
| Launcher to serving | 12 min 1 s |
| Weight load, 64 shards across 8 ranks | 505 s and 518 s by stage |
| CUDA graph capture | 12 s, and 2.75 GiB per GPU |

- The launcher prints `cluster GPUs: 8.0` before loading any weights. If it says 4, the worker did not
  join; stop there rather than discovering it twenty minutes into a weight load.
- The endpoint is on the head node only. The worker holds half the layers and refuses HTTP.
- On the direct path the server log is node-local at `/tmp/$USER/vllm/`. Under Slurm it lands in the submit
  directory.

## 4. Verify

```
KEY=$(cat secrets/Kimi-K2.7-Code-h200-4-nodes2.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the head node>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"kimi-k2.7-code","messages":[{"role":"user","content":"What is 2+2? Answer briefly."}],"max_tokens":400}'
```

| Command | Expected |
| --- | --- |
| `/v1/models` with the key | `"id": "kimi-k2.7-code"`, `"max_model_len": 262144` |
| `/v1/models` without a key | `401` |
| chat completion | `content` holds the answer, `finish_reason: stop` |

- 400 output tokens is enough: three smoke tests used 17 to 43 tokens.
- Image input is untested. Startup skips multimodal profiling, so the engine never sizes the vision path;
  text is what this recipe is validated for.

## 5. Connect a client

```
export NODE=<the head node>
source recipes/Kimi-K2.7-Code/h200-4-nodes2/client.env
claude
```

- No output cap is needed at 262144. A 32000-token output request is accepted.
- `client.env` sets `CLAUDE_CODE_MAX_CONTEXT_TOKENS=262144`. Claude Code assumes and enforces a 200k window
  for a served name it does not recognize.
- Use `ANTHROPIC_AUTH_TOKEN`. `ANTHROPIC_API_KEY` sends `x-api-key` and returns 401.
- `client.env` also sets `ANTHROPIC_SMALL_FAST_MODEL`; without it the client reaches for a hosted Haiku.
- OpenAI clients such as Codex: base URL `http://<head-node>:8000/v1`, same key, model `kimi-k2.7-code`.
  See [docs/clients.md](../../../docs/clients.md).

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
  that never becomes ready, and it fails on resources rather than memory.

## Tunable inputs

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/Kimi-K2.7-Code` | Serve a different copy of the checkpoint |
| `API_PORT` | 8000 | Listening port |
| `MAX_MODEL_LEN` | 262144 | Context window, the checkpoint maximum |
| `GPU_UTIL` | 0.90 | Fraction of VRAM for weights plus KV cache |
| `TP` | 4 | Tensor parallel size; 4 is one node's GPU count and must stay inside a node |
| `PP` | 2 | Pipeline parallel size; 2 is the node count |
| `EXTRA_ARGS` | `--skip-mm-profiling --mm-processor-cache-gb 0` | **Replaces** this default rather than adding to it. Anything you pass must repeat both flags, or startup hangs at multimodal profiling forever |
| `ENFORCE_EAGER` | unset | Set to fall back to eager, which measures 30.2 tok/s on one stream against 103.0 with graphs |
| `CUDAGRAPH_MODE` | `FULL_AND_PIECEWISE` | Graph mode, ignored when `ENFORCE_EAGER` is set |
| `ATTN_BACKEND` | unset | Override vLLM's attention backend selection |
| `TOOL_PARSER` | `kimi_k2` | Tool call parser |
| `REASONING_PARSER` | `kimi_k2` | Reasoning parser |
| `RAY_PORT` | 6379 | Ray head port |
| `RAY_HEAD_IP` | unset | Head address, when calling `serve.sh` directly |
| `GPUS_PER_NODE` | `SLURM_GPUS_ON_NODE`, then 4 | GPUs each Ray node advertises |
| `RAY_BLOCK` | unset on the SSH path, 1 under Slurm | Keep `ray start` in the foreground |
| `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME` | `ib0` | Pin cross-node transport to InfiniBand |
| `NCCL_IB_HCA` | detected | Only the InfiniBand ports actually up; an inactive HCA in the list stalls initialization |
| `NCCL_DEBUG` | `WARN` | NCCL log level; raise to `INFO` to debug a cross-node hang |
| `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` | 3600 | Raised from 480 so a storage stall that resolves is survivable |
| `VLLM_VERSION` | 0.25.1 | Release wheel version `env/build.sh` installs |
| `TRANSFORMERS_VERSION` | 5.14.1 | The transformers version the rates were measured with |
| `LOG_DIR` | `/tmp/$USER/vllm` | Where the SSH path writes the server log |
| `VENV_DIR` | under `ENV_ROOT` | Use an environment built elsewhere |
| `VLLM_CACHE_ROOT` | under `ENV_ROOT` | Where vLLM keeps compiled artifacts; the engine default is a small quota here |
| `KEY_NAME`, `KEY_FILE`, `VLLM_API_KEY` | this recipe's key | Which key `common/lib/api_key.sh` resolves; an exported `VLLM_API_KEY` wins |
| `NODE`, `KIMI_HEAD`, `KIMI_WORKER` | unset | Nodes for the SSH path; pass them as arguments instead |
| `MODELS_DIR`, `ENV_ROOT` | `common/defaults.sh` | Override there or in `common/site.conf` |

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
| Preemption | none at any level, with the KV pool far larger than the sweep needs |
| Endpoint | idle, and the benchmark client ran on a separate CPU-only node |
| Power | 700 W enforced, the card default, so not capped. Median 419 and 425 W on the two nodes, peaks of 468 and 479 W, nothing at or above 690 |

Results:

| Concurrency | Aggregate | Per stream | Spread over 3 runs | TTFT median |
| --- | --- | --- | --- | --- |
| 1 | 103.0 tok/s | 103.0 tok/s | 102.9 to 103.0 | 38 ms |
| 8 | 541.1 tok/s | 67.6 tok/s | 536.0 to 545.5 | 52 ms |
| 32 | 1250.4 tok/s | 39.1 tok/s | 1234.8 to 1255.9 | 121 ms |
| 64 | 1861.6 tok/s | 29.1 tok/s | 1852.5 to 1867.9 | 185 ms |
| 128 | 2730.7 tok/s | 21.3 tok/s | 2724.3 to 2738.1 | 270 ms |
| 256 | 3713.3 tok/s | 14.5 tok/s | 3704.7 to 3720.2 | 380 ms |
| 512 | 5741.6 tok/s | 11.2 tok/s | 5741.5 to 5753.3 | 619 ms |
| 640 | 6081.1 tok/s | 9.5 tok/s | 6073.3 to 6084.2 | 710 ms |
| 768 | 6097.7 tok/s | 7.9 tok/s | 6092.1 to 6115.2 | 823 ms |
| 896 | 6486.6 tok/s | 7.2 tok/s | 6483.9 to 6487.0 | 886 ms |
| 1024 | 7033.4 tok/s | 6.9 tok/s | 7033.0 to 7039.5 | 928 ms |

| | |
| --- | --- |
| Label | rising. The highest value is at 1024, the top of the sweep, so 7033.4 tok/s is a floor rather than a ceiling. Finding the peak needs `max_num_seqs` above 1024, a different serving configuration |
| Quote for one caller | 103.0 tok/s |
| Quote for a shared endpoint | 7033.4 tok/s at concurrency 1024 |
| KV cache | 1,466,752 tokens, 5.60 full-length requests at once. Graph capture takes 2.75 GiB per GPU, which is why the pool is 88,736 tokens smaller than eager's 1,555,488 |
| Long prompt | 180,011 tokens in 17.0 s cold; 0.31 s when the prefix is already cached |

Reproduce:

```
KEY_NAME=Kimi-K2.7-Code-h200-4-nodes2 bash common/tools/bench.sh --host <head_node> --model kimi-k2.7-code
KEY_NAME=Kimi-K2.7-Code-h200-4-nodes2 bash common/tools/bench.sh --host <head_node> --model kimi-k2.7-code \
  --sweep 1,8,32,64,128,256,512,640,768,896,1024
```

- `KEY_NAME` is required once `secrets/` holds more than one key, otherwise `bench.sh` resolves the shared
  key and every request returns 401.

## Known limits

- Graph capture needs the allreduce and RMSNorm fusion pass disabled, which `serve.sh` does through
  `--compilation-config`. With the pass on, capture takes an illegal memory access and a CUBLAS failure and the
  engine never starts. If you replace the compilation config, keep `"pass_config": {"fuse_allreduce_rms": false}`.
- Do not set `EXTRA_ARGS` without repeating `--skip-mm-profiling --mm-processor-cache-gb 0`. It replaces the
  default rather than adding to it, and without those flags a multimodal checkpoint on more than one node
  completes weight loading and then hangs at multimodal profiling with no error and no timeout.
- No speculative decoding. vLLM rejects a speculative config whenever pipeline parallelism is active, and
  554.3 GiB of weights needs two nodes.
- Keep tensor parallelism inside a node. Tensor parallelism spanning two nodes hangs at NCCL initialization.
- Leave `VLLM_USE_DEEP_GEMM` at 0, which `env/env.sh` sets. The DeepGEMM MoE path takes an illegal memory
  access on H200 for two other checkpoints on this hardware.
- Anthropic's hosted tools return HTTP 400. Use [docs/web-search.md](../../../docs/web-search.md).
