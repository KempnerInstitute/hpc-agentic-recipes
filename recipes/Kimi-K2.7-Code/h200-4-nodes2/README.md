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
| Engine | vLLM 0.25.1+cu129, eager, no speculative decoding |

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
| Launcher to serving | 14 min 42 s to 30 min 16 s, across three launches on one node pair |
| Weight load, 64 shards across 8 ranks | 821 s worker stage, 1328 s head stage |

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
| `ENFORCE_EAGER` | 1 | Eager is the default and is what the rates below were measured with. Pass an explicitly empty value to try CUDA graphs, which has never been attempted for this model on this hardware |
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
| 1 | 30.2 tok/s | 30.2 tok/s | 29.7 to 30.3 | 100 ms |
| 8 | 239.8 tok/s | 30.0 tok/s | 239.7 to 240.1 | 101 ms |
| 32 | 925.1 tok/s | 28.9 tok/s | 922.4 to 926.4 | 176 ms |
| 64 | 1750.9 tok/s | 27.4 tok/s | 1749.6 to 1763.9 | 229 ms |
| 128 | 2633.8 tok/s | 20.6 tok/s | 2626.9 to 2642.1 | 297 ms |
| 256 | 3624.5 tok/s | 14.2 tok/s | 3619.7 to 3638.0 | 415 ms |
| 512 | 5661.9 tok/s | 11.1 tok/s | 5661.3 to 5667.4 | 637 ms |
| 640 | 6125.7 tok/s | 9.6 tok/s | 6121.8 to 6129.9 | 760 ms |
| 768 | 6139.5 tok/s | 8.0 tok/s | 6137.5 to 6154.6 | 816 ms |
| 896 | 6532.1 tok/s | 7.3 tok/s | 6531.7 to 6548.9 | 959 ms |
| 1024 | 7094.3 tok/s | 6.9 tok/s | 6902.4 to 7101.6 | 1058 ms |

| | |
| --- | --- |
| Label | rising. The highest value is at 1024, the top of the sweep, so 7094.3 tok/s is a floor rather than a ceiling. Finding the peak needs `max_num_seqs` above 1024, a different serving configuration |
| Quote for one caller | 30.2 tok/s |
| Quote for a shared endpoint | 7094.3 tok/s at concurrency 1024 |
| KV cache | 1,555,488 tokens, 5.93 full-length requests at once. The pool is the same at 32768, where it holds 47.47, so the context costs nothing and only divides the same pool differently |
| Long prompt | 247,704 tokens in 27 s cold; 1.6 s when the prefix is already cached |

Reproduce:

```
KEY_NAME=Kimi-K2.7-Code-h200-4-nodes2 bash common/tools/bench.sh --host <head_node> --model kimi-k2.7-code
KEY_NAME=Kimi-K2.7-Code-h200-4-nodes2 bash common/tools/bench.sh --host <head_node> --model kimi-k2.7-code \
  --sweep 1,8,32,64,128,256,512,640,768,896,1024
```

- `KEY_NAME` is required once `secrets/` holds more than one key, otherwise `bench.sh` resolves the shared
  key and every request returns 401.

## Known limits

- Do not set `EXTRA_ARGS` without repeating `--skip-mm-profiling --mm-processor-cache-gb 0`. It replaces the
  default rather than adding to it, and without those flags a multimodal checkpoint on more than one node
  completes weight loading and then hangs at multimodal profiling with no error and no timeout.
- No speculative decoding. vLLM rejects a speculative config whenever pipeline parallelism is active, and
  554.3 GiB of weights needs two nodes.
- Keep tensor parallelism inside a node. Tensor parallelism spanning two nodes hangs at NCCL initialization.
- Leave `VLLM_USE_DEEP_GEMM` at 0, which `env/env.sh` sets. The DeepGEMM MoE path takes an illegal memory
  access on H200 for two other checkpoints on this hardware.
- Anthropic's hosted tools return HTTP 400. Use [docs/web-search.md](../../../docs/web-search.md).
