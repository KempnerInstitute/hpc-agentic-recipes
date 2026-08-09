# DeepSeek-V4-Pro on two RTX PRO 6000 nodes

Status: Validated - vLLM 0.26.0, protocol: slope(128,1152) at concurrency 1, 8, 32, 64, 128, 256, 512, 640, 768, 896 and 1024

| | |
| --- | --- |
| Served name | `deepseek-v4-pro` |
| Checkpoint | `DeepSeek-V4-Pro`, Hugging Face `deepseek-ai/DeepSeek-V4-Pro` |
| On disk | 805.4 GiB across 64 shards, FP8 with FP4 experts |
| Served precision | FP8, `weight_block_size` [128, 128], 50.95 and 52.17 GiB of weights per GPU by stage |
| Context | 1048576, the checkpoint maximum, reached by YaRN scaling factor 16 over a native 65536 |
| Hardware | 2 RTX PRO 6000 Blackwell nodes, 16 GPUs, 97887 MiB each, TP8 inside a node and PP2 between over Ray |
| Engine | vLLM 0.26.0, eager, no speculative decoding |

## 1. Create the API key

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/DeepSeek-V4-Pro-rtx-8-nodes2.key
chmod 600 secrets/DeepSeek-V4-Pro-rtx-8-nodes2.key
```

- `secrets/vllm_api_key` is read when this file is absent.
- To rotate, replace the file and relaunch; the engine reads it once at launch.

## 2. Build the environment

Run on a compute node, not a login node. Needs `uv` and `mamba` on your PATH. Build once: `ENV_ROOT` is
shared scratch, so both nodes activate the same environment.

```
module load Mambaforge
bash recipes/DeepSeek-V4-Pro/rtx-8-nodes2/env/build.sh
```

- About 11 GB under `ENV_ROOT`, default scratch, plus a conda CUDA 13.0 toolkit alongside it.
- The toolkit is required: the FlashInfer sm_120 JIT needs a complete CUDA 13 install, which the node's
  runtime-only `/usr/local/cuda-13` and the fragmented pip nvcc wheels do not provide.
- `ENV_ROOT` and `MODELS_DIR` must resolve to the same content on both nodes. The Ray worker imports vLLM
  and reads weights itself, so a path that exists only on the head fails after the cluster has formed.

## 3. Launch

Slurm, from the repo root. The batch script takes both nodes, starts Ray, then runs the engine on the head:

```
sbatch --account=<your-account> recipes/DeepSeek-V4-Pro/rtx-8-nodes2/serve.sbatch
squeue --me                                  # NODELIST, the FIRST name serves
tail -f dsv4-rtx-<jobid>.log
```

Direct, on two nodes you already hold:

```
bash recipes/DeepSeek-V4-Pro/rtx-8-nodes2/serve_ssh.sh <head_node> <worker_node>
```

| Stage | Measured |
| --- | --- |
| Launcher to serving | 58 min cold, 4 to 17 min with the checkpoint in page cache |
| Weight load, 64 shards across 16 ranks | 2438 s and 3080 s by stage when cold |

- The 58 minute figure is a cold read of 805 GiB from Lustre onto nodes whose page cache held another
  model. Staging the checkpoint on scratch, or relaunching on the same nodes, is far faster.
- Shard loading is not steady: it ran at about 90 seconds per shard early and 11 seconds per shard later in
  the same load, so a slow start is not a stall.
- The endpoint is on the head node only. The worker holds half the layers and refuses HTTP.
- On the direct path the server log is node-local at `/tmp/$USER/vllm/`. Under Slurm it lands in the submit
  directory.

## 4. Verify

```
KEY=$(cat secrets/DeepSeek-V4-Pro-rtx-8-nodes2.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the head node>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"deepseek-v4-pro","messages":[{"role":"user","content":"What is 2+2? Answer briefly."}],"max_tokens":400}'
```

| Command | Expected |
| --- | --- |
| `/v1/models` with the key | `"id": "deepseek-v4-pro"`, `"max_model_len": 1048576` |
| `/v1/models` without a key | `401` |
| chat completion | `content` holds the answer, `finish_reason: stop`, 2 output tokens |

- 400 output tokens is ample: three smoke tests each used 2.

## 5. Connect a client

```
export NODE=<the head node>
source recipes/DeepSeek-V4-Pro/rtx-8-nodes2/client.env
claude
```

- `client.env` sets `CLAUDE_CODE_MAX_CONTEXT_TOKENS=1048576`. Claude Code assumes and enforces a 200k window
  for a served name it does not recognize, far below what this endpoint serves.
- Use `ANTHROPIC_AUTH_TOKEN`. `ANTHROPIC_API_KEY` sends `x-api-key` and returns 401.
- `client.env` also sets `ANTHROPIC_SMALL_FAST_MODEL`; without it the client reaches for a hosted Haiku.
- OpenAI clients such as Codex: base URL `http://<head-node>:8000/v1`, same key, model `deepseek-v4-pro`.
  See [docs/clients.md](../../../docs/clients.md).

## 6. Stop it

```
scancel <jobid>                                        # Slurm path, owns both nodes
bash common/tools/stop.sh <head_node> <worker_node>    # direct path, name both
```

- The direct path has no scheduler to clean up, so both nodes must be named. `stop.sh` stops Ray as well as
  the server.
- Both nodes must be clean before a relaunch. A surviving Ray cluster is the most common cause of a launch
  that never becomes ready, and it fails on resources rather than memory.

## Tunable inputs

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/DeepSeek-V4-Pro` | Serve a different copy of the checkpoint |
| `API_PORT` | 8000 | Listening port |
| `MAX_MODEL_LEN` | 1048576 | Context window, the checkpoint maximum. Set 131072 to trade the window back for about 16 percent more aggregate throughput; see Benchmarking |
| `GPU_UTIL` | 0.90 | Fraction of VRAM for weights plus KV cache |
| `TP` | 8 | Tensor parallel size; 8 is one node's GPU count and must stay inside a node |
| `PP` | 2 | Pipeline parallel size; 2 is the node count |
| `PERF` | unset | Try CUDA graph capture instead of eager; the rates below are eager |
| `CUDAGRAPH_MODE` | `NONE` | Graph mode passed through when `PERF` is set |
| `EXTRA_ARGS` | unset | Extra flags appended to the `vllm serve` command line |
| `TOOL_PARSER` | `deepseek_v4` | Tool call parser |
| `REASONING_PARSER` | `deepseek_v4` | Reasoning parser |
| `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME` | `ib0` | Pin cross-node transport to InfiniBand |
| `NCCL_DEBUG` | `WARN` | NCCL log level; raise to `INFO` to debug a cross-node hang |
| `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` | 3600 | Raised from 480 so a storage stall that resolves is survivable |
| `CUDA13_DIR`, `CUDA_HOME` | under `ENV_ROOT` | The conda CUDA 13 toolkit the sm_120 JIT needs |
| `FLASHINFER_VERSION` | pinned in `env/build.sh` | FlashInfer build installed |
| `VLLM_VERSION` | 0.26.0 | Engine version `env/build.sh` installs |
| `LOG_DIR` | `/tmp/$USER/vllm` | Where the SSH path writes the server log |
| `VENV_DIR` | under `ENV_ROOT` | Use an environment built elsewhere |
| `VLLM_CACHE_ROOT` | under `ENV_ROOT` | Where vLLM keeps compiled artifacts; the engine default is a small quota here |
| `KEY_NAME`, `KEY_FILE`, `VLLM_API_KEY` | this recipe's key | Which key `common/lib/api_key.sh` resolves |
| `NODE`, `DSV4_HEAD`, `DSV4_WORKER` | unset | Nodes for the SSH path; pass them as arguments instead |
| `ACCOUNT` | unset | Read by `common/defaults.sh`; pass `--account` at submit time |
| `MODELS_DIR`, `ENV_ROOT` | `common/defaults.sh` | Override there or in `common/site.conf` |

## Benchmarking

Conditions:

| | |
| --- | --- |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| Input length | ISL 10 tokens. Rates at a long input are not measured; use `--prompt-tokens` |
| Output length | OSL 1152 tokens, output only |
| Context | `MAX_MODEL_LEN=1048576` |
| Allocation for the measurement | 2 nodes, 8 GPUs each, 128 cores and 1440 GB per node, `kempner_rtx` |
| Sequence cap | `max_num_seqs` 1024, the engine default, which equals the top sweep level |
| Preemption | none at any level. The pool is 4.18 million tokens, far more than this sweep needs |
| Endpoint | idle, and the benchmark client ran on a separate CPU-only node |
| Power | 600 W enforced, the card default, so not capped. Median 177 and 181 W on the two nodes and a 220 W peak, with nothing at or above 590 |

Results:

| Concurrency | Aggregate | Per stream | Spread over 3 runs | TTFT median |
| --- | --- | --- | --- | --- |
| 1 | 18.7 tok/s | 18.7 tok/s | 18.7 to 18.7 | 163 ms |
| 8 | 197.6 tok/s | 24.7 tok/s | 144.4 to 208.2 | 177 ms |
| 32 | 529.8 tok/s | 16.6 tok/s | 527.9 to 549.1 | 215 ms |
| 64 | 879.6 tok/s | 13.7 tok/s | 876.6 to 993.8 | 1439 ms |
| 128 | 1187.5 tok/s | 9.3 tok/s | 1156.7 to 1221.4 | 5991 ms |
| 256 | 1887.9 tok/s | 7.4 tok/s | 1874.7 to 1918.6 | 623 ms |
| 512 | 2515.4 tok/s | 4.9 tok/s | 2468.9 to 2565.4 | 1024 ms |
| 640 | 2682.8 tok/s | 4.2 tok/s | 2494.9 to 2697.2 | 1312 ms |
| 768 | 2779.8 tok/s | 3.6 tok/s | 2775.2 to 2794.6 | 1604 ms |
| 896 | 2902.0 tok/s | 3.2 tok/s | 2892.5 to 2938.1 | 2292 ms |
| 1024 | 3002.5 tok/s | 2.9 tok/s | 2992.3 to 3003.1 | 1995 ms |

| | |
| --- | --- |
| Label | rising. The highest value is at 1024, the top of the sweep, so 3002.5 tok/s is a floor rather than a ceiling. Finding the peak needs `max_num_seqs` above 1024, a different serving configuration |
| Quote for one caller | 18.7 tok/s |
| Quote for a shared endpoint | 3002.5 tok/s at concurrency 1024 |
| KV cache | 4,183,279 and 4,227,729 tokens by pipeline stage from 28.6 and 29.35 GiB per GPU, 3.99 full-length requests at once |
| What the context costs | measured against a control at 131072 on the same nodes: 2923.2 at concurrency 512 and 3599.1 at 1024, so the full window costs 14 to 17 percent of aggregate throughput above concurrency 256. Single stream is unaffected, 18.7 against 18.6. Set `MAX_MODEL_LEN=131072` to take that throughput back |
| Why the pool grows with the window | per-token KV falls from 37.2 KiB at 131072 to 7.2 KiB at 1048576, so the pool goes from 807,226 tokens to 4,183,279. This model's attention is hybrid, and KV cost is not linear in context |
| Long prompt | 1,023,874 tokens in 206 s cold; 5.3 s when the prefix is already cached |
| Spread at low concurrency | concurrency 8 measured 144.4 to 208.2, and 64 measured 876.6 to 993.8. The upper levels are tight by comparison |

Reproduce:

```
KEY_NAME=DeepSeek-V4-Pro-rtx-8-nodes2 bash common/tools/bench.sh --host <head_node> --model deepseek-v4-pro \
  --sweep 1,8,32,64,128,256,512,640,768,896,1024
```

- `KEY_NAME` is required once `secrets/` holds more than one key, otherwise `bench.sh` resolves the shared
  key and every request returns 401.

## Known limits

- The full context is reached by YaRN scaling factor 16 over a native 65536, so a million tokens is what the
  hardware serves rather than what the model was trained on. Quality at extreme lengths is not measured here.
- The context is not free on this recipe: the full window costs 14 to 17 percent of aggregate throughput
  above concurrency 256, though nothing at concurrency 1.
- No speculative decoding. vLLM rejects a speculative config whenever pipeline parallelism is active, and
  805 GiB of weights needs two nodes.
- Keep tensor parallelism inside a node. Tensor parallelism spanning two nodes hangs at NCCL initialization.
- No NVLink on these nodes, so `env/env.sh` sets `NCCL_P2P_DISABLE=1`, and all-reduce crosses PCIe. Do not
  enable `--enable-expert-parallel`, which adds all-to-all traffic and measured slower on this hardware.
- Do not add the conda CUDA 13 toolkit to `LD_LIBRARY_PATH`. Its `libcudart` shadows torch's runtime and
  pulls in a missing `libcupti.so.13`, which breaks import.
- Anthropic's hosted tools return HTTP 400. Use [docs/web-search.md](../../../docs/web-search.md).
