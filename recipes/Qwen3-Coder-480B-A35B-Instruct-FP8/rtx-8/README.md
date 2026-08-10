# Qwen3-Coder-480B-A35B-Instruct-FP8 on one RTX PRO 6000 node

Status: Validated - vLLM 0.25.1, protocol: slope(128,1152) at concurrency 1, 8, 32, 64, 128, 256, 512, 640, 768, 896 and 1024

| | |
| --- | --- |
| Served name | `qwen3-coder-480b` |
| Checkpoint | `Qwen3-Coder-480B-A35B-Instruct-FP8`, Hugging Face `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` |
| On disk | 449.04 GiB across 49 shards, FP8 |
| Served precision | FP8, `weight_block_size` [128, 128], 57.26 GiB of weights per GPU |
| Context | 262144, the checkpoint maximum, native with no rope scaling |
| Hardware | 1 RTX PRO 6000 Blackwell node, 8 GPUs, 97887 MiB each, TP4 x PP2 over PCIe with no NVLink |
| Engine | vLLM 0.25.1, CUDA graphs, prefix caching, no speculative decoding |

## 1. Create the API key

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/Qwen3-Coder-480B-A35B-Instruct-FP8-rtx-8.key
chmod 600 secrets/Qwen3-Coder-480B-A35B-Instruct-FP8-rtx-8.key
```

- `secrets/vllm_api_key` is read when this file is absent.
- To rotate, replace the file and relaunch; the engine reads it once at launch.

## 2. Build the environment

Run on a compute node, not a login node. Needs `uv` and `mamba` on your PATH.

```
module load Mambaforge
bash recipes/Qwen3-Coder-480B-A35B-Instruct-FP8/rtx-8/env/build.sh
```

- About 9.0 GB for the venv, plus a conda CUDA 13.0 toolkit alongside it under `ENV_ROOT`, default scratch.
- The toolkit is required: the FlashInfer sm_120 JIT needs a complete CUDA 13 install, which the node's
  runtime-only `/usr/local/cuda-13` and the fragmented pip nvcc wheels do not provide.
- FlashInfer is pinned to 0.6.15. vLLM 0.25.1 pins 0.6.13, which rejects the `kv_scale_format` argument the
  sm_120 backend passes and fails at the first request. No matching cubin package exists at 0.6.15, so
  `env/env.sh` bypasses the version check and kernels compile from source on the first launch.
- The vLLM wheel comes from the nightly cu130 index rather than PyPI, because sm_120 needs the CUDA 13
  build. The version is pinned explicitly, since the installed metadata carries no local version tag and an
  unpinned install drifts silently.

## 3. Launch

Slurm, from the repo root:

```
sbatch --account=<your-account> recipes/Qwen3-Coder-480B-A35B-Instruct-FP8/rtx-8/serve.sbatch
squeue --me                                  # NODELIST gives the host
tail -f qwen3-coder-<jobid>.log
```

Direct, on a node you already hold. This recipe uses all eight GPUs, so there is no device to pin:

```
bash recipes/Qwen3-Coder-480B-A35B-Instruct-FP8/rtx-8/serve_ssh.sh <node>
```

| Stage | Measured |
| --- | --- |
| Launch to serving | 7 min 49 s first time on a node, 3 min 08 s with the checkpoint in page cache |
| Weight load, 49 shards across 8 ranks | 174.77 s cold, 46.77 s warm |

- The first launch on any node compiles the sm_120 attention kernels from source, because no matching
  FlashInfer cubin package exists. That cache is node-local, so a different node pays it again.
- On the direct path the server log is node-local at `/tmp/$USER/vllm/`. Under Slurm it lands in the submit
  directory.

## 4. Verify

```
KEY=$(cat secrets/Qwen3-Coder-480B-A35B-Instruct-FP8-rtx-8.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the node serving it>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"qwen3-coder-480b","messages":[{"role":"user","content":"What is 2+2? Answer briefly."}],"max_tokens":400}'
```

| Command | Expected |
| --- | --- |
| `/v1/models` with the key | `"id": "qwen3-coder-480b"`, `"max_model_len": 262144` |
| `/v1/models` without a key | `401` |
| chat completion | `content` holds the answer, `finish_reason: stop`, 6 output tokens |

- This is not a thinking model, so no reasoning parser is passed and the answer arrives in `content`
  directly.

## 5. Connect a client

```
export NODE=<the node serving it>
source recipes/Qwen3-Coder-480B-A35B-Instruct-FP8/rtx-8/client.env
claude
```

- `client.env` sets `CLAUDE_CODE_MAX_CONTEXT_TOKENS=262144`. Claude Code assumes a 200k window for a served
  name it does not recognize, well under what this endpoint serves, so without the pin most of the window
  goes unused.
- Use `ANTHROPIC_AUTH_TOKEN`. `ANTHROPIC_API_KEY` sends `x-api-key` and returns 401.
- `client.env` also sets `ANTHROPIC_SMALL_FAST_MODEL`; without it the client reaches for a hosted Haiku.
- OpenAI clients such as Codex: base URL `http://<node>:8000/v1`, same key, model `qwen3-coder-480b`. See
  [docs/clients.md](../../../docs/clients.md).

## 6. Stop it

```
scancel <jobid>                        # Slurm path
bash common/tools/stop.sh <node>       # direct path
```

## Tunable inputs

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/Qwen3-Coder-480B-A35B-Instruct-FP8` | Serve a different copy of the checkpoint |
| `API_PORT` | 8000 | Listening port |
| `MAX_MODEL_LEN` | 262144 | Context window, and the checkpoint maximum |
| `GPU_UTIL` | 0.90 | Fraction of VRAM for weights plus KV cache |
| `TP` | 4 | Tensor parallel size. 4 is mandatory for this checkpoint; see Known limits |
| `PP` | 2 | Pipeline parallel size, which puts the node's other four GPUs to work |
| `EXECUTOR` | `mp` | Distributed executor backend; one node needs no Ray |
| `TOOL_PARSER` | `qwen3_coder` | Tool call parser |
| `REASONING_PARSER` | unset | No reasoning parser. Set one only if you want plain output parsed as reasoning |
| `EXTRA_ARGS` | unset | Extra flags appended to the `vllm serve` command line |
| `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` | 3600 | Raised from 480 so a storage stall that resolves is survivable |
| `LOG_DIR` | `/tmp/$USER/vllm` | Where the SSH path writes the server log |
| `VENV_DIR` | under `ENV_ROOT` | Use an environment built elsewhere |
| `CUDA13_DIR` | under `ENV_ROOT` | Use a CUDA 13.0 toolkit built elsewhere |
| `VLLM_VERSION` | 0.25.1 | Engine version `env/build.sh` installs |
| `FLASHINFER_VERSION` | 0.6.15 | FlashInfer version `env/build.sh` installs |
| `VLLM_CACHE_ROOT` | under `ENV_ROOT` | Where vLLM keeps compiled artifacts; the engine default is a small quota here |
| `KEY_NAME`, `KEY_FILE`, `VLLM_API_KEY` | this recipe's key | Which key `common/lib/api_key.sh` resolves; an exported `VLLM_API_KEY` wins |
| `NODE`, `QWEN3_CODER_NODE` | unset | The node to launch on or connect to; set either, or pass the node as an argument |
| `MODELS_DIR`, `ENV_ROOT` | `common/defaults.sh` | Override there or in `common/site.conf` |

## Benchmarking

Conditions:

| | |
| --- | --- |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| Input length | ISL 17 tokens. Rates at a long input are not measured; use `--prompt-tokens` |
| Output length | OSL 1152 tokens, output only |
| Context | `MAX_MODEL_LEN=262144` |
| Allocation for the measurement | 8 GPUs, 128 cores, 1440 GB, `kempner_rtx` |
| Sequence cap | `max_num_seqs` 1024, the engine default, which equals the top sweep level |
| Preemption | 4,990 across the sweep. The pool is 783,232 tokens while concurrency 1024 at this output length needs about 1.2 million, so the KV cache binds before the sequence cap. The counter is on `/metrics`; the server log does not report preemption |
| Endpoint | idle, and the benchmark client ran on a separate CPU-only node |
| Power | 600 W enforced, the card default, so not capped. Median 190 W across 15392 samples and a 286 W peak, with nothing at or above 590, clocks 2422 MHz |

Results:

| Concurrency | Aggregate | Per stream | Spread over 3 runs | TTFT median |
| --- | --- | --- | --- | --- |
| 1 | 68.0 tok/s | 68.0 tok/s | 68.0 to 68.0 | 44 ms |
| 8 | 285.7 tok/s | 35.7 tok/s | 266.3 to 290.4 | 57 ms |
| 32 | 667.8 tok/s | 20.9 tok/s | 524.8 to 686.7 | 86 ms |
| 64 | 987.8 tok/s | 15.4 tok/s | 981.1 to 1022.4 | 114 ms |
| 128 | 1587.3 tok/s | 12.4 tok/s | 1581.0 to 1594.3 | 141 ms |
| 256 | 2306.0 tok/s | 9.0 tok/s | 2305.4 to 2331.5 | 228 ms |
| 512 | 3027.2 tok/s | 5.9 tok/s | 3026.8 to 3035.8 | 339 ms |
| 640 | 3197.0 tok/s | 5.0 tok/s | 3181.8 to 3199.2 | 347 ms |
| 768 | 3186.9 tok/s | 4.1 tok/s | 3179.1 to 3187.7 | 457 ms |
| 896 | 3114.8 tok/s | 3.5 tok/s | 3107.5 to 3138.8 | 500 ms |
| 1024 | 3035.6 tok/s | 3.0 tok/s | 3026.0 to 3071.6 | 570 ms |

| | |
| --- | --- |
| Label | peak. The highest value is at concurrency 640 and the curve turns over above it, so 3197.0 tok/s is a measured ceiling rather than the edge of the sweep |
| Quote for one caller | 68.0 tok/s |
| Quote for a shared endpoint | 3197.0 tok/s at concurrency 640 |
| KV cache | 783,232 tokens from 24.98 GiB per GPU, 2.99 full-length requests at once |
| What the full context costs | nothing. The pool does not move with the window: 783,232 tokens at 131072 and 783,232 to 783,360 at 262144 across launches, so doubling the context only halves how many full-length requests fit, from 5.98 to 2.99 |
| Long prompt | 251,916 tokens in 27.6 s cold; 0.9 s when the prefix is already cached |

Reproduce:

```
KEY_NAME=Qwen3-Coder-480B-A35B-Instruct-FP8-rtx-8 bash common/tools/bench.sh --host <node> \
  --model qwen3-coder-480b --sweep 1,8,32,64,128,256,512,640,768,896,1024
```

- `KEY_NAME` is required once `secrets/` holds more than one key, otherwise `bench.sh` resolves the shared
  key and every request returns 401.

## Known limits

- **This FP8 checkpoint cannot run at TP8.** `moe_intermediate_size` is 2560 and the FP8 quantization block
  is 128, so a TP8 shard is 2560/8 = 320, which is not a multiple of 128, and vLLM refuses to start with
  `output_size of gate's and up's weight = 320 is not divisible by weight quantization block_n = 128`. TP4
  gives 640, so on an 8-GPU node the working shape is TP4 with PP2.
- No speculative decoding. vLLM rejects a speculative config whenever pipeline parallelism is active, and
  PP2 is how the other four GPUs are used. The guard is not visible in the 0.25.1 config source, so treat it
  as behavior for this version and re-check after an engine upgrade.
- **Do not move this checkpoint to H200.** Four configurations were tried and all failed at CUDA graph
  capture: memory utilization 0.90 and 0.96 both faulted, `VLLM_USE_DEEP_GEMM=1` gave an illegal memory
  access, and the Triton path gave `cutlass_gemm_caller ... Error Internal`. Memory was not the constraint,
  since the two-node runs had 65 GiB of KV per GPU. The CUTLASS w8a8 FP8 GEMM path faults on Hopper for this
  checkpoint during capture. Eager works but costs roughly 3x, at 22.2 tok/s.
- Do not add the conda CUDA 13 toolkit to `LD_LIBRARY_PATH`. Its `libcudart` shadows torch's runtime and
  pulls in a missing `libcupti.so.13`, which breaks import; the recipe exposes it through `CPATH` and
  `LIBRARY_PATH` for compilation only.
- No NVLink on these nodes, so `env/env.sh` sets `NCCL_P2P_DISABLE=1`. Without it NCCL initialization hangs
  with no error and the server never becomes ready.
- Do not enable `--enable-expert-parallel`. All-reduce already crosses PCIe rather than NVLink, and the
  added all-to-all traffic measured about 9 percent slower on this hardware.
- Anthropic's hosted tools return HTTP 400. Use [docs/web-search.md](../../../docs/web-search.md).
