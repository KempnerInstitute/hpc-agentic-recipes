# Kimi-K2.7-Code on one RTX PRO 6000 node

Status: Validated - vLLM 0.25.1, protocol: slope(128,1152) at concurrency 1, 8, 32, 64, 128, 256, 512, 640, 768, 896 and 1024

| | |
| --- | --- |
| Served name | `kimi-k2.7-code` |
| Checkpoint | `Kimi-K2.7-Code`, Hugging Face `moonshotai/Kimi-K2.7-Code` |
| On disk | 554.3 GiB across 64 shards, INT4, `KimiK25ForConditionalGeneration`, multimodal |
| Served precision | INT4 pack-quantized at group size 32, 72.03 GiB of weights per GPU |
| Context | 131072. The checkpoint declares 262144, which does not fit; see Known limits |
| Hardware | 1 RTX PRO 6000 Blackwell node, 8 GPUs, 97887 MiB each, TP8 over PCIe with no NVLink |
| Engine | vLLM 0.25.1, eager, no speculative decoding |

## 1. Create the API key

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/Kimi-K2.7-Code-rtx-8.key
chmod 600 secrets/Kimi-K2.7-Code-rtx-8.key
```

- `secrets/vllm_api_key` is read when this file is absent.
- To rotate, replace the file and relaunch; the engine reads it once at launch.

## 2. Build the environment

Run on a compute node, not a login node. Needs `uv` and `mamba` on your PATH.

```
module load Mambaforge
bash recipes/Kimi-K2.7-Code/rtx-8/env/build.sh
```

- About 9.2 GB for the venv, plus a conda CUDA 13.0 toolkit alongside it under `ENV_ROOT`, default scratch.
- The toolkit is required: the FlashInfer sm_120 JIT needs a complete CUDA 13 install, which the node's
  runtime-only `/usr/local/cuda-13` and the fragmented pip nvcc wheels do not provide.
- FlashInfer is pinned to 0.6.15. vLLM 0.25.1 pins 0.6.13, which rejects the `kv_scale_format` argument the
  sm_120 backend passes and fails at the first request. No matching cubin package exists at 0.6.15, so
  `env/env.sh` bypasses the version check and kernels compile from source on the first launch.

## 3. Launch

Slurm, from the repo root:

```
sbatch --account=<your-account> recipes/Kimi-K2.7-Code/rtx-8/serve.sbatch
squeue --me                                  # NODELIST gives the host
tail -f kimi-rtx-<jobid>.log
```

Direct, on a node you already hold. This recipe uses all eight GPUs, so there is no device to pin:

```
bash recipes/Kimi-K2.7-Code/rtx-8/serve_ssh.sh <node>
```

| Stage | Measured |
| --- | --- |
| Launch to serving, first time on a node | 19 min 55 s |
| Launch to serving, caches warm | 7 min |
| Weight load, 64 shards across 8 ranks | 989 s |

- The first launch on any node compiles the sm_120 attention kernels from source, because no matching
  FlashInfer cubin package exists. That cache is node-local, so a different node pays it again.
- On the direct path the server log is node-local at `/tmp/$USER/vllm/`. Under Slurm it lands in the submit
  directory.

## 4. Verify

```
KEY=$(cat secrets/Kimi-K2.7-Code-rtx-8.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the node serving it>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"kimi-k2.7-code","messages":[{"role":"user","content":"What is 2+2? Answer briefly."}],"max_tokens":400}'
```

| Command | Expected |
| --- | --- |
| `/v1/models` with the key | `"id": "kimi-k2.7-code"`, `"max_model_len": 131072` |
| `/v1/models` without a key | `401` |
| chat completion | `content` holds the answer, `finish_reason: stop` |

- 400 output tokens is enough: three smoke tests used 31 to 37 tokens.
- Image input is untested. This recipe is validated for text.

## 5. Connect a client

```
export NODE=<the node serving it>
source recipes/Kimi-K2.7-Code/rtx-8/client.env
claude
```

- `client.env` sets `CLAUDE_CODE_MAX_CONTEXT_TOKENS=131072`. Claude Code assumes a 200k window for a served
  name it does not recognize, which is **above** what this endpoint serves, so without the pin it overflows
  the context and requests fail with `maximum context length is 131072` rather than compacting.
- Use `ANTHROPIC_AUTH_TOKEN`. `ANTHROPIC_API_KEY` sends `x-api-key` and returns 401.
- `client.env` also sets `ANTHROPIC_SMALL_FAST_MODEL`; without it the client reaches for a hosted Haiku.
- OpenAI clients such as Codex: base URL `http://<node>:8000/v1`, same key, model `kimi-k2.7-code`. See
  [docs/clients.md](../../../docs/clients.md).

## 6. Stop it

```
scancel <jobid>                        # Slurm path
bash common/tools/stop.sh <node>       # direct path
```

## Tunable inputs

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/Kimi-K2.7-Code` | Serve a different copy of the checkpoint |
| `API_PORT` | 8000 | Listening port |
| `MAX_MODEL_LEN` | 131072 | Context window. The KV pool holds 137,664 tokens, so this is 5 percent below the hardware ceiling |
| `GPU_UTIL` | 0.90 | Fraction of VRAM for weights plus KV cache |
| `TP` | 8 | Tensor parallel size, and the node's GPU count |
| `CUDA_GRAPHS` | unset, eager | Set to serve with CUDA graphs, which needs `GPU_UTIL=0.93`. Trades aggregate for single stream; see Choosing eager or CUDA graphs |
| `CUDAGRAPH_MODE` | `FULL_AND_PIECEWISE` | Graph mode, read only when `CUDA_GRAPHS` is set |
| `EXTRA_ARGS` | unset | Extra flags appended to the `vllm serve` command line. Unlike the two-node variant, this one appends rather than replaces |
| `TOOL_PARSER` | `kimi_k2` | Tool call parser |
| `REASONING_PARSER` | `kimi_k2` | Reasoning parser |
| `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` | 3600 | Raised from 480 so a storage stall that resolves is survivable |
| `LOG_DIR` | `/tmp/$USER/vllm` | Where the SSH path writes the server log |
| `VENV_DIR` | under `ENV_ROOT` | Use an environment built elsewhere |
| `CUDA13_DIR` | under `ENV_ROOT` | Use a CUDA 13.0 toolkit built elsewhere |
| `VLLM_VERSION` | 0.25.1 | Engine version `env/build.sh` installs |
| `FLASHINFER_VERSION` | 0.6.15 | FlashInfer version `env/build.sh` installs |
| `TRANSFORMERS_VERSION` | 5.14.1 | The transformers version the rates were measured with |
| `VLLM_CACHE_ROOT` | under `ENV_ROOT` | Where vLLM keeps compiled artifacts; the engine default is a small quota here |
| `KEY_NAME`, `KEY_FILE`, `VLLM_API_KEY` | this recipe's key | Which key `common/lib/api_key.sh` resolves; an exported `VLLM_API_KEY` wins |
| `NODE`, `KIMI_RTX_NODE` | unset | The node to launch on or connect to; set either, or pass the node as an argument |
| `MODELS_DIR`, `ENV_ROOT` | `common/defaults.sh` | Override there or in `common/site.conf` |

## Benchmarking

Conditions:

| | |
| --- | --- |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| Input length | ISL 21 tokens. Rates at a long input are not measured; use `--prompt-tokens` |
| Output length | OSL 1152 tokens, output only, `ignore_eos` |
| Context | `MAX_MODEL_LEN=131072` |
| Allocation for the measurement | 8 GPUs, 128 cores, 1440 GB, `kempner_rtx` |
| Sequence cap | `max_num_seqs` 1024, the engine default, which equals the top sweep level |
| Preemption | 28,600 across the sweep. The pool is 137,664 tokens and concurrency 1024 at this output length needs far more, so the KV cache is what binds |
| Endpoint | idle, and the benchmark client ran on a separate CPU-only node |
| Power | 600 W enforced, the card default, so not capped. Median 272 W across 18200 samples and a 307 W peak, with nothing at or above 590 |

Results:

| Concurrency | Aggregate | Per stream | Spread over 3 runs | TTFT median |
| --- | --- | --- | --- | --- |
| 1 | 20.6 tok/s | 20.6 tok/s | 20.5 to 20.6 | 141 ms |
| 8 | 164.0 tok/s | 20.5 tok/s | 163.5 to 164.7 | 156 ms |
| 32 | 643.7 tok/s | 20.1 tok/s | 642.6 to 644.3 | 213 ms |
| 64 | 1101.7 tok/s | 17.2 tok/s | 1095.0 to 1104.0 | 307 ms |
| 128 | 1330.1 tok/s | 10.4 tok/s | 1329.8 to 1332.0 | 345 ms |
| 256 | 1633.8 tok/s | 6.4 tok/s | 1629.4 to 1638.7 | 597 ms |
| 512 | 1819.2 tok/s | 3.6 tok/s | 1818.7 to 1822.7 | 1029 ms |
| 640 | 1773.1 tok/s | 2.8 tok/s | 1766.0 to 1774.8 | 1294 ms |
| 768 | 1786.3 tok/s | 2.3 tok/s | 1781.8 to 1787.4 | 1458 ms |
| 896 | 1832.6 tok/s | 2.0 tok/s | 1831.2 to 1842.5 | 1652 ms |
| 1024 | 1817.9 tok/s | 1.8 tok/s | 1809.8 to 1821.8 | 1763 ms |

| | |
| --- | --- |
| Label | saturated. It varies 3.25 percent across 512 to 1024, under the 4 percent the rule uses, and the highest value is at 896 |
| Quote for one caller | 20.6 tok/s |
| Quote for a shared endpoint | 1832.6 tok/s at concurrency 896 |
| KV cache | 137,664 tokens from 9.01 GiB per GPU, 1.05 full-length requests at once |
| Long prompt | 123,863 tokens in 33 s cold; 1.4 s when the prefix is already cached |
| Against the two-node H200 variant | the same weights leave 9.01 GiB of KV per GPU here against 47.76 there, so the pool is 137,664 tokens against 1,466,752, which is why that variant serves 262144 and this one cannot |

Reproduce:

```
KEY_NAME=Kimi-K2.7-Code-rtx-8 bash common/tools/bench.sh --host <node> --model kimi-k2.7-code
KEY_NAME=Kimi-K2.7-Code-rtx-8 bash common/tools/bench.sh --host <node> --model kimi-k2.7-code \
  --sweep 1,8,32,64,128,256,512,640,768,896,1024
```

- `KEY_NAME` is required once `secrets/` holds more than one key, otherwise `bench.sh` resolves the shared
  key and every request returns 401.

## Choosing eager or CUDA graphs

Eager is the default and the table above is measured with it. CUDA graphs triple the single-stream rate and
cost about 40 percent of the aggregate, so the right choice depends on how the endpoint is used. Both were
measured on one node at 131072 context, concurrency 1, 8, 256 and 1024, 3 repeats.

```
CUDA_GRAPHS=1 GPU_UTIL=0.93 sbatch --account=<your-account> recipes/Kimi-K2.7-Code/rtx-8/serve.sbatch
```

| | Eager | CUDA graphs |
| --- | --- | --- |
| One caller | 20.8 tok/s | 64.6 tok/s |
| Concurrency 8 | 162.3 tok/s | 289.1 tok/s |
| Concurrency 256 | 1821.0 tok/s | 1021.0 tok/s |
| Concurrency 1024 | 2007.3 tok/s | 1215.5 tok/s |
| KV cache | 181,200 tokens | 152,608 tokens |

- `GPU_UTIL=0.93` is required. Graph capture takes 2.13 GiB per GPU, and at 0.90 the pool falls to 7.14 GiB
  against the 8.58 GiB one full-length request needs, so the engine refuses to start.
- Both figures in this table were measured at `GPU_UTIL=0.93`, which is also why the eager column is above the
  0.90 numbers in the table further up.
- Answers were checked on five prompts with known answers before and after the sweep, and matched in both.

## Known limits

- The checkpoint declares 262144 and the engine refuses it: one request that long needs 17.16 GiB of KV
  against 9.01 GiB available, and vLLM reports an estimated maximum of 137664. That is the pool itself, so
  serving it would leave exactly one full-length request and no headroom; 131072 keeps 1.05.
- Do not add the conda CUDA 13 toolkit to `LD_LIBRARY_PATH`. Its `libcudart` shadows torch's runtime and
  pulls in a missing `libcupti.so.13`, which breaks import; the recipe exposes it through `CPATH` and
  `LIBRARY_PATH` for compilation only.
- No NVLink on these nodes, so `env/env.sh` sets `NCCL_P2P_DISABLE=1`. Without it NCCL initialization hangs
  with no error and the server never becomes ready.
- Do not enable `--enable-expert-parallel`. All-reduce already crosses PCIe rather than NVLink, and the
  added all-to-all traffic measured about 9 percent slower on this hardware.
- Anthropic's hosted tools return HTTP 400. Use [docs/web-search.md](../../../docs/web-search.md).
