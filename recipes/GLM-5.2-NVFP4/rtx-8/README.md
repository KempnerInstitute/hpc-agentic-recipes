# GLM-5.2-NVFP4 on one RTX PRO 6000 node

Status: Validated - vLLM 0.25.1, protocol: slope(128,1152) at concurrency 1, 8, 32, 64, 128, 256 and 512

| | |
| --- | --- |
| Served name | `glm-5.2` |
| Checkpoint | `GLM-5.2-NVFP4`, Hugging Face `nvidia/GLM-5.2-NVFP4`, quantized from `zai-org/GLM-5.2` |
| On disk | 432.90 GiB across 47 shards, NVFP4, `GlmMoeDsaForCausalLM` |
| Served precision | NVFP4 weights and activations, 57.48 GiB per GPU, KV cache FP8 in sparse-MLA layout |
| Context | 217344. The checkpoint declares 1048576, which does not fit this hardware; see Known limits |
| Hardware | 1 RTX PRO 6000 Blackwell node, 8 GPUs, 97887 MiB each, TP8 over PCIe with no NVLink |
| Engine | vLLM 0.25.1, CUDA graphs, MTP speculative decoding at 3 draft tokens |

## 1. Create the API key

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/GLM-5.2-NVFP4-rtx-8.key
chmod 600 secrets/GLM-5.2-NVFP4-rtx-8.key
```

- `secrets/vllm_api_key` is read when this file is absent.
- To rotate, replace the file and relaunch; the engine reads it once at launch.

## 2. Build the environment

Run on a compute node, not a login node. Needs `uv` and `mamba` on your PATH.

```
module load Mambaforge
bash recipes/GLM-5.2-NVFP4/rtx-8/env/build.sh
```

- About 9.2 GB for the venv and 195 packages, plus a 2.9 GB conda CUDA 13.0 toolkit alongside it under
  `ENV_ROOT`, default scratch.
- The toolkit is required: the FlashInfer sm_120 JIT needs a complete CUDA 13 install, which the node's
  runtime-only `/usr/local/cuda-13` and the fragmented pip nvcc wheels do not provide.
- vLLM comes from the nightly cu130 index, because sm_120 needs a CUDA 13 build and uv's `--torch-backend`
  stops at cu129. `build.sh` pins `vllm==0.25.1` explicitly, because the installed metadata carries no local
  version tag and an unpinned install drifts silently.
- FlashInfer is pinned to 0.6.15 and installed with `--no-deps`. vLLM 0.25.1 pins 0.6.13, which rejects the
  `kv_scale_format` argument the sm_120 backend passes and fails at the first request.

## 3. Launch

Slurm, from the repo root:

```
sbatch --account=<your-account> recipes/GLM-5.2-NVFP4/rtx-8/serve.sbatch
squeue --me                                  # NODELIST gives the host
tail -f glm52-nvfp4-<jobid>.log
```

Direct, on a node you already hold. This recipe uses all eight GPUs, so there is no device to pin:

```
bash recipes/GLM-5.2-NVFP4/rtx-8/serve_ssh.sh <node>
```

| Stage | Measured |
| --- | --- |
| Launch to serving, first time on a node | 12 min 27 s |
| Launch to serving, caches warm | 5 min 31 s |
| Weight load alone | 118 to 135 s |

- The first launch on any node compiles the sm_120 attention kernels from source, because no matching
  FlashInfer cubin package exists. That cache is node-local, so a different node pays it again.
- On the direct path the server log is node-local at `/tmp/$USER/vllm/`. Under Slurm it lands in the submit
  directory.
- Over bare SSH `nproc` reports 1 while `Cpus_allowed_list` is the full set, so the server is not CPU limited
  on that path even though it looks like it.

## 4. Verify

```
KEY=$(cat secrets/GLM-5.2-NVFP4-rtx-8.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the node serving it>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"What is 17*23? Answer briefly."}],"max_tokens":400}'
```

| Command | Expected |
| --- | --- |
| `/v1/models` with the key | `"id": "glm-5.2"`, `"max_model_len": 217344` |
| `/v1/models` without a key | `401` |
| chat completion | `content` holds 391, `finish_reason: stop` |

- 400 output tokens is enough here: three smoke tests used 94, 128 and 156 tokens. Reasoning arrives in a
  separate `reasoning` field, so a much smaller budget can end with empty `content`.

## 5. Connect a client

```
export NODE=<the node serving it>
source recipes/GLM-5.2-NVFP4/rtx-8/client.env
claude
```

- `client.env` sets `CLAUDE_CODE_MAX_CONTEXT_TOKENS=217344`. Claude Code assumes and enforces a 200k window
  for a served name it does not recognize, which is below what this endpoint serves.
- Use `ANTHROPIC_AUTH_TOKEN`. `ANTHROPIC_API_KEY` sends `x-api-key` and returns 401.
- `client.env` also sets `ANTHROPIC_SMALL_FAST_MODEL`; without it the client reaches for a hosted Haiku.
- OpenAI clients such as Codex: base URL `http://<node>:8000/v1`, same key, model `glm-5.2`. See
  [docs/clients.md](../../../docs/clients.md).

## 6. Stop it

```
scancel <jobid>                        # Slurm path
bash common/tools/stop.sh <node>       # direct path
```

## Tunable inputs

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/GLM-5.2-NVFP4` | Serve a different copy of the checkpoint |
| `API_PORT` | 8000 | Listening port |
| `MAX_MODEL_LEN` | 217344 | Context window, the ceiling this hardware allows |
| `GPU_UTIL` | 0.90 | Fraction of VRAM for weights plus KV cache |
| `TP` | 8 | Tensor parallel size, and the node's GPU count |
| `MTP_TOKENS` | 3 | Speculative tokens per step |
| `NO_MTP` | unset | Set to disable speculative decoding, for a stable single stream baseline |
| `ATTN_BACKEND` | unset | Override vLLM's attention backend selection |
| `EXTRA_ARGS` | unset | Extra flags appended to the `vllm serve` command line |
| `TOOL_PARSER` | `glm45` | Tool call parser |
| `REASONING_PARSER` | `glm45` | Reasoning parser |
| `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` | 3600 | Raised from 480 so a storage stall that resolves is survivable |
| `LOG_DIR` | `/tmp/$USER/vllm` | Where the SSH path writes the server log |
| `VENV_DIR` | under `ENV_ROOT` | Use an environment built elsewhere |
| `CUDA13_DIR` | under `ENV_ROOT` | Use a CUDA 13.0 toolkit built elsewhere |
| `VLLM_VERSION` | 0.25.1 | Engine version `env/build.sh` installs |
| `FLASHINFER_VERSION` | 0.6.15 | FlashInfer version `env/build.sh` installs |
| `TRANSFORMERS_VERSION` | 5.14.1 | The transformers version the rates were measured with |
| `VLLM_CACHE_ROOT` | under `ENV_ROOT` | Where vLLM keeps compiled artifacts; the engine default is a small quota here |
| `KEY_NAME`, `KEY_FILE`, `VLLM_API_KEY` | this recipe's key | Which key `common/lib/api_key.sh` resolves; an exported `VLLM_API_KEY` wins |
| `NODE`, `GLM52_NVFP4_NODE` | unset | The node to launch on or connect to; set either, or pass the node as an argument |
| `MODELS_DIR`, `ENV_ROOT` | `common/defaults.sh` | Override there or in `common/site.conf` |

## Benchmarking

Conditions:

| | |
| --- | --- |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| Input length | ISL 21 tokens. Rates at a long input are not measured; use `--prompt-tokens` |
| Output length | OSL 1152 tokens, output only, `ignore_eos` |
| Context | `MAX_MODEL_LEN=217344` |
| Allocation for the measurement | 8 GPUs, 128 cores, 1440 GB, `kempner_rtx` |
| Sequence cap | `max_num_seqs` 1024, the engine default, above the top sweep level |
| Preemption | 617 across the sweep |
| Endpoint | idle, and the benchmark client ran on a separate CPU-only node |
| Power | 600 W enforced, the card default, so not capped. Median 209 W across 8784 samples and a 251 W peak, with no sample above 590 W, so throughput here is not power bound |

Results:

| Concurrency | Aggregate | Per stream | Spread over 3 runs | TTFT median |
| --- | --- | --- | --- | --- |
| 1 | 91.1 tok/s | 91.1 tok/s | 90.4 to 94.5 | 121 ms |
| 8 | 378.3 tok/s | 47.3 tok/s | 377.2 to 382.5 | 234 ms |
| 32 | 673.8 tok/s | 21.1 tok/s | 535.1 to 681.1 | 605 ms |
| 64 | 1047.7 tok/s | 16.4 tok/s | 1046.6 to 1052.9 | 997 ms |
| 128 | 1263.6 tok/s | 9.9 tok/s | 1260.9 to 1266.6 | 1799 ms |
| 256 | 1374.5 tok/s | 5.4 tok/s | 1360.0 to 1401.5 | 3505 ms |
| 512 | 1209.6 tok/s | 2.4 tok/s | 1166.5 to 1213.3 | 5602 ms |

| | |
| --- | --- |
| Label | peak. Throughput turns over at 256 and falls 12 percent by 512, so 1374.5 tok/s is a measured ceiling rather than the edge of the sweep |
| Single stream range | 90.4 to 94.5 across three runs, and values as high as 101.6 have been recorded. MTP gain depends on how predictable the generated text is, so single stream is less reproducible than a non-speculative model's. `NO_MTP=1` gives a stable baseline |
| Widest spread | concurrency 32 measured 535.1 to 681.1, 27 percent. Every other level held within 4 percent |
| Quote for one caller | 91.1 tok/s |
| Quote for a shared endpoint | 1374.5 tok/s at concurrency 256 |
| KV cache | 349,888 tokens from 17.84 GiB per GPU, 1.61 full-length requests at once |
| Speculative decoding | MTP accepted 65.9 percent of draft tokens across the sweep, 2,638,177 of 4,003,386 |
| Long prompt | 207,523 tokens in 64 s cold; 0.5 s when the prefix is already cached |

Reproduce:

```
KEY_NAME=GLM-5.2-NVFP4-rtx-8 bash common/tools/bench.sh --host <node> --model glm-5.2
KEY_NAME=GLM-5.2-NVFP4-rtx-8 bash common/tools/bench.sh --host <node> --model glm-5.2 \
  --sweep 1,8,32,64,128,256,512
```

- `KEY_NAME` is required once `secrets/` holds more than one key, otherwise `bench.sh` resolves the shared
  key and every request returns 401.

## Known limits

- The checkpoint declares 1048576 but the engine refuses it: one request at that length needs 53.45 GiB of
  KV against 11.08 GiB available, and vLLM reports an estimated maximum of 217344, which is what this recipe
  serves. Context also shrinks the pool, since the sparse-attention workspace scales with the declared
  window: 19.04 GiB of KV at 131072 against 17.84 at 217344.
- Do not add the conda CUDA 13 toolkit to `LD_LIBRARY_PATH`. Its `libcudart` shadows torch's runtime and
  pulls in a missing `libcupti.so.13`, which breaks import; the recipe exposes it through `CPATH` and
  `LIBRARY_PATH` for compilation only.
- Do not enable `--enable-expert-parallel`. All-reduce already crosses PCIe rather than NVLink, and the
  added all-to-all traffic measured about 9 percent slower.
- Anthropic's hosted tools return HTTP 400. Use [docs/web-search.md](../../../docs/web-search.md).
