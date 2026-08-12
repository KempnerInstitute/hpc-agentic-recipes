# DeepSeek-V4-Flash-0731 on one RTX PRO 6000 node

Status: Validated - vLLM 0.25.1, protocol: slope(128,1152) at concurrency 1, 8, 32, 64, 128, 256, 512, 640,
768, 896 and 1024

| | |
| --- | --- |
| Served name | `deepseek-v4-flash` |
| Checkpoint | `DeepSeek-V4-Flash-0731`, Hugging Face `deepseek-ai/DeepSeek-V4-Flash-0731` |
| On disk | 155.43 GiB across 48 shards, `DeepseekV4ForCausalLM` |
| Served precision | FP8 dense layers with FP4 routed experts, 20.08 GiB per GPU, KV cache FP8 in sparse-MLA layout |
| Context | 1048576, the checkpoint maximum, served in full |
| Hardware | 1 RTX PRO 6000 Blackwell node, 8 GPUs, 96 GiB each, TP8 over PCIe with no NVLink |
| Engine | vLLM 0.25.1, CUDA graphs at `FULL_AND_PIECEWISE`, no speculative decoding; see Known limits |

## 1. Create the API key

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/DeepSeek-V4-Flash-0731-rtx-8.key
chmod 600 secrets/DeepSeek-V4-Flash-0731-rtx-8.key
```

- `secrets/vllm_api_key` is read when this file is absent.
- To rotate, replace the file and relaunch; the engine reads it once at launch.

## 2. Build the environment

Run on a compute node, not a login node. Needs `uv` and `mamba` on your PATH.

```
module load Mambaforge
bash recipes/DeepSeek-V4-Flash-0731/rtx-8/env/build.sh
```

- About 9.0 GB for the venv and 195 packages, plus a 2.9 GB conda CUDA 13.0 toolkit alongside it under
  `ENV_ROOT`, default scratch.
- The toolkit is required at run time, not only to build: DeepGEMM compiles kernels on the fly and checks for
  `nvcc` on disk, so the engine exits at load without it.
- vLLM comes from the nightly cu130 index, because sm_120 needs a CUDA 13 build and uv's `--torch-backend`
  stops at cu129. `build.sh` pins `vllm==0.25.1` explicitly, because the installed metadata carries no local
  version tag and an unpinned install drifts silently.
- FlashInfer is pinned to 0.6.15 and installed with `--no-deps`. The sparse-MLA decode path passes
  `swa_topk_lens`, which does not exist before 0.6.14.

## 3. Launch

Slurm, from the repo root:

```
sbatch --account=<your-account> recipes/DeepSeek-V4-Flash-0731/rtx-8/serve.sbatch
squeue --me                                  # NODELIST gives the host
tail -f dsv4-flash-rtx-<jobid>.log
```

Direct, on a node you already hold. This recipe uses all eight GPUs, so there is no device to pin:

```
bash recipes/DeepSeek-V4-Flash-0731/rtx-8/serve_ssh.sh <node>
```

| Stage | Measured |
| --- | --- |
| Launch to serving, caches warm | 2 min to 2 min 20 s across three launches |
| Weight load alone | 10 s warm, 47 to 57 s from cold storage |
| Engine init, profiling through warmup | 56 s |
| CUDA graph capture | 20 s, and 2.61 GiB per GPU |

- The first launch on any node is several minutes longer, because it compiles the sm_120 kernels from source.
  That cache is node-local, so a different node pays it again.
- On the direct path the server log is node-local at `/tmp/$USER/vllm/`. Under Slurm it lands in the submit
  directory.
- The release ships no Jinja chat template and none is needed. vLLM selects `tokenizer_mode deepseek_v4` on
  its own and encodes messages natively, which `/tokenize` confirms:
  `<｜begin▁of▁sentence｜><｜User｜>hi<｜Assistant｜></think>`.

## 4. Verify

```
KEY=$(cat secrets/DeepSeek-V4-Flash-0731-rtx-8.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the node serving it>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"What is 17*23? Answer briefly."}],"max_tokens":600}'
```

| Command | Expected |
| --- | --- |
| `/v1/models` with the key | `"id": "deepseek-v4-flash"`, `"max_model_len": 1048576` |
| `/v1/models` without a key | `401` |
| chat completion | `content` holds 391, `finish_reason: stop` |

- Unlike the other reasoning models here, a plain request returns no `reasoning` field. The prompt closes
  thinking before the model writes, so it answers directly and any working it shows arrives as `content`.
- To get a separate `reasoning` field, ask for thinking through `chat_template_kwargs`. Only that form works;
  a top-level `thinking` or `enable_thinking` is accepted and ignored.

```
curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Is 91 prime? Think it through."}],
       "max_tokens":700,"chat_template_kwargs":{"thinking":true},"reasoning_effort":"max"}'
```

- `reasoning_effort` accepts `low`, `high` and `max`. The published scores on the model card are all measured
  at `max` with thinking on, so a default request is not the configuration those numbers describe.

## 5. Connect a client

```
export NODE=<the node serving it>
source recipes/DeepSeek-V4-Flash-0731/rtx-8/client.env
claude
```

- `client.env` sets `CLAUDE_CODE_MAX_CONTEXT_TOKENS=1048576`. Claude Code assumes and enforces a 200k window
  for a served name it does not recognize, which is far below what this endpoint serves.
- Use `ANTHROPIC_AUTH_TOKEN`. `ANTHROPIC_API_KEY` sends `x-api-key` and returns 401.
- `client.env` also sets `ANTHROPIC_SMALL_FAST_MODEL`; without it the client reaches for a hosted Haiku.
- OpenAI clients such as Codex: base URL `http://<node>:8000/v1`, same key, model `deepseek-v4-flash`. See
  [docs/clients.md](../../../docs/clients.md).

## 6. Stop it

```
scancel <jobid>                        # Slurm path
bash common/tools/stop.sh <node>       # direct path
```

## Tunable inputs

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/DeepSeek-V4-Flash-0731` | Serve a different copy of the checkpoint |
| `API_PORT` | 8000 | Listening port, and part of the SSH log file name |
| `MAX_MODEL_LEN` | 1048576 | Context window, the checkpoint maximum |
| `GPU_UTIL` | 0.90 | Fraction of VRAM for weights plus KV cache |
| `TP` | 8 | Tensor parallel size, and the node's GPU count |
| `CUDAGRAPH_MODE` | `FULL_AND_PIECEWISE` | CUDA graph mode passed to `--compilation-config` |
| `SPEC_MODE` | unset | Must stay unset on this hardware; see Known limits |
| `SPEC_TOKENS` | 1 | Draft tokens per step, read only when `SPEC_MODE` is set |
| `EXTRA_ARGS` | unset | Extra flags appended to the `vllm serve` command line |
| `TOOL_PARSER` | `deepseek_v4` | Tool call parser |
| `REASONING_PARSER` | `deepseek_v4` | Reasoning parser |
| `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` | 3600 | Raised from 480 so a storage stall that resolves is survivable |
| `LOG_DIR` | `/tmp/$USER/vllm` | Where the SSH path writes the server log |
| `VENV_DIR` | under `ENV_ROOT` | Use an environment built elsewhere |
| `CUDA13_DIR` | under `ENV_ROOT` | Use a CUDA 13.0 toolkit built elsewhere |
| `VLLM_VERSION` | 0.25.1 | Engine version `env/build.sh` installs |
| `FLASHINFER_VERSION` | 0.6.15 | FlashInfer version `env/build.sh` installs |
| `TRANSFORMERS_VERSION` | 5.14.1 | The transformers version the rates were measured with |
| `VLLM_CACHE_ROOT` | under `ENV_ROOT` | Where vLLM keeps compiled artifacts; the engine default is a small quota here |
| `KEY_NAME`, `KEY_FILE`, `VLLM_API_KEY` | this recipe's key | Which key `common/lib/api_key.sh` resolves; an exported `VLLM_API_KEY` wins |
| `NODE`, `DSV4_FLASH_NODE` | unset | The node to launch on or connect to; set either, or pass the node as an argument |
| `MODELS_DIR`, `ENV_ROOT` | `common/defaults.sh` | Override there or in `common/site.conf` |

## Benchmarking

Conditions:

| | |
| --- | --- |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| Input length | ISL 10 tokens. Rates at a long input are not measured; use `--prompt-tokens` |
| Output length | OSL 1152 tokens, output only, `ignore_eos` |
| Context | `MAX_MODEL_LEN=1048576` |
| Allocation for the measurement | 8 GPUs, 128 cores, 1440 GB, `kempner_rtx` |
| Sequence cap | `max_num_seqs` 1024, the engine default for this hardware, which equals the top sweep level |
| Preemption | zero at every level |
| Endpoint | idle, and the benchmark client ran on a separate CPU-only node |
| Power | 600 W enforced, the card default, so not capped. Mean 200 W across 2768 samples and a 256 W peak, with nothing at or above 540 W, so throughput here is not power bound |

Results:

| Concurrency | Aggregate | Per stream | Spread over 3 runs | TTFT median |
| --- | --- | --- | --- | --- |
| 1 | 106.5 tok/s | 106.5 tok/s | 105.4 to 106.6 | 36 ms |
| 8 | 603.7 tok/s | 75.5 tok/s | 601.3 to 608.5 | 48 ms |
| 32 | 1546.6 tok/s | 48.3 tok/s | 1545.4 to 1548.6 | 97 ms |
| 64 | 2076.5 tok/s | 32.4 tok/s | 2076.4 to 2091.3 | 160 ms |
| 128 | 2783.8 tok/s | 21.7 tok/s | 2783.3 to 2795.1 | 206 ms |
| 256 | 4283.6 tok/s | 16.7 tok/s | 4269.1 to 4289.9 | 292 ms |
| 512 | 5160.2 tok/s | 10.1 tok/s | 5156.9 to 5162.0 | 528 ms |
| 640 | 5316.5 tok/s | 8.3 tok/s | 5313.7 to 5318.5 | 661 ms |
| 768 | 5484.1 tok/s | 7.1 tok/s | 5483.4 to 5488.1 | 706 ms |
| 896 | 5621.2 tok/s | 6.3 tok/s | 5618.7 to 5626.8 | 778 ms |
| 1024 | 5744.6 tok/s | 5.6 tok/s | 5737.1 to 5750.6 | 843 ms |

| | |
| --- | --- |
| Label | rising. The top value is at the top of the sweep and the levels at or above 512 vary by 10.2 percent, outside the 4 percent the rule allows for `saturated`, so 5744.6 tok/s is a floor |
| Where it bends | per stream falls from the start, 106.5 to 75.5 tok/s by concurrency 8 and 16.7 by 256, while aggregate keeps climbing to the top of the sweep |
| Quote for one caller | 106.5 tok/s |
| Quote for a shared endpoint | 5744.6 tok/s at concurrency 1024 |
| KV cache | 9,247,208 tokens from 60.33 GiB per GPU, 8.82 full-length requests at once |
| Long prompt | 180,005 tokens in 18.9 s cold; 0.37 s when the prefix is already cached |
| Reproducibility | every level held within 0.6 percent across its three runs, and concurrency 1 measured 106.5 tok/s in two separate launches |

Reproduce:

```
KEY_NAME=DeepSeek-V4-Flash-0731-rtx-8 bash common/tools/bench.sh --host <node> --model deepseek-v4-flash
KEY_NAME=DeepSeek-V4-Flash-0731-rtx-8 bash common/tools/bench.sh --host <node> --model deepseek-v4-flash \
  --sweep 1,8,32,64,128,256,512,640,768,896,1024
```

- `KEY_NAME` is required once `secrets/` holds more than one key, otherwise `bench.sh` resolves the shared
  key and every request returns 401.

## Known limits

- Speculative decoding does not work on this hardware, so `SPEC_MODE` must stay unset. The sm_120 sparse-MLA
  kernel requires more than 64 tokens per batch while a draft batch is only `num_speculative_tokens` wide, so
  `dspark` loads its drafter, sizes a pool, then aborts during warmup with `Check failed: num_tokens > 64`.
  The number in that message tracks the setting, 1 at `SPEC_TOKENS=1` and 3 at 3, so no draft depth helps.
  `deepseek_mtp` fails earlier still, at `KeyError: model.layers.43.mtp_block.main_norm.weight`, because this
  checkpoint stores its head at `mtp.0`, `mtp.1` and `mtp.2` rather than as one more layer. The vLLM recipe
  page reports the same sm_120 limitation for both methods.
- The aggregate figure is a floor twice over: throughput was still climbing at the top of the sweep, and that
  level is also the engine's own admission cap, so a ceiling would need `max_num_seqs` above 1024.
- A default request returns no `reasoning` field; see Verify for the `chat_template_kwargs` form that does.
- The flag set the vLLM recipe page lists for this hardware, `--enable-expert-parallel` with
  `--kv-cache-dtype fp8` and `--block-size 256`, is measurably slower where it matters and is not adopted:
  99.7 tok/s on a single stream against this recipe's 106.5, and 546.1 against 603.7 at concurrency 8, for a
  0.9 percent gain at 256. It resolves to the same KV pool, so `fp8` and `fp8_ds_mla` cost the same here.
  This recipe keeps `fp8_ds_mla` and the default block size.
- Anthropic's hosted tools return HTTP 400. Use [docs/web-search.md](../../../docs/web-search.md).
