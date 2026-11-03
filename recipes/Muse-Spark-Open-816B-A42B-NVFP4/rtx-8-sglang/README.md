# Muse Spark Open 816B-A42B NVFP4 on one RTX PRO 6000 node, under SGLang

Status: Validated - SGLang in a container, protocol: slope(128,1152) at concurrency 1, 8, 32 and 48

| | |
| --- | --- |
| Served name | `muse-spark` |
| Checkpoint | `Muse-Spark-Open-816B-A42B-NVFP4`, Hugging Face `meta-models/Muse-Spark-Open-816B-A42B-NVFP4` |
| On disk | 497 GiB across 127 shards, NVFP4 with packed routed experts, `MuseSparkForConditionalGeneration` |
| Served precision | NVFP4 weights, `quant_method` modelopt, group size 16, BF16 KV cache |
| Context | 262144. The checkpoint declares 1048576 |
| Hardware | 1 RTX PRO 6000 Blackwell node, 8 GPUs, 97887 MiB each, sm120, TP8 with EP8 |
| Engine | SGLang in Singularity, `marlin` MoE and `flashinfer` attention, DFlash speculative decoding |
| Draft model | `Muse-Spark-Open-DFlash2`, Hugging Face `RadixArk/Muse-Spark-Open-DFlash2`, 9.2 GiB |
| Allocation for the measurement | 1 node, 8 GPUs, 128 cores and 1440 GB, `kempner_rtx` |

This is the same 816B-A42B mixture-of-experts as the BF16 recipe, quantized to NVFP4 so it fits one node.
It serves the text path only.

**The two backend choices are not interchangeable here, and the published upstream values do not work on
this hardware.** See Known limits before changing `MOE_RUNNER` or `ATTN`.

## 1. Create the API key

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/Muse-Spark-Open-816B-A42B-NVFP4-rtx-8-sglang.key
chmod 600 secrets/Muse-Spark-Open-816B-A42B-NVFP4-rtx-8-sglang.key
```

- `secrets/vllm_api_key` is read when this file is absent.
- Requests without the key receive HTTP 401.
- The key reaches SGLang through `common/tools/sglang_launch.py`, which reads it from the environment and
  appends `--api-key` inside the process. SGLang accepts a key only as a flag, and a flag lands in
  `/proc/<pid>/cmdline`, which any user on the node can read.

## 2. Get the container

The engine is a prebuilt SGLang image carrying a Muse fork at `/sgl-workspace/sglang-muse-spark`. Place it
at `$MODELS_DIR/Muse-Spark-Open-816B-A42B-NVFP4/container/sglang-muse-spark-rtxpro6000.sif`, or point
`SIF` at your own copy. There is no venv to build for this recipe.

## 3. Launch

```
sbatch --account=<your-account> recipes/Muse-Spark-Open-816B-A42B-NVFP4/rtx-8-sglang/serve.sbatch
```

To bring it up on a node you already hold:

```
bash recipes/Muse-Spark-Open-816B-A42B-NVFP4/rtx-8-sglang/serve_ssh.sh <node>
```

Startup is about 8 minutes, dominated by reading 497 GiB of weights, then CUDA graph capture and a
post-capture KV resize. The log is `$MUSE_LOG_DIR/muse-nvfp4-sglang.log`, default `/tmp/$USER/muse`.

## 4. Verify

```
KEY=$(cat secrets/Muse-Spark-Open-816B-A42B-NVFP4-rtx-8-sglang.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the node>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"muse-spark","messages":[{"role":"user","content":"What is 2+2? Answer with just the number."}],"max_tokens":4096,"temperature":0.6}'
```

| Command | Expected |
| --- | --- |
| `/v1/models` with the key | `"id": "muse-spark"` |
| `/v1/models` without a key | `401` |
| chat completion | `content` holds the answer, `finish_reason: stop` |

As on the BF16 sibling, the reasoning chain is billed in `completion_tokens` but returned in neither
`content` nor `reasoning_content`, so a small `max_tokens` yields `content: null` with
`finish_reason: length`. Allow several thousand output tokens, and prefer a temperature above 0.

## 5. Connect a client

```
export NODE=<the node>
source recipes/Muse-Spark-Open-816B-A42B-NVFP4/rtx-8-sglang/client.env
claude
```

`client.env` exports the base URL, the key, the served model name, and
`CLAUDE_CODE_MAX_CONTEXT_TOKENS=262144`, without which Claude Code assumes and enforces a 200k window.
Use `ANTHROPIC_AUTH_TOKEN`, not `ANTHROPIC_API_KEY`, which sends a header the engine ignores and yields
401 on every request.

## 6. Stop it

```
scancel <job id>
```

For an ssh launch, kill `sglang_launch.py` on the node. Confirm the GPUs are released before relaunching.

## Tunable inputs

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/Muse-Spark-Open-816B-A42B-NVFP4` | Serve a different copy of the checkpoint |
| `DRAFT` | `$MODELS_DIR/Muse-Spark-Open-DFlash2` | DFlash draft checkpoint |
| `SIF` | the path under the checkpoint's `container/` | Container image |
| `API_PORT` | 8000 | Listening port |
| `MAX_MODEL_LEN` | 262144 | Context window, fixed at init |
| `MEM_FRACTION` | 0.93 | Static VRAM fraction. This sets the KV pool: see Known limits |
| `TP` | 8 | Tensor parallel size |
| `EP` | 8 | Expert parallel size. Must equal `TP`: Marlin cannot reduce expert-TP partials |
| `MOE_RUNNER` | `marlin` | MoE backend. Do not change without reading Known limits |
| `ATTN` | `flashinfer` | Attention backend. Do not change without reading Known limits |
| `KV_DTYPE` | `bfloat16` | KV cache dtype. `fp8_e4m3` roughly doubles the pool at some accuracy cost |
| `PAGE_SIZE` | 128 | Tokens per KV page |
| `CHUNKED_PREFILL` | 4096 | Chunked prefill size |
| `NO_SPEC` | 0 | Set to 1 to serve without the DFlash draft |
| `EXTRA_ARGS` | unset | Extra flags appended to the launch command |
| `KEY_NAME`, `KEY_FILE`, `VLLM_API_KEY` | this recipe's key | Which key `common/lib/api_key.sh` resolves |

## Benchmarking

Conditions:

| | |
| --- | --- |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| Input length | ISL 151 tokens. Rates at a long input are not measured; use `--prompt-tokens` |
| Output length | OSL 1152 tokens, output only, `ignore_eos` |
| Context | `MAX_MODEL_LEN=262144`, `MEM_FRACTION=0.93`, `NCCL_P2P_DISABLE=1` |
| Allocation for the measurement | 1 node, 8 GPUs, 128 cores and 1440 GB, `kempner_rtx` |
| Sequence cap | `max_running_requests` 48, which the engine derives from the KV pool and which caps the sweep |
| Endpoint | idle; the benchmark client ran on a different node |

Results:

| Concurrency | Aggregate | Per stream | Spread over 3 runs |
| --- | --- | --- | --- |
| 1 | 102.3 tok/s | 102.3 tok/s | 97.6 to 104.3 |
| 8 | 411.3 tok/s | 51.4 tok/s | 402.4 to 420.3 |
| 32 | 936.8 tok/s | 29.3 tok/s | 921.2 to 961.5 |
| 48 | 1084.2 tok/s | 22.6 tok/s | 1080.4 to 1102.5 |

| | |
| --- | --- |
| Engine facts | KV cache 295,552 tokens, 1.13 full-length requests at once; `max_running_requests` 48 |
| Label | capped. The sweep stops at 48 because the engine admits no more, not because throughput plateaued |

## Choosing this recipe or the four-node BF16 one

One RTX node is faster for a single caller and far cheaper in hardware; four H200 nodes serve many more
callers at once. Both were measured at the same 262144 context.

| | NVFP4, 1 RTX node | BF16, 4 H200 nodes |
| --- | --- | --- |
| One caller | 102.3 tok/s | 83.0 tok/s |
| Concurrency 8 | 411.3 tok/s | 382.5 tok/s |
| Concurrency 32 | 936.8 tok/s | 727.2 tok/s |
| Highest measured | 1084.2 tok/s at 48 | 9458.8 tok/s at 1024 |
| Concurrent request cap | 48 | 1024 |

## Known limits

- **The MoE backend must be `marlin`.** This checkpoint ships packed NVFP4 experts.
  `flashinfer_cutlass`, the value published for RTX PRO 6000 upstream, is rejected at load: SGLang asks
  for the TRT-LLM backend instead. `flashinfer_trtllm` then fails on this hardware, because FlashInfer's
  trtllm-gen batched GEMM ships cubins for sm100, sm103 and sm107 and none run on sm120. SGLang's own
  auto-selection picks Marlin only for SM80 to SM90, so on sm120 `auto` chooses the backend that cannot
  work. Marlin must be named explicitly.
- **The attention backend must be `flashinfer`.** `trtllm_mha` belongs to the same TRT-LLM kernel family
  and dies with an illegal memory access during prefill graph capture, inside
  `gated_attention -> radix_attention`, after the MoE has loaded cleanly.
- `EP` must equal `TP`. The Marlin path does not reduce expert-TP partials and refuses to start otherwise.
- The KV pool is set by `MEM_FRACTION`, and the default of 0.85 is far too small here. SGLang reserves
  `pre_model_load_memory * (1 - mem_fraction_static)` as headroom, and its post-capture resize can only
  shrink the pool, never grow past the pre-capture cap. At 0.85 that left 13.98 GiB per GPU reserved and a
  20,736-token pool, which cannot hold even one full-length request; 0.93 gives 292,352 tokens. Lowering
  `MAX_MODEL_LEN` does not help, because context caps per-request length and does not size the pool.
- Greedy decoding can fail to terminate. At temperature 0, "What is 2+2? Answer with just the number." ran
  past 8192 tokens in 2 of 3 attempts and returned empty content. Other short prompts at temperature 0 were
  fine, and the same prompt at temperature 0.6 answered in 31 and 40 tokens. The BF16 sibling shows the
  same nondeterminism on different hardware and a different engine with no speculative decoding, so this is
  a model trait, not a fault of Marlin, DFlash or NVFP4. Prefer sampling over greedy decoding.
- `NCCL_P2P_DISABLE=1` costs throughput, and is set anyway. `env/env.sh` exports it because RTX multi-GPU
  NCCL init can hang without it, which is why every RTX recipe here sets it. This recipe does run without
  it, and measurably faster: 1265.5 tok/s at concurrency 48 against 1084.2 with it, about 14 percent, and
  2 to 4 percent at the lower levels. The figures above are the slower, safer configuration that ships.
- Throughput above concurrency 48 is unmeasured, because the engine admits no more at this pool size.
- Decode rate at a long input is not measured. Every figure here uses a 151-token input.
- The vision and audio towers are loaded but not exercised. This recipe is validated for text only.
