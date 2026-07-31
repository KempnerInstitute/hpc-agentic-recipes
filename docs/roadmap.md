# Deployment plan: newly downloaded models

Investigation only. No GPU jobs were run for this document. Every number below is either read from
the downloaded checkpoints, computed from their configs, or cited from upstream sources. Throughput
figures from other people's hardware are labeled as such.

Checkpoints live in `/n/netscratch/kempner_dev/Lab/mmsh/models/` (to be copied to
`/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/`).

## Hardware baseline

Per-GPU memory in decimal GB, to match Hugging Face reported sizes.

| Node type | GPUs | Per GPU | Per node | Usable at `--gpu-memory-utilization 0.90` |
|-----------|------|---------|----------|-------------------------------------------|
| H200 (`kempner_h200`) | 4 | 150.8 GB | 603 GB | about 543 GB |
| RTX PRO 6000 (`kempner_rtx`) | 8 | 103.1 GB | 825 GB | about 742 GB |

Two facts drive most decisions below:

1. Our H200 nodes have **4** GPUs, not the 8 that vendor recipes assume. Anything needing "8x H200"
   is a 2-node Ray job for us.
2. The RTX node is **Blackwell (sm_120)**, so it has native FP4 and MXFP4 math. Hopper H200 does not.
   We already proved FP4 works there: GLM-5.2-NVFP4 serves at 93.4 tok/s single stream on one RTX node,
   and DeepSeek-V4-Pro, whose experts are FP4, runs on two RTX nodes while failing on H200.

> [!NOTE]
> Gemma-4-26B, Gemma-4-31B, Qwen3-235B, and Qwen3-Coder-480B are now measured and served; see
> "Measured decode rates" in README.md for the real numbers, which supersede the estimates below.
> Kimi-K3 remains blocked: support is not merged in vLLM (PR #50000 open) or SGLang.

## Summary

| Model | On-disk | Params | Recommended target | Engine | Ready today |
|-------|---------|--------|--------------------|--------|-------------|
| gemma-4-26B-A4B-it | 51.6 GB | 26B MoE, 4B active | 1 GPU (either type) | vLLM | Yes |
| gemma-4-31B-it | 62.6 GB | 31B dense | 1 GPU (either type) | vLLM | Yes |
| Qwen3-235B-A22B | 470.2 GB | 235B MoE, 22B active | 1 RTX node TP8, or 1 H200 node TP4 | vLLM | Yes |
| Qwen3-Coder-480B-A35B | 960.3 GB | 480B MoE, 35B active | 1 RTX node **TP4 x PP2** using the FP8 repo | vLLM | Yes, measured |
| DeepSeek-V4-Pro | 864.7 GB | 1.6T MoE, 49B active | 2 RTX nodes (Blackwell FP4) | vLLM | Yes, multi-node |
| Kimi-K3 | 1561.0 GB | 2.8T MoE, 104B active | 3 RTX nodes, or 4 H200 nodes | vLLM | **No, needs a vLLM upgrade** |

## Engine support, as installed

Checked by grepping the registries in our three environments, not from memory.

| Architecture | vLLM 0.25.1 (`.venv`, `.venv-cu130`) | SGLang 0.5.11.dev (`.venv-sglang`) |
|--------------|--------------------------------------|------------------------------------|
| `Gemma4ForConditionalGeneration` | Yes, plus a `Gemma4MTPModel` draft entry | Yes (`gemma4_mm.py`, `gemma4_causal.py`) |
| `Qwen3MoeForCausalLM` | Yes | Yes (`qwen3_moe.py`) |
| `DeepseekV4ForCausalLM` | Yes, plus `DSparkDraftModel` | **No** (needs SGLang 0.5.12 or newer) |
| `KimiK3ForConditionalGeneration` | **No** | **No** |
| `KimiLinearForCausalLM` / `kimi_linear` | Yes | Yes |

Quantization kernels present in our vLLM: `compressed_tensors_moe_w4a4_mxfp4`, `..._nvfp4`, `mxfp4`,
`modelopt`, `fp8`, `marlin_utils_fp4`, `moe_wna16`. So Kimi-K3's MXFP4 format has kernel support
already; only the K3 model wrapper is missing.

Versions: vLLM latest stable is **0.26.0** (we run 0.25.1). SGLang latest is **0.5.16** (we run
0.5.11.dev).

> [!IMPORTANT]
> Claude Code needs the native Anthropic API, which only vLLM exposes. SGLang is OpenAI-only, so it
> stays an alternative for non-Claude clients. For these six models SGLang currently offers no
> capability we lack, and it is *behind* on DeepSeek-V4, so vLLM is the default everywhere below.

## KV cache math

Computed from each config: `2 (K and V) x kv_heads x head_dim x 2 bytes` per token per full-attention
layer. Sliding layers cost a fixed window instead of growing with context. This is what decides how
much context we can afford after weights.

| Model | Attention layout | KV per token | At 32K | At 128K | At 256K |
|-------|------------------|--------------|--------|---------|---------|
| gemma-4-31B-it | 10 full + 50 sliding(1024) | 160 KiB | 5.8 GB | 21 GB | 41 GB |
| gemma-4-26B-A4B-it | 5 full + 25 sliding(1024) | 40 KiB | 1.5 GB | 5.2 GB | 10 GB |
| Qwen3-235B-A22B | 94 full, GQA 4 kv heads | 188 KiB | 7.3 GB (40K max) | n/a | n/a |
| Qwen3-Coder-480B | 62 full, GQA 8 kv heads | 248 KiB | 8 GB | 31 GB | 62 GB |
| Kimi-K3 | 24 gated MLA + 69 KDA recurrent | about 27 KiB | 0.9 GB | 3.4 GB | 6.8 GB |
| DeepSeek-V4-Pro | hybrid CSA + HCA | very small | small | small | small |

Two useful consequences:

- **Kimi-K3 and DeepSeek-V4-Pro are cheap in KV despite 1M context.** K3 pays full-attention KV on
  only 24 of 93 layers, and those use MLA compression; the 69 KDA layers hold a constant-size
  recurrent state. DeepSeek claims 10 percent of DeepSeek-V3.2's KV at 1M. For both, weights dominate
  and context length is nearly free.
- **Qwen3-Coder-480B is the opposite.** At 256K its KV alone is 62 GB, so context length is a real
  memory lever there.

`--kv-cache-dtype fp8` halves every number in that table and is worth using on all of these.

## Per model

### gemma-4-31B-it and gemma-4-26B-A4B-it

Both are `Gemma4ForConditionalGeneration`, bf16, multimodal, 256K context, 5:1 sliding-to-full
attention. Supported by our vLLM today.

- **26B-A4B** (MoE, 128 experts, top 8): 51.6 GB weights. One GPU of either type covers weights plus
  256K context. Start with `--tensor-parallel-size 1`; use TP2 only for throughput.
- **31B** (dense): 62.6 GB. One H200 GPU holds weights plus 256K context (104 GB of 150.8). On one RTX
  GPU that is marginal at 256K, so either cap context at 128K or use TP2.

These are the cheapest wins and the right first test: they validate the Gemma 4 path end to end
without a multi-node job. Worth checking whether `Gemma4MTPModel` gives speculative decoding, which
would help the dense 31B most.

Options that help: TP2 for latency, `--kv-cache-dtype fp8` for long context, and
`--enable-prefix-caching` for agentic reuse.

### Qwen3-235B-A22B

`Qwen3MoeForCausalLM`, bf16, 470.2 GB, 128 experts top 8, GQA with only 4 KV heads. Native context is
**40960**, shorter than the others; longer needs YaRN scaling.

- **1 RTX node, TP8**: 470 GB across 742 GB usable, about 59 GB per GPU. Comfortable. Recommended.
- **1 H200 node, TP4**: 470 + 7 GB KV = 477 GB against 543 GB usable. Fits, but only about 66 GB of
  slack, so leave context at native 40K.
- **FP8 alternative**: `Qwen3-235B-A22B-FP8` is 239.1 GB upstream. That halves the footprint and would
  fit 2 GPUs, freeing a node. Worth downloading if we want this model resident cheaply.

Add `--tool-call-parser`/`--reasoning-parser` for the Qwen3 family. Do **not** add
`--enable-expert-parallel`: it measured 9 percent slower on the RTX node, which has no NVLink, so the
extra all-to-all traffic costs more than the sharding saves.

### Qwen3-Coder-480B-A35B-Instruct

The strongest agentic-coding candidate here. `Qwen3MoeForCausalLM`, bf16, 960.3 GB, 160 experts top 8,
262144 context. Supported today.

- As downloaded (bf16) it **does not fit one RTX node** (960 > 742) and needs 2 H200 nodes.
- **Download `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8` (482.2 GB) instead.** That fits **one RTX node**
  with room for the full 256K context (482 + 62 = 544 GB of 742), or one H200 node at moderate context
  (482 + 31 = 513 GB of 543 at 128K, tight).

Single node matters: it keeps the collectives inside one box. Note that using PP here does forbid
speculative decoding, but this checkpoint ships no draft model, so nothing is lost.

Measured shape: 1 RTX node, **TP4 x PP2**, FP8 weights, `--tool-call-parser qwen3_coder`,
`--max-model-len` 131072. TP8 is impossible for this checkpoint: `moe_intermediate_size` is 2560 and
the FP8 block is 128, so 2560/8 = 320 is not divisible by 128. Skip `--enable-expert-parallel` for the
same reason as the 235B. It does not run on H200 with CUDA graphs at all; see README.md.

### DeepSeek-V4-Pro

1.6T total, 49B active, 1M context, hybrid Compressed Sparse Attention plus Heavily Compressed
Attention, `mHC` residuals. The checkpoint is mixed precision: **FP8 attention and dense, FP4
experts** (`expert_dtype: fp4`). It also has `num_nextn_predict_layers: 1`, so an MTP head exists, and
our vLLM registry already carries a `DSparkDraftModel` entry.

Supported by our vLLM 0.25.1. **Not** supported by our SGLang 0.5.11.

- **2 RTX nodes** is the natural target: 865 GB across 1484 GB usable, and Blackwell executes the FP4
  experts natively. This is the configuration I would try.
- **2 H200 nodes** fits by memory (1086 GB usable) but Hopper has no FP4 hardware. The FP4 experts
  would fall back to emulation or a Marlin path, which is a correctness and speed risk. If we want
  this on H200, plan on an FP8 expert build instead (DeepSeek's own `convert.py` supports
  `--expert-dtype fp8`, and `nvidia/DeepSeek-V4-Pro-NVFP4` exists at 913 GB as another variant).

Trade-off to decide at test time: TP8 within each node plus PP2 across nodes is the shape that worked
for our GLM-5.2 and Kimi-K2.7 multi-node runs, but **pipeline parallelism disables speculative
decoding in vLLM**, so we would give up the MTP/DSpark speedup. Pure TP16 across two nodes preserves
MTP in principle but hung for Kimi-K2.7 in our earlier testing. Try TP8xPP2 first for a working
baseline, then measure what MTP would be worth.

### Kimi-K3

The hardest one, and the only one blocked today.

2.8T total, 104B active, 93 layers (69 Kimi Delta Attention plus 24 gated MLA), 896 experts with 16
selected plus 2 shared, native multimodal including video, 1M context, weights already **MXFP4**
(that is how 2.8T params fit in 1561 GB).

- **Blocker: `KimiK3ForConditionalGeneration` is absent from vLLM 0.25.1 and from SGLang 0.5.11.**
  Upstream announced day-0 support on 2026-07-27 for both engines, so this needs vLLM 0.26.0 or a
  nightly, in a separate environment so the current endpoints stay untouched.
- Memory: 1561 GB of weights plus a few GB of KV.
  - **3 RTX nodes**: 2226 GB usable. Comfortable, and Blackwell runs MXFP4 natively through the
    FlashInfer trtllm-gen path.
  - **4 H200 nodes**: 2030 GB usable. Works, but Hopper falls back to Marlin W4A16 for MXFP4, which is
    slower.
  - 3 H200 nodes is 1629 GB usable against 1564 GB needed. It technically fits with about 65 GB of
    slack, which is too little for comfort. 2 RTX nodes (1484 GB) does not fit at all.
- Upstream reports, on far larger nodes than ours (8x B300 minimum, 16x B200 also supported):
  111 tok/s per user at TP8, 118 at TP16, rising to 331 and 370 with DSpark speculative decoding, a
  3.14x speedup. The DSpark draft model is a small separate download (`Inferact/Kimi-K3-DSpark`,
  7.1 GB, or `RadixArk/Kimi-K3-DSpark`, 4.5 GB).
- Same PP caveat as DeepSeek: on our 4-GPU or 8-GPU nodes this is a 3 to 4 node job, and if we reach
  it with pipeline parallelism we lose DSpark.

My recommendation is to treat K3 as a later milestone: get the single-node models working first, then
attempt K3 on 3 RTX nodes once a newer vLLM is built and validated.

## Options that help, in rough order of value

1. **Use official FP8 checkpoints where they exist.** Qwen3-Coder-480B goes 960 to 482 GB and becomes a
   single-node model; Qwen3-235B goes 470 to 239 GB. This is the single biggest lever we have.
2. **`--kv-cache-dtype fp8`.** Halves KV. Matters most for Qwen3-Coder at long context.
3. **Prefer one node over two.** Single node keeps TP inside NVLink or PCIe, avoids Ray, and keeps
   speculative decoding legal. Two nodes cost us MTP whenever we need PP.
4. **Speculative decoding where the model provides a head.** DeepSeek-V4 has MTP, Kimi-K3 has DSpark
   (3.14x upstream), Gemma 4 may have an MTP draft entry. This is the largest decode-speed lever after
   quantization, and it is single-node only.
5. **MoE backend flags**, such as the `flashinfer_trtllm` runner upstream recommends for K3 at TP > 1.
   Do not assume `--enable-expert-parallel` helps: on the RTX node, which has no NVLink, it measured 9
   percent slower than plain tensor parallelism.
6. **Cap `--max-model-len`.** Agentic coding rarely needs 256K, and dropping to 128K or 32K buys back
   tens of GB on the Qwen models.
7. **`--enable-prefix-caching`** for agentic sessions, where the system prompt and tool definitions
   repeat on every turn.
8. **`--load-format fastsafetensors`** to cut load time on the multi-hundred-GB checkpoints.
9. **Right-size the node type.** Send FP4 and MXFP4 work to the RTX Blackwell node; send bf16 and FP8
   work wherever there is room. Hopper cannot do FP4 natively.

## Suggested order when GPUs are available

1. `gemma-4-26B-A4B-it` on 1 GPU. Fastest validation of the Gemma 4 path.
2. `gemma-4-31B-it` on 1 GPU. Confirms the dense variant and 256K context.
3. `Qwen3-235B-A22B` on 1 RTX node, TP8, as downloaded.
4. Download the Qwen3-Coder FP8 repo, then serve it on 1 RTX node, TP4 x PP2. Expected to be the best
   coding endpoint of this batch.
5. `DeepSeek-V4-Pro` on 2 RTX nodes, TP8xPP2. Multi-node, so budget time for Ray and NCCL issues.
6. Build a vLLM 0.26.0 or nightly environment, then attempt `Kimi-K3` on 3 RTX nodes.

## Open questions to settle by testing

- Does vLLM 0.25.1 run DeepSeek-V4's FP4 experts on Hopper at all, or is Blackwell mandatory?
- Does `Gemma4MTPModel` actually enable speculative decoding for these two Gemma checkpoints?
- Does pure TP across nodes work in 0.25.1/0.26.0, which would preserve MTP for the multi-node models,
  or does it still hang as it did for Kimi-K2.7?
- Does Qwen3-235B tolerate YaRN scaling past its native 40960 without quality loss?
- Throughput numbers for models we have not yet served. Figures quoted for those come from other
  people's GPUs. Our own 14 recipes were measured on 2026-07-31 across concurrency 1 to 512 and are
  recorded in each recipe README.
