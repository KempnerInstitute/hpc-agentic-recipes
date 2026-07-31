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
   We already proved FP4 works there: GLM-5.2-NVFP4 serves at about 90 tok/s on one RTX node.

> [!NOTE]
> Gemma-4-26B, Gemma-4-31B, Qwen3-235B, and Qwen3-Coder-480B are now measured and served; see
> "Measured decode rates" in README.md for the real numbers, which supersede the estimates below.
> Kimi-K3 was measured on 4 H200 nodes on 2026-07-31 under the SGLang container: 40.3 tok/s single
> stream, 87.1 with DSpark. See the measured section under Kimi-K3.

## Summary

| Model | On-disk | Params | Recommended target | Engine | Ready today |
|-------|---------|--------|--------------------|--------|-------------|
| gemma-4-26B-A4B-it | 51.6 GB | 26B MoE, 4B active | 1 GPU (either type) | vLLM | Yes |
| gemma-4-31B-it | 62.6 GB | 31B dense | 1 GPU (either type) | vLLM | Yes |
| Qwen3-235B-A22B | 470.2 GB | 235B MoE, 22B active | 1 RTX node TP8, or 1 H200 node TP4 | vLLM | Yes |
| Qwen3-Coder-480B-A35B | 960.3 GB | 480B MoE, 35B active | 1 RTX node **TP4 x PP2** using the FP8 repo | vLLM | Yes, measured |
| DeepSeek-V4-Pro | 864.7 GB | 1.6T MoE, 49B active | 2 RTX nodes (Blackwell FP4) | vLLM | Yes, multi-node |
| Kimi-K3 | 1561.0 GB | 2.8T MoE, 104B active | 4 H200 nodes, TP16 with EP16 | SGLang container | **Yes, measured 2026-07-31** |

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

#### Upstream survey, 2026-07-30

**Bottom line.** K3 became servable today: vLLM merged support into `main` at 10:49 UTC, and upstream now
marks H200 as verified. But it is in no release of either engine, and the only target on our fleet that
can physically hold the checkpoint is **4 H200 nodes, TP16, on the Marlin W4A16 kernel, under SGLang**.
Whether that is worth doing turns entirely on **DSpark**, and the numbers are further apart than they
look: 16.8 tok/s measured on 16 H200 *without* speculative decoding, 5.8 tok/s on 32 H100 without it, but
**674.94 output tok/s at batch 8 on 32 H200 with DSpark against 239.71 without**, a 2.8x gain on the same
Hopper Marlin path. So the honest read is that plain Marlin decode is poor and DSpark is the whole game on
Hopper. Two RTX nodes cannot hold the model at all, and 3 H200 nodes were never a candidate because TP12
is illegal. The most immediately useful findings are elsewhere: a 1.5 GB test model that validates the
code path on one GPU for free, an expert-pruned checkpoint at 837 GB that fits our 2 RTX nodes, and a
measured warning that FP8 KV cache **hurts** badly when DSpark is on.

##### What changed since 2026-07-29

| Thing checked | Status on 2026-07-30 | Source |
|---------------|----------------------|--------|
| vLLM latest release | **0.26.0**, PyPI 2026-07-25, GitHub tag 2026-07-27. No 0.27.0. | [PyPI](https://pypi.org/pypi/vllm/json), [releases](https://github.com/vllm-project/vllm/releases) |
| `KimiK3ForConditionalGeneration` in 0.26.0 | **No.** Absent from `registry.py` at tag `v0.26.0`, and `vllm/models/` has no `kimi_k3`. | [registry.py@v0.26.0](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/model_executor/models/registry.py) |
| `KimiK3ForConditionalGeneration` in vLLM `main` | **Yes.** PR #50000 merged **2026-07-30 10:49 UTC**, 82 files. Registry maps it to `vllm.models.kimi_k3`, a package with `nvidia/`, `amd/` and `common/` subtrees. `KimiK3MTPModel` and `K3DSparkModel` are registered too. | [PR #50000](https://github.com/vllm-project/vllm/pull/50000), [registry.py@main](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py) |
| vLLM docs | `supported_models.md` now lists the architecture, `moonshotai/Kimi-K3`, T + I. | [supported_models.md](https://github.com/vllm-project/vllm/blob/main/docs/models/supported_models.md) |
| Extra dependency | PR #50000 says verbatim: "**after merging, it still needs to install** https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.6.16rc5 **to run K3**". That release is dated 2026-07-30 07:10 UTC, while `requirements/cuda.txt` still pins `flashinfer-python==0.6.15.post1`. | [PR #50000](https://github.com/vllm-project/vllm/pull/50000), [FlashInfer releases](https://github.com/flashinfer-ai/flashinfer/releases) |
| vLLM recipes page | Still `"min_vllm_version": "0.27.0"` and `"nightly_required": true`, but the hardware map is now `h200: verified`, `b200/b300/gb200/gb300/mi355x: verified`, updated 2026-07-30. **`h100` is not in the verified map.** Estimated `vram_minimum_gb: 1680`. | [recipes.vllm.ai](https://recipes.vllm.ai/moonshotai/Kimi-K3) |
| SGLang `main` | **Still no K3.** `python/sglang/srt/models/` has `kimi_k25`, `kimi_linear`, `kimi_vl`, no `kimi_k3`. PR #32541 is open at 292 files and +51956 lines. Latest release is 0.5.16 (2026-07-25). | [PR #32541](https://github.com/sgl-project/sglang/pull/32541) |
| SGLang shipping path | The `kimi-k3` **branch** plus day-0 images `lmsysorg/sglang:kimi-k3` (CUDA 13) and `lmsysorg/sglang:kimi-k3-cu12` (**CUDA 12**), with Dockerfiles at `docker/kimi_k3/`. | [branch](https://github.com/sgl-project/sglang/tree/kimi-k3) |
| Tracking issue #50001 | Open, 7 comments. Every hardware failure reported in it is Blackwell, including a memory access violation on DGX B300 at TP8 under concurrent load ([#50147](https://github.com/vllm-project/vllm/issues/50147)). No Hopper reports either way. | [#50001](https://github.com/vllm-project/vllm/issues/50001) |

The blocker moved rather than cleared. On both engines this is pre-release code reached through a branch
or a container, not something we can `pip install`. The vLLM blog is explicit about that too: "Because of
complicated dependencies, **only Docker images are usable now**"
([blog](https://vllm.ai/blog/2026-07-27-k3)).

##### Is Hopper viable? Yes, and we can say why

MXFP4 has no Hopper tensor core, but neither engine dequantizes to bf16, which would double the footprint
and end the discussion. Both fall back to **Marlin W4A16**, which keeps weights packed at 4 bits in HBM
and runs the math at bf16 rate. This is verified in vLLM's source rather than inferred:
`MarlinExpertsBase._supports_current_device()` returns `has_device_capability((7, 5))`, and
`_supports_activation()` includes `MoEActivation.SITU`, which is the unusual SiTU-GLU activation K3 needs
([marlin_moe.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py)).
The backend oracle states the rule in its own docstring: "SM100+ prefers DeepGEMM FP4 / TRTLLM MXFP8;
**SM90 falls through to Triton_unfused or Marlin**"
([oracle/mxfp4.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py)).
SGLang says the same in one sentence: "the FlashInfer MXFP4 (trtllm-gen SiTU) runner serves them on
Blackwell, **Marlin (W4A16) elsewhere**", and "H100/H200 pin Marlin"
([cookbook](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3)). The KDA kernels are fine
on Hopper as well: `kda.py` accepts SM90, SM10x and SM12x and otherwise raises "FlashKDA requires CUDA
SM90/SM10x/SM12x, bfloat16"
([kda.py](https://github.com/vllm-project/vllm/blob/main/vllm/models/kimi_k3/nvidia/kda.py)). NVIDIA also
funded Hopper-specific kernel work: the blog credits "Shikhar for Flash-Flash-KDA", described as
optimizing "the kernels for H100".

The sharpest point, and the one that tells us what would have to change: **Hopper is limited by K3's SiTU
activation, not by the 4-bit format.** Two other MXFP4 paths are device-capable on sm_90 and are excluded
on activation grounds alone. FlashInfer's mixed-input CUTLASS GEMM advertises `(kMxfp4Static, None)` under
`is_device_capability(90)`, commented "wmxfp4a16 on 9.0", but its `_supports_activation` allows only SILU,
GELU_TANH, RELU2_NO_MUL and SWIGLUOAI, so K3's `hidden_act: situ` is filtered out. And the Triton
`matmul_ogs` kernel that made **gpt-oss** fast on H100 with the same MXFP4 format does not transfer either,
because `TRITON` is simply not in K3's backend priority list, with `TRITON_UNFUSED` commented out as having
"bug with MTP support". So the fix is a SiTU-capable sm_90 MXFP4 kernel, which is exactly what SGLang's
Humming port is, rather than anything to do with 4-bit storage.

Two consequences matter more than the Hopper question itself.

**Our RTX nodes do not get the fast path either.** The trtllm-gen MXFP4 kernel, the tcgen05 latent-MoE
tail fusion, and the fused AttnRes kernel are all gated on `is_device_capability_family(100)`, which is
B200 and B300 only. That helper compares `capability // 10`, so sm_120 returns False
([interface.py](https://github.com/vllm-project/vllm/blob/main/vllm/platforms/interface.py),
[latent_moe_tail.py](https://github.com/vllm-project/vllm/blob/main/vllm/models/kimi_k3/nvidia/ops/latent_moe_tail.py)).
Our sm_120 cards get `DEEPGEMM_MXFP4` at best, which does allow family 120 with W4A8 FP8 activations
([deep_gemm_moe.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py)),
and Marlin otherwise. The claim above that "Blackwell runs MXFP4 natively through the FlashInfer
trtllm-gen path" holds for B200 and B300 and **not** for RTX PRO 6000.

**Marlin costs about 5 percent in weight bytes.** It rounds the MoE intermediate size up to 128 and the
hidden size up to 256 and then repacks, so resident weights exceed the 1561 GB on disk. A measured
deployment reports **102.75 GB of weights per GPU at TP16**
([HF discussion #136](https://huggingface.co/moonshotai/Kimi-K3/discussions/136)), and the recipes page
estimates `vram_minimum_gb: 1680`. Every calculation below uses those numbers rather than dividing
1561 GB, because dividing the on-disk size understates the requirement and is what made 2 RTX nodes and
3 H200 nodes look feasible in the estimates above.

##### Evidence, separated by strength

**Measured by someone, on Hopper:**

- **2 nodes x 8 H200 141GB, SGLang.** `lmsysorg/sglang:kimi-k3`, TP16 + EP16, Marlin MoE backend.
  102.75 GB weights per GPU, 7.95 GB bf16 KV pool giving 308,608 tokens, cold start 13 min 02 s of which
  11.2 min is weight loading, TTFT 0.37 to 0.54 s, **16.8 tok/s single request**, about **147 tok/s
  aggregate at 54 concurrent requests**. Two operational fixes were required and both apply to us: pin
  `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` per rank, and set
  `TRITON_CACHE_DIR=/tmp/triton-cache-rank$RANK`, because 16 ranks racing on a shared filesystem Triton
  cache throw `FileNotFoundError` during CUDA graph capture. The same author reports the vLLM `kimi-k3`
  image **failed** on this hardware: weights measured 130.2 GB per GPU at TP16 and 127.3 GB at TP32
  against an expected 98 and 49, so per-GPU weights do not shrink as TP grows, which is a replication
  bug rather than tight memory; KV came up 1 to 2 GB short even at `--gpu-memory-utilization 0.97
  --max-model-len 65536`; `VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=1` hard-errors with "K3 latent-MoE tail
  fusion requires SM100."; and TP8 with PP4 crashed in graph capture with "cp_world_size must be
  positive". ([HF discussion #136](https://huggingface.co/moonshotai/Kimi-K3/discussions/136))
- **4 nodes x 8 H200, SGLang 0.5.16, TP32/EP32, Marlin plus FlashMLA, by an SGLang maintainer.** The most
  important Hopper datapoint, because it is the only one with speculative decoding on. Decode output at
  batch 8: **674.94 tok/s with DSpark and bf16 KV, against 239.71 tok/s without DSpark**, an acceptance
  length of 6.08. It also carries a trap that contradicts the advice above: switching only the KV dtype to
  FP8 E4M3 **collapses** DSpark throughput to 173.32 tok/s, a 74.3 percent regression, while making no
  difference without DSpark (240.57 versus 239.71). So `--kv-cache-dtype fp8` is a good idea for K3 in
  general and a bad idea on Hopper the moment DSpark is enabled, at least until this is fixed. Its
  reproduction command also shows the real env set: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
  `SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0`, `SGLANG_K3_ATTN_RES_MODE=jit`,
  `SGLANG_MOE_FUSED_GATE_RADIX=1`, plus `--mem-fraction-static 0.88 --mamba-full-memory-ratio 0.040
  --mamba-radix-cache-strategy extra_buffer_lazy --max-running-requests 32`.
  ([issue #32938](https://github.com/sgl-project/sglang/issues/32938))
- **4 nodes x 8 H100 80GB, SGLang, TP32/EP32, Marlin plus FlashMLA.** This is the only measured H100 run
  I found. It served at the full 1M context, came up 11 minutes after launch, and reached **5.8 tok/s
  single stream and 14.0 tok/s aggregate at 4-way concurrency** with no speculative decoding. It also
  confirms the checkpoint at 1,560,936,091,448 bytes across 96 shards, matching our copy.
  ([Hyperstack tutorial](https://www.hyperstack.cloud/technical-resources/tutorials/deploy-kimi-k3-on-gpu-cloud-for-multi-node-2.8t-inference))
- **2 nodes x 8 H20 (SM90), SGLang PR #32778.** Full 1319-example GSM8K at PP1/TP16/EP16 with DSpark and
  FP8 KV cache. The Marlin baseline scores 1288/1319 = 0.97650; the proposed Humming W4A8 path matches it
  at 0.97650 while cutting TTFT 19.6 to 35.8 percent and TPOT 4.5 to 17.4 percent. The PR is **open** and
  its CI is failing, and the author labels the speed numbers "trend results, not publication-grade".
  ([PR #32778](https://github.com/sgl-project/sglang/pull/32778))
- **16x H200, `vessl/Kimi-K3-W4AFP8`.** Against the native MXFP4 base with the checkpoint as the only
  changed variable: output throughput 277.7 to 327.5 tok/s (+17.9 percent), TTFT p50 0.924 to 0.611 s,
  TPOT p50 23.21 to 21.32 ms. ([repo](https://huggingface.co/vessl/Kimi-K3-W4AFP8))

**Claimed or implied, not measured:**

- The vLLM blog's TL;DR now claims "NVIDIA (**Hopper** and Blackwell) and AMD (MI355X) support at
  launch", but the FAQ in the same document still answers the GPU question with "At least one 8x B300 (or
  GB300 NVL72) node is required; 16x B200 is also supported", and the 118 and 370 tok/s headline figures
  are explicitly "on 16 NVIDIA GB300 NVL72 GPUs". So the blog asserts Hopper support without publishing a
  single Hopper configuration or number. ([blog](https://vllm.ai/blog/2026-07-27-k3))
- SGLang's cookbook publishes H200 2x8 and H100 4x8 cells but closes its platform table with "**No cell
  has a serving round in this exact shape, treat them as starting points to verify**". Those hardware
  entries are engineering intent. ([cookbook](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3))
- Moonshot's own eval footnotes mention "H20 GPUs", but **this is not evidence that Moonshot serves K3 on
  Hopper**, and I initially misread it. Both footnotes describe the *benchmark environment*, the GPUs the
  agent's own tasks run on inside SWE-Marathon and PostTrainBench, not the inference fleet
  ([model card](https://huggingface.co/moonshotai/Kimi-K3)). The 47-page tech report names no chip for
  training or serving anywhere. The genuine adjacent evidence is different and better: Moonshot's own
  [FlashKDA](https://github.com/MoonshotAI/FlashKDA) repo ships a `BENCHMARK_H20.md` headed "KDA forward
  benchmark (Hopper / H20)" and requires SM90 or above, so Moonshot demonstrably develops and benchmarks
  the KDA kernels on Hopper. That predates K3's release and is a kernel benchmark, not K3 serving.
- Red Hat publicly pitched `Kimi-K3-FP8-BLOCK` at Hopper on the grounds that "FP8 is native to Hopper's
  tensor cores", but the checkpoint's own model card contains no Hopper mention and no throughput, and at
  2820 GB it needs 24 to 32 Hopper GPUs for weights alone. A vendor pitch, not a result.

- TokenSpeed, the third engine Moonshot's card recommends, does **not** support Hopper for K3. Its day-0
  post lists only "NVIDIA (G)B200/(G)B300 and AMD MI350X/MI355X", and its recipe says "The fused MoE path
  needs a Blackwell GPU (B200/B300); on other NVIDIA platforms use `--moe-backend triton`", which is a
  fallback rather than a validated path.
  ([blog](https://lightseek.org/blog/tokenspeed-kimi-k3.html), [recipes](https://lightseek.org/tokenspeed/recipes/models#kimi-k3))

**No evidence found:** any successful vLLM run of the *official MXFP4 checkpoint* on Hopper. Every Hopper
success on record is SGLang. The nearest vLLM counterexample is a PR that loaded K3 on 8x H200 at TP8, but
against `unsloth/Kimi-K3-GGUF UD-Q2_K_XL` at 4096 context with bf16 activations
([PR #50404](https://github.com/vllm-project/vllm/pull/50404)), so the K3 *architecture* runs in vLLM on
Hopper while the real checkpoint does not. No upstream issue has been filed for the vLLM H200
weight-loading bug, which means it is not being tracked and may not be fixed soon. Coverage gap to be
honest about: reddit.com is blocked to our tooling, so r/LocalLLaMA was not searched; that is an access
failure rather than a confirmed absence. Note also a hard constraint that shapes everything below: tensor
parallelism must divide both the 96 attention heads and the 7168 hidden size, whose greatest common
divisor is 32, so **the only legal TP sizes are 1, 2, 4, 8, 16 and 32**
([Hyperstack](https://www.hyperstack.cloud/technical-resources/tutorials/deploy-kimi-k3-on-gpu-cloud-for-multi-node-2.8t-inference)).
That rules out TP12 and TP24 outright, independent of memory.

##### Candidate checkpoints

Sizes are decimal GB summed from Hugging Face blob listings. "Helps on Hopper" means helps us on H200 or
H100 specifically.

| Repo id | Size | Precision | Helps on Hopper? |
|---------|------|-----------|------------------|
| `moonshotai/Kimi-K3` | 1561.0 GB | MXFP4 W4A16, compressed-tensors, group 32 | **Yes, and measured.** Runs via Marlin. The reference case. |
| `vessl/Kimi-K3-W4AFP8` | 1518.5 GB | INT4 group 128 experts + static per-tensor FP8 E4M3 activations | **Best Hopper option.** Smaller than base, measured faster, SGLang only. |
| `RedHatAI/Kimi-K3-FP8-BLOCK` | 2820.0 GB | FP8 E4M3 block 128; attention left unquantized | **No.** 1.8x the base and beyond our whole fleet. |
| `PatronusAI/kimi-k3-nvfp4` | 1646.1 GB | NVFP4, bit-exact cast from MXFP4 | Runs, but as W4A16 Marlin anyway and 85 GB larger. No gain. |
| `RedHatAI/Kimi-K3-NVFP4` | 1646.2 GB | NVFP4 | Blackwell oriented, and still needs open vLLM PR #50500. |
| `GrEarl/Kimi-K3-NVFP4A16-Transcoded` | 1646.2 GB | NVFP4A16, lossless transcode | Same as above. |
| `mgoin/Kimi-K3-pruned50` | 837.1 GB | MXFP4, 448 of 896 experts | Fits 2 RTX nodes. Quality regression expected. |
| `lovedheart/Kimi-K3-Lite` | 630.3 GB | MXFP4, 320 of 896 experts, AIMER pruning | Fits 1 RTX node. Unevaluated. |
| `mgoin/Kimi-K3-pruned75` | 475.2 GB | MXFP4, 224 experts | Fits 1 RTX node. Heavier regression. |
| `unsloth/Kimi-K3-GGUF` | 594 to 1509 GB | GGUF IQ1_S through Q8 | No. llama.cpp only, and K3 is in no llama.cpp release. |
| `inference-optimization/Kimi-K3-0.40B-MXFP4` | 1.5 GB | MXFP4, 8 layers, 8 experts | Not a real model. A build smoke test, see below. |
| `Inferact/Kimi-K3-DSpark` | 7.1 GB | BF16 draft, `K3DSparkModel`, 3.56B | vLLM draft. Needs `pp_size == 1`. |
| `RadixArk/Kimi-K3-DSpark` | 4.5 GB | BF16 draft, `DSparkDraftModel`, 2.25B | SGLang draft. Needs `pp_size == 1`. |

The ones that look tempting and are not:

- **There is no smaller official K3.** Moonshot has exactly one K3 repo. No Instruct, Base, Thinking,
  mini, air, flash, or distilled sibling exists, and thinking is always on in the single checkpoint
  ([moonshotai](https://huggingface.co/moonshotai)). **No AWQ, GPTQ, or INT8 build exists** from any of
  the usual publishers.
- **Dropping the vision tower buys nothing.** MoonViT-V2 is 401M params, about 0.9 GB of 1561 GB. The
  switch exists, `--language-model-only`, and the recipes page exposes it as the `text_only` feature, but
  the win is skipped preprocessing rather than memory.
- **The pruned variants are unvalidated.** `lovedheart/Kimi-K3-Lite` says outright "**The weights have
  not been fine-tuned or evaluated, use with caution**", yet its card reproduces Moonshot's full
  benchmark table verbatim, so those scores describe the unpruned model
  ([card](https://huggingface.co/lovedheart/Kimi-K3-Lite)). `mgoin/Kimi-K3-pruned50` is the better bet:
  it comes from a vLLM maintainer, reports GSM8K 68.23, and warns to "expect substantial quality
  regression". Treat all of them as unknown quality until we measure them.
- **NVFP4 cannot beat the base checkpoint here.** Patronus is explicit that NVFP4's accuracy advantage is
  only reachable by re-quantizing from BF16, which for 2.8T params means materializing about 5.6 TB first
  ([patronus-ai/kimi-k3-nvfp4](https://github.com/patronus-ai/kimi-k3-nvfp4)).

##### Conversion tooling, and why the FP8 idea fails

There is **no** conversion or quantization script in the checkpoint or in Moonshot's GitHub repo. Our
local copy contains only modeling and tokenizer code (`modeling_kimi_k3.py`, `configuration_kimi_k3.py`,
`encoding_k3.py`, `kimi_k3_processor.py`, `media_utils.py`, `tokenization_kimi.py`), and
[github.com/MoonshotAI/Kimi-K3](https://github.com/MoonshotAI/Kimi-K3) holds four files: LICENSE,
README.md, the logo, and the tech report PDF. There is nothing like DeepSeek's
`inference/convert.py --expert-dtype fp8`.

The equivalent tool lives in llm-compressor instead, as `model_free_ptq` with a
`CompressedTensorsDequantizer`, added in **open** PR
[#2978](https://github.com/vllm-project/llm-compressor/pull/2978) along with K3 examples for `FP8_BLOCK`,
NVFP4 and MXFP4; `scheme=` is a free parameter, so it can emit W4A16 or INT8 too. But we already know
what the FP8 route costs, because `RedHatAI/Kimi-K3-FP8-BLOCK` is that script's output: **2820 GB**.
Converting the experts to FP8 is what would give Hopper native math, and it also puts the model out of
reach of every configuration we own. That is the central trade of this whole investigation, and it does
not go our way. The base MXFP4 checkpoint on Marlin is the better Hopper answer, and `vessl`'s W4AFP8 is
better still because it gets FP8 *activations* without doubling the *weights*.

##### Per hardware target

Marlin resident weights are 102.75 GB per GPU at TP16, measured. Because that report's units are ambiguous
between GB and GiB, the safe form is the ratio: at TP16 the weights occupy **about 73 percent of one
H200**. Do not assume that halves at TP32. It does not, because the quantization config's `ignore` list
keeps `self_attn`, `shared_experts`, the dense `mlp`, `lm_head`, `vision_tower` and `mm_projector` in
bf16, and those are **replicated on every rank**. The measured totals show it: 102.75 GB x 16 = 1644 GB at
TP16, but 59.63 GB x 32 = 1908 GB at TP32
([Hyperstack logs](https://www.hyperstack.cloud/technical-resources/tutorials/deploy-kimi-k3-on-gpu-cloud-for-multi-node-2.8t-inference)).
Per-GPU cost falls with TP, but sublinearly, and total memory rises. That is also why our 2-RTX-node case
below is settled by a direct TP16-to-TP16 comparison rather than by any scaling rule.

**2 RTX PRO 6000 nodes, 16 GPUs, sm_120. Not viable for the base checkpoint.** Each card has 97887 MiB
against the H200's 143771 MiB, which is 68 percent of an H200, while weights alone need about 73 percent
of an H200 per GPU at TP16. They do not fit even at utilization 1.0, before any KV cache, activation
buffer or CUDA graph. Upstream's own estimate agrees from the other direction: `vram_minimum_gb` is 1680,
and 16 cards give 1650 GB. The 1650 versus 1561 comparison in the estimates above is exactly the trap,
since the on-disk size is the wrong number to compare against. These nodes are instead the right home
for a **pruned** variant, and the arithmetic there is comfortable rather than marginal:

| Checkpoint | Shape | Per GPU with 5 percent Marlin padding | Experts per rank |
|------------|-------|----------------------------------------|------------------|
| `mgoin/Kimi-K3-pruned50`, 837 GB | 2 nodes, TP16 | 55 GB of 103 | 448 / 16 = 28 |
| `lovedheart/Kimi-K3-Lite`, 630 GB | 1 node, TP8 | 83 GB of 103 | 320 / 8 = 40 |
| `mgoin/Kimi-K3-pruned75`, 475 GB | 1 node, TP8 | 62 GB of 103 | 224 / 8 = 28 |

TP8 and TP16 are both legal sizes, and all three expert counts divide evenly by their EP size, so none of
these hits the divisibility walls that kill TP12 and TP24.

**3 H200 nodes, 12 GPUs. Not viable, for two independent reasons.** First, **TP12 is illegal**: 7168 is
not divisible by 12, and the legal TP sizes are 1, 2, 4, 8, 16, 32. Second, even if it were legal, the
weights would need `73 x 16 / 12`, about 97 percent of each GPU, leaving nothing for KV or activations,
when upstream's Hopper profile already asks for `--gpu-memory-utilization 0.97` at TP16. EP12 fails too,
since 896 experts is not divisible by 12 and the Hopper recipes pin EP equal to TP. The estimate above
that 3 H200 nodes fits "with about 65 GB of slack" was computed from the on-disk size and does not survive
Marlin's padding. This is the cleanest correction in this survey: 3 nodes was never a candidate.

**4 H200 nodes, 16 GPUs. The only viable target for the base checkpoint, and it hinges on DSpark.** Same
GPU model and same GPU count as the measured 2x8 deployment: 102.75 GB of weights against 150.8 GB per
card, so at `--mem-fraction-static 0.85` a 128.2 GB budget leaves about 25 GB per GPU for KV, activations
and graphs. K3's KV is tiny, so context is cheap here. **Do not reach for `--kv-cache-dtype fp8` as a
reflex**, though: it is free without speculative decoding and costs 74 percent of decode throughput with
DSpark on Hopper (see the maintainer's measurement above). Keep bf16 KV until that is fixed.

What to expect, and the honest uncertainty:

1. Plain Marlin decode is slow. TP16 without spec decoding measured 16.8 tok/s single stream. DSpark
   measured a 2.8x gain on this exact Hopper Marlin path at TP32, and DSpark is legal for us because our
   only viable shape is pure TP16 where `pp_size == 1` holds. So the realistic target is tens of tok/s,
   not 16.8 and not the 370 tok/s from the Blackwell headline. We cannot narrow it further from published
   data, because nobody has published a TP16 Hopper run with DSpark on.
2. Our nodes have 4 GPUs, so TP16 and EP16 span **four** boxes instead of two, roughly doubling the
   cross-node hops on every all-reduce and all-to-all relative to the 2x8 run we are extrapolating from.
   Our nearest local reference, Kimi-K2.7-Code at about 29 tok/s, was already a 2-node H200 job.
3. On vLLM specifically the context story collapses: its Hopper profile sets `--max-model-len 32768` and
   `--max-num-seqs 5`, against Blackwell's 1048576 and prefix caching. SGLang's measured Hopper run kept a
   308,608-token pool, so the 1M-context selling point survives on SGLang and not on vLLM today.
4. It consumes all four H200 nodes, so Qwen3-235B and everything else on H200 stops while it runs.

**H100, `kempner_h100`, 80 GB per GPU. Works, and is not worth it.** This is the one target where we have
a measured end-to-end result rather than a projection, and the result is the argument against it: 32 H100
GPUs at TP32/EP32 served the full 1M context at **5.8 tok/s single stream and 14.0 tok/s at 4-way
concurrency**
([Hyperstack](https://www.hyperstack.cloud/technical-resources/tutorials/deploy-kimi-k3-on-gpu-cloud-for-multi-node-2.8t-inference)).
Memory is not the constraint; allocation size and speed are. At 4 GPUs per node, TP32 is **8 nodes**, and
TP16 on 4 nodes would put about 103 GB of weights on an 80 GB card, so 32 is the floor. That matches the
recipes page, which sets `strategy_min_gpus` for `h100` to 32 and does not list h100 as verified at all,
and SGLang's cookbook, which calls H100 the platform with "least post-weight headroom (80 GB)" and notes
it needs an "SM90a build of the K3 image". Six nodes is not a way out either: TP24 is illegal because
7168 is not divisible by 24. The recipes page also marks `multi_node_dep` unsupported on both h100 and
h200, so the `deep_gemm_mega_moe` advice does not apply to us.

##### DSpark speculative decoding

Both drafts are confirmed and they are **not** interchangeable. `Inferact/Kimi-K3-DSpark` is 7.1 GB,
3.56B params, arch `K3DSparkModel`, MLA-native, and is the vLLM one. `RadixArk/Kimi-K3-DSpark` is 4.5 GB,
2.25B params, arch `DSparkDraftModel`, GQA, and is the SGLang one. Moonshot publishes neither.

The pipeline-parallelism caveat above is confirmed from two directions. The recipes page lists
`spec_decoding.strategies` as `single_node_tp`, `multi_node_tp`, `multi_node_tep`, `multi_node_dep`,
`multi_node_tp_dp` and `pd_cluster`, and **omits `multi_node_tp_pp`**. SGLang disables DSPARK on its
long-context recipes because "**DSPARK currently requires `pp_size == 1`**". vLLM `main` no longer carries
an explicit PP guard in `speculative.py` and has an open, empty-bodied PR
[#50138](https://github.com/vllm-project/vllm/pull/50138) titled "Codex/kimi k3 dspark pp", so the
combination is being built rather than either working or forbidden. None of this hurts us, because our
only viable shape is pure TP16 where `pp_size == 1` holds anyway.

**DSpark is the difference between K3 being worth serving on Hopper and not.** The 3.14x figure quoted
above is a GB300 number, but Hopper now has its own: 674.94 against 239.71 output tok/s at batch 8 on
32 H200, a 2.8x gain on the Marlin path, at acceptance length 6.08
([issue #32938](https://github.com/sgl-project/sglang/issues/32938)). Because Marlin decode is the weak
part of the Hopper path, speculation is worth proportionally more to us than it is on Blackwell, not less.

Two cautions remain. SGLang still warns "**No serving round on the final draft checkpoint has landed,
measure against the same recipe running NOSPEC before adopting**", so run both arms. And `RadixArk`'s card
notes the draft was trained at 4,096 context only, so acceptance should fall in exactly the long agentic
sessions we care about, which is the one place the 6.08 acceptance length may not hold for us.

##### Recipes people are actually running, verbatim

vLLM's own **Hopper** override block, which is the most useful single find here
([recipes.vllm.ai](https://recipes.vllm.ai/moonshotai/Kimi-K3)):

```
--gpu-memory-utilization 0.97 --max-num-seqs 5 --max-model-len 32768 --moe-backend marlin
--disable-custom-all-reduce --no-enable-flashinfer-autotune --max-num-batched-tokens 4096
--attention-backend FLASHMLA
env: VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_RUST_FRONTEND=1
     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Its Blackwell counterpart, for contrast, shows what we would be giving up: `--load-format fastsafetensors
--no-enable-flashinfer-autotune --max-model-len 1048576 --kv-cache-dtype fp8 --attention-config
'{"use_prefill_query_quantization":true}' --enable-prefix-caching`, with
`VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=1` and `VLLM_ALLREDUCE_USE_FLASHINFER=1`. Base args for both are
`--trust-remote-code --moe-backend auto --gpu-memory-utilization 0.95`.

SGLang, H200 2x8, from the cookbook's generator:

```
--trust-remote-code --model-path moonshotai/Kimi-K3 --tp-size 16 --ep-size 16
--moe-runner-backend marlin --decode-attention-backend flashmla --enable-symm-mem
--mem-fraction-static 0.85 --reasoning-parser kimi_k3 --tool-call-parser kimi_k3
```

H100 4x8 is the same with `--tp-size 32 --ep-size 32 --dist-timeout 3600` and no `--enable-symm-mem`.
Multi-node adds `--nnodes N --node-rank {{NODE_RANK}} --dist-init-addr {{NODE0_IP}}:20000`, and the
Hopper note says to export the cross-node NIC (`GLOO_SOCKET_IFNAME`, `NCCL_SOCKET_IFNAME`,
`SGLANG_HOST_IP`) and keep `NCCL_MNNVL_ENABLE=1 NCCL_CUMEM_ENABLE=1`. DSpark layers on as
`--speculative-algorithm DSPARK --speculative-draft-model-path RadixArk/Kimi-K3-DSpark
--speculative-dspark-block-size 7`.

vLLM's DSpark config, verbatim from the blog and the recipes page:

```
--speculative-config '{"model":"Inferact/Kimi-K3-DSpark","method":"dspark","num_speculative_tokens":7,"attention_backend":"FLASHINFER_MLA","draft_sample_method":"probabilistic","rejection_sample_method":"block"}'
```

Other flags worth knowing before we try anything:

- **Prefix caching is off by default for K3** while the hybrid cache design settles, so
  `--enable-prefix-caching` must be passed explicitly. That matters a lot for agentic reuse.
- **All-to-all backend:** `flashinfer_nvlink_one_sided` for NVLink, `deepep_v2` for RDMA. A 4-node H200
  job is RDMA across nodes, so `deepep_v2`. The `flashinfer_trtllm` MoE backend the blog recommends for
  TP > 1 is Blackwell only, so on Hopper the answer is `marlin`.
- **Do not try `--moe-backend humming` in vLLM.** The Humming experts class omits `MoEActivation.SITU`
  from its supported list, so it will be rejected for K3 even though SGLang's Hopper Humming port exists
  ([fused_humming_moe.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py)).
- **Tool calling is not yet dependable.** The blog says plainly: "We've occasionally seen K3 emit a
  tool-call format its own parser does not expect, yielding an empty `tool_calls` result". For an agentic
  coding endpoint that is a first-order problem, not a footnote.
- **Never set `VLLM_ENABLE_K3_LATENT_MOE_TAIL_FUSION=1` on Hopper.** It is SM100 only and raises
  `ValueError: K3 latent-MoE tail fusion requires SM100.` The recipes page correctly leaves it out of the
  Hopper block and sets it in the Blackwell block, so it is easy to copy the wrong profile.
- **Driver, and the recipes page addresses our exact situation.** Its prerequisites read verbatim: "The
  kimi-k3 image ships as a CUDA 13 (cu130) build only, there is no -cu129 tag, and the K3-enabled wheels
  are not on the cu129 nightly index. The host needs an r580+ NVIDIA driver; **on a CUDA 12.9 (r575) host,
  upgrade the driver or build vLLM from the K3 branch against cu129 PyTorch yourself.**" That is precisely
  our H200 configuration, driver 575 on CUDA 12.9, and vLLM `main` now pins `torch==2.13.0`,
  `nvidia-cutlass-dsl[cu13]` and `humming-kernels[cu13]`
  ([requirements/cuda.txt](https://github.com/vllm-project/vllm/blob/main/requirements/cuda.txt)), so a
  from-source cu129 build is not a small job
  ([CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/)). SGLang's prebuilt
  `lmsysorg/sglang:kimi-k3-cu12` avoids the whole problem, which is the strongest practical argument for
  taking the SGLang path on H200.
- **The same page says "At least 8x GB300" under Hardware while its own badge marks h200 verified.** Read
  the badge as "someone got it up", not as a supported configuration.

##### Corrections to the estimates above

- 0.26.0 does **not** contain K3; the section above expects it to. `main` or a container is required.
- "3 RTX nodes" is not a configuration we have. We have two.
- On RTX PRO 6000 (sm_120), MXFP4 does **not** go through the FlashInfer trtllm-gen path; that is B200
  and B300 only.
- 3 H200 nodes are not a candidate at all: TP12 is illegal, since 7168 is not divisible by 12. Legal TP
  sizes are 1, 2, 4, 8, 16 and 32 only. 2 RTX nodes also do not fit once Marlin padding is counted.
- The 111 to 370 tok/s figures are GB300 NVL72 numbers, specifically 111 at TP8 and 118 at TP16 at batch
  size 1, rising to 331 and 370 with DSpark. The measured Hopper figures without speculative decoding are
  16.8 tok/s on 16 H200 and 5.8 tok/s on 32 H100; with DSpark, 32 H200 reached 674.94 output tok/s at
  batch 8 against 239.71 without. Quote Hopper numbers only alongside whether DSpark was on.
- `--kv-cache-dtype fp8` is recommended for all six models in the options list further down. That advice is
  **wrong for K3 on Hopper with DSpark**, where it costs 74 percent of decode throughput.
- 4 H200 nodes were called "works, but slower". That is right, and it is now measured: about 16.8 tok/s on
  a better topology than ours, at 32K context and 5 concurrent sequences on upstream's Hopper profile.
- The DSpark repo ids in the list above are both real, but they belong to different engines and are not
  interchangeable. Pick by engine, not by size.

##### Recommended next action

Do not attempt the base checkpoint yet. In order:

1. **Validate the code path for free.** Serve `inference-optimization/Kimi-K3-0.40B-MXFP4`, 1.5 GB, 8
   layers, 8 experts, on a single GPU. It is the real `KimiK3ForConditionalGeneration` architecture with
   the real MXFP4 quantization config, so it exercises the registry entry, the KDA layers, the MXFP4 MoE
   method and the `kimi_k3` parsers. It tells us in minutes whether a build works, at essentially no cost.
2. **Build against SGLang's `kimi-k3` branch rather than vLLM.** The only successful Hopper run on record
   is SGLang, the only failed one is vLLM, and SGLang is the only engine shipping a CUDA 12 image that
   matches our H200 driver. The cost is that SGLang is OpenAI-only, so K3 will not serve Claude Code
   until vLLM catches up. That is a real reason to treat this as an experiment rather than an endpoint.
3. **Try a pruned variant on 2 RTX nodes before committing 4 H200 nodes.**
   `mgoin/Kimi-K3-pruned50` at 837 GB fits comfortably, comes from a vLLM maintainer, and has at least one
   published accuracy number. If its quality holds it converts K3 from a fleet-consuming 4-node job into a
   2-node one; if it does not, we learn that cheaply.
4. **Only then consider 4 H200 nodes, and measure NOSPEC against DSpark as the very first thing.** That
   single A/B decides whether this is a 17 tok/s curiosity or a usable endpoint, since DSpark measured
   2.8x on this exact path. Run it with **bf16 KV**, not fp8, given the measured DSpark regression. Then
   compare against what we already have: GLM-5.2-NVFP4 at about 90 tok/s on one RTX node, Kimi-K2.7-Code
   at about 29 tok/s on two H200 nodes. The bar is not "does it run", it is "does it beat those while
   occupying twice the hardware". Note the context story differs by engine: the SGLang run kept a
   308,608-token pool, while vLLM's Hopper profile drops to `--max-model-len 32768`, so on vLLM we would
   lose the long context that is K3's main draw.

**What would have to become true for the base model to be a good option here:** a tagged vLLM release
containing K3 with the H200 weight-loading bug fixed, so we can serve it over the Anthropic API that
Claude Code needs; SGLang PR #32778's Humming W4A8 path merged, since that is the only measured route to
better-than-Marlin speed on Hopper, or `vessl/Kimi-K3-W4AFP8` proving out, since it claims parity at a
smaller footprint; a driver bump on the H200 nodes or a maintained CUDA 12 build; and the tool-call parser
becoming reliable. Absent those, K3 stays a later milestone and the near-term moves are step 1 and step 3.

#### Measured on 4 H200 nodes, 2026-07-31

**It runs, and DSpark is worth 2.16x on single stream but costs 27 percent at concurrency 32.** Served
`moonshotai/Kimi-K3` on 16 H200 GPUs across 4 nodes, TP16 with EP16, MXFP4 through Marlin W4A16, under
`lmsysorg/sglang:kimi-k3-cu12` via Singularity. Deliberately outside the repo recipe format: it needs a
container rather than a venv and SGLang rather than vLLM.

| Concurrency | No speculation | DSpark | Ratio | Per stream, nospec | Per stream, DSpark |
| --- | --- | --- | --- | --- | --- |
| 1 | 40.3 tok/s | 87.1 tok/s | **2.16x** | 40.3 | 87.1 |
| 8 | 245.4 tok/s | 395.7 tok/s | 1.61x | 30.7 | 49.5 |
| 32 | 709.0 tok/s | 516.8 tok/s | **0.73x** | 22.2 | 16.2 |

TTFT median rises with DSpark at every level: 208 to 296 ms at c=1, 213 to 317 at c=8, and 394 to 1553 at
c=32. Draft acceptance ran 2.86 to 3.27 tokens of a 7-token proposal window. Protocol slope(128,1152),
output tokens only, 3 repeats per level, ISL 130, OSL 1152, bf16 KV. Endpoint answered 200 after both
sweeps and returned 401 without a key.

**Read the crossover, not the headline.** Upstream's 3.14x was measured on Blackwell; on Hopper through
Marlin we got 2.16x, and only below roughly 8 concurrent streams. Past that, verification burns compute the
batch already needed, so a shared endpoint is faster without DSpark while a single developer is much faster
with it. The two regimes want opposite flags.

**In context:** 40.3 tok/s single stream beats Kimi-K2.7-Code on 2 H200 nodes (30.4) and with DSpark 87.1
approaches GLM-5.2-NVFP4 on one RTX node (93.4), but it occupies four H200 nodes to do it. K3 is cheaper per
token than its size suggests because 69 of its 93 layers are KDA recurrent rather than full attention.

Five things had to be right, and four of them were not obvious:

| Setting | Why it is mandatory |
| --- | --- |
| `--ep-size 16` | Under pure TP16 each rank gets 3072/16 = 192 of the MoE intermediate dim, and Marlin needs a contraction dim that is a multiple of 128, so `w2` pads to 256. Weights measured **131.62 GB per GPU**, 94 percent of the card, and the KDA state cache could not be allocated at all: `total_rest_memory` came out **negative** at -21.18 GB. With expert parallelism each rank holds whole experts, K stays 3072, and weights drop to **102.75 GB** with 35.42 GB free. Worth 462 GB across the fleet. |
| `--moe-runner-backend marlin` | With `auto`, SGLang selects `Mxfp4MoEMethod`, whose fallback branch calls `upcast_from_mxfp` to dequantize every expert to bf16. It OOMed with 135 of 139.8 GiB per GPU in weights. There is an SM90 path that keeps 4 bits, FlashInfer `cutlass_fused_moe` with `use_w4_group_scaling`, but flashinfer 0.6.15.post1 in this image lacks the `interleave_moe_*_for_sm90_mixed_gemm` helpers, so Marlin is the only 4-bit-preserving option. |
| `TRITON_CACHE_DIR` node-local | It defaults to `~/.triton/cache`, which is NFS home. Sixteen ranks JIT-compiling the same attention metadata kernel into one shared directory raced, and CUDA graph capture died with `FileNotFoundError` on a half-written entry, because Triton's rename-based atomicity does not hold over NFS. SGLang's own hint at that point suggests memory knobs, which would have been the wrong fix. |
| `--weight-loader-disable-mmap` | With mmap, loading ran at about 80 seconds per shard, a 2 hour load, with the node 90 percent idle and 7.7 percent in iowait and loader threads parked in D state. That is mmap paging 1.56 TB in small random reads over a network filesystem. Without it, **8 to 10 minutes**. |
| `--trust-remote-code` | Needed for the tokenizer, not the model: SGLang registers `KimiK3Config` and implements `DSparkDraftModel` itself, but `tokenizer_config.json` maps AutoTokenizer to `TikTokenTokenizer` in `tokenization_kimi.py`. The vocabulary is the local `tiktoken.model`, so this needs no network. |

Also worth recording:

- Ready in **14m 46s** without speculation and **13m 28s** with it, dominated by the weight read.
- KV cache 9.87 GB per GPU in bf16, 383,223 tokens at `--context-length 32768`.
- `max_running_requests` is capped to **67** by the KDA state cache, 338 slots at 5 per request and
  26.78 MB per request, not by KV. Raising concurrency past 67 means raising that, not the KV pool.
- Attention resolves to a hybrid: `fa3` for the 24 full-attention layers, `KDAAttnBackend` for the other
  69. Prefill CUDA graphs are disabled automatically as incompatible with MLA.
- Multimodal is opt-in in SGLang, so omitting `--enable-multimodal` skips the vision tower and sidesteps
  the profiling stall this family caused on two nodes, with nothing lost for coding.
- No `--reasoning-parser` was used, on purpose. K3 always emits reasoning, and a parser moves that text
  into `reasoning_content`, where a first-token measurement watching `content` would miss it.
- The **1.56 TB checkpoint is MXFP4**, contrary to a note made earlier in this session. Verifying that
  needs care: `quantization_config` is nested under `text_config`, not at the top of `config.json`, and
  shard 1 holds the unquantized dense layer, so probing either alone reports bf16. Probe a middle shard
  and expect thousands of `U8` tensors.

Scripts and full logs are outside the repo, at `kimi-k3-test/` on Lustre with `serve_node.sh`,
`run_k3.sh`, `k3.sbatch`, and per-rank logs under `results/`.

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
- Actual tok/s on our hardware. Every throughput number in this document is from other people's GPUs;
  our own reference points are GLM-5.2-NVFP4 at about 90 tok/s on 1 RTX node, and Kimi-K2.7-Code at
  about 21 tok/s on 1 RTX node and about 29 tok/s on 2 H200 nodes.
