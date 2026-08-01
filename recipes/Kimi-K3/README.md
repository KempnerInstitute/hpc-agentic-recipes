# Kimi-K3: blocked for vLLM, measured under SGLang

Status: Blocked - vLLM cannot load `KimiK3ForConditionalGeneration` in any release we run. Served and measured under SGLang in a container, outside this repo.

This is a documentation-only entry: there is no hardware subdirectory and no scripts, because the only
engine that runs this checkpoint is SGLang inside a container, which does not fit the environment build
contract every recipe here follows. It also serves an OpenAI-only API, so Claude Code would need a proxy
rather than the native `/v1/messages` endpoint the vLLM recipes provide.

It does run, though, and the numbers are worth knowing before anyone plans around this model. On 4 H200
nodes at TP16 with EP16 through Marlin W4A16, using `lmsysorg/sglang:kimi-k3-cu12`:

| | |
| --- | --- |
| Single stream | 40.3 tok/s, and 87.1 with the DSpark draft, a 2.16x speedup |
| Saturated | 1392 tok/s at concurrency 96, which is its measured ceiling |
| Concurrency limit | 156 requests, set by the KDA state pool rather than by KV or compute |
| Long context | 38.9 tok/s at a 131,072-token prompt, only 3.5 percent below its short-prompt rate |
| Weights | 102.75 GB per GPU, 69 percent of all HBM across the 16 GPUs |
| Startup | about 20 minutes, dominated by reading 1.56 TB |

DSpark is not a uniform win: 2.16x at concurrency 1 and 1.61x at 8, but 0.73x at 32, because verification
burns compute the batch already needed. Full method, flag derivations and failure modes are recorded
outside the repo, since this is not a recipe.

This page records what the checkpoint is, what vLLM is still missing, what hardware would be needed, and
what to re-check when a newer vLLM lands.

## What is staged

- Checkpoint directory: `Kimi-K3`
- Hugging Face repo: `moonshotai/Kimi-K3`
- Testbed path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Kimi-K3`
- Also on scratch: `/n/netscratch/kempner_dev/Lab/mmsh/models/Kimi-K3`
- On disk: 1453.8 GiB, 1561.0 GB decimal, about 1.5 TiB, in 96 shards

Read from `config.json` and the model card:

| Property | Value |
| --- | --- |
| Architecture | `KimiK3ForConditionalGeneration`, `model_type` `kimi_k3` |
| Text tower | `KimiLinearForCausalLM`, `model_type` `kimi_linear` |
| Parameters | 2.8T total, 104B activated |
| Layers | 93, of which 1 is dense |
| Attention | Kimi Delta Attention on 69 layers, gated MLA on 24 |
| Full-attention placement | every fourth layer plus the final layer, so layers 4, 8, ... 92, 93 |
| Attention dimensions | hidden 7168, 96 heads, `kv_lora_rank` 512, `q_lora_rank` 1536, `v_head_dim` 128 |
| Experts | 896, 16 selected per token, 2 shared, MoE hidden 3072, latent MoE 3584 |
| Activation | SiTU-GLU |
| Quantization | `compressed-tensors`, format `mxfp4-pack-quantized`, `group_size` 32, `num_bits` 4, symmetric, group strategy |
| Left unquantized | attention, shared experts, the dense `mlp` gate, up, gate_up and down projections, `lm_head`, and the vision tower |
| Modality | text and image, natively multimodal |
| Vision encoder | MoonViT-V2, 401M parameters, patch size 14, 2x2 merge, `patchmergerv2` projector |
| Context | 1048576 |

MXFP4 is not a post-training compression here. The card states the model was quantization-aware
trained from the SFT stage onward with MXFP4 weights and MXFP8 activations, which is how 2.8T
parameters fit in 1.5 TiB. That also means there is no bf16 twin to fall back on.

The model always thinks: it returns `reasoning_content`, thinking effort is set with a top-level
`reasoning_effort` field taking `low`, `high`, or `max` with `max` as the default, and multi-turn use
requires passing the complete previous assistant message back, `reasoning_content` and `tool_calls`
included, because it was trained in preserved-thinking-history mode. Any future recipe here has to get
that right in the client configuration, not just in the launch flags.

## The blocker

`KimiK3ForConditionalGeneration` is not implemented by any engine we can install today. Checked
directly rather than from memory:

| Engine | Result |
| --- | --- |
| vLLM 0.25.1, the version this repo runs | absent from the model registry |
| vLLM `main` | absent |
| SGLang v0.5.16 | absent |

In our installed vLLM 0.25.1 the Kimi model files are `kimi_audio`, `kimi_k25`, `kimi_k25_vit`,
`kimi_linear`, and `kimi_vl`, and the registry entries are `KimiK25ForConditionalGeneration` and
`KimiLinearForCausalLM`. There is no `kimi_k3` module and no K3 registry entry. SGLang v0.5.16's model
files cover `kimi_k25`, `kimi_k25_eagle3`, `kimi_linear`, `kimi_vl`, and `kimi_vl_moonvit`, again with
no K3 entry.

**The missing piece is the K3 multimodal wrapper, not the linear attention core.** vLLM already
implements `kimi_linear`, which is the text tower's own `model_type`, and it already carries MXFP4
fused-MoE kernels (`vllm/model_executor/layers/quantization/mxfp4.py`). What does not exist is the
`KimiK3ForConditionalGeneration` model class that wires the vision tower, the projector, and the
93-layer KDA plus gated-MLA stack together. That is a smaller gap than "no support at all", which is
why this is worth revisiting on the next engine release rather than writing off.

Version arithmetic, which is the actual reason nothing can be done right now: the latest vLLM on PyPI
is 0.26.0, and the official Kimi-K3 recipe at `recipes.vllm.ai/moonshotai/Kimi-K3` requires vLLM
0.27.0 or newer. 0.27.0 is unreleased. Installing a nightly is not a shortcut here, because the
architecture is absent from `main` as well.

### How each claim above was checked

Older notes in this repo said Kimi-K3 needed vLLM 0.26.0, which is now wrong, so it is worth being
explicit about where each statement comes from and how to re-check it.

| Claim | How it was checked |
| --- | --- |
| Checkpoint properties, sizes, shard count | read from `config.json`, the model card, and the staged files |
| Absent from vLLM 0.25.1 | grep of the installed registry and model directory in this repo's environment |
| `kimi_linear` and MXFP4 kernels present in 0.25.1 | same grep: `KimiLinearForCausalLM` is registered and `mxfp4.py` exists |
| Absent from vLLM `main` | upstream registry inspected on the web, not reproducible offline |
| Absent from SGLang v0.5.16 | upstream model directory listing on the web; the SGLang installed here is 0.5.11.dev, which also lacks it |
| Latest PyPI vLLM is 0.26.0, official recipe needs 0.27.0 or newer | PyPI and `recipes.vllm.ai` read on the web |
| B300, GB300, or B200 hardware requirement | the official vLLM recipe page for this model |

The rows marked as read on the web are the ones to re-verify first, since they are the ones that change
without anything in this repo changing.

## Hardware, if the software existed

Capacity is not the blocker, and it is worth writing down which of our node types would work, because
the answer is not the one the memory arithmetic suggests.

| Target | GPUs | Raw VRAM | Verdict |
| --- | --- | --- | --- |
| 4 H200 nodes | 16 at 143771 MiB | about 2246 GiB | fits with room to spare, but wrong architecture |
| 2 RTX PRO 6000 nodes | 16 at 97887 MiB | about 1530 GiB | right architecture, does not fit |
| 3 RTX PRO 6000 nodes | 24 at 97887 MiB | about 2294 GiB | right architecture and fits |

Four H200 nodes give about 2246 GiB against 1454 GiB of weights, or 1536 GiB if you round the
checkpoint up to a full 1.5 TiB for headroom, so roughly 710 to 790 GiB is left over. KV is cheap for
this model despite the 1M context, since only 24 of 93 layers hold growing KV and those use MLA
compression while the 69 KDA layers hold a constant-size recurrent state. So by memory alone, four
H200 nodes look like the obvious target.

They are the wrong hardware anyway. MXFP4 is a Blackwell-native format, and the official deployment
guide asks for at least one 8x B300 or GB300 node, or 16x B200; it never mentions H200, H100, or Hopper
at all. On Hopper the MXFP4 experts would need a 4-bit weight-only fallback such as Marlin, at unknown
speed and unverified numerics for a quantization-aware-trained model.

Our RTX PRO 6000 nodes are Blackwell sm_120 and are the architecturally correct hardware, but two of
them give only about 1530 GiB raw, which is roughly 1377 GiB at a realistic
`--gpu-memory-utilization 0.90` and therefore short of the 1454 GiB of weights before any KV cache at
all. Two nodes are about one node short. Three RTX nodes fit comfortably, and that is the shape to plan
for, with the usual multi-node caveats: tensor parallelism inside each node, pipeline parallelism
across them, Ray as the executor, and no speculative decoding, since vLLM rejects a speculative config
when pipeline parallelism is active. The K3 DSpark drafter that upstream reports as a roughly 3x decode
speedup is therefore unavailable to us at three nodes, which is worth knowing before comparing our
future numbers against published ones.

## What to re-check, and when

Revisit when vLLM 0.27.0 is released, and check two independent things rather than one:

1. **Is the architecture registered?** Look for a `kimi_k3` model module and a
   `KimiK3ForConditionalGeneration` registry entry in the release you install, not in the release notes.
   This is the hard blocker.
2. **Is Hopper supported for MXFP4 in that release?** If it is, four H200 nodes become a legitimate
   option and are easier to schedule than three RTX nodes. If it is not, this stays an RTX-only model
   and needs three nodes.

If both come back positive, the next steps are a dedicated environment on the newer vLLM, so the
existing endpoints stay untouched, then a three-node RTX bring-up, then the reasoning and tool-calling
client configuration described above.
