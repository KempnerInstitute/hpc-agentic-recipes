# GLM-5.3-NVFP4

The NVFP4 build of GLM-5.3: the same 743B-parameter mixture-of-experts with 39B active per token,
re-quantized to four-bit weights by Inferact rather than by the model author. Four-bit weights are what
make a model this size fit a single node, and on Blackwell they run natively.

It serves through vLLM's Anthropic-compatible API as `glm-5.3`, the same served name the FP8 build uses,
so a client moves between them by changing only the host.

## Hardware variants

| Variant | Engine | Shape | Single stream | Aggregate | Status |
| --- | --- | --- | --- | --- | --- |
| [`rtx-8`](rtx-8/README.md) | vLLM | 1 RTX node, 8 GPUs, TP8 | not measured | not measured | Untested |

There is no Hopper variant of this checkpoint. NVFP4 execution needs Blackwell; on H200 it would fall
back to emulation. The native FP8 build of the same base model is a separate checkpoint with its own
recipe, and at 703.5 GiB it needs two H200 nodes.

## Checkpoint provenance

- Checkpoint directory: `GLM-5.3-NVFP4`
- Hugging Face repo: `Inferact/GLM-5.3-NVFP4`
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/GLM-5.3-NVFP4`
- Size: 432.9 GiB across 88 shards
- Architecture: `GlmMoeDsaForCausalLM`, `model_type` `glm_moe_dsa`, native to vLLM 0.32.0

Read from the checkpoint: `quant_method` modelopt at group size 16, with no excluded modules, against
the FP8 build's 703.5 GiB.
