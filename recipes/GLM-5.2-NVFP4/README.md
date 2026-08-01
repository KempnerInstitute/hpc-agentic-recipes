# GLM-5.2-NVFP4

GLM-5.2 quantized to NVFP4 by NVIDIA Model Optimizer: a 744B-parameter mixture-of-experts reasoning
and coding model that uses DeepSeek-style sparse attention for long context, at near-FP8 quality. It
serves through vLLM's Anthropic-compatible API as `glm-5.2`, so Claude Code connects with no proxy.

4-bit weights and activations are what make a model this size fit a single node, and on Blackwell they
run natively. This is the fastest large model in this repo; the small Gemma models are faster outright.

## Hardware variants

| Variant | Shape | Single stream | Aggregate |
| --- | --- | --- | --- |
| [`rtx-8`](rtx-8/README.md) | 1 RTX PRO 6000 node, 8 GPUs, TP8 | 93.4 tok/s | 1389 at c=256, peak |

There is no Hopper variant of this checkpoint. NVFP4 execution needs Blackwell; on H200 it would fall
back to emulation. The FP8 build of the same base model is a separate checkpoint with its own recipes.

## Checkpoint provenance

- Checkpoint directory: `GLM-5.2-NVFP4`
- Hugging Face repo: `nvidia/GLM-5.2-NVFP4`, quantized from `zai-org/GLM-5.2`
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/GLM-5.2-NVFP4`
- Size: about 465 GB across 47 shards
- License: MIT, the same as the base model
- Architecture: `GlmMoeDsaForCausalLM`, native to vLLM 0.25.1

Quantization is 4-bit weights and 4-bit input activations at group size 16, with the embeddings,
`lm_head` and the first layer left unquantized.
