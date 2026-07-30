# Kimi-K2.7-Code

A 1T-parameter mixture-of-experts coding model with 32B parameters active per token: 384 experts with 8
routed per token, 61 layers, MLA attention, and a MoonViT vision tower, so it accepts images as well as
text. It is thinking-mode only, always emitting reasoning before its answer, and it is the strongest
coder in this repo. It serves through vLLM's Anthropic-compatible API as `kimi-k2.7-code`, so Claude
Code connects with no proxy.

The checkpoint is natively INT4 quantization-aware trained rather than post-quantized: the routed
experts are compressed-tensors pack-quantized W4A16 at group size 32, while attention, the shared
expert, the dense layers, `lm_head` and the vision tower stay bf16.

## Hardware variants

| Variant | Shape | Decode rate | Protocol |
| --- | --- | --- | --- |
| [`rtx-8`](rtx-8/README.md) | 1 RTX PRO 6000 node, 8 GPUs, TP8 | about 21 tok/s | single-generation |
| [`h200-4-nodes2`](h200-4-nodes2/README.md) | 2 H200 nodes, 4 GPUs each, TP4 x PP2 | about 29 tok/s | single-generation |

Both variants are now recipe directories, and both rates above were measured with the pre-restructure
scripts rather than with the recipes as they now stand.

The two-node H200 configuration is the faster of the two, because its tensor-parallel all-reduces stay
inside a node and cross NVLink, while the RTX node has to use PCIe. The RTX variant wins on cost: one
node instead of two. About 595 GB of weights does not fit a single 4-GPU H200 node, which is why Hopper
forces two.

## Checkpoint provenance

- Checkpoint directory: `Kimi-K2.7-Code`
- Hugging Face repo: `moonshotai/Kimi-K2.7-Code`
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Kimi-K2.7-Code`
- Size: about 595 GB across 64 shards
- License: modified MIT
- Architecture: `KimiK25ForConditionalGeneration`, native to vLLM 0.25.1

vLLM ships its own implementation of this architecture, so the model card's note about a `transformers`
version bound applies only to the Hugging Face reference path, not to serving here.
