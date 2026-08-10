# Kimi-K3

A 2.8T-parameter mixture-of-experts model with 104B parameters active per token: 896 experts with 16
routed per token and 2 shared, 93 layers, and a MoonViT-V2 vision tower, so it accepts images as well as
text. Attention is mixed rather than uniform, Kimi Delta Attention on 69 layers and gated MLA on the other
24. It always thinks, returning `reasoning_content` before its answer. It serves through SGLang's
Anthropic-compatible API as `kimi-k3`, so Claude Code connects with no proxy.

The checkpoint is MXFP4 quantization-aware trained rather than post-quantized: the model card records
MXFP4 weights and MXFP8 activations from the SFT stage onward, which is how 2.8T parameters fit in about
1561 GB. Attention, the shared experts, the dense layer's projections, `lm_head`, the vision tower and the
projector stay unquantized. There is no BF16 twin to fall back on.

## Hardware variants

| Variant | Shape | Single stream | Aggregate | Status |
| --- | --- | --- | --- | --- |
| [`h200-4-nodes4-sglang`](h200-4-nodes4-sglang/README.md) | 4 H200 nodes, 16 GPUs, TP16 x EP16 | 40.3 tok/s | 1067 at c=64, capped | Validated |

Measured with protocol slope(128,1152), each configuration swept only to the concurrency its own engine
admits. The recipe carries four configurations, and they trade against each other rather than ranking. The
defaults give 40.3 tok/s single stream and 1067.1 at concurrency 64. `WIDE=1` reaches the highest
aggregate, 1442.6 tok/s at concurrency 156. Speculation with the wide pool gives the highest single stream,
94.1 tok/s, but admits only 48 requests, because speculation takes its state slots from the same budget as
the KV cache.

The request cap, 67 at the defaults and 156 under `WIDE=1`, is set by the KDA state pool rather than by
throughput turning over, so these aggregates are ceilings rather than peaks. Long context costs little:
38.9 tok/s at an input of 115292 tokens, 3.2 percent below the short-prompt rate.

The engine is SGLang in a container rather than vLLM in a virtual environment, because no vLLM release
checked here implements `KimiK3ForConditionalGeneration`, including 0.26.0, the newest this repo installs.
SGLang serves both an Anthropic-compatible `/v1/messages` and an OpenAI-compatible `/v1`, so clients
connect the same way as the vLLM recipes.

The checkpoint declares a 1048576 context. The engine's token pool is far smaller, so the recipe serves
383216.

## Checkpoint provenance

- Checkpoint directory: `Kimi-K3`
- Hugging Face repo: `moonshotai/Kimi-K3`
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Kimi-K3`
- Size: about 1561 GB across 96 shards
- License: Kimi K3 License, MIT with added compliance and attribution conditions
- Architecture: `KimiK3ForConditionalGeneration`, text tower `KimiLinearForCausalLM`

Read from `config.json` and the model card:

| Property | Value |
| --- | --- |
| Parameters | 2.8T total, 104B activated |
| Layers | 93, of which 1 is dense |
| Attention | Kimi Delta Attention on 69 layers, gated MLA on 24 |
| Full-attention placement | every fourth layer plus the final layer, so layers 4, 8, ... 92, 93 |
| Attention dimensions | hidden 7168, 96 heads, `kv_lora_rank` 512, `q_lora_rank` 1536, `v_head_dim` 128 |
| Experts | 896, 16 selected per token, 2 shared, MoE hidden 3072, latent MoE 3584 |
| Activation | SiTU-GLU |
| Quantization | `compressed-tensors`, format `mxfp4-pack-quantized`, `group_size` 32, `num_bits` 4, symmetric, group strategy |
| Modality | text and image, natively multimodal |
| Vision encoder | MoonViT-V2, 401M parameters, patch size 14, 2x2 merge, `patchmergerv2` projector |
| Context | 1048576 |

Multi-turn use has to pass the complete previous assistant message back, `reasoning_content` and
`tool_calls` included, because the model was trained in preserved-thinking-history mode. Thinking effort is
set with a top-level `reasoning_effort` field taking `low`, `high` or `max`, with `max` as the default.
