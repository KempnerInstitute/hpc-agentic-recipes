# Qwen3-235B-A22B

A bf16 mixture-of-experts model with 235B total parameters and 22B active per token: 94 layers, 128
experts with 8 routed per token, and GQA with 64 query and 4 key-value heads. It is a thinking model,
and switches between thinking and non-thinking behavior within the same weights. It serves through
vLLM's Anthropic-compatible API as `qwen3-235b`, so Claude Code connects with no proxy.

## Hardware variants

| Variant | Shape | Single stream | Aggregate |
| --- | --- | --- | --- |
| [`rtx-8`](rtx-8/README.md) | 1 RTX PRO 6000 node, 8 GPUs, TP8 | 63.3 tok/s | 3984 at c=512, peak |

The weights are bf16 and unquantized, at about 470 GB, so a whole 8-GPU node is the smallest thing
that holds them. Quantizing to FP8 on load measured no faster, because decode at TP8 on this hardware
is limited by cross-GPU communication rather than weight bandwidth.

## Checkpoint provenance

- Checkpoint directory: `Qwen3-235B-A22B`
- Hugging Face repo: `Qwen/Qwen3-235B-A22B`
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Qwen3-235B-A22B`
- Size: about 470 GB across 118 shards
- License: Apache 2.0
- Architecture: `Qwen3MoeForCausalLM`, native to vLLM 0.25.1

The checkpoint's `max_position_embeddings` is 40960, which is what the recipe serves.
