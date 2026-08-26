# Qwen3.8-27B

A 27B dense hybrid-attention model with vision and reasoning: 64 layers, of which 48 are linear attention
and 16 are full attention, hidden size 5120, and a 262144-token context. It ships an MTP draft head, is a
thinking model with native tool calling, and serves through vLLM's Anthropic-compatible API as
`qwen3.8-27b`, so Claude Code connects with no proxy.

At 122.1 tok/s for a single caller it is the fastest large model in this repo, and it fits one GPU, which is
the shortest queue wait here.

## Hardware variants

| Variant | Shape | Single stream | Aggregate | Status |
| --- | --- | --- | --- | --- |
| [`h200-1`](h200-1/README.md) | 1 H200 GPU, TP1 | 122.1 tok/s | 3957 at c=1024, rising | Validated |

51.75 GiB of BF16 weights leave 63.95 GiB for KV on a 141 GB card, so the full context fits with 6.88
full-length requests at once. The FP8 build of the same model is a separate checkpoint with its own recipe.

## Checkpoint provenance

- Checkpoint directory: `Qwen3.8-27B`
- Hugging Face repo: `Qwen/Qwen3.8-27B`
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Qwen3.8-27B`
- Size: 51.75 GiB across 18 shards
- License: Apache 2.0
- Architecture: `Qwen3_5ForConditionalGeneration`, native to vLLM, no `--trust-remote-code` needed

The context is 262144 natively and the card describes extending it to 1M with rope overrides, which this
recipe does not use.
