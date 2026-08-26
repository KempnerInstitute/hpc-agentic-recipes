# Qwen3.8-27B-FP8

The FP8 release build of Qwen3.8-27B: a 27B dense hybrid-attention model with vision and reasoning, 64
layers of which 48 are linear attention and 16 are full attention, and a 262144-token context. It ships an
MTP draft head, is a thinking model with native tool calling, and serves through vLLM's
Anthropic-compatible API as `qwen3.8-27b-fp8`, so Claude Code connects with no proxy.

At 170.6 tok/s for a single caller it is the fastest model in this repo on one GPU, and FP8 weights leave
more room for KV than the BF16 build does.

## Hardware variants

| Variant | Shape | Single stream | Aggregate | Status |
| --- | --- | --- | --- | --- |
| [`h200-1`](h200-1/README.md) | 1 H200 GPU, TP1 | 170.6 tok/s | 4855 at c=1024, saturated | Validated |

28.99 GiB of weights leave 86.58 GiB for KV on a 141 GB card, which holds 9.32 full-length requests at
once against 6.88 for the BF16 build. The BF16 checkpoint is a separate recipe.

## Checkpoint provenance

- Checkpoint directory: `Qwen3.8-27B-FP8`
- Hugging Face repo: `Qwen/Qwen3.8-27B-FP8`
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Qwen3.8-27B-FP8`
- Size: 28.75 GiB across 66 shards
- License: Apache 2.0
- Architecture: `Qwen3_5ForConditionalGeneration`, native to vLLM, no `--trust-remote-code` needed

Quantization is FP8 e4m3 with `weight_block_size` [128, 128] and dynamic activation scaling. The vision
tower is left unquantized.
