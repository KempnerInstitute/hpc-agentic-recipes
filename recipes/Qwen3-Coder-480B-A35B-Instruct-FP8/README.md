# Qwen3-Coder-480B-A35B-Instruct-FP8

Qwen3-Coder-480B-A35B-Instruct, FP8 quantized: a mixture-of-experts coding model with 480B total
parameters and 35B active per token (62 layers, 160 experts with 8 routed per token, GQA with 96 query
and 8 key-value heads), a native 262144-token context, and a tool call format built for agentic coding
clients. It emits no reasoning blocks, which is a property of the weights rather than a setting. It
serves through vLLM's Anthropic-compatible API as `qwen3-coder-480b`, so Claude Code connects with no
proxy. It is the largest coding model in this repo that fits a single node.

## Hardware variants

| Variant | Shape | Single stream | Aggregate |
| --- | --- | --- | --- |
| [`rtx-8`](rtx-8/README.md) | 1 RTX PRO 6000 node, 8 GPUs, TP4 x PP2 | 67.7 tok/s | 3238 at c=768, peak |
| [`h200-4`](h200-4/README.md) | 4 H200 GPUs, TP4 | blocked, 22.2 tok/s eager only | not measurable |

Serve this checkpoint on RTX. Two constraints shape both variants. TP8 is impossible: the
`moe_intermediate_size` is 2560 against an FP8 quantization block of 128, so a 320-column shard is not
a multiple of the block and vLLM refuses to start, which is why the RTX node runs TP4 x PP2 rather than
TP8. And on H200 every CUDA graph capture attempt failed in the CUTLASS w8a8 FP8 GEMM path, leaving
only an eager fallback that costs roughly 3x, which is why the Hopper variant exists to record the
failure rather than to be used.

## Checkpoint provenance

- Checkpoint directory: `Qwen3-Coder-480B-A35B-Instruct-FP8`
- Hugging Face repo: `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8`
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Qwen3-Coder-480B-A35B-Instruct-FP8`
- Size: about 482 GB across 49 shards
- License: Apache 2.0
- Architecture: `Qwen3MoeForCausalLM`, native to vLLM 0.25.1

Quantization is FP8 with a dynamic activation scheme and a 128-wide weight block, with `lm_head`, the
router gates and the first three layers' norms left unconverted. The bf16 release of the same model is
a separate, larger checkpoint and is not what these recipes serve.
