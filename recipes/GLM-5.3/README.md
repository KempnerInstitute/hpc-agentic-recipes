# GLM-5.3

GLM-5.3 in its native FP8 release build: a 743B-parameter mixture-of-experts, 39B activated per token,
reasoning and coding model on the same sparse-attention backbone as GLM-5.2. It serves through vLLM's
Anthropic-compatible API as `glm-5.3`, so Claude Code connects with no proxy.

The checkpoint is FP8 as published; there is no separate FP8 build to fetch. BF16 weights live under a
suffixed repo and an NVFP4 re-quantization is published by Inferact for Blackwell.

## Hardware variants

| Variant | Engine | Shape | Single stream | Aggregate | Status |
| --- | --- | --- | --- | --- | --- |
| [`h200-4-nodes2`](h200-4-nodes2/README.md) | vLLM | 2 H200 nodes, 4 GPUs each, TP4 x PP2 | not measured | not measured | Untested |

At 703.5 GiB the weights do not fit one H200 node, which holds 562 GiB, so two nodes and pipeline
parallelism are required exactly as for GLM-5.2-FP8. They do not fit one RTX node either, which holds
765 GiB raw but only 688 at the utilization this repo serves at.

The checkpoint ships a 5-token MTP head. Whether it can be used here is open: this engine version does
not list MTP among the methods it refuses under pipeline parallelism, which the GLM-5.2 generation did,
so it is worth testing rather than assumed.

## Checkpoint provenance

- Checkpoint directory: `GLM-5.3`
- Hugging Face repo: `zai-org/GLM-5.3`
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/GLM-5.3`
- Size: 703.5 GiB across 141 shards
- Architecture: `GlmMoeDsaForCausalLM`, `model_type` `glm_moe_dsa`, native to vLLM 0.32.0

Read from the checkpoint: 78 layers, 256 routed experts plus 1 shared with 8 routed per token, 64
attention heads, MLA with `kv_lora_rank` 512 and `q_lora_rank` 2048, `index_topk` 2048, and
`max_position_embeddings` 1048576.
