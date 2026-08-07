# GLM-4.6-FP8

A mixture-of-experts reasoning and coding model in FP8: 92 layers, 160 routed experts plus 1 shared with
8 routed per token, and GQA with 96 query and 8 key-value heads. It serves through vLLM's
Anthropic-compatible API as `glm-4.6`, so Claude Code connects with no proxy.

## Hardware variants

| Variant | Shape | Single stream | Aggregate |
| --- | --- | --- | --- |
| [`h200-4`](h200-4/README.md) | 1 H200 node, 4 GPUs, TP4 | 19.1 tok/s | 8127 at c=1024, rising |

FP8 weights are what let this model fit one node, which matters for more than memory: staying
single-node means no pipeline parallelism, and vLLM rejects a speculative config whenever pipeline
parallelism is active. The checkpoint ships a speculative head, `num_nextn_predict_layers` 1, so this
is one of the few recipes here that actually gets MTP speculative decoding.

The aggregate figure is `rising`, meaning throughput was still climbing at concurrency 1024, the top of
the sweep, so 8127 tok/s is a floor rather than a ceiling. Measured with protocol slope(128,1152) over
output tokens only, 3 repeats per level. MTP accepted 89.2 percent of its draft tokens across that sweep.

## Checkpoint provenance

- Checkpoint directory: `GLM-4.6-FP8`
- Hugging Face repo: `zai-org/GLM-4.6-FP8`
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/GLM-4.6-FP8`
- Size: about 361 GB, 336 GiB, across 93 shards
- License: MIT
- Architecture: `Glm4MoeForCausalLM`, native to vLLM

The checkpoint's `max_position_embeddings` is 202752, which the recipe serves in full. The KV pool is
395,392 tokens either way, so the context costs concurrent capacity rather than pool: 1.95 full-length
requests at 202752 against 3.02 at 131072.
