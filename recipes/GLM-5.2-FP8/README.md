# GLM-5.2-FP8

GLM-5.2 in its FP8 release build: a 744B-parameter mixture-of-experts reasoning and coding model that
uses DeepSeek-style sparse attention for long context. It is a thinking model with native tool calling,
and it serves through vLLM's Anthropic-compatible API as `glm-5.2`, so Claude Code connects with no
proxy.

This is the long-context endpoint in this repo. The checkpoint's `max_position_embeddings` is 1048576,
a million tokens, which no other checkpoint here reaches. The price is speed: at 756 GB the weights need
two H200 nodes, two nodes need pipeline parallelism, and a two-stage pipeline with one request in flight
is the slowest configuration in this repo.

## Hardware variants

| Variant | Engine | Shape | Decode rate | Protocol | Status |
| --- | --- | --- | --- | --- | --- |
| [`h200-4-nodes2`](h200-4-nodes2/README.md) | vLLM | 2 H200 nodes, 4 GPUs each, TP4 x PP2 | about 13 tok/s | single-generation | Untested (migrated) |
| [`h200-4-nodes2-sglang`](h200-4-nodes2-sglang/README.md) | SGLang | 2 H200 nodes, TP8, no PP, EAGLE | never measured | none | Blocked |

Use the vLLM variant. The SGLang variant is documentation only: it has never successfully loaded
weights, and it is kept because it is the one configuration that could use this checkpoint's MTP
speculative head, which vLLM cannot reach here.

There is no single-node H200 variant, and there cannot be one: 756 GB of weights does not fit a 4-GPU
node, which holds 562 GiB. If you want GLM-5.2 on one node, use the separately quantized NVFP4
checkpoint on an RTX node, which measured about 90 tok/s at a 128K context. If you want a smaller model
with the same tool calling and reasoning parsers on a single H200 node, use GLM-4.6-FP8.

## Checkpoint provenance

- Checkpoint directory: `GLM-5.2-FP8`
- Hugging Face repo: `zai-org/GLM-5.2-FP8`, **unverified**
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/GLM-5.2-FP8`
- Size: about 756 GB across 141 shards, 704 GiB
- License: MIT
- Architecture: `GlmMoeDsaForCausalLM`, `model_type` `glm_moe_dsa`, native to vLLM 0.25.1

The repo id is the one unverified fact here. The pre-restructure repo recorded only the local path this
checkpoint was downloaded to, never a Hub id. The id above is inferred from the sibling checkpoints,
`zai-org/GLM-4.6-FP8` and `zai-org/GLM-5.2`, and has not been confirmed against the Hub. Confirm it
before using it in a download command.

Read from the checkpoint on 2026-07-29: 78 layers, 256 routed experts plus 1 shared with 8 routed per
token, `moe_intermediate_size` 2048, `hidden_size` 6144, 64 attention heads, and a sparse attention
indexer with `index_n_heads` 32, `index_head_dim` 128, and `index_topk` 2048. Quantization is FP8
`e4m3` with dynamic activations and `weight_block_size` [128, 128], which is what constrains legal
tensor-parallel sizes. `num_nextn_predict_layers` is 1, so an MTP speculative head ships with the
weights; whether it can be used depends entirely on the engine and the parallelism shape, and each
variant's README says which.
