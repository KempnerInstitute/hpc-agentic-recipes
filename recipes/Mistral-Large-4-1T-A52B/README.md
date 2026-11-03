# Mistral Large 4 1T-A52B

Mistral Large 4 in its native FP8 release build: a 1T-parameter mixture-of-experts, 52B activated per
token, on a DeepSeek-style sparse attention stack. It serves as `mistral-large-4`.

The checkpoint is FP8 as published. An NVFP4 re-quantization is published for Blackwell and is a separate
checkpoint and recipe.

## Hardware variants

| Variant | Engine | Shape | Single stream | Aggregate | Status |
| --- | --- | --- | --- | --- | --- |
| [`h200-4-nodes4-sglang`](h200-4-nodes4-sglang/README.md) | SGLang | 4 H200 nodes, 4 GPUs each, TP16 | 67.1 tok/s | 1517.5 at c=48, capped | Validated |

At 988 GiB the weights need four H200 nodes, which hold 562 GiB each. The recipe runs SGLang rather than
vLLM because the only vLLM build carrying this model ships without Ray and so cannot span nodes; its
Known limits records the exact error.

The NVFP4 build fits one RTX node, is faster for a single caller and serves a shorter context. See
[Mistral-Large-4-1T-A52B-NVFP4](../Mistral-Large-4-1T-A52B-NVFP4/README.md).

## Checkpoint provenance

- Checkpoint directory: `Mistral-Large-4-1T-A52B`
- Hugging Face repo: `mistralai/Mistral-Large-4-1T-A52B`
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Mistral-Large-4-1T-A52B`
- Size: 988 GiB across 54 shards, plus a 12.5 GiB container under `container/`
- Architecture: `MistralLarge4ForConditionalGeneration`, `quant_method` fp8, `weight_block_size` [128, 128]

Read from the checkpoint: 72 layers, hidden size 6144, 64 attention heads over 64 key-value heads with
`head_dim` 64, `max_position_embeddings` 1048576, and a vision encoder. The attention stack uses MLA with
a DSA top-k indexer, NoPE on one layer in four (`rope,rope,rope,nope`), SSMax scaling and cross-layer KV
sharing, which is why only the `dsa` attention backend implements it.

A DSpark drafter ships inside the checkpoint, so speculative decoding needs no separate download and no
draft path.
