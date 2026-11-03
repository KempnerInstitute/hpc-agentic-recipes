# Muse Glimmer 30B

A dense 29.6B vision-language model distilled from Muse Spark for local deployment: a 52-layer text
decoder plus a ViT-G/14 perception encoder, with a 131072-token trained context. It serves as
`muse-glimmer`.

Unlike the Muse Spark family this is dense rather than mixture-of-experts, and it fits one GPU.

## Hardware variants

| Variant | Engine | Shape | Single stream | Aggregate | Status |
| --- | --- | --- | --- | --- | --- |
| [`h200-1`](h200-1/README.md) | vLLM | 1 H200 GPU, TP1 | 61.0 tok/s | 8925.0 at c=256, rising | Validated |

At 55.5 GiB the weights fit a single H200 GPU with room for a 3.8M-token KV cache, which is 29 full-length
requests at once. Per-stream throughput holds near 55 tok/s all the way to concurrency 64, so one GPU
serves many callers without each one feeling slower.

Upstream also verifies this model on a single GB10 DGX Spark, consumer RTX 5090 and AMD MI300 series
cards, since it is built for local deployment rather than datacenter serving.

## Checkpoint provenance

- Checkpoint directory: `Muse-Glimmer-30B`
- Hugging Face repo: `meta-models/Muse-Glimmer-30B`
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Muse-Glimmer-30B`
- Size: 55.5 GiB across 2 shards
- Architecture: `MuseGlimmerForConditionalGeneration`, dense, native to vLLM 0.28.0 and later

Read from the checkpoint: 52 text layers, hidden size 6656, 2 key-value heads with `head_dim` 128, a
repeating local, local, local, global attention pattern with a 2048-token sliding window, and
`max_position_embeddings` 131072.

A separate DFlash assistant, `meta-models/Muse-Glimmer-30B-assistant` at 4.76 GiB, provides optional
speculative decoding. It is off by default; see the recipe's Known limits.
