# Muse Spark Open 816B-A42B

Muse Spark Open in its native BF16 release build: an 816B-parameter mixture-of-experts, 42B activated per
token, on a hybrid backbone that mixes full and sliding attention. It serves through vLLM's
Anthropic-compatible API as `muse-spark`, so Claude Code connects with no proxy.

The checkpoint carries vision and audio towers alongside the text path. These recipes serve text only.

## Hardware variants

| Variant | Engine | Shape | Single stream | Aggregate | Status |
| --- | --- | --- | --- | --- | --- |
| [`h200-4-nodes4`](h200-4-nodes4/README.md) | vLLM | 4 H200 nodes, 4 GPUs each, TP4 x PP4 | 83.0 tok/s | 9458.8 at c=1024, rising | Validated |

At 1.5 TiB the weights need four H200 nodes, which hold 562 GiB each. The NVFP4 re-quantization is a
separate checkpoint and a separate recipe: it fits one RTX node and is faster for a single caller, but
admits far fewer concurrent requests. See [Muse-Spark-Open-816B-A42B-NVFP4](../Muse-Spark-Open-816B-A42B-NVFP4/README.md).

A DFlash draft model is published for speculative decoding, but it is used by the NVFP4 SGLang recipe, not
by this one: vLLM does not carry a DFlash speculative method.

## Checkpoint provenance

- Checkpoint directory: `Muse-Spark-Open-816B-A42B`
- Hugging Face repo: `meta-models/Muse-Spark-Open-816B-A42B`
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Muse-Spark-Open-816B-A42B`
- Size: 1.5 TiB across 65 shards
- Architecture: `MuseSparkForConditionalGeneration`, native to vLLM 0.33.0, no `--trust-remote-code`

Read from the checkpoint: 62 layers, 256 routed experts with 8 per token, hidden size 8192, 128 attention
heads over 16 key-value heads with `head_dim` 64, `layer_types` mixing `full_attention` and
`sliding_attention`, and `max_position_embeddings` 1048576. The config also carries `vision_config`,
`audio_config` and image, video and audio token ids.
