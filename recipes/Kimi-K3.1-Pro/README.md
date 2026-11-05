# Kimi-K3.1-Pro

A 1T-class multimodal MoE from Moonshot AI: 93 layers, 896 experts, hidden size 7168, shipped as
`compressed-tensors` in the `mxfp4-pack-quantized` format. It serves as `kimi-k3.1-pro`.

## Hardware variants

| Variant | Engine | Shape | Single stream | Aggregate | Status |
| --- | --- | --- | --- | --- | --- |
| [`h200-4-nodes4-sglang`](h200-4-nodes4-sglang/README.md) | SGLang | 4 nodes, 16 H200 GPUs, TP16 EP16 | 40.4 tok/s | 707.6 at c=32, climbing | Validated |

At 1454.0 GiB the weights need all sixteen GPUs of four H200 nodes. Throughput lands within a few
percent of the sibling [Kimi-K3](../Kimi-K3/README.md) recipe on the same hardware, so the two models
cost the same to serve.

vLLM is not an option here: its recipe for this model excludes H200 and pins a release that does not
exist. SGLang's cookbook gives the Hopper shape as TP16 with EP16 and publishes a day-zero image,
which is what this recipe uses.

## Checkpoint provenance

- Checkpoint directory: `Kimi-K3.1-Pro`
- Hugging Face repo: `moonshotai/Kimi-K3.1-Pro`
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Kimi-K3.1-Pro`
- Size: 1454.0 GiB across 96 shards, plus an 11.8 GiB container under `container/`
- Architecture: `KimiK31ForConditionalGeneration`, `model_type` `kimi_k31`, needs `--trust-remote-code`

The checkpoint declares a 1048576 context; the recipe serves 262144, which is what has been measured.
The tool-call and reasoning parsers are both `kimi_k3`, and unlike the Kimi-K3 recipe this image needs
no reasoning-parser patch.
