# Ripple-65B-Preview

A dense 65B transformer from Ricursive Intelligence, pretrained on 12T tokens of predominantly English
text and code, with a hybrid sliding-window and full-attention backbone. It serves as `ripple`.

The release is Apache 2.0.

## Hardware variants

| Variant | Engine | Shape | Single stream | Aggregate | Status |
| --- | --- | --- | --- | --- | --- |
| [`h200-2`](h200-2/README.md) | SGLang | 2 H200 GPUs of one node, TP2 | 40.5 tok/s | 587.7 at c=16, capped | Validated |

At 121 GiB the weights need two H200 GPUs, not one: a single card has about 129 GiB usable, which
leaves no room for the KV cache. Two give a 392,960-token pool at a 262144 context, and per-stream
throughput falls only from 40.5 to 36.7 tok/s out to sixteen concurrent callers.

Neither vLLM nor SGLang ships a recipe for this model, and `RicursiveForCausalLM` is in no vLLM
release. The checkpoint carries its own SGLang implementation under `inference/server`, which this
recipe registers; its Known limits records the two settings that must differ from the vendor's own
defaults on this hardware.

## Checkpoint provenance

- Checkpoint directory: `Ripple-65B-Preview`
- Hugging Face repo: `ricursiveAI/Ripple-65B-Preview`
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Ripple-65B-Preview`
- Size: 121 GiB across 16 shards, plus a 12.7 GiB container under `container/`
- Architecture: `RicursiveForCausalLM`, `model_type` `ricursive`, needs `--trust-remote-code`

Read from the checkpoint: 80 layers, hidden size 5120, 80 attention heads over 8 key-value heads with
`head_dim` 128, a 1024-token sliding window with a hybrid layer pattern, and
`max_position_embeddings` 2097152.

The repository also ships `inference/server`, a reference SGLang deployment: `model.py` implements the
architecture, `ripple_format.py` registers the `ripple` tool-call and reasoning parsers, and
`serve.py` launches the two together. A separate DFlash2 draft,
`ricursiveAI/Ripple-65B-Preview-DFlash2` at 7.5 GiB, is published for speculative decoding but does
not work on H200; see the recipe's Known limits.
