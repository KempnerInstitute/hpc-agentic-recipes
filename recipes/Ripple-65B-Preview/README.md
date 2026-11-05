# Ripple-65B-Preview

A dense 65B transformer from Ricursive Intelligence, pretrained on 12T tokens of predominantly English
text and code, with a hybrid sliding-window and full-attention backbone. It serves as `ripple`.

The release is Apache 2.0.

## Hardware variants

| Variant | Engine | Shape | Single stream | Aggregate | Status |
| --- | --- | --- | --- | --- | --- |
| [`h200-2`](h200-2/README.md) | SGLang | 2 H200 GPUs of one node, TP2 | 40.5 tok/s | 587.7 at c=16, capped | Validated |
| [`h200-1`](h200-1/README.md) | SGLang | 1 H200 GPU, TP1 | 25.1 tok/s | 188.8 at c=8, capped | Validated |

Pick by context, not by rate. One card serves a real endpoint at 25.1 tok/s, but its KV pool caps the
usable context near 33,000 tokens however high `--context-length` is set, because SGLang sizes the
pool from the memory left after the weights. Two cards give a 392,960-token pool, the full 262144
context, twice the concurrency and 62 percent more single stream throughput. The model fits on one
card at all only because of its hybrid attention: the 1024-token sliding window layers cost far less
KV than full attention layers would.

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
