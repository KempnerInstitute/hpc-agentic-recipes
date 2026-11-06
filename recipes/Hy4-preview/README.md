# Hy4-preview

A 770B Mixture-of-Experts from Tencent with 49B active per token, Apache 2.0. It serves as
`hy4-preview`. 78 layers, the first dense and the other 77 with 256 routed experts plus 1 shared,
top-8 routed; hidden size 6144, 64 attention heads, vocabulary 120832. Attention is Gated DeepSeek
Sparse Attention with an IndexCache, and the residual path uses identity Hyper-Connections. It ships
a native MTP layer for speculative decoding, 10B more parameters with 0.7B active.

Tencent positions it for long-horizon software engineering, office documents, game prototypes and
scientific work.

## Hardware variants

| Variant | Engine | Shape | Single stream | Aggregate | Status |
| --- | --- | --- | --- | --- | --- |
| [`h200-4-nodes4`](h200-4-nodes4/README.md) | SGLang | 4 nodes, 16 H200 GPUs, TP16 | 48.8 tok/s | 836.3 at c=32, rising | Validated |

836.3 tok/s at concurrency 32 is the highest aggregate of any four-node recipe here, and 48.8 tok/s on
a single stream beats both Kimi recipes on the same hardware.

Four nodes is the only workable shape. The model rejects pipeline parallelism, so parallelism is pure
tensor parallel, and TP must be a power of two dividing the 64 attention heads. Eight GPUs cannot hold
1452.8 GiB; twelve would need TP12 and twenty TP20, neither legal. So it is 16 GPUs or nothing.

**This recipe bind-mounts a patched file into the container.** SGLang's post-load barrier is hardcoded
at 480 s and sixteen ranks loading this checkpoint finish 600 to 700 s apart, which kills ranks that
are loading correctly. The variant's Known limits and its `patches/README.md` carry the measurements
and the reasoning.

## Hardware that does not work

The MXFP8 sibling, `tencent/Hy4-preview-FP8` at 757.9 GiB, has no recipe here. SGLang marks MXFP8
unsupported on H200 because the kernel path requires SM100, and the checkpoint is too large for a
single RTX PRO 6000 node: 757.9 GiB against 764.7 GiB leaves nothing for a KV cache, so it would need
two.

## Checkpoint provenance

- Checkpoint directory: `Hy4-preview`
- Hugging Face repo: `tencent/Hy4-preview`
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Hy4-preview`
- Size: 1452.8 GiB across 131 shards, BF16, plus an 11.6 GiB container under `container/`
- Architecture: `HYV4ForCausalLM`, `model_type` `hy_v4`, declares a 1048576 context
- Verified against the Hugging Face API: every shard's size matches, the index references all 131, and
  a SHA256 spot check matches

No SGLang release carries this model, so the engine is the published `lmsysorg/sglang:hy4-preview`
image. The tool-call and reasoning parsers are both `auto`.
