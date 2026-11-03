# Muse Spark Open 816B-A42B NVFP4

The NVFP4 re-quantization of Muse Spark Open 816B-A42B, published for Blackwell. At 497 GiB it fits one
RTX PRO 6000 node, where the 1.5 TiB BF16 release needs four H200 nodes. It serves as `muse-spark`.

This build runs under SGLang, not vLLM. vLLM selects a TRT-LLM MoE backend for this checkpoint's packed
NVFP4 experts and then refuses it on sm120, so there is no working vLLM path on RTX PRO 6000 today.

## Hardware variants

| Variant | Engine | Shape | Single stream | Aggregate | Status |
| --- | --- | --- | --- | --- | --- |
| [`rtx-8-sglang`](rtx-8-sglang/README.md) | SGLang | 1 RTX node, 8 GPUs, TP8 with EP8 | 102.3 tok/s | 1084.2 at c=48, capped | Validated |

One RTX node is faster for a single caller than four H200 nodes on BF16, 102.3 against 83.0, helped by
DFlash speculative decoding and by having no pipeline-parallel hop between nodes. It admits only 48
concurrent requests against the BF16 recipe's 1024, so it suits one or a few interactive users rather than
a busy shared endpoint. See [Muse-Spark-Open-816B-A42B](../Muse-Spark-Open-816B-A42B/README.md).

Two backend choices are forced by missing sm120 kernels and are not interchangeable: the MoE runner must
be `marlin` and attention must be `flashinfer`. The values published upstream for RTX PRO 6000 do not work
on this checkpoint. The recipe's Known limits section records what each wrong choice does.

## Checkpoint provenance

- Checkpoint directory: `Muse-Spark-Open-816B-A42B-NVFP4`
- Hugging Face repo: `meta-models/Muse-Spark-Open-816B-A42B-NVFP4`
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Muse-Spark-Open-816B-A42B-NVFP4`
- Size: 497 GiB across 127 shards
- Architecture: `MuseSparkForConditionalGeneration`, `quant_method` modelopt, NVFP4, group size 16

The routed experts are stored packed, which is what forces the MoE backend choice. Layers 0 and 61,
attention and `lm_head` are excluded from quantization, as are the audio tower's first MLP blocks.

A DFlash draft model, `RadixArk/Muse-Spark-Open-DFlash2` at 9.2 GiB, provides speculative decoding and is
served by default.
