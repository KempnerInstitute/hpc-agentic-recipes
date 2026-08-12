# DeepSeek-V4-Flash-0731

A mixture-of-experts model of about 284B parameters with 13.8B activated per token: 43 layers, 256 routed
experts with 6 per token plus 1 shared, hidden size 4096, and a 1M-token context. It is a thinking model with
native tool calling, and it serves through vLLM's Anthropic-compatible API as `deepseek-v4-flash`, so Claude
Code connects with no proxy.

The routed experts are MXFP4 and the dense path is FP8, and that one fact decides the hardware: only Blackwell
executes FP4 natively.

The checkpoint also carries a 19.8B speculative head, the DSpark module, which this hardware cannot run. It is
what would normally lift a 13.8B-activated model well above 15.1 tok/s on a single stream, so its absence is
the recipe's main cost; see [Known limits](rtx-8/README.md#known-limits).

## Hardware variants

| Variant | Shape | Single stream | Aggregate | Status |
| --- | --- | --- | --- | --- |
| [`rtx-8`](rtx-8/README.md) | 1 RTX PRO 6000 node, 8 GPUs, TP8 | 15.1 tok/s | 5772 at c=1024, rising | Validated |

There is no Hopper variant. The experts are FP4 and Hopper has no FP4 tensor cores, the same reason
DeepSeek-V4-Pro has no H200 recipe. Capacity is not what stops it: 155.43 GiB spread over four H200s would be
38.9 GiB per GPU.

The vLLM recipe page does list H200 for this checkpoint, but only as single-host 8-GPU disaggregated serving,
4 prefill plus 4 decode GPUs in one chassis. The H200 nodes here carry 4 GPUs each, so that shape does not
exist on this cluster.

The RTX figure is `rising`, meaning throughput was still climbing at concurrency 1024, the top of the sweep,
so 5772 tok/s is a floor rather than a ceiling. Measured with protocol slope(128,1152) over output tokens
only, 3 repeats per level.

## Checkpoint provenance

- Checkpoint directory: `DeepSeek-V4-Flash-0731`
- Hugging Face repo: `deepseek-ai/DeepSeek-V4-Flash-0731`
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/DeepSeek-V4-Flash-0731`
- Size: 155.43 GiB across 48 shards
- License: MIT
- Architecture: `DeepseekV4ForCausalLM`, native to vLLM, no `--trust-remote-code` needed

The card publishes no parameter count. The figures above are counted from the shard headers, where the expert
weights are packed two 4-bit values per byte with one E8M0 scale per block of 32.

This release supersedes the preview and has the same structure as `DeepSeek-V4-Flash-DSpark`, so the
speculative head is part of the checkpoint rather than a separate download.

It ships no Jinja chat template and does not need one: vLLM selects its `deepseek_v4` tokenizer mode without
being told to and encodes messages natively. A default request answers directly and returns no reasoning,
because the prompt closes thinking before the model writes; the recipe page shows how to ask for it.
