# DeepSeek-V4-Pro

A 1.6T-parameter mixture-of-experts model with 49B activated per token: 61 layers, 384 routed experts
plus 1 shared with 6 routed per token, and a hybrid compressed-attention design with a 1M-token context.
It is a thinking model with native tool calling, and it serves through vLLM's Anthropic-compatible API
as `deepseek-v4-pro`, so Claude Code connects with no proxy.

The weights are FP8 with `weight_block_size` [128, 128], but the routed experts are FP4, and that one
fact decides the hardware: only Blackwell executes FP4 natively.

## Hardware variants

| Variant | Shape | Single stream | Aggregate | Status |
| --- | --- | --- | --- | --- |
| [`rtx-8-nodes2`](rtx-8-nodes2/README.md) | 2 RTX PRO 6000 nodes, 8 GPUs each, TP8 x PP2 | 18.7 tok/s | 3003 at c=1024, rising | Validated |

There is no Hopper variant. The same checkpoint was run end to end on two H200 nodes and fails during
engine initialization, inside the CUTLASS w8a8 kernel dispatch, because Hopper has no FP4 tensor cores
for the expert layers.

The RTX figure is `rising`, meaning throughput was still climbing at concurrency 1024, the top of the
sweep, so 3003 tok/s is a floor rather than a ceiling. Measured with protocol slope(128,1152) over
output tokens only, 3 repeats per level.

## Checkpoint provenance

- Checkpoint directory: `DeepSeek-V4-Pro`
- Hugging Face repo: `deepseek-ai/DeepSeek-V4-Pro`
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/DeepSeek-V4-Pro`
- Size: 805.3 GiB across 64 shards
- License: MIT
- Architecture: `DeepseekV4ForCausalLM`, native to vLLM, no `--trust-remote-code` needed

The checkpoint ships a speculative head, `num_nextn_predict_layers` 1, which neither variant can use:
both need pipeline parallelism to span two nodes, and vLLM rejects a speculative config whenever
pipeline parallelism is active.
