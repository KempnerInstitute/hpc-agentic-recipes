# gemma-4-31B-it

Gemma 4 31B, instruction tuned: a dense model, so every parameter is read for every token, with native
tool calling and a separate reasoning channel. It runs on a single GPU and is the higher-quality half of
the single-GPU Gemma 4 pair, at roughly a third the decode rate of the 26B mixture-of-experts sibling.

Served as `gemma-4-31b` on `http://<node>:8000/v1`, and on vLLM's Anthropic-compatible API at
`http://<node>:8000`, so Claude Code connects with no proxy.

## Variants

One per GPU type, because the toolchains differ: RTX PRO 6000 (sm_120) needs torch cu130, a conda CUDA
13.0 toolkit and FlashInfer 0.6.15, while the Hopper nodes run driver 575 and take the cu129 release
wheel. All three are TP1 on one GPU with FP8 weights and serve the same name.

| Variant | GPU | Single stream | Aggregate | KV cache | Status |
| --- | --- | --- | --- | --- | --- |
| [`h200-1`](h200-1/README.md) | 1 H200, 143771 MiB | 85.0 tok/s | 3154 at c=768, saturated | 894,418 tokens | Validated |
| [`h100-1`](h100-1/README.md) | 1 H100, 81559 MiB | 68.7 tok/s | 2471 at c=512, saturated | 365,231 tokens | Validated |
| [`rtx-1`](rtx-1/README.md) | 1 RTX PRO 6000 Blackwell, 97887 MiB | 39.5 tok/s | 2136 at c=768, saturated | 401,491 tokens | Validated |

Measured in FP8 at the 262144 default, protocol slope(128,1152) over output tokens only, 3 repeats per
level. Rates are defined in [docs/benchmarking.md](../../docs/benchmarking.md), the aggregate labels in
[docs/choosing-a-model.md](../../docs/choosing-a-model.md).

- All three are `saturated`, varying under 4 percent across concurrency 512 to 1024, so concurrency beyond
  that buys only queueing delay. On the H100 the highest value sits at 512.
- Single stream varies 2.1x across the three GPU types and tracks HBM bandwidth, so unlike the 26B sibling
  it is worth queueing for the faster card.
- `QUANT=fp8` is the default everywhere and is worth 72 percent on RTX, 69 percent on H100 and 51 percent on
  H200, against BF16 rates of 23.0, 40.7 and 56.3 tok/s that were not re-measured in this sweep. The 26B
  sibling is the opposite case, where the same flag measured no change.
- `SPEC_DRAFT` works on all three and is worth 2.6 to 2.7x on single stream, at a KV pool cost of 2.5 to 6.0
  percent. Each variant page carries its own figures.
- KV cost is not linear in context, because 50 of the 60 layers keep only a 1024-token sliding window.

## Checkpoint

| | |
| --- | --- |
| Directory | `gemma-4-31B-it` |
| Hugging Face repo | `google/gemma-4-31B-it` |
| Documented path | `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/gemma-4-31B-it` |
| Size on disk | 62.6 GB, BF16, quantized to FP8 at load time rather than from a second checkpoint |
| Architecture | `Gemma4ForConditionalGeneration`, dense, multimodal |
| Context | 262144, the checkpoint maximum, served by default on all three variants |
| Drafter | `gemma-4-31B-it-assistant`, under 1 GB, wired through `SPEC_DRAFT` and usable on all three |
