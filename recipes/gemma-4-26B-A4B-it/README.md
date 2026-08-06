# gemma-4-26B-A4B-it

Gemma 4 26B-A4B, instruction tuned: a mixture-of-experts model that activates 4B of its 26B parameters
per token, with native tool calling and a separate reasoning channel. It runs on one GPU and has the
highest single stream rate in this repo, which makes it the best default for interactive coding.

Served as `gemma-4-26b` on `http://<node>:8000/v1`, and on vLLM's Anthropic-compatible API at
`http://<node>:8000`, so Claude Code connects with no proxy.

## Variants

One per GPU type, because the toolchains differ: RTX PRO 6000 (sm_120) needs torch cu130, a conda CUDA
13.0 toolkit and FlashInfer 0.6.15, while the Hopper nodes run driver 575 and take the cu129 release
wheel. All three are TP1 on one GPU and serve the same name.

| Variant | GPU | Single stream | Aggregate | KV cache | Status |
| --- | --- | --- | --- | --- | --- |
| [`h200-1`](h200-1/README.md) | 1 H200, 143771 MiB | 250.5 tok/s | 10905 at c=1024, saturated | 2,776,615 tokens | Validated |
| [`h100-1`](h100-1/README.md) | 1 H100, 81559 MiB | 204.5 tok/s | 7165 at c=640, peak | 661,098 tokens | Validated |
| [`rtx-1`](rtx-1/README.md) | 1 RTX PRO 6000 Blackwell, 97887 MiB | 141.1 tok/s | 5798 at c=1024, rising | 1,013,590 tokens | Validated |

Measured in bf16 at the 262144 default, protocol slope(128,1152) over output tokens only, 3 repeats per
level. Rates are defined in [docs/benchmarking.md](../../docs/benchmarking.md), the aggregate labels in
[docs/choosing-a-model.md](../../docs/choosing-a-model.md).

- Single stream varies only 1.8x across the three GPU types, so an idle GPU beats queueing for a faster one.
- `QUANT=fp8` measured no change in rate on any of them, so every variant serves bf16.

## Checkpoint

| | |
| --- | --- |
| Directory | `gemma-4-26B-A4B-it` |
| Hugging Face repo | `google/gemma-4-26B-A4B-it` |
| Documented path | `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/gemma-4-26B-A4B-it` |
| Size on disk | 51.6 GB, bf16 |
| Architecture | `Gemma4ForConditionalGeneration`, 128 experts with top-8 routing, multimodal |
| Context | 262144, the checkpoint maximum, served by default on all three variants |
| Drafter | `gemma-4-26B-A4B-it-assistant`, under 1 GB, wired through `SPEC_DRAFT`, unusable on vLLM 0.25.1 and 0.26.0 |
