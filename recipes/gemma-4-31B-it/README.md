# gemma-4-31B-it

Gemma 4 31B, instruction tuned: a dense model, so every parameter is read for every token, with native
tool calling and a separate reasoning channel. It runs on a single GPU and is the higher-quality half of
the single-GPU Gemma 4 pair, at roughly a third the decode rate of the 26B mixture-of-experts sibling.

Served as `gemma-4-31b` on `http://<node>:8000/v1`, and on vLLM's Anthropic-compatible API at
`http://<node>:8000`, so Claude Code connects with no proxy.

## Variants

One per GPU type, because the toolchains genuinely differ: on RTX PRO 6000 (sm_120) this checkpoint
needs torch cu130, a conda CUDA 13.0 toolkit, and FlashInfer 0.6.15, while the Hopper nodes run driver
575 and need the cu129 release wheel instead. All three are TP1 on one GPU with FP8 weights and serve
the same name, so they differ only in hardware, environment, and rate.

| Variant | GPU | Single stream, FP8 | Aggregate, FP8 | Status |
| --- | --- | --- | --- | --- |
| [`h200-1`](h200-1/README.md) | 1 H200, 143771 MiB | 85.1 tok/s | 3136 at c=1024, saturated | Validated |
| [`h100-1`](h100-1/README.md) | 1 H100, 80 GB | 67.4 tok/s | 2680 at c=512, peak | Validated |
| [`rtx-1`](rtx-1/README.md) | 1 RTX PRO 6000 Blackwell, 97887 MiB | 39.5 tok/s | 2139 at c=768, saturated | Validated |

Single stream is one request at a time, which is what interactive coding feels like. Saturated is total
output across every concurrent stream, and it says nothing about how fast a single reply arrives.
The H200 and RTX variants are `saturated`, varying by under 4 percent across concurrency 512 to 1024, so
adding concurrency past them buys only queueing delay. The H100 turned over at 512, so its figure is a
measured `peak`. All three are real ceilings. This model reaches
its limit at lower concurrency than its 26B sibling, which is consistent with it being memory bandwidth
bound. Measured at 32K context, protocol slope(128,1152) over output tokens only, 3 repeats
per level, concurrency 1 through 1024.

Running the same weights in bf16 instead of FP8 measured 56.3, 40.7 and 23.0 tok/s single stream on the
same three GPUs. Those bf16 figures were not re-measured in this
sweep, so treat them as indicative of the roughly 1.7x FP8 advantage
rather than as current numbers.

The spread across GPU types is 2.1x and tracks HBM bandwidth, because this model is memory bandwidth
bound, so unlike the 26B sibling it is worth queueing for the faster card.

## Checkpoint

| | |
| --- | --- |
| Directory | `gemma-4-31B-it` |
| Hugging Face repo | `google/gemma-4-31B-it` |
| Documented path | `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/gemma-4-31B-it` |
| Size on disk | 62.6 GB, bf16, quantized to FP8 at load time rather than from a second checkpoint |
| Architecture | `Gemma4ForConditionalGeneration`, dense, multimodal |
| Context | 256K supported, 32K default in every variant |
| KV cache | 160 KiB per token: 5.8 GB at 32K, 21 GB at 128K, 41 GB at 256K |
| Drafter | `gemma-4-31B-it-assistant`, under 1 GB, wired through `SPEC_DRAFT` and unusable on vLLM 0.25.1 and 0.26.0 |

FP8 weights are the default everywhere, and this is the most interesting measured fact about the model:
they are worth 74 percent on RTX, 69 percent on H100, and 52 percent on H200, because a dense model
reads all of its weights for every token and halving the bytes per weight buys most of a proportional
speedup. The `gemma-4-26B-A4B-it` sibling is the opposite case, where the same flag measured no change at
all. FP8 also decides the usable context on the smaller cards. Each variant page carries the numbers and
the reasoning for its own hardware.
