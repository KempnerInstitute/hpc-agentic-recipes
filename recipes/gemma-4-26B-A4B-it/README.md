# gemma-4-26B-A4B-it

Gemma 4 26B-A4B, instruction tuned: a mixture-of-experts model that activates 4B of its 26B parameters
per token, with native tool calling and a separate reasoning channel. It runs on a single GPU and has
the highest decode rate of anything in this repo, which makes it the best default for interactive work.

Served as `gemma-4-26b` on `http://<node>:8000/v1`, and on vLLM's Anthropic-compatible API at
`http://<node>:8000`, so Claude Code connects with no proxy.

## Variants

One per GPU type, because the toolchains genuinely differ: on RTX PRO 6000 (sm_120) this checkpoint
needs torch cu130, a conda CUDA 13.0 toolkit, and FlashInfer 0.6.15, while the Hopper nodes run driver
575 and need the cu129 release wheel instead. All three are TP1 on one GPU and serve the same name, so
they differ only in hardware, environment, and rate.

| Variant | GPU | Single stream | Aggregate | Status |
| --- | --- | --- | --- | --- |
| [`h200-1`](h200-1/README.md) | 1 H200, 143771 MiB | 236.3 tok/s | 10727 at c=256, peak | Validated |
| [`h100-1`](h100-1/README.md) | 1 H100, 80 GB | 203.4 tok/s | 7243 at c=640, saturated | Validated |
| [`rtx-1`](rtx-1/README.md) | 1 RTX PRO 6000 Blackwell, 97887 MiB | 140.6 tok/s | 5972 at c=1024, rising | Validated |

Single stream is one request at a time, which is what interactive coding feels like. Saturated is total
output across every concurrent stream at the stated concurrency, 36 to 45 times larger here, and it says
nothing about how fast a single reply arrives. The labels differ by GPU: the H200 turned over at concurrency 256, so that is a measured `peak`; the
H100 is `saturated`, varying by 2.2 percent from 512 upward; and the RTX variant was still `rising` at
1024, so its figure is a floor. All three measured with bf16, 32K context, protocol
slope(128,1152) over output tokens only, 3 repeats per level, concurrency 1 through 1024.

Pick by what you can get. The spread across GPU types is only 1.7x, because this model is host overhead
bound rather than memory bandwidth bound, so an idle RTX GPU beats a queued H200.

## Checkpoint

| | |
| --- | --- |
| Directory | `gemma-4-26B-A4B-it` |
| Hugging Face repo | `google/gemma-4-26B-A4B-it` |
| Documented path | `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/gemma-4-26B-A4B-it` |
| Size on disk | 51.6 GB, bf16 |
| Architecture | `Gemma4ForConditionalGeneration`, 128 experts with top-8 routing, multimodal |
| Context | 256K supported, 32K default in every variant |
| KV cache | 40 KiB per token: 1.5 GB at 32K, 5.2 GB at 128K, 10 GB at 256K |
| Drafter | `gemma-4-26B-A4B-it-assistant`, under 1 GB, wired through `SPEC_DRAFT` and unusable on vLLM 0.25.1 and 0.26.0 |

Quantization is deliberately not used. FP8 weights measured no change in decode rate on any GPU type,
because only 4B parameters are read per token and there is almost nothing for a narrower dtype to save.
The dense `gemma-4-31B-it` sibling is the opposite case, where FP8 is worth 52 to 74 percent depending
on the card. Each variant page carries the numbers and the reasoning for its own hardware.
