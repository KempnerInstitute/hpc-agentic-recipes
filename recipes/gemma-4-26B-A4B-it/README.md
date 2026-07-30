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

| Variant | GPU | Decode rate | Status |
| --- | --- | --- | --- |
| [`h200-1`](h200-1/README.md) | 1 H200, 143771 MiB | 236.0 tok/s | Untested (migrated) |
| [`h100-1`](h100-1/README.md) | 1 H100, 80 GB | 183.9 tok/s | Untested (migrated) |
| [`rtx-1`](rtx-1/README.md) | 1 RTX PRO 6000 Blackwell, 97887 MiB | 140.5 tok/s | Untested (migrated) |

Rates are sustained single-stream decode, bf16, 32K context, measured 2026-07-27 with the
pre-restructure scripts, protocol slope(128,1152). None of the three has been run from these recipe
files yet.

Pick by what you can get. The spread across GPU types is only 1.7x, because this model is host overhead
bound rather than memory bandwidth bound, so an idle RTX GPU beats a queued H200.

## Checkpoint

| | |
| --- | --- |
| Directory | `gemma-4-26B-A4B-it` |
| Hugging Face repo | not recorded before the restructure; the testbed copy is the system of record |
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
