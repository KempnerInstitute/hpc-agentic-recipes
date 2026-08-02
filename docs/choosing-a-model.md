# Choosing a model

Decode rate is the wrong first question. The fastest model here activates 4B parameters and is not the
one you want for hard agentic work.

## Start here

**Best default for interactive coding on one GPU:** Gemma-4-26B-A4B. It is a mixture of experts with 4B
active parameters, so it is fast and cheap to schedule, and one GPU means the shortest queue wait.

**Fastest large model:** GLM-5.2 in NVFP4 on one RTX node, at 93.4 tok/s single stream, helped by MTP
speculative decoding that works because the model fits one node with no pipeline parallelism. Its NVFP4
card measures no meaningful quality loss against FP8; see below.

**Best quality-per-second on one node:** Qwen3-Coder-480B-FP8 on one RTX node at 67.7 tok/s, slightly
faster than the much smaller Qwen3-235B on the same hardware.

**Highest published coding scores:** Kimi-K3, a 2.8T MoE in MXFP4, which needs 4 H200 nodes and runs only
under SGLang. Kimi-K2.7-Code, 1T in INT4, is the largest coding-specialized checkpoint and fits a single
RTX node. Both trade a lot of latency for that quality: 40.3 and 20.7 tok/s.

**Longest context:** GLM-5.2-FP8 across two H200 nodes and Kimi-K3 both support 1M tokens. The recipes
serve less by default and raising it costs decode rate.

No published source ranks all of these against each other, so pick on the scores below and on your own
task rather than on this ordering.

## The table

The full comparison lives in the top-level README so that the entry point and the numbers stay together.
Each row links to a recipe, and every recipe repeats its own number with its measurement protocol.

## Published quality scores

Everything above this line is throughput measured here. Quality is not, so these are the numbers the model
authors publish on their own Hugging Face cards. **Read them with three caveats.** They are self-reported by
the people who trained the model, not independent evaluation. Scores are only comparable *within* one card,
because each vendor picks its own benchmark set and harness. And each card reports the reference model,
while the recipes here often serve a quantized copy.

### Coding and agentic benchmarks

From the Kimi K3 card, which is the only source here that scores several of these models under one harness.
Higher is better.

| Benchmark | Kimi K3 | GLM-5.2 |
| --- | --- | --- |
| Terminal-Bench 2.1 | 88.3 | 82.7 |
| FrontierSWE | 81.2 | 67.3 |
| ProgramBench | 77.8 | 63.7 |
| DeepSWE | 67.5 | 46.2 |
| SWE-Marathon | 42.0 | 13.0 |
| GPQA Diamond | 93.5 | 91.2 |

Source: [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3). That card also scores several
closed models; K3 leads GLM-5.2 on every row above, the first five being coding and agentic and
GPQA Diamond being general reasoning. Its own note says K3 was run with the Kimi Code
harness while other models take the best score across harnesses, which favors K3.

From the Kimi K2.7-Code card:

| Benchmark | Kimi K2.6 | Kimi K2.7 Code |
| --- | --- | --- |
| Kimi Code Bench v2 | 50.9 | 62.0 |
| Program Bench | 48.3 | 53.6 |
| MLS Bench Lite | 26.7 | 35.1 |
| MCP Atlas | 69.4 | 76.0 |
| MCP Mark Verified | 72.8 | 81.1 |

Source: [moonshotai/Kimi-K2.7-Code](https://huggingface.co/moonshotai/Kimi-K2.7-Code). The same card puts
GPT-5.5 and Claude Opus 4.8 ahead of K2.7-Code on most of these, so treat it as the strongest open coder
here rather than the strongest available.

From the DeepSeek-V4-Pro card, at its `max` reasoning effort:

| Benchmark | DeepSeek-V4-Pro |
| --- | --- |
| LiveCodeBench (Pass@1) | 93.5 |
| SWE Verified (Resolved) | 80.6 |
| SWE Multilingual (Resolved) | 76.2 |
| SWE Pro (Resolved) | 55.4 |
| Codeforces (Rating) | 3206 |

Source: `DeepSeek-V4-Pro` model card. The card shows these collapse without reasoning effort, to 56.8 on
LiveCodeBench and 73.6 on SWE Verified, so the headline numbers depend on spending output tokens on
thinking.

From the shared Gemma 4 card. Both Gemma models here appear in one table, which makes this the one
apples-to-apples comparison in this document:

| Benchmark | Gemma 4 31B | Gemma 4 26B A4B |
| --- | --- | --- |
| LiveCodeBench v6 | 80.0% | 77.1% |
| MMLU Pro | 85.2% | 82.6% |
| GPQA Diamond | 84.3% | 82.3% |
| AIME 2026, no tools | 89.2% | 88.3% |

Source: the `gemma-4-31B-it` and `gemma-4-26B-A4B-it` cards, which publish the same family table. The 26B
model with 4B active parameters lands within 3 points of the 31B dense model on every row while decoding
about 2.8x faster here, which is why it is the recommended default.

### Models with no published scores

`GLM-4.6`, `Qwen3-235B-A22B`, and `Qwen3-Coder-480B` ship no benchmark table on their cards. The
Qwen3-Coder card points to [the Qwen blog](https://qwenlm.github.io/blog/qwen3-coder/) for evaluation, where
the figures are published as images rather than text. Nothing is quoted for them here rather than sourcing a
number to an aggregator.

### Quantization cost

Serving a quantized copy is the usual reason a published score would not transfer. GLM-5.2-NVFP4 is the one
checkpoint here that publishes a like-for-like comparison against its own FP8 baseline:

| Precision | GPQA Diamond | SciCode | IFBench | AA-LCR | τ²-Bench Telecom |
| --- | --- | --- | --- | --- | --- |
| FP8 baseline | 89.52 | 49.85 | 74.95 | 69.38 | 97.9 |
| NVFP4 | 89.39 | 49.04 | 75.81 | 70.13 | 98.25 |

Source: the `GLM-5.2-NVFP4` model card. The two are within a point either way, which is the evidence for
preferring the NVFP4 recipe on RTX: it is far faster here and its own author measures no meaningful quality
loss. No comparable table exists for the INT4 and FP8 checkpoints of the other models, so treat their
published scores as upper bounds.

## Read the protocol column

Rates measured with different protocols are not comparable:

- `slope(128,1152)` times the same request at two output lengths and divides the difference, which
  cancels prefill and fixed per-request cost. This is the honest sustained decode rate.
- `single-generation` times one request and divides tokens by wall time, counting prefill and fixed cost
  as decode, so it understates sustained decode. The shorter the generation, the larger the error.

Every rate in this repo is slope-measured. The vLLM recipes use `common/tools/bench.sh`; the Kimi-K3
figures came from a separate harness applying the same `slope(128,1152)` protocol, which is why that
recipe is Untested until it is re-measured from its own scripts. Details in
[benchmarking.md](benchmarking.md).

## Read the status column

`Validated` means the recipe in this repo was run end to end with the stated engine version, and its
rates were measured from those files rather than carried over. `Untested` means it has
never been launched and every number in it is a prediction. `Blocked` means it does not currently run at
all, and the recipe says why.

Aggregate rates carry one of three labels. `peak` means throughput turned over inside the sweep, so it is
a measured maximum. `saturated` means it varies by under 4 percent from concurrency 512 to 1024, so it is
a real ceiling even though no single level stands out. `rising` means it was still climbing at 1024, the
top of the sweep, so the figure is a floor and the true peak is higher.

## Three performance regimes, and why flags help or do not

Measured here, and the reason the same flag helps one model and does nothing for another:

**Memory-bandwidth bound.** A dense model such as Gemma-4-31B reads every weight for every token. FP8
weights halve the bytes moved and gave a large speedup, scaling with the GPU's memory bandwidth.

**Host-overhead bound.** A sparse MoE such as Gemma-4-26B-A4B activates 4B of its parameters, so the GPU
finishes each step before the host can feed it the next. Utilization sat at 35 to 40 percent and power at
roughly 210 W of a 700 W budget. FP8 gave exactly zero benefit, because bandwidth was never the limit.

**Communication bound.** A large model at TP8 on an RTX node is limited by all-reduce traffic crossing
PCIe, because there is no NVLink. FP8 weights again bought nothing, and expert parallelism made it worse
by adding all-to-all traffic, measuring about 9 percent slower.

The practical rule: FP8 weights pay off for dense models on high-bandwidth GPUs, and do nothing for sparse
MoE models or for anything already limited by the interconnect.
