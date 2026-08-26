# Choosing a model

Decode rate is the wrong first question. The fastest model here activates 4B parameters and is not the one
you want for hard agentic work. The full comparison is the table in the top-level README; this page says how
to read it and what the published quality scores are worth.

## Start here

- **Interactive coding on one GPU:** Gemma-4-26B-A4B. A mixture of experts with 4B active parameters, so it
  is fast, and one GPU is the shortest queue wait.
- **Coding scores on one GPU:** Qwen3.8-27B, a dense 27B serving its full 262144 context on one H200 GPU.
  The FP8 build is the faster of the two, 170.6 tok/s against 122.1, and holds 9.32 full-length requests at
  once against 6.88. Both decode slower than Gemma-4-26B-A4B on the same hardware, and this is the only
  single-GPU model here whose card scores it against a frontier model.
- **Fastest large model:** Kimi-K2.7-Code on two H200 nodes, 103.0 tok/s. GLM-5.2-NVFP4 is the fastest that
  fits a single node, 91.1 tok/s on one RTX node, helped by MTP speculative decoding that works because the
  model fits one node with no pipeline parallelism.
- **Best quality per second on one node:** Qwen3-Coder-480B-FP8 on one RTX node, 68.0 tok/s, slightly faster
  than the much smaller Qwen3-235B on the same hardware.
- **Highest published coding scores:** Kimi-K3, 2.8T in MXFP4, needing 4 H200 nodes and SGLang.
  Kimi-K2.7-Code, 1T in INT4, is the largest coding-specialized checkpoint that fits one RTX node. Both cost
  latency for it: 40.3 and 20.6 tok/s.
- **Longest context:** GLM-5.2-FP8 on two H200 nodes, serving 626K by default.

No published source ranks all of these against each other, so pick on the scores below and on your own task
rather than on this ordering.

## Aggregate labels

Aggregate rates carry one of four labels, shown as a symbol in the table and keyed under it:

- `peak` means throughput turned over inside the sweep, so it is a measured maximum.
- `saturated` means it varies by under 4 percent across the levels at or above concurrency 512, so it is a
  real ceiling even though no single level stands out.
- `rising` means it was still climbing at 1024, the top of the sweep, so the figure is a floor and the
  true peak is higher.
- `capped` means the engine admits fewer concurrent requests than the sweep would reach, so the figure is
  the most that endpoint accepts rather than a point on a curve.

## How the rates were measured

Every rate here is slope-measured with `common/tools/bench.sh` against an endpoint the recipe's own scripts
launched, and each recipe repeats its own number with the protocol and the concurrency levels it swept.
Rates measured with different protocols are not comparable:

- `slope(128,1152)` times the same request at two output lengths and divides the difference, which cancels
  prefill and fixed per-request cost. This is the honest sustained decode rate.
- `single-generation` times one request and divides tokens by wall time, counting prefill and fixed cost as
  decode, so it understates sustained decode. The shorter the generation, the larger the error.

Details in [benchmarking.md](benchmarking.md).

## Published quality scores

Throughput here is measured. Quality is not, so the rest of this page is what the model authors publish on
their own cards. Three caveats:

- They are self-reported by the people who trained the model, not independent evaluation.
- Scores are comparable only *within* one card, since each vendor picks its own benchmark set and harness.
- Each card reports the reference model, while the recipes here often serve a quantized copy.

### Coding and agentic

From the [Kimi K3 card](https://huggingface.co/moonshotai/Kimi-K3), the only source here scoring several of
these models under one harness. Higher is better.

| Benchmark | Kimi K3 | GLM-5.2 |
| --- | --- | --- |
| Terminal-Bench 2.1 | 88.3 | 82.7 |
| FrontierSWE | 81.2 | 67.3 |
| ProgramBench | 77.8 | 63.7 |
| DeepSWE | 67.5 | 46.2 |
| SWE-Marathon | 42.0 | 13.0 |
| GPQA Diamond | 93.5 | 91.2 |

K3 leads on every row, the first five coding and agentic and the last general reasoning. The card's own note
says K3 was run with the Kimi Code harness while other models take the best score across harnesses, which
favors K3.

From the [Kimi K2.7-Code card](https://huggingface.co/moonshotai/Kimi-K2.7-Code):

| Benchmark | Kimi K2.6 | Kimi K2.7 Code |
| --- | --- | --- |
| Kimi Code Bench v2 | 50.9 | 62.0 |
| Program Bench | 48.3 | 53.6 |
| MLS Bench Lite | 26.7 | 35.1 |
| MCP Atlas | 69.4 | 76.0 |
| MCP Mark Verified | 72.8 | 81.1 |

The same card puts GPT-5.5 and Claude Opus 4.8 ahead on most rows, so treat it as the strongest open coder
here rather than the strongest available.

From the shared Gemma 4 card, the one apples-to-apples comparison in this document since both models appear
in one table:

| Benchmark | Gemma 4 31B | Gemma 4 26B A4B |
| --- | --- | --- |
| LiveCodeBench v6 | 80.0% | 77.1% |
| MMLU Pro | 85.2% | 82.6% |
| GPQA Diamond | 84.3% | 82.3% |
| AIME 2026, no tools | 89.2% | 88.3% |

The 26B model with 4B active parameters lands within 3 points on every row while decoding about 2.9x faster
here, which is why it is the recommended default.

From the [Qwen3.8-27B card](https://huggingface.co/Qwen/Qwen3.8-27B), scored against its own predecessor
and a frontier model. Higher is better.

| Benchmark | Qwen3.8-27B | Qwen3.6-27B | Opus 4.6 Max |
| --- | --- | --- | --- |
| Terminal Bench 2.1 | 73.0 | 63.4 | 78.2 |
| SWE-bench Pro | 61.7 | 53.5 | 53.4 |
| NL2Repo-Bench | 42.3 | 36.2 | 47.6 |
| QwenSWEBench | 79.0 | 49.3 | 63.8 |
| LiveCodeBench v6 | 90.3 | 83.9 | 88.8 |
| GPQA Diamond | 89.2 | 87.8 | 91.3 |

It leads Opus 4.6 Max on SWE-bench Pro, QwenSWEBench and LiveCodeBench v6, and trails it on Terminal Bench,
NL2Repo-Bench and GPQA Diamond. QwenSWEBench is the vendor's own benchmark. The card's vision scores are
left out because this recipe is validated for text. The FP8 build reprints the same table and states that
its metrics are nearly identical, without publishing a comparison, so read these as an upper bound for that
recipe.

### No published scores

`GLM-4.6`, `Qwen3-235B-A22B` and `Qwen3-Coder-480B` ship no benchmark table. The Qwen3-Coder card points to
[the Qwen blog](https://qwenlm.github.io/blog/qwen3-coder/), where the figures are images rather than text.
Nothing is quoted for them here rather than sourcing a number to an aggregator.

### Quantization cost

Serving a quantized copy is the usual reason a published score would not transfer. GLM-5.2-NVFP4 is the one
checkpoint here that publishes a like-for-like comparison against its own FP8 baseline:

| Precision | GPQA Diamond | SciCode | IFBench | AA-LCR | τ²-Bench Telecom |
| --- | --- | --- | --- | --- | --- |
| FP8 baseline | 89.52 | 49.85 | 74.95 | 69.38 | 97.9 |
| NVFP4 | 89.39 | 49.04 | 75.81 | 70.13 | 98.25 |

The two are within a point either way, which is the evidence for preferring the NVFP4 recipe on RTX: far
faster here, and its own author measures no meaningful quality loss. No comparable table exists for the INT4
and FP8 checkpoints of the other models, so treat their published scores as upper bounds.

## Why a flag helps one model and not another

Three regimes, measured here:

- **Memory-bandwidth bound.** A dense model such as Gemma-4-31B reads every weight for every token. FP8
  halves the bytes moved and gave a large speedup, scaling with the GPU's memory bandwidth.
- **Host-overhead bound.** A sparse MoE such as Gemma-4-26B-A4B activates 4B parameters, so the GPU finishes
  each step before the host can feed it the next. Utilization sat at 35 to 40 percent and power at roughly
  210 W of a 700 W budget. FP8 gave exactly zero benefit, because bandwidth was never the limit.
- **Communication bound.** A large model at TP8 on an RTX node is limited by all-reduce crossing PCIe, since
  there is no NVLink. FP8 again bought nothing, and expert parallelism made it worse by adding all-to-all
  traffic, measuring about 9 percent slower.

The rule: FP8 weights pay off for dense models on high-bandwidth GPUs, and do nothing for sparse MoE models
or for anything already limited by the interconnect.
