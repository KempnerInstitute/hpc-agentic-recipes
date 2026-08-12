# Choosing a model

Decode rate is the wrong first question. The fastest model here activates 4B parameters and is not the one
you want for hard agentic work. The full comparison is the table in the top-level README; this page says how
to read it and what the published quality scores are worth.

## Start here

- **Interactive coding on one GPU:** Gemma-4-26B-A4B. A mixture of experts with 4B active parameters, so it
  is fast, and one GPU is the shortest queue wait.
- **Fastest large model:** GLM-5.2-NVFP4 on one RTX node, 91.1 tok/s, helped by MTP speculative decoding
  that works because the model fits one node with no pipeline parallelism.
- **Best quality per second on one node:** Qwen3-Coder-480B-FP8 on one RTX node, 68.0 tok/s, slightly faster
  than the much smaller Qwen3-235B on the same hardware.
- **Highest published coding scores:** Kimi-K3, 2.8T in MXFP4, needing 4 H200 nodes and SGLang.
  Kimi-K2.7-Code, 1T in INT4, is the largest coding-specialized checkpoint that fits one RTX node. Both cost
  latency for it: 40.3 and 20.6 tok/s.
- **Longest context:** DeepSeek-V4-Pro on two RTX nodes, serving its full 1M window by default. It is the one
  case here where context is not free, costing 14 to 17 percent of aggregate throughput above concurrency
  256, though nothing at concurrency 1.
- **A 1M window on one node:** DeepSeek-V4-Flash, which serves the same full context from a single RTX node
  and holds 9.06 full-length requests at once. Its card also reports the strongest agentic scores of any
  checkpoint here that fits one node. The cost is latency: 15.1 tok/s on a single stream, because the
  speculative head it ships cannot run on this hardware.

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

From the `DeepSeek-V4-Pro` card, at its `max` reasoning effort:

| Benchmark | DeepSeek-V4-Pro |
| --- | --- |
| LiveCodeBench (Pass@1) | 93.5 |
| SWE Verified (Resolved) | 80.6 |
| SWE Multilingual (Resolved) | 76.2 |
| SWE Pro (Resolved) | 55.4 |
| Codeforces (Rating) | 3206 |

Without reasoning effort the card shows these collapse, to 56.8 on LiveCodeBench and 73.6 on SWE Verified,
so the headline numbers depend on spending output tokens on thinking.

From the `DeepSeek-V4-Flash-0731` card, which is the second place in this document where two models served
here appear in one table, Flash and GLM-5.2. Its DeepSeek-V4-Pro column is the Preview, not the Pro the
recipes here serve, so it is not the same model as the table above:

| Benchmark | V4-Flash-0731 | V4-Pro Preview | GLM-5.2 | Opus-4.8 |
| --- | --- | --- | --- | --- |
| Terminal Bench 2.1 | 82.7 | 72.1 | 81.0 | 85.0 |
| NL2Repo | 54.2 | 38.5 | 48.9 | 69.7 |
| Cybergym | 76.7 | 52.7 | not reported | 83.1 |
| DeepSWE | 54.4 | 12.8 | 46.2 | 58.0 |
| Toolathlon-Verified | 70.3 | 55.9 | 59.9 | 76.2 |
| Agents' Last Exam | 25.2 | 16.5 | 23.8 | 25.7 |
| AutomationBench Public | 25.1 | 12.8 | 12.9 | 27.2 |

Flash leads GLM-5.2 on every row and trails Opus-4.8 on every row. Two internal DeepSeek test sets on the
card are left out here. These are agent-harness scores, run with the `max` reasoning effort and thinking
enabled, which is not what a default request to this endpoint does; the recipe page shows how to match it.

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
