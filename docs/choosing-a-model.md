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
- **Largest checkpoint:** Qwen3.8-2.4T-A95B-FP8, 2.4T parameters with 95B active, on six H200 nodes at
  52.6 tok/s. Size buys quality, not speed: a dense 27B reaches 170.6 tok/s on a single GPU here.
- **Most aggregate throughput on four nodes:** Hy4-preview, 836.3 tok/s at concurrency 32 and still
  climbing, with 48.8 tok/s on a single stream, which beats both Kimi recipes on the same hardware.
  A 770B MoE with only 49B active is what buys that. It serves 131072 rather than its declared 1M, and
  its recipe bind-mounts a patched file because SGLang's load barrier is too short for the checkpoint.
- **Newest frontier MoE on four nodes:** Kimi-K3.1-Pro, 40.4 tok/s, within a percent of Kimi-K3 on the
  same four H200 nodes, so the two cost the same to serve and the choice is on published scores.
- **Full context on one GPU for a 65B:** Ripple-65B-Preview in NVFP4, 36.2 tok/s, serving the whole
  262144 window on a single H200 where its BF16 build needs two GPUs for the same window and caps near
  33,000 tokens on one. It leads nothing on rate; it is the cheap way to a long context for this model.
- **Fastest two-node decode:** MiniMax-M3, 111.5 tok/s on two H200 nodes at pure TP8. GLM-5.3 is still
  faster on the same hardware at 142.3, but M3 is the only recipe here that runs cross-node tensor
  parallelism in vLLM, which needs two optional all-reduce accelerators switched off. Its KV pool is
  the catch: 482,176 tokens, so only 3.68 callers fit at its 131072 context.
- **Agentic coding on one node:** MiniMax-M2.7, 214.3 GiB in FP8 on a single H200 node at TP4, the
  cheapest recipe here to reach a published SWE-Pro score near a frontier model. It is slow per stream
  at 20.8 tok/s, but scales almost linearly to 642 tok/s at concurrency 32, so it suits many parallel
  agents rather than one interactive session.

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

From the [Hy4-preview card](https://huggingface.co/tencent/Hy4-preview), a 770B MoE with 49B active:

| Benchmark | Hy4 preview |
| --- | --- |
| GPQA Diamond | 92.3 |
| Terminal-Bench 2.1 | 85.4 |
| SWE-bench Pro | 65.7 |
| Deep SWE | 64.3 |
| Toolathlon, verified | 74.1 |
| Apex Agents | 37.1 |

The card also reports an internal blind evaluation over 203 engineering tasks where Hy4 scores 2.99
against GLM 5.3 at 2.92 and Kimi K3 at 2.94, which is the vendor's own harness and its own task set.
Two rows look comparable with the Kimi K3 table above and are not safely so: K3 reports Terminal-Bench
2.1 at 88.3 and GPQA Diamond at 93.5, both above Hy4, but each card runs its own harness. The card
lists over-reasoning and over-verification as known limitations.

From the [Kimi K3.1 Pro card](https://huggingface.co/moonshotai/Kimi-K3.1-Pro), a 2.8T MoE with 104B
active, scored on **Terminal-Bench 4.0** rather than the 2.1 used above, so its 65.7 is not comparable
with any number in the earlier tables. The card states it was evaluated with the Claude Code harness
while most comparison scores are the best reported elsewhere, which favors it.

From the [Ripple-65B-Preview card](https://huggingface.co/ricursiveAI/Ripple-65B-Preview), a dense 65B:
GPQA Diamond 89.0 and AIME 2024 97.1, against a column of comparison models on the card itself. It is
the only model here small enough to serve on one GPU in NVFP4 at the full context, and its scores are
reported against reasoning benchmarks rather than the agentic coding sets the larger cards use.

From the [MiniMax-M3 card](https://huggingface.co/MiniMaxAI/MiniMax-M3), which scores both MiniMax
models against the same comparison set:

| Benchmark | MiniMax M3 | MiniMax M2.7 | Claude Opus 4.7 | GPT 5.5 | Kimi K2.6 Thinking |
| --- | --- | --- | --- | --- | --- |
| SWE-Bench Verified | 80.5 | 79.9 | 87.6 | 82.9 | 80.2 |
| SWE-Bench Pro | 59.0 | 56.2 | 64.3 | 58.6 | 58.6 |
| Terminal Bench 2.1 | 66.0 | 51.1 | 66.1 | 78.2 | 53.9 |
| NL2Repo | 42.1 | 35.0 | 56.3 | 52.9 | 42.8 |
| SWE-fficiency | 34.8 | 14.0 | 42.2 | 46.6 | - |
| VIBE-V2 | 50.1 | 37.9 | 55.9 | 50.5 | 46.0 |
| KernelBench Hard | 28.8 | 10.5 | 30.7 | 20.9 | - |
| PaperBench | 52.6 | 30.6 | 58.5 | 57.5 | - |
| BrowseComp | 83.5 | 76.3 | 79.3 | 84.4 | 83.2 |
| MCP Atlas | 74.2 | 49.4 | 77.0 | 75.3 | 66.6 |

M3 leads M2.7 on every row, by a wide margin on several: Terminal Bench 2.1 66.0 against 51.1,
SWE-fficiency 34.8 against 14.0, KernelBench Hard 28.8 against 10.5. On Terminal Bench 2.1 the card
puts M3 level with Claude Opus 4.7 at 66.1, though GPT 5.5 leads at 78.2.

Two caveats on this table. The card states these were run on MiniMax's own infrastructure using Claude
Code as the scaffolding with the default system prompt overridden, and that some comparison columns
are taken from other leaderboards rather than rerun, so it is not a like-for-like harness across
columns. And this is a different Terminal-Bench version from the Kimi K3 and Hy4 tables above only in
part: those report 2.1 as well, so **those rows are comparable**, unlike the Kimi K3.1 Pro figure,
which uses Terminal-Bench 4.0.

### No published scores

`GLM-4.6`, `Qwen3-235B-A22B` and `Qwen3-Coder-480B` ship no benchmark table. `Qwen3.8-2.4T-A95B` names
GPQA Diamond and SWE-bench Pro on its card but publishes no scores in text, so nothing is quoted for it
here either. The Qwen3-Coder card points to
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
