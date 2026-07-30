# Choosing a model

Decode rate is the wrong first question. The fastest model here activates 4B parameters and is not the
one you want for hard agentic work.

## Start here

**Best default for interactive coding on one GPU:** Gemma-4-26B-A4B. It is a mixture of experts with 4B
active parameters, so it is fast and cheap to schedule, and one GPU means the shortest queue wait.

**Strongest coder overall:** Kimi-K2.7-Code, a 1T-parameter MoE quantized to INT4. Use it when quality
matters more than latency and you can hold a full node.

**Largest coding model that fits a single node:** Qwen3-Coder-480B-FP8 on one RTX node, at a decode rate
close to the much smaller Qwen3-235B, which makes it a strong quality-per-second choice.

**Best reasoning and the fastest large model measured here:** GLM-5.2 in NVFP4 on one RTX node, at 101.6 tok/s measured with the slope method, helped by MTP speculative decoding that works because the model fits one node with no pipeline
parallelism.

**Longest context:** GLM-5.2-FP8 across two H200 nodes reaches 1M tokens, at a much lower decode rate.

## The table

The full comparison lives in the top-level README so that the entry point and the numbers stay together.
Each row links to a recipe, and every recipe repeats its own number with its measurement protocol.

## Read the protocol column

Rates measured with different protocols are not comparable:

- `slope(128,1152)` times the same request at two output lengths and divides the difference, which
  cancels prefill and fixed per-request cost. This is the honest sustained decode rate.
- `single-generation` times one request and divides tokens by wall time, counting prefill and fixed cost
  as decode. It understates sustained decode by up to 40 percent.

Where a model shows a single-generation number, treat it as a conservative floor. Re-measuring with
`common/tools/bench.sh` will usually produce a higher figure. Details in [benchmarking.md](benchmarking.md).

## Read the status column

`Validated` means the recipe in this repo was run end to end on the stated date with the stated engine
version. `Untested (migrated)` means the number is real but was measured with the older pre-restructure
scripts, so the recipe itself has not been exercised. `Blocked` means it does not currently run at all,
and the recipe says why.

## Three performance regimes, and why flags help or do not

Measured here, and the reason the same flag helps one model and does nothing for another:

**Memory-bandwidth bound.** A dense model such as Gemma-4-31B reads every weight for every token. FP8
weights halve the bytes moved and gave a large speedup, scaling with the GPU's memory bandwidth.

**Host-overhead bound.** A sparse MoE such as Gemma-4-26B-A4B activates 4B of its parameters, so the GPU
finishes each step before the host can feed it the next. Utilization sat at 35 to 40 percent and power at
roughly 210 W of a 700 W budget. FP8 gave exactly zero benefit, because bandwidth was never the limit.

**Communication bound.** A large model at TP8 on an RTX node spends about 16 ms per token on all-reduce
against roughly 3 ms of weight reading, because there is no NVLink. FP8 weights again bought nothing, and
expert parallelism made it worse by adding all-to-all traffic.

The practical rule: FP8 weights pay off for dense models on high-bandwidth GPUs, and do nothing for sparse
MoE models or for anything already limited by the interconnect.
