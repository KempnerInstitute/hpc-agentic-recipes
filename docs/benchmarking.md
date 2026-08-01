# Measuring decode rate

## Decide which number you want first

There are two, and they are not comparable.

**Single stream** is one request at a time. It is what an interactive coding session feels, and it is
the right number for agentic work where one person waits on one response. Every figure in this repo's
model table is single stream.

**Saturated** is total tokens per second across many simultaneous requests. Continuous batching decodes
many sequences in one forward pass, so aggregate throughput keeps climbing with concurrency until
compute, memory bandwidth, or KV cache space runs out. It is the right number when one endpoint serves
several people.

The gap is large, and larger than a three-point measurement suggests. GLM-5.2-NVFP4 on one RTX node
measured 93.4 tok/s single stream and 1389 tok/s aggregate at concurrency 256, a factor of 15. Sweeping
only to concurrency 32 would have reported 683 and understated the endpoint's capacity by half. Single
stream measurement leaves the GPU nowhere near saturated, which is correct for latency and misleading
for capacity.

Sweep far enough to see throughput turn over, and be explicit when you have not. A first sweep here
reached concurrency 512 and left 12 of 14 recipes still climbing, so those figures were floors rather
than ceilings. Extending to 1024 in 128 increments resolved most of them: 6 recipes turned over at a
measured peak, 4 proved flat to within 4 percent across concurrency 512 to 1024, and 4 were still rising
at 1024 and remain floors. Publishing the first sweep's maxima as ceilings would have understated six
recipes and overstated nothing, which is the safer direction to be wrong in but still wrong.

Report the sequence cap next to the rate. vLLM sets `max_num_seqs` from device memory when the flag is
absent, and on every GPU here that resolves to 1024, read from four running engines rather than from
source. So this sweep ran entirely below the cap. Forcing the cap down does throttle the result:
gemma-4-26B on one RTX GPU gave 5429 tok/s at concurrency 512 at the default and 4290 with the cap at 256,
with the running batch reaching 512 and 256 respectively. A rate quoted without its cap is not
reproducible.

What binds first is model-dependent, so measure it rather than assuming. On gemma-4-26B on one RTX GPU, KV
cache usage sat at 99 to 100 percent from concurrency 256 upward and requests queued on KV blocks. On
Qwen3-235B across a whole node, KV usage stayed near 24 percent and the sequence count was the limit.
Neither preempted at any level.

```
bash common/tools/bench.sh --host <node> --model <name>                   single stream
bash common/tools/bench.sh --host <node> --model <name> --concurrency 16  saturated at 16
bash common/tools/bench.sh --host <node> --model <name> --sweep 1,4,16,32 find the plateau
```

## Prompt length is a third axis

The slope method cancels prefill, so prompt length never distorts the measurement. Whether it changes
the result is an empirical question, and the answer depends on the model's attention design rather than
on any general rule.

Measured on GLM-5.2-NVFP4, one RTX node, concurrency 1:

| ISL, input tokens | Decode | TTFT |
| --- | --- | --- |
| 21 | 97.0 tok/s | 91 ms |
| 7052 | 97.8 tok/s | 123 ms |
| 26379 | 95.0 tok/s | 131 ms |

Decode is flat to 26K within noise. Only time to first token grows, which is prefill doing more work.
The reason is architectural: MLA compresses the KV cache, this recipe stores it as fp8, and DeepSeek
sparse attention reads only a subset of the context, so the per-step read barely grows with length.

Do not generalize that. A dense model with full attention and a bf16 KV cache reads the whole cache
every step and should degrade noticeably. Measure with `--prompt-tokens` rather than assuming, in either
direction. The tool reports the prompt length the server actually counted, not an estimate.

## Use the slope method

```
bash common/tools/bench.sh --host <node> --model <served-name>
```

It times the same greedy request at 128 and 1152 output tokens, three times, and reports

```
rate = (1152 - 128) / (t_1152 - t_128)
```

along with the literal protocol string `slope(128,1152)` to paste into a recipe README.

## Why not just time one generation

A single timed generation includes prefill, scheduling, detokenization, and HTTP overhead, and divides all
of it by the output token count as though it were decode. Subtracting two lengths cancels every cost that
does not scale with output tokens.

This is not a small correction. On one H100 measurement here, a 256-token generation
reported 116.2 tok/s where the sustained rate was 183.9 tok/s. Short generations can understate decode by up to 40 percent, and
the error grows as the generation gets shorter.

`bench.sh --single` still offers the old behavior, and prints a warning naming the bias, because it is
occasionally useful for measuring what a user actually experiences on a short reply.

## Making runs comparable

Both requests use `temperature 0` and `ignore_eos` so they generate exactly the requested number of
tokens deterministically. Without `ignore_eos` the model may stop early and the two timings would measure
different amounts of work.

Warm the endpoint first, which `bench.sh` does with a throwaway 64-token request. A cold endpoint pays
first-time kernel compilation, and on a fresh environment FlashInfer compiles sm_120 kernels from source
on the first request.

Run on an otherwise idle endpoint. Concurrent traffic inflates both timings unequally, and `bench.sh` will
warn and skip a run if the long request did not take longer than the short one.

## Recording a result

A recipe's Measured performance section states the number, the protocol, and what did not help. The audit
refuses a `Validated` status without a date, an engine version, and a protocol label, precisely so that a
number measured one way is never silently compared against a number measured the other way.
