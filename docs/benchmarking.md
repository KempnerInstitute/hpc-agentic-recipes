# Measuring decode rate

## Decide which number you want first

There are two, and they are not comparable.

**Single stream** is one request at a time. It is what an interactive coding session feels, and it is the
right number for agentic work where one person waits on one response.

**Aggregate** is total tokens per second across many simultaneous requests. Continuous batching decodes many
sequences in one forward pass, so aggregate throughput keeps climbing with concurrency until compute, memory
bandwidth, or KV cache space runs out. It is the right number when one endpoint serves several people.

The gap is large. GLM-5.2-NVFP4 on one RTX node measures 91.1 tok/s single stream and 1374.5 tok/s aggregate at
concurrency 256, a factor of 15.

```
bash common/tools/bench.sh --host <node> --model <name>                        single stream
bash common/tools/bench.sh --host <node> --model <name> --concurrency 16       aggregate at 16
bash common/tools/bench.sh --host <node> --model <name> --sweep 1,8,32,128     find where it turns over
```

## Sweep until throughput turns over

A sweep that stops early reports a floor, not a ceiling, and the two look identical. That same GLM-5.2-NVFP4
endpoint reports 674 tok/s if the sweep stops at concurrency 32, half its actual capacity at 256. Of the 14
rates the labeling rule covers, 5 turn over at a peak, 5 are flat to within 4 percent between 512 and 1024,
and 4 were still climbing at the top of their sweep, so those figures are floors. Twelve sweeps ran to 1024
and two stopped at 512.

**Report the sequence cap next to the rate.** vLLM sets `max_num_seqs` from device memory when the flag is
absent, and that resolves to 1024 on every GPU here. It binds: every vLLM recipe that swept to 1024 reached
a running batch of 1024 there. The SGLang recipe is the exception, admitting 67 requests, which is why its
rate is labeled capped. A rate quoted without its cap is not reproducible, and forcing the cap down
throttles the result.

**What binds first is model-dependent, so measure it.** KV cache usage reached 100 percent in most recipes,
so requests queue on KV blocks. It reached only 78 percent for DeepSeek-V4-Pro across two RTX nodes and 73
percent for Kimi-K2.7-Code across two H200 nodes, where the sequence cap alone is the limit.

Preemption depends on the recipe, and the gemma recipes now preempt heavily at a 262144 context. The 31B on
H100 reached 110,565 preemptions across one sweep, with the cache at 100 percent from concurrency 256 upward,
the 31B on H200 124,318, and the 26B on RTX 11,070 in a single run at concurrency 1024. Where that happens the sequence cap is not
what bounds the top of the curve.

## Prompt length is a third axis

The slope method cancels prefill, so prompt length never distorts the measurement. Whether it changes the
result depends on the model's attention design.

GLM-5.2-NVFP4, one RTX node, concurrency 1. All three rows come from one run, separate from the sweep in
that recipe, so compare them against each other rather than against the recipe's 91.1 tok/s:

| ISL, input tokens | Decode | TTFT |
| --- | --- | --- |
| 21 | 97.0 tok/s | 91 ms |
| 7052 | 97.8 tok/s | 123 ms |
| 26379 | 95.0 tok/s | 131 ms |

Decode is flat to 26K within noise; only time to first token grows, which is prefill doing more work. That
is architectural: MLA compresses the KV cache, this recipe stores it in DeepSeek's sparse-MLA fp8 layout, and
sparse attention reads only a subset of the context, so the per-step read barely grows with length.

Do not generalize it. A dense model with full attention and a BF16 KV cache reads the whole cache every step
and should degrade. Measure with `--prompt-tokens`; the tool reports the prompt length the server actually
counted, not an estimate.

## Use the slope method

```
bash common/tools/bench.sh --host <node> --model <served-name>
```

It times the same greedy request at 128 and 1152 output tokens, three times, and reports

```
rate = concurrency * (1152 - 128) / (t_1152 - t_128)
```

along with the protocol string `slope(128,1152)` to paste into a recipe README. It posts to
`/v1/chat/completions`, so it measures a vLLM or an SGLang endpoint the same way.

Timing one generation instead counts prefill, scheduling, detokenization, and HTTP overhead as decode.
Subtracting two lengths cancels every cost that does not scale with output tokens, whatever it happens to be.

How much that matters depends on the endpoint. Measured on H200 against a warm endpoint with a 19-token
prompt, both protocols agree closely, and the gap shrinks as the generation lengthens:

| Model | slope(128,1152) | single(128) | single(256) | single(512) |
| --- | --- | --- | --- | --- |
| gemma-4-26B-A4B | 236.4 tok/s | 233.1 | 235.4 | 236.3 |
| gemma-4-31B | 85.1 tok/s | 85.2 | 85.4 | 85.4 |

Those two rows were measured at a 32768 context, before both recipes moved to 262144.

The largest gap is 1.4 percent, on the faster model at the shortest generation, where fixed cost weighs
most. Prefer the slope method anyway: it removes that cost by construction, so you do not have to know
whether a cold endpoint, a long prompt or a short reply is about to make it matter.

`bench.sh --single` keeps the one-generation behavior and warns about the bias.

## Making runs comparable

Both requests use `temperature 0` and `ignore_eos`, so they generate exactly the requested token count
deterministically. Without `ignore_eos` the model may stop early and the two timings would measure different
amounts of work.

Warm the endpoint first, which `bench.sh` does with a throwaway 64-token request. A cold endpoint pays
first-time kernel compilation, and on a fresh environment FlashInfer compiles sm_120 kernels from source on
the first request.

Run against an otherwise idle endpoint. Concurrent traffic inflates both timings unequally, and `bench.sh`
warns and skips a run if the long request did not take longer than the short one.

## Recording a result

A recipe's Measured performance section states the rate, the concurrency, and the protocol, and a
`Validated` status has to name both an engine version and a protocol label. That is what stops a number
measured one way from being compared against one measured the other way.
