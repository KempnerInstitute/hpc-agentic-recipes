# Measuring decode rate

## Decide which number you want first

There are two, and they are not comparable.

**Single stream** is one request at a time. It is what an interactive coding session feels, and it is the
right number for agentic work where one person waits on one response.

**Aggregate** is total tokens per second across many simultaneous requests. Continuous batching decodes many
sequences in one forward pass, so aggregate throughput keeps climbing with concurrency until compute, memory
bandwidth, or KV cache space runs out. It is the right number when one endpoint serves several people.

The gap is large. GLM-5.2-NVFP4 on one RTX node measures 93.4 tok/s single stream and 1389 tok/s aggregate at
concurrency 256, a factor of 15.

```
bash common/tools/bench.sh --host <node> --model <name>                        single stream
bash common/tools/bench.sh --host <node> --model <name> --concurrency 16       aggregate at 16
bash common/tools/bench.sh --host <node> --model <name> --sweep 1,8,32,128     find where it turns over
```

## Sweep until throughput turns over

A sweep that stops early reports a floor, not a ceiling, and the two look identical. That same
GLM-5.2-NVFP4 endpoint reports 683 tok/s if the sweep stops at concurrency 32, half its actual capacity at
256. Of the 14 recipes here, sweeping to 1024 found 6 that turn over at a peak, 4 flat to within 4 percent
between 512 and 1024, and 4 still climbing at 1024 whose figures remain floors.

**Report the sequence cap next to the rate.** vLLM sets `max_num_seqs` from device memory when the flag is
absent, which is 1024 on every GPU here. Forcing it down throttles the result: gemma-4-26B on one RTX GPU
gives 5429 tok/s at concurrency 512 at the default and 4290 with the cap at 256. A rate quoted without its
cap is not reproducible.

**What binds first is model-dependent, so measure it.** On gemma-4-26B on one RTX GPU, KV cache usage sits at
99 to 100 percent from concurrency 256 upward and requests queue on KV blocks. On Qwen3-235B across a whole
node, KV usage stays near 24 percent and the sequence count is the limit. Neither preempts at any level.

## Prompt length is a third axis

The slope method cancels prefill, so prompt length never distorts the measurement. Whether it changes the
result depends on the model's attention design.

GLM-5.2-NVFP4, one RTX node, concurrency 1:

| ISL, input tokens | Decode | TTFT |
| --- | --- | --- |
| 21 | 97.0 tok/s | 91 ms |
| 7052 | 97.8 tok/s | 123 ms |
| 26379 | 95.0 tok/s | 131 ms |

Decode is flat to 26K within noise; only time to first token grows, which is prefill doing more work. That
is architectural: MLA compresses the KV cache, this recipe stores it in DeepSeek's sparse-MLA fp8 layout, and
sparse attention reads only a subset of the context, so the per-step read barely grows with length.

Do not generalize it. A dense model with full attention and a bf16 KV cache reads the whole cache every step
and should degrade. Measure with `--prompt-tokens`; the tool reports the prompt length the server actually
counted, not an estimate.

## Use the slope method

```
bash common/tools/bench.sh --host <node> --model <served-name>
```

It times the same greedy request at 128 and 1152 output tokens, three times, and reports

```
rate = (1152 - 128) / (t_1152 - t_128)
```

along with the protocol string `slope(128,1152)` to paste into a recipe README.

Timing one generation instead counts prefill, scheduling, detokenization, and HTTP overhead as decode.
Subtracting two lengths cancels every cost that does not scale with output tokens, whatever it happens to be.

How much that matters depends on the endpoint. Measured on H200 against a warm endpoint with a 19-token
prompt, both protocols agree closely, and the gap shrinks as the generation lengthens:

| Model | slope(128,1152) | single(128) | single(256) | single(512) |
| --- | --- | --- | --- | --- |
| gemma-4-26B-A4B | 236.4 tok/s | 233.1 | 235.4 | 236.3 |
| gemma-4-31B | 85.1 tok/s | 85.2 | 85.4 | 85.4 |

The largest gap is 1.4 percent, on the faster model at the shortest generation, which is where fixed cost
weighs most. So prefer the slope method because it removes fixed cost by construction rather than because it
will change your number here. It matters more on a cold endpoint, a long prompt, or a very short reply, and
the point of subtracting is that you do not have to know which of those applies.

`bench.sh --single` keeps the one-generation behavior and warns about the bias, which is occasionally what
you want if you are measuring a short reply as a user experiences it.

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

A recipe's Measured performance section states the rate, the concurrency, and the protocol.
`common/tools/audit_recipes.sh` refuses a `Validated` status that does not name both an engine version and a
protocol label, so a number measured one way is never silently compared against one measured the other way.
