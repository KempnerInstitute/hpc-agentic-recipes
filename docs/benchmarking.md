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

The gap is large. Measured on an RTX node here: 90.0 tok/s single stream, and 404.6 tok/s aggregate at
concurrency 8, with each individual stream seeing 50.6 tok/s. Single stream measurement leaves the GPU
nowhere near saturated, which is correct for latency and misleading for capacity.

```
bash common/tools/bench.sh --host <node> --model <name>                   single stream
bash common/tools/bench.sh --host <node> --model <name> --concurrency 16  saturated at 16
bash common/tools/bench.sh --host <node> --model <name> --sweep 1,4,16,32 find the plateau
```

## Prompt length is a third axis

The slope method cancels prefill, so prompt length does not distort the measurement. But it is not
neutral to the result: attention reads the whole KV cache on every decode step, so a long context
genuinely slows decode. A rate measured with a 20-token prompt overstates what you will see at 30K.
Use `--prompt-tokens 8000` to measure at a realistic context before quoting a number for long-context
work. The tool reports the prompt length the server actually counted, not an estimate.

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

This is not a small correction. On one H100 measurement here, a 256-token generation reported 116.2 tok/s
where the sustained rate was 183.9 tok/s. Short generations can understate decode by up to 40 percent, and
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
