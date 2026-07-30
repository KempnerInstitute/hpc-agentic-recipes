# Measuring decode rate

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
