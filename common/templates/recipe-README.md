# <Model> on <hardware>

Status: Untested - not yet run end to end

Everything needed to build, launch, verify, connect to, and debug this endpoint is on this page.
Do not link out for anything required: repeat it here instead. Issue text is injected from
common/issues by common/tools/audit_recipes.sh --fix, so add empty marker pairs rather than prose:

    <!-- issue:<slug> begin -->
    <!-- issue:<slug> end -->

## Configure once

<!-- API key creation, and the variables this recipe reads. A fresh clone must work without edits. -->

## Status

<!-- Validated needs an engine version and a protocol label, or the audit rejects it. No dates: the
     prose checker rejects them, because a version says whether a number still holds and a date does not. -->

## What this is

<!-- Model, checkpoint directory, Hugging Face repo id, documented testbed path, faster-copy note. -->

## Hardware

<!-- GPU type, count, nodes, partition, per-GPU CPU and memory limits, max wall time. -->

## Environment build

<!-- The full command sequence inline, plus why any non-obvious step exists. -->

## Launch

<!-- The sbatch path from the repo root, then the direct SSH alternative. State the submit directory. -->

## Verify

<!-- curl commands, including that a keyless request must return 401. -->

## Connect a client

<!-- The full client block. Anthropic for vLLM, OpenAI-compatible for SGLang. -->

## Tunable inputs

<!-- Every variable this recipe honors, with default and effect. -->

## Web search

<!-- Required for vLLM recipes: the hosted-tool failure and the local replacement. -->

## Measured performance

<!-- Rate, protocol, and what was tried that did not help. -->

## Parallelism and quantization

<!-- Why this TP, PP and quantization, including divisibility constraints. -->

## Gotchas

<!-- Every issue from the matrix, in full. -->

## Stop the endpoint

<!-- Teardown, and confirming GPU memory was released before relaunching. -->

## Expected startup time

<!-- Environment build, weight load, JIT. Measure it or say to be measured. -->
