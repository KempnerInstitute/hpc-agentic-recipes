# Local Agentic Coding

Recipes for serving open-weight coding models on the Kempner AI Cluster and driving them with Claude Code
or any OpenAI-compatible client. Each runnable recipe is self-contained: one directory holds the environment
build, the launch scripts, the measured performance, and every known failure mode for one model on one GPU
configuration. A few entries in the table below are documentation only, recording why a configuration does
not work; those ship no scripts and say so.

Every runnable recipe uses vLLM, which serves an Anthropic-compatible endpoint that Claude Code talks to
directly. SGLang appears twice in the table and runs neither model here; see
[docs/engines.md](docs/engines.md).

## If someone is already serving a model

Set four variables:

```
export ANTHROPIC_BASE_URL=http://<node>:8000
export ANTHROPIC_AUTH_TOKEN=<the api key>
export ANTHROPIC_MODEL=<served model name>
export ANTHROPIC_SMALL_FAST_MODEL=<the same name>
claude
```

Use `ANTHROPIC_AUTH_TOKEN`, not `ANTHROPIC_API_KEY`, or every request returns 401. Get the served model
name from the endpoint:

```
curl -s -H "Authorization: Bearer $ANTHROPIC_AUTH_TOKEN" http://<node>:8000/v1/models
```

Walkthrough in [docs/quickstart.md](docs/quickstart.md).

## To serve your own

First create an API key. The recipes pass it through the environment so it never appears in `ps` output or
in a tracked file:

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/vllm_api_key
chmod 600 secrets/vllm_api_key
```

With a key in place, requests without it receive HTTP 401. Without one the launcher prints a warning and
serves the endpoint **ungated**, so create the file before launching on a shared network. `secrets/` is
gitignored. To rotate, replace the file and restart. The SGLang recipe passes no key at all.

Then start with a single-GPU recipe, which needs one GPU rather than a whole node and so queues fastest:
[recipes/gemma-4-26B-A4B-it/h200-1](recipes/gemma-4-26B-A4B-it/h200-1/README.md). Follow it from the top.
Runnable recipes all have the same steps: configure once, build the environment, launch, verify, connect.
Run them from the repo root.

Each launches two ways. Use the Slurm path unless you already hold nodes through a reservation, which puts
them outside the scheduler.

To choose a model, see [docs/choosing-a-model.md](docs/choosing-a-model.md). The fastest model here is not
the best coder.

## Models

Rates measured with `common/tools/bench.sh`; method in [docs/benchmarking.md](docs/benchmarking.md).
`c=256` means 256 concurrent requests. `Context` is what the recipe serves by default; raise it with
`MAX_MODEL_LEN` up to what the checkpoint supports, which each recipe states.

| Model | Precision | Hardware | Parallelism | Single stream | Aggregate | Context | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [GLM-5.2](recipes/GLM-5.2-NVFP4/rtx-8) | NVFP4 | 1 RTX node, 8 GPUs | TP8 | 93.4 tok/s | 1389 at c=256, peak | 128K | Validated |
| [GLM-5.2](recipes/GLM-5.2-FP8/h200-4-nodes2) | FP8 | 2 H200 nodes | TP4 x PP2 | 13.0 tok/s | 5421 at c=640, peak | 128K | Validated |
| [GLM-4.6](recipes/GLM-4.6-FP8/h200-4) | FP8 | 1 H200 node, 4 GPUs | TP4 | 19.2 tok/s | 8130 at c=1024, rising | 128K | Validated |
| [Kimi-K2.7-Code](recipes/Kimi-K2.7-Code/rtx-8) | INT4 | 1 RTX node, 8 GPUs | TP8 | 20.7 tok/s | 1839 at c=896, saturated | 32K | Validated |
| [Kimi-K2.7-Code](recipes/Kimi-K2.7-Code/h200-4-nodes2) | INT4 | 2 H200 nodes | TP4 x PP2 | 30.4 tok/s | 7140 at c=1024, rising | 32K | Validated |
| [Qwen3-235B-A22B](recipes/Qwen3-235B-A22B/rtx-8) | bf16 | 1 RTX node, 8 GPUs | TP8 | 63.3 tok/s | 3984 at c=512, peak | 40K | Validated |
| [Qwen3-Coder-480B](recipes/Qwen3-Coder-480B-A35B-Instruct-FP8/rtx-8) | FP8 | 1 RTX node, 8 GPUs | TP4 x PP2 | 67.7 tok/s | 3238 at c=768, peak | 128K | Validated |
| [DeepSeek-V4-Pro](recipes/DeepSeek-V4-Pro/rtx-8-nodes2) | FP8 with FP4 experts | 2 RTX nodes | TP8 x PP2 | 18.6 tok/s | 3582 at c=1024, rising | 128K | Validated |
| [Gemma-4-26B-A4B](recipes/gemma-4-26B-A4B-it/h200-1) | bf16 | 1 H200 GPU | TP1 | 236.3 tok/s | 10727 at c=256, peak | 32K | Validated |
| [Gemma-4-26B-A4B](recipes/gemma-4-26B-A4B-it/h100-1) | bf16 | 1 H100 GPU | TP1 | 203.4 tok/s | 7243 at c=640, saturated | 32K | Validated |
| [Gemma-4-26B-A4B](recipes/gemma-4-26B-A4B-it/rtx-1) | bf16 | 1 RTX GPU | TP1 | 140.6 tok/s | 5972 at c=1024, rising | 32K | Validated |
| [Gemma-4-31B](recipes/gemma-4-31B-it/h200-1) | FP8 | 1 H200 GPU | TP1 | 85.1 tok/s | 3136 at c=1024, saturated | 32K | Validated |
| [Gemma-4-31B](recipes/gemma-4-31B-it/h100-1) | FP8 | 1 H100 GPU | TP1 | 67.4 tok/s | 2680 at c=512, peak | 32K | Validated |
| [Gemma-4-31B](recipes/gemma-4-31B-it/rtx-1) | FP8 | 1 RTX GPU | TP1 | 39.5 tok/s | 2139 at c=768, saturated | 32K | Validated |
| [Qwen3-Coder-480B](recipes/Qwen3-Coder-480B-A35B-Instruct-FP8/h200-4) | FP8 | 1 H200 node, 4 GPUs | TP4 | 22.2 tok/s eager only | n/a | n/a | Blocked, serve on RTX |
| [DeepSeek-V4-Pro](recipes/DeepSeek-V4-Pro/h200-4-nodes2) | FP8 with FP4 experts | 2 H200 nodes | TP4 x PP2 | does not run | n/a | n/a | Blocked, serve on RTX |
| [GLM-5.2 on SGLang](recipes/GLM-5.2-FP8/h200-4-nodes2-sglang) | FP8 | 2 H200 nodes | TP4 x PP2 | never loaded weights | n/a | n/a | Blocked |
| [Qwen3-Coder-480B](recipes/Qwen3-Coder-480B-A35B-Instruct) | bf16 | 2 to 4 H200 nodes | TP4 x PP | not measured | n/a | n/a | Untested |
| [Kimi-K3](recipes/Kimi-K3) | MXFP4, QAT | 4 H200 nodes, 16 GPUs | TP16 x EP16 | 40.3 tok/s | 1405 at c=128 | n/a | Blocked for vLLM, SGLang only |

**Single stream** is one request at a time, and it is the only number that describes how fast a reply feels
to one person. **Aggregate** is total output across every concurrent stream at the stated concurrency, tens
to hundreds of times larger.

Each label is computed from the numbers in that recipe's own table, as the spread across the levels from
concurrency 512 upward, `(max - min) / max`. Under 4 percent is `saturated`, meaning more concurrency buys
only queueing delay. Otherwise `rising` if the highest value sits at the top of the sweep, so the figure is
a floor, and `peak` if throughput turned over before then. `Status` is defined in
[docs/choosing-a-model.md](docs/choosing-a-model.md).

Kimi-K3 carries no label: it was measured under SGLang on a shorter sweep, capped at 156 concurrent
requests, so a rule defined from concurrency 512 upward does not apply. Two other rows also stopped at 512
because they turned over at 256.

## Hardware

| Type | GPUs per node | Memory per GPU | Partition |
| --- | --- | --- | --- |
| RTX PRO 6000 Blackwell | 8 | 96 GiB | `kempner_rtx` |
| H200 | 4 | 140 GiB | `kempner_h200` |
| H100 | 4 | 80 GiB | `kempner_h100` |

All nodes of a given type are identical, so any node in a partition works. Details, including allocation
limits and interconnect, in [docs/hardware.md](docs/hardware.md).

## Layout

```
recipes/<Checkpoint-Name>/<hardware>/   one self-contained recipe
common/                                 shared config, libraries, and tools
docs/                                   guides, indexed below
```

Hardware directory names are `<gpu-type>-<gpus-per-node>[-nodes<N>][-<engine>]`, so `rtx-8` is one full RTX
node, `h200-4-nodes2` is two H200 nodes, and `h100-1` is a single H100. Documentation-only entries have no
hardware level, just the checkpoint directory.

## Guides

| Guide | For |
| --- | --- |
| [quickstart.md](docs/quickstart.md) | Connecting to an endpoint someone else runs |
| [choosing-a-model.md](docs/choosing-a-model.md) | Which model to serve, and what the rates mean |
| [clients.md](docs/clients.md) | Cline, Aider, Continue, or any OpenAI-compatible client |
| [engines.md](docs/engines.md) | Why vLLM is the default and when SGLang is worth it |
| [hardware.md](docs/hardware.md) | Node types, partitions, allocation limits |
| [downloading-weights.md](docs/downloading-weights.md) | Fetching a new checkpoint |
| [benchmarking.md](docs/benchmarking.md) | How the rates here were measured, and how to measure your own |
| [web-search.md](docs/web-search.md) | Giving a local model web search, which the built-in tool cannot do |
| [troubleshooting.md](docs/troubleshooting.md) | Symptoms across all recipes, for pattern spotting |
| [adding-a-model.md](docs/adding-a-model.md) | Contributing a recipe |

## Contributing a model

[docs/adding-a-model.md](docs/adding-a-model.md) is the checklist. Run `common/tools/audit_recipes.sh`
before opening a pull request.

## Weights

Checkpoints are read from `MODELS_DIR`, which defaults to
`/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models`. Copying one to your own scratch space loads
faster and only `MODELS_DIR` changes.
[docs/downloading-weights.md](docs/downloading-weights.md) covers fetching new checkpoints.
