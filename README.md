# Local Agentic Coding

Recipes for serving open-weight coding models on the Kempner AI Cluster and driving them with Claude Code
or any OpenAI-compatible client. Each runnable recipe is self-contained: one directory holds the environment
build, the launch scripts, the measured performance, and every known failure mode for one model on one GPU
configuration. A few entries in the table below are documentation only, recording why a configuration does
not work; those ship no scripts and say so.

Both vLLM and SGLang are used here. vLLM serves an Anthropic-compatible endpoint that Claude Code talks to
directly, and runs most recipes here. SGLang serves Kimi-K3, which no vLLM release available here can
load, from a container over an OpenAI-compatible API. See [docs/engines.md](docs/engines.md).

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
gitignored. To rotate, replace the file and restart. The Kimi-K3 recipe gates its port the same way;
the blocked GLM-5.2 SGLang recipe passes no key at all.

Then start with a single-GPU recipe, which needs one GPU rather than a whole node and so queues fastest:
[recipes/gemma-4-26B-A4B-it/h200-1](recipes/gemma-4-26B-A4B-it/h200-1/README.md). Follow it from the top.
Runnable recipes all have the same steps: configure once, build the environment, launch, verify, connect.
Run them from the repo root.

Each launches two ways: an sbatch submission and a direct launch on nodes you already hold. Use the
sbatch path unless you have a reason not to.

To choose a model, see [docs/choosing-a-model.md](docs/choosing-a-model.md). The fastest model here is not
the best coder.

## Models

Rows are grouped by model, one per hardware configuration, and the hardware cell links to that recipe.
Rates for the vLLM recipes are measured with `common/tools/bench.sh` and Kimi-K3 with the same protocol
under SGLang; method in [docs/benchmarking.md](docs/benchmarking.md). `c=256` means 256 concurrent
requests. `Context` is what the recipe serves by default; raise it with `MAX_MODEL_LEN` up to what the
checkpoint supports, which each recipe states.

| Model | Precision | Hardware | Parallelism | Single stream | Aggregate | Context | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **GLM-5.2** | NVFP4 | [1 RTX node, 8 GPUs](recipes/GLM-5.2-NVFP4/rtx-8) | TP8 | 93.4 tok/s | 1389 at c=256, peak | 128K | Validated |
| | FP8 | [2 H200 nodes](recipes/GLM-5.2-FP8/h200-4-nodes2) | TP4 x PP2 | 13.0 tok/s | 5421 at c=640, peak | 128K | Validated |
| | FP8 | [2 H200 nodes, SGLang](recipes/GLM-5.2-FP8/h200-4-nodes2-sglang) | TP8, no PP | never loaded weights | n/a | n/a | Blocked |
| **GLM-4.6** | FP8 | [1 H200 node, 4 GPUs](recipes/GLM-4.6-FP8/h200-4) | TP4 | 19.2 tok/s | 8130 at c=1024, rising | 128K | Validated |
| **Kimi-K2.7-Code** | INT4 | [1 RTX node, 8 GPUs](recipes/Kimi-K2.7-Code/rtx-8) | TP8 | 20.7 tok/s | 1839 at c=896, saturated | 32K | Validated |
| | INT4 | [2 H200 nodes](recipes/Kimi-K2.7-Code/h200-4-nodes2) | TP4 x PP2 | 30.4 tok/s | 7140 at c=1024, rising | 32K | Validated |
| **Kimi-K3** | MXFP4, QAT | [4 H200 nodes, 16 GPUs, SGLang](recipes/Kimi-K3/h200-4-nodes4-sglang) | TP16 x EP16 | 40.2 tok/s | 1069 at c=64, capped | 256K | Validated |
| **Qwen3-235B-A22B** | bf16 | [1 RTX node, 8 GPUs](recipes/Qwen3-235B-A22B/rtx-8) | TP8 | 63.3 tok/s | 3984 at c=512, peak | 40K | Validated |
| **Qwen3-Coder-480B** | FP8 | [1 RTX node, 8 GPUs](recipes/Qwen3-Coder-480B-A35B-Instruct-FP8/rtx-8) | TP4 x PP2 | 67.7 tok/s | 3238 at c=768, peak | 128K | Validated |
| | FP8 | [1 H200 node, 4 GPUs](recipes/Qwen3-Coder-480B-A35B-Instruct-FP8/h200-4) | TP4 | 22.2 tok/s eager only | n/a | n/a | Blocked, serve on RTX |
| | bf16 | [2 to 4 H200 nodes](recipes/Qwen3-Coder-480B-A35B-Instruct) | TP4 x PP | not measured | n/a | n/a | Untested |
| **DeepSeek-V4-Pro** | FP8 with FP4 experts | [2 RTX nodes](recipes/DeepSeek-V4-Pro/rtx-8-nodes2) | TP8 x PP2 | 18.6 tok/s | 3582 at c=1024, rising | 128K | Validated |
| | FP8 with FP4 experts | [2 H200 nodes](recipes/DeepSeek-V4-Pro/h200-4-nodes2) | TP4 x PP2 | does not run | n/a | n/a | Blocked, serve on RTX |
| **Gemma-4-26B-A4B** | bf16 | [1 H200 GPU](recipes/gemma-4-26B-A4B-it/h200-1) | TP1 | 236.3 tok/s | 10727 at c=256, peak | 32K | Validated |
| | bf16 | [1 H100 GPU](recipes/gemma-4-26B-A4B-it/h100-1) | TP1 | 203.4 tok/s | 7243 at c=640, saturated | 32K | Validated |
| | bf16 | [1 RTX GPU](recipes/gemma-4-26B-A4B-it/rtx-1) | TP1 | 140.6 tok/s | 5972 at c=1024, rising | 32K | Validated |
| **Gemma-4-31B** | FP8 | [1 H200 GPU](recipes/gemma-4-31B-it/h200-1) | TP1 | 85.1 tok/s | 3136 at c=1024, saturated | 32K | Validated |
| | FP8 | [1 H100 GPU](recipes/gemma-4-31B-it/h100-1) | TP1 | 67.4 tok/s | 2680 at c=512, peak | 32K | Validated |
| | FP8 | [1 RTX GPU](recipes/gemma-4-31B-it/rtx-1) | TP1 | 39.5 tok/s | 2139 at c=768, saturated | 32K | Validated |

**Single stream** is one request at a time, and it is the only number that describes how fast a reply feels
to one person. **Aggregate** is total output across every concurrent stream at the stated concurrency, tens
to hundreds of times larger.

Each label comes from that recipe's own table. Where the sweep has two or more levels at concurrency 512
and above, a spread of `(max - min) / max` under 4 percent across them is `saturated`, meaning more
concurrency buys only queueing delay. Otherwise the label is `rising` if the highest value sits at the top
of the sweep, so the figure is a floor, and `peak` if throughput turned over before then. Two sweeps
stopped at 512 because throughput had already turned over at 256, and both are `peak`. `Status` is defined
in [docs/choosing-a-model.md](docs/choosing-a-model.md).

Kimi-K3 is labeled `capped` rather than by the rule above: its defaults admit only 67 concurrent
requests, set by the KDA state pool, so the sweep stops at 64 and a rule defined from concurrency 512
upward cannot apply. Its row is the default configuration; the recipe also measures a speculative setting
reaching 94.1 tok/s single stream and a wide setting reaching 1442.6 tok/s at concurrency 156. Its
`Context` cell is the 256K the measured runs served; the checkpoint supports 1M.

## Hardware

| Type | GPUs per node | Memory per GPU | Partition |
| --- | --- | --- | --- |
| RTX PRO 6000 Blackwell | 8 | 96 GiB | `kempner_rtx` |
| H200 | 4 | 140 GiB | `kempner_h200` |
| H100 | 4 | 80 GiB | `kempner_h100` |
| A100 | 4 | 40 GiB | `kempner` |

All nodes of a given type are identical, so any node in a partition works. Details, including allocation
limits and interconnect, in [docs/hardware.md](docs/hardware.md).

No recipe here targets the A100 yet. It is listed because it is available and may suit a smaller
checkpoint later.

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
| [engines.md](docs/engines.md) | Which engine a model needs, and what each one serves |
| [hardware.md](docs/hardware.md) | Node types, partitions, allocation limits |
| [downloading-weights.md](docs/downloading-weights.md) | Fetching a new checkpoint |
| [benchmarking.md](docs/benchmarking.md) | How the rates here were measured, and how to measure your own |
| [web-search.md](docs/web-search.md) | Giving a local model web search, which the built-in tool cannot do |
| [troubleshooting.md](docs/troubleshooting.md) | Symptoms across all recipes, for pattern spotting |
| [adding-a-model.md](docs/adding-a-model.md) | Contributing a recipe |

## Contributing a model

[docs/adding-a-model.md](docs/adding-a-model.md) is the checklist. Run `common/tools/audit_recipes.sh`
before opening a pull request.

## License

MIT, in [LICENSE](LICENSE). Model weights are not covered by it: each checkpoint carries its own license
from whoever published it, and the recipe for that model names the Hugging Face repo where it is stated.

## Weights

Checkpoints are read from `MODELS_DIR`, which defaults to
`/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models`. Every model a recipe references is already
there. That directory is read-only for most users, so download your own copies anywhere you can write and
point `MODELS_DIR` there; scratch also loads faster.
[docs/downloading-weights.md](docs/downloading-weights.md) covers both.
