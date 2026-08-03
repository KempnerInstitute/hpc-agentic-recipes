# HPC Agentic Recipes

Recipes for serving open-weight models on the Kempner AI Cluster and driving them with Claude Code or any
OpenAI-compatible client. Each runnable recipe is self-contained. A few entries in the table below are
documentation only.

Both vLLM and SGLang are used here, and both serve an Anthropic-compatible endpoint that Claude Code talks
to directly. vLLM runs most recipes. SGLang runs Kimi-K3 from a container. See
[docs/engines.md](docs/engines.md).

## If someone is already serving a model

Set five variables:

```
export ANTHROPIC_BASE_URL=http://<node>:8000
export ANTHROPIC_AUTH_TOKEN=<the api key>
export ANTHROPIC_MODEL=<served model name>
export ANTHROPIC_SMALL_FAST_MODEL=<the same name>
export CLAUDE_CODE_ATTRIBUTION_HEADER=0
claude
```

Use `ANTHROPIC_AUTH_TOKEN`, not `ANTHROPIC_API_KEY`, or every request returns 401. Get the served model
name from the endpoint:

```
curl -s -H "Authorization: Bearer $ANTHROPIC_AUTH_TOKEN" http://<node>:8000/v1/models
```

Walkthrough in [docs/quickstart.md](docs/quickstart.md).

## To serve your own

First create an API key. The name is the recipe path with a hyphen, so `recipes/GLM-5.2-NVFP4/rtx-8` reads:

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/GLM-5.2-NVFP4-rtx-8.key
chmod 600 secrets/GLM-5.2-NVFP4-rtx-8.key
```

Every recipe names its file at the top of its README. `secrets/vllm_api_key` is read when that file is
absent, so a single key there still gates everything.

With a key in place, requests without it receive HTTP 401. `secrets/` is gitignored. To rotate, replace the
file and restart: the engine reads it once at launch.

Then start with a single-GPU recipe, which needs one GPU rather than a whole node and so queues fastest:
[recipes/gemma-4-26B-A4B-it/h200-1](recipes/gemma-4-26B-A4B-it/h200-1/README.md). Follow it from the top.
Runnable recipes all have the same steps: configure once, build the environment, launch, verify, connect.

Each launches two ways: an sbatch submission and a direct launch on nodes you already hold.

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

`Status` is defined in [docs/choosing-a-model.md](docs/choosing-a-model.md).

## Weights

Checkpoints are read from `MODELS_DIR`, which defaults to
`/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models`. Every model a recipe references is already
there. That directory is read-only for most users, so download your own copies anywhere you can write and
point `MODELS_DIR` there; scratch also loads faster.
[docs/downloading-weights.md](docs/downloading-weights.md) covers both.

## Hardware

| Type | GPUs per node | Memory per GPU | Partition |
| --- | --- | --- | --- |
| RTX PRO 6000 Blackwell | 8 | 96 GiB | `kempner_rtx` |
| H200 | 4 | 140 GiB | `kempner_h200` |
| H100 | 4 | 80 GiB | `kempner_h100` |
| A100 | 4 | 40 GiB | `kempner` |

Details, including allocation limits and interconnect, in [docs/hardware.md](docs/hardware.md).

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
