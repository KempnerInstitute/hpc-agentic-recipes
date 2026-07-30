# Local Agentic Coding

Recipes for serving open-weight coding models on the Kempner AI Cluster and driving them with Claude
Code or any OpenAI-compatible client. Each recipe is self-contained: one directory holds the
environment build, the launch scripts, the measured performance, and every known failure mode for one
model on one hardware shape. You should not need to read a second file to use one.

Both vLLM and SGLang are covered. vLLM is the default, because it exposes an Anthropic-compatible
`/v1/messages` endpoint that Claude Code uses directly with no proxy.

## Already have an endpoint

If a colleague is already serving a model, you need four variables and no build. About a minute.

```
export ANTHROPIC_BASE_URL=http://<node>:8000
export ANTHROPIC_AUTH_TOKEN=<the api key>
export ANTHROPIC_MODEL=<served model name, for example glm-5.2>
export ANTHROPIC_SMALL_FAST_MODEL=<the same name>
claude
```

Use `ANTHROPIC_AUTH_TOKEN`, not `ANTHROPIC_API_KEY`. The latter makes the client send an `x-api-key`
header, which vLLM ignores, and every request returns 401. Walkthrough in
[docs/quickstart.md](docs/quickstart.md).

## Want your own endpoint

Start with a single-GPU recipe. It queues fastest and needs one GPU rather than a whole node:

```
# recipes are run from the repo root, so read this one and follow its commands from here
less recipes/gemma-4-31B-it/h200-1/README.md
```

Then follow that recipe's README from the top. Every recipe has the same shape: configure once, build
the environment, launch, verify, connect.

Choosing between models is [docs/choosing-a-model.md](docs/choosing-a-model.md). Highest decode rate is
not the same as best at coding: the fastest model here activates only 4B parameters.

## Models

| Model | Precision | Hardware | Parallelism | Decode | Protocol | Context | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [GLM-5.2](recipes/GLM-5.2-NVFP4/rtx-8) | NVFP4 | 1 RTX node, 8 GPUs | TP8 | 101.6 tok/s | slope(128,1152) | 128K | Validated 2026-07-29 |
| [GLM-5.2](recipes/GLM-5.2-FP8/h200-4-nodes2) | FP8 | 2 H200 nodes | TP4 x PP2 | 13.0 tok/s | slope(128,1152) | 1M | Validated 2026-07-29 |
| [GLM-4.6](recipes/GLM-4.6-FP8/h200-4) | FP8 | 1 H200 node, 4 GPUs | TP4 | 18.5 tok/s | slope(128,1152) | 200K | Validated 2026-07-29 |
| [Kimi-K2.7-Code](recipes/Kimi-K2.7-Code/rtx-8) | INT4 | 1 RTX node, 8 GPUs | TP8 | about 21 tok/s | single-generation | 32K | Untested (migrated) |
| [Kimi-K2.7-Code](recipes/Kimi-K2.7-Code/h200-4-nodes2) | INT4 | 2 H200 nodes | TP4 x PP2 | 30.0 tok/s | slope(128,1152) | 32K | Validated 2026-07-29 |
| [Qwen3-235B-A22B](recipes/Qwen3-235B-A22B/rtx-8) | bf16 | 1 RTX node, 8 GPUs | TP8 | about 63 tok/s | slope(128,1152) | 40K | Untested (migrated) |
| [Qwen3-Coder-480B](recipes/Qwen3-Coder-480B-A35B-Instruct-FP8/rtx-8) | FP8 | 1 RTX node, 8 GPUs | TP4 x PP2 | 63.9 tok/s | slope(128,1152) | 128K | Untested (migrated) |
| [Qwen3-Coder-480B](recipes/Qwen3-Coder-480B-A35B-Instruct-FP8/h200-4) | FP8 | 1 H200 node, 4 GPUs | TP4 | does not run | n/a | n/a | Blocked, serve on RTX |
| [Gemma-4-26B-A4B](recipes/gemma-4-26B-A4B-it) | bf16 | 1 GPU | TP1 | 140 to 236.1 tok/s | slope(128,1152) | 32K | h200-1 Validated 2026-07-29 |
| [Gemma-4-31B](recipes/gemma-4-31B-it) | FP8 | 1 GPU | TP1 | 40 to 85.2 tok/s | slope(128,1152) | 32K | h200-1 Validated 2026-07-29 |
| [DeepSeek-V4-Pro](recipes/DeepSeek-V4-Pro) | FP8 with FP4 experts | 2 RTX nodes | TP8 x PP2 | not measured | n/a | 1M | Untested on RTX, Blocked on H200 |
| [Qwen3-Coder-480B](recipes/Qwen3-Coder-480B-A35B-Instruct) | bf16 | 2 to 4 H200 nodes | TP4 x PP | not measured | n/a | 256K | Untested |
| [Kimi-K3](recipes/Kimi-K3) | MXFP4 | Blackwell only | n/a | n/a | n/a | 1M | Blocked upstream |

Two columns deserve care. **Protocol**: slope-measured numbers time two generation lengths and
subtract, which cancels prefill and fixed per-request cost, while single-generation numbers count those
as decode time and understate the rate by up to 40 percent. The two are not directly comparable, and
the single-generation figures are conservative. **Status**: `Untested (migrated)` means the number was
measured with the pre-restructure scripts and the recipe here has not itself been run end to end.

## Hardware

| Type | Per node | Notes |
| --- | --- | --- |
| RTX PRO 6000 Blackwell | 8 GPUs, 97887 MiB each | sm_120, CUDA 13, PCIe with no NVLink |
| H200 | 4 GPUs, 143771 MiB each | NVLink, CUDA 12.9, driver 575 |
| H100 | 4 GPUs, 80 GB each | NVLink, CUDA 12.9 |

Partitions are `kempner_rtx`, `kempner_h200`, and `kempner_h100`. Per-GPU allocation limits are 16 CPUs
on the RTX and H200 partitions and 24 on H100. Maximum wall time is 2 days. All nodes of a given type
share one hardware specification, so any node in a partition works. More in
[docs/hardware.md](docs/hardware.md).

## Layout

```
recipes/<Checkpoint-Name>/<hardware>/     one self-contained recipe
common/defaults.sh                        cluster paths, tracked so a fresh clone works
common/site.conf.example                  optional overrides: your nodes, your account
common/lib/                               repo root and API key resolution
common/issues/                            canonical failure-mode text plus the recipe matrix
common/tools/                             benchmarking, smoke tests, search, audit
docs/                                     cross-model guides
```

Hardware directory names are `<gpu-type>-<gpus-per-node>[-nodes<N>][-<engine>]`, so `rtx-8` is one full
RTX node, `h200-4-nodes2` is two H200 nodes, and `h100-1` is a single H100.

## Contributing a model

[docs/adding-a-model.md](docs/adding-a-model.md) is the checklist. In short: scaffold from the nearest
existing recipe, pin an environment, write the serve invocation, measure with `common/tools/bench.sh`,
write the README, then run `common/tools/audit_recipes.sh`, which enforces recipe structure, required
sections, propagated failure-mode text, flag provenance, and agreement between each recipe and the
tables in this file.

Recipes deliberately repeat the same warnings instead of linking to a shared page, so a reader of one
recipe sees everything that affects it. That duplication is generated from `common/issues/` and verified
by the audit, so it cannot silently drift.

## Access

Every vLLM endpoint here requires an API key, supplied through the environment so it never appears in
`ps` output or in a tracked file:

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/vllm_api_key
chmod 600 secrets/vllm_api_key
```

Requests without the key receive HTTP 401. `secrets/` is gitignored. To rotate, replace the file and
restart the endpoint. The one exception is the [SGLang recipe](recipes/GLM-5.2-FP8/h200-4-nodes2-sglang), which does not
currently gate its port and says so in its README.

## Weights

Checkpoints are read from `MODELS_DIR`, which defaults to
`/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models`. Copying a checkpoint to your own VAST
scratch space loads faster, and the directory names are identical in both locations, so only
`MODELS_DIR` changes. Scratch is a 90-day cache rather than the system of record.
[docs/downloading-weights.md](docs/downloading-weights.md) covers fetching new checkpoints.
