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

Every recipe launches two ways. **Slurm submission is the default** and is what you want: any cluster
user can submit it and the scheduler picks the node. **Direct SSH is the advanced path**, for people who
already hold nodes through a reservation and for administrators standing up an endpoint on behalf of
others, since reserved nodes are removed from the scheduler and cannot be reached with sbatch.

Choosing between models is [docs/choosing-a-model.md](docs/choosing-a-model.md). Highest decode rate is
not the same as best at coding: the fastest model here activates only 4B parameters.

## Models

Every rate below was measured on 2026-07-31 with `common/tools/bench.sh` at concurrency 1, 8, 32, 64,
128, 256 and 512, using slope(128,1152) over output tokens only, 3 repeats per level, median reported.

| Model | Precision | Hardware | Parallelism | Single stream | Saturated | Context | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [GLM-5.2](recipes/GLM-5.2-NVFP4/rtx-8) | NVFP4 | 1 RTX node, 8 GPUs | TP8 | 93.4 tok/s | 1389 at c=256, peak | 128K | Validated |
| [GLM-5.2](recipes/GLM-5.2-FP8/h200-4-nodes2) | FP8 | 2 H200 nodes | TP4 x PP2 | 13.0 tok/s | 5405 at c=512, rising | 1M | Validated |
| [GLM-4.6](recipes/GLM-4.6-FP8/h200-4) | FP8 | 1 H200 node, 4 GPUs | TP4 | 19.2 tok/s | 6300 at c=512, rising | 200K | Validated |
| [Kimi-K2.7-Code](recipes/Kimi-K2.7-Code/rtx-8) | INT4 | 1 RTX node, 8 GPUs | TP8 | 20.7 tok/s | 1819 at c=512, rising | 32K | Validated |
| [Kimi-K2.7-Code](recipes/Kimi-K2.7-Code/h200-4-nodes2) | INT4 | 2 H200 nodes | TP4 x PP2 | 30.4 tok/s | 5669 at c=512, rising | 32K | Validated |
| [Qwen3-235B-A22B](recipes/Qwen3-235B-A22B/rtx-8) | bf16 | 1 RTX node, 8 GPUs | TP8 | 63.3 tok/s | 3984 at c=512, rising | 40K | Validated |
| [Qwen3-Coder-480B](recipes/Qwen3-Coder-480B-A35B-Instruct-FP8/rtx-8) | FP8 | 1 RTX node, 8 GPUs | TP4 x PP2 | 67.7 tok/s | 3040 at c=512, rising | 128K | Validated |
| [DeepSeek-V4-Pro](recipes/DeepSeek-V4-Pro/rtx-8-nodes2) | FP8 with FP4 experts | 2 RTX nodes | TP8 x PP2 | 18.6 tok/s | 2959 at c=512, rising | 1M | Validated |
| [Gemma-4-26B-A4B](recipes/gemma-4-26B-A4B-it/h200-1) | bf16 | 1 H200 GPU | TP1 | 236.3 tok/s | 10727 at c=256, peak | 32K | Validated |
| [Gemma-4-26B-A4B](recipes/gemma-4-26B-A4B-it/h100-1) | bf16 | 1 H100 GPU | TP1 | 203.4 tok/s | 7165 at c=512, rising | 32K | Validated |
| [Gemma-4-26B-A4B](recipes/gemma-4-26B-A4B-it/rtx-1) | bf16 | 1 RTX GPU | TP1 | 140.6 tok/s | 5404 at c=512, rising | 32K | Validated |
| [Gemma-4-31B](recipes/gemma-4-31B-it/h200-1) | FP8 | 1 H200 GPU | TP1 | 85.1 tok/s | 3097 at c=512, rising | 32K | Validated |
| [Gemma-4-31B](recipes/gemma-4-31B-it/h100-1) | FP8 | 1 H100 GPU | TP1 | 67.4 tok/s | 2680 at c=512, rising | 32K | Validated |
| [Gemma-4-31B](recipes/gemma-4-31B-it/rtx-1) | FP8 | 1 RTX GPU | TP1 | 39.5 tok/s | 2101 at c=512, rising | 32K | Validated |
| [Qwen3-Coder-480B](recipes/Qwen3-Coder-480B-A35B-Instruct-FP8/h200-4) | FP8 | 1 H200 node, 4 GPUs | TP4 | does not run | n/a | n/a | Blocked, serve on RTX |
| [DeepSeek-V4-Pro](recipes/DeepSeek-V4-Pro/h200-4-nodes2) | FP8 with FP4 experts | 2 H200 nodes | TP4 x PP2 | does not run | n/a | n/a | Blocked, serve on RTX |
| [GLM-5.2 on SGLang](recipes/GLM-5.2-FP8/h200-4-nodes2-sglang) | FP8 | 2 H200 nodes | TP4 x PP2 | never loaded weights | n/a | n/a | Blocked |
| [Qwen3-Coder-480B](recipes/Qwen3-Coder-480B-A35B-Instruct) | bf16 | 2 to 4 H200 nodes | TP4 x PP | not measured | n/a | 256K | Untested |
| [Kimi-K3](recipes/Kimi-K3) | MXFP4, QAT | 4 H200 nodes, 16 GPUs | TP16 | not measured | n/a | 1M | Blocked, no vLLM support |

Three columns deserve care. **Single stream** is what one person waiting on one response experiences, and
it is the only number that describes interactive coding. **Saturated** is total output across all streams
at the stated concurrency, which here runs from 15x the single stream rate to 416x, and says nothing
about how fast a reply feels. Never quote one where the other belongs. The ratio is largest for the slow
big-MoE recipes and smallest for GLM-5.2-NVFP4, whose speculative decoding already spends spare capacity
on latency rather than leaving it for extra streams. `rising` means throughput had not yet turned over at
concurrency 512, the top of the sweep, so that figure is a floor rather than a ceiling; only the two
marked `peak` were measured past their maximum. These numbers were taken at vLLM's default
`max_num_seqs`, which resolves to 1024 on every GPU here, so the sweep ran entirely below the cap and no
level was queue-limited. Forcing the cap below the requested concurrency does throttle the result:
gemma-4-26B on one RTX GPU measured 5429 tok/s at concurrency 512 at the default and 4290 with the cap at
256. Which resource binds first is model-dependent, so measure rather than assume: that recipe held KV
cache usage at 99 to 100 percent from concurrency 256 upward, while Qwen3-235B across a whole node stayed
near 24 percent. Neither preempted at any level.

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
common/tools/                             for recipe users: bench.sh, search.sh, stop.sh,
                                          rebuild_envs.sh
                                          for maintainers: audit_recipes.sh, new_recipe.sh
common/skills/local-search/               skill that makes a client use search.sh
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
