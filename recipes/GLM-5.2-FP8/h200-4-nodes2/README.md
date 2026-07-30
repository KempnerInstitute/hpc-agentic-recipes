# GLM-5.2-FP8 on two H200 nodes

Status: Untested (migrated) - numbers below were measured 2026-07-19 with the pre-restructure scripts, protocol: single-generation

Everything needed to build, launch, verify, connect to, and debug this endpoint is on this page.

## Configure once

Create the API key. The endpoint refuses requests without it, and the key is passed through the
environment rather than the command line so it never appears in `ps` output.

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/vllm_api_key
chmod 600 secrets/vllm_api_key
```

Nothing else is required. Cluster paths come from `common/defaults.sh`, which is tracked with working
defaults, so a fresh clone runs as is. Optional overrides, either exported or set in
`common/site.conf`:

| Variable | Default | Why you might change it |
| --- | --- | --- |
| `ACCOUNT` | unset | Your Slurm account, or pass `--account` at submit time |
| `GLM52_HEAD`, `GLM52_WORKER` | unset | Two nodes you already hold, for the SSH path |
| `MODELS_DIR` | shared testbed path | Point at your own faster copy of the checkpoint |
| `ENV_ROOT` | VAST scratch | Where this recipe builds its environment |

One constraint is specific to a two-node recipe: `ENV_ROOT` and `MODELS_DIR` must both resolve to the
same content on both nodes. The Ray worker imports vLLM from `ENV_ROOT` and reads weights from
`MODELS_DIR` itself, so a path that exists only on the head node fails after the cluster has already
formed, which reads as an engine problem rather than a path problem.

## Status

Untested (migrated). This configuration ran end to end before the restructure, twice on 2026-07-19 on
vLLM 0.25.1, on two different node pairs, and the decode rate below came from those runs. What has not
been done is a run of the recipe in this directory: the environment has not been built from
`env/build.sh`, and `serve.sbatch` and `serve_ssh.sh` here have not been submitted. The engine
invocation, the flags, and every warning on this page were carried over from the scripts that produced
those runs.

The rate was measured with the older single-generation protocol rather than the slope method, so it is
conservative and not directly comparable to the slope-measured numbers elsewhere in this repo.

## What this is

GLM-5.2, FP8 quantized: a 744B-parameter mixture-of-experts reasoning and coding model that uses
DeepSeek-style sparse attention for long context. It is a thinking model with native tool calling, and
it exposes vLLM's Anthropic-compatible API, so Claude Code connects to it directly with no proxy. Its
reason to exist in this repo is context length: this is the only endpoint here whose checkpoint reaches
a million tokens.

- Checkpoint directory: `GLM-5.2-FP8`
- Hugging Face repo: `zai-org/GLM-5.2-FP8`, unverified, see below
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/GLM-5.2-FP8`

The repo id is the one place on this page that is not sourced from a measurement or a file. The
pre-restructure repo recorded no Hugging Face id for this checkpoint, only the local path it was
downloaded to. `zai-org/GLM-5.2-FP8` is the id implied by the sibling checkpoints, since the FP8 build
of GLM-4.6 in this repo is `zai-org/GLM-4.6-FP8` and the NVFP4 build of this same base model was
quantized from `zai-org/GLM-5.2`, but it has not been confirmed against the Hub. Verify before relying
on it for a download.

Read from the checkpoint on 2026-07-29:

| Property | Value |
| --- | --- |
| Architecture | `GlmMoeDsaForCausalLM`, `model_type` `glm_moe_dsa`, native to vLLM 0.25.1 |
| On disk | about 756 GB across 141 shards, 704 GiB |
| Layers | 78 |
| Experts | 256 routed plus 1 shared, 8 routed per token, `moe_intermediate_size` 2048 |
| Attention | 64 heads, `hidden_size` 6144, sparse indexer with `index_n_heads` 32, `index_head_dim` 128, `index_topk` 2048 |
| Context | `max_position_embeddings` 1048576 |
| Quantization | `quant_method` `fp8`, `fmt` `e4m3`, `weight_block_size` [128, 128], dynamic activations |
| Speculative head | `num_nextn_predict_layers` 1, so an MTP head ships in the checkpoint |
| License | MIT |

The MTP head cannot be used in this configuration, and the reason is structural rather than a missing
flag. See "Parallelism and quantization".

The testbed path works out of the box. Copying the checkpoint into your own VAST scratch space loads
faster, because VAST outperforms Lustre for this workload, and the directory names are identical in both
locations so only `MODELS_DIR` changes. Scratch has a 90-day retention policy, so treat it as a fast
cache and keep testbed as the system of record.

## Hardware

| Requirement | Value |
| --- | --- |
| GPU | H200, 4 per node, 143771 MiB each |
| Nodes | 2, so 8 GPUs and about 1123 GiB of VRAM |
| Parallelism | TP4 inside each node, PP2 between them, over Ray |
| Interconnect | NVLink within a node, InfiniBand between nodes |
| Partition | `kempner_h200` |
| Per-GPU allocation limit | 16 CPUs, 360 GB host memory |
| Maximum wall time | 2 days |

All H200 nodes on this cluster share one hardware specification, so any two nodes in the partition
work. About 756 GB of weights, 704 GiB, does not fit one node's four cards at 143771 MiB each, which is
562 GiB, and that is what forces two nodes and therefore pipeline parallelism. `serve.sbatch` requests
48 CPUs and 500 GB per node, both inside the per-GPU limits for four GPUs.

The two nodes are not symmetric in what they hold. On 2026-07-19 each rank of the head stage loaded
91.31 GiB of weights and had 38.25 GiB left for KV cache, while each rank of the worker stage loaded
84.71 GiB and had 29.99 GiB, because pipeline parallelism splits layers between stages and the stages are
not identical in weight footprint. That is expected, not a misconfiguration.

## Environment build

This recipe builds its own environment, shared with no other recipe. Roughly 13 GB, and it lands under
`ENV_ROOT` on VAST scratch rather than in the repo, because startup is dominated by page faulting the
torch shared objects and stat-ing tens of thousands of small package files: measured on one node,
importing torch and vLLM took about 14 minutes from Lustre and 9.2 seconds from VAST.

```
bash recipes/GLM-5.2-FP8/h200-4-nodes2/env/build.sh
```

That is the only supported build path, because the install needs uv flags a requirements file cannot
express. Build it once, from a login node or either compute node: `ENV_ROOT` is shared scratch, so both
nodes then activate the same environment. `ray[default]` is installed alongside vLLM, because this
recipe spans two nodes through the Ray executor and `ray start` needs those extras. What else it does,
and why:

<!-- issue:hopper-cu129-wheel begin -->
**Hopper nodes need the cu129 wheel, not vLLM's default.** These nodes run NVIDIA driver 575
(CUDA 12.9), which cannot run vLLM's default CUDA 13 PyPI wheel. The recipe installs the cu129
release wheel from the vLLM GitHub release with `--torch-backend=cu129`.
<!-- issue:hopper-cu129-wheel end -->

`build.sh` treats an existing directory as suspect rather than as success: it checks for the installed
`vllm` dist-info and rebuilds anything less, because an interrupted install leaves a venv skeleton
behind that would otherwise be served from silently.

Scratch expires after 90 days, so this environment is disposable. Rebuild it with the same command, or
`--force` to replace an existing one. `env/requirements.lock` records the exact resolution that was
tested, once one is recorded for this recipe.

## Launch

Canonical path, submitted from the repo root:

```
sbatch --account=<your-account> recipes/GLM-5.2-FP8/h200-4-nodes2/serve.sbatch
```

The batch script allocates two nodes, reads each one's ib0 address, starts a Ray head on the first and a
worker on the second, then runs the engine on the head. The endpoint is on the **first** allocated
node:

```
squeue --me                       # NODELIST column, first name
tail -f glm52-fp8-<jobid>.log
```

Secondary path, for two nodes you already hold. Reserved nodes are removed from the scheduler, which is
why this exists:

```
bash recipes/GLM-5.2-FP8/h200-4-nodes2/serve_ssh.sh <head_node> <worker_node>
```

That path prints the cluster's GPU count before it loads any weights. It should say 8. If it says 4 the
worker did not join, and stopping there is much cheaper than discovering it 15 minutes into a weight
load.

Submit from the repo root either way. Slurm stages the batch script into its own spool directory, so
the script cannot locate the repo from its own path and resolves paths against the submit directory
instead.

`ray_head.sh` and `ray_worker.sh` in this directory are the Ray bring-up, and both launch paths call
them rather than duplicating the commands. They take their GPU count from `GPUS_PER_NODE`, then
`SLURM_GPUS_ON_NODE`, then 4, instead of the hardcoded 4 the pre-restructure scripts used.

## Verify

```
KEY=$(cat secrets/vllm_api_key)
NODE=<the head node serving it>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models          # must print 401

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"What is 2+2? Answer briefly."}],"max_tokens":400}'
```

A keyless request returning 401 is the expected, correct behavior, and it was confirmed on both
2026-07-19 runs.

<!-- issue:thinking-model-max-tokens begin -->
**Give thinking models room, or `content` comes back empty.** This model emits reasoning before its
answer, and vLLM returns that in a separate `reasoning` field (not `reasoning_content`). With a small
budget the whole allowance is spent reasoning, `finish_reason` is `length` or `stop_reason` is
`max_tokens`, and `content` is empty, which looks like a broken endpoint but is not. Measured on
2026-07-29: GLM-4.6 consumed a full 400-token budget on reasoning alone and returned no answer. Use at
least 400 output tokens for a smoke test and 800 or more for a model that reasons at length. If
`content` is empty, raise the budget before suspecting the endpoint.
<!-- issue:thinking-model-max-tokens end -->

## Connect a client

```
export NODE=<the head node serving it>
source recipes/GLM-5.2-FP8/h200-4-nodes2/client.env
claude
```

<!-- issue:anthropic-auth-token begin -->
**Use `ANTHROPIC_AUTH_TOKEN`, never `ANTHROPIC_API_KEY`.** vLLM accepts only
`Authorization: Bearer <key>`. Setting `ANTHROPIC_API_KEY` makes Claude Code send an `x-api-key`
header instead, which vLLM ignores, and every request returns HTTP 401. Also set
`ANTHROPIC_SMALL_FAST_MODEL` to this same served model, or the client reaches for a hosted Haiku that
this endpoint does not serve.
<!-- issue:anthropic-auth-token end -->

Point the client at the head node. The worker node runs Ray and holds half the layers, but it serves no
HTTP endpoint and will refuse the connection.

For an OpenAI-compatible client instead (Cline, Aider, Continue, OpenHands), use base URL
`http://<head-node>:8000/v1`, the same key, and model name `glm-5.2`.

## Tunable inputs

Every variable this recipe honors, with its default and effect.

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/GLM-5.2-FP8` | Serve a different copy of the checkpoint |
| `API_PORT` | 8000 | Listening port |
| `MAX_MODEL_LEN` | 131072 | Context window; the checkpoint supports 1048576, at a KV cache cost |
| `GPU_UTIL` | 0.90 | Fraction of VRAM for weights plus KV cache |
| `TP` | 4 | Tensor parallel size; 4 is one node's GPU count and must stay inside a node |
| `PP` | 2 | Pipeline parallel size; 2 is the node count |
| `PERF` | unset | Retry CUDA graph capture instead of eager; currently crashes |
| `CUDAGRAPH_MODE` | `NONE` | Graph mode passed through when `PERF` is set |
| `EXTRA_ARGS` | unset | Extra `vllm serve` flags, for experiments |
| `TOOL_PARSER` | `glm45` | Tool call parser |
| `REASONING_PARSER` | `glm45` | Reasoning parser |
| `RAY_PORT` | 6379 | Ray head port |
| `RAY_HEAD_IP` | unset | Head address, when calling `serve.sh` directly |
| `GPUS_PER_NODE` | `SLURM_GPUS_ON_NODE`, then 4 | GPUs each Ray node advertises |
| `RAY_BLOCK` | unset in the SSH path, 1 under Slurm | Keep `ray start` in the foreground |
| `GLM52_HEAD`, `GLM52_WORKER` | unset | Default nodes for the SSH path |
| `LOG_DIR` | `/tmp/$USER/vllm` | Where the SSH path writes the server log |
| `VENV_DIR` | under `ENV_ROOT` | Use an environment built elsewhere |

There is deliberately no `NO_MTP` switch, unlike the single-node GLM-4.6 recipe. The pre-restructure
launcher forwarded one, but nothing consumed it, because speculative decoding is unavailable here for
the reason in the next section but one.

## Web search

<!-- issue:anthropic-hosted-tools-400 begin -->
**Anthropic's hosted tools fail against this endpoint with HTTP 400.** Claude Code's built-in web
search sends tool definitions of type `web_search_20250305` that carry no `input_schema`, and vLLM
rejects them:

```
API Error: 400 1 validation error: 'loc': ('body', 'tools', 0, 'input_schema'),
'msg': 'Field required', 'type': 'web_search_20250305'
```

Client-side tools (file edits, shell, and anything you define) work normally. For web access, install
the repo's keyless search tool and skill:

```
ln -sf "$REPO_ROOT/common/tools/search.sh" ~/.local/bin/search.sh
cp -r "$REPO_ROOT/.claude/skills/local-search" ~/.claude/skills/
```

Then the model searches through `search.sh` (web, arxiv, crossref, pubmed, openalex, wiki, fetch)
instead of the hosted tool.
<!-- issue:anthropic-hosted-tools-400 end -->

## Measured performance

| Configuration | Decode rate | Protocol |
| --- | --- | --- |
| TP4 x PP2, eager, 2 nodes, no speculative decoding | about 13 tok/s | single-generation |

Measured 2026-07-19. The server's own throughput logging over several requests across both runs
reported 12.5 to 13.1 tokens per second of generation, which agrees with the figure. Context window
131072 as served, against a checkpoint that supports 1048576.

Re-measure with the slope method, which cancels prefill and fixed per-request cost:

```
bash common/tools/bench.sh --host <head-node> --model glm-5.2
```

Expect the correction to be small here. The slope method matters most for fast models: at 13 tok/s a
1152-token generation takes about 90 seconds, so a fraction of a second of fixed cost is noise. On the
sibling GLM-4.6 recipe the same correction moved 18 to 18.5 tok/s.

This is the slowest endpoint in this repo, and the two-node pipeline is why. Every token crosses the
InfiniBand link between the two pipeline stages, and with a single request in flight there is no second
micro-batch to overlap that hop with, so the second stage waits on the first. Pipeline parallelism pays
off in aggregate throughput under concurrency, not in single-stream latency, and interactive coding is
single-stream. If you want speed from GLM-5.2, the NVFP4 build on one RTX node measured about 90 tok/s;
if you want a million tokens of context, this is the only recipe that has it.

What is not available here: MTP speculative decoding, CUDA graphs, and torch.compile. All three are
ruled out, the first by topology and the other two by crashes, so there is no easy decode win left in
this configuration.

## Parallelism and quantization

The shape is TP4 inside each node and PP2 between the two nodes, over Ray. It is the only shape that
works: 756 GB of weights needs both nodes, tensor parallelism cannot cross the node boundary without
hanging, so what crosses is pipeline parallelism.

TP4 satisfies the FP8 block constraint. The weights are block quantized with `weight_block_size`
[128, 128], so every tensor-parallel shard of a quantized dimension must be a multiple of 128, and
`moe_intermediate_size` is 2048:

| Tensor parallel size | Shard of 2048 | Multiple of 128 | Verdict |
| --- | --- | --- | --- |
| 4 | 512 | yes, 4 x 128 | legal, and what this recipe uses |
| 8 | 256 | yes, 2 x 128 | legal on paper, but cross-node TP hangs at NCCL init |

So divisibility is not what rules out TP8 across both nodes; the fabric is. That distinction matters
because the SGLang variant of this recipe does attempt TP8 across two nodes, and its problem is
elsewhere entirely.

FP8 with dynamic activations is what makes 744B parameters fit eight H200 GPUs at all, at
`--gpu-memory-utilization 0.90`. `--disable-custom-all-reduce` is passed because vLLM's custom
all-reduce kernel is a single-node optimization and this engine spans two.

**The checkpoint's MTP head is dead weight in this configuration.** `num_nextn_predict_layers` is 1, so
a speculative head ships with the weights, and vLLM 0.25.1 can drive it: the sibling GLM-4.6 recipe
gets its decode speedup exactly that way. It is unusable here because a speculative config and
pipeline parallelism are mutually exclusive in vLLM, and pipeline parallelism is not optional at this
model size on this hardware. The one engine that can use the head at TP8 across two nodes is SGLang,
which is why `recipes/GLM-5.2-FP8/h200-4-nodes2-sglang` exists as a documented negative result rather
than being deleted.

## Gotchas

<!-- issue:h200-cudagraph-crash begin -->
**CUDA graphs and torch.compile crash on H200 for this model, so eager is the default.** Graph capture
hits an illegal memory access on vLLM 0.25.1, so `serve.sh` passes `--enforce-eager`. Set `PERF=1` to
retry the compile path after a vLLM upgrade.
<!-- issue:h200-cudagraph-crash end -->

<!-- issue:deepgemm-h200-crash begin -->
**Leave `VLLM_USE_DEEP_GEMM` at 0 on H200.** The DeepGEMM MoE path takes an illegal memory access on
GLM-5.2's sparse attention, and forcing `VLLM_USE_DEEP_GEMM=1` on H200 independently reproduced the
same crash for Qwen3-Coder-480B-FP8. It is load-bearing for more than one model on this hardware, so
do not flip it without re-testing the model you are serving.
<!-- issue:deepgemm-h200-crash end -->

This model is where that DeepGEMM crash was first diagnosed, which is why `env/env.sh` sets
`VLLM_USE_DEEP_GEMM=0` with a verified provenance note rather than an inherited one.

<!-- issue:cross-node-tp-hangs begin -->
**Keep tensor parallelism inside a node and use pipeline parallelism across nodes.** Pure tensor
parallelism spanning two nodes hangs at NCCL initialization. The working shape is TP within each node,
where all-reduce uses NVLink, and PP between nodes.
<!-- issue:cross-node-tp-hangs end -->

<!-- issue:pp-forbids-spec-decode begin -->
**Pipeline parallelism disables speculative decoding.** vLLM rejects a speculative config when
pipeline parallelism is in use, so no MTP or draft-model speedup is available in any recipe that needs
PP to span nodes. A checkpoint that ships an MTP head cannot use it in this configuration. This was
observed when configuring GLM-5.2 across two H200 nodes, and it is the reason the SGLang recipe exists
at all, since SGLang can run TP8 across two nodes with EAGLE instead. Note that the guard was not
located in vLLM 0.25.1's config source, so treat it as observed behavior for this version rather than a
documented API contract, and re-check after an engine upgrade.
<!-- issue:pp-forbids-spec-decode end -->

<!-- issue:jit-cache-node-local begin -->
**JIT caches must be node-local.** Concurrent multi-node compiles against a shared NFS home hit stale
file handles. `env/env.sh` points `TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR` at
`/tmp/$USER`, which also means the first launch on a fresh node pays the compile cost again.
<!-- issue:jit-cache-node-local end -->

<!-- issue:lustre-watchdog begin -->
**A storage stall can kill the endpoint even after it recovers.** PyTorch kills the process when the
NCCL watchdog thread stops sending heartbeats, on the assumption that a collective hung. A stalled
network filesystem freezes every rank the same way, so at the 480 second default a storage outage
that later recovers still takes the endpoint down permanently. Observed on 2026-07-29: a holylfs06
OSS failover froze two unrelated endpoints on two nodes within one second of each other, and both
were killed by their own watchdog eight minutes later while reporting `Last enqueued NCCL work: -1`,
meaning no collective was ever in flight. `env/env.sh` sets
`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600` so a transient stall is survivable.
<!-- issue:lustre-watchdog end -->

<!-- issue:engine-ready-timeout begin -->
**Startup exceeds vLLM's default readiness timeout.** Weight load plus torch.compile plus CUDA graph
capture routinely takes longer than the 600 second default, so `env/env.sh` sets
`VLLM_ENGINE_READY_TIMEOUT_S=3600`. A first launch that looks hung is usually still loading; check
the log before killing it.
<!-- issue:engine-ready-timeout end -->

<!-- issue:node-local-logs begin -->
**Logs are written to node-local `/tmp`, not to the repo.** Every rank writes stderr for the life of
the endpoint, so a log on a network filesystem puts a blocking write on the critical path. During a
filesystem stall that write hangs, which freezes the server. `LOG_DIR` defaults to
`/tmp/$USER/vllm`, so read logs over SSH on the node that runs the server.
<!-- issue:node-local-logs end -->

**A Ray worker can die under you, and the head log will not say why.** The longest recorded run of this
endpoint served requests intermittently for about 21 hours on 2026-07-19 and then ended with
`RayWorkerProc rank=[1] died unexpectedly, shutting down executor`, followed by
`RuntimeError: Executor failed`. Rank 1 is the second pipeline stage, on the worker node. The head's log
records the death and no cause, because the cause was on the other node, and Ray's own worker logs under
`/tmp/ray` on the worker were not captured. If this happens to you, look there before theorizing:
`ssh <worker_node> 'ls -t /tmp/ray/session_latest/logs | head'`. A single-node recipe cannot fail this
way, which is why it is worth saying here.

**Both nodes need a clean slate before a relaunch.** A leftover Ray cluster on either node is the most
common cause of a second launch that never becomes ready, and it looks nothing like a Ray problem from
the client side: the engine simply waits for GPUs that a stale Ray head believes are already in use.

## Stop the endpoint

For a Slurm job, `scancel <jobid>`, which also tears down both Ray processes. For the SSH path, both
nodes need attention:

```
bash common/tools/stop.sh <head_node> <worker_node>
ssh <head_node> ray stop
ssh <worker_node> ray stop
```

`stop.sh` kills the server processes and waits for GPU memory to be released. Confirm with
`ssh <node> nvidia-smi --query-gpu=memory.used --format=csv,noheader`, which should read 0 MiB on all
four GPUs of both nodes before you relaunch, or the next start will fail on memory. `ray stop` is listed
separately because `stop.sh` does not touch Ray, and a surviving Ray cluster fails the next launch on
resources instead of on memory.

## Expected startup time

| Stage | Warm nodes | Cold nodes |
| --- | --- | --- |
| Environment build, one time | skipped | 5 to 15 min |
| Ray head plus worker bring-up | seconds | seconds |
| Weight load, 141 shards across 8 ranks | 39 s | 14 min 23 s |
| Engine init: profile, KV cache, warmup | 109 s | 84 s |
| Total, vLLM banner to serving | 5 min 5 s | 19 min 36 s |

Both columns are measured, on 2026-07-19, on two different node pairs, with the checkpoint read from
VAST scratch in both cases. The difference is page cache: the fast pair had already loaded this
checkpoint, the freshly allocated pair had not. Reading from the Lustre testbed path instead is slower
again.

Nothing in this window looks like progress if you only watch the port, so watch the log. A first launch
that appears hung is almost always still loading weights, and `VLLM_ENGINE_READY_TIMEOUT_S=3600` in
`env/env.sh` exists precisely because the cold case exceeds vLLM's 600 second default.
