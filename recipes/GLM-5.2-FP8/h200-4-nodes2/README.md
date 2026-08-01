# GLM-5.2-FP8 on two H200 nodes

Status: Validated - vLLM 0.25.1+cu129, protocol: slope(128,1152) swept at concurrency 1 through 1024

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
| `ENV_ROOT` | scratch | Where this recipe builds its environment |

One constraint is specific to a two-node recipe: `ENV_ROOT` and `MODELS_DIR` must both resolve to the
same content on both nodes. The Ray worker imports vLLM from `ENV_ROOT` and reads weights from
`MODELS_DIR` itself, so a path that exists only on the head node fails after the cluster has already
formed, which reads as an engine problem rather than a path problem.

## Status

Validated. The environment was built from `env/build.sh`, the endpoint was
launched with `serve_ssh.sh` on two H200 nodes, 8 GPUs via Ray, and throughput was measured with `common/tools/bench.sh`
across concurrency 1, 8, 32, 64, 128, 256, 512, 640, 768, 896 and 1024. Ready 16 minutes 23 seconds after launch. The endpoint was still answering after the sweep finished.

Single stream and saturated throughput are different measurements and neither substitutes for
the other. See Measured performance below for the full curve and the disclosure block.

## What this is

GLM-5.2, FP8 quantized: a 753B-parameter mixture-of-experts, 40B activated per token, reasoning and coding model that uses
DeepSeek-style sparse attention for long context. It is a thinking model with native tool calling, and
it exposes vLLM's Anthropic-compatible API, so Claude Code connects to it directly with no proxy. Its
reason to exist in this repo is context length: this is the only endpoint here whose checkpoint reaches
a million tokens.

- Checkpoint directory: `GLM-5.2-FP8`
- Hugging Face repo: `zai-org/GLM-5.2-FP8`, unverified, see below
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/GLM-5.2-FP8`

The repo id is the one place on this page that is not sourced from a measurement or a file. The
checkpoint was staged from a local path with no Hugging Face id recorded alongside it. `zai-org/GLM-5.2-FP8` is the id implied by the sibling checkpoints, since the FP8 build
of GLM-4.6 in this repo is `zai-org/GLM-4.6-FP8` and the NVFP4 build of this same base model was
quantized from `zai-org/GLM-5.2`, but it has not been confirmed against the Hub. Verify before relying
on it for a download.

Read from the checkpoint:

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

The testbed path works out of the box. Copying the checkpoint into your own scratch space loads
faster, because scratch outperforms Lustre for this workload, and the directory names are identical in both
locations so only `MODELS_DIR` changes. Scratch has a 90-day retention policy, so treat it as a fast
cache and keep testbed as the permanent copy.

## Hardware

| Requirement | Value |
| --- | --- |
| GPU | H200, 4 per node, 143771 MiB each |
| Nodes | 2, so 8 GPUs and about 1123 GiB of VRAM |
| Parallelism | TP4 inside each node, PP2 between them, over Ray |
| Interconnect | NVLink within a node, InfiniBand between nodes |
| Partition | `kempner_h200` |
| Per-GPU allocation limit | 16 CPUs, about 378 GiB host memory |
| Maximum wall time | 2 days |

All H200 nodes on this cluster share one hardware specification, so any two nodes in the partition
work. About 756 GB of weights, 704 GiB, does not fit one node's four cards at 143771 MiB each, which is
562 GiB, and that is what forces two nodes and therefore pipeline parallelism. `serve.sbatch` requests
48 CPUs and 500 GB per node, both inside the per-GPU limits for four GPUs.

The two nodes are not symmetric in what they hold. Each rank of the head stage loaded
91.31 GiB of weights and had 38.25 GiB left for KV cache, while each rank of the worker stage loaded
84.71 GiB and had 29.99 GiB, because pipeline parallelism splits layers between stages and the stages are
not identical in weight footprint. That is expected, not a misconfiguration.

## Environment build

This recipe builds its own environment, shared with no other recipe. Roughly 13 GB, and it lands under
`ENV_ROOT` on scratch rather than in the repo, because startup is dominated by page faulting the
torch shared objects and stat-ing tens of thousands of small package files: measured on one node,
importing torch and vLLM took about 14 minutes from Lustre and 9.2 seconds from scratch.

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

Slurm path, submitted from the repo root:

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

Direct path, for two nodes you already hold. Use the Slurm submission above unless you already have
the nodes, or you are deploying an endpoint on behalf of others:

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
`SLURM_GPUS_ON_NODE`, then 4, so a node with a different GPU count is not mis-advertised to Ray.

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
runs.

<!-- issue:thinking-model-max-tokens begin -->
**Give thinking models room, or `content` comes back empty.** This model emits reasoning before its
answer, and vLLM returns that in a separate `reasoning` field, not `reasoning_content`. With a small
budget the whole allowance is spent reasoning, `finish_reason` is `length`, and `content` is empty,
which looks like a broken endpoint but is not. Use at least 400 output tokens for a smoke test, and 800
or more for a model that reasons at length. If `content` is empty, raise the budget before suspecting
the endpoint.
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

There is deliberately no `NO_MTP` switch, unlike the single-node GLM-4.6 recipe. The earlier
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
cp -r "$REPO_ROOT/common/skills/local-search" ~/.claude/skills/
```

Then the model searches through `search.sh` (web, arxiv, crossref, pubmed, openalex, wiki, fetch)
instead of the hosted tool.
<!-- issue:anthropic-hosted-tools-400 end -->

## Measured performance

| Configuration | Aggregate rate | Per stream | Latency |
| --- | --- | --- | --- |
| Single stream, concurrency 1 | 13.0 tok/s | 13.0 tok/s | TTFT median 235 ms, n=3 spanning 12.9 to 13.0 |
| Concurrency 8 | 101.0 tok/s | 12.6 tok/s | TTFT median 239 ms, p90 295 ms, n=3 spanning 100.9 to 101.5 |
| Concurrency 32 | 393.1 tok/s | 12.3 tok/s | TTFT median 239 ms, p90 313 ms, n=3 spanning 393.0 to 396.9 |
| Concurrency 64 | 786.3 tok/s | 12.3 tok/s | TTFT median 384 ms, p90 1050 ms, n=3 spanning 785.4 to 797.8 |
| Concurrency 128 | 1575.3 tok/s | 12.3 tok/s | TTFT median 407 ms, p90 561 ms, n=3 spanning 1565.5 to 1591.0 |
| Concurrency 256 | 3066.0 tok/s | 12.0 tok/s | TTFT median 559 ms, p90 794 ms, n=3 spanning 3028.5 to 3079.8 |
| Concurrency 512 | 5404.6 tok/s | 10.6 tok/s | TTFT median 820 ms, p90 1252 ms, n=3 spanning 5391.0 to 5475.6 |
| Concurrency 640 (peak) | 5420.7 tok/s | 8.5 tok/s | TTFT median 938 ms, p90 1522 ms, n=3 spanning 5259.4 to 5425.6 |
| Concurrency 768 | 5034.0 tok/s | 6.6 tok/s | TTFT median 1189 ms, p90 1753 ms, n=3 spanning 4995.6 to 5034.5 |
| Concurrency 896 | 4942.5 tok/s | 5.5 tok/s | TTFT median 1222 ms, p90 1920 ms, n=3 spanning 4901.8 to 4953.4 |
| Concurrency 1024 | 5049.3 tok/s | 4.9 tok/s | TTFT median 1325 ms, p90 2092 ms, n=3 spanning 5032.8 to 5056.3 |

Measured with `common/tools/bench.sh`, endpoint ready 16m 23s after launch. Full disclosure, without which a tokens
per second figure cannot be compared against anything:

| Parameter | Value |
| --- | --- |
| ISL, input tokens | 21 |
| OSL, output tokens | 1152, as the slope between 128 and 1152 |
| Counted | output tokens only, never input plus output |
| Concurrency levels | 1,8,32,64,128,256,512,640,768,896,1024 |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| `max_num_seqs` | engine default, 1024 on this hardware |
| Hardware | two H200 nodes, 8 GPUs |

Quote 13.0 tok/s for interactive coding, where one person waits on one
response. Quote 5420.7 tok/s at concurrency 640 for a shared endpoint under load.
The two measure different things and neither substitutes for the other.

Throughput **peaks at concurrency 640** and falls to 5049 tok/s by concurrency 1024, so 5420.7 tok/s is a
measured ceiling for this recipe rather than the edge of the sweep.

Concurrency 512 was measured in both runs, at 5404.6 and 5416.6 tok/s, a +0.2 percent
difference. That is the check that the two halves of this curve are comparable.

Scheduler counters over the extended levels: KV cache usage reached 100 percent, the running
batch reached 1024 requests, and there were no preemptions at any level.

The input sequence here is short, which is the best case for decode. Measure with
`--prompt-tokens` at your working context before quoting a number for long-context work.

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

FP8 with dynamic activations is what makes 753B parameters fit eight H200 GPUs at all, at
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
PP to span nodes, even when the checkpoint ships an MTP head. This is why an SGLang recipe exists for
GLM-5.2: SGLang can run TP8 across two nodes with EAGLE speculative decoding, where vLLM would need PP
and therefore lose it. The guard is not visible in vLLM 0.25.1's config source, so treat it as behavior
for this version rather than a documented API contract, and re-check after an engine upgrade.
<!-- issue:pp-forbids-spec-decode end -->

<!-- issue:jit-cache-node-local begin -->
**JIT caches must be node-local.** Concurrent multi-node compiles against a shared NFS home hit stale
file handles. `env/env.sh` points `TRITON_CACHE_DIR` and `TORCHINDUCTOR_CACHE_DIR` at
`/tmp/$USER`, which also means the first launch on a fresh node pays the compile cost again.
<!-- issue:jit-cache-node-local end -->

<!-- issue:lustre-watchdog begin -->
**A storage stall can kill the endpoint even after the storage recovers.** PyTorch kills the process
when the NCCL watchdog thread stops sending heartbeats, on the assumption that a collective hung. A
stalled network filesystem freezes every rank the same way, so at the 480 second default a transient
storage outage takes the endpoint down permanently rather than pausing it. The signature is every rank
reporting `Last enqueued NCCL work: -1`, meaning no collective was ever in flight, so the process was
frozen rather than genuinely hung on communication. `env/env.sh` sets
`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600` so a stall that resolves within an hour is survivable.
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
endpoint served requests intermittently for about 21 hours and then ended with
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

For a Slurm job, `scancel <jobid>` is all of it: the job requests both nodes and starts Ray through
`srun` inside that allocation, so Slurm's cgroups own every process on both nodes and remove them.

The SSH path has no such owner, so name both nodes explicitly:

```
bash common/tools/stop.sh <head_node> <worker_node>
```

That stops Ray as well as the server. Do not reach for `ssh <node> ray stop` by hand: `ray` lives in the
recipe venv and is not on `PATH` in a non-interactive shell, so it silently does nothing. `stop.sh` finds
the venv from a running process instead, then confirms both the GPUs and the host are clear. Ray matters
here even when nvidia-smi looks clean, because its GCS server and dashboard workers hold no GPU memory
while still occupying several GB of host RAM.

`stop.sh` kills the server processes and waits for GPU memory to be released. Confirm with
`ssh <node> nvidia-smi --query-gpu=memory.used --format=csv,noheader`, which should read 0 MiB on all
four GPUs of both nodes before you relaunch, or the next start will fail on memory. A surviving Ray
cluster is the other common cause of a failed relaunch, and it fails on resources rather than on memory,
which reads like a different problem entirely. `stop.sh` clears both.

## Expected startup time

| Stage | Warm nodes | Cold nodes |
| --- | --- | --- |
| Environment build, one time | skipped | 5 to 15 min |
| Ray head plus worker bring-up | seconds | seconds |
| Weight load, 141 shards across 8 ranks | 39 s | 14 min 23 s |
| Engine init: profile, KV cache, warmup | 109 s | 84 s |
| Total, vLLM banner to serving | 5 min 5 s | 19 min 36 s |

The table starts at the vLLM banner. The validated run reached serving 16 minutes 23 seconds after the
launcher was invoked; the extra time is Ray bring-up and the Python import of torch and vLLM, before
vLLM logs anything.

Both columns are measured on two different node pairs, with the checkpoint read from
scratch in both cases. The difference is page cache: the fast pair had already loaded this
checkpoint, the freshly allocated pair had not. Reading from the Lustre testbed path instead is slower
again.

Nothing in this window looks like progress if you only watch the port, so watch the log. A first launch
that appears hung is almost always still loading weights, and `VLLM_ENGINE_READY_TIMEOUT_S=3600` in
`env/env.sh` exists precisely because the cold case exceeds vLLM's 600 second default.
