# Kimi-K2.7-Code on two H200 nodes

Status: Validated - vLLM 0.25.1+cu129, protocol: slope(128,1152) swept at concurrency 1 through 1024

Everything needed to build, launch, verify, connect to, and debug this endpoint is on this page.

## Configure once

Create the API key. The endpoint refuses requests without it, and the key reaches the engine through
the environment rather than as an argument, so it does not appear in `ps` output.

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/Kimi-K2.7-Code-h200-4-nodes2.key
chmod 600 secrets/Kimi-K2.7-Code-h200-4-nodes2.key
```

That key gates this recipe alone. When it is absent `secrets/vllm_api_key` is read instead, so a setup
made before per-model keys keeps working.

Nothing else is required. Cluster paths come from `common/defaults.sh`, which is tracked with working
defaults, so a fresh clone runs as is. Optional overrides, either exported or set in
`common/site.conf`:

| Variable | Default | Why you might change it |
| --- | --- | --- |
| `ACCOUNT` | unset | Your Slurm account, or pass `--account` at submit time |
| `KIMI_HEAD`, `KIMI_WORKER` | unset | Two nodes you already hold, for the SSH path |
| `MODELS_DIR` | shared repository path | Point at your own faster copy of the checkpoint |
| `ENV_ROOT` | scratch | Where this recipe builds its environment |

One constraint is specific to a two-node recipe: `ENV_ROOT` and `MODELS_DIR` must both resolve to the
same content on both nodes. The Ray worker imports vLLM from `ENV_ROOT` and reads weights from
`MODELS_DIR` itself, so a path that exists only on the head node fails after the cluster has already
formed, which reads as an engine problem rather than a path problem.

## Status

Validated. The environment was built from `env/build.sh`, the endpoint was
launched with `serve_ssh.sh` on two H200 nodes, 8 GPUs via Ray, and throughput was measured with `common/tools/bench.sh`
across concurrency 1, 8, 32, 64, 128, 256, 512, 640, 768, 896 and 1024. Ready 17 minutes 43 seconds after launch. The endpoint was still answering after the sweep finished.

The multimodal profiling deadlock did not occur, confirming that `--skip-mm-profiling
--mm-processor-cache-gb 0` is still doing its job on this checkpoint.

Single stream and saturated throughput are different measurements and neither substitutes for
the other. See Measured performance below for the full curve and the disclosure block.

## What this is

Kimi-K2.7-Code, a 1T-parameter mixture-of-experts coding model with 32B parameters active per token,
MLA attention, and a MoonViT vision tower, so it accepts images as well as text. It is thinking-mode
only: it always emits reasoning before its answer. The
checkpoint is natively INT4 quantization-aware trained, not post-quantized. It exposes vLLM's
Anthropic-compatible API, so Claude Code connects to it directly with no proxy.

- Checkpoint directory: `Kimi-K2.7-Code`
- Hugging Face repo: `moonshotai/Kimi-K2.7-Code`
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Kimi-K2.7-Code`

Read from the checkpoint:

| Property | Value |
| --- | --- |
| Architecture | `KimiK25ForConditionalGeneration`, native to vLLM 0.25.1 |
| Language model | `model_type` `kimi_k2`, on a DeepSeek-V3 style decoder |
| On disk | about 595 GB across 64 shards, 554 GiB |
| Layers | 61 |
| Experts | 384 routed plus 1 shared, 8 routed per token, `moe_intermediate_size` 2048 |
| Attention | 64 heads, `hidden_size` 7168, MLA |
| Context | `max_position_embeddings` 262144, served here at 32768 |
| Speculative head | `num_nextn_predict_layers` 0, so none ships |
| License | modified MIT |

vLLM 0.25.1 implements this architecture, so no out-of-tree model code is needed.
`--trust-remote-code` is still passed, because the checkpoint ships its own configuration and processor
modules, which the multimodal path loads.

The shared repository path works out of the box. Copying the checkpoint into your own scratch space loads
faster, because scratch outperforms Lustre for this workload, and the directory names are identical in both
locations so only `MODELS_DIR` changes. Scratch has a 90-day retention policy, so treat it as a fast
cache and keep the shared repository as the permanent copy.

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

All H200 nodes on this cluster share one hardware specification, so any two nodes in the partition work.
About 595 GB of weights, 554 GiB, does not fit one node's four cards at 143771 MiB each, which is
562 GiB once you account for activations and any KV cache at all, and that is what forces two nodes on
Hopper and therefore pipeline parallelism. `serve.sbatch` requests 48 CPUs and 500 GB per node, both
inside the per-GPU limits for four GPUs.

The two nodes are not symmetric in what they hold. Each rank of the head stage loaded
71.06 GiB of weights and had 52.16 GiB left for KV cache, while each rank of the worker stage loaded
70.81 GiB and had 50.07 GiB, because pipeline parallelism splits layers between stages and the stages are
not identical in weight footprint. That is expected, not a misconfiguration.

The single-node RTX variant of this model exists for a reason: eight 97887 MiB Blackwell cards hold the
same checkpoint on one node. This recipe is the faster of the two and the more expensive.

## Environment build

This recipe builds its own environment, shared with no other recipe. Roughly 13 GB, and it lands under
`ENV_ROOT` on scratch rather than in the repo, because startup is dominated by page faulting the
torch shared objects and stat-ing tens of thousands of small package files: measured on GPU nodes, the interval from
process start to the first vLLM log line was about 14 minutes from Lustre and 58 seconds from scratch.
A bare torch and vLLM import from scratch is 9.2 seconds, so most of that 58 seconds is engine
startup rather than filesystem cost.

```
bash recipes/Kimi-K2.7-Code/h200-4-nodes2/env/build.sh
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

This is why the H200 and RTX variants of the same model cannot share an environment: the RTX variant
needs a CUDA 13 nightly wheel for sm_120, and this one needs the cu129 release wheel for driver 575.
Building both is not redundant work.

`build.sh` treats an existing directory as suspect rather than as success: it checks for the installed
`vllm` dist-info and rebuilds anything less, because an interrupted install leaves a venv skeleton behind
that would otherwise be served from silently.

Scratch expires after 90 days, so this environment is disposable. Rebuild it with the same command, or
`--force` to replace an existing one. Record the exact resolution in `env/requirements.lock` after a build, which is what makes a
drifted rebuild visible.

## Launch

Slurm path, submitted from the repo root:

```
sbatch --account=<your-account> recipes/Kimi-K2.7-Code/h200-4-nodes2/serve.sbatch
```

The batch script allocates two nodes, reads each one's ib0 address, starts a Ray head on the first and a
worker on the second, then runs the engine on the head. The endpoint is on the **first** allocated node:

```
squeue --me                       # NODELIST column, first name
tail -f kimi-h200-<jobid>.log
```

Direct path, for two nodes you already hold. Use the Slurm submission above unless you already have
the nodes, or you are deploying an endpoint on behalf of others:

```
bash recipes/Kimi-K2.7-Code/h200-4-nodes2/serve_ssh.sh <head_node> <worker_node>
```

That path prints the cluster's GPU count before it loads any weights. It should say 8. If it says 4 the
worker did not join, and stopping there is much cheaper than discovering it minutes into a weight load.

Submit from the repo root either way. Slurm stages the batch script into its own spool directory, so the
script cannot locate the repo from its own path and resolves paths against the submit directory instead.

`ray_head.sh` and `ray_worker.sh` in this directory are the Ray bring-up, and both launch paths call
them rather than duplicating the commands. They take their GPU count from `GPUS_PER_NODE`, then
`SLURM_GPUS_ON_NODE`, then 4, so a node with a different GPU count is not mis-advertised to Ray.

## Verify

```
KEY=$(cat secrets/Kimi-K2.7-Code-h200-4-nodes2.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the head node serving it>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models          # must print 401

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"kimi-k2.7-code","messages":[{"role":"user","content":"What is 2+2? Answer briefly."}],"max_tokens":400}'
```

A keyless request returning 401 is the expected, correct behavior, and it was confirmed on that
run.

<!-- issue:thinking-model-max-tokens begin -->
**Give thinking models room, or `content` comes back empty.** This model emits reasoning before its
answer, and vLLM returns that in a separate `reasoning` field, not `reasoning_content`. With a small
budget the whole allowance is spent reasoning, `finish_reason` is `length`, and `content` is empty,
which looks like a broken endpoint but is not. Use at least 400 output tokens for a smoke test, and 800
or more for a model that reasons at length. If `content` is empty, raise the budget before suspecting
the endpoint.
<!-- issue:thinking-model-max-tokens end -->

This model is thinking-mode only, so that applies to every request, not just complex ones.

Because startup deliberately skips multimodal profiling, image input is the one capability to verify
explicitly rather than assume. Skipping profiling means the engine never sizes a multimodal batch during
startup, so the first image request is also the first test of that path.

## Connect a client

```
export NODE=<the head node serving it>
source recipes/Kimi-K2.7-Code/h200-4-nodes2/client.env
claude
```

<!-- issue:anthropic-auth-token begin -->
**Use `ANTHROPIC_AUTH_TOKEN`, never `ANTHROPIC_API_KEY`.** Both engines accept only
`Authorization: Bearer <key>`. Setting `ANTHROPIC_API_KEY` makes Claude Code send an `x-api-key`
header instead, which the engine ignores, and every request returns HTTP 401. Also set
`ANTHROPIC_SMALL_FAST_MODEL` to this same served model, or the client reaches for a hosted Haiku that
this endpoint does not serve.
<!-- issue:anthropic-auth-token end -->

Point the client at the head node. The worker node runs Ray and holds half the layers, but it serves no
HTTP endpoint and will refuse the connection.

For an OpenAI-compatible client instead (Cline, Aider, Continue, OpenHands), use base URL
`http://<head-node>:8000/v1`, the same key, and model name `kimi-k2.7-code`.

## Tunable inputs

Every variable this recipe honors, with its default and effect.

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/Kimi-K2.7-Code` | Serve a different copy of the checkpoint |
| `API_PORT` | 8000 | Listening port |
| `MAX_MODEL_LEN` | 32768 | Context window; the checkpoint supports 262144, at a KV cache cost |
| `GPU_UTIL` | 0.90 | Fraction of VRAM for weights plus KV cache |
| `TP` | 4 | Tensor parallel size; 4 is one node's GPU count and must stay inside a node |
| `PP` | 2 | Pipeline parallel size; 2 is the node count |
| `ENFORCE_EAGER` | 1 | Pass empty to try CUDA graph capture, which is untested here |
| `ATTN_BACKEND` | unset | Override vLLM's attention backend selection |
| `EXTRA_ARGS` | `--skip-mm-profiling --mm-processor-cache-gb 0` | **Replaces** the default; see Gotchas |
| `TOOL_PARSER` | `kimi_k2` | Tool call parser |
| `REASONING_PARSER` | `kimi_k2` | Reasoning parser |
| `RAY_PORT` | 6379 | Ray head port |
| `RAY_HEAD_IP` | unset | Head address, when calling `serve.sh` directly |
| `GPUS_PER_NODE` | `SLURM_GPUS_ON_NODE`, then 4 | GPUs each Ray node advertises |
| `RAY_BLOCK` | unset in the SSH path, 1 under Slurm | Keep `ray start` in the foreground |
| `KIMI_HEAD`, `KIMI_WORKER` | unset | Default nodes for the SSH path |
| `LOG_DIR` | `/tmp/$USER/vllm` | Where the SSH path writes the server log |
| `VENV_DIR` | under `ENV_ROOT` | Use an environment built elsewhere |

`EXTRA_ARGS` is the one entry in this table that can break the launch by being set carelessly. It
replaces the default rather than adding to it, so anything you pass must repeat both multimodal flags.
An empty value falls back to the default rather than clearing it, deliberately, because the failure mode
of clearing it costs a two-node allocation and produces no error message.

## Web search

<!-- issue:anthropic-hosted-tools-400 begin -->
**Anthropic's hosted tools do not work against a local endpoint.** Server-side tools such as
`web_search_20250305`, `web_fetch_20250910` and `code_execution_20250522` are executed by Anthropic's own
API rather than by the model, so no endpoint here can run them. What you see differs by engine, measured
on both.

vLLM rejects all three with HTTP 400, because their definitions carry no `input_schema`:

```
1 validation error:
  {'type': 'missing', 'loc': ('body', 'tools', 0, 'input_schema'), 'msg': 'Field required',
   'input': {'type': 'web_search_20250305', 'name': 'web_search'}}
```

SGLang is harder to diagnose. It accepts `web_search_20250305` with HTTP 200 and drops the tool, logging
that it has no native support, so the model answers without searching and nothing in the reply says why.
It rejects `web_fetch_20250910` and `code_execution_20250522` with HTTP 400.

Client-side tools (file edits, shell, and anything you define) work normally on both. For web access,
install the repo's keyless search tool and skill, from the repo root:

```
mkdir -p ~/.local/bin ~/.claude/skills
ln -sf "$PWD/common/tools/search.sh" ~/.local/bin/search.sh
cp -r common/skills/local-search ~/.claude/skills/
```

Check it with `search.sh wiki "tensor parallelism" 1`, and add `~/.local/bin` to your `PATH` if the
command is not found. The model then searches through `search.sh` (web, arxiv, crossref, pubmed,
openalex, wiki, fetch) instead of the hosted tool. Full details in
[docs/web-search.md](../../docs/web-search.md).
<!-- issue:anthropic-hosted-tools-400 end -->

## Measured performance

| Configuration | Aggregate rate | Per stream | Latency |
| --- | --- | --- | --- |
| Single stream, concurrency 1 | 30.4 tok/s | 30.4 tok/s | TTFT median 102 ms, n=3 spanning 30.3 to 30.5 |
| Concurrency 8 | 241.4 tok/s | 30.2 tok/s | TTFT median 100 ms, p90 148 ms, n=3 spanning 240.9 to 242.0 |
| Concurrency 32 | 929.0 tok/s | 29.0 tok/s | TTFT median 175 ms, p90 239 ms, n=3 spanning 927.9 to 930.2 |
| Concurrency 64 | 1755.1 tok/s | 27.4 tok/s | TTFT median 221 ms, p90 314 ms, n=3 spanning 1751.6 to 1759.4 |
| Concurrency 128 | 2648.9 tok/s | 20.7 tok/s | TTFT median 302 ms, p90 406 ms, n=3 spanning 2645.7 to 2649.9 |
| Concurrency 256 | 3641.4 tok/s | 14.2 tok/s | TTFT median 410 ms, p90 585 ms, n=3 spanning 3640.5 to 3641.6 |
| Concurrency 512 | 5668.7 tok/s | 11.1 tok/s | TTFT median 630 ms, p90 988 ms, n=3 spanning 5667.3 to 5676.9 |
| Concurrency 640 | 6144.7 tok/s | 9.6 tok/s | TTFT median 772 ms, p90 1185 ms, n=3 spanning 6140.9 to 6157.0 |
| Concurrency 768 | 6170.5 tok/s | 8.0 tok/s | TTFT median 856 ms, p90 1385 ms, n=3 spanning 6167.9 to 6176.9 |
| Concurrency 896 | 6577.4 tok/s | 7.3 tok/s | TTFT median 964 ms, p90 1597 ms, n=3 spanning 6576.5 to 6577.9 |
| Concurrency 1024 (rising) | 7140.3 tok/s | 7.0 tok/s | TTFT median 1011 ms, p90 1689 ms, n=3 spanning 7136.3 to 7148.0 |

Measured with `common/tools/bench.sh`, endpoint ready 17m 43s after launch. Full disclosure, without which a tokens
per second figure cannot be compared against anything:

| Parameter | Value |
| --- | --- |
| ISL, input tokens | 15 |
| OSL, output tokens | 1152, as the slope between 128 and 1152 |
| Counted | output tokens only, never input plus output |
| Concurrency levels | 1,8,32,64,128,256,512,640,768,896,1024 |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| `max_num_seqs` | engine default, 1024 on this hardware |
| Hardware | two H200 nodes, 8 GPUs |

Quote 30.4 tok/s for interactive coding, where one person waits on one
response. Quote 7140.3 tok/s at concurrency 1024 for a shared endpoint under load.
The two measure different things and neither substitutes for the other.

Throughput was **still rising at concurrency 1024**, 26 percent above its own concurrency 512 at the top of the sweep,
so 7140.3 tok/s is a floor rather than a ceiling. The sequence cap is not what stopped it:
`max_num_seqs` resolves to 1024 on this hardware and the sweep ran to that level, so finding the
true peak needs the cap raised, which is a different serving configuration.

Concurrency 512 was measured in both runs, at 5668.7 and 5677.6 tok/s, a +0.2 percent
difference. That is the check that the two halves of this curve are comparable.

Scheduler counters over the extended levels: KV cache usage reached 73 percent, the running
batch reached 1024 requests, and there were no preemptions at any level.

The input sequence here is short, which is the best case for decode. Measure with
`--prompt-tokens` at your working context before quoting a number for long-context work.

## Parallelism and quantization

The shape is TP4 inside each node and PP2 between the two nodes, over Ray. It is the only shape that
works on Hopper for this checkpoint: 595 GB of weights needs both nodes, tensor parallelism cannot cross
the node boundary without hanging, so what crosses is pipeline parallelism.

The quantization is native INT4 quantization-aware training, not a post-hoc conversion: the routed
experts are compressed-tensors pack-quantized W4A16 at group size 32, while attention, the shared
expert, the dense layers, `lm_head` and the vision tower stay bf16. Without the INT4 experts a
1T-parameter model would not fit eight GPUs of any kind here.

`--disable-custom-all-reduce` is passed because vLLM's custom all-reduce kernel is a single-node
optimization and this engine spans two nodes.

The vision tower is placed with `--mm-encoder-tp-mode data`, so the encoder runs data parallel across the
tensor-parallel ranks rather than being sharded across them. That is also what makes the profiling step
skippable: the encoder is replicated per rank rather than split, so nothing about its placement depends
on a profiled batch shape.

## Gotchas

<!-- issue:multimodal-mm-profiling begin -->
**Multimodal profiling deadlocks across nodes.** With a multimodal checkpoint on more than one node,
startup completes weight loading and then hangs at multimodal profiling with 0 percent GPU
utilization, indefinitely. Raw two-node NCCL all-reduce is healthy at that point, so it is not a
fabric problem. `serve.sh` passes `--skip-mm-profiling --mm-processor-cache-gb 0`, which is required,
not optional.
<!-- issue:multimodal-mm-profiling end -->

That is the single most important line in this recipe, and it is worth knowing how it was established,
because the symptom points at the fabric and the fabric was innocent. The launch loads all 64 shards,
reports success, and then sits at 0 percent GPU utilization forever with no error and no timeout. A
two-node hang with no error looks exactly like a broken InfiniBand path, so the first step was to rule
that out: a raw two-node NCCL `all_reduce` between the same eight ranks completed normally. Only then
was the multimodal profiling step the remaining suspect, and skipping it fixed the launch. If you ever
see this hang again, prove the fabric healthy first; it saves a day of chasing NCCL flags.

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

That one costs this recipe nothing, because this checkpoint ships no speculative head to lose. It is
recorded here because it is the reason the shape cannot be changed to recover one.

<!-- issue:deepgemm-h200-crash begin -->
**Leave `VLLM_USE_DEEP_GEMM` at 0 on H200.** The DeepGEMM MoE path takes an illegal memory access on
GLM-5.2's sparse attention, and forcing `VLLM_USE_DEEP_GEMM=1` on H200 independently reproduced the
same crash for Qwen3-Coder-480B-FP8. It is load-bearing for more than one model on this hardware, so
do not flip it without re-testing the model you are serving.
<!-- issue:deepgemm-h200-crash end -->

Nothing about this model has been tested with DeepGEMM either way. `env/env.sh` keeps
`VLLM_USE_DEEP_GEMM=0` as an inherited setting, matching the configuration the measured rate came from,
and its provenance comment says inherited rather than verified for exactly that reason.

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

**A Ray worker can die under you, and the head log will not say why.** The recorded run of this endpoint
served requests, then ended 47 minutes after startup with
`RayWorkerProc rank=[1] died unexpectedly, shutting down executor`, followed by
`RuntimeError: Executor failed`. Rank 1 is the second pipeline stage, on the worker node. The head's log
records the death and no cause, because the cause was on the other node, and Ray's own worker logs under
`/tmp/ray` on the worker were not captured. The same ending was recorded for GLM-5.2-FP8 on the same
hardware in the same week, so it is not specific to this model. If it happens to you, look there before
theorizing: `ssh <worker_node> 'ls -t /tmp/ray/session_latest/logs | head'`.

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

| Stage | Measured |
| --- | --- |
| Environment build, one time | 5 to 15 min, once |
| Ray head plus worker bring-up | seconds |
| Banner to the start of the weight load | about 3 min |
| Weight load, 64 shards across 8 ranks | 2 min 6 s, reported as 132 s per rank |
| Engine init: profile, KV cache, warmup | 165 s |
| Total, vLLM banner to serving | 9 min 11 s |

The validated run reached serving 17 minutes 43 seconds after the launcher was invoked. The difference
from the total above is Ray bring-up and the Python import of torch and vLLM, before vLLM logs anything.

Measured from the server log, with the checkpoint read from scratch rather than the Lustre repository
path. The table starts at the vLLM banner; everything before it is the Python import of torch and
vLLM, which is fast because `ENV_ROOT` is on scratch and slow from Lustre.

A first launch that appears hung is usually still loading weights, so check the log before killing it.
The one exception is the multimodal profiling hang above, which is genuinely permanent rather than slow,
and which the default flags prevent.
