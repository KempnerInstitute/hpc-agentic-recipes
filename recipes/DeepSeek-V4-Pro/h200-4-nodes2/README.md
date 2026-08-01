# DeepSeek-V4-Pro on two H200 nodes

Status: Blocked - vLLM 0.25.1, CUTLASS w8a8 kernel dispatch fails on Hopper

Everything needed to build, launch, verify, connect to, and debug this endpoint is on this page.

> **Read this before you allocate two H200 nodes.** This checkpoint's routed experts are FP4 and
> Hopper has no FP4 hardware. It was run here and it does not start. Use
> `recipes/DeepSeek-V4-Pro/rtx-8-nodes2` on Blackwell RTX nodes, which execute FP4 natively. This page
> records the H200 failure so it does not have to be rediscovered.

## Configure once

Create the API key. The endpoint refuses requests without it, and the key is passed through the
environment rather than the command line so it never appears in `ps` output.

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/vllm_api_key
chmod 600 secrets/vllm_api_key
```

Cluster paths otherwise come from `common/defaults.sh`, which is tracked with working defaults.
Optional overrides, either exported or set in `common/site.conf`:

| Variable | Default | Why you might change it |
| --- | --- | --- |
| `ACCOUNT` | unset | Your Slurm account, or pass `--account` at submit time |
| `DSV4_H200_HEAD`, `DSV4_H200_WORKER` | unset | Two nodes you already hold, for the SSH path |
| `MODELS_DIR` | shared testbed path | Point at your own faster copy of the checkpoint |
| `ENV_ROOT` | scratch | Where this recipe builds its environment |

## Status

Blocked, measured rather than predicted. This configuration was run end to end on
two H200 nodes and does not work. Use `../rtx-8-nodes2` instead.

The environment built, Ray formed an 8-GPU cluster, and all 64 shards loaded in 12 minutes 24 seconds,
so nothing is wrong with the recipe's plumbing. Engine initialization then failed on every worker:

```
RuntimeError: dispatch_scaled_mm,
  csrc/libtorch_stable/quantization/w8a8/cutlass/c3x/scaled_mm_helper.hpp:17
RuntimeError: Engine core initialization failed
  [repeated 12x across cluster]
```

This is the CUTLASS w8a8 scaled matrix multiply refusing to dispatch a kernel for this data type on
Hopper. It is the same class of failure that blocks Qwen3-Coder-480B-FP8 on H200, and it confirms what
this model's `config.json` implies: `expert_dtype` is `fp4`, Blackwell executes FP4 natively, and
Hopper has no such hardware path.

Two things worth knowing before trying to work around it. The checkpoint ships an
`inference/convert.py` that can emit fp8 experts, and vLLM has an `expert_dtype == "fp8"` path, so
converting the experts is the only real Hopper option. And an attempt without that flag failed differently, with
`DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache, got auto`, because the recipe was missing
`--kv-cache-dtype fp8_ds_mla`; that flag is now present, which is how this run got far enough to reach
the FP4 problem.

The cause is precision, not capacity. `config.json` sets `expert_dtype: fp4`, and vLLM
0.25.1 resolves that to its MXFP4 fused-MoE method, since the checkpoint sets no
`moe_quant_algo: NVFP4` override (`vllm/models/deepseek_v4/quant_config.py`). MXFP4 is a
Blackwell-native format, and Hopper has no FP4 tensor cores, so the expert layers must either be
emulated or routed through a 4-bit weight-only kernel such as Marlin. Of the three possible outcomes,
a clean fallback that is merely slow, a numerically wrong fallback, or a hard startup failure, the
measured one is the third: engine initialization fails on every worker, as recorded above.

If H200 is the only hardware available, the one option that addresses the problem is requantizing the experts to FP8: the checkpoint ships an
`inference/convert.py` whose `--expert-dtype` argument accepts `fp8` and `fp4`, and vLLM's DeepSeek-V4
quantization config has an explicit `expert_dtype == "fp8"` path that falls through to the ordinary
block-scaled FP8 MoE method, which Hopper runs natively. That has not been tried here, and it costs a
conversion pass over 806 GiB plus roughly double the expert memory.

Note that the `nvidia/DeepSeek-V4-Pro-NVFP4` variant upstream does **not** solve this: NVFP4 is also a
Blackwell format, it would simply take vLLM's ModelOpt NVFP4 path instead of the MXFP4 one. It is an
alternative for the RTX variant, not a Hopper workaround.

## What this is

DeepSeek-V4-Pro, the larger of the two DeepSeek-V4 preview models: 1.6T total parameters, 49B
activated, a 1M-token context, and a hybrid attention design combining Compressed Sparse Attention
with Heavily Compressed Attention, plus manifold-constrained hyper-connections on the residual path.
It is a thinking model with native tool calling, and it exposes vLLM's Anthropic-compatible API, so
Claude Code connects to it directly with no proxy.

- Checkpoint directory: `DeepSeek-V4-Pro`
- Hugging Face repo: `deepseek-ai/DeepSeek-V4-Pro`
- Testbed path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/DeepSeek-V4-Pro`

Read from the checkpoint, since these are the facts the whole recipe rests on:

| Property | Value |
| --- | --- |
| Architecture | `DeepseekV4ForCausalLM`, `model_type` `deepseek_v4` |
| On disk | 805.4 GiB, 864.7 GB decimal, 64 shards, 145116 tensors |
| Layers | 61 |
| Experts | 384 routed plus 1 shared, 6 routed experts per token |
| `moe_intermediate_size` | 3072 |
| Attention | 128 heads, `head_dim` 512, `q_lora_rank` 1536, `o_lora_rank` 1024, 1 KV head |
| Sparse attention indexer | `index_n_heads` 64, `index_head_dim` 128, `index_topk` 1024 |
| Context | `max_position_embeddings` 1048576, YaRN scaling factor 16 over an original 65536 |
| Quantization | `quant_method` `fp8`, `weight_block_size` [128, 128], `scale_fmt` `ue8m0`, `fmt` `e4m3`, dynamic activations |
| Experts precision | `expert_dtype` `fp4`, which is the problem on this hardware |
| Speculative head | `num_nextn_predict_layers` 1, and 2343 `mtp.0.*` tensors ship in the checkpoint |

Two consequences of the checkpoint's contents that are easy to get wrong:

- **It carries no chat template.** `tokenizer_config.json` has no `chat_template` and there is no
  `chat_template.jinja`. Nothing needs to be supplied: vLLM 0.25.1 implements the DeepSeek-V4 prompt
  encoding itself and switches `tokenizer_mode` to `deepseek_v4` automatically for this architecture,
  so do not pass `--chat-template`.
- **It needs no `--trust-remote-code`.** `config.json` has no `auto_map` and ships no modeling code
  for the language model, so the engine's own implementation is used.

Copying the checkpoint into scratch loads faster than Lustre for this workload, and the directory
names are identical in both locations so only `MODELS_DIR` changes. Scratch has a 90-day retention
policy, so treat it as a fast cache and keep testbed as the permanent copy.

## Hardware

| Requirement | Value |
| --- | --- |
| GPU | H200, 4 per node, 143771 MiB each |
| Nodes | 2, so 8 GPUs and about 1123 GiB of VRAM |
| Parallelism | TP4 inside each node, PP2 between them, Ray |
| Partition | `kempner_h200` |
| Per-GPU allocation limit | 16 CPUs, about 378 GiB host memory |
| Maximum wall time | 2 days |

All H200 nodes on this cluster share one hardware specification, so any two nodes in the partition
work.

Memory is comfortable: 8 GPUs at 143771 MiB is about 1123 GiB, and at `--gpu-memory-utilization 0.90`
about 1011 GiB is usable against 806 GiB of weights, leaving roughly 200 GiB for KV cache and
activations. This model is unusually cheap in KV for its context length, because the hybrid compressed
attention is designed to be, so 200 GiB is more than it needs. Capacity is not why this recipe is
expected to struggle; the FP4 experts are.

For comparison, the recommended RTX variant has 16 GPUs at 97887 MiB, about 1530 GiB, and executes the
FP4 experts natively.

## Environment build

This recipe builds its own environment, shared with no other recipe. Roughly 13 GB, and it lands under
`ENV_ROOT` on scratch rather than in the repo, because startup is dominated by page faulting the
torch shared objects and stat-ing tens of thousands of small package files: measured on GPU nodes,
the interval from process start to the first vLLM log line was about 14 minutes from Lustre and 58
seconds from scratch. A bare torch and vLLM import from scratch is 9.2 seconds, so most of that 58 seconds
is engine startup rather than filesystem cost.

```
bash recipes/DeepSeek-V4-Pro/h200-4-nodes2/env/build.sh
```

That is the only supported build path, because the install needs uv flags a requirements file cannot
express. Ray is installed alongside vLLM, since this recipe spans two nodes and uses the Ray executor.
What else it does, and why:

<!-- issue:hopper-cu129-wheel begin -->
**Hopper nodes need the cu129 wheel, not vLLM's default.** These nodes run NVIDIA driver 575
(CUDA 12.9), which cannot run vLLM's default CUDA 13 PyPI wheel. The recipe installs the cu129
release wheel from the vLLM GitHub release with `--torch-backend=cu129`.
<!-- issue:hopper-cu129-wheel end -->

Scratch expires after 90 days, so this environment is disposable. Rebuild it with the same command, or
`--force` to replace an existing one. Record the exact resolution in `env/requirements.lock` and any non-PyPI artifact URLs with their
hashes in `env/WHEELS` after a build, which is what makes a drifted rebuild visible.

## Launch

Slurm path, submitted from the repo root:

```
sbatch --account=<your-account> recipes/DeepSeek-V4-Pro/h200-4-nodes2/serve.sbatch
```

The batch script allocates two nodes, starts a Ray head on the first and a worker on the second, then
runs the engine on the head. The endpoint is on the **first** allocated node:

```
squeue --me                       # NODELIST column, first name
tail -f dsv4-h200-<jobid>.log
```

Watch that log closely on the first attempt. The interesting moment is when the MoE layers are built,
which is where an FP4 expert format on Hopper either falls back or fails.

Direct path, for two nodes you already hold. Use the Slurm submission above unless you already have
the nodes, or you are deploying an endpoint on behalf of others:

```
bash recipes/DeepSeek-V4-Pro/h200-4-nodes2/serve_ssh.sh <head_node> <worker_node>
```

Submit from the repo root either way. Slurm stages the batch script into its own spool directory, so
the script cannot locate the repo from its own path and resolves paths against the submit directory
instead.

## Verify

```
KEY=$(cat secrets/vllm_api_key)
NODE=<the head node serving it>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models          # must print 401

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"deepseek-v4-pro","messages":[{"role":"user","content":"What is 2+2? Answer briefly."}],"max_tokens":400}'
```

A keyless request returning 401 is the expected, correct behavior.

If this recipe ever does start, verify output **quality** and not just liveness, because a silent
numerical fallback on the FP4 experts would show up as fluent nonsense rather than as an error. Ask it
something with a checkable answer, and compare against the RTX variant if both can be held at once.

Reasoning is off by default for this model in vLLM, because the engine's DeepSeek-V4 encoding closes
the thinking block unless it is asked not to. To exercise the thinking path, pass it explicitly:

```
  -d '{"model":"deepseek-v4-pro","messages":[{"role":"user","content":"Prove that 2 is irrational."}],
       "max_tokens":2000,"chat_template_kwargs":{"thinking":true}}'
```

The model card recommends `temperature 1.0` and `top_p 1.0` for local deployment, and at least a 384K
context window for its maximum reasoning effort mode.

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
source recipes/DeepSeek-V4-Pro/h200-4-nodes2/client.env
claude
```

<!-- issue:anthropic-auth-token begin -->
**Use `ANTHROPIC_AUTH_TOKEN`, never `ANTHROPIC_API_KEY`.** vLLM accepts only
`Authorization: Bearer <key>`. Setting `ANTHROPIC_API_KEY` makes Claude Code send an `x-api-key`
header instead, which vLLM ignores, and every request returns HTTP 401. Also set
`ANTHROPIC_SMALL_FAST_MODEL` to this same served model, or the client reaches for a hosted Haiku that
this endpoint does not serve.
<!-- issue:anthropic-auth-token end -->

For an OpenAI-compatible client instead (Cline, Aider, Continue, OpenHands), use base URL
`http://<head-node>:8000/v1`, the same key, and model name `deepseek-v4-pro`.

## Tunable inputs

Every variable this recipe honors, with its default and effect.

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/DeepSeek-V4-Pro` | Serve a different copy of the checkpoint |
| `MODELS_DIR` | shared testbed path | Point at your own faster copy of the checkpoint |
| `API_PORT` | 8000 | Listening port |
| `MAX_MODEL_LEN` | 131072 | Context window; the checkpoint supports 1048576 |
| `GPU_UTIL` | 0.90 | Fraction of VRAM for weights plus KV cache |
| `TP` | 4 | Tensor parallel size; 4 is the node's GPU count |
| `PP` | 2 | Pipeline parallel size; 2 is the node count |
| `PERF` | unset | Attempt CUDA graph capture instead of eager |
| `CUDAGRAPH_MODE` | `NONE` | Graph mode passed through when `PERF` is set |
| `EXTRA_ARGS` | unset | Extra `vllm serve` flags, for experiments |
| `TOOL_PARSER` | `deepseek_v4` | Tool call parser |
| `REASONING_PARSER` | `deepseek_v4` | Reasoning parser |
| `RAY_PORT` | 6379 | Ray head port |
| `RAY_HEAD_IP` | unset | Head address, when calling `serve.sh` directly |
| `DSV4_H200_HEAD`, `DSV4_H200_WORKER` | unset | Default nodes for the SSH path |
| `LOG_DIR` | `/tmp/$USER/vllm` | Where the SSH path writes the server log |
| `VENV_DIR` | under `ENV_ROOT` | Use an environment built elsewhere |

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

| Configuration | Decode rate | Protocol |
| --- | --- | --- |
| TP4 x PP2, eager, 2 nodes | not measured | not applicable |

**No decode rate has been measured for this model on this hardware, and none is estimated here.** The
useful measurement from this recipe is not a rate at all: it is whether the FP4 experts run on Hopper,
and if they do, whether the answers are correct and how much slower the fallback is than the RTX
variant. Record that outcome here when it is known, including a failure, which is the result this
recipe most expects.

When and if it comes up, measure with the slope method rather than timing one generation:

```
bash common/tools/bench.sh --host <head-node> --model deepseek-v4-pro
```

For scale, two H200 nodes served Kimi-K2.7-Code at 30.4 tok/s single stream, which is the closest measured
two-node Hopper reference point in this repo.

To measure total throughput with concurrent requests, and to find where it stops rising:

```
bash common/tools/bench.sh --host <head-node> --model deepseek-v4-pro --sweep 1,8,32,128
```

Aggregate throughput is several times the single stream figure, while per stream latency falls. Quote
the single stream number for interactive use and the aggregate for serving several people at once, and
never compare one against the other.

Prompt length is a separate axis, and how much it costs depends entirely on the model's attention
design. The slope method cancels prefill, so a long prompt never distorts the measurement, but a larger
KV cache can slow every decode step because attention reads it on each one. How much is an empirical
question: measured on GLM-5.2-NVFP4, decode was flat from 21 to 26379 input tokens (97.0, 97.8 and 95.0
tok/s), because that model combines MLA compression, an fp8 KV cache and sparse attention that reads only
a subset of the context. A dense model with full attention and a bf16 KV cache should be expected to
degrade far more. Measure with `--prompt-tokens` rather than assuming either way.

## Parallelism and quantization

The shape is TP4 inside each node and PP2 between the two nodes, over Ray.

**TP4 satisfies the FP8 block constraint.** The FP8 weights are block quantized with
`weight_block_size` [128, 128], so every tensor-parallel shard of a quantized dimension must be a
multiple of 128. `moe_intermediate_size` is 3072, so:

| Tensor parallel size | Shard of 3072 | Multiple of 128 | Verdict |
| --- | --- | --- | --- |
| 4 | 768 | yes, 6 x 128 | legal, and what this recipe uses |
| 8 | 384 | yes, 3 x 128 | legal, but there are only 4 GPUs per node here |
| 16 | 192 | no, 1.5 x 128 | rejected at startup |

The mixed precision is where this hardware choice breaks down. Attention and dense weights are FP8
with `ue8m0` scales, which Hopper runs natively, while the routed experts are FP4, which it does not,
and the experts are the majority of the 806 GiB. That is the whole argument for the RTX variant.

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

**The in-checkpoint MTP head cannot be used in this configuration.** The checkpoint ships 2343
`mtp.0.*` tensors and declares `num_nextn_predict_layers: 1`, so a speculative head is present and
vLLM 0.25.1 has both the MTP methods and a `dspark` draft path that could use it. Pipeline parallelism
is required to span two nodes, and a speculative config is rejected when pipeline parallelism is
active, so those weights sit idle. Pure TP8 across both nodes would preserve speculative decoding in
principle and is divisibility-legal, but cross-node tensor parallelism hangs at NCCL initialization on
this cluster.

## Gotchas

<!-- issue:deepgemm-h200-crash begin -->
**Leave `VLLM_USE_DEEP_GEMM` at 0 on H200.** The DeepGEMM MoE path takes an illegal memory access on
GLM-5.2's sparse attention, and forcing `VLLM_USE_DEEP_GEMM=1` on H200 independently reproduced the
same crash for Qwen3-Coder-480B-FP8. It is load-bearing for more than one model on this hardware, so
do not flip it without re-testing the model you are serving.
<!-- issue:deepgemm-h200-crash end -->

`env/env.sh` therefore keeps `VLLM_USE_DEEP_GEMM=0` for this recipe as a counter-warning rather than
an optimistic flip. Nothing about DeepSeek-V4 has been tested with it either way, and the two models
that were tested on this hardware both crashed with it on.

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

**An illegal memory access during startup is not automatically the FP4 issue.** Two other models on
this hardware crash during CUDA graph capture for unrelated reasons, which is why this recipe passes
`--enforce-eager` by default. If it fails in eager mode, the failure is more likely to be about the
expert format; if it only fails with `PERF=1`, that is the graph capture problem this hardware already
has with other checkpoints.

## Stop the endpoint

For a Slurm job, `scancel <jobid>` is all of it: the job requests both nodes and starts Ray through
`srun` inside that allocation, so Slurm's cgroups own every process on both nodes and remove them.

The SSH path has no such owner, so name both nodes explicitly:

```
bash common/tools/stop.sh <head_node> <worker_node>
```

That stops Ray as well as the server, then confirms both the GPUs and the host are clear. Do not reach for
`ssh <node> ray stop` by hand: `ray` lives in the recipe venv and is not on `PATH` in a non-interactive
shell, so it silently does nothing. A leftover Ray cluster is the other common cause of a failed relaunch,
and it fails on resources rather than on memory, which reads like a different problem entirely.

## Expected startup time

| Stage | Cold | Warm |
| --- | --- | --- |
| Environment build, one time | 5 to 15 min | skipped |
| Weight load, 806 GiB across 8 ranks | to be measured | to be measured |
| Total to first token | to be measured | to be measured |

First launch on a fresh node is slower than later ones: page cache is cold and any just-in-time kernel
compilation happens once. A launch that looks hung during this window is usually still loading. Check
the log before killing it. These numbers will be filled in if this recipe is run on hardware; they are
deliberately blank rather than guessed.
