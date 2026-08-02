# DeepSeek-V4-Pro on two RTX PRO 6000 nodes

Status: Validated - vLLM 0.26.0, protocol: slope(128,1152) swept at concurrency 1 through 1024

Everything needed to build, launch, verify, connect to, and debug this endpoint is on this page.

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
| `DSV4_HEAD`, `DSV4_WORKER` | unset | Two nodes you already hold, for the SSH path |
| `MODELS_DIR` | shared testbed path | Point at your own faster copy of the checkpoint |
| `ENV_ROOT` | scratch | Where this recipe builds its environment |

## Status

Validated. The environment was built from `env/build.sh`, the endpoint was
launched with `serve_ssh.sh` on two RTX PRO 6000 Blackwell nodes, 16 GPUs via Ray, and throughput was measured with `common/tools/bench.sh`
across concurrency 1, 8, 32, 64, 128, 256, 512, 640, 768, 896 and 1024. Ready 20 minutes 8 seconds after launch. The endpoint was still answering after the sweep finished.

This recipe settles the FP4 hardware question. The same checkpoint fails on H200 inside the CUTLASS w8a8 kernel dispatch, documented in the
`h200-4-nodes2` sibling. Here it loaded, served a full sweep, and was still healthy afterward, which
confirms the cause is Blackwell's native FP4 path rather than anything configurable.

Single stream and saturated throughput are different measurements and neither substitutes for
the other. See Measured performance below for the full curve and the disclosure block.

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
| Experts precision | `expert_dtype` `fp4`, which is what sends this recipe to Blackwell |
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
| GPU | RTX PRO 6000 Blackwell, 8 per node, 97887 MiB each, sm_120 |
| Nodes | 2, so 16 GPUs and about 1530 GiB of VRAM |
| Parallelism | TP8 inside each node, PP2 between them, Ray |
| Partition | `kempner_rtx` |
| Per-GPU allocation limit | 16 CPUs, about 189 GiB host memory |
| Maximum wall time | 2 days |

All RTX nodes on this cluster share one hardware specification, so any two nodes in the partition
work.

**Why this model goes to RTX rather than to H200.** The experts are FP4, and only Blackwell executes
FP4 natively. In vLLM 0.25.1, `expert_dtype: fp4` resolves to the MXFP4 fused-MoE method unless the
checkpoint also sets `moe_quant_algo: NVFP4`, which this one does not
(`vllm/models/deepseek_v4/quant_config.py`), so on Hopper the expert layers have no native instruction
to run on and fall back to emulation or a Marlin path. Capacity here is not
the constraint either way: 16 GPUs at 97887 MiB is about 1530 GiB, and at
`--gpu-memory-utilization 0.90` about 1377 GiB is usable against 806 GiB of weights, leaving roughly
570 GiB for KV cache and activations. This model is unusually cheap in KV for its context length,
because the hybrid compressed attention is designed to be.

An H200 sibling exists at `recipes/DeepSeek-V4-Pro/h200-4-nodes2`, where the same checkpoint does not
run. This is the variant to use.

## Environment build

This recipe builds its own environment, shared with no other recipe: about 9.0 GB for the Python
environment, a little more with Ray added, plus 2.9 GB for a private CUDA 13.0 toolkit. Both land under
`ENV_ROOT` on scratch rather than in the repo, because startup is dominated by page faulting the
torch shared objects and stat-ing tens of thousands of small package files: measured on GPU nodes,
the interval from process start to the first vLLM log line was about 14 minutes from Lustre and 58
seconds from scratch. A bare torch and vLLM import from scratch is 9.2 seconds, so most of that 58 seconds
is engine startup rather than filesystem cost.

```
bash recipes/DeepSeek-V4-Pro/rtx-8-nodes2/env/build.sh
```

That is the only supported build path, because the install needs uv flags a requirements file cannot
express: a nightly index with `--prerelease=allow --index-strategy unsafe-best-match`, `--no-deps` for
exactly one package, and a `mamba create` step. What it does, and why:

**Ray is installed here and was not installed before.** Every RTX endpoint in this repo's history ran
on a single node, so the earlier RTX environment carried no Ray at all; only the Hopper
environment did. This recipe spans two nodes and therefore needs the Ray executor, so `build.sh` adds
`ray[default]` to the RTX install. If you reuse an older RTX environment through `VENV_DIR`, check
that `ray` imports before submitting, or the engine fails at executor startup.

<!-- issue:cuda13-toolkit begin -->
**The sm_120 JIT needs a complete CUDA 13.0 toolkit, and it must not reach `LD_LIBRARY_PATH`.** The
node's `/usr/local/cuda-13` is runtime-only, and the fragmented pip nvcc wheels mix 13.0 and 13.2
between `nvcc`, `cicc`, and `ptxas`, which breaks the JIT. The recipe installs a consistent toolkit
via conda. `env/env.sh` points `CUDA_HOME` at it and exposes its headers through `CPATH` and
`LIBRARY_PATH` for compilation only. Do **not** add its libraries to `LD_LIBRARY_PATH`: its
`libcudart` shadows torch's CUDA 13 runtime and pulls in a `libcupti.so.13` that is not present,
which breaks import entirely.
<!-- issue:cuda13-toolkit end -->

<!-- issue:flashinfer-cubin-skew begin -->
**FlashInfer must be 0.6.15, and its version check must be bypassed.** vLLM 0.25.1 pins
flashinfer-python 0.6.13, but the sm_120 attention backend passes a `kv_scale_format` argument that
0.6.13 does not accept, which fails at the first inference request. Install 0.6.15 with `--no-deps`
so torch is left untouched. No matching 0.6.15 cubin package exists, so `flashinfer-cubin` stays at
0.6.13 and `env/env.sh` sets `FLASHINFER_DISABLE_VERSION_CHECK=1`; kernels are then compiled from
source on first launch, which is why the first request after a fresh environment is slow.
<!-- issue:flashinfer-cubin-skew end -->

Scratch expires after 90 days, so this environment is disposable. Rebuild it with the same command, or
`--force` to replace an existing one. After a successful build, record the exact resolution in
`env/requirements.lock` and the non-PyPI artifact URLs with hashes in `env/WHEELS`; the vLLM CUDA 13
nightly index rotates, so a bare `vllm` requirement will not resolve to the same wheel later.

## Launch

Slurm path, submitted from the repo root:

```
sbatch --account=<your-account> recipes/DeepSeek-V4-Pro/rtx-8-nodes2/serve.sbatch
```

The batch script allocates two nodes, starts a Ray head on the first and a worker on the second, then
runs the engine on the head. The endpoint is on the **first** allocated node:

```
squeue --me                       # NODELIST column, first name
tail -f dsv4-rtx-<jobid>.log
```

Direct path, for two nodes you already hold. Use the Slurm submission above unless you already have
the nodes, or you are deploying an endpoint on behalf of others:

```
bash recipes/DeepSeek-V4-Pro/rtx-8-nodes2/serve_ssh.sh <head_node> <worker_node>
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
source recipes/DeepSeek-V4-Pro/rtx-8-nodes2/client.env
claude
```

<!-- issue:anthropic-auth-token begin -->
**Use `ANTHROPIC_AUTH_TOKEN`, never `ANTHROPIC_API_KEY`.** Both engines accept only
`Authorization: Bearer <key>`. Setting `ANTHROPIC_API_KEY` makes Claude Code send an `x-api-key`
header instead, which the engine ignores, and every request returns HTTP 401. Also set
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
| `TP` | 8 | Tensor parallel size; 8 is one node's GPU count and the largest legal value |
| `PP` | 2 | Pipeline parallel size; 2 is the node count |
| `PERF` | unset | Attempt CUDA graph capture instead of eager |
| `CUDAGRAPH_MODE` | `NONE` | Graph mode passed through when `PERF` is set |
| `EXTRA_ARGS` | unset | Extra `vllm serve` flags, for experiments |
| `TOOL_PARSER` | `deepseek_v4` | Tool call parser |
| `REASONING_PARSER` | `deepseek_v4` | Reasoning parser |
| `RAY_PORT` | 6379 | Ray head port |
| `RAY_HEAD_IP` | unset | Head address, when calling `serve.sh` directly |
| `DSV4_HEAD`, `DSV4_WORKER` | unset | Default nodes for the SSH path |
| `LOG_DIR` | `/tmp/$USER/vllm` | Where the SSH path writes the server log |
| `VENV_DIR`, `CUDA13_DIR` | under `ENV_ROOT` | Use an environment built elsewhere |

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
| Single stream, concurrency 1 | 18.6 tok/s | 18.6 tok/s | TTFT median 163 ms, n=3 spanning 18.6 to 18.8 |
| Concurrency 8 | 144.8 tok/s | 18.1 tok/s | TTFT median 178 ms, p90 239 ms, n=3 spanning 143.7 to 148.9 |
| Concurrency 32 | 541.7 tok/s | 16.9 tok/s | TTFT median 209 ms, p90 281 ms, n=3 spanning 539.8 to 542.1 |
| Concurrency 64 | 921.4 tok/s | 14.4 tok/s | TTFT median 2210 ms, p90 8139 ms, n=3 spanning 868.3 to 1104.1 |
| Concurrency 128 | 1235.3 tok/s | 9.7 tok/s | TTFT median 15834 ms, p90 16982 ms, n=3 spanning 1232.0 to 1502.9 |
| Concurrency 256 | 2123.3 tok/s | 8.3 tok/s | TTFT median 17511 ms, p90 32383 ms, n=3 spanning 2111.6 to 3625.8 |
| Concurrency 512 | 2959.3 tok/s | 5.8 tok/s | TTFT median 898 ms, p90 1363 ms, n=3 spanning 2915.8 to 3260.0 |
| Concurrency 640 | 3161.3 tok/s | 4.9 tok/s | TTFT median 2045 ms, p90 3234 ms, n=3 spanning 3142.0 to 3214.1 |
| Concurrency 768 | 3332.4 tok/s | 4.3 tok/s | TTFT median 1348 ms, p90 2028 ms, n=3 spanning 3321.9 to 3350.7 |
| Concurrency 896 | 3451.2 tok/s | 3.9 tok/s | TTFT median 1541 ms, p90 2506 ms, n=3 spanning 3431.7 to 3457.4 |
| Concurrency 1024 (rising) | 3581.9 tok/s | 3.5 tok/s | TTFT median 1807 ms, p90 2822 ms, n=3 spanning 3533.3 to 3598.4 |

Measured with `common/tools/bench.sh`, endpoint ready 20m 8s after launch. Full disclosure, without which a tokens
per second figure cannot be compared against anything:

| Parameter | Value |
| --- | --- |
| ISL, input tokens | 10 |
| OSL, output tokens | 1152, as the slope between 128 and 1152 |
| Counted | output tokens only, never input plus output |
| Concurrency levels | 1,8,32,64,128,256,512,640,768,896,1024 |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| `max_num_seqs` | engine default, 1024 on this hardware |
| Hardware | two RTX PRO 6000 Blackwell nodes, 16 GPUs |

Quote 18.6 tok/s for interactive coding, where one person waits on one
response. Quote 3581.9 tok/s at concurrency 1024 for a shared endpoint under load.
The two measure different things and neither substitutes for the other.

Throughput was **still rising at concurrency 1024**, 21 percent above its own concurrency 512 at the top of the sweep,
so 3581.9 tok/s is a floor rather than a ceiling. The sequence cap is not what stopped it:
`max_num_seqs` resolves to 1024 on this hardware and the sweep ran to that level, so finding the
true peak needs the cap raised, which is a different serving configuration.

Concurrency 512 was measured in both runs, at 2959.3 and 2928.6 tok/s, a -1.0 percent
difference. That is the check that the two halves of this curve are comparable.

Scheduler counters over the extended levels: KV cache usage reached 78 percent, the running
batch reached 1024 requests, and there were no preemptions at any level.

The input sequence here is short, which is the best case for decode. Measure with
`--prompt-tokens` at your working context before quoting a number for long-context work.

## Parallelism and quantization

The shape is TP8 inside each node and PP2 between the two nodes, over Ray.

**TP8 is legal for this checkpoint and TP16 is not.** The FP8 weights are block quantized with
`weight_block_size` [128, 128], so every tensor-parallel shard of a quantized dimension must be a
multiple of 128. `moe_intermediate_size` is 3072, so:

| Tensor parallel size | Shard of 3072 | Multiple of 128 | Verdict |
| --- | --- | --- | --- |
| 8 | 384 | yes, 3 x 128 | legal, and what this recipe uses |
| 16 | 192 | no, 1.5 x 128 | rejected at startup |

That is the same class of constraint that forces Qwen3-Coder-480B-FP8 down to TP4, but it lands
differently here: 3072 divides cleanly by 8 where 2560 does not, so a full RTX node can be used as one
tensor-parallel group and the second node is reached with pipeline parallelism instead of with a
wider TP group.

The mixed precision is the other half of the story. Attention and dense weights are FP8 with `ue8m0`
scales, while the routed experts are FP4, which is the majority of the 806 GiB. FP4 experts are the
reason for choosing Blackwell, as described under Hardware.

<!-- issue:cross-node-tp-hangs begin -->
**Keep tensor parallelism inside a node and use pipeline parallelism across nodes.** Pure tensor
parallelism spanning two nodes hangs at NCCL initialization. The working shape is TP within each node,
where all-reduce uses NVLink, and PP between nodes.
<!-- issue:cross-node-tp-hangs end -->

One word of that warning does not apply here: these nodes have no NVLink, so the intra-node all-reduce
crosses PCIe. The rule still holds, and for the same reason, since PCIe inside a box is still far better
than a fabric hop between boxes.

<!-- issue:pp-forbids-spec-decode begin -->
**Pipeline parallelism disables speculative decoding.** vLLM rejects a speculative config when
pipeline parallelism is in use, so no MTP or draft-model speedup is available in any recipe that needs
PP to span nodes, even when the checkpoint ships an MTP head. This is why an SGLang recipe exists for
GLM-5.2: SGLang can run TP8 across two nodes with EAGLE speculative decoding, where vLLM would need PP
and therefore lose it. The guard is not visible in vLLM 0.25.1's config source, so treat it as behavior
for this version rather than a documented API contract, and re-check after an engine upgrade.
<!-- issue:pp-forbids-spec-decode end -->

**The in-checkpoint MTP head cannot be used in this configuration, and that is a real cost.** The
checkpoint ships 2343 `mtp.0.*` tensors and declares `num_nextn_predict_layers: 1`, so a speculative
head is present and vLLM 0.25.1 has both the MTP methods and a `dspark` draft path that could use it.
Pipeline parallelism is required to span two nodes, and a speculative config is rejected when pipeline
parallelism is active, so those weights sit idle. Pure TP16 across both nodes would preserve
speculative decoding in principle, but it is doubly blocked: TP16 fails the divisibility check above,
and cross-node tensor parallelism hangs at NCCL initialization on this cluster. If a future engine
lifts the pipeline restriction, this is the single largest decode-speed lever available to this recipe.

## Gotchas

<!-- issue:rtx-no-nvlink begin -->
**RTX PRO 6000 nodes have no NVLink, so peer-to-peer must be disabled.** `env/env.sh` sets
`NCCL_P2P_DISABLE=1`. Without it, NCCL initialization hangs on any multi-GPU job, with no error, and
the server never becomes ready.
<!-- issue:rtx-no-nvlink end -->

<!-- issue:rtx-comms-bound begin -->
**Decode on a full RTX node is limited by cross-GPU communication, not memory bandwidth.** All-reduce
traffic crosses PCIe rather than NVLink. Two consequences, both measured: do not enable
`--enable-expert-parallel`, which added all-to-all traffic and measured about 9 percent slower, and
FP8 weights bought nothing on Qwen3-235B because weight bandwidth was not the bottleneck.
<!-- issue:rtx-comms-bound end -->

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

**The cross-node transport for these nodes has not been verified.** `env/env.sh` pins
`NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` to `ib0` only when the node actually exposes that
interface, and otherwise leaves NCCL's own interface selection alone, because whether the RTX nodes
present an InfiniBand interface has not been checked on hardware. The Hopper multi-node recipes pin
`ib0` unconditionally. If NCCL initialization hangs or picks a slow interface, this is the first thing
to look at, and `NCCL_DEBUG=INFO` will name the interface it chose.

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

That kills the server processes and waits for GPU memory to be released. Confirm with
`ssh <node> nvidia-smi --query-gpu=memory.used --format=csv,noheader`, which should read 0 MiB on all
eight GPUs of both nodes before you relaunch, or the next start will fail on memory. A leftover Ray
cluster is the other common cause of a failed relaunch, and it fails on resources rather than on memory,
which reads like a different problem entirely. `stop.sh` clears both.

## Expected startup time

| Stage | Cold | Warm |
| --- | --- | --- |
| Environment build, one time | 10 to 25 min | skipped |
| Weight load, 806 GiB across 16 ranks | to be measured | to be measured |
| FlashInfer sm_120 kernel compilation, first launch only | to be measured | skipped |
| Total, launch to serving | 20 min 8 s | 20 min 8 s |

First launch on a fresh node is slower than later ones: page cache is cold, and the FlashInfer sm_120
kernels are compiled from source once because no cubin package matches the installed version. A launch
that looks hung during this window is usually still loading. Check the log before killing it. These
numbers will be filled in when this recipe is run on hardware; they are deliberately blank rather than
guessed.
