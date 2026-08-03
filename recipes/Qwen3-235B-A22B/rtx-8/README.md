# Qwen3-235B-A22B on one RTX PRO 6000 node

Status: Validated - vLLM 0.25.1, protocol: slope(128,1152) swept at concurrency 1 through 1024

Everything needed to build, launch, verify, connect to, and debug this endpoint is on this page.

## Configure once

Create the API key. The endpoint refuses requests without it, and the key reaches the engine through
the environment rather than as an argument, so it does not appear in `ps` output.

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/Qwen3-235B-A22B-rtx-8.key
chmod 600 secrets/Qwen3-235B-A22B-rtx-8.key
```

That key gates this recipe alone. When it is absent `secrets/vllm_api_key` is read instead, so a setup
made before per-model keys keeps working.

Nothing else is required. Cluster paths come from `common/defaults.sh`, which is tracked with working
defaults, so a fresh clone runs as is. Four optional overrides, either exported or set in
`common/site.conf`:

| Variable | Default | Why you might change it |
| --- | --- | --- |
| `ACCOUNT` | unset | Your Slurm account, or pass `--account` at submit time |
| `QWEN3_235B_NODE` | unset | A node you already hold, for the SSH path |
| `MODELS_DIR` | shared testbed path | Point at your own faster copy of the checkpoint |
| `ENV_ROOT` | scratch | Where this recipe builds its environment |

## Status

Validated. The environment was built from `env/build.sh`, the endpoint was
launched with `serve_ssh.sh` on one RTX PRO 6000 Blackwell node, and throughput was measured with `common/tools/bench.sh`
across concurrency 1, 8, 32, 64, 128, 256, 512, 640, 768, 896 and 1024. Ready 4 minutes 55 seconds after launch. The endpoint was still answering after the sweep finished.

Single stream and saturated throughput are different measurements and neither substitutes for
the other. See Measured performance below for the full curve and the disclosure block.

## What this is

Qwen3-235B-A22B, a bf16 mixture-of-experts model with 235B total parameters and 22B active per token
(94 layers, 128 experts, 8 routed per token, GQA with 64 query and 4 key-value heads). It is a thinking
model, and switches between thinking and non-thinking behavior within the same weights. It exposes
vLLM's Anthropic-compatible API, so Claude Code connects to it directly with no proxy.

- Checkpoint directory: `Qwen3-235B-A22B`
- Hugging Face repo: `Qwen/Qwen3-235B-A22B`
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Qwen3-235B-A22B`

The testbed path works out of the box. Copying the checkpoint into your own scratch space loads
faster, because scratch outperforms Lustre for this workload, and the directory names are identical in
both locations so only `MODELS_DIR` changes. Scratch has a 90-day retention policy, so treat it as a
fast cache and keep testbed as the permanent copy.

`qwen3_moe` is native to vLLM 0.25.1, so no out-of-tree model code and no `--trust-remote-code` are
needed here.

## Hardware

| Requirement | Value |
| --- | --- |
| GPU | RTX PRO 6000 Blackwell, 8 per node, 97887 MiB each |
| Architecture | sm_120, CUDA 13 |
| Interconnect | PCIe, no NVLink |
| Nodes | 1 |
| Parallelism | TP8 |
| Partition | `kempner_rtx` |
| Per-GPU allocation limit | 16 CPUs, about 189 GiB host memory |
| Maximum wall time | 2 days |

All RTX PRO 6000 nodes on this cluster share one hardware specification, so any node in the partition
works. The checkpoint is bf16 and about 470 GB across 118 shards, so it needs the whole node.
`serve.sbatch` requests 96 CPUs and 1000 GB, both inside the per-GPU limits for a whole node.

## Environment build

This recipe builds its own environment, shared with no other recipe: about 9.0 GB for the Python
environment plus 2.9 GB for a private CUDA 13.0 toolkit. Both land under `ENV_ROOT` on scratch
rather than in the repo, because startup is dominated by page faulting the torch shared objects and
stat-ing tens of thousands of small package files: measured on GPU nodes, the interval from process start to the first vLLM log line was about 14
minutes from Lustre and 58 seconds from scratch. A bare torch and vLLM import from scratch is 9.2
seconds, so most of that 58 seconds is engine startup rather than filesystem cost.

```
bash recipes/Qwen3-235B-A22B/rtx-8/env/build.sh
```

That is the only supported build path, because the install needs uv flags a requirements file cannot
express and a conda step no Python environment can carry. What it does, and why:

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

The vLLM wheel comes from `https://wheels.vllm.ai/nightly/cu130`, not from PyPI, because sm_120 needs
the CUDA 13 build and uv's `--torch-backend` maxes out at cu129. `build.sh` pins `vllm==0.25.1`
explicitly rather than leaving the version to resolve: the installed metadata reads a plain `0.25.1`
with no local version tag, so an unpinned install silently drifts, either to whatever the nightly
index holds that day or to the PyPI CUDA 13 wheel, and neither is the build these numbers were
measured on. Nightly wheels also rotate and are deleted, so a much later rebuild can fail outright
instead of drifting quietly.

Scratch expires after 90 days, so this environment is disposable. Rebuild it with the same command,
or `--force` to replace an existing one. Record the exact resolution in `env/requirements.lock` after a build, which is what makes a drifted
rebuild visible.

## Launch

Slurm path, submitted from the repo root:

```
sbatch --account=<your-account> recipes/Qwen3-235B-A22B/rtx-8/serve.sbatch
```

Find the host once it starts, then use that node name with the client:

```
squeue --me                       # NODELIST column
tail -f qwen3-235b-<jobid>.log
```

Direct path, for a node you already hold. Use the Slurm submission above unless you already have
the node, or you are deploying an endpoint on behalf of others:

```
bash recipes/Qwen3-235B-A22B/rtx-8/serve_ssh.sh <node>
```

Submit from the repo root either way. Slurm stages the batch script into its own spool directory, so
the script cannot locate the repo from its own path and resolves paths against the submit directory
instead.

Over bare SSH `nproc` reports 1 on these nodes while `Cpus_allowed_list` is the full set, so the
server is not actually CPU limited on that path even though it looks like it.

## Verify

```
KEY=$(cat secrets/Qwen3-235B-A22B-rtx-8.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the node serving it>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models          # must print 401

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"qwen3-235b","messages":[{"role":"user","content":"What is 2+2? Answer briefly."}],"max_tokens":400}'
```

A keyless request returning 401 is the expected, correct behavior.

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
export NODE=<the node serving it>
source recipes/Qwen3-235B-A22B/rtx-8/client.env
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
`http://<node>:8000/v1`, the same key, and model name `qwen3-235b`.

## Tunable inputs

Every variable this recipe honors, with its default and effect.

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/Qwen3-235B-A22B` | Serve a different copy of the checkpoint |
| `API_PORT` | 8000 | Listening port |
| `MAX_MODEL_LEN` | 40960 | Context window; larger costs KV cache memory |
| `GPU_UTIL` | 0.90 | Fraction of VRAM for weights plus KV cache |
| `TP` | 8 | Tensor parallel size; 8 is the node's GPU count |
| `QUANT` | unset | Quantize weights on load, for example `fp8`; measured no faster here |
| `TOOL_PARSER` | `hermes` | Tool call parser |
| `REASONING_PARSER` | `qwen3` | Reasoning parser |
| `LOG_DIR` | `/tmp/$USER/vllm` | Where the SSH path writes the server log |
| `VENV_DIR` | under `ENV_ROOT` | Use an environment built elsewhere |
| `CUDA13_DIR` | under `ENV_ROOT` | Use a CUDA 13.0 toolkit built elsewhere |
| `VLLM_VERSION` | 0.25.1 | Engine version `env/build.sh` installs |
| `FLASHINFER_VERSION` | 0.6.15 | FlashInfer version `env/build.sh` installs |

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
| Single stream, concurrency 1 | 63.3 tok/s | 63.3 tok/s | TTFT median 63 ms, n=3 spanning 62.4 to 63.4 |
| Concurrency 8 | 312.2 tok/s | 39.0 tok/s | TTFT median 74 ms, p90 77 ms, n=3 spanning 311.2 to 313.6 |
| Concurrency 32 | 901.0 tok/s | 28.2 tok/s | TTFT median 108 ms, p90 119 ms, n=3 spanning 817.8 to 901.8 |
| Concurrency 64 | 1286.9 tok/s | 20.1 tok/s | TTFT median 131 ms, p90 164 ms, n=3 spanning 1273.2 to 1288.4 |
| Concurrency 128 | 1734.0 tok/s | 13.5 tok/s | TTFT median 192 ms, p90 255 ms, n=3 spanning 1733.5 to 1745.9 |
| Concurrency 256 | 3127.4 tok/s | 12.2 tok/s | TTFT median 293 ms, p90 414 ms, n=3 spanning 3122.5 to 3134.7 |
| Concurrency 512 (peak) | 3983.5 tok/s | 7.8 tok/s | TTFT median 473 ms, p90 631 ms, n=3 spanning 3970.3 to 3984.6 |
| Concurrency 640 | 3696.6 tok/s | 5.8 tok/s | TTFT median 541 ms, p90 683 ms, n=3 spanning 3689.9 to 3713.7 |
| Concurrency 768 | 3541.6 tok/s | 4.6 tok/s | TTFT median 672 ms, p90 1020 ms, n=3 spanning 3526.7 to 3547.8 |
| Concurrency 896 | 3641.1 tok/s | 4.1 tok/s | TTFT median 608 ms, p90 916 ms, n=3 spanning 3637.5 to 3671.7 |
| Concurrency 1024 | 3750.3 tok/s | 3.7 tok/s | TTFT median 658 ms, p90 1012 ms, n=3 spanning 3741.6 to 3754.9 |

Measured with `common/tools/bench.sh`, endpoint ready 4m 55s after launch. Full disclosure, without which a tokens
per second figure cannot be compared against anything:

| Parameter | Value |
| --- | --- |
| ISL, input tokens | 17 |
| OSL, output tokens | 1152, as the slope between 128 and 1152 |
| Counted | output tokens only, never input plus output |
| Concurrency levels | 1,8,32,64,128,256,512,640,768,896,1024 |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| `max_num_seqs` | engine default, 1024 on this hardware |
| Hardware | one RTX PRO 6000 Blackwell node, 8 GPUs |

Quote 63.3 tok/s for interactive coding, where one person waits on one
response. Quote 3983.5 tok/s at concurrency 512 for a shared endpoint under load.
The two measure different things and neither substitutes for the other.

Throughput **peaks at concurrency 512** and falls to 3750 tok/s by concurrency 1024, so 3983.5 tok/s is a
measured ceiling for this recipe rather than the edge of the sweep.

Concurrency 512 was measured in both runs, at 3983.5 and 3979.4 tok/s, a -0.1 percent
difference. That is the check that the two halves of this curve are comparable.

Scheduler counters over the extended levels: KV cache usage reached 100 percent, the running
batch reached 1024 requests, and there were no preemptions at any level.

The input sequence here is short, which is the best case for decode. Measure with
`--prompt-tokens` at your working context before quoting a number for long-context work.

## Parallelism and quantization

TP8 spans all eight GPUs of one node. The weights are bf16 and unquantized, which is what the
checkpoint ships, and the model needs the full node for that reason.

The tensor-parallel all-reduces in all 94 layers cross host memory over PCIe, because the node has no
NVLink and peer-to-peer has to be disabled. That is the ceiling here, and it is why neither FP8 nor
expert parallelism helps. Prefix caching is enabled, which is a straight win for agentic use, where a
long unchanged system prompt and file context are re-sent on every turn.

Pipeline parallelism is not used and not needed on one node. This checkpoint ships no draft head, so
there is no speculative decoding to lose either way.

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

## Stop the endpoint

If you launched with Slurm, which is the default path, `scancel <jobid>` is all you need: Slurm terminates the job step and reclaims the node. The tool below is only for the direct SSH path, which has no scheduler to clean up after it.

```
bash common/tools/stop.sh <node>
```

That kills the server processes and waits for GPU memory to be released. Confirm with
`ssh <node> nvidia-smi --query-gpu=memory.used --format=csv,noheader`, which should read 0 MiB on all
eight GPUs before you relaunch, or the next start will fail on memory.

## Expected startup time

| Stage | Cold | Warm |
| --- | --- | --- |
| Environment build, one time | to be measured | skipped |
| First-time FlashInfer JIT | to be measured | skipped, cached under `/tmp/$USER/flashinfer` |
| Weight load | to be measured | to be measured |
| Total, launch to serving | 4 min 55 s | 4 min 55 s |

First launch on a fresh node is slower than later ones: page cache is cold, and the sm_120 attention
kernels are compiled from source once because no matching cubin package exists. Both caches are
node-local, so a different node pays the JIT cost again. A launch that looks hung during this window
is usually still loading. Check the log before killing it. These numbers will be filled in when this
recipe is validated on hardware; they are deliberately blank rather than guessed.
