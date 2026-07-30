# GLM-4.6-FP8 on one H200 node

Status: Validated - 2026-07-29, vLLM 0.25.1, protocol: slope(128,1152)

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
defaults, so a fresh clone runs as is. Four optional overrides, either exported or set in
`common/site.conf`:

| Variable | Default | Why you might change it |
| --- | --- | --- |
| `ACCOUNT` | unset | Your Slurm account, or pass `--account` at submit time |
| `GLM46_NODE` | unset | A node you already hold, for the SSH path |
| `MODELS_DIR` | shared testbed path | Point at your own faster copy of the checkpoint |
| `ENV_ROOT` | VAST scratch | Where this recipe builds its environment |

## Status

Validated on 2026-07-29. This recipe was run end to end on an H200 node: the environment was built from
`env/build.sh`, the endpoint was launched with `serve_ssh.sh`, and the decode rate was measured with
`common/tools/bench.sh` using the slope method. Key gating and the Anthropic endpoint were both
confirmed.

## What this is

GLM-4.6, FP8 quantized, a mixture-of-experts model with strong agentic coding behavior and native tool
calling. It exposes vLLM's Anthropic-compatible API, so Claude Code connects to it directly with no
proxy.

- Checkpoint directory: `GLM-4.6-FP8`
- Hugging Face repo: `zai-org/GLM-4.6-FP8`
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/GLM-4.6-FP8`

The testbed path works out of the box. Copying the checkpoint into your own VAST scratch space loads
faster, because VAST outperforms Lustre for this workload, and the directory names are identical in
both locations so only `MODELS_DIR` changes. Scratch has a 90-day retention policy, so treat it as a
fast cache and keep testbed as the system of record.

## Hardware

| Requirement | Value |
| --- | --- |
| GPU | H200, 4 per node, 143771 MiB each |
| Nodes | 1 |
| Parallelism | TP4 |
| Partition | `kempner_h200` |
| Per-GPU allocation limit | 16 CPUs, 360 GB host memory |
| Maximum wall time | 2 days |

All H200 nodes on this cluster share one hardware specification, so any node in the partition works.

## Environment build

This recipe builds its own environment, shared with no other recipe. Roughly 13 GB, and it lands under
`ENV_ROOT` on VAST scratch rather than in the repo, because startup is dominated by page faulting the
torch shared objects and stat-ing tens of thousands of small package files: measured on GPU nodes,
the interval from process start to the first vLLM log line was about 14 minutes from Lustre and 58
seconds from VAST. A bare torch and vLLM import from VAST is 9.2 seconds, so most of that 58 seconds
is engine startup rather than filesystem cost.

```
bash recipes/GLM-4.6-FP8/h200-4/env/build.sh
```

That is the only supported build path, because the install needs uv flags a requirements file cannot
express. What it does, and why:

<!-- issue:hopper-cu129-wheel begin -->
**Hopper nodes need the cu129 wheel, not vLLM's default.** These nodes run NVIDIA driver 575
(CUDA 12.9), which cannot run vLLM's default CUDA 13 PyPI wheel. The recipe installs the cu129
release wheel from the vLLM GitHub release with `--torch-backend=cu129`.
<!-- issue:hopper-cu129-wheel end -->

Scratch expires after 90 days, so this environment is disposable. Rebuild it with the same command,
or `--force` to replace an existing one. `env/requirements.lock` records the exact resolution that was
tested, and `env/WHEELS` records the non-PyPI artifact URLs with hashes.

## Launch

Canonical path, submitted from the repo root:

```
sbatch --account=<your-account> recipes/GLM-4.6-FP8/h200-4/serve.sbatch
```

Find the host once it starts, then use that node name with the client:

```
squeue --me                       # NODELIST column
tail -f glm46-<jobid>.log
```

Secondary path, for a node you already hold. Reserved nodes are removed from the scheduler, which is
why this exists:

```
bash recipes/GLM-4.6-FP8/h200-4/serve_ssh.sh <node>
```

Submit from the repo root either way. Slurm stages the batch script into its own spool directory, so
the script cannot locate the repo from its own path and resolves paths against the submit directory
instead.

## Verify

```
KEY=$(cat secrets/vllm_api_key)
NODE=<the node serving it>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models          # must print 401

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"glm-4.6","messages":[{"role":"user","content":"What is 2+2? Answer briefly."}],"max_tokens":400}'
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
source recipes/GLM-4.6-FP8/h200-4/client.env
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
`http://<node>:8000/v1`, the same key, and model name `glm-4.6`.

## Tunable inputs

Every variable this recipe honors, with its default and effect.

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/GLM-4.6-FP8` | Serve a different copy of the checkpoint |
| `API_PORT` | 8000 | Listening port |
| `MAX_MODEL_LEN` | 131072 | Context window; larger costs KV cache memory |
| `GPU_UTIL` | 0.90 | Fraction of VRAM for weights plus KV cache |
| `TP` | 4 | Tensor parallel size; 4 is the node's GPU count |
| `MTP_TOKENS` | 1 | Speculative tokens per step |
| `NO_MTP` | unset | Set to disable speculative decoding |
| `PERF` | unset | Retry CUDA graphs after an engine upgrade; currently crashes |
| `TOOL_PARSER` | `glm45` | Tool call parser |
| `REASONING_PARSER` | `glm45` | Reasoning parser |
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
| TP4, eager, MTP 1 token | 18.5 tok/s | slope(128,1152) |

Context window 200K. Measured 2026-07-29 on an idle node, two runs agreeing to within 0.01 tok/s, at
54.05 ms per token. Re-measure with:

```
bash common/tools/bench.sh --host <node> --model glm-4.6
```

The pre-restructure single-generation figure for this model was about 18 tok/s, so the slope method
barely moved it. That is expected here and worth understanding: the single-generation protocol
overstates decode time by counting prefill and fixed per-request cost, and at 18 tok/s a 1152-token
generation takes 62 seconds, so a fraction of a second of fixed cost is negligible. The bias matters for
fast models, not slow ones. On a fast single-GPU model in this repo the same correction moved a
measurement from 116 to 184 tok/s.

What did not help: CUDA graphs and torch.compile crash outright on this hardware, so the usual decode
win is unavailable and MTP speculative decoding is the substitute.

## Parallelism and quantization

TP4 uses all four GPUs of one node, where all-reduce runs over NVLink. The FP8 weights halve memory
against bf16 and are what let this model fit one node with a 200K context.

Pipeline parallelism is not used and not needed here, which matters because vLLM rejects speculative
decoding when pipeline parallelism is active. Staying single-node is what makes MTP available.

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

For a Slurm job, `scancel <jobid>`. For the SSH path:

```
bash common/tools/stop.sh <node>
```

That kills the server processes and waits for GPU memory to be released. Confirm with
`ssh <node> nvidia-smi --query-gpu=memory.used --format=csv,noheader`, which should read 0 MiB on all
four GPUs before you relaunch, or the next start will fail on memory.

## Expected startup time

| Stage | Measured |
| --- | --- |
| Environment build, one time | about 6 min, 215 packages |
| Launch to serving, cold page cache | 6 min 6 s |

Measured 2026-07-29 with the checkpoint read from VAST scratch rather than the default testbed path;
reading from Lustre is slower. First launch on a fresh node is slower than later ones, because the page
cache is cold and any just-in-time kernel compilation happens once. A launch that looks hung during this
window is usually still loading, so check the log before killing it.
