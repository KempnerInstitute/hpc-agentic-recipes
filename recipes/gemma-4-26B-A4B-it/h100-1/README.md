# gemma-4-26B-A4B-it on one H100 GPU

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
| `GEMMA26_H100_NODE` | unset | An H100 node you already hold, for the SSH path |
| `MODELS_DIR` | shared testbed path | Point at your own faster copy of the checkpoint |
| `ENV_ROOT` | VAST scratch | Where this recipe builds its environment |

## Status

Validated. The environment was built from `env/build.sh`, the endpoint was
launched with `serve.sbatch` through Slurm on one H100 GPU, and throughput was measured with `common/tools/bench.sh`
across concurrency 1, 8, 32, 64, 128, 256 and 512. The endpoint was still answering after the sweep finished.

Single stream and saturated throughput are different measurements and neither substitutes for
the other. See Measured performance below for the full curve and the disclosure block.

## What this is

Gemma 4 26B-A4B, instruction tuned: a mixture-of-experts model that activates 4B of its 26B
parameters per token, with native tool calling and a separate reasoning channel. It is the cheapest
endpoint in this repo, one GPU with no NCCL and no multi-node coordination, and among the fastest by
decode rate, which makes it a good default for interactive work. It exposes vLLM's
Anthropic-compatible API, so Claude Code connects to it directly with no proxy.

- Checkpoint directory: `gemma-4-26B-A4B-it`
- Hugging Face repo: not recorded upstream; the testbed copy is the system of record
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/gemma-4-26B-A4B-it`
- On disk: 51.6 GB, bf16, `Gemma4ForConditionalGeneration`, multimodal, 256K context
- Optional drafter: `gemma-4-26B-A4B-it-assistant`, under 1 GB, wired through `SPEC_DRAFT` and
  currently unusable on this engine (see Gotchas)

The testbed path works out of the box. Copying the checkpoint into your own VAST scratch space loads
faster, because VAST outperforms Lustre for this workload, and the directory names are identical in
both locations so only `MODELS_DIR` changes. Scratch has a 90-day retention policy, so treat it as a
fast cache and keep testbed as the system of record.

## Hardware

| Requirement | Value |
| --- | --- |
| GPU | H100, 1 of the 4 on the node, 80 GB |
| Nodes | 1 |
| Parallelism | TP1 |
| Partition | `kempner_h100` |
| Per-GPU allocation limit | 24 CPUs |
| Maximum wall time | 2 days |

All H100 nodes on this cluster share one hardware specification, so any node in the partition works.
This recipe claims one GPU and leaves the other three to other jobs, so expect to share the node; on
the SSH path set `GPU=<n>` to pin a device.

Weights are 51.6 GB and the KV cache costs 40 KiB per token, from 5 full attention layers plus 25
sliding-window layers of 1024 tokens. So 32K of context costs 1.5 GB and the full 256K costs 10 GB.
This is the tightest of the three GPU types for this checkpoint: at the 0.90 utilization default an
80 GB card leaves roughly 20 GB after weights, which still covers the full 256K context, but there is
no headroom for a second model on the same device.

Driver 575 and CUDA 12.9 are why this variant has its own toolchain, and why the RTX, H200, and H100
variants of this checkpoint are three separate recipes rather than one recipe with a switch.

## Environment build

This recipe builds its own environment, shared with no other recipe. Roughly 13 GB, and it lands under
`ENV_ROOT` on VAST scratch rather than in the repo, because startup is dominated by page faulting the
torch shared objects and stat-ing tens of thousands of small package files: measured on GPU nodes,
the interval from process start to the first vLLM log line was about 14 minutes from Lustre and 58
seconds from VAST. A bare torch and vLLM import from VAST is 9.2 seconds, so most of that 58 seconds
is engine startup rather than filesystem cost.

```
bash recipes/gemma-4-26B-A4B-it/h100-1/env/build.sh
```

That is the only supported build path, because the install needs uv flags a requirements file cannot
express. What it does, and why:

<!-- issue:hopper-cu129-wheel begin -->
**Hopper nodes need the cu129 wheel, not vLLM's default.** These nodes run NVIDIA driver 575
(CUDA 12.9), which cannot run vLLM's default CUDA 13 PyPI wheel. The recipe installs the cu129
release wheel from the vLLM GitHub release with `--torch-backend=cu129`.
<!-- issue:hopper-cu129-wheel end -->

Ray is installed alongside vLLM. A TP1 endpoint never uses it, but it is what the earlier
environment contained, so keeping it means the rate below was measured in this exact environment.

Scratch expires after 90 days, so this environment is disposable. Rebuild it with the same command,
or `--force` to replace an existing one. `env/requirements.lock` records the exact resolution that was
tested, and `env/WHEELS` records the non-PyPI artifact URLs with hashes.

## Launch

Canonical path, submitted from the repo root:

```
sbatch --account=<your-account> recipes/gemma-4-26B-A4B-it/h100-1/serve.sbatch
```

Find the host once it starts, then use that node name with the client:

```
squeue --me                       # NODELIST column
tail -f gemma26-h100-1-<jobid>.log
```

Advanced path, for a node you already hold. Use the Slurm submission above unless you already have
the node, or you are deploying an endpoint on behalf of others:

```
bash recipes/gemma-4-26B-A4B-it/h100-1/serve_ssh.sh <node>
GPU=1 bash recipes/gemma-4-26B-A4B-it/h100-1/serve_ssh.sh <node>    # pin one device of a busy node
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
  -d '{"model":"gemma-4-26b","messages":[{"role":"user","content":"What is 2+2? Answer briefly."}],"max_tokens":400}'
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
source recipes/gemma-4-26B-A4B-it/h100-1/client.env
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
`http://<node>:8000/v1`, the same key, and model name `gemma-4-26b`.

## Tunable inputs

Every variable this recipe honors, with its default and effect.

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/gemma-4-26B-A4B-it` | Serve a different copy of the checkpoint |
| `API_PORT` | 8000 | Listening port, and part of the SSH log file name |
| `MAX_MODEL_LEN` | 32768 | Context window; 262144 is supported and costs 10 GB of KV cache |
| `GPU_UTIL` | 0.90 | Fraction of VRAM for weights plus KV cache |
| `TP` | 1 | Tensor parallel size; one GPU holds this checkpoint at full context |
| `QUANT` | unset, meaning bf16 | `fp8` quantizes weights on load; measured no change in rate for this MoE, so use it only to free VRAM |
| `KV_FP8` | unset | `--kv-cache-dtype fp8`; halves KV bytes, measured no change in rate |
| `ENFORCE_EAGER` | unset | Skip torch.compile and CUDA graph capture, for debugging a startup failure |
| `SPEC_DRAFT` | unset | Path to the drafter checkpoint. Must stay unset on vLLM 0.25.1 and 0.26.0, where it fails at the first request; see Gotchas |
| `SPEC_TOKENS` | 3 | Speculative tokens per step, only read when `SPEC_DRAFT` is set |
| `EXTRA_ARGS` | unset | Extra flags appended to the `vllm serve` command line |
| `TOOL_PARSER` | `gemma4` | Tool call parser |
| `REASONING_PARSER` | `gemma4` | Reasoning parser |
| `GPU` | unset | SSH path only: pin one device of a shared node, for example `GPU=1` |
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

| Configuration | Aggregate rate | Per stream | Latency |
| --- | --- | --- | --- |
| Single stream, concurrency 1 | 203.4 tok/s | 203.4 tok/s | TTFT median 21 ms, n=3 spanning 203.4 to 203.4 |
| Concurrency 8 | 1302.8 tok/s | 162.8 tok/s | TTFT median 36 ms, p90 45 ms, n=3 spanning 1302.7 to 1323.4 |
| Concurrency 32 | 3591.1 tok/s | 112.2 tok/s | TTFT median 79 ms, p90 97 ms, n=3 spanning 3576.4 to 3592.9 |
| Concurrency 64 | 5273.5 tok/s | 82.4 tok/s | TTFT median 130 ms, p90 166 ms, n=3 spanning 5267.2 to 5345.9 |
| Concurrency 128 | 5417.1 tok/s | 42.3 tok/s | TTFT median 197 ms, p90 293 ms, n=3 spanning 5275.6 to 5465.3 |
| Concurrency 256 | 6614.8 tok/s | 25.8 tok/s | TTFT median 332 ms, p90 545 ms, n=3 spanning 6579.0 to 6619.5 |
| Concurrency 512 | 7164.7 tok/s | 14.0 tok/s | TTFT median 566 ms, p90 974 ms, n=3 spanning 7142.7 to 7173.4 |
| Concurrency 640 (peak) | 7242.8 tok/s | 11.3 tok/s | TTFT median 703 ms, p90 1214 ms, n=3 spanning 7236.1 to 7251.9 |
| Concurrency 768 | 7196.0 tok/s | 9.4 tok/s | TTFT median 835 ms, p90 1443 ms, n=3 spanning 7161.1 to 7243.5 |
| Concurrency 896 | 7147.8 tok/s | 8.0 tok/s | TTFT median 948 ms, p90 1653 ms, n=3 spanning 7139.1 to 7218.1 |
| Concurrency 1024 | 7082.7 tok/s | 6.9 tok/s | TTFT median 1107 ms, p90 1929 ms, n=3 spanning 7073.2 to 7137.4 |

Measured with `common/tools/bench.sh`. Full disclosure, without which a tokens
per second figure cannot be compared against anything:

| Parameter | Value |
| --- | --- |
| ISL, input tokens | 19 |
| OSL, output tokens | 1152, as the slope between 128 and 1152 |
| Counted | output tokens only, never input plus output |
| Concurrency levels | 1,8,32,64,128,256,512,640,768,896,1024 |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| `max_num_seqs` | engine default, 1024 on this hardware |
| Hardware | one H100 GPU |

Quote 203.4 tok/s for interactive coding, where one person waits on one
response. Quote 7242.8 tok/s at concurrency 640 for a shared endpoint under load.
The two measure different things and neither substitutes for the other.

Throughput **peaks at concurrency 640** and falls to 7083 tok/s by concurrency 1024, so 7242.8 tok/s is a
measured ceiling for this recipe rather than the edge of the sweep.

Concurrency 512 was measured in both runs, at 7164.7 and 6773.9 tok/s, a -5.5 percent
difference. That is the check that the two halves of this curve are comparable.

The input sequence here is short, which is the best case for decode. Measure with
`--prompt-tokens` at your working context before quoting a number for long-context work.

## Parallelism and quantization

TP1 is not a compromise here. One 80 GB GPU holds the 51.6 GB of bf16 weights plus a 10 GB KV cache at
the full 256K context, and at TP1 there is no all-reduce at all, so the node's NVLink fabric is left
for jobs that need it.

bf16, with no `--quantization` flag, is the measured right answer for this checkpoint, and it is the
most interesting thing about it. FP8 weights changed the decode rate by nothing. The model activates
only 4B of its 26B parameters per token, so there is very little weight traffic for a narrower dtype
to save: during decode, GPU utilization measured 35 to 40 percent and power draw about 210 W of a
700 W budget. This endpoint is host overhead bound, not memory bandwidth bound, so quantization
addresses the wrong constraint. The same regime explains why the rate scales only about 1.3x per GPU
tier instead of tracking HBM bandwidth.

`QUANT=fp8` is still worth knowing about on this GPU type for a different reason: it is the only one of
the three where VRAM is not abundant, so quantizing weights is how you would fit a long context and a
second workload on the same card. It costs nothing in rate, and it gains nothing either.
`KV_FP8=1` also measured no change in rate.

The contrast with the dense 31B sibling is the point: there FP8 is worth 69 percent on this same GPU,
because a dense model reads all of its weights for every token. Same family, same flag, opposite
answer.

TP2 was not measured. The bottleneck is host overhead rather than weight bandwidth, so more GPUs
address the wrong constraint, but the honest statement is that it is untested.

## Gotchas

<!-- issue:gemma4-mtp-broken begin -->
**Speculative decoding is unavailable for Gemma 4 on this engine.** The drafter checkpoints
(`*-it-assistant`) are downloaded and wired through `SPEC_DRAFT`, but the `gemma4_mtp` method is
broken in both vLLM 0.25.1 and 0.26.0, failing with a shape mismatch:

```
a and b must have same reduction dim, but got [s47, 3840] X [5632, 1024]
```

The drafter expects twice 2816 rather than 5632. The fix (`_maybe_share_embeddings`) exists only on
vLLM's `main` branch, so this needs a build newer than 0.26.0. Do not enable `SPEC_DRAFT` on 0.25.1
or 0.26.0; it fails at the first request.
<!-- issue:gemma4-mtp-broken end -->

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
`ssh <node> nvidia-smi --query-gpu=memory.used --format=csv,noheader`, which should read 0 MiB on the
device this endpoint used before you relaunch, or the next start will fail on memory. On a shared node
check only that device: another job may legitimately be holding the other three.

## Expected startup time

| Stage | Cold | Warm |
| --- | --- | --- |
| Environment build, one time | 5 to 15 min | skipped |
| Weight load | to be measured | to be measured |
| Total to first token | to be measured | to be measured |

First launch on a fresh node is slower than later ones: page cache is cold and any just-in-time kernel
compilation happens once. A launch that looks hung during this window is usually still loading. Check
the log before killing it. These numbers will be filled in when this recipe is validated on hardware;
they are deliberately blank rather than guessed.
