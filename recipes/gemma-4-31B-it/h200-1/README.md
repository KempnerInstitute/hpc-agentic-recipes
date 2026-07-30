# gemma-4-31B-it on one H200 GPU

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
defaults, so a fresh clone runs as is. Optional overrides, either exported or set in
`common/site.conf`:

| Variable | Default | Why you might change it |
| --- | --- | --- |
| `ACCOUNT` | unset | Your Slurm account, or pass `--account` at submit time |
| `GEMMA31_H200_NODE` | unset | An H200 node you already hold, for the SSH path |
| `MODELS_DIR` | shared testbed path | Point at your own faster copy of the checkpoint |
| `ENV_ROOT` | VAST scratch | Where this recipe builds its environment |

## Status

Validated on 2026-07-29. This recipe was run end to end on an H200 node: the environment was built
from `env/build.sh`, the endpoint launched with `serve_ssh.sh`, and the rate measured with
`common/tools/bench.sh`. Ready in 7 minutes 38 seconds from launch, including CUDA graph capture in 20
seconds. Measured 85.2 tok/s, which independently reproduces the 85.3 tok/s recorded before the
restructure. Key gating and the Anthropic endpoint were both confirmed.

Untested (migrated). The decode rates below were measured on this GPU type on 2026-07-27, with the
pre-restructure scripts whose serve flags this recipe reproduces exactly, using the slope protocol.
The recipe files themselves have not been run yet, so treat startup behavior as unverified while the
rates are trustworthy.

## What this is

Gemma 4 31B, instruction tuned: a dense model, so every parameter is read for every token, with native
tool calling and a separate reasoning channel. It is the higher-quality half of the single-GPU Gemma 4
pair and roughly a third the decode rate of its 26B mixture-of-experts sibling. This is the fastest of
the three GPU variants of this checkpoint. It exposes vLLM's Anthropic-compatible API, so Claude Code
connects to it directly with no proxy.

- Checkpoint directory: `gemma-4-31B-it`
- Hugging Face repo: not recorded before the restructure; the testbed copy is the system of record
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/gemma-4-31B-it`
- On disk: 62.6 GB, bf16, `Gemma4ForConditionalGeneration`, multimodal, 256K context
- Optional drafter: `gemma-4-31B-it-assistant`, under 1 GB, wired through `SPEC_DRAFT` and currently
  unusable on this engine (see Gotchas)

This recipe serves the bf16 checkpoint with FP8 weight quantization applied at load time, which is why
there is no separate FP8 checkpoint directory.

The testbed path works out of the box. Copying the checkpoint into your own VAST scratch space loads
faster, because VAST outperforms Lustre for this workload, and the directory names are identical in
both locations so only `MODELS_DIR` changes. Scratch has a 90-day retention policy, so treat it as a
fast cache and keep testbed as the system of record.

## Hardware

| Requirement | Value |
| --- | --- |
| GPU | H200, 1 of the 4 on the node, 143771 MiB |
| Nodes | 1 |
| Parallelism | TP1 |
| Partition | `kempner_h200` |
| Per-GPU allocation limit | 16 CPUs, 360 GB host memory |
| Maximum wall time | 2 days |

All H200 nodes on this cluster share one hardware specification, so any node in the partition works.
This recipe claims one GPU and leaves the other three to other jobs, so expect to share the node; on
the SSH path set `GPU=<n>` to pin a device.

The KV cache costs 160 KiB per token, from 10 full attention layers plus 50 sliding-window layers of
1024 tokens: 5.8 GB at 32K, 21 GB at 128K, and 41 GB at the full 256K. Weights are 62.6 GB in bf16 and
about half that in FP8. This is the roomiest of the three GPU types for this checkpoint: one 141 GB card
holds bf16 weights plus the full 256K context, so here quantization is a speed decision rather than a
memory one.

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
bash recipes/gemma-4-31B-it/h200-1/env/build.sh
```

That is the only supported build path, because the install needs uv flags a requirements file cannot
express. What it does, and why:

<!-- issue:hopper-cu129-wheel begin -->
**Hopper nodes need the cu129 wheel, not vLLM's default.** These nodes run NVIDIA driver 575
(CUDA 12.9), which cannot run vLLM's default CUDA 13 PyPI wheel. The recipe installs the cu129
release wheel from the vLLM GitHub release with `--torch-backend=cu129`.
<!-- issue:hopper-cu129-wheel end -->

Ray is installed alongside vLLM. A TP1 endpoint never uses it, but it is what the pre-restructure
environment contained, so keeping it means the rates below were measured in this exact environment.

Scratch expires after 90 days, so this environment is disposable. Rebuild it with the same command,
or `--force` to replace an existing one. `env/requirements.lock` records the exact resolution that was
tested, and `env/WHEELS` records the non-PyPI artifact URLs with hashes.

## Launch

Canonical path, submitted from the repo root:

```
sbatch --account=<your-account> recipes/gemma-4-31B-it/h200-1/serve.sbatch
```

Find the host once it starts, then use that node name with the client:

```
squeue --me                       # NODELIST column
tail -f gemma31-h200-1-<jobid>.log
```

Secondary path, for a node you already hold. Reserved nodes are removed from the scheduler, which is
why this exists:

```
bash recipes/gemma-4-31B-it/h200-1/serve_ssh.sh <node>
GPU=1 bash recipes/gemma-4-31B-it/h200-1/serve_ssh.sh <node>    # pin one device of a busy node
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
  -d '{"model":"gemma-4-31b","messages":[{"role":"user","content":"What is 2+2? Answer briefly."}],"max_tokens":400}'
```

A keyless request returning 401 is the expected, correct behavior.

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
export NODE=<the node serving it>
source recipes/gemma-4-31B-it/h200-1/client.env
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
`http://<node>:8000/v1`, the same key, and model name `gemma-4-31b`.

## Tunable inputs

Every variable this recipe honors, with its default and effect.

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/gemma-4-31B-it` | Serve a different copy of the checkpoint |
| `API_PORT` | 8000 | Listening port, and part of the SSH log file name |
| `MAX_MODEL_LEN` | 32768 | Context window; 262144 is supported and costs 41 GB of KV cache |
| `GPU_UTIL` | 0.90 | Fraction of VRAM for weights plus KV cache |
| `TP` | 1 | Tensor parallel size; one GPU holds this checkpoint at full context |
| `QUANT` | `fp8` | Weight quantization applied at load. Set `QUANT=` (empty) for bf16, which measured 34 percent slower on this GPU |
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

`QUANT` is forwarded as `${QUANT-fp8}` rather than `${QUANT:-}` by `serve_ssh.sh`, so that an unset
variable keeps the FP8 default instead of arriving on the far side as set and empty, which would
silently serve bf16 at 66 percent of the rate.

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
| TP1, FP8 weights, 32K context | 85.3 tok/s | slope(128,1152) |
| TP1, bf16 weights, 32K context | 56.3 tok/s | slope(128,1152) |

Sustained single-stream decode, greedy, measured on one H200 on 2026-07-27. The slope protocol times
the same request at 128 and at 1152 output tokens and divides the difference, which cancels prefill and
per-request overhead; a single timed generation understates decode by up to 40 percent. This is the
fastest of the three GPU types for this checkpoint, which is expected: the model is memory bandwidth
bound and this card has the most of it.

What did and did not help, all measured rather than assumed:

| Change | Effect on decode rate |
| --- | --- |
| FP8 weights | 52 percent faster, and the reason it is the default |
| FP8 KV cache | no change in rate, and unnecessary on a 141 GB card |

To re-measure:

```
bash common/tools/bench.sh --host <node> --model gemma-4-31b
```

## Parallelism and quantization

TP1 is enough: one 141 GB card holds bf16 weights plus the full 256K context, and at TP1 there is no
all-reduce at all, so the node's NVLink fabric is left for jobs that need it.

FP8 weights are the default because this model is dense and therefore memory bandwidth bound. Every one
of its 31B parameters is read for every token, so halving the bytes per weight buys most of a
proportional speedup: 85.3 tok/s against 56.3 in bf16, 52 percent faster on this GPU. The same
reasoning shows up across GPU types, where the bf16 rates track HBM bandwidth: 1.77x from RTX to H100
and 1.38x from H100 to H200, against bandwidth ratios of about 1.86x and 1.43x. The gain from FP8 is
smallest on this card for the same reason it is the fastest card, since the more bandwidth there is,
the less of the total time is spent moving weights.

This is exactly the opposite of the 26B-A4B sibling, where FP8 measured no change at all. That model
activates only 4B of 26B parameters, so it is host overhead bound and there is almost no weight traffic
for a narrower dtype to save. Same family, same flag, opposite answer, and the reason is the
architecture rather than anything about the quantization implementation.

Because VRAM is not the constraint on this card, `QUANT=` (empty) for bf16 fidelity is a defensible
choice here in a way it is not on the 96 GB or 80 GB variants: it costs 34 percent of the rate and
nothing in context. `KV_FP8=1` halves the 160 KiB per token that the KV cache costs and measured no
change in rate; it has no purpose on this card unless you push well past the model's 256K limit.

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

This is the model that would benefit most from a working drafter, since it is dense and bandwidth
bound, which is why the plumbing is kept rather than deleted.

<!-- issue:deepgemm-h200-crash begin -->
**Leave `VLLM_USE_DEEP_GEMM` at 0 on H200.** The DeepGEMM MoE path takes an illegal memory access on
GLM-5.2's sparse attention, and forcing `VLLM_USE_DEEP_GEMM=1` on H200 independently reproduced the
same crash for Qwen3-Coder-480B-FP8. It is load-bearing for more than one model on this hardware, so
do not flip it without re-testing the model you are serving.
<!-- issue:deepgemm-h200-crash end -->

This checkpoint is dense, so the DeepGEMM MoE path is not exercised and the flag is inert here.
`env/env.sh` still pins it to 0, which is how the rates above were measured, and flipping it is both
untested and pointless for this model.

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
`ssh <node> nvidia-smi --query-gpu=memory.used --format=csv,noheader`, which should read 0 MiB on the
device this endpoint used before you relaunch, or the next start will fail on memory. On a shared node
check only that device: another job may legitimately be holding the other three.

## Expected startup time

| Stage | Cold | Warm |
| --- | --- | --- |
| Environment build, one time | 5 to 15 min | skipped |
| Weight load | to be measured | to be measured |
| Total to first token | to be measured | to be measured |

First launch on a fresh node is slower than later ones: page cache is cold, any just-in-time kernel
compilation happens once, and FP8 quantization happens during weight load. A launch that looks hung
during this window is usually still loading. Check the log before killing it. These numbers will be
filled in when this recipe is validated on hardware; they are deliberately blank rather than guessed.
