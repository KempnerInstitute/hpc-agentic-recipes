# Kimi-K3 on four H200 nodes, SGLang

Status: Validated - SGLang 0.5.16, protocol: slope(128,1152) swept to each configuration's concurrency cap

Everything needed to build, launch, verify, connect to, and debug this endpoint is on this page.

This is the only recipe here that runs on SGLang rather than vLLM, and the only one whose engine is a
container rather than a virtual environment. Both follow from the model: no vLLM release available here
implements `KimiK3ForConditionalGeneration`, and the engine that does ships as an image.

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
| `K3_HEAD_NODE` | unset | The head node, for the client configuration |
| `MODELS_DIR` | shared testbed path | Point at your own faster copy of the checkpoint |
| `SIF` | testbed copy, or `$MODELS_DIR` if it has one | Use a container image staged elsewhere |

## Status

Validated. The environment is the staged container, and all four configurations this recipe exposes were
launched with `serve_ssh.sh` on four H200 nodes, 16 GPUs at TP16 with expert parallelism, then measured
with `common/tools/bench.sh`. Each was swept only up to the concurrency its own engine admits, and each
was also measured at three prompt lengths. Every endpoint was still answering after its sweep finished.

Time to serving was 13 to 15 minutes in each of the four launches.

The engine is SGLang 0.5.16 as shipped in `lmsysorg/sglang:kimi-k3-cu12`. That is worth stating
precisely: the upstream SGLang 0.5.16 release carries no `kimi_k3` model module and no K3 registry
entry, so a stock 0.5.16 install cannot serve this checkpoint. The image is a K3-specific build that
reports the same version.

## What this is

Kimi-K3, a 2.8T-parameter mixture of experts with 104B activated per token, quantization-aware trained
in MXFP4 from the SFT stage onward. It is natively multimodal and always emits reasoning before its
answer. SGLang serves it as `kimi-k3` on both an Anthropic-compatible and an OpenAI-compatible API.

- Checkpoint directory: `Kimi-K3`
- Hugging Face repo: `moonshotai/Kimi-K3`
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Kimi-K3`
- On disk: 1453.8 GiB, 1561.0 GB decimal, in 96 shards
- Container: `Kimi-K3/container/sglang-kimi-k3-cu12.sif`, 16 GB, staged beside the weights

Read from `config.json` and the model card:

| Property | Value |
| --- | --- |
| Architecture | `KimiK3ForConditionalGeneration`, `model_type` `kimi_k3` |
| Text tower | `KimiLinearForCausalLM`, `model_type` `kimi_linear` |
| Parameters | 2.8T total, 104B activated |
| Layers | 93, of which 1 is dense |
| Attention | Kimi Delta Attention on 69 layers, gated MLA on 24 |
| Attention dimensions | hidden 7168, 96 heads, `kv_lora_rank` 512, `q_lora_rank` 1536, `v_head_dim` 128 |
| Experts | 896, 16 selected per token, 2 shared, `moe_intermediate_size` 3072 |
| Quantization | `compressed-tensors`, `mxfp4-pack-quantized`, `group_size` 32, `num_bits` 4, symmetric |
| Context | `max_position_embeddings` 1048576 |

There is no bf16 twin to fall back on: the MXFP4 weights are what the model was trained with.

The testbed path works out of the box. Copying the checkpoint into your own scratch space loads faster,
because scratch outperforms Lustre for this workload, and the directory names are identical in both
locations so only `MODELS_DIR` changes. Scratch has a 90-day retention policy, so treat it as a fast
cache and keep testbed as the permanent copy.

## Hardware

| Requirement | Value |
| --- | --- |
| GPU | H200, 4 per node, 143771 MiB each, sm_90 |
| Nodes | 4, so 16 GPUs and about 2246 GiB of VRAM |
| Parallelism | TP16 with expert parallelism across all 16 ranks, no pipeline parallelism |
| Partition | `kempner_h200` |
| Per-GPU allocation limit | 16 CPUs, about 378 GiB host memory |
| Maximum wall time | 2 days |

All H200 nodes on this cluster share one hardware specification, so any four nodes in the partition
work. Sixteen GPUs is the per-user cap, so this recipe uses your entire allowance.

Weights measured 102.75 GB per GPU in use. What is left funds the KV and KDA state pools, and how it is
split between them is what the configurations below trade off.

## Environment build

There is no virtual environment to build. The engine is a container, already staged beside the weights:

```
bash recipes/Kimi-K3/h200-4-nodes4-sglang/env/build.sh
```

That verifies the staged image and reports its SGLang version. It pulls only if the image is missing,
which took 2 hours 6 minutes when it was done here and resolves a moving upstream tag, so a rebuilt
image is not guaranteed to match the one these numbers came from. Prefer the staged copy.

## Launch

Slurm path, submitted from the repo root:

```
sbatch --account=<your-account> recipes/Kimi-K3/h200-4-nodes4-sglang/serve.sbatch
```

Find the head node once it starts, then use that name with the client:

```
squeue --me                       # NODELIST column, the endpoint runs on the first node
tail -f kimi-k3-<jobid>.log
```

Direct path, for four nodes you already hold. Use the Slurm submission above unless you already have
the nodes, or you are deploying an endpoint on behalf of others:

```
bash recipes/Kimi-K3/h200-4-nodes4-sglang/serve_ssh.sh <node0> <node1> <node2> <node3>
```

Both commands above launch the default configuration. To select one of the others, set the variables in
the environment of the launch. The four configurations, and the two paths for each:

```
# default
sbatch --account=<acct> recipes/Kimi-K3/h200-4-nodes4-sglang/serve.sbatch
bash recipes/Kimi-K3/h200-4-nodes4-sglang/serve_ssh.sh <node0> <node1> <node2> <node3>

# speculative decoding, fastest for one caller
SPEC_MODE=dspark sbatch --account=<acct> recipes/Kimi-K3/h200-4-nodes4-sglang/serve.sbatch
SPEC_MODE=dspark bash recipes/Kimi-K3/h200-4-nodes4-sglang/serve_ssh.sh <node0> <node1> <node2> <node3>

# wide pool, highest total throughput
WIDE=1 sbatch --account=<acct> recipes/Kimi-K3/h200-4-nodes4-sglang/serve.sbatch
WIDE=1 bash recipes/Kimi-K3/h200-4-nodes4-sglang/serve_ssh.sh <node0> <node1> <node2> <node3>

# both, fastest single stream
SPEC_MODE=dspark WIDE=1 sbatch --account=<acct> recipes/Kimi-K3/h200-4-nodes4-sglang/serve.sbatch
SPEC_MODE=dspark WIDE=1 bash recipes/Kimi-K3/h200-4-nodes4-sglang/serve_ssh.sh <node0> <node1> <node2> <node3>
```

Which one to pick is under Measured performance below. `sbatch` passes the submitting environment to the
job by default, so a variable set on that line reaches every rank.

Submit from the repo root either way. Slurm stages the batch script into its own spool directory, so
the script cannot locate the repo from its own path and resolves paths against the submit directory
instead.

Every rank dials the head node's InfiniBand address to form the 16-rank group. Both launchers read that
address from the node rather than assuming a naming scheme.

## Verify

```
KEY=$(cat secrets/vllm_api_key)
NODE=<the head node>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models          # must print 401

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"kimi-k3","messages":[{"role":"user","content":"What is 17*23? Answer briefly."}],"max_tokens":800}'
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

The shared warning above names vLLM's field. This endpoint is SGLang, where the field is
`reasoning_content` instead. Everything else about it applies unchanged.

## Connect a client

```
export NODE=<the head node>
source recipes/Kimi-K3/h200-4-nodes4-sglang/client.env
```

SGLang serves an Anthropic-compatible `/v1/messages` alongside the OpenAI `/v1`, so Claude Code connects
with no proxy. `client.env` sets `CLAUDE_CODE_ATTRIBUTION_HEADER=0`, without which every turn of a
conversation re-prefills the whole history, and `--tool-call-parser kimi_k3` is what makes tool calls
arrive as calls rather than as text. For an OpenAI-compatible client instead, use base URL
`http://<node>:8000/v1`, the same key, and model name `kimi-k3`. See
[clients.md](../../../docs/clients.md).

Multi-turn use requires passing the complete previous assistant message back, `reasoning_content` and
`tool_calls` included, because the model was trained in preserved-thinking-history mode. Thinking
effort is set with a top-level `reasoning_effort` field taking `low`, `high` or `max`, defaulting to
`max`.

## Tunable inputs

Every variable this recipe honors, with its default and effect.

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/Kimi-K3` | Serve a different copy of the checkpoint |
| `DRAFT` | `$MODELS_DIR/Kimi-K3-DSpark` | The speculative draft, used only when `SPEC_MODE=dspark` |
| `SIF` | testbed copy, or `$MODELS_DIR/Kimi-K3/container/…` if present | Use a container staged elsewhere |
| `SPEC_MODE` | `none` | Set to `dspark` for speculative decoding |
| `API_PORT` | 8000 | Listening port |
| `DIST_PORT` | 29500 | Port the 16 ranks use to form their group |
| `MAX_MODEL_LEN` | 262144 | Context window; the checkpoint supports 1048576 |
| `MEM_FRACTION` | 0.88, 0.90 under `WIDE=1` | Static memory fraction; it feeds the KDA state pool, so lowering it cuts the concurrency cap |
| `WIDE` | 0 | Set to 1 for the configuration that lifted the concurrency cap from 67 to 156 |
| `MAMBA_RATIO` | unset, 3.2 under `WIDE=1` | Size of the KDA state pool relative to KV |
| `MAMBA_CACHE_STRATEGY` | unset, `extra_buffer_lazy` under `WIDE=1` | Cuts the state slots per request from 5 to 4 |
| `K3_LOG_DIR` | `/tmp/$USER/k3` | Node-local logs and JIT caches |

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

This applies here: Claude Code does reach this endpoint, and its built-in web search is one of the
hosted tools the engine rejects. Use the search tool above to give the model web access.

## Measured performance

Four configurations, each swept only up to the concurrency its own engine admits. A rate measured above
that cap includes queueing rather than throughput, so every table below states the cap first.

**Default.** Cap 67 requests, token pool 383,223.

| Concurrency | Aggregate | Per stream | Latency |
| --- | --- | --- | --- |
| 1 | 40.2 tok/s | 40.2 tok/s | TTFT median 213 ms, n=3 spanning 40.2 to 40.3 |
| 8 | 245.7 tok/s | 30.7 tok/s | TTFT median 210 ms, p90 214 ms, n=3 spanning 244.1 to 246.0 |
| 16 | 423.2 tok/s | 26.5 tok/s | TTFT median 212 ms, p90 221 ms, n=3 spanning 422.2 to 423.7 |
| 32 | 710.1 tok/s | 22.2 tok/s | TTFT median 383 ms, p90 1905 ms, n=3 spanning 708.6 to 711.1 |
| 48 | 957.8 tok/s | 20.0 tok/s | TTFT median 531 ms, p90 2071 ms, n=3 spanning 951.4 to 963.2 |
| 64 | 1069.2 tok/s | 16.7 tok/s | TTFT median 2300 ms, p90 2372 ms, n=3 spanning 1068.5 to 1074.6 |

**`SPEC_MODE=dspark`.** Cap 23 requests, token pool 302,711. The draft model needs its own state slots
from the same pool, so speculation costs concurrency.

| Concurrency | Aggregate | Per stream | Latency |
| --- | --- | --- | --- |
| 1 | 84.8 tok/s | 84.8 tok/s | TTFT median 380 ms, n=3 spanning 84.8 to 84.9 |
| 4 | 262.6 tok/s | 65.6 tok/s | TTFT median 281 ms, p90 421 ms, n=3 spanning 256.0 to 264.9 |
| 8 | 378.9 tok/s | 47.4 tok/s | TTFT median 245 ms, p90 394 ms, n=3 spanning 374.8 to 382.3 |
| 16 | 511.5 tok/s | 32.0 tok/s | TTFT median 321 ms, p90 1875 ms, n=3 spanning 505.0 to 515.9 |
| 20 | 538.7 tok/s | 26.9 tok/s | TTFT median 1359 ms, p90 1774 ms, n=3 spanning 531.0 to 551.5 |

**`WIDE=1`.** Cap 156 requests, token pool 198,936. This is the highest aggregate throughput available.

| Concurrency | Aggregate | Per stream | Latency |
| --- | --- | --- | --- |
| 1 | 40.3 tok/s | 40.3 tok/s | TTFT median 206 ms, n=3 spanning 40.3 to 40.4 |
| 8 | 245.4 tok/s | 30.7 tok/s | TTFT median 216 ms, p90 312 ms, n=3 spanning 245.0 to 245.6 |
| 32 | 707.6 tok/s | 22.1 tok/s | TTFT median 377 ms, p90 386 ms, n=3 spanning 706.9 to 709.4 |
| 64 | 1064.2 tok/s | 16.6 tok/s | TTFT median 388 ms, p90 1959 ms, n=3 spanning 1054.6 to 1069.6 |
| 96 | 1389.1 tok/s | 14.5 tok/s | TTFT median 402 ms, p90 2349 ms, n=3 spanning 1388.4 to 1392.6 |
| 128 | 1398.6 tok/s | 10.9 tok/s | TTFT median 612 ms, p90 2512 ms, n=3 spanning 1397.6 to 1403.9 |
| 156 | 1442.6 tok/s | 9.2 tok/s | TTFT median 1971 ms, p90 2407 ms, n=3 spanning 1442.4 to 1444.2 |

**`SPEC_MODE=dspark WIDE=1`.** Cap 48 requests, token pool 159,445. This is the highest single stream
rate available.

| Concurrency | Aggregate | Per stream | Latency |
| --- | --- | --- | --- |
| 1 | 94.1 tok/s | 94.1 tok/s | TTFT median 222 ms, n=3 spanning 94.1 to 94.2 |
| 8 | 378.7 tok/s | 47.3 tok/s | TTFT median 320 ms, p90 394 ms, n=3 spanning 374.8 to 381.4 |
| 16 | 506.5 tok/s | 31.7 tok/s | TTFT median 323 ms, p90 455 ms, n=3 spanning 496.2 to 517.0 |
| 32 | 715.9 tok/s | 22.4 tok/s | TTFT median 404 ms, p90 2292 ms, n=3 spanning 715.2 to 730.8 |
| 48 | 928.4 tok/s | 19.3 tok/s | TTFT median 1770 ms, p90 2554 ms, n=3 spanning 732.4 to 936.6 |

The c=48 spread, 732.4 to 936.6 across three repeats, is far wider than any non-speculative level. Draft
acceptance varies with the text being generated, and at the cap that variance is not absorbed by spare
capacity.

| Parameter | Value |
| --- | --- |
| ISL, input tokens | 130 |
| OSL, output tokens | 1152, as the slope between 128 and 1152 |
| Counted | output tokens only, never input plus output |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| Context served | 262144, the recipe default, in all four arms |
| Hardware | four H200 nodes, 16 GPUs |

### Prompt length

Decode rate at concurrency 1 against the prompt length the server counted. Raising the context ceiling
changes neither the token pool nor the request cap, so all four arms ran at 262144 and these figures come
from the same launches as the sweeps above.

| Configuration | ISL 1012 | ISL 28247 | ISL 115292 | Fall |
| --- | --- | --- | --- | --- |
| default | 40.2 tok/s | 39.9 tok/s | 38.9 tok/s | 3.2 percent |
| `WIDE=1` | 40.4 tok/s | 39.9 tok/s | 39.0 tok/s | 3.5 percent |
| `SPEC_MODE=dspark` | 86.1 tok/s | 67.4 tok/s | 36.1 tok/s | 58.1 percent |
| `SPEC_MODE=dspark WIDE=1` | 82.4 tok/s | 64.4 tok/s | 37.3 tok/s | 54.7 percent |

**Speculation does not survive long context.** The two non-speculative configurations are almost flat,
because 69 of the 93 layers use Kimi Delta Attention and hold a fixed-size recurrent state whatever the
prompt length, while only 24 hold a growing KV cache. The draft model has no such shortcut, so its
verification cost grows with context: by an ISL of 115292 both speculative arms are slower than the
default. Their repeat spans are also wide at every length, 67.1 to 96.3 at ISL 1012 for instance, against
40.2 to 40.3 for the default.

Time to first token grows with prompt length in every arm, from about 200 ms to about 260 ms, which is
prefill doing more work. The slope method cancels prefill, so it does not enter the decode figures.

### Which configuration to use

| Situation | Configuration | What you get |
| --- | --- | --- |
| One person, short prompts | `SPEC_MODE=dspark WIDE=1` | 94.1 tok/s, 2.3x the default |
| Long prompts | default, or `WIDE=1` | about 39 tok/s and nearly flat to an ISL of 115292 |
| Shared endpoint under load | `WIDE=1` | 1442.6 tok/s at concurrency 156 |
| A few users, long context | default | the largest token pool of the four, 383,223 |

`WIDE=1` costs nothing at concurrency 1, 40.3 against the default's 40.2, so it is the better base for
anything that might serve more than one caller with ordinary prompt lengths.

For a handful of callers at long context the request cap is not what runs out first, so the two rows
above point in opposite directions. The token pool is shared by every resident sequence, and the four
configurations trade pool for cap: default 383,223 tokens against 67 requests, `WIDE=1` 198,936 against
156. Dividing the pool by the context each caller actually holds gives roughly three 128K conversations
under the default and one under `WIDE=1`. Beyond that the scheduler still serves everyone, by evicting
and re-prefilling, which is paid for in latency rather than in errors. So the default is the better
choice for a small group of Claude Code sessions, and `WIDE=1` for many short requests. This is
arithmetic on the measured pools, not a measured five-caller run.

`WIDE=1` is one switch rather than three knobs because all three settings are needed together. The pool
has to grow (`--mamba-full-memory-ratio 3.2`), the cheaper cache strategy has to cut the state slots per
request from 5 to 4 (`--mamba-radix-cache-strategy extra_buffer_lazy`), and the static budget has to grow
from 0.88 to 0.90 to pay for both. An attempt that only forced the pool larger, with
`--max-mamba-cache-size 1280`, starved the KV pool and died after 17 minutes.

## Parallelism and quantization

TP16 with expert parallelism spans all sixteen GPUs, and there is no pipeline parallelism at all. That
is not a preference: DSpark speculative decoding requires `pp_size 1`, so any TP8 by PP2 split would
forfeit it.

`--ep-size 16` matters for memory, not just speed. Under pure TP16 each rank gets 3072/16 = 192 of the
MoE intermediate dimension, and 192 is not a multiple of the 128 that Marlin needs for a contraction
dimension, so `w2` pads to 256. Weights then measured 131.62 GiB per GPU against 97.5 expected, 94
percent of the card's 143771 MiB, and the KDA state cache could not be allocated at all. With expert parallelism each
rank holds whole experts, so `w2` keeps K=3072 and needs no padding.

MXFP4 is a Blackwell-native format and these are Hopper cards, so the experts run through Marlin W4A16
rather than a native FP4 path. That is what `--moe-runner-backend marlin` selects, and it is mandatory:
the `auto` backend dequantizes every expert to bf16 and runs out of memory.

## Gotchas

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

**Down InfiniBand ports stall initialization.** `mlx5_0` and `mlx5_1` on these nodes are dead 40 Gb QDR
ports while `mlx5_2` through `mlx5_5` are live 400 Gb NDR. Leaving a down HCA in `NCCL_IB_HCA` stalls
the job with no error, so `serve.sh` detects the active set rather than hardcoding it.

**`--trust-remote-code` is required, for the tokenizer rather than the model.** SGLang registers
`KimiK3Config` and implements the draft model itself, but `tokenizer_config.json` maps `AutoTokenizer`
to `TikTokenTokenizer` in `tokenization_kimi.py`, and the processor loader refuses custom code without
the flag. The vocabulary is the local `tiktoken.model`, so this needs no network access.

**mmap loading is pathologically slow for this checkpoint.** The first attempt ran at about 80 seconds
per shard, a two hour load, with the node 90 percent idle, 7.7 percent in iowait and loader threads in
D state. That is mmap paging 1.56 TB in small random reads over a network filesystem.
`--weight-loader-disable-mmap` brought it to about 12 minutes.

<!-- issue:node-local-logs begin -->
**Logs are written to node-local `/tmp`, not to the repo.** Every rank writes stderr for the life of
the endpoint, so a log on a network filesystem puts a blocking write on the critical path. During a
filesystem stall that write hangs, which freezes the server. `LOG_DIR` defaults to
`/tmp/$USER/vllm`, so read logs over SSH on the node that runs the server.
<!-- issue:node-local-logs end -->

**Multimodal support is left off.** The vision tower is opt-in in SGLang, and enabling it risks the
cross-node multimodal profiling stall this model family caused elsewhere, with nothing gained for
coding.

## Stop the endpoint

For a Slurm job, `scancel <jobid>` and nothing else: `srun` launches every rank inside the allocation,
so Slurm's cgroups own all sixteen and tear them down together.

For the direct SSH path, stop each node:

```
bash common/tools/stop.sh <node0> <node1> <node2> <node3>
```

## Expected startup time

| Stage | Measured |
| --- | --- |
| Container staging, one time | none, the image is staged in the testbed |
| Weight load, 96 shards across 16 ranks | about 12 min |
| Total, launch to serving | 13 min 54 s to 15 min 35 s |

The range is the four launches this recipe was measured from, in order: 13 min 54 s, 14 min 35 s,
14 min 20 s and 15 min 35 s. Startup is dominated by reading 1.5 TiB of weights, so a launch that appears
hung is almost always still loading; check the rank 0 log before killing it.
