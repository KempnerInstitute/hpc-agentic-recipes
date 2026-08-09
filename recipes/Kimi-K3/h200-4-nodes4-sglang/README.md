# Kimi-K3 on four H200 nodes, SGLang

Status: Validated - SGLang 0.5.16, protocol: slope(128,1152) swept to each configuration's concurrency cap

| | |
| --- | --- |
| Served name | `kimi-k3` |
| Checkpoint | `Kimi-K3`, Hugging Face `moonshotai/Kimi-K3` |
| On disk | 1453.8 GiB across 96 shards, MXFP4 quantization-aware trained |
| Served precision | MXFP4 through Marlin W4A16, 102.75 GB of weights per GPU |
| Context | 383216. The checkpoint declares 1048576, which the engine will not accept; see Known limits |
| Hardware | 4 H200 nodes, 16 GPUs, 143771 MiB each, TP16 x EP16, no pipeline parallelism |
| Engine | SGLang 0.5.16 in `lmsysorg/sglang:kimi-k3-cu12`, a container rather than a virtual environment |

## 1. Create the API key

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/Kimi-K3-h200-4-nodes4-sglang.key
chmod 600 secrets/Kimi-K3-h200-4-nodes4-sglang.key
```

- `secrets/vllm_api_key` is read when this file is absent.
- To rotate, replace the file and relaunch; the engine reads it once at launch.

## 2. Build the environment

There is no virtual environment. The engine is a container, already staged beside the weights.

```
bash recipes/Kimi-K3/h200-4-nodes4-sglang/env/build.sh
```

- That verifies the staged image and reports its SGLang version. It pulls only if the image is missing,
  which took 2 hours 6 minutes and resolves a moving upstream tag, so a rebuilt image is not guaranteed to
  match the one these numbers came from. Prefer the staged copy.
- Upstream SGLang 0.5.16 carries no `kimi_k3` model module, so a stock install cannot serve this
  checkpoint. The image is a K3-specific build reporting the same version.

## 3. Launch

Slurm, from the repo root:

```
sbatch --account=<your-account> recipes/Kimi-K3/h200-4-nodes4-sglang/serve.sbatch
squeue --me                                  # NODELIST, the FIRST node serves
tail -f kimi-k3-<jobid>.log
```

Direct, on four nodes you already hold:

```
bash recipes/Kimi-K3/h200-4-nodes4-sglang/serve_ssh.sh <node0> <node1> <node2> <node3>
```

Four serving configurations, selected by environment variables on either launch line:

| Configuration | Set | Use it for |
| --- | --- | --- |
| default | nothing | long prompts, and a few callers at long context |
| speculative | `SPEC_MODE=dspark` | short prompts, one caller |
| wide pool | `WIDE=1` | a shared endpoint under load |
| both | `SPEC_MODE=dspark WIDE=1` | the fastest single stream, short prompts only |

- `K3_PARSER_PATCH=1` composes with any of them and changes no performance characteristic. Set it, or raw
  channel markers such as `<|open|>think<|sep|` appear in the visible reply.
- `sbatch` passes the submitting environment to the job, so a variable set on that line reaches every rank.

| Stage | Measured |
| --- | --- |
| Weight load, 96 shards across 16 ranks | about 12 min |
| Launcher to serving | 8 to 9 min here, 13 min 54 s to 15 min 35 s across four earlier launches |

- Startup is dominated by reading 1.5 TiB of weights, so a launch that appears hung is almost always still
  loading. Check the rank 0 log before killing it.
- The rank logs are appended across launches, not truncated, so date-filter before reading anything from
  them.

## 4. Verify

```
KEY=$(cat secrets/Kimi-K3-h200-4-nodes4-sglang.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the head node>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"kimi-k3","messages":[{"role":"user","content":"What is 2+2? Answer briefly."}],"max_tokens":800}'
```

| Command | Expected |
| --- | --- |
| `/v1/models` with the key | `"id": "kimi-k3"`, `"max_model_len": 383216` |
| `/v1/models` without a key | `401` |
| chat completion | `content` holds the answer, `finish_reason: stop`, about 30 output tokens |

- This model always emits reasoning before its answer, returned in a separate field. Use at least 800
  output tokens.
- With `K3_PARSER_PATCH=1` the visible `content` carries no channel markers. Without it, it does.

## 5. Connect a client

```
export NODE=<the head node>
source recipes/Kimi-K3/h200-4-nodes4-sglang/client.env
claude
```

- `client.env` sets `CLAUDE_CODE_MAX_CONTEXT_TOKENS=383216`. Claude Code assumes and enforces a 200k window
  for a served name it does not recognize, well below what this endpoint serves.
- Use `ANTHROPIC_AUTH_TOKEN`. `ANTHROPIC_API_KEY` sends `x-api-key` and returns 401.
- `client.env` also sets `ANTHROPIC_SMALL_FAST_MODEL`; without it the client reaches for a hosted Haiku.
- OpenAI clients such as Codex: base URL `http://<head-node>:8000/v1`, same key, model `kimi-k3`. See
  [docs/clients.md](../../../docs/clients.md).

## 6. Stop it

```
scancel <jobid>                                              # Slurm path, owns all sixteen ranks
bash common/tools/stop.sh <node0> <node1> <node2> <node3>    # direct path, name every node
```

- SGLang renames its workers to `sglang::scheduler_TP0_EP0` and similar, so a `pgrep` for the launcher
  finds nothing. Check `nvidia-smi` for GPU memory instead.

## Tunable inputs

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/Kimi-K3` | Serve a different copy of the checkpoint |
| `DRAFT` | `$MODELS_DIR/Kimi-K3-DSpark` | The speculative draft, used only when `SPEC_MODE=dspark` |
| `SIF` | shared repository copy | Use a container staged elsewhere |
| `SPEC_MODE` | `none` | Set to `dspark` for speculative decoding |
| `API_PORT` | 8000 | Listening port |
| `DIST_PORT` | 29500 | Port the 16 ranks use to form their group |
| `MAX_MODEL_LEN` | 383216 | Context window, the largest the engine will accept |
| `MEM_FRACTION` | 0.88, 0.90 under `WIDE=1` | Static memory fraction; it feeds the KDA state pool, so lowering it cuts the concurrency cap |
| `WIDE` | 0 | Set to 1 for the configuration that lifts the request cap from 67 to 156 |
| `MAMBA_RATIO` | unset, 3.2 under `WIDE=1` | Size of the KDA state pool relative to KV |
| `MAMBA_CACHE_STRATEGY` | unset, `extra_buffer_lazy` under `WIDE=1` | Cuts the state slots per request from 5 to 4 |
| `K3_PARSER_PATCH` | 0 | Set to 1 to keep channel markers out of the visible text, using the files in `patches/` |
| `K3_LOG_DIR` | `/tmp/$USER/k3` | Node-local logs and JIT caches |
| `KEY_NAME`, `KEY_FILE`, `VLLM_API_KEY` | this recipe's key | Which key `common/lib/api_key.sh` resolves |
| `NODE`, `K3_HEAD_NODE` | unset | The head node for the client; pass nodes as arguments to the launcher |
| `ACCOUNT` | unset | Read by `common/defaults.sh`; pass `--account` at submit time |
| `MODELS_DIR`, `ENV_ROOT` | `common/defaults.sh` | Override there or in `common/site.conf` |

## Benchmarking

Conditions:

| | |
| --- | --- |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| Input length | ISL 130 tokens for the sweeps; prompt-length sensitivity is measured separately below |
| Output length | OSL 1152 tokens, output only |
| Context | `MAX_MODEL_LEN=383216` for the default arm below, 262144 for the other three |
| Allocation for the measurement | 4 nodes, 4 GPUs each, 64 cores and 1440 GB per node, `kempner_eng` |
| Sequence cap | each configuration is swept only to the concurrency its own engine admits |
| Endpoint | idle, and the benchmark client ran on a separate CPU-only node |
| Power | 700 W enforced, the card default, so not capped. Median 346 to 361 W across the four nodes and a 462 W peak, with nothing at or above 690 |

**Default.** Cap 67 requests, token pool 383,223.

| Concurrency | Aggregate | Per stream | Spread over 3 runs | TTFT median |
| --- | --- | --- | --- | --- |
| 1 | 40.3 tok/s | 40.3 tok/s | 40.2 to 40.3 | 209 ms |
| 8 | 245.2 tok/s | 30.7 tok/s | 245.0 to 245.3 | 209 ms |
| 16 | 421.9 tok/s | 26.4 tok/s | 421.7 to 423.9 | 215 ms |
| 32 | 709.2 tok/s | 22.2 tok/s | 708.5 to 710.1 | 370 ms |
| 48 | 955.1 tok/s | 19.9 tok/s | 952.4 to 957.9 | 426 ms |
| 64 | 1067.1 tok/s | 16.7 tok/s | 1065.1 to 1070.5 | 2322 ms |

**`SPEC_MODE=dspark`.** Cap 23 requests, token pool 302,711. The draft needs its own state slots from the
same pool, so speculation costs concurrency.

| Concurrency | Aggregate | Per stream |
| --- | --- | --- |
| 1 | 84.8 tok/s | 84.8 tok/s |
| 4 | 262.6 tok/s | 65.6 tok/s |
| 8 | 378.9 tok/s | 47.4 tok/s |
| 16 | 511.5 tok/s | 32.0 tok/s |
| 20 | 538.7 tok/s | 26.9 tok/s |

**`WIDE=1`.** Cap 156 requests, token pool 198,936. Highest aggregate throughput available.

| Concurrency | Aggregate | Per stream |
| --- | --- | --- |
| 1 | 40.3 tok/s | 40.3 tok/s |
| 8 | 245.4 tok/s | 30.7 tok/s |
| 32 | 707.6 tok/s | 22.1 tok/s |
| 64 | 1064.2 tok/s | 16.6 tok/s |
| 96 | 1389.1 tok/s | 14.5 tok/s |
| 128 | 1398.6 tok/s | 10.9 tok/s |
| 156 | 1442.6 tok/s | 9.2 tok/s |

**`SPEC_MODE=dspark WIDE=1`.** Cap 48 requests, token pool 159,445. Highest single stream rate.

| Concurrency | Aggregate | Per stream |
| --- | --- | --- |
| 1 | 94.1 tok/s | 94.1 tok/s |
| 8 | 378.7 tok/s | 47.3 tok/s |
| 16 | 506.5 tok/s | 31.7 tok/s |
| 32 | 715.9 tok/s | 22.4 tok/s |
| 48 | 928.4 tok/s | 19.3 tok/s |

Decode rate against prompt length, at concurrency 1:

| Configuration | ISL 1012 | ISL 28247 | ISL 115292 | Fall |
| --- | --- | --- | --- | --- |
| default | 40.2 tok/s | 39.9 tok/s | 38.9 tok/s | 3.2 percent |
| `WIDE=1` | 40.4 tok/s | 39.9 tok/s | 39.0 tok/s | 3.5 percent |
| `SPEC_MODE=dspark` | 86.1 tok/s | 67.4 tok/s | 36.1 tok/s | 58.1 percent |
| `SPEC_MODE=dspark WIDE=1` | 82.4 tok/s | 64.4 tok/s | 37.3 tok/s | 54.7 percent |

| | |
| --- | --- |
| Label | capped. Each arm is swept only to the concurrency its engine admits, so these are ceilings set by the request cap rather than by throughput turning over |
| Quote for one caller | 40.3 tok/s on the default, or 94.1 with `SPEC_MODE=dspark WIDE=1` at short prompts |
| Quote for a shared endpoint | 1442.6 tok/s at concurrency 156 under `WIDE=1` |
| KV cache | 383,223 tokens on the default arm, unchanged whether the context is set to 262144, 383216 or 1048576 |
| Speculation and long context | both speculative arms are slower than the default by an ISL of 115292. 69 of the 93 layers use Kimi Delta Attention and hold a fixed-size state whatever the prompt length, while the draft's verification cost grows with context |
| Long prompt | 345,653 tokens in 61 s cold; 0.8 s when the prefix is already cached |

Reproduce:

```
KEY_NAME=Kimi-K3-h200-4-nodes4-sglang bash common/tools/bench.sh --host <head_node> --model kimi-k3 \
  --sweep 1,8,16,32,48,64
```

- `KEY_NAME` is required once `secrets/` holds more than one key, otherwise `bench.sh` resolves the shared
  key and every request returns 401.
- Sweep each configuration only to its own cap. A rate measured above the cap includes queueing rather than
  throughput.

## Known limits

- The checkpoint declares 1048576 but the engine accepts 383,217, which is the token pool. SGLang starts
  happily at a larger setting and then rejects the request: `MAX_MODEL_LEN=1048576` advertises a million
  tokens through `/v1/models` while returning HTTP 400 above 383,217. Do not set it above the pool.
- Raising the context costs nothing. The pool and the request cap are identical at 262144, 383216 and
  1048576, so the only thing that changes is what a single request may claim.
- Keep `--ep-size 16`. Under pure TP16 the MoE intermediate dimension does not divide into the 128 Marlin
  needs, `w2` pads, weights reach 131.62 GiB per GPU, and the KDA state cache cannot be allocated at all.
- Keep `--moe-runner-backend marlin`. MXFP4 is Blackwell-native and these are Hopper cards; the `auto`
  backend dequantizes every expert to bf16 and runs out of memory.
- Down InfiniBand ports stall initialization with no error, so `serve.sh` detects the active HCAs rather
  than hardcoding them.
- Multimodal is left off. The vision tower is opt-in, and enabling it risks the cross-node profiling stall
  this model family caused elsewhere, with nothing gained for coding.
- Anthropic's hosted tools return HTTP 200 with the tool silently dropped on this engine, rather than an
  error. Use [docs/web-search.md](../../../docs/web-search.md).
