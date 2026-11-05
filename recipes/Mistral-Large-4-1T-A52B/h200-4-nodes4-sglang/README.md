# Mistral Large 4 1T-A52B FP8 on four H200 nodes, under SGLang

Status: Validated - SGLang in a container, protocol: slope(128,1152) at concurrency 1, 8, 32 and 48

| | |
| --- | --- |
| Served name | `mistral-large-4` |
| Checkpoint | `Mistral-Large-4-1T-A52B`, Hugging Face `mistralai/Mistral-Large-4-1T-A52B` |
| On disk | 988 GiB across 54 shards, FP8 block-scaled, `MistralLarge4ForConditionalGeneration` |
| Served precision | FP8 with `weight_block_size` [128, 128], BF16 KV cache on Hopper |
| Context | 262144. The checkpoint declares 1048576; upstream validated to 256k |
| Hardware | 4 H200 nodes, 16 GPUs, 143771 MiB each, TP16 across all four |
| Engine | SGLang in Singularity, DSpark speculative decoding from the checkpoint's own draft |
| Allocation for the measurement | 4 nodes, 4 GPUs each, 64 cores and 1440 GB per node, `kempner_h200` |

Mistral Large 4 is a 1T-parameter mixture-of-experts with 52B active per token, 72 layers, and a
DeepSeek-style sparse attention stack: MLA with a DSA top-k indexer, NoPE on one layer in four, SSMax
scaling and cross-layer KV sharing. It carries a vision encoder; this recipe serves the text path.

**This recipe runs SGLang, not vLLM, and that is not a preference.** See Known limits.

## 1. Create the API key

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/Mistral-Large-4-1T-A52B-h200-4-nodes4.key
chmod 600 secrets/Mistral-Large-4-1T-A52B-h200-4-nodes4.key
```

- `secrets/vllm_api_key` is read when this file is absent.
- Requests without the key receive HTTP 401.
- The key reaches SGLang through `common/tools/sglang_launch.py`, which reads it from the environment and
  appends `--api-key` inside the process, keeping it out of `/proc/<pid>/cmdline`.

## 2. Build the environment

A prebuilt SGLang image carrying Mistral Large 4 support and mistral-common 1.13.0 or later, which is
required to read the checkpoint's `tekken.json`. There is no venv to build.

```
bash recipes/Mistral-Large-4-1T-A52B/h200-4-nodes4-sglang/env/build.sh
```

That verifies the staged image and reports its SGLang version, failing if `sglang` does not import
inside it. It does not pull: the image is prebuilt and cannot be reconstructed from a version number.

`env/env.sh` resolves it in order: an exported `SIF`, then
`$MODELS_DIR/Mistral-Large-4-1T-A52B/container/sglang-mistral-large-4.sif`, then the NVFP4 sibling's
copy of the same image, then the shared testbed. One image serves both checkpoints, so a second copy is
not needed.

A copy of this checkpoint needs the `container/` subdirectory alongside the weights, 12.5 GiB on top of
the 988 GiB.

## 3. Launch

```
sbatch --account=<your-account> recipes/Mistral-Large-4-1T-A52B/h200-4-nodes4-sglang/serve.sbatch
```

On nodes you already hold:

```
bash recipes/Mistral-Large-4-1T-A52B/h200-4-nodes4-sglang/serve_ssh.sh <node0> <node1> <node2> <node3>
```

Startup is about 7 minutes once the page cache is warm, dominated by reading 988 GiB. The per-rank logs
are `$ML4_LOG_DIR/ml4-rank<N>.log`, default under `/tmp/$USER/mistral-large-4`.

## 4. Verify

```
KEY=$(cat secrets/Mistral-Large-4-1T-A52B-h200-4-nodes4.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the first node>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"mistral-large-4","messages":[{"role":"user","content":"What is 2+2? Answer with just the number."}],"max_tokens":2048,"temperature":0.6}'
```

| Command | Expected |
| --- | --- |
| `/v1/models` with the key | `"id": "mistral-large-4"` |
| `/v1/models` without a key | `401` |
| chat completion | `content` holds the answer, `finish_reason: stop` |

Reasoning comes back in a separate field and the answer in `content`. Without the reasoning parser the
thinking stays inline as `[THINK]...[/THINK]`, and without the tool-call parser tool calls arrive as
`[TOOL_CALLS]...` text; this recipe passes both.

## 5. Connect a client

```
export NODE=<the first node>
source recipes/Mistral-Large-4-1T-A52B/h200-4-nodes4-sglang/client.env
claude
```

`client.env` exports the base URL, the key, the served model name, and
`CLAUDE_CODE_MAX_CONTEXT_TOKENS=262144`, without which Claude Code assumes and enforces a 200k window.
Use `ANTHROPIC_AUTH_TOKEN`, not `ANTHROPIC_API_KEY`.

## 6. Stop it

```
scancel <job id>
```

For an ssh launch, kill `sglang_launch.py` on all four nodes. Confirm the GPUs are released before
relaunching.

## Tunable inputs

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/Mistral-Large-4-1T-A52B` | Serve a different copy of the checkpoint |
| `SIF` | resolved as in step 2 | Container image |
| `API_PORT` | 8000 | Listening port |
| `DIST_PORT` | 29500 | Rank rendezvous port |
| `MAX_MODEL_LEN` | 262144 | Context window, fixed at init |
| `MEM_FRACTION` | 0.85 | Static VRAM fraction for weights plus KV |
| `TP` | 16 | Tensor parallel size, the total GPU count |
| `NNODES` | 4 | Node count |
| `MAX_RUNNING` | 48 | Concurrent request cap, which also caps the sweep |
| `NO_SPEC` | 0 | Set to 1 to serve without DSpark drafting |
| `EXTRA_ARGS` | unset | Extra flags appended to the launch command |
| `KEY_NAME`, `KEY_FILE`, `VLLM_API_KEY` | this recipe's key | Which key `common/lib/api_key.sh` resolves |

## Benchmarking

Conditions:

| | |
| --- | --- |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| Input length | ISL 23 tokens. Rates at a long input are not measured; use `--prompt-tokens` |
| Output length | OSL 1152 tokens, output only, `ignore_eos` |
| Context | `MAX_MODEL_LEN=262144`, `MEM_FRACTION=0.85` |
| Allocation for the measurement | 4 nodes, 4 GPUs each, 64 cores and 1440 GB per node, `kempner_h200` |
| Sequence cap | `max_running_requests` 48, which the engine sets with DSpark active |
| Endpoint | idle, and the benchmark client ran on the first node |

Results:

| Concurrency | Aggregate | Per stream | Spread over 3 runs |
| --- | --- | --- | --- |
| 1 | 67.1 tok/s | 67.1 tok/s | 67.0 to 67.1 |
| 8 | 395.4 tok/s | 49.4 tok/s | 394.4 to 396.5 |
| 32 | 1117.3 tok/s | 34.9 tok/s | 1114.9 to 1122.0 |
| 48 | 1517.5 tok/s | 31.6 tok/s | 1509.7 to 1523.6 |

| | |
| --- | --- |
| Engine facts | KV cache 690,816 tokens, 2.63 full-length requests at once; 17.17 GB free per GPU after sizing |
| Label | capped. The sweep stops at 48 because the engine admits no more, not because throughput plateaued |

## Known limits

- **vLLM cannot serve this model across nodes here.** No vLLM release or nightly wheel carries Mistral
  Large 4; the published image is the only vLLM build that has it, and that image ships without Ray:
  `ValidationError for ParallelConfig, Failed to import Ray: No module named 'ray'`. Its remaining
  backends, `mp` and `uni`, are single-node only, and a read-only SIF cannot be patched. SGLang needs no
  Ray and coordinates with `--nnodes` and `--node-rank`, so it is the only working multi-node path.
- Throughput above concurrency 48 is unmeasured, because the engine admits no more with DSpark active.
  Upstream notes `--max-running-requests 64` helps bursts of short requests and that 48 or lower suits
  long-context traffic.
- Context is served at 262144 rather than the checkpoint's 1048576. Upstream validated to 256k and caps
  its own recipe the same way.
- Decode rate at a long input is not measured. Every figure here uses a 23-token input.
- The vision encoder is loaded but not exercised. This recipe is validated for text only.
- The NVFP4 sibling serves a shorter context on one RTX node and uses vLLM rather than SGLang. See
  [Mistral-Large-4-1T-A52B-NVFP4](../../Mistral-Large-4-1T-A52B-NVFP4/README.md).
