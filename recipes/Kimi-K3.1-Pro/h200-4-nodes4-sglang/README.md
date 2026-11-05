# Kimi-K3.1-Pro on four H200 nodes, under SGLang

Status: Validated - SGLang day-zero image in a container, protocol: slope(128,1152) at concurrency 1, 8 and 32

| | |
| --- | --- |
| Served name | `kimi-k3.1-pro` |
| Checkpoint | `Kimi-K3.1-Pro`, Hugging Face `moonshotai/Kimi-K3.1-Pro` |
| On disk | 1454.0 GiB across 96 shards, `KimiK31ForConditionalGeneration` |
| Served precision | MXFP4 experts, `fp8_e4m3` KV cache; the engine refuses any other KV dtype |
| Context | 262144. The checkpoint declares 1048576 |
| Hardware | 4 nodes, 4 H200 GPUs each, 16 GPUs total, TP16 with EP16 |
| Engine | SGLang `0.0.0.dev1+g41208373c` in Singularity, `marlin` MoE runner |
| Allocation for the measurement | 4 nodes, 4 GPUs each, 64 cores and 1440 GB per node, `kempner_h200` |

Kimi-K3.1-Pro is a 1T-class multimodal MoE from Moonshot AI: 93 layers, 896 experts, hidden size
7168, shipped as `compressed-tensors` in the `mxfp4-pack-quantized` format.

**Three settings here are not tuning knobs**, and each one was a failed bring-up before it was a
default. See Known limits before changing `KV_DTYPE`, `MOE_RUNNER`, or the container path.

## 1. Create the API key

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/Kimi-K3.1-Pro-h200-4-nodes4.key
chmod 600 secrets/Kimi-K3.1-Pro-h200-4-nodes4.key
```

- `secrets/vllm_api_key` is read when this file is absent.
- Requests without the key receive HTTP 401.
- `serve.sh` refuses to start when no key resolves, rather than serving an open endpoint on four
  nodes. The key reaches SGLang through `common/tools/sglang_launch.py`, which appends `--api-key`
  inside the process so it stays out of `/proc/<pid>/cmdline`.
- SGLang logs the parsed `server_args`, the key among them, so `serve.sh` creates the log directory
  mode 700 and each rank log mode 600, and aborts if it cannot.

## 2. Build the environment

K3.1 is served from SGLang's `kimi-k3.1-day-0` branch, which no release carries, so the published
image `lmsysorg/sglang:dev-kimi-k31-pro` is the only practical engine. A venv is not an option.

```
bash recipes/Kimi-K3.1-Pro/h200-4-nodes4-sglang/env/build.sh
```

`env/env.sh` resolves the image in order: an exported `SIF`, then
`$MODELS_DIR/Kimi-K3.1-Pro/container/sglang-kimi-k31-pro.sif`, then the shared testbed copy.

**The layers must be unpacked on node-local disk.** The image carries a `.wh.rust` whiteout whose
removal walks `rust/target/release/build` recursively, and on a parallel filesystem the rapid
recursive unlink leaves `.nfs*` placeholders behind, so the pull dies with
`whiteout remove: ... unpriv.removeall: remove ...: directory not empty`. Unpacking locally and
writing only the finished SIF to shared storage takes about 15 minutes; the same pull against shared
storage failed after two hours.

A copy of this checkpoint needs the `container/` subdirectory alongside the weights, 11.8 GiB on top
of the 1454.0 GiB.

## 3. Launch

On four nodes you already hold, by ssh:

```
bash recipes/Kimi-K3.1-Pro/h200-4-nodes4-sglang/serve_ssh.sh <n0> <n1> <n2> <n3>
```

Through Slurm:

```
sbatch --account=<your-account> recipes/Kimi-K3.1-Pro/h200-4-nodes4-sglang/serve.sbatch
```

The first node is the head: it is read for its `ib0` address, which becomes `--dist-init-addr`, and
it serves the endpoint. `serve_ssh.sh` launches each rank, then confirms all four are alive rather
than reporting success unconditionally.

Startup is about 26 minutes: 808 seconds to load the weights, then graph capture. The logs are
`$K31_LOG_DIR/k31-rank<N>.log` on each node, default under `/tmp/$USER/k31`.

## 4. Verify

```
KEY=$(cat secrets/Kimi-K3.1-Pro-h200-4-nodes4.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the first node>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/completions \
  -d '{"model":"kimi-k3.1-pro","prompt":"The capital of France is","max_tokens":14,"temperature":0}'

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"kimi-k3.1-pro","messages":[{"role":"user","content":"Write a Python function that returns the n-th Fibonacci number iteratively."}],"max_tokens":700,"temperature":0.6}'
```

| Command | Expected |
| --- | --- |
| `/v1/models` with the key | `"id": "kimi-k3.1-pro"` |
| `/v1/models` without a key | `401` |
| raw completion | a coherent continuation, for example ` Paris. The Eiffel Tower is located in Paris.` |
| chat completion | a correct Fibonacci function, `finish_reason: stop` |
| the chat response | `reasoning_content` holds the reasoning, and `content` has no `<think>` or `<|` markers |

Check the reasoning split, not only that an answer came back. The sibling Kimi-K3 recipe needs a
patched reasoning parser, bound in through `K3_PARSER_PATCH=1`; this image needs no patch, and the
way to confirm that is a chat response whose `reasoning_content` is populated and whose `content` is
free of markers.

## 5. Connect a client

```
export NODE=<the first node>
source recipes/Kimi-K3.1-Pro/h200-4-nodes4-sglang/client.env
claude
```

`client.env` exports the base URL, the key, the served model name and
`CLAUDE_CODE_MAX_CONTEXT_TOKENS`, which follows a `MAX_MODEL_LEN` override. Use
`ANTHROPIC_AUTH_TOKEN`, not `ANTHROPIC_API_KEY`, which sends a header the engine ignores and yields
401 on every request.

## 6. Stop it

```
scancel <job id>
```

For an ssh launch, kill `sglang_launch.py` on each of the four nodes. Kill by that name rather than
by GPU, in case anything else on the node is not yours.

## Tunable inputs

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/Kimi-K3.1-Pro` | Serve a different copy of the checkpoint |
| `SIF` | resolved as in step 2 | Container image |
| `API_PORT` | 8000 | Listening port on the head node |
| `DIST_PORT` | 29501 | Rendezvous port for `--dist-init-addr` |
| `MAX_MODEL_LEN` | 262144 | Context window, fixed at init |
| `MEM_FRACTION` | 0.85 | Static VRAM fraction. Lower it if MegaMoE's symmetric workspace exhausts memory at startup |
| `TP` | 16 | Tensor parallel size across all four nodes |
| `EP` | 16 | Expert parallel size |
| `NNODES` | 4 | Node count |
| `KV_DTYPE` | `fp8_e4m3` | KV cache dtype. The engine rejects anything else, see Known limits |
| `MOE_RUNNER` | `marlin` | MoE runner backend. Do not change on Hopper, see Known limits |
| `ATTN` | `auto` | Attention backend. Unset passes no flag, so the engine chooses |
| `EXTRA_ARGS` | unset | Extra flags appended to the launch command |
| `KEY_NAME`, `KEY_FILE`, `VLLM_API_KEY` | this recipe's key | Which key `common/lib/api_key.sh` resolves |

## Benchmarking

Conditions:

| | |
| --- | --- |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| Input length | ISL 23 tokens. Rates at a long input are not measured; use `--prompt-tokens` |
| Output length | OSL 1152 tokens, output only, `ignore_eos` |
| Context | `MAX_MODEL_LEN=262144`, `MEM_FRACTION=0.85`, `KV_DTYPE=fp8_e4m3`, `MOE_RUNNER=marlin` |
| Allocation for the measurement | 4 nodes, 4 GPUs each, 64 cores and 1440 GB per node, `kempner_h200` |
| Sequence cap | `max_running_requests` 51 |
| Endpoint | idle, and the benchmark client ran on the head node |

Results:

| Concurrency | Aggregate | Per stream | Spread over 3 runs |
| --- | --- | --- | --- |
| 1 | 40.4 tok/s | 40.4 tok/s | 40.4 to 40.4 |
| 8 | 252.7 tok/s | 31.6 tok/s | 252.5 to 254.0 |
| 32 | 707.6 tok/s | 22.1 tok/s | 706.7 to 708.1 |

| | |
| --- | --- |
| Engine facts | KV cache 426,688 tokens, 1.63 full-length requests at once; 102.93 GB of weights per GPU and 20.44 GB free after sizing |
| Label | climbing. Aggregate throughput was still rising at 32, and the engine admits 51, so the ceiling is unmeasured |

Throughput is within a few percent of the sibling Kimi-K3 recipe on the same hardware, which reaches
40.3 tok/s single stream and 709.0 at concurrency 32. The two models cost the same to serve.

## Known limits

- **The KV cache dtype must be `fp8_e4m3`.** The engine validates it per model in
  `_kimi_k31_kv_cache_dtype` and raises
  `ValueError: Kimi-K3.1 requires --kv-cache-dtype fp8_e4m3 (the reference fp8_ds_mla format)`.
  Argparse accepts `bf16` and `auto`, and the sibling Kimi-K3 recipe serves `bf16` happily, so
  testing the parser alone is misleading: the check runs at model load, after the ranks have started.
- **`--moe-runner-backend marlin` is mandatory on Hopper.** MXFP4 is Blackwell-native, and the `auto`
  backend dequantizes every expert to BF16 and runs out of memory. The Kimi-K3 recipe pins it for the
  same reason.
- The container must be unpacked on node-local disk, see step 2.
- **Context is served at 262144, not the declared 1048576**, which is untested here. The pool holds
  426,688 tokens, so a single full-length request already claims most of it.
- Throughput above concurrency 32 is unmeasured, although the engine admits 51 and the curve was
  still climbing.
- Decode rate at a long input is not measured. Every figure here uses a 23-token input.
- Multimodal is left off. The architecture is `KimiK31ForConditionalGeneration` and the config
  carries image placeholders, but the vision path is unexercised and adds nothing for coding.
- Speculative decoding is not configured. A DSpark draft exists for this family and the vLLM recipe
  names `Inferact/Kimi-K3.1-Pro-DSpark-preview`, but it is not downloaded or measured here.
