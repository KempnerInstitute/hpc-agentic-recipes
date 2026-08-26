# Qwen3.8-27B on one H200 GPU

Status: Validated - vLLM 0.25.1+cu129, protocol: slope(128,1152) at concurrency 1, 8, 32, 64, 128, 256, 512,
640, 768, 896 and 1024

| | |
| --- | --- |
| Served name | `qwen3.8-27b` |
| Checkpoint | `Qwen3.8-27B`, Hugging Face `Qwen/Qwen3.8-27B` |
| On disk | 51.75 GiB across 18 shards, BF16, `Qwen3_5ForConditionalGeneration` |
| Served precision | BF16, 51.92 GiB of weights per GPU |
| Context | 262144, the checkpoint maximum |
| Hardware | 1 H200 GPU, 143771 MiB, TP1, no NCCL |
| Engine | vLLM 0.25.1+cu129, CUDA graphs at `FULL_AND_PIECEWISE`, MTP speculative decoding at 3 draft tokens |

Hybrid attention: 48 of the 64 layers are linear attention and 16 are full attention. The checkpoint also
carries a vision tower and an MTP draft head. This recipe is validated for text.

## 1. Create the API key

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/Qwen3.8-27B-h200-1.key
chmod 600 secrets/Qwen3.8-27B-h200-1.key
```

- `secrets/vllm_api_key` is read when this file is absent.
- To rotate, replace the file and relaunch; the engine reads it once at launch.

## 2. Build the environment

Run on a compute node, not a login node.

```
bash recipes/Qwen3.8-27B/h200-1/env/build.sh
```

- About 13 GB under `ENV_ROOT`, default scratch. Set `ENV_ROOT` in `common/site.conf` to move it.
- Installs the cu129 wheel. These nodes run driver 575, CUDA 12.9, and vLLM's default wheel is CUDA 13.
- Removes `torchcodec`, which arrives as a vLLM dependency and needs FFmpeg 4 to 7 shared libraries that
  these nodes do not carry. The video backend defaults to opencv, so text serving does not use it.

## 3. Launch

Slurm, from the repo root:

```
sbatch --account=<your-account> recipes/Qwen3.8-27B/h200-1/serve.sbatch
squeue --me                                  # NODELIST gives the host
tail -f qwen38-27b-h200-1-<jobid>.log
```

Direct, on a node you already hold:

```
# whole node to yourself
bash recipes/Qwen3.8-27B/h200-1/serve_ssh.sh <node>

# shared node: read the device Slurm gave you, then pin it
scontrol show job -d <jobid> | grep -oE 'IDX:[0-9]+' | cut -d: -f2
GPU=<n> bash recipes/Qwen3.8-27B/h200-1/serve_ssh.sh <node>
```

| Stage | Measured |
| --- | --- |
| Launch to serving, first time on a node | 11 min 22 s |
| Launch to serving, caches warm | 2 min to 3 min 20 s |
| Weight load alone | 8.9 s warm |
| CUDA graph capture | 8 s, and 1.88 GiB |

- The first launch on any node compiles the hybrid attention kernels. That cache is node-local, so a
  different node pays it again.
- The server log is node-local at `/tmp/$USER/vllm/` on the direct path. Under Slurm it lands in the submit
  directory.

## 4. Verify

```
KEY=$(cat secrets/Qwen3.8-27B-h200-1.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the node serving it>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"What is 2+2? Answer briefly."}],"max_tokens":600}'
```

| Command | Expected |
| --- | --- |
| `/v1/models` with the key | `"id": "qwen3.8-27b"`, `"max_model_len": 262144` |
| `/v1/models` without a key | `401` |
| chat completion | `content` holds 4, `finish_reason: stop` |

- Thinking is on by default and arrives in a separate `reasoning` field, so `content` stays clean. The 2+2
  answer above returns 4 in `content` with 105 characters of `reasoning`.
- To turn thinking off, pass `"chat_template_kwargs": {"enable_thinking": false}`, which returns the same
  answer with an empty `reasoning`.
- The card recommends `temperature 1.0`, `top_p 0.95`, `top_k 20` for thinking mode, and `temperature 0.7`,
  `top_p 0.8`, `top_k 20`, `presence_penalty 1.5` for non-thinking mode.

## 5. Connect a client

```
export NODE=<the node serving it>
source recipes/Qwen3.8-27B/h200-1/client.env
claude
```

- `client.env` sets `CLAUDE_CODE_MAX_CONTEXT_TOKENS=262144`. Claude Code assumes and enforces a 200k window
  for a served name it does not recognize, which is below what this endpoint serves.
- Use `ANTHROPIC_AUTH_TOKEN`. `ANTHROPIC_API_KEY` sends `x-api-key` and returns 401.
- `client.env` also sets `ANTHROPIC_SMALL_FAST_MODEL`; without it the client reaches for a hosted Haiku.
- OpenAI clients such as Codex: base URL `http://<node>:8000/v1`, same key, model `qwen3.8-27b`. See
  [docs/clients.md](../../../docs/clients.md).

## 6. Stop it

```
scancel <jobid>                        # Slurm path
bash common/tools/stop.sh <node>       # direct path
```

## Tunable inputs

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/Qwen3.8-27B` | Serve a different copy of the checkpoint |
| `API_PORT` | 8000 | Listening port, and part of the SSH log file name |
| `MAX_MODEL_LEN` | 262144 | Context window, the checkpoint maximum |
| `GPU_UTIL` | 0.90 | Fraction of VRAM for weights plus KV cache |
| `TP` | 1 | Tensor parallel size |
| `KV_DTYPE` | `fp8` | KV cache dtype |
| `MTP_TOKENS` | 3 | Speculative draft tokens; set to 0 to serve without MTP |
| `SPEC_METHOD` | `mtp` | Speculative method |
| `ENFORCE_EAGER` | unset | Skip torch.compile and graph capture, to debug a startup failure |
| `CUDAGRAPH_MODE` | `FULL_AND_PIECEWISE` | Graph mode, ignored when `ENFORCE_EAGER` is set |
| `EXTRA_ARGS` | unset | Extra flags appended to the `vllm serve` command line |
| `TOOL_PARSER` | `qwen3_coder` | Tool call parser |
| `REASONING_PARSER` | `qwen3` | Reasoning parser |
| `LOG_DIR` | `/tmp/$USER/vllm` | Where the SSH path writes the server log |
| `VENV_DIR` | under `ENV_ROOT` | Use an environment built elsewhere |
| `VLLM_VERSION` | 0.25.1 | Engine version `env/build.sh` installs |
| `TRANSFORMERS_VERSION` | 5.14.1 | The transformers version the rates were measured with |
| `VLLM_CACHE_ROOT` | under `ENV_ROOT` | Where vLLM keeps compiled artifacts; the engine default is a small quota here |
| `KEY_NAME`, `KEY_FILE`, `VLLM_API_KEY` | this recipe's key | Which key `common/lib/api_key.sh` resolves; an exported `VLLM_API_KEY` wins |
| `NODE`, `QWEN38_27B_NODE` | unset | The node to launch on or connect to; set either, or pass the node as an argument |
| `MODELS_DIR`, `ENV_ROOT` | `common/defaults.sh` | Override there or in `common/site.conf` |

## Benchmarking

Conditions:

| | |
| --- | --- |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| Input length | ISL 61 tokens. Rates at a long input are not measured; use `--prompt-tokens` |
| Output length | OSL 1152 tokens, output only, `ignore_eos` |
| Context | `MAX_MODEL_LEN=262144` |
| Allocation for the measurement | 1 GPU, 16 cores, 360 GB, `kempner_h200` |
| Sequence cap | `max_num_seqs` 1024, the engine default, which equals the top sweep level |
| Preemption | none at any level |
| Endpoint | idle, and the benchmark client ran on the serving node |

Results:

| Concurrency | Aggregate | Per stream | Spread over 3 runs | TTFT median |
| --- | --- | --- | --- | --- |
| 1 | 122.1 tok/s | 122.1 tok/s | 122.1 to 122.1 | 61 ms |
| 8 | 900.0 tok/s | 112.5 tok/s | 899.8 to 939.4 | 92 ms |
| 32 | 2493.2 tok/s | 77.9 tok/s | 2484.9 to 2836.6 | 253 ms |
| 64 | 3408.5 tok/s | 53.3 tok/s | 3407.1 to 3576.9 | 414 ms |
| 128 | 3485.7 tok/s | 27.2 tok/s | 3301.9 to 3498.1 | 650 ms |
| 256 | 3680.7 tok/s | 14.4 tok/s | 3663.8 to 3860.7 | 1116 ms |
| 512 | 3623.2 tok/s | 7.1 tok/s | 3593.0 to 3972.1 | 1921 ms |
| 640 | 3681.9 tok/s | 5.8 tok/s | 3624.5 to 3978.4 | 2389 ms |
| 768 | 3736.1 tok/s | 4.9 tok/s | 3694.8 to 3969.5 | 2711 ms |
| 896 | 3790.1 tok/s | 4.2 tok/s | 3699.6 to 4036.7 | 3171 ms |
| 1024 | 3957.3 tok/s | 3.9 tok/s | 3652.6 to 4045.9 | 3633 ms |

| | |
| --- | --- |
| Label | rising. The highest value is at 1024, the top of the sweep, and the levels at or above 512 vary by 8.4 percent, so 3957.3 tok/s is a floor |
| Quote for one caller | 122.1 tok/s |
| Quote for a shared endpoint | 3957.3 tok/s at concurrency 1024 |
| KV cache | 1,804,253 tokens from 63.95 GiB, 6.88 full-length requests at once |
| Speculative decoding | MTP accepted 0.76, 0.49 and 0.30 of draft tokens by position, a mean accepted length of 2.55 |
| Long prompt | 180,052 tokens in 32.3 s cold; 0.99 s when the prefix is already cached |

Reproduce:

```
KEY_NAME=Qwen3.8-27B-h200-1 bash common/tools/bench.sh --host <node> --model qwen3.8-27b
KEY_NAME=Qwen3.8-27B-h200-1 bash common/tools/bench.sh --host <node> --model qwen3.8-27b \
  --sweep 1,8,32,64,128,256,512,640,768,896,1024
```

- `KEY_NAME` is required once `secrets/` holds more than one key, otherwise `bench.sh` resolves the shared
  key and every request returns 401.

## Choosing MTP or plain decoding

MTP is on by default because it nearly doubles the rate a single caller sees. It costs aggregate throughput
above concurrency 64, so set `MTP_TOKENS=0` for a busy shared endpoint.

| | MTP 3, the default | `MTP_TOKENS=0` |
| --- | --- | --- |
| One caller | 122.1 tok/s | 64.8 tok/s |
| Concurrency 8 | 900.0 tok/s | 483.5 tok/s |
| Concurrency 256 | 3680.7 tok/s | 5434.5 tok/s |
| Concurrency 1024 | 3957.3 tok/s | 5346.8 tok/s |

## Known limits

- Prefix caching must stay enabled, which `serve.sh` does with `--enable-prefix-caching`. Without it a
  repeated long prefix is not reused: the same 180,052-token prompt took 30.88 s on the second request
  rather than 0.99 s.
- Image input is untested. This recipe is validated for text.
- Anthropic's hosted tools return HTTP 400. Use [docs/web-search.md](../../../docs/web-search.md).
