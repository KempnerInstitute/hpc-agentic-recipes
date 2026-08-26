# Qwen3.8-27B-FP8 on one H200 GPU

Status: Validated - vLLM 0.25.1+cu129, protocol: slope(128,1152) at concurrency 1, 8, 32, 64, 128, 256, 512,
640, 768, 896 and 1024

| | |
| --- | --- |
| Served name | `qwen3.8-27b-fp8` |
| Checkpoint | `Qwen3.8-27B-FP8`, Hugging Face `Qwen/Qwen3.8-27B-FP8` |
| On disk | 28.75 GiB across 66 shards, FP8, `Qwen3_5ForConditionalGeneration` |
| Served precision | FP8 e4m3 with `weight_block_size` [128, 128], 28.99 GiB of weights per GPU. The vision tower stays unquantized |
| Context | 262144, the checkpoint maximum |
| Hardware | 1 H200 GPU, 143771 MiB, TP1, no NCCL |
| Engine | vLLM 0.25.1+cu129, CUDA graphs at `FULL_AND_PIECEWISE`, MTP speculative decoding at 3 draft tokens |

Hybrid attention: 48 of the 64 layers are linear attention and 16 are full attention. The checkpoint also
carries a vision tower and an MTP draft head. This recipe is validated for text.

## 1. Create the API key

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/Qwen3.8-27B-FP8-h200-1.key
chmod 600 secrets/Qwen3.8-27B-FP8-h200-1.key
```

- `secrets/vllm_api_key` is read when this file is absent.
- To rotate, replace the file and relaunch; the engine reads it once at launch.

## 2. Build the environment

Run on a compute node, not a login node.

```
bash recipes/Qwen3.8-27B-FP8/h200-1/env/build.sh
```

- About 13 GB under `ENV_ROOT`, default scratch. Set `ENV_ROOT` in `common/site.conf` to move it.
- Installs the cu129 wheel. These nodes run driver 575, CUDA 12.9, and vLLM's default wheel is CUDA 13.
- Removes `torchcodec`, which arrives as a vLLM dependency and needs FFmpeg 4 to 7 shared libraries that
  these nodes do not carry. The video backend defaults to opencv, so text serving does not use it.

## 3. Launch

Slurm, from the repo root:

```
sbatch --account=<your-account> recipes/Qwen3.8-27B-FP8/h200-1/serve.sbatch
squeue --me                                  # NODELIST gives the host
tail -f qwen38-27b-fp8-h200-1-<jobid>.log
```

Direct, on a node you already hold:

```
# whole node to yourself
bash recipes/Qwen3.8-27B-FP8/h200-1/serve_ssh.sh <node>

# shared node: read the device Slurm gave you, then pin it
scontrol show job -d <jobid> | grep -oE 'IDX:[0-9]+' | cut -d: -f2
GPU=<n> bash recipes/Qwen3.8-27B-FP8/h200-1/serve_ssh.sh <node>
```

| Stage | Measured |
| --- | --- |
| Launch to serving, first time on a node | 14 min 42 s |
| Launch to serving, caches warm | 2 min 20 s |
| Weight load alone | 29.4 s |
| CUDA graph capture | 141 s, and 2.62 GiB |

- The first launch on any node compiles the hybrid attention kernels. That cache is node-local, so a
  different node pays it again.
- The server log is node-local at `/tmp/$USER/vllm/` on the direct path. Under Slurm it lands in the submit
  directory.

## 4. Verify

```
KEY=$(cat secrets/Qwen3.8-27B-FP8-h200-1.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the node serving it>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"qwen3.8-27b-fp8","messages":[{"role":"user","content":"What is 2+2? Answer briefly."}],"max_tokens":600}'
```

| Command | Expected |
| --- | --- |
| `/v1/models` with the key | `"id": "qwen3.8-27b-fp8"`, `"max_model_len": 262144` |
| `/v1/models` without a key | `401` |
| chat completion | `content` holds 4, `finish_reason: stop` |

- Thinking is on by default and arrives in a separate `reasoning` field, so `content` stays clean. The 2+2
  answer above returns 4 in `content` with 57 characters of `reasoning`.
- To turn thinking off, pass `"chat_template_kwargs": {"enable_thinking": false}`, which returns the same
  answer with an empty `reasoning`.
- The card recommends `temperature 1.0`, `top_p 0.95`, `top_k 20` for thinking mode, and `temperature 0.7`,
  `top_p 0.8`, `top_k 20`, `presence_penalty 1.5` for non-thinking mode.

## 5. Connect a client

```
export NODE=<the node serving it>
source recipes/Qwen3.8-27B-FP8/h200-1/client.env
claude
```

- `client.env` sets `CLAUDE_CODE_MAX_CONTEXT_TOKENS=262144`. Claude Code assumes and enforces a 200k window
  for a served name it does not recognize, which is below what this endpoint serves.
- Use `ANTHROPIC_AUTH_TOKEN`. `ANTHROPIC_API_KEY` sends `x-api-key` and returns 401.
- `client.env` also sets `ANTHROPIC_SMALL_FAST_MODEL`; without it the client reaches for a hosted Haiku.
- OpenAI clients such as Codex: base URL `http://<node>:8000/v1`, same key, model `qwen3.8-27b-fp8`. See
  [docs/clients.md](../../../docs/clients.md).

## 6. Stop it

```
scancel <jobid>                        # Slurm path
bash common/tools/stop.sh <node>       # direct path
```

## Tunable inputs

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/Qwen3.8-27B-FP8` | Serve a different copy of the checkpoint |
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
| `NODE`, `QWEN38_27B_FP8_NODE` | unset | The node to launch on or connect to; set either, or pass the node as an argument |
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
| 1 | 170.6 tok/s | 170.6 tok/s | 170.4 to 170.7 | 47 ms |
| 8 | 1209.3 tok/s | 151.2 tok/s | 1208.8 to 1285.7 | 70 ms |
| 32 | 3265.9 tok/s | 102.1 tok/s | 3124.9 to 7728.3 | 208 ms |
| 64 | 4236.8 tok/s | 66.2 tok/s | 4224.7 to 4643.9 | 321 ms |
| 128 | 4749.0 tok/s | 37.1 tok/s | 4709.0 to 4821.8 | 574 ms |
| 256 | 4758.0 tok/s | 18.6 tok/s | 4720.6 to 4860.3 | 791 ms |
| 512 | 4750.2 tok/s | 9.3 tok/s | 4729.4 to 4761.4 | 1321 ms |
| 640 | 4748.7 tok/s | 7.4 tok/s | 4531.3 to 4783.9 | 1776 ms |
| 768 | 4809.6 tok/s | 6.3 tok/s | 4745.2 to 4971.7 | 1897 ms |
| 896 | 4839.1 tok/s | 5.4 tok/s | 4553.2 to 4871.5 | 2374 ms |
| 1024 | 4855.4 tok/s | 4.7 tok/s | 4805.5 to 4860.9 | 2555 ms |

| | |
| --- | --- |
| Label | saturated. The levels at or above 512 vary by 2.2 percent, under the 4 percent the rule uses, and the highest value is at 1024 |
| Quote for one caller | 170.6 tok/s |
| Quote for a shared endpoint | 4855.4 tok/s at concurrency 1024 |
| KV cache | 2,442,772 tokens from 86.58 GiB, 9.32 full-length requests at once |
| Speculative decoding | MTP accepted 0.76, 0.49 and 0.31 of draft tokens by position, a mean accepted length of 2.57 |
| Long prompt | 180,052 tokens in 27.0 s cold; 0.93 s when the prefix is already cached |

Reproduce:

```
KEY_NAME=Qwen3.8-27B-FP8-h200-1 bash common/tools/bench.sh --host <node> --model qwen3.8-27b-fp8
KEY_NAME=Qwen3.8-27B-FP8-h200-1 bash common/tools/bench.sh --host <node> --model qwen3.8-27b-fp8 \
  --sweep 1,8,32,64,128,256,512,640,768,896,1024
```

- `KEY_NAME` is required once `secrets/` holds more than one key, otherwise `bench.sh` resolves the shared
  key and every request returns 401.

## Choosing MTP or plain decoding

MTP is on by default because it raises the rate a single caller sees by three quarters. It costs aggregate throughput
above concurrency 64, so set `MTP_TOKENS=0` for a busy shared endpoint.

| | MTP 3, the default | `MTP_TOKENS=0` |
| --- | --- | --- |
| One caller | 170.6 tok/s | 98.3 tok/s |
| Concurrency 8 | 1209.3 tok/s | 721.6 tok/s |
| Concurrency 256 | 4758.0 tok/s | 6181.8 tok/s |
| Concurrency 1024 | 4855.4 tok/s | 6081.8 tok/s |

## Known limits

- Prefix caching must stay enabled, which `serve.sh` does with `--enable-prefix-caching`. Without it a
  repeated long prefix is not reused: a repeated 180,052-token prompt is prefilled again in full
  rather than returning in 0.93 s.
- Image input is untested. This recipe is validated for text.
- Anthropic's hosted tools return HTTP 400. Use [docs/web-search.md](../../../docs/web-search.md).
