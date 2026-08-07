# gemma-4-31B-it on one RTX PRO 6000 GPU

Status: Validated - vLLM 0.26.1rc1.dev162+g8700f86a7, protocol: slope(128,1152) at concurrency 1, 64, 128, 256, 512, 768 and 1024

| | |
| --- | --- |
| Served name | `gemma-4-31b` |
| Checkpoint | `gemma-4-31B-it`, Hugging Face `google/gemma-4-31B-it` |
| On disk | 62.6 GB, bf16, `Gemma4ForConditionalGeneration` |
| Served precision | FP8, quantized on load, 31.73 GiB of weights |
| Context | 262144, the checkpoint maximum |
| Hardware | 1 RTX PRO 6000 Blackwell, 97887 MiB, TP1 |
| Engine | vLLM 0.26.1rc1.dev162+g8700f86a7 |

## 1. Create the API key

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/gemma-4-31B-it-rtx-1.key
chmod 600 secrets/gemma-4-31B-it-rtx-1.key
```

- `secrets/vllm_api_key` is read when this file is absent.
- To rotate, replace the file and relaunch; the engine reads it once at launch.

## 2. Build the environment

Run on a compute node, not a login node. Needs `uv` and `mamba` on your PATH.

```
bash recipes/gemma-4-31B-it/rtx-1/env/build.sh
```

- About 7.8 GiB for the venv, plus a conda CUDA 13 toolkit alongside it under `ENV_ROOT`.
- The toolkit is required: the FlashInfer sm_120 JIT needs a complete CUDA 13 install, which the pip
  wheels do not provide.
- vLLM comes from the nightly cu130 index, because sm_120 needs a CUDA 13 build that ships nowhere else.
  Nothing in this environment is version pinned: the index rotates, and vLLM and torch carry no constraint.
  `env/build.sh` asks for FlashInfer 0.6.15 and got 0.6.15.post1. The build in the table above is already
  gone from the index, so a rebuild installs a newer one and its rates will differ.

## 3. Launch

Slurm, from the repo root:

```
sbatch --account=<your-account> recipes/gemma-4-31B-it/rtx-1/serve.sbatch
squeue --me                                  # NODELIST gives the host
tail -f gemma31-rtx-1-<jobid>.log
```

Direct, on a node you already hold:

```
# whole node to yourself
bash recipes/gemma-4-31B-it/rtx-1/serve_ssh.sh <node>

# shared node, the usual case: read the device Slurm gave you, then pin it
scontrol show job -d <jobid> | grep -oE 'IDX:[0-9]+' | cut -d: -f2
GPU=<n> bash recipes/gemma-4-31B-it/rtx-1/serve_ssh.sh <node>
```

- Startup is about 2 minutes with a warm compile cache, though a first launch on a node can be far slower.
- On the direct path the server log is node-local at `/tmp/$USER/vllm/`. Under Slurm it lands in the submit
  directory.

## 4. Verify

```
KEY=$(cat secrets/gemma-4-31B-it-rtx-1.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the node serving it>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"gemma-4-31b","messages":[{"role":"user","content":"What is 2+2? Answer briefly."}],"max_tokens":400}'
```

| Command | Expected |
| --- | --- |
| `/v1/models` with the key | `"id": "gemma-4-31b"`, `"max_model_len": 262144` |
| `/v1/models` without a key | `401` |
| chat completion | `content` holds the answer, `finish_reason: stop` |

- Use at least 400 output tokens. Reasoning arrives in a separate `reasoning` field, so a smaller budget
  can end with `finish_reason: length` and empty `content`.

## 5. Connect a client

```
export NODE=<the node serving it>
source recipes/gemma-4-31B-it/rtx-1/client.env
claude
```

- No output cap is needed at 262144.
- `client.env` sets `CLAUDE_CODE_MAX_CONTEXT_TOKENS=262144`. Claude Code does not recognize this served name
  and assumes a 200k window, which client 2.1.223 enforces.
- Use `ANTHROPIC_AUTH_TOKEN`. `ANTHROPIC_API_KEY` sends `x-api-key` and returns 401.
- `client.env` also sets `ANTHROPIC_SMALL_FAST_MODEL`; without it the client reaches for a hosted Haiku.
- OpenAI clients such as Codex: base URL `http://<node>:8000/v1`, same key, model `gemma-4-31b`. See
  [docs/clients.md](../../../docs/clients.md).

## 6. Stop it

```
scancel <jobid>                        # Slurm path
bash common/tools/stop.sh <node>       # direct path
```

## Tunable inputs

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/gemma-4-31B-it` | Serve a different copy of the checkpoint |
| `API_PORT` | 8000 | Listening port, and part of the SSH log file name |
| `MAX_MODEL_LEN` | 262144 | Context window, the checkpoint maximum |
| `GPU_UTIL` | 0.90 | Fraction of VRAM for weights plus KV cache |
| `TP` | 1 | Tensor parallel size |
| `QUANT` | `fp8` | Set but empty for bf16. bf16 will not start at 262144 on this card: it needs 33.29 GiB of KV against 23.57 available, and the engine reports a maximum length of 134624, so cap `MAX_MODEL_LEN` at 131072 for bf16 |
| `KV_FP8` | unset | `--kv-cache-dtype fp8`; halves KV bytes, and measured no change in rate |
| `ENFORCE_EAGER` | unset | Skip torch.compile and CUDA graph capture, to debug a startup failure |
| `SPEC_DRAFT` | unset | Path to the drafter checkpoint, `$MODELS_DIR/gemma-4-31B-it-assistant`. Works on this build and measured 2.6 times faster; see Benchmarking |
| `SPEC_TOKENS` | 3 | Speculative tokens per step, read only when `SPEC_DRAFT` is set |
| `EXTRA_ARGS` | unset | Extra flags appended to the `vllm serve` command line |
| `TOOL_PARSER` | `gemma4` | Tool call parser. Not forwarded by `serve_ssh.sh`, so it applies on the Slurm path |
| `REASONING_PARSER` | `gemma4` | Reasoning parser. Not forwarded by `serve_ssh.sh` either |
| `GPU` | unset | SSH path only: pin one device of a shared node |
| `LOG_DIR` | `/tmp/$USER/vllm` | Where the SSH path writes the server log |
| `VENV_DIR` | under `ENV_ROOT` | Use an environment built elsewhere |
| `CUDA13_DIR`, `CUDA_HOME` | under `ENV_ROOT` | The conda CUDA 13 toolkit the sm_120 JIT needs |
| `FLASHINFER_VERSION` | 0.6.15 | FlashInfer build installed |
| `KEY_NAME`, `KEY_FILE`, `VLLM_API_KEY` | this recipe's key | Which key `common/lib/api_key.sh` resolves; an exported `VLLM_API_KEY` wins |
| `NODE`, `GEMMA31_RTX_NODE` | unset | The node to launch on or connect to; set either, or pass the node as an argument |
| `MODELS_DIR`, `ENV_ROOT`, `VLLM_CACHE_ROOT`, `UV_CACHE_DIR` | `common/defaults.sh` | Override there or in `common/site.conf` |

## Benchmarking

Conditions:

| | |
| --- | --- |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| Input length | ISL 19 tokens. Rates at a long input are not measured; use `--prompt-tokens` |
| Output length | OSL 1152 tokens, output only, `ignore_eos` |
| Context | `MAX_MODEL_LEN=262144` |
| Allocation | 1 GPU, 16 cores, 180 GB, `kempner_rtx` |
| Sequence cap | `max_num_seqs` 1024, which equals the top sweep level |
| Preemption | 118,116 requests across the sweep, so the KV cache saturates before the sequence cap does |
| Endpoint | idle, and the benchmark client ran on a separate CPU-only node |
| Power | 600 W enforced, the card default, so not capped |

Results:

| Concurrency | Aggregate | Per stream | Spread over 3 runs |
| --- | --- | --- | --- |
| 1 | 39.5 tok/s | 39.5 tok/s | 39.5 to 39.5 |
| 64 | 1157.4 tok/s | 18.1 tok/s | 1156.3 to 1157.9 |
| 128 | 1470.9 tok/s | 11.5 tok/s | 1470.1 to 1471.3 |
| 256 | 1790.2 tok/s | 7.0 tok/s | 1785.5 to 1790.4 |
| 512 | 2095.9 tok/s | 4.1 tok/s | 2094.5 to 2096.8 |
| 768 | 2136.2 tok/s | 2.8 tok/s | 2121.7 to 2138.0 |
| 1024 | 2131.2 tok/s | 2.1 tok/s | 2102.3 to 2133.3 |

| | |
| --- | --- |
| Label | saturated. It varies 1.89 percent across 512 to 1024, the window the rule uses, under its 4 percent threshold. The highest median is at 768 |
| Quote for one caller | 39.5 tok/s |
| Quote for a shared endpoint | 2136.2 tok/s at concurrency 768 |
| KV cache | 401,491 tokens from 50.99 GiB, 1.53 full-length requests at once. The pool varies about 0.15 percent between launches |
| Speculative decoding | `SPEC_DRAFT=$MODELS_DIR/gemma-4-31B-it-assistant` measured 101.0 tok/s single stream against 39.5, a 2.6 times gain, with 64 percent of draft tokens accepted. It costs 5.7 percent of the KV pool, which drops to 378,673 tokens. Aggregate throughput with it is not measured |
| bf16 rate cost | bf16 measured 23.0 tok/s single stream against 39.5 for FP8, so bf16 costs about 42 percent |
| Long prompt, cold | 30,048 tokens in 7.8 s, 120,049 in 76.2 s, 240,049 in 272 s |

Reproduce:

```
KEY_NAME=gemma-4-31B-it-rtx-1 bash common/tools/bench.sh --host <node> --model gemma-4-31b
KEY_NAME=gemma-4-31B-it-rtx-1 bash common/tools/bench.sh --host <node> --model gemma-4-31b \
  --sweep 1,64,128,256,512,768,1024
```

- `KEY_NAME` is required once `secrets/` holds both this recipe's key and `vllm_api_key`. The server uses
  the recipe key, `bench.sh` alone resolves the shared one, and every request returns 401.

## Known limits

- Nothing in the environment is version pinned, so a rebuild installs different versions than the ones
  measured here.
- Do not add the conda CUDA 13 toolkit to `LD_LIBRARY_PATH`. Its `libcudart` shadows torch's runtime, so the
  recipe exposes it through `CPATH` and `LIBRARY_PATH` for compilation only. FlashInfer runs with its version
  check bypassed and compiles kernels from source on the first launch.
- Anthropic's hosted tools return HTTP 400. Use [docs/web-search.md](../../../docs/web-search.md).
