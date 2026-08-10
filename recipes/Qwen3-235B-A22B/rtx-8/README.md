# Qwen3-235B-A22B on one RTX PRO 6000 node

Status: Validated - vLLM 0.25.1, protocol: slope(128,1152) at concurrency 1, 8, 32, 64, 128, 256, 512, 640, 768, 896 and 1024

| | |
| --- | --- |
| Served name | `qwen3-235b` |
| Checkpoint | `Qwen3-235B-A22B`, Hugging Face `Qwen/Qwen3-235B-A22B` |
| On disk | 437.90 GiB across 118 shards, BF16 |
| Served precision | BF16 as the checkpoint ships, 54.92 GiB of weights per GPU |
| Context | 40960, the checkpoint maximum. The spare KV cannot buy a larger one; see Known limits |
| Hardware | 1 RTX PRO 6000 Blackwell node, 8 GPUs, 97887 MiB each, TP8 over PCIe with no NVLink |
| Engine | vLLM 0.25.1, CUDA graphs, prefix caching, no speculative decoding |

## 1. Create the API key

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/Qwen3-235B-A22B-rtx-8.key
chmod 600 secrets/Qwen3-235B-A22B-rtx-8.key
```

- `secrets/vllm_api_key` is read when this file is absent.
- To rotate, replace the file and relaunch; the engine reads it once at launch.

## 2. Build the environment

Run on a compute node, not a login node. Needs `uv` and `mamba` on your PATH.

```
module load Mambaforge
bash recipes/Qwen3-235B-A22B/rtx-8/env/build.sh
```

- About 9.0 GB for the venv, plus a conda CUDA 13.0 toolkit alongside it under `ENV_ROOT`, default scratch.
- The toolkit is required: the FlashInfer sm_120 JIT needs a complete CUDA 13 install, which the node's
  runtime-only `/usr/local/cuda-13` and the fragmented pip nvcc wheels do not provide.
- FlashInfer is pinned to 0.6.15. vLLM 0.25.1 pins 0.6.13, which rejects the `kv_scale_format` argument the
  sm_120 backend passes and fails at the first request. No matching cubin package exists at 0.6.15, so
  `env/env.sh` bypasses the version check and kernels compile from source on the first launch.
- The vLLM wheel comes from the nightly cu130 index rather than PyPI, because sm_120 needs the CUDA 13
  build. The version is pinned explicitly, since the installed metadata carries no local version tag and an
  unpinned install drifts silently.

## 3. Launch

Slurm, from the repo root:

```
sbatch --account=<your-account> recipes/Qwen3-235B-A22B/rtx-8/serve.sbatch
squeue --me                                  # NODELIST gives the host
tail -f qwen3-235b-<jobid>.log
```

Direct, on a node you already hold. This recipe uses all eight GPUs, so there is no device to pin:

```
bash recipes/Qwen3-235B-A22B/rtx-8/serve_ssh.sh <node>
```

| Stage | Measured |
| --- | --- |
| Launch to serving | 3 min 02 s to 6 min 29 s across four launches on one node |
| Weight load, 118 shards across 8 ranks | 182.75 s first time on the node, 63.52 s once the page cache is warm |

- The first launch on any node compiles the sm_120 attention kernels from source, because no matching
  FlashInfer cubin package exists. That cache is node-local, so a different node pays it again.
- On the direct path the server log is node-local at `/tmp/$USER/vllm/`. Under Slurm it lands in the submit
  directory.

## 4. Verify

```
KEY=$(cat secrets/Qwen3-235B-A22B-rtx-8.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the node serving it>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/chat/completions \
  -d '{"model":"qwen3-235b","messages":[{"role":"user","content":"What is 2+2? Answer briefly."}],"max_tokens":400}'
```

| Command | Expected |
| --- | --- |
| `/v1/models` with the key | `"id": "qwen3-235b"`, `"max_model_len": 40960` |
| `/v1/models` without a key | `401` |
| chat completion | `content` holds the answer, `finish_reason: stop` |

- This is a thinking model and vLLM returns its reasoning in a separate `reasoning` field. Give it at
  least 400 output tokens: with a smaller budget the whole allowance goes on reasoning, `finish_reason`
  is `length`, and `content` comes back empty, which looks like a broken endpoint but is not. The smoke
  test above used 123 tokens.

## 5. Connect a client

```
export NODE=<the node serving it>
source recipes/Qwen3-235B-A22B/rtx-8/client.env
claude
```

- `client.env` sets `CLAUDE_CODE_MAX_CONTEXT_TOKENS=40960`. Claude Code assumes a 200k window for a served
  name it does not recognize, five times what this endpoint serves, so without the pin it overflows and
  requests fail with `maximum context length is 40960` rather than compacting.
- `client.env` also caps `CLAUDE_CODE_MAX_OUTPUT_TOKENS` at 4096. The client asks for 32000 output tokens
  by default, which would leave too little of a 40960 window for the prompt. Raise it only if you also
  raise `MAX_MODEL_LEN`, which on this recipe you cannot; see Known limits.
- 40960 is small for agentic work, and Claude Code holds back a fixed allowance for auto-compaction on top
  of it, so the room left for the conversation itself is a fraction of the window. Prefer an endpoint with a
  larger context for long sessions.
- Use `ANTHROPIC_AUTH_TOKEN`. `ANTHROPIC_API_KEY` sends `x-api-key` and returns 401.
- `client.env` also sets `ANTHROPIC_SMALL_FAST_MODEL`; without it the client reaches for a hosted Haiku.
- OpenAI clients such as Codex: base URL `http://<node>:8000/v1`, same key, model `qwen3-235b`. See
  [docs/clients.md](../../../docs/clients.md).

## 6. Stop it

```
scancel <jobid>                        # Slurm path
bash common/tools/stop.sh <node>       # direct path
```

## Tunable inputs

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/Qwen3-235B-A22B` | Serve a different copy of the checkpoint |
| `API_PORT` | 8000 | Listening port |
| `MAX_MODEL_LEN` | 40960 | Context window, and the checkpoint maximum. Raising it needs rope scaling, which breaks this model; see Known limits |
| `GPU_UTIL` | 0.90 | Fraction of VRAM for weights plus KV cache |
| `TP` | 8 | Tensor parallel size, and the node's GPU count |
| `QUANT` | unset | Quantize weights on load, for example `fp8`. Measured no faster here |
| `TOOL_PARSER` | `hermes` | Tool call parser |
| `REASONING_PARSER` | `qwen3` | Reasoning parser |
| `EXTRA_ARGS` | unset | Extra flags appended to the `vllm serve` command line |
| `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` | 3600 | Raised from 480 so a storage stall that resolves is survivable |
| `LOG_DIR` | `/tmp/$USER/vllm` | Where the SSH path writes the server log |
| `VENV_DIR` | under `ENV_ROOT` | Use an environment built elsewhere |
| `CUDA13_DIR` | under `ENV_ROOT` | Use a CUDA 13.0 toolkit built elsewhere |
| `VLLM_VERSION` | 0.25.1 | Engine version `env/build.sh` installs |
| `FLASHINFER_VERSION` | 0.6.15 | FlashInfer version `env/build.sh` installs |
| `VLLM_CACHE_ROOT` | under `ENV_ROOT` | Where vLLM keeps compiled artifacts; the engine default is a small quota here |
| `KEY_NAME`, `KEY_FILE`, `VLLM_API_KEY` | this recipe's key | Which key `common/lib/api_key.sh` resolves; an exported `VLLM_API_KEY` wins |
| `NODE`, `QWEN3_235B_NODE` | unset | The node to launch on or connect to; set either, or pass the node as an argument |
| `MODELS_DIR`, `ENV_ROOT` | `common/defaults.sh` | Override there or in `common/site.conf` |

## Benchmarking

Conditions:

| | |
| --- | --- |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| Input length | ISL 17 tokens. Rates at a long input are not measured; use `--prompt-tokens` |
| Output length | OSL 1152 tokens, output only |
| Context | `MAX_MODEL_LEN=40960` |
| Allocation for the measurement | 8 GPUs, 128 cores, 1440 GB, `kempner_rtx` |
| Sequence cap | `max_num_seqs` 1024, the engine default, which equals the top sweep level |
| Preemption | 5,762 across the sweep. The pool is 575,440 tokens while concurrency 1024 at this output length needs about 1.2 million, so the KV cache binds before the sequence cap. The counter is on `/metrics`; the server log does not report preemption |
| Endpoint | idle, and the benchmark client ran on a separate CPU-only node |
| Power | 600 W enforced, the card default, so not capped. Median 234 W across 12608 samples and a 302 W peak, with nothing at or above 590, clocks 2422 MHz |

Results:

| Concurrency | Aggregate | Per stream | Spread over 3 runs | TTFT median |
| --- | --- | --- | --- | --- |
| 1 | 62.7 tok/s | 62.7 tok/s | 61.8 to 62.9 | 63 ms |
| 8 | 312.1 tok/s | 39.0 tok/s | 311.8 to 313.6 | 74 ms |
| 32 | 890.7 tok/s | 27.8 tok/s | 857.8 to 890.8 | 97 ms |
| 64 | 1273.9 tok/s | 19.9 tok/s | 1273.4 to 1274.6 | 135 ms |
| 128 | 1757.8 tok/s | 13.7 tok/s | 1742.7 to 1760.4 | 191 ms |
| 256 | 3131.6 tok/s | 12.2 tok/s | 3084.8 to 3141.3 | 289 ms |
| 512 | 3947.5 tok/s | 7.7 tok/s | 3623.8 to 3950.3 | 457 ms |
| 640 | 3691.0 tok/s | 5.8 tok/s | 3684.5 to 3694.0 | 548 ms |
| 768 | 3561.9 tok/s | 4.6 tok/s | 3555.7 to 3614.9 | 530 ms |
| 896 | 3664.1 tok/s | 4.1 tok/s | 3632.1 to 3669.8 | 567 ms |
| 1024 | 3736.5 tok/s | 3.6 tok/s | 3718.8 to 3751.8 | 614 ms |

| | |
| --- | --- |
| Label | peak. The highest value is at concurrency 512 and the curve turns over above it, so 3947.5 tok/s is a measured ceiling rather than the edge of the sweep |
| Quote for one caller | 62.7 tok/s |
| Quote for a shared endpoint | 3947.5 tok/s at concurrency 512 |
| KV cache | 575,440 tokens from 25.79 GiB per GPU, 14.05 full-length requests at once |
| Long prompt | 40,081 tokens in 7.5 s cold; 0.3 s when the prefix is already cached |

Reproduce:

```
KEY_NAME=Qwen3-235B-A22B-rtx-8 bash common/tools/bench.sh --host <node> --model qwen3-235b
KEY_NAME=Qwen3-235B-A22B-rtx-8 bash common/tools/bench.sh --host <node> --model qwen3-235b \
  --sweep 1,8,32,64,128,256,512,640,768,896,1024
```

- `KEY_NAME` is required once `secrets/` holds more than one key, otherwise `bench.sh` resolves the shared
  key and every request returns 401.

## Known limits

- **The KV headroom cannot be spent on a longer context.** The pool holds 575,440 tokens against a 40960
  window, which is 14 full-length requests at once, so the hardware would serve far more. Reaching past
  40960 needs static YaRN rope scaling, and that breaks this model's output. Both documented factors were
  measured on this node, with a marker planted at the head of the prompt and varied filler after it:

  | Configuration | 1K prompt | 30K | 38K | 62K and beyond |
  | --- | --- | --- | --- | --- |
  | 40960, no scaling | correct | correct | correct | not served |
  | YaRN factor 2, 65536 | correct | garbage | garbage | garbage |
  | YaRN factor 4, 131072 | correct | garbage | garbage | garbage |

  Garbage is literal, not a wrong answer: `8118811116144348668888888868888888888848888848888848888` in
  place of the marker. The engine reports no error and advertises the full window through `/v1/models`, so
  nothing but reading the output reveals it. The failure starts at 30K, well below where either scaling is
  needed, which is the short-context degradation that comes with static YaRN rather than a limit of the
  extension. Serve 40960.
- Give thinking models room. With a small output budget the whole allowance goes on reasoning and `content`
  comes back empty; use at least 400 output tokens, and more for a question that reasons at length.
- Do not add the conda CUDA 13 toolkit to `LD_LIBRARY_PATH`. Its `libcudart` shadows torch's runtime and
  pulls in a missing `libcupti.so.13`, which breaks import; the recipe exposes it through `CPATH` and
  `LIBRARY_PATH` for compilation only.
- No NVLink on these nodes, so `env/env.sh` sets `NCCL_P2P_DISABLE=1`. Without it NCCL initialization hangs
  with no error and the server never becomes ready.
- Do not enable `--enable-expert-parallel`. All-reduce already crosses PCIe rather than NVLink, and the
  added all-to-all traffic measured about 9 percent slower on this hardware. FP8 weights bought nothing
  for the same reason: decode here is limited by cross-GPU communication, not weight bandwidth.
- Anthropic's hosted tools return HTTP 400. Use [docs/web-search.md](../../../docs/web-search.md).
