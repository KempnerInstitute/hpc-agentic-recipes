# MiMo-V2.6-Pro-RL on one RTX PRO 6000 node

Status: Validated - vLLM 0.33.0+cu130, protocol: slope(128,1152) at concurrency 1, 8, 32, 64 and 128

| | |
| --- | --- |
| Served name | `mimo-v2.6-pro` |
| Checkpoint | `MiMo-V2.6-Pro-RL`, Hugging Face `XiaomiMiMo/MiMo-V2.6-Pro-RL` |
| On disk | 534 GiB across 130 shards, FP8 compute with MXFP4-stored MoE weights, `MiMoV2ForCausalLM` |
| Served precision | FP8 e4m3 compute, MXFP4 expert storage, BF16 MoE router |
| Context | 65536. The checkpoint declares 1048576; this is a memory choice, see Known limits |
| Hardware | 1 RTX PRO 6000 Blackwell node, 8 GPUs, 97887 MiB each, sm120, TP8 inside the node |
| Engine | vLLM 0.33.0+cu130, CUDA graphs on, no speculative decoding |
| Allocation for the measurement | 1 node, 8 GPUs, 128 cores and 1440 GB, `kempner_rtx` |

MiMo-V2.6-Pro is Xiaomi's flagship mixture-of-experts: 1.02T total parameters, 42B active per token, 70
layers with a hybrid sliding-window and global attention backbone at a 128-token window. It carries
omnimodal encoders; this recipe serves the text path.

**This recipe runs on one RTX node rather than H200 nodes, and that is a hardware constraint, not a
preference.** The model needs its eight tensor-parallel ranks inside a single box. See Known limits.

## 1. Create the API key

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/MiMo-V2.6-Pro-RL-rtx-8.key
chmod 600 secrets/MiMo-V2.6-Pro-RL-rtx-8.key
```

- `secrets/vllm_api_key` is read when this file is absent.
- Requests without the key receive HTTP 401.

## 2. Build the environment

```
bash recipes/MiMo-V2.6-Pro-RL/rtx-8/env/build.sh
```

Builds into `$ENV_ROOT/MiMo-V2.6-Pro-RL/rtx-8/venv` and pins vLLM 0.33.0+cu130 with transformers 5.20.2.
Run it on a compute node: a CPU allocation is enough.

`env/env.sh` loads `gcc/13.2.0-fasrc01` and `cuda/13.3.1-fasrc01`, which FlashInfer needs to compile its
sampling kernel on first launch. Without them the build fails with ninja exit 127 and the engine never
starts.

## 3. Launch

```
sbatch --account=<your-account> recipes/MiMo-V2.6-Pro-RL/rtx-8/serve.sbatch
```

On a node you already hold:

```
bash recipes/MiMo-V2.6-Pro-RL/rtx-8/serve_ssh.sh <node>
```

Startup is about 17 minutes, dominated by reading 534 GiB, then 18 s of graph capture for 0.43 GiB per
GPU. The log is `$MIMO_LOG_DIR/vllm-mimo.log`, default under `/tmp/$USER/mimo`.

## 4. Verify

```
KEY=$(cat secrets/MiMo-V2.6-Pro-RL-rtx-8.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the node>

curl -s -H "Authorization: Bearer $KEY" http://$NODE:8000/v1/models

curl -s -o /dev/null -w '%{http_code}\n' http://$NODE:8000/v1/models

curl -s -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  http://$NODE:8000/v1/completions \
  -d '{"model":"mimo-v2.6-pro","prompt":"The capital of France is","max_tokens":16,"temperature":0}'
```

| Command | Expected |
| --- | --- |
| `/v1/models` with the key | `"id": "mimo-v2.6-pro"` |
| `/v1/models` without a key | `401` |
| raw completion | coherent continuation, for example ` Paris. In the early Middle Ages, ...` |

The raw completion probe is in this list deliberately. A misconfigured parallel layout can serve happily
while generating nonsense, and a raw completion bypasses the chat template and both parsers, so it is the
quickest way to tell a healthy deployment from a silently broken one. See Known limits.

Reasoning arrives in `reasoning_content` and the answer in `content`. The MiMo reasoning parser honors
`chat_template_kwargs.enable_thinking`.

## 5. Connect a client

```
export NODE=<the node>
source recipes/MiMo-V2.6-Pro-RL/rtx-8/client.env
claude
```

`client.env` exports the base URL, the key, the served model name, and
`CLAUDE_CODE_MAX_CONTEXT_TOKENS=65536`, matching what this recipe serves. Use `ANTHROPIC_AUTH_TOKEN`, not
`ANTHROPIC_API_KEY`.

## 6. Stop it

```
scancel <job id>
```

For an ssh launch, kill the `vllm serve` process on the node.

## Tunable inputs

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/MiMo-V2.6-Pro-RL` | Serve a different copy of the checkpoint |
| `API_PORT` | 8000 | Listening port |
| `MAX_MODEL_LEN` | 65536 | Context window, fixed at init |
| `GPU_UTIL` | 0.92 | Fraction of VRAM for weights plus KV cache |
| `TP` | 8 | Tensor parallel size. Must equal the node's GPU count; see Known limits |
| `ENFORCE_EAGER` | unset | Set to fall back to eager instead of CUDA graphs |
| `TOOL_PARSER` | `mimo` | Tool call parser |
| `REASONING_PARSER` | `mimo` | Reasoning parser |
| `EXTRA_ARGS` | unset | Extra flags appended to the `vllm serve` command line |
| `KEY_NAME`, `KEY_FILE`, `VLLM_API_KEY` | this recipe's key | Which key `common/lib/api_key.sh` resolves |

There is deliberately no pipeline parallel knob. See Known limits.

## Benchmarking

Conditions:

| | |
| --- | --- |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| Input length | ISL 23 tokens. Rates at a long input are not measured; use `--prompt-tokens` |
| Output length | OSL 1152 tokens, output only, `ignore_eos` |
| Context | `MAX_MODEL_LEN=65536`, `GPU_UTIL=0.92`, no speculative decoding |
| Allocation for the measurement | 1 node, 8 GPUs, 128 cores and 1440 GB, `kempner_rtx` |
| Sequence cap | `max_num_seqs` 1024, the engine default, well above the top sweep level |
| Endpoint | idle; the benchmark client ran on another node |

Results:

| Concurrency | Aggregate | Per stream | Spread over 3 runs |
| --- | --- | --- | --- |
| 1 | 84.9 tok/s | 84.9 tok/s | 83.8 to 85.3 |
| 8 | 315.1 tok/s | 39.4 tok/s | 315.0 to 315.2 |
| 32 | 610.9 tok/s | 19.1 tok/s | 605.4 to 767.8 |
| 64 | 815.5 tok/s | 12.7 tok/s | 815.3 to 818.9 |
| 128 | 1136.0 tok/s | 8.9 tok/s | 1135.3 to 1140.2 |

| | |
| --- | --- |
| Engine facts | KV cache 895,431 tokens, 13.66 full-length requests at once; graph capture 18 s and 0.43 GiB per GPU |
| Label | rising. The curve is still climbing at 128, so 1136.0 is a floor |

## Known limits

- **Tensor parallelism must stay inside one node, and pipeline parallelism must not be used.** This is the
  reason the recipe targets an 8-GPU RTX node rather than the H200 nodes, which hold 4 GPUs each:
  - TP8 split across two 4-GPU H200 nodes **hangs**. vLLM warns `tensor_parallel_size=8 is bigger than a
    reserved number of GPUs (4 GPUs) in a node`, Ray forms `world_size=8`, NCCL initializes on all ranks,
    and then nothing happens: no weight loading at all, GPUs holding only bare CUDA contexts.
  - TP4 x PP2 across the same two nodes **loads, serves, and generates garbage**. The engine looks
    healthy, reporting an 8,668,322-token KV cache and capturing graphs on all 8 workers, but raw
    completions return whitespace: `"The capital of France is"` produced `'            :           '`.
    `MiMoV2ForCausalLM` declares `SupportsPP`, so this is a bug rather than an unsupported configuration,
    most likely a split that breaks the 70-layer sliding-window backbone.
  - TP8 inside one RTX node works, with identical weights, engine and parsers. 765 GiB of VRAM covers the
    model's 633 GiB need.
  Treat a silently wrong deployment as the risk here, not a crash: always run the raw completion probe in
  step 4 after changing the parallel layout.
- Context is served at 65536 rather than the checkpoint's 1048576. That is a deliberate margin rather than
  a measured ceiling: the 895,431-token pool would cover a longer window, and raising `MAX_MODEL_LEN` is
  untested here.
- Throughput is still rising at concurrency 128, so the ceiling is unmeasured.
- Concurrency 32 is the one noisy level, 605.4 to 767.8 across three runs, while every other level
  repeats to within 1 percent. The median is reported; treat that row as approximate.
- Tool-call repetition, which the upstream recipe cites as the reason to prefer the MOPD variant for
  agentic workloads, did not reproduce: three multi-step tool prompts each returned exactly one tool call
  with `finish_reason: tool_calls`. Three runs do not prove absence, and `XiaomiMiMo/MiMo-V2.6-Pro-MOPD`
  remains the upstream recommendation for agentic use.
- Speculative decoding is not used. The checkpoint ships a DFlash drafter in a `dflash/` subdirectory and
  the SGLang cookbook uses it, but this recipe does not and it is unmeasured here.
- Decode rate at a long input is not measured. Every figure here uses a 23-token input.
- The omnimodal encoders are loaded but not exercised. This recipe is validated for text only.
