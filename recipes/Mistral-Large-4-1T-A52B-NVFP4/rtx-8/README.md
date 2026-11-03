# Mistral Large 4 1T-A52B NVFP4 on one RTX PRO 6000 node

Status: Validated - vLLM in a container, protocol: slope(128,1152) at concurrency 1, 8, 32 and 48

| | |
| --- | --- |
| Served name | `mistral-large-4` |
| Checkpoint | `Mistral-Large-4-1T-A52B-NVFP4`, Hugging Face `mistralai/Mistral-Large-4-1T-A52B-NVFP4` |
| On disk | 568 GiB across 32 shards, NVFP4 routed experts with block-FP8 attention and dense layers |
| Served precision | NVFP4 weights, ModelOpt mixed precision, `fp8_ds_mla` KV cache chosen by the engine |
| Context | 98304. The checkpoint declares 1048576; the memory budget sets the ceiling, see Known limits |
| Hardware | 1 RTX PRO 6000 Blackwell node, 8 GPUs, 97887 MiB each, sm120, TP8 |
| Engine | vLLM in Singularity, `FLASHINFER_MLA_SPARSE_SM120` attention, DSpark speculative decoding |
| Allocation for the measurement | 1 node, 8 GPUs, 128 cores and 1440 GB, `kempner_rtx` |

The same 1T-A52B mixture-of-experts as the FP8 recipe, quantized to NVFP4 so it fits one node. It serves
the text path only.

**This recipe runs vLLM while the FP8 sibling runs SGLang.** SGLang cannot serve this checkpoint on
sm120 at all. See Known limits.

## 1. Create the API key

```
mkdir -p secrets
printf '%s' "sk-local-$(openssl rand -hex 24)" > secrets/Mistral-Large-4-1T-A52B-NVFP4-rtx-8.key
chmod 600 secrets/Mistral-Large-4-1T-A52B-NVFP4-rtx-8.key
```

- `secrets/vllm_api_key` is read when this file is absent.
- Requests without the key receive HTTP 401.
- The key is passed to the container in the environment as `VLLM_API_KEY`, never on the command line.

## 2. Get the container

No vLLM release or nightly wheel carries Mistral Large 4, so the engine is the published image. There is
no venv to build.

`env/env.sh` resolves it in order: an exported `SIF`, then
`$MODELS_DIR/Mistral-Large-4-1T-A52B-NVFP4/container/vllm-mistral-large-4.sif`, then the FP8 sibling's
copy of the same image, then the shared testbed.

A copy of this checkpoint needs the `container/` subdirectory alongside the weights, 8.2 GiB on top of
the 568 GiB.

## 3. Launch

```
sbatch --account=<your-account> recipes/Mistral-Large-4-1T-A52B-NVFP4/rtx-8/serve.sbatch
```

On a node you already hold:

```
bash recipes/Mistral-Large-4-1T-A52B-NVFP4/rtx-8/serve_ssh.sh <node>
```

Startup is about 8 minutes warm. The log is `$ML4N_LOG_DIR/vllm-mistral-nvfp4.log`, default under
`/tmp/$USER/mistral-large-4-nvfp4`.

## 4. Verify

```
KEY=$(cat secrets/Mistral-Large-4-1T-A52B-NVFP4-rtx-8.key 2>/dev/null || cat secrets/vllm_api_key)
NODE=<the node>

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

Without the reasoning parser the thinking stays inline in `content` as `[THINK]...[/THINK]`; this recipe
passes both parsers, so reasoning arrives in its own field.

## 5. Connect a client

```
export NODE=<the node>
source recipes/Mistral-Large-4-1T-A52B-NVFP4/rtx-8/client.env
claude
```

`client.env` exports the base URL, the key, the served model name, and
`CLAUDE_CODE_MAX_CONTEXT_TOKENS=98304`, matching what this recipe serves. Use `ANTHROPIC_AUTH_TOKEN`, not
`ANTHROPIC_API_KEY`.

## 6. Stop it

```
scancel <job id>
```

For an ssh launch, kill the `vllm serve` process on the node. Confirm the GPUs are released before
relaunching.

## Tunable inputs

| Variable | Default | Effect |
| --- | --- | --- |
| `MODEL` | `$MODELS_DIR/Mistral-Large-4-1T-A52B-NVFP4` | Serve a different copy of the checkpoint |
| `SIF` | resolved as in step 2 | Container image |
| `API_PORT` | 8000 | Listening port |
| `MAX_MODEL_LEN` | 98304 | Context window. Raising it needs `NO_SPEC=1`, see Known limits |
| `GPU_UTIL` | 0.92 | Fraction of VRAM for weights plus KV. 0.96 fails, see Known limits |
| `TP` | 8 | Tensor parallel size, the node's GPU count |
| `MAX_NUM_SEQS` | 128 | Sequence cap |
| `MAX_BATCHED_TOKENS` | 8192 | Batched prefill tokens |
| `NO_SPEC` | 0 | Set to 1 to serve without DSpark and raise the context ceiling |
| `SPEC_TOKENS` | 3 | DSpark draft depth, fixed at 3 by the checkpoint |
| `TOOL_PARSER` | `mistral` | Tool call parser |
| `REASONING_PARSER` | `mistral` | Reasoning parser |
| `EXTRA_ARGS` | unset | Extra flags appended to the `vllm serve` command line |
| `KEY_NAME`, `KEY_FILE`, `VLLM_API_KEY` | this recipe's key | Which key `common/lib/api_key.sh` resolves |

## Benchmarking

Conditions:

| | |
| --- | --- |
| Protocol | slope(128,1152), 3 repeats per level, median reported |
| Input length | ISL 23 tokens. Rates at a long input are not measured; use `--prompt-tokens` |
| Output length | OSL 1152 tokens, output only, `ignore_eos` |
| Context | `MAX_MODEL_LEN=98304`, `GPU_UTIL=0.92`, DSpark at depth 3 |
| Allocation for the measurement | 1 node, 8 GPUs, 128 cores and 1440 GB, `kempner_rtx` |
| Sequence cap | swept to 48 to match the FP8 sibling, so the two tables compare directly |
| Endpoint | idle; the benchmark client ran on another node |

Results:

| Concurrency | Aggregate | Per stream | Spread over 3 runs |
| --- | --- | --- | --- |
| 1 | 129.7 tok/s | 129.7 tok/s | 123.7 to 132.8 |
| 8 | 381.8 tok/s | 47.7 tok/s | 377.3 to 385.1 |
| 32 | 727.4 tok/s | 22.7 tok/s | 721.5 to 730.8 |
| 48 | 825.9 tok/s | 17.2 tok/s | 825.4 to 829.6 |

| | |
| --- | --- |
| Engine facts | KV cache 123,391 tokens, 1.26 full-length requests at once |
| Label | rising. The curve is still climbing at 48; the sweep stops there only to match the FP8 sibling |

## Choosing this recipe or the four-node FP8 one

One RTX node is about twice as fast for a single caller and uses a quarter of the hardware; four H200
nodes serve a longer context and more aggregate throughput. Both were measured with the same protocol.

| | NVFP4, 1 RTX node | FP8, 4 H200 nodes |
| --- | --- | --- |
| One caller | 129.7 tok/s | 67.1 tok/s |
| Concurrency 8 | 381.8 tok/s | 395.4 tok/s |
| Concurrency 32 | 727.4 tok/s | 1117.3 tok/s |
| Concurrency 48 | 825.9 tok/s | 1517.5 tok/s |
| Context | 98304 | 262144 |
| Engine | vLLM | SGLang |

The crossover is between concurrency 1 and 8: below it the single node wins on latency, above it the
four-node deployment wins on throughput and serves a 2.7x longer context.

## Known limits

- **SGLang cannot serve this checkpoint on sm120.** Three attempts, each refused differently: the
  default backends die with `size of tensor a (128) must match the size of tensor b (16)`, where 128 is
  `head_dim` and 16 is the NVFP4 `group_size`; `--attention-backend flashinfer` is rejected because this
  model's NoPE, SSMax and KV-sharing attention is implemented only by the `dsa` backend; and `dsa` itself
  fails with `TllmGenFmhaRunner ... Unsupported architecture`, because it routes into TRT-LLM-gen FMHA
  which ships no sm120 kernels. vLLM instead selects `FLASHINFER_MLA_SPARSE_SM120` and works. Neither
  vendor claims sm120 for this checkpoint: vLLM lists `[gb300, b300, b200]` and SGLang says
  "Blackwell only: 4x GB300".
- **Context is capped by memory, not by the checkpoint.** All three edges were measured at TP8:
  262144 without DSpark needs 11.51 GiB of KV against 8.17 available and fails; 131072 without DSpark
  works and gives a 246,912-token pool; 131072 with DSpark needs 7.05 GiB against 6.62 and fails, because
  the draft costs about 1.5 GiB per GPU; 98304 with DSpark works and is what ships. Set `NO_SPEC=1` to
  trade drafting for a 131072 context.
- `GPU_UTIL` above about 0.92 is not usable: 0.96 runs out of memory inside
  `compile_or_warm_up_model` with `Cuda failure 'out of memory'`.
- This recipe does not set `NCCL_P2P_DISABLE=1`, which the venv-based RTX recipes here do. The engine
  disables the P2P-dependent collectives itself on these GPUs, reporting that custom allreduce is not
  supported on more than two PCIe-only GPUs, and TP8 ran without it. Setting it anyway cost about
  14 percent at the top concurrency level on the Muse NVFP4 recipe, so it is left off here. If NCCL
  init ever hangs on a node, set it and expect that cost.
- Throughput is still rising at concurrency 48. The sweep stops there to match the FP8 sibling, so
  825.9 tok/s is a floor rather than a plateau.
- Decode rate at a long input is not measured. Every figure here uses a 23-token input.
- The vision encoder is loaded but not exercised. This recipe is validated for text only.
