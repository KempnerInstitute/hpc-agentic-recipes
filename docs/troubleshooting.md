# Troubleshooting

Aggregated view of failure modes across models, for browsing and for spotting patterns. It is not the home
of any of them: every recipe repeats in full the issues that affect it, so if you are working on one model
you do not need this page.

The source text for each issue lives in `common/issues/`, and the recipes embed copies of it.
`common/tools/audit_recipes.sh` checks that every copy still matches its source and fails if one has
drifted.

| Symptom | Cause | Affects |
| --- | --- | --- |
| Two unrelated endpoints die within seconds of each other | A storage stall froze every rank, and PyTorch's NCCL heartbeat monitor killed each process on the assumption a collective hung | every multi-rank recipe |
| Endpoint dies about 8 minutes after a filesystem hiccup | `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` defaults to 480 seconds | every multi-rank recipe |
| Server freezes with no error during normal operation | Server log was on a network filesystem, so a blocked write stalled the process | any recipe with `LOG_DIR` off node-local disk |
| Launch produces no log output for many minutes and looks hung | Weight load, compile, and graph capture routinely exceed vLLM's 600 second readiness default, so the recipes raise it to 3600 | every vLLM recipe |
| vLLM's default wheel will not run on an H200 or H100 node | Those nodes run driver 575, CUDA 12.9, and the default wheel is built for CUDA 13 | every H200 and H100 recipe |
| Multi-node launch fails with a stale file handle during compilation | Triton and Inductor caches sat on a shared network home and raced across ranks | every multi-node recipe |
| `output_size of gate's and up's weight = 320 is not divisible by weight quantization block_n = 128` | TP does not divide `moe_intermediate_size` into multiples of the FP8 block | Qwen3-Coder-480B-FP8 at TP8 |
| Illegal memory access during CUDA graph capture on H200 | CUTLASS w8a8 FP8 GEMM faults on Hopper for some checkpoints | Qwen3-Coder-480B-FP8 on H200, GLM-4.6, GLM-5.2-FP8 |
| Illegal memory access with DeepGEMM enabled on H200 | DeepGEMM MoE path faults on sparse attention, reproduced independently for a second model | every H200 recipe |
| Startup hangs at 0 percent GPU utilization after weights load | Cross-node multimodal profiling deadlocks | multimodal models on more than one node |
| NCCL initialization hangs with no error on an RTX node | No NVLink, so peer-to-peer must be disabled | every multi-GPU RTX recipe |
| Expert parallelism makes throughput worse rather than better | All-to-all traffic added on top of an interconnect that is already the limit | every full-RTX-node recipe |
| First inference request fails on a `kv_scale_format` argument | flashinfer-python 0.6.13 does not accept it; 0.6.15 is required | every RTX recipe |
| `import torch` fails after adding the CUDA toolkit to the library path | The toolkit's `libcudart` shadows torch's runtime | every RTX recipe |
| `a and b must have same reduction dim, but got [s47, 3840] X [5632, 1024]` | `gemma4_mtp` is broken in vLLM 0.25.1 and 0.26.0 | both Gemma models with `SPEC_DRAFT` |
| HTTP 400 mentioning `input_schema` and `web_search_20250305` | Anthropic hosted tools carry no input schema and vLLM rejects them | every vLLM recipe |
| Every request returns 401 with an apparently correct key | Client sent `x-api-key` because `ANTHROPIC_API_KEY` was set instead of `ANTHROPIC_AUTH_TOKEN` | every vLLM recipe |
| Response has empty `content` and `stop_reason: max_tokens` | Thinking model spent the whole budget on the `reasoning` field | every thinking model |
| Cross-node tensor parallelism hangs at NCCL init | TP must stay inside a node; use PP across nodes | every multi-node recipe |
| Speculative decoding rejected at startup | vLLM does not allow it with pipeline parallelism | every multi-node vLLM recipe |
| `assert param.size() == loaded_weight.size()` during weight loading | Shape assertion in the DeepSeek weight loader; this recipe has never loaded weights | GLM-5.2-FP8 on SGLang |
| An endpoint serves requests that carry no API key | The SGLang launcher passes no `--api-key` | GLM-5.2-FP8 on SGLang |
| Relaunch fails with out of memory right after stopping a server | The old process had not released the KV cache yet | any recipe, use `common/tools/stop.sh` which waits |
| A sourced setup script exits silently with no output | A sourced file returning non-zero aborts a caller running under `set -e` | fixed in `common/defaults.sh`, checked by the audit |
