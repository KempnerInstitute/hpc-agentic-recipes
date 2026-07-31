# Qwen3-Coder-480B-A35B-Instruct, bf16: never served

Status: Untested - staged and supported, never launched, no hardware variant authored yet

This is a documentation-only entry. There is no hardware subdirectory and there are no scripts, because
this checkpoint has never been served here. It is kept because it answers a question the FP8 twin
cannot, and this page records what it is, why it is interesting, and what shape it would need.

The FP8 version of this model is the one in production use: it serves at 67.7 tok/s single stream at TP4 x PP2 on
one RTX node. This bf16 version is twice the size and has no measured rate at all.

## What is staged

- Checkpoint directory: `Qwen3-Coder-480B-A35B-Instruct`
- Hugging Face repo: `Qwen/Qwen3-Coder-480B-A35B-Instruct`
- Testbed path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Qwen3-Coder-480B-A35B-Instruct`
- On disk: 894.4 GiB, 960.3 GB decimal, 241 shards, 30321 tensors

Read from `config.json` on 2026-07-29:

| Property | Value |
| --- | --- |
| Architecture | `Qwen3MoeForCausalLM`, `model_type` `qwen3_moe` |
| Precision | bf16, no `quantization_config` at all |
| Layers | 62 |
| Experts | 160, 8 active per token, `decoder_sparse_step` 1, no shared expert |
| `moe_intermediate_size` | 2560 |
| Attention | hidden 6144, 96 heads, 8 KV heads, `head_dim` 128, QK norm on |
| Context | `max_position_embeddings` 262144, `rope_theta` 1e7, no rope scaling |

It ships its own `chat_template.jinja` and a `qwen3coder_tool_parser.py`, and the architecture is
supported by vLLM 0.25.1 today, so nothing about the software side is blocked. The only reason this has
not been served is that it costs twice the VRAM of the FP8 twin for what is expected to be the same
quality.

## Why this checkpoint is worth keeping

**It is the only way to find out whether Coder-480B can serve on H200 at all.** The FP8 twin cannot:
four configurations were tried on H200 and all four failed during CUDA graph capture, and the failure
was traced to the CUTLASS w8a8 FP8 GEMM path faulting on Hopper for that checkpoint. A bf16 checkpoint
never touches that path, because there is no FP8 GEMM to call. So if Coder-480B is wanted on Hopper,
this is the version to try, and a successful bring-up here would confirm that the FP8 failure is a
kernel problem rather than anything about the model.

**The TP8 limit that blocks the FP8 twin does not apply here.** The FP8 checkpoint cannot run at TP8
because its `moe_intermediate_size` of 2560 divided by 8 gives 320, which is not a multiple of the
128-element FP8 quantization block, so vLLM refuses to start and the working shape becomes TP4 x PP2.
This checkpoint has no `quantization_config` and therefore no quantization block, so nothing has to
divide by 128 and 2560/8 = 320 is a perfectly ordinary shard. TP8 across a full node is legal here.
That is a real difference in shape, not a footnote: on an 8-GPU node this model can be one
tensor-parallel group instead of two pipeline stages.

## Sizing, if it is brought up

| Target | GPUs | Raw VRAM | Usable at 0.90 | Verdict |
| --- | --- | --- | --- | --- |
| 1 RTX PRO 6000 node | 8 at 97887 MiB | about 765 GiB | about 689 GiB | does not fit, 894 GiB of weights |
| 2 RTX PRO 6000 nodes | 16 at 97887 MiB | about 1530 GiB | about 1377 GiB | comfortable |
| 2 H200 nodes | 8 at 143771 MiB | about 1123 GiB | about 1011 GiB | fits, tight for KV |
| 4 H200 nodes | 16 at 143771 MiB | about 2246 GiB | about 2021 GiB | comfortable |

The tightness on two H200 nodes is about KV cache, not weights. This model pays full-attention KV on
all 62 layers with 8 KV heads, about 248 KiB per token, so a 128K context costs about 31 GiB and the
full 256K context about 62 GiB. Against 1011 GiB usable and 894 GiB of weights that leaves roughly 117
GiB, which covers 256K with some margin but not much room for anything unexpected. `--kv-cache-dtype
fp8` halves the KV figure and is the obvious first lever if it turns out to be too close.

Four H200 nodes is comfortable but expensive to schedule. Two RTX nodes fit easily, but that spends two
Blackwell nodes on a bf16 model whose FP8 twin already runs on one of them, which is the wrong trade
unless the point is specifically to compare precisions.

## What a real recipe here would have to settle

None of these has been tested, and each would need to be answered before this became a validated
recipe:

- Whether it starts on 2 H200 nodes at TP4 x PP2 with CUDA graphs, which is the question the FP8 twin
  fails and this one might pass.
- Whether TP8 on a single 8-GPU node plus PP across nodes beats TP4 x PP2, now that TP8 is legal.
- Whether the quality difference against the FP8 twin is measurable at all. If it is not, this
  checkpoint stays an experiment rather than a production endpoint.
- Cross-node topology, since every shape above needs two nodes: tensor parallelism inside a node and
  pipeline parallelism between nodes, Ray as the executor, and no speculative decoding, which this
  checkpoint does not offer anyway since it ships no draft model.

Until somebody answers those, this page is the honest state of it: staged, supported, and never run.
