# Qwen3.8-2.4T-A95B-FP8

A 2.4T parameter hybrid Mixture-of-Experts from Qwen with 95B active per token: 92 layers, 512
experts, hidden size 8192. It serves as `qwen3.8-2.4t`, and at 2324.7 GiB it is the largest checkpoint
in this repo.

## Hardware variants

| Variant | Engine | Shape | Single stream | Aggregate | Status |
| --- | --- | --- | --- | --- | --- |
| [`h200-4-nodes6`](h200-4-nodes6/README.md) | vLLM | 6 nodes, 24 H200 GPUs, TP4 EP1 PP6 | 52.6 tok/s | 534.7 at c=32, rising | Validated |

52.6 tok/s on a single stream is the fastest H200 figure in the repo, which is what a 95B active MoE
buys: only a fraction of the 2.4T parameters runs per token.

Six nodes is the minimum. Sixteen GPUs give 2246 GiB against 2324.7 GiB of weights, so four nodes
cannot load the model at all, and five leave almost nothing for the KV cache. Upstream sizes this
checkpoint at 32 H200 GPUs; on 24 the engine still reported a 9,294,877 token KV pool at the full
262144 context.

The BF16 sibling, `Qwen3.8-2.4T-A95B` at 4556.4 GiB, does not fit this cluster and has no recipe.

## Checkpoint provenance

- Checkpoint directory: `Qwen3.8-2.4T-A95B-FP8`
- Hugging Face repo: `Qwen/Qwen3.8-2.4T-A95B-FP8`
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Qwen3.8-2.4T-A95B-FP8`
- Size: 2324.7 GiB across 213 shards, FP8
- Architecture: `Qwen3_5MoeForCausalLM`, declares a 262144 context

There is no container for this recipe. The engine is a venv with vLLM 0.27.1 and transformers 5.14.1,
built by `env/build.sh` on shared scratch so every node imports the same interpreter.
