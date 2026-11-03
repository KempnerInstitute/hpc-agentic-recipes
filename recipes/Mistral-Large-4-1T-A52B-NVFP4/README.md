# Mistral Large 4 1T-A52B NVFP4

The NVFP4 re-quantization of Mistral Large 4, published for Blackwell. At 568 GiB it fits one RTX PRO
6000 node, where the 988 GiB FP8 release needs four H200 nodes. It serves as `mistral-large-4`.

## Hardware variants

| Variant | Engine | Shape | Single stream | Aggregate | Status |
| --- | --- | --- | --- | --- | --- |
| [`rtx-8`](rtx-8/README.md) | vLLM | 1 RTX node, 8 GPUs, TP8 | 129.7 tok/s | 825.9 at c=48, rising | Validated |

One RTX node is about twice as fast for a single caller than four H200 nodes on FP8, 129.7 against 67.1,
helped by DSpark drafting and by having no cross-node hop. It serves a 98304 context against 262144,
because the memory budget is tighter, and it is overtaken on aggregate throughput from concurrency 8 up. See [Mistral-Large-4-1T-A52B](../Mistral-Large-4-1T-A52B/README.md).

**This build runs vLLM, while the FP8 sibling runs SGLang.** Each is the engine that works for that
shape: SGLang has no sm120 kernels for this model's mandatory `dsa` attention, and the vLLM image cannot
span nodes because it ships without Ray. Both recipes record the exact errors in their Known limits.

## Checkpoint provenance

- Checkpoint directory: `Mistral-Large-4-1T-A52B-NVFP4`
- Hugging Face repo: `mistralai/Mistral-Large-4-1T-A52B-NVFP4`
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Mistral-Large-4-1T-A52B-NVFP4`
- Size: 568 GiB across 32 shards, plus an 8.2 GiB container under `container/`
- Architecture: NVFP4 routed experts with block-FP8 attention and dense layers, ModelOpt mixed precision

This checkpoint ships in Mistral's own format rather than the Hugging Face one: it has no `config.json`,
instead carrying `params.json`, `consolidated.safetensors.index.json` and `tekken.json`. The recipe
therefore passes `--tokenizer-mode mistral --config-format mistral --load-format mistral`.

A DSpark drafter ships inside the checkpoint, so speculative decoding needs no separate download.
