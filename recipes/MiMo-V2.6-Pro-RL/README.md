# MiMo-V2.6-Pro-RL

Xiaomi's flagship mixture-of-experts: 1.02T total parameters, 42B active per token, with a hybrid
sliding-window and global attention backbone and omnimodal encoders. It serves as `mimo-v2.6-pro`.

The checkpoint is FP8 compute with MXFP4-stored expert weights as published.

## Hardware variants

| Variant | Engine | Shape | Single stream | Aggregate | Status |
| --- | --- | --- | --- | --- | --- |
| [`rtx-8`](rtx-8/README.md) | vLLM | 1 RTX node, 8 GPUs, TP8 | 84.9 tok/s | 1136.0 at c=128, rising | Validated |

At 534 GiB the weights fit one RTX PRO 6000 node, which holds 765 GiB. They would also fit two H200
nodes by size, but this model needs its eight tensor-parallel ranks inside a single box: split across two
4-GPU nodes it either hangs or, under pipeline parallelism, serves silently wrong output. The recipe's
Known limits records both failures with their evidence.

Upstream verifies this model on H200, GB300 and MI355X, in every case as eight GPUs in one node.

## Checkpoint provenance

- Checkpoint directory: `MiMo-V2.6-Pro-RL`
- Hugging Face repo: `XiaomiMiMo/MiMo-V2.6-Pro-RL`
- Documented path: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/MiMo-V2.6-Pro-RL`
- Size: 534 GiB across 130 shards
- Architecture: `MiMoV2ForCausalLM`, needs `--trust-remote-code`, native to vLLM 0.31.0 and later

Read from the checkpoint: 70 layers, hidden size 6144, 8 key-value heads with `head_dim` 192, 8 experts
per token, `sliding_window` 128 and `max_position_embeddings` 1048576.

A DFlash drafter ships in a `dflash/` subdirectory and an audio tokenizer in `audio_tokenizer/`. Neither
is used by this recipe.

Upstream publishes `XiaomiMiMo/MiMo-V2.6-Pro-MOPD` as the default variant and recommends it over this RL
release for agentic workloads, citing tool-call repetition. That defect did not reproduce here; see the
recipe's Known limits.
