**Pipeline parallelism disables speculative decoding.** vLLM rejects a speculative config when
pipeline parallelism is in use, so no MTP or draft-model speedup is available in any recipe that needs
PP to span nodes. A checkpoint that ships an MTP head cannot use it in this configuration. This was
observed when configuring GLM-5.2 across two H200 nodes, and it is the reason the SGLang recipe exists
at all, since SGLang can run TP8 across two nodes with EAGLE instead. Note that the guard was not
located in vLLM 0.25.1's config source, so treat it as observed behavior for this version rather than a
documented API contract, and re-check after an engine upgrade.
