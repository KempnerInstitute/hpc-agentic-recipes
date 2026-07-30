**Speculative decoding is unavailable for Gemma 4 on this engine.** The drafter checkpoints
(`*-it-assistant`) are downloaded and wired through `SPEC_DRAFT`, but the `gemma4_mtp` method is
broken in both vLLM 0.25.1 and 0.26.0, failing with a shape mismatch:

```
a and b must have same reduction dim, but got [s47, 3840] X [5632, 1024]
```

The drafter expects twice 2816 rather than 5632. The fix (`_maybe_share_embeddings`) exists only on
vLLM's `main` branch, so this needs a build newer than 0.26.0. Do not enable `SPEC_DRAFT` on 0.25.1
or 0.26.0; it fails at the first request.
