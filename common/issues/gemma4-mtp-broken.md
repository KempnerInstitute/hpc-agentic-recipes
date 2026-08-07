**Speculative decoding failed for Gemma 4 26B on the engines tested here.** The drafter checkpoints
(`*-it-assistant`) are wired through `SPEC_DRAFT`. With the 26B drafter, `gemma4_mtp` failed on vLLM 0.25.1
and 0.26.0 with a shape mismatch:

```
a and b must have same reduction dim, but got [s47, 3840] X [5632, 1024]
```

The failure kills the engine during startup, so the server never comes up. Retested on vLLM 0.25.1 with
the same result.

The 31B drafter is not affected: it works on vLLM 0.25.1 and on a 0.26.1 dev build, measured 2.6 to 2.7
times faster than without it on all three cards.
