# vLLM and SGLang

Both engines are approved here. They differ in which API they serve and which checkpoints they can load, so
the choice follows the model rather than a preference.

## Pick by what the model needs

**Use vLLM when it supports the checkpoint.** It is the better-trodden path here: nineteen of the twenty-one
serve scripts run it, so its failure modes are the ones this repo documents. Both engines serve an
OpenAI-compatible `/v1` and an Anthropic-compatible `/v1/messages`, so the client story is the same
either way.

**Use SGLang when vLLM cannot load or run the model.** Kimi-K3 is that case: no vLLM release available here
implements `KimiK3ForConditionalGeneration`, while SGLang serves it from a container, which is what
`recipes/Kimi-K3/h200-4-nodes4-sglang` runs. Measured on 4 H200 nodes it gives 40.3 tok/s single stream by
default, up to 94.1 with the DSpark draft and the wider pool, and 1442.6 tok/s aggregate at concurrency 156,
holding 38.9 tok/s at an input of 115,292 tokens. SGLang serves an Anthropic-compatible `/v1/messages` as
well as the OpenAI `/v1`, so Claude Code reaches it directly, provided the launcher passes a tool call
parser. Without one, tool calls arrive as raw text the client cannot execute.

Muse Spark Open NVFP4 is a second case, and a different one. vLLM implements the architecture, but for
this checkpoint's packed NVFP4 experts it selects a FlashInfer TRT-LLM MoE backend and then refuses it on
sm120, where that kernel family ships no cubins. SGLang serves the same checkpoint on one RTX node
through its Marlin MoE path, which `recipes/Muse-Spark-Open-816B-A42B-NVFP4/rtx-8-sglang` runs.

Whether speculative decoding survives pipeline parallelism depends on the vLLM version. Older releases
refused a speculative config whenever PP was in use, which is why GLM-5.2-FP8 on two H200 nodes serves
without it even though its checkpoint ships an MTP head. vLLM 0.32.0 refuses only a short list of methods,
and MTP is not among them: GLM-5.3 runs MTP at 5 draft tokens across two H200 nodes at TP4 x PP2, worth
1.89x on a single stream. Check the engine version before assuming either way.

The two coexist. Each recipe builds its own environment, so an SGLang environment never disturbs a vLLM one.

## Authentication

Both engines accept `Authorization: Bearer <key>` only. They ignore the `x-api-key` header that Anthropic's
own API accepts, which is the most common configuration mistake here: setting `ANTHROPIC_API_KEY` instead of
`ANTHROPIC_AUTH_TOKEN` makes the client send `x-api-key`, and every request returns 401.

Every recipe gates its endpoint, and none of them puts the key on a command line, where any user on the node
could read it out of `/proc`. vLLM reads `VLLM_API_KEY` from the environment. SGLang accepts a key only as
an argument, so that recipe starts through `common/tools/sglang_launch.py`, which supplies it after exec.

## Hosted tools work on neither engine

Anthropic's server-side tools (`web_search_20250305`, `web_fetch_20250910`, `code_execution_20250522`) are
executed by Anthropic's own API rather than by the model, so neither engine can run them. Their definitions
carry no `input_schema`, and vLLM rejects all three with HTTP 400 on that basis. SGLang instead accepts
`web_search_20250305` with HTTP 200 and drops the tool, which is harder to diagnose because the model simply
answers without searching, and rejects the other two with 400. Client-side tools work normally on both,
which covers most of what an agent needs. [web-search.md](web-search.md) has a keyless replacement for
search.
