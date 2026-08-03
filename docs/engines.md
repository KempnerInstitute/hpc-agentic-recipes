# vLLM and SGLang

Both engines are approved here. They differ in which API they serve and which checkpoints they can load, so
the choice follows the model rather than a preference.

## Pick by what the model needs

**Use vLLM when it supports the checkpoint.** It is the better-trodden path here: fifteen of the sixteen
serve scripts run it, so its failure modes are the ones this repo documents. Both engines serve an
OpenAI-compatible `/v1` and an Anthropic-compatible `/v1/messages`, so the client story is the same
either way.

**Use SGLang when vLLM cannot load the model.** Kimi-K3 is that case: no vLLM release available here
implements `KimiK3ForConditionalGeneration`, while SGLang serves it from a container, which is what
`recipes/Kimi-K3/h200-4-nodes4-sglang` runs. Measured on 4 H200 nodes it gives 40.2 tok/s single stream by
default, up to 94.1 with the DSpark draft and the wider pool, and 1442.6 tok/s aggregate at concurrency 156,
holding 38.9 tok/s at an input of 115,292 tokens. SGLang serves an Anthropic-compatible `/v1/messages` as
well as the OpenAI `/v1`, so Claude Code reaches it directly, provided the launcher passes a tool call
parser. Without one, tool calls arrive as raw text the client cannot execute.

SGLang is also the only way to combine speculative decoding with a model that must span nodes. vLLM rejects
a speculative config whenever pipeline parallelism is in use, so a checkpoint needing PP to fit loses its MTP
or draft-model speedup. SGLang can run TP8 across two nodes with EAGLE speculative decoding instead. The
GLM-5.2 SGLang recipe here attempts exactly that and is marked Blocked: every rank raises a shape assertion
inside the DeepSeek weight loader, and the recipe records the traceback.

The two coexist. Each recipe builds its own environment, so an SGLang environment never disturbs a vLLM one.

## Authentication

vLLM accepts `Authorization: Bearer <key>` only. It ignores the `x-api-key` header that Anthropic's own API
accepts, which is the most common configuration mistake here: setting `ANTHROPIC_API_KEY` instead of
`ANTHROPIC_AUTH_TOKEN` makes the client send `x-api-key`, and every request returns 401.

The vLLM recipes pass `--api-key`. The SGLang recipe passes none, so its port accepts any request that can
reach it. Do not run it on a shared network.

## Hosted tools work on neither engine

Anthropic's server-side tools (`web_search_20250305`, `web_fetch_20250910`, `code_execution_20250522`) are
executed by Anthropic's own API rather than by the model, so neither engine can run them. Their definitions
carry no `input_schema`, and vLLM rejects all three with HTTP 400 on that basis. SGLang instead accepts
`web_search_20250305` with HTTP 200 and drops the tool, which is harder to diagnose because the model simply
answers without searching, and rejects the other two with 400. Client-side tools work normally on both,
which covers most of what an agent needs. [web-search.md](web-search.md) has a keyless replacement for
search.
