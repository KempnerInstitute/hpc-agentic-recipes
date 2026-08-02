# vLLM and SGLang

Both engines are approved here. They differ in which API they serve and which checkpoints they can load, so
the choice follows the model rather than a preference.

## Pick by what the model needs

**Use vLLM when it supports the checkpoint.** vLLM serves two APIs at once, an OpenAI-compatible `/v1` and
an Anthropic-compatible `/v1/messages`. The second is why Claude Code connects with no proxy and no
translation layer. Fifteen serve scripts here run vLLM.

**Use SGLang when vLLM cannot load the model.** Kimi-K3 is that case: no vLLM release available here
implements `KimiK3ForConditionalGeneration`, while SGLang serves it. Measured on 4 H200 nodes it gives
40.3 tok/s single stream, 87.1 with the DSpark draft, and 1405 tok/s aggregate at concurrency 128, holding
38.9 tok/s at a 131,072-token prompt. SGLang exposes no Anthropic-compatible endpoint, so Claude Code needs
an OpenAI-compatible client instead, covered in [clients.md](clients.md).

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
executed by Anthropic's own API rather than by the model. Their definitions carry no `input_schema`, so
vLLM's validation rejects them with HTTP 400. Client-side tools work normally, which covers most of what
agentic coding needs. [web-search.md](web-search.md) has a keyless replacement for search.
