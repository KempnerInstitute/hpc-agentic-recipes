# vLLM and SGLang

Both engines are covered here. They are not interchangeable for this use case.

## vLLM is the default

vLLM serves two APIs at once: an OpenAI-compatible `/v1` and an Anthropic-compatible `/v1/messages`.
The second is why Claude Code connects directly with no proxy and no translation layer. Every validated
recipe in this repo uses vLLM.

Authentication is `Authorization: Bearer <key>` only. vLLM does not read the `x-api-key` header that
Anthropic's own API accepts, which is the single most common configuration mistake here.

Anthropic's server-side tools (`web_search_20250305`, `web_fetch_20250910`, `code_execution_20250522`)
do not work against vLLM: those definitions carry no `input_schema`, and vLLM's validation rejects them
with HTTP 400. Client-side tools work normally, which is most of what agentic coding needs.

## SGLang is OpenAI-only

SGLang exposes no Anthropic-compatible endpoint, so Claude Code cannot use it without a translating
proxy. Use an OpenAI-compatible client instead, covered in [clients.md](clients.md).

The reason to care about SGLang at all is speculative decoding topology. GLM-5.2's MTP head needs the
whole model in one tensor-parallel group, and on two H200 nodes vLLM would need pipeline parallelism to
span them, which vLLM refuses to combine with speculative decoding. SGLang can run TP8 across two nodes
with EAGLE speculative decoding, so in principle it reaches a decode rate vLLM cannot on that hardware.

In practice our SGLang recipe has never successfully loaded weights: it fails with a shape assertion in
the DeepSeek weight loader. It is kept as a documented starting point, marked Blocked, with the exact
traceback recorded. It also does not gate its port with an API key, unlike every vLLM recipe here.

## Choosing

Use vLLM unless you specifically want to finish the SGLang work. If you do, the two engines coexist:
each recipe builds its own environment, so an SGLang environment never disturbs a vLLM one.
