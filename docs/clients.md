# OpenAI-compatible clients

Every vLLM recipe here also serves an OpenAI-compatible API at `/v1`, so tools that speak that protocol
work without a proxy. SGLang serves only this API and no Anthropic-compatible endpoint, so an
SGLang-served model such as Kimi-K3 needs a client from this page rather than Claude Code.

Three settings, whatever the client:

| Setting | Value |
| --- | --- |
| Base URL | `http://<node>:8000/v1` |
| API key | the contents of `secrets/vllm_api_key` |
| Model | the served model name, for example `glm-5.2` |

Confirm the model name from the endpoint rather than guessing:

```
curl -s -H "Authorization: Bearer $(cat secrets/vllm_api_key)" http://<node>:8000/v1/models
```

Cline, Aider, Continue, and OpenHands all take these three settings. Any library that accepts a custom base
URL works too, including the official `openai` Python package:

```
from openai import OpenAI
client = OpenAI(base_url="http://<node>:8000/v1", api_key=open("secrets/vllm_api_key").read().strip())
```

Thinking models return their reasoning in a separate field rather than inside `content`: vLLM calls it
`reasoning`, SGLang calls it `reasoning_content`. If `content` comes back empty and `finish_reason` is
`length`, raise `max_tokens` to at least 400, because a small budget is spent entirely on reasoning before
any answer is emitted.
