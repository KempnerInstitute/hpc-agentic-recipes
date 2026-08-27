# OpenAI-compatible clients

Every endpoint here serves an OpenAI-compatible API at `/v1` alongside the Anthropic-compatible one, so
tools that speak either protocol work without a proxy. That holds under both engines, so a client from this
page reaches the same endpoints Claude Code does.

Three settings, whatever the client:

| Setting | Value |
| --- | --- |
| Base URL | `http://<node>:8000/v1` |
| API key | the contents of the recipe's key file, or `secrets/vllm_api_key` |
| Model | the served model name, for example `glm-5.2` |

Confirm the model name from the endpoint rather than guessing:

```
KEY=$(cat secrets/GLM-5.2-NVFP4-rtx-8.key 2>/dev/null || cat secrets/vllm_api_key)
curl -s -H "Authorization: Bearer $KEY" http://<node>:8000/v1/models
```

Any library that accepts a custom base URL works, including the official `openai` Python package:

```
from openai import OpenAI
key = open("secrets/GLM-5.2-NVFP4-rtx-8.key").read().strip()
client = OpenAI(base_url="http://<node>:8000/v1", api_key=key)
```

Thinking models return their reasoning in a separate field rather than inside `content`: vLLM calls it
`reasoning`, SGLang calls it `reasoning_content`. If `content` comes back empty and `finish_reason` is
`length`, raise `max_tokens` to at least 400, because a small budget is spent entirely on reasoning before
any answer is emitted.

## Codex

Install it with `curl -fsSL https://chatgpt.com/codex/install.sh | sh`. Codex needs two settings beyond
the three above. In `~/.codex/config.toml`:

```
model = "glm-5.2"
model_provider = "cluster"

[model_providers.cluster]
name = "cluster"
base_url = "http://<node>:8000/v1"
env_key = "CLUSTER_API_KEY"
wire_api = "responses"

[sandbox_workspace_write]
network_access = true
```

`wire_api = "responses"` is required. Recent versions dropped the chat protocol, so `/v1/responses` is the
only route they speak, and it is served alongside `/v1/chat/completions` on both engines.

`network_access = true` matters only if you want the model to search. Codex sandboxes the commands it runs
and blocks outbound network by default, so `search.sh` fails there while working fine in your own shell,
and Codex reports that web access is disabled, which reads like a missing tool rather than a sandbox
setting. It is a real relaxation of the sandbox, not a feature flag: every command Codex runs gains network
access, the same latitude Claude Code already has through Bash.

Export the key named by `env_key`, then run `codex`. Two more things to expect. `codex exec` reads standard
input, so redirect it from `/dev/null` for a non-interactive run or it waits with no output. And Codex warns
that it has no metadata for the model name; it prints that for any custom provider, no setting suppresses
it, and it does not stop anything working. One consequence is real: with fallback metadata Codex does not
know the context limit, so it will not manage the window for you on a short-context endpoint.
