# Quickstart: connect to an endpoint someone else is serving

Using an endpoint someone else runs needs no environment build and no queue wait.

Get three things from whoever is serving it: the node name, the port (8000 unless they say otherwise), and
the API key.

```
export ANTHROPIC_BASE_URL=http://<node>:8000
export ANTHROPIC_AUTH_TOKEN=<the api key>
export ANTHROPIC_MODEL=<served model name>
export ANTHROPIC_SMALL_FAST_MODEL=<the same name>
claude
```

Ask the endpoint for the served model name rather than guessing it, since it differs from the checkpoint
directory name:

```
curl -s -H "Authorization: Bearer $ANTHROPIC_AUTH_TOKEN" http://<node>:8000/v1/models
```

The variables above work against either engine here: vLLM and SGLang both serve an Anthropic-compatible
API. For an OpenAI-compatible client instead, see [clients.md](clients.md).

## Three common mistakes

**Use `ANTHROPIC_AUTH_TOKEN`, not `ANTHROPIC_API_KEY`.** vLLM accepts only `Authorization: Bearer`.
`ANTHROPIC_API_KEY` makes Claude Code send `x-api-key` instead, which vLLM ignores, so every request
returns 401.

**Set `ANTHROPIC_SMALL_FAST_MODEL`.** Without it the client reaches for a hosted Haiku model that your
local endpoint does not serve.

**Web search will fail.** Claude Code's built-in web search sends a tool definition with no
`input_schema`, and vLLM rejects it with HTTP 400. Client-side tools such as file editing and shell
commands work normally. See [web-search.md](web-search.md) for the keyless replacement.

You will also see a warning that claude.ai connectors are disabled because another auth source is set.
That is expected when pointing at a local endpoint and affects nothing.

## Verify before blaming the client

```
KEY=<the api key>
curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $KEY" http://<node>:8000/v1/models
curl -s -o /dev/null -w '%{http_code}\n' http://<node>:8000/v1/models
```

The first should print 200 and the second 401. If the first prints 000 the server is not listening, and
if it prints 401 the key is wrong.
