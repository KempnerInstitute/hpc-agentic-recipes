# Quickstart: connect to an endpoint someone else is serving

Using an endpoint someone else runs needs no environment build and no queue wait.

Get three things from whoever is serving it: the node name, the port (8000 unless they say otherwise), and
the API key.

```
export ANTHROPIC_BASE_URL=http://<node>:8000
export ANTHROPIC_AUTH_TOKEN=<the api key>
export ANTHROPIC_MODEL=<served model name>
export ANTHROPIC_SMALL_FAST_MODEL=<the same name>
export CLAUDE_CODE_ATTRIBUTION_HEADER=0
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

**Use `ANTHROPIC_AUTH_TOKEN`, not `ANTHROPIC_API_KEY`.** Both engines accept only
`Authorization: Bearer`. `ANTHROPIC_API_KEY` makes Claude Code send `x-api-key` instead, which the
engine ignores, so every request returns 401.

**Set `ANTHROPIC_SMALL_FAST_MODEL`.** Without it the client reaches for a hosted Haiku model that your
local endpoint does not serve.

**Web search will not work.** Claude Code's built-in web search sends a tool definition with no
`input_schema`. vLLM rejects it with HTTP 400. SGLang returns 200 and silently drops the tool, so the
model answers without searching and nothing says why. Client-side tools such as file editing and shell
commands work normally. See [web-search.md](web-search.md) for the keyless replacement.

You will also see a warning that claude.ai connectors are disabled because another auth source is set.
That is expected when pointing at a local endpoint and affects nothing.

## On a shared endpoint, drop the client's attribution block

```
export CLAUDE_CODE_ATTRIBUTION_HEADER=0
```

By default Claude Code puts a line reading `x-anthropic-billing-header: cc_version=...; cc_entrypoint=...;`
at the very start of the prompt, ahead of its own system prompt. Despite the name it is not an HTTP header,
just text, and setting this to 0 removes it.

A prefix cache hit needs the prompt to match from the first token, so that line gates reuse of everything
behind it: the client's system prompt and every tool definition. Because it carries the client version,
callers on different versions match on nothing, and a self-updating client drifts into that state on its
own. Callers on the same version send an identical line, so between them this changes nothing, and the same
holds turn to turn within one conversation. The cross-version case is the reason to set it, and the one case
not tested here, since only one client version was installed.

## A small context needs the output request capped

Claude Code's own system prompt and tool definitions run to about 24K tokens before you type anything, and
it asks for 32000 output tokens by default. Against an endpoint serving 32K or 40K that exceeds the context
and every request fails with a maximum context length error. Cap the request:

```
export CLAUDE_CODE_MAX_OUTPUT_TOKENS=4096
```

The recipes that serve a small context set this in their `client.env` already. The alternative is to raise
the endpoint's own context: every one of those checkpoints supports far more than the recipe serves by
default, at the cost of KV cache.

## Verify before blaming the client

```
KEY=<the api key>
curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $KEY" http://<node>:8000/v1/models
curl -s -o /dev/null -w '%{http_code}\n' http://<node>:8000/v1/models
```

The first should print 200 and the second 401. If the first prints 000 the server is not listening, and
if it prints 401 the key is wrong.
