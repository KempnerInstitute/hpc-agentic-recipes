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

**Use `ANTHROPIC_AUTH_TOKEN`, not `ANTHROPIC_API_KEY`.** Both engines accept only
`Authorization: Bearer`. `ANTHROPIC_API_KEY` makes Claude Code send `x-api-key` instead, which the
engine ignores, so every request returns 401.

**Set `ANTHROPIC_SMALL_FAST_MODEL`.** Without it the client reaches for a hosted Haiku model that your
local endpoint does not serve.

**Web search will fail.** Claude Code's built-in web search sends a tool definition with no
`input_schema`, and both engines reject it with HTTP 400. Client-side tools such as file editing and shell
commands work normally. See [web-search.md](web-search.md) for the keyless replacement.

You will also see a warning that claude.ai connectors are disabled because another auth source is set.
That is expected when pointing at a local endpoint and affects nothing.

## On a shared endpoint, drop the client's attribution block

```
export CLAUDE_CODE_ATTRIBUTION_HEADER=0
```

By default Claude Code prepends a system block reading
`x-anthropic-billing-header: cc_version=...; cc_entrypoint=...;` ahead of its own system prompt. Despite
the name it is not sent as a header. It is about 74 characters at the very front of the prompt, and a
prefix cache hit requires the prompt to match from the first token, so those characters decide whether
the much larger remainder, the client's own system prompt and every tool definition, can be reused at
all. `cc_version` changes when the client updates, and `cc_entrypoint` differs between an interactive
session and `claude -p`, so two callers that differ in either share no prefix and each pays its own
prefill. Setting this to 0 removes the block, so every caller starts identically and shares one cache.

It is constant within a session, so it changes nothing about reuse between turns of one conversation.
The gain is across callers and across client upgrades, which is what matters on an endpoint several
people share. Every recipe's `client.env` sets it.

## Verify before blaming the client

```
KEY=<the api key>
curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $KEY" http://<node>:8000/v1/models
curl -s -o /dev/null -w '%{http_code}\n' http://<node>:8000/v1/models
```

The first should print 200 and the second 401. If the first prints 000 the server is not listening, and
if it prints 401 the key is wrong.
