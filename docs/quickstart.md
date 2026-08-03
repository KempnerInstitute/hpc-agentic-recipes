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

By default Claude Code prepends a system block reading
`x-anthropic-billing-header: cc_version=...; cc_entrypoint=...;` ahead of its own system prompt. Despite
the name it is not sent as a header: it is text, 70 characters in an interactive session and 74 under
`claude -p`, at the very front of the prompt. A prefix cache hit requires the prompt to match from the
first token, so those characters gate reuse of the much larger remainder, the client's own system prompt
and every tool definition. Setting this to 0 removes that block and changes nothing else in the request.

What it is worth, measured rather than assumed. Two callers on the same client version and the same
entrypoint send a byte-identical block, so between them this setting gains nothing. The block carries the
client version, so callers on different versions differ at the first character, and since the client
updates itself a shared endpoint drifts into that state on its own. That is the case worth setting it
for, and it is the one not verified here, because only one client version was available to test against.

Do not expect it to help within a conversation: every turn of one conversation sends the same block.

## Verify before blaming the client

```
KEY=<the api key>
curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $KEY" http://<node>:8000/v1/models
curl -s -o /dev/null -w '%{http_code}\n' http://<node>:8000/v1/models
```

The first should print 200 and the second 401. If the first prints 000 the server is not listening, and
if it prints 401 the key is wrong.
