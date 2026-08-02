**Anthropic's hosted tools do not work against a local endpoint.** Server-side tools such as
`web_search_20250305`, `web_fetch_20250910` and `code_execution_20250522` are executed by Anthropic's own
API rather than by the model, so no endpoint here can run them. What you see differs by engine, measured
on both.

vLLM rejects all three with HTTP 400, because their definitions carry no `input_schema`:

```
1 validation error:
  {'type': 'missing', 'loc': ('body', 'tools', 0, 'input_schema'), 'msg': 'Field required',
   'input': {'type': 'web_search_20250305', 'name': 'web_search'}}
```

SGLang is harder to diagnose. It accepts `web_search_20250305` with HTTP 200 and drops the tool, logging
that it has no native support, so the model answers without searching and nothing in the reply says why.
It rejects `web_fetch_20250910` and `code_execution_20250522` with HTTP 400.

Client-side tools (file edits, shell, and anything you define) work normally on both. For web access,
install the repo's keyless search tool and skill, from the repo root:

```
mkdir -p ~/.local/bin ~/.claude/skills
ln -sf "$PWD/common/tools/search.sh" ~/.local/bin/search.sh
cp -r common/skills/local-search ~/.claude/skills/
```

Check it with `search.sh wiki "tensor parallelism" 1`, and add `~/.local/bin` to your `PATH` if the
command is not found. The model then searches through `search.sh` (web, arxiv, crossref, pubmed,
openalex, wiki, fetch) instead of the hosted tool. Full details in
[docs/web-search.md](../../docs/web-search.md).
