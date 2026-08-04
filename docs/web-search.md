# Web search for local models

## The problem

Claude Code's built-in web search does not work against a local endpoint. Invoking it makes the client
send Anthropic's hosted `web_search_20250305` tool, which carries no `input_schema` and which no local
engine implements. vLLM rejects it with HTTP 400:

```
1 validation error:
  {'type': 'missing', 'loc': ('body', 'tools', 0, 'input_schema'), 'msg': 'Field required',
   'input': {'type': 'web_search_20250305', 'name': 'web_search'}}
```

SGLang instead returns HTTP 200 and drops the tool, so the model answers without searching and nothing in
the reply says why. `web_fetch_20250910` and `code_execution_20250522` are the same kind of tool, and both
engines reject those with 400.

Client-side tools, which is everything else the model drives (file edits, shell commands, and any tool you
define), work normally on both engines.

## The replacement

`common/tools/search.sh` is a keyless search tool with seven modes:

```
bash common/tools/search.sh web      "vllm pipeline parallel speculative decoding"
bash common/tools/search.sh arxiv    "mixture of experts routing"
bash common/tools/search.sh crossref "attention is all you need"
bash common/tools/search.sh pubmed   "protein language model"
bash common/tools/search.sh openalex "scaling laws"
bash common/tools/search.sh wiki     "tensor parallelism"
bash common/tools/search.sh fetch    https://example.com/page
```

Only `web` mode needs a package. On first use it builds `.venv-tools` in the repo root and installs `ddgs`
there with `uv`, so `uv` has to be available, or set `SEARCH_PYTHON` to an interpreter that already has
`ddgs`. The other six modes run on plain `python3` and need nothing but network access.

## Making the model use it

Install the tool and the skill. Run this from the repo root:

```
mkdir -p ~/.local/bin ~/.claude/skills
ln -sf "$PWD/common/tools/search.sh" ~/.local/bin/search.sh
cp -r common/skills/local-search ~/.claude/skills/
```

Confirm it works, from any directory:

```
search.sh wiki "tensor parallelism" 1
```

If that reports `search.sh: command not found`, add `~/.local/bin` to your `PATH` in your shell profile.
The skill also gives the model the full path as a fallback, so it works either way.

The skill tells the model what the tool does and when to use each mode. It can only react to a failure it
can see, which on vLLM is the 400. On SGLang the hosted search is dropped silently, so nothing signals the
model to switch: ask for the search tool explicitly there, or expect an answer written without searching.

Both steps above are specific to Claude Code. Another client needs two things instead. Name the script in
the prompt, since the skill format is Claude Code's and no other client reads it. And check whether the
client sandboxes the commands it runs: Codex blocks outbound network by default, so the tool fails under it
while working in your own shell, which looks like a missing tool rather than a sandbox setting. See the
Codex section of clients.md for the setting that opens it.

## Reliability

Web mode scrapes a search engine, so it is the least stable of the seven: results can change format or
rate-limit without notice. The literature modes (arxiv, crossref, pubmed, openalex) call documented APIs
and are more reliable. Prefer them when the question is academic.
