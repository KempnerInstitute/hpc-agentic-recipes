---
name: local-search
description: Search the web, arxiv, Crossref, PubMed, OpenAlex, or Wikipedia, and fetch page text, using common/tools/search.sh. Use whenever the user asks to search online, look something up, or find papers while served by a local model, and whenever the built-in WebSearch tool fails with a 400 input_schema validation error.
---

# Local search

The built-in WebSearch tool runs on Anthropic infrastructure and is rejected by local vLLM or SGLang
endpoints with `400 ... tools.0.input_schema Field required`. Use `search.sh` through Bash instead.
It needs no API key.

Call it as `search.sh` if it is on PATH, otherwise as `bash <repo>/common/tools/search.sh`.

## Commands

| Command | Use for |
|---------|---------|
| `search.sh web "<query>" [n]` | general web search |
| `search.sh arxiv "<query>" [n]` | preprints, with abstracts |
| `search.sh crossref "<query>" [n]` | published papers by DOI |
| `search.sh pubmed "<query>" [n]` | biomedical literature |
| `search.sh openalex "<query>" [n]` | papers with citation counts |
| `search.sh wiki "<query>" [n]` | background definitions |
| `search.sh fetch "<url>"` | plain text of one page |

`n` is the result count and defaults to 5.

## How to use it

1. Start with `web` for general questions, or `arxiv` and `openalex` for research literature.
2. Read the returned titles, URLs, and snippets.
3. Run `fetch` on the URLs worth reading in full before answering.
4. Cite the URLs you used.

Run two or three different modes for a literature question, since each source indexes different
material. Quote the query; results are plain text, one numbered block per hit.
