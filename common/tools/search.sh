#!/usr/bin/env bash
# Web and literature search for local models, which cannot use Anthropic's hosted web search tool.
# Usage: search.sh <web|arxiv|crossref|pubmed|openalex|wiki|fetch> "<query or url>" [count]
# Web search needs the ddgs package; it is installed into .venv-tools on first use.
# Override the interpreter with SEARCH_PYTHON, the venv location with SEARCH_VENV, or the result
# count with the third argument.
set -euo pipefail
# The install puts a symlink on PATH, so resolve it before locating the repo.
S="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
source "$S/../lib/repo_root.sh"
REPO_DIR="$REPO_ROOT"
MODE="${1:?usage: search.sh <web|arxiv|crossref|pubmed|openalex|wiki|fetch> \"<query or url>\" [count]}"
ARG="${2:?usage: search.sh <mode> \"<query or url>\" [count]}"
N="${3:-5}"

if [ "$MODE" = web ]; then
  VENV="${SEARCH_VENV:-$REPO_DIR/.venv-tools}"
  PY="${SEARCH_PYTHON:-}"
  if [ -z "$PY" ] && [ -x "$VENV/bin/python" ]; then PY="$VENV/bin/python"; fi
  if [ -z "$PY" ] && python3 -c 'import ddgs' 2>/dev/null; then PY="python3"; fi
  if [ -z "$PY" ]; then
    export PATH="$HOME/.local/bin:$PATH"
    command -v uv >/dev/null 2>&1 || { echo "search.sh: need uv, or install ddgs and set SEARCH_PYTHON" >&2; exit 1; }
    echo "search.sh: creating $VENV with ddgs (one time)" >&2
    uv venv "$VENV" --python 3.12 --quiet >&2
    uv pip install --python "$VENV/bin/python" --quiet ddgs >&2
    PY="$VENV/bin/python"
  fi
  exec "$PY" - "$ARG" "$N" <<'PY'
import sys
from ddgs import DDGS
query, n = sys.argv[1], int(sys.argv[2])
rows = list(DDGS().text(query, max_results=n))
if not rows:
    print("no results")
for i, r in enumerate(rows, 1):
    print(f"{i}. {(r.get('title') or '').strip()}")
    print(f"   {(r.get('href') or '').strip()}")
    body = " ".join((r.get("body") or "").split())
    if body:
        print(f"   {body[:300]}")
PY
fi

exec python3 - "$MODE" "$ARG" "$N" <<'PY'
import html, json, sys, urllib.parse, urllib.request, xml.etree.ElementTree as ET
from html.parser import HTMLParser

mode, arg = sys.argv[1], sys.argv[2]
n = int(sys.argv[3])
UA = {"User-Agent": "kempner-local-search (research use)"}


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read()


def clean(s):
    return " ".join(html.unescape(html.unescape(s or "")).split())


def show(i, title, url, extra=""):
    print(f"{i}. {clean(title)}")
    if url:
        print(f"   {url}")
    if extra:
        print(f"   {clean(extra)[:300]}")


def arxiv(q):
    ns = {"a": "http://www.w3.org/2005/Atom"}

    def query(expr):
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"search_query": expr, "start": 0, "max_results": n})
        return ET.fromstring(get(url)).findall("a:entry", ns)

    entries = query(f'all:"{q}"')
    if not entries and len(q.split()) > 1:
        entries = query(" AND ".join(f"all:{t}" for t in q.split()))
    for i, e in enumerate(entries, 1):
        link = e.find("a:id", ns)
        summary = e.find("a:summary", ns)
        authors = [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)][:4]
        show(i, e.findtext("a:title", "", ns), link.text if link is not None else "",
             (", ".join(authors) + ". " + (summary.text if summary is not None else "")))
    if not entries:
        print("no results")


def crossref(q):
    d = json.loads(get("https://api.crossref.org/works?query=" + urllib.parse.quote(q) + f"&rows={n}"))
    items = d["message"]["items"]
    for i, it in enumerate(items, 1):
        year = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
        venue = (it.get("container-title") or [""])[0]
        show(i, (it.get("title") or ["?"])[0], it.get("URL", ""), f"{venue} {year or ''}".strip())
    if not items:
        print("no results")


def pubmed(q):
    ids = json.loads(get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json"
                         f"&retmax={n}&term=" + urllib.parse.quote(q)))["esearchresult"].get("idlist", [])
    if not ids:
        print("no results")
        return
    d = json.loads(get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&retmode=json&id="
                       + ",".join(ids)))["result"]
    for i, pmid in enumerate(ids, 1):
        r = d.get(pmid, {})
        show(i, r.get("title", "?"), f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
             f"{r.get('source', '')} {r.get('pubdate', '')}")


def openalex(q):
    d = json.loads(get("https://api.openalex.org/works?search=" + urllib.parse.quote(q) + f"&per-page={n}"))
    items = d.get("results", [])
    for i, it in enumerate(items, 1):
        loc = (it.get("primary_location") or {}).get("source") or {}
        show(i, it.get("display_name", "?"),
             it.get("doi") or (it.get("primary_location") or {}).get("landing_page_url", ""),
             f"{loc.get('display_name', '')} {it.get('publication_year', '')} "
             f"cited_by={it.get('cited_by_count', 0)}")
    if not items:
        print("no results")


def wiki(q):
    d = json.loads(get("https://en.wikipedia.org/w/api.php?action=query&list=search&format=json"
                       f"&srlimit={n}&srsearch=" + urllib.parse.quote(q)))
    hits = d.get("query", {}).get("search", [])
    for i, h in enumerate(hits, 1):
        title = h.get("title", "?")
        snippet = h.get("snippet", "").replace("<span class=\"searchmatch\">", "").replace("</span>", "")
        show(i, title, "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_")), snippet)
    if not hits:
        print("no results")


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.out = []
        self.skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript") and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip and data.strip():
            self.out.append(data.strip())


def fetch(url):
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    raw = get(url).decode("utf-8", "replace")
    p = Text()
    p.feed(raw)
    text = " ".join(" ".join(p.out).split())
    limit = n * 2000 if n > 5 else 10000
    print(text[:limit])


handlers = {"arxiv": arxiv, "crossref": crossref, "pubmed": pubmed,
            "openalex": openalex, "wiki": wiki, "fetch": fetch}
if mode not in handlers:
    sys.exit(f"search.sh: unknown mode '{mode}' (use web, arxiv, crossref, pubmed, openalex, wiki, fetch)")
handlers[mode](arg)
PY
