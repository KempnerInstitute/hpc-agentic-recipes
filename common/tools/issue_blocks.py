#!/usr/bin/env python3
"""Inject and verify canonical issue text in recipe READMEs.

Recipes deliberately repeat the same issue text, so a reader of one recipe never has to open another
file. Hand-maintained duplication drifts, so the text lives once in common/issues/<slug>.md and is
injected into blocks delimited by:

    <!-- issue:<slug> begin -->
    <!-- issue:<slug> end -->

Usage:  issue_blocks.py <repo_root> [--fix]

Checks in both directions: a recipe listed in the matrix must carry the block, and a recipe carrying a
block the matrix does not list for it is stale and fails. The second direction is what catches copies
left behind when a recipe stops being affected.
"""
import fnmatch
import re
import sys
from pathlib import Path


def recipes(root: Path):
    base = root / "recipes"
    if not base.is_dir():
        return []
    out = []
    for model in sorted(base.iterdir()):
        if not model.is_dir():
            continue
        for hw in sorted(model.iterdir()):
            if hw.is_dir():
                out.append(f"{model.name}/{hw.name}")
    return out


def expand(patterns: str, all_recipes):
    if patterns.strip() in ("", "-"):
        return set()
    hit = set()
    for pat in (p.strip() for p in patterns.split(",")):
        if not pat:
            continue
        hit |= {r for r in all_recipes if fnmatch.fnmatchcase(r, pat)}
    return hit


def load_matrix(root: Path, all_recipes):
    """Return {slug: set(recipes)} after applying excludes."""
    path = root / "common" / "issues" / "matrix.tsv"
    table = {}
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        parts = [p.strip() for p in line.split("\t") if p.strip() != ""]
        if len(parts) < 2:
            print(f"FAIL  matrix.tsv line {lineno}: expected tab separated slug, include, exclude")
            table.setdefault("__error__", set())
            continue
        slug, include = parts[0], parts[1]
        exclude = parts[2] if len(parts) > 2 else "-"
        table[slug] = expand(include, all_recipes) - expand(exclude, all_recipes)
    return table


def block_re(slug: str):
    # Tolerates an empty block, which is how a new recipe is authored before the first injection.
    return re.compile(
        r"(<!--\s*issue:%s\s+begin\s*-->)(.*?)(<!--\s*issue:%s\s+end\s*-->)" % (re.escape(slug), re.escape(slug)),
        re.S,
    )


def norm(text: str):
    return " ".join(text.split())


def main():
    root = Path(sys.argv[1]).resolve()
    fix = "--fix" in sys.argv[2:]
    issues_dir = root / "common" / "issues"
    all_recipes = recipes(root)
    matrix = load_matrix(root, all_recipes)
    failures = 0

    if "__error__" in matrix:
        return 1

    canonical = {}
    for slug in matrix:
        src = issues_dir / f"{slug}.md"
        if not src.is_file():
            print(f"FAIL  matrix names {slug} but {src.relative_to(root)} does not exist")
            failures += 1
            continue
        canonical[slug] = src.read_text().strip()

    # Unused canonical files are dead weight: flag them so the corpus cannot silently accumulate.
    for src in sorted(issues_dir.glob("*.md")):
        if src.stem not in matrix:
            print(f"FAIL  common/issues/{src.name} is not referenced by matrix.tsv")
            failures += 1

    for recipe in all_recipes:
        readme = root / "recipes" / recipe / "README.md"
        if not readme.is_file():
            continue
        text = original = readme.read_text()
        expected = {s for s, rs in matrix.items() if recipe in rs}
        present = {m.group(1) for m in re.finditer(r"<!--\s*issue:([a-z0-9-]+)\s+begin\s*-->", text)}

        for slug in sorted(expected - present):
            print(f"FAIL  {recipe}: missing required issue block {slug}")
            failures += 1
        for slug in sorted(present - expected):
            print(f"FAIL  {recipe}: carries issue block {slug} which the matrix does not list for it")
            failures += 1

        for slug in sorted(expected & present):
            if slug not in canonical:
                continue
            body = canonical[slug]
            if fix:
                text = block_re(slug).sub(
                    lambda m: f"{m.group(1)}\n{body}\n{m.group(3)}", text, count=1
                )
            found = block_re(slug).search(text)
            if not found or norm(found.group(2)) != norm(body):
                print(f"FAIL  {recipe}: issue block {slug} does not match common/issues/{slug}.md"
                      + ("" if fix else " (run --fix)"))
                failures += 1

        if fix and text != original:
            readme.write_text(text)
            print(f"  injected canonical text into {recipe}/README.md")

    if failures == 0:
        print(f"  {len(all_recipes)} recipes, {len(matrix)} issues, all blocks match")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
