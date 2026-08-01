#!/usr/bin/env python3
"""Flag calendar dates and internal project vocabulary in prose, ignoring fenced code.

A date does not tell a reader whether a number still holds; the engine version does. Dates inside fenced
blocks are exempt because those are verbatim engine output pasted as evidence, and editing a quoted log
line to remove its timestamp would falsify it.

Shell and Python files are checked too. Their comments carry the same kind of claim a README does, and a
flag justified by a date is exactly the case the rule exists for.

No file is exempt. There used to be a carve-out for two planning documents; they were deleted instead,
because a page full of predictions is not something a recipe reader should find.
"""
import pathlib, re, subprocess, sys
# Cluster arrangements local to one group, which mean nothing to a reader outside it. The word is
# matched, not one phrasing of it: a sentence about reserved nodes survived an earlier sweep that
# searched only for the noun.
JARGON = re.compile(r'pre-restructure|the restructure|this session|our campaign|reserv(?:ation|ed node)', re.I)
DATE = re.compile(r'20[0-9]{2}-[0-9]{2}-[0-9]{2}')
# Every tracked text file, not a chosen few: the first version scanned only markdown and missed the
# same vocabulary in .gitignore and in shell comments.
SKIP = {".safetensors", ".png", ".jpg", ".gz", ".sif", ".whl", ".lock"}
TRACKED = [f for f in subprocess.check_output(["git", "ls-files"]).decode().split()
           if pathlib.Path(f).suffix not in SKIP]
bad = 0
for f in TRACKED:
    if pathlib.Path(f).name == pathlib.Path(__file__).name:
        continue
    fence = False
    for i, l in enumerate(pathlib.Path(f).read_text().split("\n"), 1):
        if l.lstrip().startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        # A guard that must name the flag it forbids marks itself, so the rule can stay a plain
        # word match everywhere else.
        if 'check_prose: allow' in l:
            continue
        if DATE.search(l):
            print("  DATE   %s:%d  %s" % (f, i, l.strip()[:100])); bad += 1
        if JARGON.search(l):
            print("  JARGON %s:%d  %s" % (f, i, l.strip()[:100])); bad += 1
print("  %d finding(s)" % bad)
sys.exit(1 if bad else 0)
