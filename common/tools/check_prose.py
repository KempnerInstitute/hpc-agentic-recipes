#!/usr/bin/env python3
"""Flag calendar dates and internal project vocabulary in prose, ignoring fenced code.

A date does not tell a reader whether a number still holds; the engine version does. Dates inside fenced
blocks are exempt because those are verbatim engine output pasted as evidence, and editing a quoted log
line to remove its timestamp would falsify it.

No file is exempt. There used to be a carve-out for two planning documents; they were deleted instead,
because a page full of predictions is not something a recipe reader should find.
"""
import pathlib, re, subprocess, sys
JARGON = re.compile(r'pre-restructure|the restructure|this session|our campaign', re.I)
DATE = re.compile(r'20[0-9]{2}-[0-9]{2}-[0-9]{2}')
bad = 0
for f in subprocess.check_output(["git", "ls-files", "*.md"]).decode().split():
    fence = False
    for i, l in enumerate(pathlib.Path(f).read_text().split("\n"), 1):
        if l.lstrip().startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        if DATE.search(l):
            print("  DATE   %s:%d  %s" % (f, i, l.strip()[:100])); bad += 1
        if JARGON.search(l):
            print("  JARGON %s:%d  %s" % (f, i, l.strip()[:100])); bad += 1
print("  %d finding(s)" % bad)
sys.exit(1 if bad else 0)
