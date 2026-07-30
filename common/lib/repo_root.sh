# Resolve REPO_ROOT once, from any depth. Source this, do not execute.
#
# Every file that needs a repo-relative path must use this rather than counting ".." levels. Counting
# levels is what breaks when a file moves: a tool at common/tools/ and a client env at
# recipes/<model>/<hardware>/ sit at different depths, and a hardcoded "one level up" silently
# resolves to the wrong directory, which then fails as a missing API key rather than a bad path.
if [ -z "${REPO_ROOT:-}" ]; then
  _rr_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(cd "$_rr_here" && git rev-parse --show-toplevel 2>/dev/null || true)"
  if [ -z "$REPO_ROOT" ] || [ ! -f "$REPO_ROOT/common/defaults.sh" ]; then
    # Not a git checkout, or a stripped copy: walk up looking for the marker file.
    _rr_d="$_rr_here"
    while [ "$_rr_d" != "/" ] && [ ! -f "$_rr_d/common/defaults.sh" ]; do
      _rr_d="$(dirname "$_rr_d")"
    done
    REPO_ROOT="$_rr_d"
  fi
  unset _rr_here _rr_d
fi
export REPO_ROOT
