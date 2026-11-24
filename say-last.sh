#!/usr/bin/env bash
# say-last: read aloud the last thing a Claude in Orca said (see say-last.py --help).
#   say-last            the Claude in the focused Orca pane, else the shown worktree's newest
#   say-last --any      the most recently updated Claude anywhere
#   say-last --print    show the text instead of speaking it
#   say-last -w         (and any say flag) passed through to say
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/.venv/Scripts/python.exe"; [ -x "$PY" ] || PY="$HERE/.venv/bin/python"   # Windows, else Linux/macOS
[ -x "$PY" ] || { echo "say-last: venv missing, run: bash $HERE/install.sh" >&2; exit 1; }
exec "$PY" "$HERE/say-last.py" "$@"
