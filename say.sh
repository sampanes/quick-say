#!/usr/bin/env bash
# say: read text aloud on the GPU. Text from args, a pipe, or -f FILE.
#   say "hello"               speak; returns at once (a warm local server plays it)
#   some-command | say        speak its output (Markdown cleaned first)
#   say -w ...                wait until the audio finishes
#   say --stop   (alias shh)  silence it
#   say --status / --quit     server state / shut it down (frees the GPU)
#   say --list-voices         voices; pick with -v am_adam, rate with -r 1.2
# First run: ./install.sh. The first say after that starts the server (about 30 s).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/.venv/Scripts/python.exe"; [ -x "$PY" ] || PY="$HERE/.venv/bin/python"   # Windows, else Linux/macOS
[ -x "$PY" ] || { echo "say: venv missing, run: bash $HERE/install.sh" >&2; exit 1; }
exec "$PY" "$HERE/say.py" "$@"
