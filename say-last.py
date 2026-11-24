#!/usr/bin/env python
"""say-last.py: read aloud the last thing a Claude in Orca said.

Asks the running Orca app (its CLI) for every agent's last assistant message,
picks the one you mean, and pipes it into say.py. No transcript parsing.

Which agent, first match wins:
  1. the pane this shell runs in (Orca sets ORCA_PANE_KEY), when it is an
     agent: a `!` command typed into a Claude pane reads THAT Claude
  2. the newest agent in this shell's worktree (ORCA_WORKTREE_ID): an Orca
     Quick Command opens a fresh tab beside the Claude, and this finds it
  3. outside Orca: the agent in the pane Orca has focused, else the newest
     agent in that pane's worktree
  4. --any: the most recently updated agent across every worktree

    say-last.py                 speak it (returns at once, like say)
    say-last.py --print         print the text instead of speaking it
    say-last.py --any           most recent agent anywhere
    say-last.py --close-tab     then close the Orca tab this runs in (for a Quick Command)
    say-last.py -- -w -v am_adam   anything after -- goes to say.py
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import say  # noqa: E402 -- stdlib-only at import time; loads say.conf into the environment

PY = HERE / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
SAY = HERE / "say.py"
# SAY_ORCA: the orca CLI, if it is not where the Windows installer puts it or on PATH.
ORCA = Path(os.environ.get("SAY_ORCA") or (
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "orca" / "resources" / "bin" / "orca.exe"
    if os.name == "nt" else shutil.which("orca") or "orca"))


def orca(*args):
    """Run an orca CLI command with --json and return the parsed reply (or None)."""
    try:
        out = subprocess.run([str(ORCA), *args, "--json"], capture_output=True, text=True,
                             encoding="utf-8", errors="replace", timeout=20)
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"say-last: orca {' '.join(args)}: {e}", file=sys.stderr)
        return None
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError:
        print(f"say-last: orca {' '.join(args)} gave no JSON: {out.stdout[:200]} {out.stderr[:200]}",
              file=sys.stderr)
        return None
    return data.get("result") if data.get("ok") else None


def server_outlives_tab():
    """True when the voice server is up before this tab starts it: already running (it
    lives wherever it was started), or started now by SAY_BROKER. Then closing this tab
    cannot take it down. False means say would start it inside this tab."""
    return say.server_alive() or (say.broker_start() and say.server_alive())


def newest(agents):
    return max(agents, key=lambda a: a.get("updatedAt") or 0) if agents else None


def pick(worktrees, any_worktree):
    """Return (agent, worktree, how)."""
    if any_worktree:
        all_agents = [(a, w) for w in worktrees for a in w.get("agents", [])]
        if not all_agents:
            return None, None, "no agents anywhere"
        a, w = max(all_agents, key=lambda aw: aw[0].get("updatedAt") or 0)
        return a, w, "newest anywhere"

    env_pane = os.environ.get("ORCA_PANE_KEY")
    env_wt = os.environ.get("ORCA_WORKTREE_ID")
    if env_pane:
        for w in worktrees:
            for a in w.get("agents", []):
                if a.get("paneKey") == env_pane:
                    return a, w, "this pane"
    if env_wt:
        for w in worktrees:
            if w.get("worktreeId") == env_wt:
                a = newest(w.get("agents", []))
                if a:
                    return a, w, "newest in this tab's worktree"
                return None, w, "no agent in this tab's worktree"

    shown = orca("terminal", "show")           # the pane Orca has focused
    term = (shown or {}).get("terminal")
    if term:
        key = f"{term.get('tabId')}:{term.get('leafId')}"
        for w in worktrees:
            for a in w.get("agents", []):
                if a.get("paneKey") == key:
                    return a, w, "focused pane"
        for w in worktrees:
            if w.get("worktreeId") == term.get("worktreeId"):
                a = newest(w.get("agents", []))
                if a:
                    return a, w, "newest in this worktree"
                return None, w, "no agent in this worktree"

    for w in worktrees:
        if w.get("isActive"):
            a = newest(w.get("agents", []))
            if a:
                return a, w, "newest in the worktree Orca shows"
            return None, w, "no agent in the worktree Orca shows"
    return None, None, "no active worktree"


def main(argv):
    do_print = any_wt = close_tab = False
    say_args = []
    it = iter(argv)
    for arg in it:
        if arg == "--":
            say_args = list(it)
            break
        if arg == "--print":
            do_print = True
        elif arg in ("--any", "-a"):
            any_wt = True
        elif arg == "--close-tab":
            close_tab = True
        elif arg in ("-h", "--help"):
            print(__doc__)
            return 0
        else:
            say_args.append(arg)

    if not ORCA.exists():
        print(f"say-last: orca CLI not found at {ORCA}", file=sys.stderr)
        return 2
    ps = orca("worktree", "ps")
    if not ps:
        print("say-last: Orca is not answering (is the app open?)", file=sys.stderr)
        return 2

    agent, wt, how = pick(ps.get("worktrees", []), any_wt)
    where = Path(wt["path"]).name if wt else "?"
    if not agent:
        print(f"say-last: nothing to read: {how} ({where})", file=sys.stderr)
        return 1
    text = (agent.get("lastAssistantMessage") or "").strip()
    label = agent.get("taskTitle") or agent.get("displayName") or agent.get("agentType") or "agent"
    if not text:
        print(f"say-last: {where} / {label} has not said anything yet ({how})", file=sys.stderr)
        return 1

    print(f"say-last: {where} / {label}, {len(text)} chars ({how})")
    if do_print:
        print(text)
        return 0
    if not PY.exists():
        print(f"say-last: venv missing, run: bash {HERE / 'install.sh'}", file=sys.stderr)
        return 2
    if close_tab and "-w" not in say_args and "--wait" not in say_args and not server_outlives_tab():
        # Why: anything started from an Orca tab is killed when the tab closes, and the tab
        # refuses to let a child break away. With no server running and no SAY_BROKER to
        # start one outside the tab, it would start here and die with this tab, so wait
        # for the reading to finish before closing it.
        say_args.append("-w")
    r = subprocess.run([str(PY), str(SAY), *say_args], input=text, text=True, encoding="utf-8")
    if close_tab:
        handle = os.environ.get("ORCA_TERMINAL_HANDLE")
        if handle:
            # Why: a Quick Command run from the tab bar lives in a tab of its own; the
            # audio is already playing from the warm server, so the tab has no further use.
            subprocess.run([str(ORCA), "terminal", "close", "--terminal", handle, "--tab", "--json"],
                           capture_output=True, timeout=20)
    return r.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
