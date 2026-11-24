# say: GPU text-to-speech for the terminal

Reads any text aloud with [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) on an
NVIDIA GPU (it falls back to the CPU without one). Everything runs locally; nothing leaves
the machine after the one-time download, so it is fine for work output. While it speaks, a
small popup gives you pause, back, skip and stop, and flashes each word as it is spoken.

## Setup

Needs [uv](https://docs.astral.sh/uv/) and bash (Git Bash on Windows). Built and used daily
on Windows 11 with an NVIDIA GPU. Linux and macOS have their code paths (venv layout,
fonts, detaching the server) but are untested.

```bash
bash install.sh                        # venv, CUDA torch, kokoro, model download (about 3.5 GB)
alias say='bash /path/to/say/say.sh'   # put these two in your shell profile
alias shh='bash /path/to/say/say.sh --stop'
cp say.conf.example say.conf           # optional: your own defaults (below)
```

`TORCH_INDEX=<url> bash install.sh` picks another torch build (the default is CUDA 12.8;
macOS gets plain PyPI).

## Use

```bash
say "hello there"                 # returns at once; a new say interrupts the old
claude -p "explain X" | say       # or any command's output
say -f notes.md                   # a file; Markdown is cleaned before speaking
say -w "..."                      # block until the audio finishes
shh                               # stop it (= say --stop)
say --pause                       # pause; run again to resume
say --no-popup "..."              # speak without the control popup
say --list-voices                 # af_heart (default), am_adam, bf_emma, ...
say -v bm_george -r 1.2 "..."     # voice and speed
say -o out.wav "..."              # write a WAV instead of playing
say --raw ...                     # skip the Markdown cleanup
say --status                      # is the server up, what is it doing
say --quit                        # shut the server down (frees the GPU)
say-last                          # read the last thing a Claude in Orca said (below)
```

Markdown cleanup: fenced code blocks become "Code block, N lines, skipped", inline
code keeps its words, bold/italic/header markers go, table rows read as comma lists,
URLs read as "a link".

## The popup: controls and RSVP

While anything is spoken, a small always-on-top popup shows near the bottom of the screen:

- **One word at a time (RSVP)**, timed to the voice from Kokoro's per-word timings. The
  focus letter (about a third into the word) is red and always sits under the notch, so
  the eye never moves. Punctuation rides on its word (`there,`).
- **Buttons:** back a sentence (press again to keep going back), pause/resume, skip a
  sentence (past the last one ends the reading), stop. `Aa` hides the word display and
  leaves a slim control bar; the X hides the popup until the next reading, the voice keeps going.
- **Click the word to pause.** Pause cuts in within 50 ms and resumes at the start of the
  word that was cut off. Back and skip work while paused, and the word display follows.
- **Drag anywhere that is not a button** to move it. The spot and the word display on/off
  are kept in `.run/popup.json`; a spot on a monitor that is no longer attached is ignored.
- It never takes keyboard focus and has no taskbar button. It hides 1.5 s after the reading ends.

It runs inside the server, not the terminal that asked, so it outlives that terminal.
`--no-popup` skips it for one reading; `SAY_POPUP=0` turns it off.

## How it runs

The first `say` starts a warm server in the background (`say.py --serve`, its own hidden
console and process group, log in `.run/server.log`) that loads the model onto the GPU
once. That costs about 30 s, mostly importing torch and kokoro. Every later `say` connects
to it on `127.0.0.1:47337` and audio starts within about a second. The whole text is
synthesized ahead of playback, about 25x realtime, so back and skip can land anywhere.

The server exits by itself after an idle hour (`SAY_IDLE_MIN`; a paused reading counts as
busy). A cold start right after another one takes about 8 s; after hours away it takes
about 40 s, because the 4.6 GB venv has fallen out of the file cache, so if you read all
day, `SAY_IDLE_MIN=480` keeps it warm. Holding the model costs about 600 MB of VRAM;
`say --quit` frees it.

Measured on an RTX 4000 Ada laptop GPU: first call 33 s, later calls 0.8 s, a 540-word
page renders in about 10 s, `shh` lands within 2 s.

## Knobs: environment or `say.conf`

Set these in the environment, or copy `say.conf.example` to `say.conf` (gitignored) so
they hold however say is launched. The environment wins over the file.

| Knob | Default | What |
|---|---|---|
| `SAY_VOICE` | `af_heart` | default voice (`say --list-voices`) |
| `SAY_RATE` | `1.0` | default speed multiplier |
| `SAY_IDLE_MIN` | `60` | the server exits after this many idle minutes |
| `SAY_POPUP` | `1` | `0` = no control popup |
| `SAY_PORT` | `47337` | where the server listens on 127.0.0.1 |
| `SAY_BROKER` | empty | a local URL that starts the server for you (below) |
| `SAY_ORCA` | Orca's install spot, else `PATH` | the orca CLI, for `say-last` |

Any other line in `say.conf` is passed into the server's environment as well (a process
label your dashboard reads, `CUDA_VISIBLE_DEVICES`, ...).

**`SAY_BROKER`, only if your terminal kills its children.** Some terminals kill every
process a tab started when that tab closes, and refuse to let one break away (Orca does).
A server started from such a tab dies with it, mid-sentence if the tab closes while it
speaks. If you run a long-lived local process anyway (a dashboard, a tray app), give it an
endpoint that runs `<venv python> say.py --start` and answers `{"ok": true}`, and point
`SAY_BROKER` at it: when the server is down, `say` POSTs there first and starts the server
itself only if that fails. Without a broker everything works; the server just lives as
long as the terminal that first started it.

## say-last: the last thing a Claude in Orca said (optional)

For Orca users; `say` itself does not need it.
`say-last` asks the running Orca app (`orca worktree ps --json` carries every agent's
`lastAssistantMessage`) and pipes the text into `say`. No transcript parsing, no model turn.
Which Claude, first match wins: the pane the shell runs in (`ORCA_PANE_KEY`), the newest
agent in the shell's worktree (`ORCA_WORKTREE_ID`), outside Orca the pane Orca has focused,
or `--any` for the newest anywhere. `--print` shows the text instead of speaking it;
`--close-tab` closes the Orca tab it runs in afterward; other flags pass through to `say`.

**As a button in Orca:** Settings > Quick Commands > add one, label `Read aloud`, command
`bash /path/to/say/say-last.sh --close-tab`, scope global. The button then appears in the
tab bar of every worktree; clicking it opens a fresh tab beside the Claude, speaks, and
closes that tab. Without a running server or a `SAY_BROKER`, the tab stays open until the
reading ends (closing it would kill the server it just started), and the next press is a
cold start.

## Files

| File | Role |
|---|---|
| `say.sh` | Wrapper the `say`/`shh` aliases call: runs `say.py` with the venv's Python. |
| `say.py` | Client and server in one file: Markdown cleanup, Kokoro on CUDA, streaming playback, the popup (Tk), stop/pause/status/quit over a local socket. |
| `say-last.sh`, `say-last.py` | Pick a Claude running in Orca and feed its last assistant message to `say`. |
| `install.sh` | One-time setup: `uv` venv (Python 3.12), CUDA torch, kokoro, model prefetch. Re-runnable. |
| `say.conf.example` | The knobs; copy to `say.conf` to use. |
| `.venv/`, `.run/`, `say.conf` | Local, not in git. The venv is 4.6 GB; `.run/` holds the server pid, log and `popup.json`. |

Model weights cache under `~/.cache/huggingface/`.

## Why Kokoro

82M parameters, Apache-licensed, near-instant on a laptop GPU, and it never loses its
place in long text, which is what reading an agent's output needs. Bigger local models
(Chatterbox, Orpheus, Dia) buy voice cloning and emotion, not better narration, and they
need chunking on long input. Windows' own System.Speech voices (David, Zira) are the 2012
SAPI set; the Windows 11 natural voices are locked to Narrator and Edge.
