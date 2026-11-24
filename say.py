#!/usr/bin/env python
"""say.py: read text aloud on the GPU (Kokoro-82M).

Client and server in one file. The first `say` starts a warm server that
loads the model onto CUDA once (about 30 s); every later `say` talks to it
over 127.0.0.1 and starts speaking within about a second. The server exits
on its own after SAY_IDLE_MIN minutes without a request (default 60). Knobs come
from the environment or from say.conf beside this file (see say.conf.example).

    say.py words...           speak, return at once (a new say interrupts the old)
    cmd | say.py              speak stdin; Markdown is cleaned first
    say.py -f FILE            speak a file
    say.py -w ...             block until the audio ends
    say.py -o out.wav ...     render to a WAV instead of playing
    say.py --stop             silence (what the shh alias calls)
    say.py --pause            pause, or resume a paused reading
    say.py --no-popup ...     speak without the control popup
    say.py --status           is the server up, what is it doing
    say.py --quit             shut the server down (frees the GPU)
    say.py --serve            run the server in the foreground (normally spawned)

While it speaks, the server shows an always-on-top popup: back a sentence, pause,
skip a sentence, stop, and an RSVP display that flashes each word as it is spoken.
"""
import argparse
import bisect
import json
import os
import queue
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE / ".run"
PIDFILE = RUN / "server.pid"
LOGFILE = RUN / "server.log"
CONF = HERE / "say.conf"


def _load_conf():
    """KEY=VALUE lines from say.conf beside this file become defaults under the real
    environment, so the knobs hold however say is launched (a terminal, a button, a broker)."""
    try:
        lines = CONF.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


_load_conf()
PORT = int(os.environ.get("SAY_PORT", "47337"))
# Holding the model costs about 600 MB of VRAM; `say --quit` frees it. Longer keeps reads
# instant: a cold start after hours away costs about 40 s (the 4.6 GB venv falls out of
# the file cache and is re-read and re-scanned), against about 8 s right after another.
IDLE_MIN = float(os.environ.get("SAY_IDLE_MIN", "60"))
# Optional: the URL of a long-lived local process that starts the server when it is down,
# for terminals that kill everything a tab started when the tab closes. POST to it must
# run `<venv python> say.py --start` and answer {"ok": true}. Empty (the default): say
# starts the server itself.
BROKER = os.environ.get("SAY_BROKER", "")
# SAY_POPUP=0 runs the server without the control popup.
POPUP = os.environ.get("SAY_POPUP", "1") != "0"
SR = 24000
REPO = "hexgrad/Kokoro-82M"
DEFAULT_VOICE = "af_heart"
VOICE = os.environ.get("SAY_VOICE", DEFAULT_VOICE)   # what -v and -r default to
RATE = float(os.environ.get("SAY_RATE", "1.0"))
# A hand-picked subset of the model's voices. Prefix: a=American b=British, f/m=gender.
VOICES = {
    "af_heart": "American, warm",
    "af_bella": "American, bright",
    "af_nicole": "American, soft, close-mic",
    "af_sarah": "American, neutral",
    "am_adam": "American, deep",
    "am_michael": "American, neutral",
    "am_fenrir": "American, gravelly",
    "bf_emma": "British, calm",
    "bf_isabella": "British, crisp",
    "bm_george": "British, measured",
    "bm_lewis": "British, light",
}


# ---------- text ----------

def clean_markdown(text: str) -> str:
    def fence(m):
        n = m.group(1).count("\n")
        return f"\nCode block, {n} lines, skipped.\n"
    text = re.sub(r"```[^\n]*\n(.*?)```", fence, text, flags=re.S)
    text = re.sub(r"`([^`\n]+)`", r"\1", text)                              # inline code keeps its words
    text = re.sub(r"^\s{0,3}#{1,6}\s+", "", text, flags=re.M)               # headers
    text = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", text)                        # bold
    text = re.sub(r"(?<!\w)[*_]([^*_\n]+)[*_](?!\w)", r"\1", text)           # italic
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)                     # links keep their label
    text = re.sub(r"https?://\S+", "a link", text)
    text = re.sub(r"^[ \t]*\|?[ \t]*:?-{2,}:?[ \t]*(\|[ \t]*:?-{2,}:?[ \t]*)*\|?[ \t]*$", "", text, flags=re.M)  # table rules
    text = re.sub(r"^[ \t]*\|(.+)\|[ \t]*$",
                  lambda m: ", ".join(c.strip() for c in m.group(1).split("|") if c.strip()) + ".",
                  text, flags=re.M)                                           # table rows
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.M)                    # bullets
    text = re.sub(r"^\s*\d+[.)]\s+", "", text, flags=re.M)                  # numbered lists
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_input(args) -> str:
    if args.file:
        text = Path(args.file).read_bytes().decode("utf-8", "replace")
        if text.strip():
            return text
    if args.text:
        return " ".join(args.text)
    if not sys.stdin.isatty():
        # Empty under scripts and agents whose stdin is /dev/null: that is fine.
        return sys.stdin.buffer.read().decode("utf-8", "replace")
    return ""


# ---------- server ----------

_SENTENCE_END = re.compile(r"[.!?…][\"'”’)\]]*$")


def timed_words(tokens, offset):
    """Kokoro's tokens for one segment -> ([start sample, display word], sentence starts).

    Punctuation rides on its neighbor ("there," and "(big)"), so a flash is a word as
    written, not a token. A segment's first word, and every word after one that ends a
    sentence, is a sentence start: where back and skip land."""
    words, marks = [], []
    pending, glue = "", False   # glue: the last token joined a word and had no space after it
    for t in tokens or []:
        txt = (t.text or "").replace("`", "")
        joined = False
        if txt:
            if glue and words:
                words[-1][1] += txt
                joined = True
            elif not re.search(r"\w", txt):
                pending += txt          # opening punctuation waits for its word
            elif t.start_ts is None and words:
                words[-1][1] += " " + pending + txt
                pending, joined = "", True
            else:
                start = offset + int((t.start_ts or 0.0) * SR)
                if not words or _SENTENCE_END.search(words[-1][1]):
                    marks.append(start)
                words.append([start, pending + txt])
                pending, joined = "", True
        glue = joined and not t.whitespace
    return words, marks


class Job:
    def __init__(self, text, voice, rate, out=None, popup=True):
        self.text, self.voice, self.rate, self.out, self.popup = text, voice, rate, out, popup
        self.lang = "b" if voice.startswith("b") else "a"
        self.cancel = threading.Event()
        self.done = threading.Event()
        self.wake = threading.Event()   # nudges the player: new audio, pause, seek, stop
        self.error = None
        self.seconds = 0.0
        # The whole reading as one timeline, filled as segments are synthesized. Every
        # position is a sample index from the start of the text, so the player, pause,
        # back/skip and the RSVP word all share one coordinate.
        self.segs, self.offsets = [], []
        self.total = 0                       # samples synthesized so far
        self.synth_done = False
        self.words, self.word_starts = [], []
        self.marks = []                      # sentence starts
        self.cursor = 0                      # next sample the player writes
        self.run = None                      # (first sample, monotonic time it is heard) while the stream plays
        self.paused = False
        self.seek_to = None

    def add(self, audio, tokens):
        off = self.total
        words, marks = timed_words(tokens, off)
        self.segs.append(audio)
        self.offsets.append(off)
        self.words.extend(words)             # before word_starts: an index the UI finds always exists
        self.word_starts.extend(w[0] for w in words)
        self.marks.extend(marks)
        self.total = off + len(audio)
        self.wake.set()

    def slice(self, pos, n):
        """Up to n samples from pos, never crossing a segment; None past what is synthesized."""
        if pos >= self.total:
            return None
        k = bisect.bisect_right(self.offsets, pos) - 1
        i = pos - self.offsets[k]
        return self.segs[k][i:i + n]

    def heard(self):
        """The sample the listener hears now. A stream starts with its buffer primed with
        silence (0.18 s on MME), so its first sample is heard one output latency after start."""
        run = self.run
        if run is None:
            return self.cursor
        origin, t = run
        return max(origin, min(origin + int((time.monotonic() - t) * SR), self.cursor))

    def word_index(self, pos):
        return bisect.bisect_right(self.word_starts, pos) - 1

    def word_start(self, pos):
        i = self.word_index(pos)
        return self.word_starts[i] if i >= 0 else pos

    def toggle_pause(self):
        self.paused = not self.paused
        self.wake.set()
        return self.paused

    def seek(self, step):
        """Back: to the start of this sentence, or the one before if this one began under
        1.5 s ago. Skip: to the next sentence; past the last one, the reading ends.
        Presses chain, so three quick backs go three sentences back."""
        here = self.seek_to if self.seek_to is not None else self.heard()
        if step < 0:
            k = bisect.bisect_right(self.marks, here - int(1.5 * SR)) - 1
            target = self.marks[max(k, 0)] if self.marks else 0
        else:
            k = bisect.bisect_right(self.marks, here + int(0.3 * SR))
            if k < len(self.marks):
                target = self.marks[k]
            elif self.synth_done:
                target = self.total
            else:
                return
        self.seek_to = target
        self.wake.set()

    def take_seek(self):
        target, self.seek_to = self.seek_to, None
        return target


class Speaker:
    """Owns the GPU pipelines and the audio output. One job at a time; a new job cancels the old."""

    def __init__(self, force_cpu=False):
        import torch
        self.device = "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"
        self.pipelines = {}
        self.jobs = queue.Queue()
        self.current = None
        self.pipeline("a")
        # Warm up: the first synthesis compiles kernels and loads the voice (about 5 s).
        for _ in self._synth(Job("Warm up.", DEFAULT_VOICE, 1.0, popup=False)):
            pass
        threading.Thread(target=self._loop, daemon=True).start()

    def pipeline(self, lang):
        if lang not in self.pipelines:
            from kokoro import KPipeline
            self.pipelines[lang] = KPipeline(lang_code=lang, repo_id=REPO, device=self.device)
        return self.pipelines[lang]

    def busy(self):
        return self.current is not None or not self.jobs.empty()

    def submit(self, job):
        self.stop()
        self.jobs.put(job)

    def stop(self):
        while True:
            try:
                j = self.jobs.get_nowait()
            except queue.Empty:
                break
            j.cancel.set()
            j.done.set()
        cur = self.current
        if cur is not None:
            cur.cancel.set()
            cur.wake.set()
            return True
        return False

    def toggle_pause(self):
        """True = now paused, False = now playing, None = nothing to pause."""
        cur = self.current
        return None if cur is None else cur.toggle_pause()

    def seek(self, step):
        cur = self.current
        if cur is not None:
            cur.seek(step)

    def _synth(self, job):
        """Yield (float32 audio at 24 kHz, Kokoro tokens with word timings), one per segment."""
        import numpy as np
        pad = np.zeros(int(SR * 0.12), dtype=np.float32)
        pipe = self.pipeline(job.lang)
        for result in pipe(job.text, voice=job.voice, speed=job.rate, split_pattern=r"\n+"):
            if job.cancel.is_set():
                return
            audio = result.audio
            if audio is None:
                continue
            yield np.concatenate([audio.detach().cpu().numpy().astype(np.float32), pad]), result.tokens

    def _loop(self):
        while True:
            job = self.jobs.get()
            if job.cancel.is_set():
                job.done.set()
                continue
            self.current = job
            try:
                self._speak(job)
            except Exception as e:  # noqa: BLE001
                job.error = f"{type(e).__name__}: {e}"
                log(f"error: {job.error}")
            finally:
                self.current = None
                job.done.set()

    def _speak(self, job):
        import numpy as np

        def produce():
            # The whole text is synthesized ahead (about 25x realtime, about 6 MB of audio a
            # minute), so back and skip can land anywhere already rendered.
            try:
                for audio, tokens in self._synth(job):
                    job.add(audio, tokens)
            except Exception as e:  # noqa: BLE001
                job.error = f"{type(e).__name__}: {e}"
            finally:
                job.synth_done = True
                job.wake.set()

        producer = threading.Thread(target=produce, daemon=True)
        producer.start()
        try:
            if job.out:
                producer.join()
                if job.segs and not job.cancel.is_set():
                    import soundfile as sf
                    audio = np.concatenate(job.segs)
                    sf.write(job.out, audio, SR)
                    job.seconds = len(audio) / SR
                return
            self._play(job)
        finally:
            # One synthesis at a time: the pipelines are not shared across threads.
            producer.join()

    def _play(self, job):
        import sounddevice as sd
        # PortAudio snapshots the device list at init: re-scan so headphones that
        # came on after the server started, or a changed default device, are used.
        try:
            sd._terminate()
            sd._initialize()
        except Exception:  # noqa: BLE001
            pass
        log(f"output device: {sd.query_devices(kind='output')['name']}")
        step = SR // 20  # 50 ms slices so pause and stop land fast
        stream = sd.OutputStream(samplerate=SR, channels=1, dtype="float32")
        latency = float(stream.latency)

        def halt(drain=False):
            """Stop the stream: drain plays what is queued, otherwise it is dropped at once."""
            if job.run is not None:
                (stream.stop if drain else stream.abort)()
                job.run = None

        try:
            while not job.cancel.is_set():
                target = job.take_seek()
                if target is not None:
                    halt()
                    job.cursor = target
                    continue
                if job.paused:
                    if job.run is not None:
                        heard = job.heard()
                        halt()
                        job.cursor = job.word_start(heard)  # resume on the word that was cut off
                    job.wake.wait(0.25)
                    job.wake.clear()
                    continue
                chunk = job.slice(job.cursor, step)
                if chunk is None:
                    if job.synth_done:
                        break
                    halt(drain=True)  # synthesis is behind: restart with a fresh clock on new audio
                    job.wake.wait(0.05)
                    job.wake.clear()
                    continue
                if job.run is None:
                    stream.start()
                    job.run = (job.cursor, time.monotonic() + latency)
                stream.write(chunk)
                job.cursor += len(chunk)
            halt(drain=not job.cancel.is_set())
        finally:
            job.seconds = job.cursor / SR
            try:
                halt()
            finally:
                stream.close()


def _mmss(samples):
    s = int(samples / SR)
    return f"{s // 60}:{s % 60:02d}"


class Popup:
    """The control popup: back a sentence, pause, skip a sentence, stop, and an RSVP word
    display that follows the audio. It runs on its own Tk thread inside the server, which
    outlives the terminal that asked for the reading. It stays
    on top and never takes keyboard focus. Click the word to pause; drag anywhere that is
    not a button to move it. Position and the word display's on/off live in .run/popup.json."""

    BG, FG, DIM, HOVER, EDGE = "#18181c", "#ececf0", "#70707a", "#2d2d36", "#3a3a44"
    PIVOT, ACCENT, STOP, TRACK = "#ff5f56", "#4ea1ff", "#ff6b6b", "#2a2a31"
    PREFS = RUN / "popup.json"
    LINGER = 1.5  # seconds the popup stays after a reading ends, so back-to-back reads do not flicker
    PLAY, PAUSE, BACK, SKIP, STOPSQ, CLOSE = "", "", "", "", "", ""  # Segoe MDL2 Assets

    def __init__(self, speaker):
        self.sp = speaker
        self.job = self.dismissed = self.drawn = self.hint = None
        self.shown = False
        self.last_live = 0.0
        self.grab = None
        self.clock_text = self.pause_glyph = None
        self.warned = False
        threading.Thread(target=self._main, name="popup", daemon=True).start()

    # ----- setup -----

    def _main(self):
        try:
            self._build()
        except Exception as e:  # noqa: BLE001 -- no desktop, no popup; the voice still works
            log(f"popup unavailable: {type(e).__name__}: {e}")
            return
        log("popup ready")
        self.root.after(200, self._tick)
        self.root.mainloop()

    def _build(self):
        import tkinter as tk
        import tkinter.font as tkfont
        if os.name == "nt":
            import ctypes
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)  # crisp text; Tk then scales points by the DPI
            except (AttributeError, OSError):
                pass
        self.prefs = self._load_prefs()
        r = self.root = tk.Tk()
        r.withdraw()
        r.title("say")
        r.overrideredirect(True)
        r.attributes("-topmost", True)
        r.configure(bg=self.BG, highlightthickness=1, highlightbackground=self.EDGE, highlightcolor=self.EDGE)
        s = self.scale = r.winfo_fpixels("1i") / 96
        self.W, self.H = int(420 * s), int(84 * s)
        families = set(r.tk.call("font", "families"))
        if "Segoe UI Semibold" in families:
            self.f_word = tkfont.Font(root=r, family="Segoe UI Semibold", size=26)
            self.f_word_small = tkfont.Font(root=r, family="Segoe UI Semibold", size=17)
            self.f_ui = tkfont.Font(root=r, family="Segoe UI", size=9)
        else:  # off Windows: Tk maps Helvetica to the system sans
            self.f_word = tkfont.Font(root=r, family="Helvetica", size=26, weight="bold")
            self.f_word_small = tkfont.Font(root=r, family="Helvetica", size=17, weight="bold")
            self.f_ui = tkfont.Font(root=r, family="Helvetica", size=9)
        if "Segoe MDL2 Assets" in families:
            self.f_icon = tkfont.Font(root=r, family="Segoe MDL2 Assets", size=11)
        else:  # its icons are private-use code points: use plain Unicode media symbols
            self.PLAY, self.PAUSE, self.BACK, self.SKIP, self.STOPSQ, self.CLOSE = "▶", "⏸", "⏮", "⏭", "⏹", "✕"
            self.f_icon = tkfont.Font(root=r, size=12)

        self.word = tk.Canvas(r, width=self.W, height=self.H, bg=self.BG, highlightthickness=0)
        self.bar = tk.Canvas(r, width=self.W, height=max(2, round(3 * s)), bg=self.TRACK, highlightthickness=0)
        self.fill = self.bar.create_rectangle(0, 0, 0, 100, fill=self.ACCENT, width=0)
        row = tk.Frame(r, bg=self.BG)
        self.b_back = self._button(row, self.BACK, lambda: self.sp.seek(-1), "Back a sentence")
        self.b_pause = self._button(row, self.PAUSE, self.sp.toggle_pause, "Pause / resume (or click the word)")
        self.b_skip = self._button(row, self.SKIP, lambda: self.sp.seek(+1), "Skip a sentence")
        self.b_stop = self._button(row, self.STOPSQ, self.sp.stop, "Stop reading", fg=self.STOP)
        self._button(row, self.CLOSE, self._dismiss, "Hide until the next reading", side="right")
        self._button(row, "Aa", self._toggle_words, "Word display on / off", side="right", font=self.f_ui)
        self.clock = tk.Label(row, text="", font=self.f_ui, fg=self.DIM, bg=self.BG, anchor="w", padx=int(8 * s))
        self.clock.pack(side="left", fill="x", expand=True)
        if self.prefs.get("words", True):
            self.word.pack(side="top")
        self.bar.pack(side="top", fill="x")
        row.pack(side="top", fill="x")
        self.row = row
        for w in (self.word, self.bar, row, self.clock):
            w.bind("<ButtonPress-1>", self._press)
            w.bind("<B1-Motion>", self._drag)
            w.bind("<ButtonRelease-1>", self._release)
        self._draw_guides()

    def _button(self, parent, glyph, action, hint, fg=None, side="left", font=None):
        import tkinter as tk
        b = tk.Label(parent, text=glyph, font=font or self.f_icon, fg=fg or self.FG, bg=self.BG,
                     width=4, pady=int(6 * self.scale))

        def release(e):
            if 0 <= e.x < b.winfo_width() and 0 <= e.y < b.winfo_height():
                action()
        b.bind("<Enter>", lambda e: (b.configure(bg=self.HOVER), self._set_hint(hint)))
        b.bind("<Leave>", lambda e: (b.configure(bg=self.BG), self._set_hint(None)))
        b.bind("<ButtonRelease-1>", release)
        b.pack(side=side)
        return b

    def _draw_guides(self):
        """Spritz-style rails: two faint lines with a notch over the focus letter."""
        c, s = self.word, self.scale
        x = self.W * 0.38
        for y, notch in ((self.H * 0.16, 1), (self.H * 0.84, -1)):
            c.create_line(self.W * 0.06, y, self.W * 0.94, y, fill=self.TRACK, width=max(1, round(s)))
            c.create_line(x, y, x, y + notch * 7 * s, fill=self.DIM, width=max(1, round(s)))

    # ----- the loop -----

    def _tick(self):
        try:
            self._update()
        except Exception as e:  # noqa: BLE001
            if not self.warned:
                log(f"popup error: {type(e).__name__}: {e}")
                self.warned = True
        self.root.after(30 if self.shown else 150, self._tick)

    def _update(self):
        job = self.sp.current
        now = time.monotonic()
        if job is not None and job.popup and not job.out and job is not self.dismissed:
            self.last_live = now
            if job is not self.job:
                self.job, self.drawn = job, None
            if not self.shown:
                self._show()
            self._render(job)
        elif self.shown and now - self.last_live > self.LINGER:
            self._hide()

    def _render(self, job):
        heard = job.heard()
        i = job.word_index(heard)
        text = job.words[i][1] if i >= 0 else ("" if job.words else "…")
        key = (id(job), i, text, job.paused)
        if key != self.drawn:
            self._draw_word(text, job.paused, waiting=not job.words)
            self.drawn = key
        total = job.total
        self.bar.coords(self.fill, 0, 0, int(self.bar.winfo_width() * (heard / total if total else 0)), 100)
        glyph = self.PLAY if job.paused else self.PAUSE
        if glyph != self.pause_glyph:
            self.b_pause.configure(text=glyph)
            self.pause_glyph = glyph
        clock = f"{_mmss(heard)} / {_mmss(total)}" + ("" if job.synth_done else "+")
        if job.paused:
            clock += "   paused"
        self._set_clock(clock)

    def _draw_word(self, text, paused, waiting=False):
        """One word, its focus letter red and fixed under the notch (the optimal
        recognition point: about a third in), so the eye never has to move."""
        c, s = self.word, self.scale
        c.delete("w")
        if not text:
            return
        core = re.search(r"\w(?:.*\w)?", text)
        lead, n = (core.start(), len(core.group())) if core else (0, len(text))
        k = min(len(text) - 1, lead + (0 if n <= 1 else 1 if n <= 5 else 2 if n <= 9 else 3 if n <= 13 else 4))
        left, piv, right = text[:k], text[k], text[k + 1:]
        x, y, pad = self.W * 0.38, self.H / 2, 10 * s
        for font in (self.f_word, self.f_word_small):
            half = font.measure(piv) / 2
            if font.measure(left) + half <= x - pad and font.measure(right) + half <= self.W - x - pad:
                break
        fg = self.DIM if paused or waiting else self.FG
        c.create_text(x - half, y, text=left, anchor="e", font=font, fill=fg, tags="w")
        c.create_text(x, y, text=piv, anchor="center", font=font, fill=fg if waiting else self.PIVOT, tags="w")
        c.create_text(x + half, y, text=right, anchor="w", font=font, fill=fg, tags="w")

    def _set_clock(self, text):
        self.clock_text = text
        if self.hint is None and self.clock.cget("text") != text:
            self.clock.configure(text=text)

    def _set_hint(self, hint):
        self.hint = hint
        self.clock.configure(text=hint if hint is not None else (self.clock_text or ""))

    # ----- show, hide, move -----

    def _show(self):
        r = self.root
        x, y = self._place()
        r.geometry(f"+{x}+{y}")
        r.deiconify()
        r.attributes("-topmost", True)
        self._win32_style()
        self.shown = True

    def _hide(self):
        self.root.withdraw()
        self.shown = False

    def _dismiss(self):
        self.dismissed = self.sp.current
        self._hide()

    def _toggle_words(self):
        self.prefs["words"] = not self.prefs.get("words", True)
        if self.prefs["words"]:
            self.word.pack(side="top", before=self.bar)
            self.drawn = None
        else:
            self.word.pack_forget()
        self._save_prefs()

    def _press(self, e):
        self.grab = (e.x_root, e.y_root, self.root.winfo_x(), self.root.winfo_y(), e.widget, False)

    def _drag(self, e):
        if not self.grab:
            return
        x0, y0, wx, wy, widget, moved = self.grab
        dx, dy = e.x_root - x0, e.y_root - y0
        if moved or abs(dx) + abs(dy) > 4 * self.scale:
            self.grab = (x0, y0, wx, wy, widget, True)
            self.root.geometry(f"+{wx + dx}+{wy + dy}")

    def _release(self, e):
        grab, self.grab = self.grab, None
        if not grab:
            return
        if grab[5]:
            self.prefs["x"], self.prefs["y"] = self.root.winfo_x(), self.root.winfo_y()
            self._save_prefs()
        elif grab[4] is self.word:
            self.sp.toggle_pause()

    def _place(self):
        """The saved spot if it is still on a monitor (docks come and go), else bottom center."""
        r = self.root
        r.update_idletasks()
        w, h = r.winfo_reqwidth(), r.winfo_reqheight()
        x, y = self.prefs.get("x"), self.prefs.get("y")
        if isinstance(x, int) and isinstance(y, int) and self._on_screen(x + w // 2, y + h // 2):
            return x, y
        left, top, right, bottom = self._work_area()
        return left + (right - left - w) // 2, bottom - h - int(56 * self.scale)

    # ----- Windows -----

    def _win32_style(self):
        """No activation (clicks never steal keyboard focus), no taskbar button, round corners."""
        if os.name != "nt":
            return
        import ctypes
        import ctypes.wintypes as wt
        u = ctypes.windll.user32
        u.GetAncestor.restype = wt.HWND
        u.GetAncestor.argtypes = [wt.HWND, ctypes.c_uint]
        u.GetWindowLongPtrW.restype = ctypes.c_ssize_t
        u.GetWindowLongPtrW.argtypes = [wt.HWND, ctypes.c_int]
        u.SetWindowLongPtrW.argtypes = [wt.HWND, ctypes.c_int, ctypes.c_ssize_t]
        hwnd = u.GetAncestor(self.root.winfo_id(), 2)  # GA_ROOT: Tk's wrapper window
        if not hwnd:
            return
        ex = u.GetWindowLongPtrW(hwnd, -20)  # GWL_EXSTYLE
        want = ex | 0x08000000 | 0x80        # WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW
        if want != ex:
            u.SetWindowLongPtrW(hwnd, -20, want)
        try:
            pref = ctypes.c_int(2)  # DWMWA_WINDOW_CORNER_PREFERENCE = DWMWCP_ROUND (Windows 11)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(wt.HWND(hwnd), 33, ctypes.byref(pref), 4)
        except (AttributeError, OSError):
            pass

    def _on_screen(self, x, y):
        if os.name != "nt":
            return 0 <= x < self.root.winfo_screenwidth() and 0 <= y < self.root.winfo_screenheight()
        import ctypes
        import ctypes.wintypes as wt
        u = ctypes.windll.user32
        u.MonitorFromPoint.restype = ctypes.c_void_p
        u.MonitorFromPoint.argtypes = [wt.POINT, wt.DWORD]
        return bool(u.MonitorFromPoint(wt.POINT(x, y), 0))  # MONITOR_DEFAULTTONULL

    def _work_area(self):
        if os.name == "nt":
            import ctypes
            import ctypes.wintypes as wt
            rc = wt.RECT()
            if ctypes.windll.user32.SystemParametersInfoW(0x30, 0, ctypes.byref(rc), 0):  # SPI_GETWORKAREA
                return rc.left, rc.top, rc.right, rc.bottom
        return 0, 0, self.root.winfo_screenwidth(), self.root.winfo_screenheight()

    def _load_prefs(self):
        try:
            return json.loads(self.PREFS.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_prefs(self):
        try:
            RUN.mkdir(exist_ok=True)
            self.PREFS.write_text(json.dumps(self.prefs), encoding="utf-8")
        except OSError:
            pass


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)


def handle(conn, speaker, state):
    try:
        conn.settimeout(30)
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = conn.recv(1 << 16)
            if not chunk:
                break
            buf += chunk
        req = json.loads(buf.decode("utf-8", "replace") or "{}")
        cmd = req.get("cmd")
        conn.settimeout(None)
        if cmd == "ping":
            reply = {"ok": True}
        elif cmd == "status":
            cur = speaker.current
            reply = {"ok": True, "device": speaker.device, "speaking": speaker.busy(),
                     "paused": bool(cur and cur.paused),
                     "voices_loaded": sorted(speaker.pipelines), "pid": os.getpid(),
                     "up_min": round((time.time() - state["t0"]) / 60, 1), "idle_min": IDLE_MIN}
        elif cmd == "stop":
            reply = {"ok": True, "stopped": speaker.stop()}
        elif cmd == "pause":
            reply = {"ok": True, "paused": speaker.toggle_pause()}
        elif cmd == "quit":
            speaker.stop()
            state["quit"] = True
            reply = {"ok": True}
            conn.sendall((json.dumps(reply) + "\n").encode())
            conn.close()
            state["srv"].close()  # makes the accept loop raise and exit at once
            return
        elif cmd == "say":
            job = Job(req["text"], req.get("voice", DEFAULT_VOICE), float(req.get("rate", 1.0)), req.get("out"),
                      popup=req.get("popup", True))
            words = len(job.text.split())
            log(f"say {words} words, {job.voice}, rate {job.rate}" + (f", out {job.out}" if job.out else ""))
            speaker.submit(job)
            if req.get("wait") or job.out:
                job.done.wait()
                reply = {"ok": job.error is None, "error": job.error, "words": words,
                         "seconds": round(job.seconds, 1), "cancelled": job.cancel.is_set()}
            else:
                reply = {"ok": True, "words": words}
        else:
            reply = {"ok": False, "error": f"unknown cmd {cmd!r}"}
        conn.sendall((json.dumps(reply) + "\n").encode())
    except Exception as e:  # noqa: BLE001
        log(f"handler error: {type(e).__name__}: {e}")
    finally:
        conn.close()


def serve(force_cpu=False):
    import logging
    import warnings
    warnings.filterwarnings("ignore")
    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
    RUN.mkdir(exist_ok=True)
    PIDFILE.write_text(str(os.getpid()))
    t0 = time.time()
    log(f"loading Kokoro on {'cpu' if force_cpu else 'cuda if available'} (pid {os.getpid()})")
    speaker = Speaker(force_cpu)
    if POPUP:
        Popup(speaker)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if os.name != "nt":
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", PORT))
    srv.listen(8)
    srv.settimeout(2.0)
    log(f"ready on 127.0.0.1:{PORT}, device {speaker.device}, load {time.time() - t0:.1f}s, idle exit {IDLE_MIN:g} min")
    state = {"t0": t0, "quit": False, "srv": srv}
    last = time.time()
    try:
        while not state["quit"]:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                if not speaker.busy() and time.time() - last > IDLE_MIN * 60:
                    log("idle, exiting")
                    break
                continue
            except OSError:
                break  # listening socket closed by --quit
            last = time.time()
            threading.Thread(target=handle, args=(conn, speaker, state), daemon=True).start()
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()
        try:
            if PIDFILE.read_text().strip() == str(os.getpid()):
                PIDFILE.unlink()
        except OSError:
            pass
        log("bye")
        os._exit(0)


# ---------- client ----------

def request(msg, connect_timeout=2.0):
    s = socket.create_connection(("127.0.0.1", PORT), timeout=connect_timeout)
    try:
        s.sendall((json.dumps(msg) + "\n").encode())
        s.settimeout(None)
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(1 << 16)
            if not chunk:
                break
            buf += chunk
        return json.loads(buf.decode("utf-8", "replace") or "{}")
    finally:
        s.close()


def server_alive():
    try:
        return request({"cmd": "ping"}).get("ok", False)
    except OSError:
        return False


def _pid_is_our_python(pid: int) -> bool:
    """True only if PID is our server (guards against PID reuse): on Windows this venv's
    python.exe, on Linux a process whose command line runs say.py --serve, elsewhere never."""
    if os.name != "nt":
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        except OSError:
            return False
        return b"--serve" in cmd and any(c.endswith(b"say.py") for c in cmd)
    import ctypes
    import ctypes.wintypes as wt
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = wt.DWORD(1024)
        if not k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            return False
        return Path(buf.value).resolve() == Path(sys.executable).resolve()
    finally:
        k32.CloseHandle(h)


def kill_stale_server():
    try:
        pid = int(PIDFILE.read_text().strip())
    except (OSError, ValueError):
        return False
    if pid != os.getpid() and _pid_is_our_python(pid):
        try:
            os.kill(pid, signal.SIGTERM)
            return True
        except OSError:
            pass
    return False


def broker_start(quiet=False):
    """Ask SAY_BROKER to start the server, so its parent is a process that outlives any
    terminal. Some terminals (Orca) kill everything a tab started when the tab closes and
    refuse to let a child break away, which killed the warm server whenever a Read aloud
    button closed its tab. True once the server answers."""
    import urllib.request
    if not BROKER:
        return False
    try:
        req = urllib.request.Request(BROKER, data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=3) as r:
            if not json.loads(r.read().decode("utf-8", "replace") or "{}").get("ok"):
                return False
    except (OSError, ValueError):
        return False
    if not quiet:
        print("say: starting the GPU server via SAY_BROKER (first use, about 10-40 s)...", file=sys.stderr, flush=True)
    deadline = time.time() + 180
    while time.time() < deadline:
        time.sleep(0.5)
        if server_alive():
            return True
    return False


def ensure_server(force_cpu=False, quiet=False, use_broker=True):
    if server_alive():
        return
    if use_broker and not force_cpu and broker_start(quiet):
        return
    RUN.mkdir(exist_ok=True)
    kill_stale_server()
    if not quiet:
        print("say: starting the GPU server (first use, about 10-40 s)...", file=sys.stderr, flush=True)
    # Every say.conf line is already in os.environ, so the server inherits the knobs and
    # anything else put there (a process label a dashboard reads, say).
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    flags = 0
    if os.name == "nt":
        # Why: a hidden console, not DETACHED_PROCESS. The venv's python.exe is a stub that
        # runs the real interpreter as a console child; with no console to inherit, Windows
        # opened a new one for it, a visible Windows Terminal window, and closing that
        # window aborted the server mid-sentence (torch's Fortran runtime exits on a
        # console close event).
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    with open(LOGFILE, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--serve"] + (["--cpu"] if force_cpu else []),
                                stdin=subprocess.DEVNULL, stdout=logf, stderr=subprocess.STDOUT,
                                env=env, creationflags=flags, close_fds=True,
                                start_new_session=os.name != "nt")  # POSIX: survive the terminal's hangup
    deadline = time.time() + 180
    while time.time() < deadline:
        time.sleep(0.5)
        if server_alive():
            return
        if proc.poll() is not None:
            sys.exit(f"say: server exited with {proc.returncode}, see {LOGFILE}")
    sys.exit(f"say: server did not come up in time, see {LOGFILE}")


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="*", help="words to speak (or pipe text in, or use --file)")
    ap.add_argument("-f", "--file", help="read text from this file")
    ap.add_argument("-v", "--voice", default=VOICE, help=f"voice id (default {VOICE}, SAY_VOICE); see --list-voices")
    ap.add_argument("-r", "--rate", type=float, default=RATE, help=f"speed multiplier (default {RATE:g}, SAY_RATE)")
    ap.add_argument("-w", "--wait", action="store_true", help="block until the audio finishes")
    ap.add_argument("-o", "--out", help="write a 24 kHz WAV here instead of playing")
    ap.add_argument("--raw", action="store_true", help="skip the Markdown cleanup")
    ap.add_argument("--cpu", action="store_true", help="start the server on the CPU instead of CUDA")
    ap.add_argument("-q", "--quiet", action="store_true", help="no status lines")
    ap.add_argument("--list-voices", action="store_true")
    ap.add_argument("--stop", action="store_true", help="silence a running say")
    ap.add_argument("--pause", action="store_true", help="pause, or resume a paused reading")
    ap.add_argument("--no-popup", action="store_true", help="speak without the control popup")
    ap.add_argument("--status", action="store_true", help="server state")
    ap.add_argument("--quit", action="store_true", help="shut the server down (frees the GPU)")
    ap.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--start", action="store_true", help=argparse.SUPPRESS)  # what a SAY_BROKER runs
    args = ap.parse_args()

    if args.serve:
        serve(args.cpu)
        return 0
    if args.start:
        ensure_server(args.cpu, quiet=True, use_broker=False)
        return 0
    if args.list_voices:
        for k, v in VOICES.items():
            print(f"{k:12s} {v}" + ("  (default)" if k == VOICE else ""))
        return 0
    if args.status:
        if not server_alive():
            print("say: server not running")
            return 0
        s = request({"cmd": "status"})
        print(f"say: server pid {s['pid']} on {s['device']}, up {s['up_min']} min, "
              f"{'paused' if s.get('paused') else 'speaking' if s['speaking'] else 'idle'}, "
              f"voices {','.join(s['voices_loaded'])}, "
              f"exits after {s['idle_min']:g} idle min")
        return 0
    if args.stop:
        if not server_alive():
            print("say: nothing playing")
            return 0
        print("say: stopped" if request({"cmd": "stop"}).get("stopped") else "say: nothing playing")
        return 0
    if args.pause:
        state = request({"cmd": "pause"}).get("paused") if server_alive() else None
        print({True: "say: paused", False: "say: resumed"}.get(state, "say: nothing playing"))
        return 0
    if args.quit:
        if server_alive():
            request({"cmd": "quit"})
            deadline = time.time() + 10
            while server_alive() and time.time() < deadline:
                time.sleep(0.2)
            print("say: server stopped" if not server_alive() else "say: server still answering after quit")
        elif kill_stale_server():
            print("say: killed a stale server")
        else:
            print("say: server not running")
        return 0

    text = read_input(args)
    if not args.raw:
        text = clean_markdown(text)
    if not text.strip():
        print("say: no text", file=sys.stderr)
        return 1
    if args.voice not in VOICES and not re.fullmatch(r"[ab][fm]_[a-z]+", args.voice):
        print(f"say: unknown voice {args.voice!r}, see --list-voices", file=sys.stderr)
        return 2

    ensure_server(args.cpu, args.quiet)
    out = str(Path(args.out).resolve()) if args.out else None
    reply = request({"cmd": "say", "text": text, "voice": args.voice, "rate": args.rate,
                     "wait": args.wait, "out": out, "popup": not args.no_popup})
    if not reply.get("ok"):
        print(f"say: {reply.get('error')}", file=sys.stderr)
        return 1
    if out and not args.quiet:
        print(f"say: wrote {out} ({reply.get('seconds')}s of audio)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
