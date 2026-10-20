"""suzu - a small bell for your LLM engine: live monitor for a
speculative-decoding inference server.

This is the public single-file release of an internal tool.

The full-screen speculative-decoding panel idea comes from a dashboard posted
in the NVIDIA DGX Spark forum thread; the 2026-09 rework borrowed visual ideas
from DingoOz/llm-visuals (MIT): half-block big digits, VU gauges with
peak-hold, attack/release fade, the expert heat map, theme-aware RGB ramps,
and the views (`p` performance, `b` token pipeline).

*** The measurement that dictates this codebase's whole architecture ***
llama.cpp's /metrics counters DO NOT MOVE during a generation. They are
flushed at request completion. Measured: 6.2 s of active slot produced delta
0 on tokens_predicted_total and on every spec_decode_* counter, while the
model was emitting 20.5 tok/s. A 1 s monitor built on /metrics alone would
read "0 tok/s" for minutes under full load. Hence the split:

  live rate       <- /slots  next_token.n_decoded   (advances every token)
  prefill         <- /slots  n_prompt_tokens_processed
  KV/prefix cache <- /slots  (llama.cpp exposes NEITHER on /metrics)
  acceptance, tau, p0..pN <- /metrics, shown WITH THE AGE of the last closed
                             request rather than passed off as instantaneous

Sources, all already exposed - no engine patch required:
  /slots    next_token.n_decoded, n_prompt_tokens{,_processed,_cache}, n_ctx,
            is_processing
  /metrics  spec_decode_num_{draft,accepted}_tokens_total,
            ..._num_drafts_total, ..._accepted_tokens_per_pos_total{position=N}
  ps -ww    server cmdline -> draft model, spec method, k, uptime, plus what
            vLLM adds: quantisation, kv dtype, tensor-parallel size, nnodes,
            max-num-seqs, max model len (the model strip reads these)
  nvidia-smi  GPU utilisation, VRAM, watts, degC, SM clock (the GB10 panel)
  --lanes-url  optional gateway lanes endpoint: the queue you actually sit
            in, which /metrics cannot see (the gateway answers before
            touching the scheduler, so polling it is free)
  --rank1-url  rank 1 of a tensor-parallel pair (Prometheus textformat),
            probed only when the cmdline says --nnodes 2: the cluster is
            only as alive as its quietest rank

*** 29-08: TabbyAPI/exllamav3 (entry glm53x, the daily driver at the time)
exposes NONE of this - /metrics and /slots answer 404. The `tabby` adapter
(fetch/tabby.py, synth_tabby) rebuilds the same keys from /props + the
gateway stats + the `Metrics (ID: ...)` line in the engine log, so the rest
of the code does not change. Live tokens come from the gateway, the only
component that sees them pass.

*** 30-09: TensorFold (entry glm53tf, the main engine since) has /metrics
but in its own words (tensorfold_*), flushed at request close like
llama.cpp, and no /slots. fetch/tensorfold.py (synth_tensorfold) maps it onto
the same keys: closed counters from /metrics, the in-flight part from the
gateway's SSE events (one event per verify round) times tokens/round, the
last request's prompt/cached from its request log. It counts no drafted
tokens, so the panel shows tokens/round instead of an acceptance percentage.
The same request log gives the all-time totals (`since DD-MM` row), which
survive engine restarts; the /metrics ones start again at every boot.

Views:  full (default) . p = performance zoom . b = token pipeline (vertical
VU meters + bottleneck verdict, the repo's bandwidth view adapted to GB10
unified memory: no PCIe hop, but the gateway lanes and the rank-1 link are
ours) . g = full-screen layer grid.

usage: suzu.py [interval] [--port N] [--theme NAME] [--color auto|truecolor|256]
               [--tokens-db PATH] [--rank1-url URL] [--lanes-url URL]
               [--no-log] [--once] [--list-themes] [--layers] [--demo]
keys:  t = cycle theme   l = toggle log   L = layer grid   g = full-screen layer grid
       p = performance zoom   b = token pipeline   q = quit

The code lives in this package (see ARCHITECTURE.md); this module is the
command line front end and the redraw loop.
"""
import atexit
import os
import select
import signal
import socket
import subprocess
import sys
import time
from urllib.request import urlopen

from .ansi import ESC, RST
from . import config
from .themes import THEMES, THEME_ORDER, set_color_mode, _theme_read
from .render.view import View

try:
    import termios
    import tty
    HAVE_TTY = True
except ImportError:                     # not a POSIX terminal - keys disabled
    HAVE_TTY = False


def term_size(fallback=(100, 40)):
    """Real terminal size, from the tty.

    COLUMNS/LINES in the environment are a snapshot taken by the shell that
    launched us and never change afterwards: a fish (or a bash that exported
    them) hands suzu the size the pane had at launch, so a resize would keep
    drawing the old layout forever. The kernel's winsize is the only live
    source; the env is a last resort for when stdout/stdin are pipes.
    """
    for fd in (1, 0, 2):
        try:
            s = os.get_terminal_size(fd)
            if s.columns > 0 and s.lines > 0:
                return s
        except (OSError, ValueError, AttributeError):
            pass
    try:
        cols = int(os.environ.get("COLUMNS", ""))
        lines = int(os.environ.get("LINES", ""))
    except ValueError:
        cols = lines = 0
    return os.terminal_size((cols or fallback[0], lines or fallback[1]))


def _free_port():
    """A port the kernel says is free right now (bind to 0, read it, release)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


_DEMO = []


def _demo_stop():
    for proc in _DEMO:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
    _DEMO.clear()


def _demo_start(port, port_given):
    """Start demo/fake_engine.py next to the script; return the port to watch.

    The repo ships the synthetic engine; a bare `curl` of suzu.py does not.
    Returns None on failure (the reason is already on stderr).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    cand = [os.path.join(here, "demo", "fake_engine.py"),
            os.path.join(here, os.pardir, "demo", "fake_engine.py")]
    script = next((p for p in cand if os.path.isfile(p)), None)
    if script is None:
        sys.stderr.write("--demo: demo/fake_engine.py not found next to suzu.py "
                         "(it ships in the repo; a lone suzu.py has no demo)\n")
        return None
    if not port_given:
        port = _free_port()
    proc = subprocess.Popen([sys.executable, script, "--port", str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _DEMO.append(proc)
    atexit.register(_demo_stop)
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if proc.poll() is not None:
            sys.stderr.write("--demo: fake engine exited (code %s)\n" % proc.returncode)
            return None
        try:
            with urlopen("http://127.0.0.1:%d/health" % port, timeout=0.5):
                return port
        except Exception:
            time.sleep(0.1)
    sys.stderr.write("--demo: fake engine did not answer on :%d within 5s\n" % port)
    return None


def main():
    port, interval, show_log, once = 30001, 1.0, True, False
    demo, port_given = False, False
    layers = False
    theme = _theme_read() or os.environ.get("SUZU_THEME_NAME") or "yuzu"
    color_mode = None
    tokens_db = rank1_url = lanes_url = None
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        theme = "mono"  # piped/NO_COLOR default; an explicit --theme still wins below
    args, rest, i = sys.argv[1:], [], 0
    while i < len(args):
        a = args[i]
        if a == "--port":
            i += 1
            port = int(args[i])
            port_given = True
        elif a.startswith("--port="):
            port = int(a.split("=", 1)[1])
            port_given = True
        elif a == "--theme":
            i += 1
            theme = args[i]
        elif a.startswith("--theme="):
            theme = a.split("=", 1)[1]
        elif a == "--color":
            i += 1
            color_mode = args[i]
        elif a.startswith("--color="):
            color_mode = a.split("=", 1)[1]
        elif a == "--list-themes":
            print("themes: " + ", ".join(THEME_ORDER))
            return 0
        elif a in ("--no-log", "-n"):
            show_log = False
        elif a == "--once":
            once = True
        elif a == "--tokens-db":
            i += 1
            tokens_db = args[i]
        elif a.startswith("--tokens-db="):
            tokens_db = a.split("=", 1)[1]
        elif a == "--rank1-url":
            i += 1
            rank1_url = args[i]
        elif a.startswith("--rank1-url="):
            rank1_url = a.split("=", 1)[1]
        elif a == "--lanes-url":
            i += 1
            lanes_url = args[i]
        elif a.startswith("--lanes-url="):
            lanes_url = a.split("=", 1)[1]
        elif a in ("--layers", "-g"):
            layers = True
        elif a == "--demo":
            demo = True
        elif a in ("-h", "--help"):
            print(__doc__)
            return 0
        else:
            rest.append(a)
        i += 1
    if theme not in THEMES:
        sys.stderr.write("unknown theme %r - known: %s\n"
                         % (theme, ", ".join(THEME_ORDER)))
        return 2
    if color_mode is not None:
        set_color_mode(color_mode)
    if rest:
        try:
            interval = max(0.25, float(rest[0]))
        except ValueError:
            pass

    if demo:
        got = _demo_start(port, port_given)
        if got is None:
            return 2
        port = got

    if tokens_db:
        config.HERMES_DB = os.path.expanduser(tokens_db)
    config.LANES_URL = lanes_url or None
    config.RANK1_URL = rank1_url or None
    if config.RANK1_URL:
        from .fetch import rank1
        rank1._RANK1 = rank1._rank1_feed(config.RANK1_URL)   # rebuild the process-wide feed with the URL

    view = View(port, interval, show_log, theme)
    if layers:
        view.mode = "layers"
    out = sys.stdout

    if once:
        size = term_size()
        w, h = min(size.columns, 500), size.lines
        if os.environ.get("HERDR_ENV"):
            # herdr replaces the last cell of a row that spans the width exactly
            # with a wrap glyph, hiding the box border: leave that column empty.
            w = max(20, w - 1)
        # One sample has no delta - every rate would read 0 and look like a fault.
        # Take two; the first is only the baseline.
        view.frame(w, h)
        time.sleep(max(1.0, min(2.0, interval)))
        for ln in view.frame(w, h):
            print(ln)
        return 0

    # SIGWINCH: redraw at once on resize instead of waiting out the interval,
    # otherwise a resize leaves a torn panel on screen for up to `interval`.
    resized = {"flag": False}
    try:
        signal.signal(signal.SIGWINCH, lambda *_: resized.__setitem__("flag", True))
    except (AttributeError, ValueError):
        pass

    israw = HAVE_TTY and sys.stdin.isatty()
    old = termios.tcgetattr(sys.stdin) if israw else None

    def bye(*_):
        if old is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
            except termios.error:
                pass
        out.write(ESC + "?25h" + ESC + "?1049l" + RST)
        out.flush()
        sys.exit(0)

    signal.signal(signal.SIGINT, bye)
    signal.signal(signal.SIGTERM, bye)

    if israw:
        tty.setcbreak(sys.stdin.fileno())
    out.write(ESC + "?1049h" + ESC + "?25l")
    lastsize = None
    try:
        while True:
            size = term_size()
            w, h = min(size.columns, 500), size.lines
            if os.environ.get("HERDR_ENV"):
                # Same wrap-glyph cell as above: leave the last column empty.
                w = max(20, w - 1)
            if resized["flag"] or lastsize != (w, h):
                # A shrink leaves stale glyphs to the right and below. Full erase
                # on the resize frame only - doing it every frame flickers.
                out.write(ESC + "2J")
                resized["flag"] = False
                lastsize = (w, h)
            # 05-10 ("the top right glitches: it disappears and reappears moving the whole page"):
            # a "\n" after the LAST line, on a frame as tall as the terminal (the expert map now takes every free row),
            # scrolled the whole screen up one line at every redraw. Lines are joined, the last one has no newline.
            rows = view.frame(w, h)
            buf = [ESC + "H", (ESC + "K\n").join(rows), ESC + "K", ESC + "J"]
            out.write("".join(buf))
            out.flush()

            deadline = time.time() + view.interval
            while True:
                left = deadline - time.time()
                if left <= 0 or resized["flag"]:
                    break
                if not israw:
                    time.sleep(min(left, 0.2))
                    continue
                r, _, _ = select.select([sys.stdin], [], [], min(left, 0.2))
                if not r:
                    continue
                ch = sys.stdin.read(1)
                if ch in ("q", "Q"):
                    bye()
                elif ch in ("t", "T"):
                    view.cycle_theme(1 if ch == "t" else -1)
                    out.write(ESC + "2J")
                    break
                elif ch == "l":
                    view.show_log = not view.show_log
                    out.write(ESC + "2J")
                    break
                elif ch == "L":
                    view.show_layers = not view.show_layers
                    out.write(ESC + "2J")
                    break
                elif ch in ("g", "G"):
                    view.mode = "layers" if view.mode != "layers" else "spec"
                    out.write(ESC + "2J")
                    break
                elif ch in ("p", "P"):
                    view.view = "perf" if view.view != "perf" else "full"
                    out.write(ESC + "2J")
                    break
                elif ch == "b":
                    view.view = "pipes" if view.view != "pipes" else "full"
                    out.write(ESC + "2J")
                    break
    finally:
        bye()
