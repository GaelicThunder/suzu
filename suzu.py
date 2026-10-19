#!/usr/bin/env python3
"""suzu - a small bell for your LLM engine: live monitor for a
speculative-decoding inference server.

*** GENERATED FILE - edit suzu/ and run tools/bundle.py, not this file. ***

The measurement that dictates this file's whole architecture:
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

See ARCHITECTURE.md for the module map; this file is those modules inlined.
"""

import json
import os
import re
import select
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import deque
import atexit
import signal
import socket
from urllib.request import urlopen

# ==================== suzu/ansi.py
"""ANSI primitives: SGR helpers, regexes, escape constants."""



ESC = "\033["



RST = ESC + "0m"



def fg(n):
    return ESC + "38;5;%dm" % n



def bg(n):
    return ESC + "48;5;%dm" % n



BOLD = ESC + "1m"



DIM = ESC + "2m"



ANSI_RE = re.compile(r"\033\[[0-9;?]*[a-zA-Z]")



SGR_RE = re.compile(r"\033\[[0-9;]*m")

# ==================== suzu/text.py
"""Text measurement and formatting: visible width, pad, clip, units."""



def vis_len(s):
    return len(SGR_RE.sub("", s))



def pad(s, w):
    n = w - vis_len(s)
    return s + " " * n if n > 0 else s



def clip(s, w):
    """Truncate by VISIBLE characters - len() lies on coloured strings."""
    if w <= 0:
        return ""
    if vis_len(s) <= w:
        return s
    out, seen, i, had = [], 0, 0, False
    while i < len(s) and seen < w:
        if s[i] == "\033":
            j = s.find("m", i)
            if j < 0:
                break
            out.append(s[i:j + 1])
            had = True
            i = j + 1
            continue
        out.append(s[i])
        seen += 1
        i += 1
    # A cut can land inside a coloured run, so close it - but only when the
    # string had any escape at all: appending RST unconditionally painted a
    # reset into plain (mono) output and broke the "no ANSI in mono" contract.
    return "".join(out) + (RST if had else "")



def human(n):
    return "{:,}".format(int(n))



def si_bytes(n):
    """Bytes the short way: 950 B, 12.3 kB, 456 MB, 1.23 GB (decimal units, like the link's line rate)."""
    n = float(n or 0)
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= div:
            v = n / div
            return ("%.2f %s" if v < 10 else "%.1f %s" if v < 100 else "%.0f %s") % (v, unit)
    return "%.0f B" % n



def fmt_hms(sec):
    if sec is None:
        return "--:--"
    h, rem = divmod(int(sec), 3600)
    m, s = divmod(rem, 60)
    return "%dh%02dm" % (h, m) if h else "%02d:%02d" % (m, s)



def ago(sec):
    if sec is None:
        return "never"
    if sec < 90:
        return "%.0fs ago" % sec
    if sec < 5400:
        return "%.0fm ago" % (sec / 60)
    return "%.1fh ago" % (sec / 3600)

# ==================== suzu/config.py
"""Runtime configuration set by the CLI (paths/URLs)."""



TF_REQLOG = os.path.expanduser("~/models/glm53-tf/sessions/requests.jsonl")



HERMES_DB = os.path.expanduser("~/.hermes/state.db")   # --tokens-db overrides



LANES_URL = None        # optional gateway lanes endpoint (--lanes-url)



RANK1_URL = None        # optional rank-1 metrics endpoint (--rank1-url)


class _Config:
    pass


config = _Config()
for _k in ('TF_REQLOG', 'HERMES_DB', 'LANES_URL', 'RANK1_URL'):
    setattr(config, _k, globals()[_k])
del _k

# ==================== suzu/themes.py
"""Themes: role->colour tables, the 5-step quality ramp, truecolor ramps."""



THEMES = {
    # Every theme is multicolour ON PURPOSE. "scale" is the semantic ramp
    # bad -> good, and it is the only thing that tells you WHAT a bar is worth:
    # each cell of a bar is painted by its own position on this ramp, so a bar
    # is its own legend. A theme whose 5 steps sit inside one hue looks pretty
    # and says nothing - 30% and 70% draw the same picture. So every ramp below
    # walks a real hue distance; only the furniture (title/frame/label) carries
    # the theme's identity.
    "neon": {                       # the look of the forum screenshot
        "title": fg(213), "accent": fg(205), "head": fg(44), "frame": fg(238),
        "label": fg(244), "value": fg(231),
        "scale": [fg(196), fg(202), fg(220), fg(118), fg(46)],
    },
    "yuzu": {                       # the warm citrus default (born 2026-09-19)
        "title": fg(220), "accent": fg(184), "head": fg(149), "frame": fg(239),
        "label": fg(250), "value": fg(231),
        "scale": [fg(131), fg(179), fg(220), fg(191), fg(107)],
    },
    "matrix": {
        "title": fg(46), "accent": fg(82), "head": fg(40), "frame": fg(22),
        "label": fg(71), "value": fg(120),
        "scale": [fg(124), fg(166), fg(184), fg(82), fg(46)],
    },
    "amber": {                      # amber CRT
        "title": fg(214), "accent": fg(208), "head": fg(180), "frame": fg(94),
        "label": fg(137), "value": fg(223),
        "scale": [fg(124), fg(166), fg(214), fg(226), fg(154)],
    },
    "phosphor": {                   # green/amber phosphor, EVA terminal (19-10)
        "title": fg(46), "accent": fg(214), "head": fg(184), "frame": fg(28),
        "label": fg(64), "value": fg(120),
        "scale": [fg(130), fg(178), fg(184), fg(82), fg(46)],
    },
    "ice": {
        "title": fg(45), "accent": fg(51), "head": fg(39), "frame": fg(24),
        "label": fg(109), "value": fg(195),
        "scale": [fg(161), fg(135), fg(63), fg(45), fg(51)],
    },
    "grape": {
        "title": fg(141), "accent": fg(177), "head": fg(105), "frame": fg(54),
        "label": fg(139), "value": fg(225),
        "scale": [fg(160), fg(168), fg(176), fg(141), fg(85)],
    },
    "sunset": {
        "title": fg(209), "accent": fg(203), "head": fg(215), "frame": fg(95),
        "label": fg(180), "value": fg(230),
        "scale": [fg(89), fg(131), fg(173), fg(215), fg(229)],
    },
    "oxide": {                      # rust -> patina
        "title": fg(166), "accent": fg(130), "head": fg(108), "frame": fg(58),
        "label": fg(144), "value": fg(230),
        "scale": [fg(88), fg(130), fg(178), fg(107), fg(79)],
    },
    "vapor": {
        "title": fg(212), "accent": fg(207), "head": fg(117), "frame": fg(60),
        "label": fg(146), "value": fg(231),
        "scale": [fg(197), fg(171), fg(99), fg(75), fg(87)],
    },
    "paper": {                      # light terminals: dark inks, still 5 hues
        "title": fg(53), "accent": fg(89), "head": fg(24), "frame": fg(250),
        "label": fg(240), "value": fg(232),
        "scale": [fg(124), fg(130), fg(94), fg(28), fg(22)],
    },
    # NOT in the cycle: colour is the point of this panel. Still reachable by
    # name (--theme mono) and picked automatically for pipes / NO_COLOR / dumb
    # terminals, where the bars fall back to ASCII # and -.
    "mono": {
        "title": "", "accent": "", "head": "", "frame": "", "label": "",
        "value": "", "scale": ["", "", "", "", ""],
    },
}



THEME_ORDER = ["yuzu", "neon", "matrix", "amber", "phosphor", "ice", "grape",
               "sunset", "oxide", "vapor", "paper"]



BLOCKS = "▁▂▃▄▅▆▇█"



DITHER = ("·", "░", "▒", "▓", "█")          # empty, then dithered density: the heatmap's levels and every bar's edge



PULSE = "◐◓◑◒"          # pulsing head on the context row while the engine works



def _rgb(t):
    return ESC + "38;2;%d;%d;%dm" % t



def _lerp_rgb(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))



def _grad(stops, pos):
    pos = max(0.0, min(1.0, pos))
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        if pos <= p1:
            t = 0.0 if p1 == p0 else (pos - p0) / (p1 - p0)
            return _lerp_rgb(c0, c1, t)
    return stops[-1][1]



def _rgb256(t):
    """Nearest 6x6x6 xterm cube cell for an RGB triple."""
    q = [min(5, int(round(v / 255.0 * 5))) for v in t]
    return 16 + 36 * q[0] + 6 * q[1] + q[2]



_TRUECOLOR = None



_COLOR_MODE = None          # None = auto, else "truecolor" | "256" | "off"



def set_color_mode(mode):
    global _COLOR_MODE
    _COLOR_MODE = mode if mode in ("truecolor", "256", "off") else None
    global _TRUECOLOR
    _TRUECOLOR = None



def _have_truecolor():
    if _COLOR_MODE == "truecolor":
        return True
    if _COLOR_MODE in ("256", "off"):
        return False
    global _TRUECOLOR
    if _TRUECOLOR is None:
        ct = (os.environ.get("COLORTERM") or "").lower()
        _TRUECOLOR = "truecolor" in ct or "24bit" in ct
    return _TRUECOLOR



def _col(t):
    """RGB tuple -> SGR; truecolor when advertised, else the nearest
    256-cube cell (also forced by --color 256 / --color off)."""
    if _have_truecolor():
        return _rgb(t)
    return fg(_rgb256(t))



HEAT_COOL = [(0.00, (20, 24, 44)), (0.18, (36, 58, 170)), (0.42, (0, 190, 235)),
             (0.66, (120, 240, 140)), (0.84, (255, 200, 50)), (1.00, (255, 250, 240))]



HEAT_GOLD = [(0.00, (26, 22, 12)), (0.18, (94, 66, 16)), (0.42, (176, 128, 28)),
             (0.66, (228, 168, 54)), (0.84, (255, 208, 63)), (1.00, (255, 248, 225))]



VU_COOL = [(0.00, (40, 200, 110)), (0.55, (140, 230, 80)), (0.72, (255, 200, 40)),
           (0.88, (255, 120, 50)), (1.00, (255, 60, 80))]



VU_GOLD = [(0.00, (146, 176, 58)), (0.55, (214, 186, 44)), (0.72, (244, 208, 63)),
           (0.88, (236, 148, 44)), (1.00, (224, 96, 52))]



FLOW_COOL = [(0.00, (90, 70, 200)), (0.45, (0, 190, 255)),
             (0.80, (120, 250, 230)), (1.00, (255, 255, 255))]



FLOW_GOLD = [(0.00, (122, 92, 22)), (0.45, (201, 162, 39)),
             (0.80, (244, 208, 63)), (1.00, (255, 248, 225))]



GOLD_FAMILY = {"yuzu", "amber", "sunset", "oxide"}


PHOSPHOR_FAMILY = {"phosphor"}


HEAT_PHOS = [(0.00, (8, 18, 10)), (0.18, (14, 60, 30)), (0.42, (30, 140, 60)),
             (0.66, (120, 220, 90)), (0.84, (240, 220, 80)), (1.00, (255, 250, 220))]


VU_PHOS = [(0.00, (40, 200, 90)), (0.55, (150, 230, 80)), (0.72, (250, 200, 50)),
           (0.88, (255, 140, 40)), (1.00, (255, 70, 60))]


FLOW_PHOS = [(0.00, (10, 60, 30)), (0.45, (40, 200, 90)),
             (0.80, (190, 250, 140)), (1.00, (255, 255, 230))]


TRACK_BG_PHOS = (14, 30, 18)



TRACK_BG_COOL = (36, 40, 54)



TRACK_BG_GOLD = (52, 44, 26)



WHITE_FG = (250, 250, 255)



def _bg_col(t):
    if _have_truecolor():
        return ESC + "48;2;%d;%d;%dm" % t
    return bg(_rgb256(t))



BIG_GLYPHS = {
    "0": ["▄▀▄", "█ █", "▀▀▀"], "1": ["▄█ ", " █ ", "▀▀▀"],
    "2": ["▀▀▄", "▄▀ ", "▀▀▀"], "3": ["▀▀▄", " ▀▄", "▀▀ "],
    "4": ["▄ ▄", "▀▀█", "  ▀"], "5": ["█▀▀", "▀▀▄", "▀▀ "],
    "6": ["▄▀▀", "█▀▄", "▀▀▀"], "7": ["▀▀█", " ▄▀", " ▀ "],
    "8": ["▄▀▄", "█▀█", "▀▀▀"], "9": ["▄▀▄", "▀▀█", "▀▀▀"],
    ".": [" ", " ", "▄"], ",": [" ", " ", "▄"], " ": [" ", " ", " "],
}



BIGFONT = True



THEME_STATE = os.environ.get("SUZU_THEME", os.path.expanduser("~/.config/suzu/theme"))



def _theme_read():
    """Last theme chosen with t, or None. Piped/mono runs never persist it."""
    try:
        with open(THEME_STATE) as f:
            n = f.read().strip()
        return n if n in THEMES else None
    except OSError:
        return None



def _theme_save(name):
    try:
        with open(THEME_STATE, "w") as f:
            f.write(name + "\n")
    except OSError:
        pass   # cosmetic state; a failed save must never kill the view



class Theme:
    def __init__(self, name):
        self.name = name if name in THEMES else "neon"
        self.c = THEMES[self.name]
        self.plain = self.name == "mono"

    def __getattr__(self, k):
        try:
            return self.__dict__["c"][k]
        except KeyError:
            raise AttributeError(k)

    # ramp cutoffs, shared by heat() and by the footer legend so that what the
    # legend shows and what the bars do can never drift apart
    CUTS = (0.25, 0.40, 0.55, 0.75)

    def heat(self, frac):
        """Colour for a 0..1 quality fraction."""
        s = self.c["scale"]
        c = Theme.CUTS
        i = (0 if frac < c[0] else 1 if frac < c[1] else
             2 if frac < c[2] else 3 if frac < c[3] else 4)
        return s[i]

    def heat_bg(self, frac):
        """Background colour for a 0..1 fraction - a filled box, not a glyph.
        Same ramp as heat(); in mono/plain it degrades to empty."""
        if self.plain:
            return ""
        return self.heat(frac).replace("38;5;", "48;5;")

    def legend(self):
        """The ramp itself, so the colours in the bars can be decoded."""
        if self.plain:
            return ""
        return "".join(c + "█" for c in self.c["scale"]) + RST

    def rst(self):
        return "" if self.plain else RST

    def bold(self):
        return "" if self.plain else BOLD

    def dim(self):
        return "" if self.plain else DIM

# ==================== suzu/fetch/http.py
"""HTTP + Prometheus text parsing shared by every engine adapter."""



def http_get(url, timeout=4, tries=2):
    """With -np 1 the HTTP handler can wait on the in-flight decode, so an
    isolated timeout is NOT a dead engine - retry before declaring one."""
    for n in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except (urllib.error.URLError, OSError, ValueError):
            if n + 1 < tries:
                time.sleep(0.2)
    return None



METRIC_RE = re.compile(r'^([a-zA-Z_:][\w:]*)(\{[^}]*\})?\s+([-\d.eE+]+)\s*$')



POS_RE = re.compile(r'position="(\d+)"')



def parse_metrics(text):
    out = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        m = METRIC_RE.match(line.strip())
        if not m:
            continue
        try:
            val = float(m.group(3))
        except ValueError:
            continue
        name, labels = m.group(1), m.group(2) or ""
        out[name + labels] = val
        if labels:
            # vLLM labels EVERY metric (engine="0", model_name="..."), llama.cpp
            # none. Without this alias on the bare name every vllm:* lookup below
            # missed and the WHOLE vllm branch was dead code (found 21-08 on dsflash).
            out.setdefault(name, val)
            pos = POS_RE.search(labels)
            if pos:
                out.setdefault('%s{position="%s"}' % (name, pos.group(1)), val)
    return out



def gl(mx, name, *subs):
    """First metric whose key is `name` carrying every one of `subs` in its
    labels. parse_metrics aliases the bare name to whichever label set it saw
    FIRST, which is fine for the metrics vLLM emits once but not for the ones
    split by a label (prompt_tokens_by_source_total has three)."""
    pre = name + "{"
    for k, v in mx.items():
        if k.startswith(pre) and all(s in k for s in subs):
            return v
    return None



def g(mx, *names):
    for n in names:
        if n in mx:
            return mx[n]
    return 0.0

# ==================== suzu/fetch/memory.py
"""Host memory, process RSS and llama.cpp /slots."""



_MEMINFO = {"at": 0.0, "info": None, "pid": None}



def fetch_memory(pid=None):
    """Host memory picture, /proc only - the repo's RAM stage adapted to
    GB10 unified memory, where nvidia-smi reports VRAM as [N/A] because the
    weights and the desktop swim in the same pool. The engine number is its
    full resident set (RssFile+RssAnon): on GB10 the weights sit in CUDA
    unified memory, which RSS cannot tag as 'file', so RssFile alone would
    show a meaningless ~0.4 GB."""
    now = time.time()
    if _MEMINFO["info"] is not None and now - _MEMINFO["at"] < 2.0 \
            and _MEMINFO["pid"] == pid:
        return _MEMINFO["info"]
    mi = {"total": None, "avail": None, "used_frac": None, "rss": None}
    try:
        with open("/proc/meminfo") as f:
            vals = {}
            for line in f:
                k, _, rest = line.partition(":")
                vals[k.strip()] = float(rest.split()[0])   # kB
        mi["total"] = vals.get("MemTotal")
        mi["avail"] = vals.get("MemAvailable")
        if mi["total"]:
            used = mi["total"] - (mi["avail"] or 0.0)
            mi["used_frac"] = max(0.0, min(1.0, used / mi["total"]))
    except (OSError, ValueError):
        pass
    if pid:
        try:
            with open("/proc/%s/status" % pid) as f:
                rf = ra = None
                for line in f:
                    if line.startswith("RssFile:"):
                        rf = float(line.split()[1])    # kB, file-backed
                    elif line.startswith("RssAnon:"):
                        ra = float(line.split()[1])    # kB, anonymous
                if rf is not None or ra is not None:
                    mi["rss"] = (rf or 0.0) + (ra or 0.0)
        except (OSError, ValueError):
            pass
    _MEMINFO["info"], _MEMINFO["at"], _MEMINFO["pid"] = mi, now, pid
    return mi



def fetch_slots(port):
    txt = http_get("http://127.0.0.1:%d/slots" % port, timeout=3, tries=1)
    if not txt:
        return None
    try:
        j = json.loads(txt)
        return j if isinstance(j, list) else None
    except (json.JSONDecodeError, ValueError):
        return None

# ==================== suzu/fetch/gpu.py
"""GPU rows: nvidia-smi first, rocm-smi fallback."""



_GPU_SMI = {"at": 0.0, "rows": None}



def fetch_gpu():
    """One row per GPU: util %, VRAM used/total, watts, degC, SM clock.
    Cached 2 s - the panel ticks faster than a GPU counter meaningfully moves,
    and nvidia-smi is not free (it spawns a process).
    Tries nvidia-smi first; falls back to rocm-smi (AMD ROCm) when nvidia-smi
    is absent or returns no rows."""
    now = time.time()
    if _GPU_SMI["rows"] is not None and now - _GPU_SMI["at"] < 2.0:
        return _GPU_SMI["rows"]
    rows = None
    try:
        rows = []
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,"
             "memory.total,power.draw,temperature.gpu,clocks.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3).stdout
        def _f(s):
            try:
                return float(s)
            except ValueError:
                return None
        for line in out.splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) < 6:
                continue
            rows.append({"util": _f(p[0]), "mem": _f(p[1]), "memtot": _f(p[2]),
                         "w": _f(p[3]), "c": _f(p[4]), "mhz": _f(p[5])})
        if not rows:
            rows = _fetch_rocm_gpu()
    except (OSError, subprocess.SubprocessError, ValueError):
        rows = _fetch_rocm_gpu()
    if rows:
        _GPU_SMI["rows"], _GPU_SMI["at"] = rows, now
        return rows
    return None



def _fetch_rocm_gpu():
    """AMD ROCm fallback via rocm-smi. `rocm-smi --csv` is not supported on this
    build ("NOT_SUPPORTED: Cannot print JSON/CSV output"), so parse the supported
    tabular (concise) output instead. Per device data row, after the
    (Device, Node, DID, GUID, Temp, Power, Mem, Compute, PartID) prefix the
    tokens are: SCLK, MCLK, Fan, Perf, PwrCap, VRAM%, GPU%. rocm-smi reports VRAM
    as a percent only, so absolute used/total (MB) comes from /sys/class/drm; the
    two enumerations are ordered differently, so each device is matched to the card
    whose used/total ratio is closest to its reported VRAM%. Returns rows in the
    same shape as the nvidia-smi parser (util, mem, memtot, w, c, mhz)."""
    try:
        out = subprocess.run(["/opt/rocm/bin/rocm-smi"],
                             capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.SubprocessError, ValueError):
        return None

    def _num(tok):
        try:
            s = "".join(ch for ch in str(tok) if ch.isdigit() or ch in ".-")
            return float(s) if s else None
        except (ValueError, TypeError):
            return None

    cards = []
    try:
        import glob
        for path in sorted(glob.glob("/sys/class/drm/card*/device/mem_info_vram_total")):
            try:
                tot = int(open(path).read().strip()) / 1024.0
            except (OSError, ValueError, AttributeError):
                continue
            used = None
            try:
                used = int(open(path.replace("_total", "_used")).read().strip()) / 1024.0
            except (OSError, ValueError, AttributeError):
                pass
            ratio = (used / tot) if (used is not None and tot) else None
            cards.append({"tot": tot, "used": used, "ratio": ratio, "taken": False})
    except Exception:
        cards = []

    dev_rows = []
    for line in out.splitlines():
        toks = line.split()
        if not toks or not toks[0].isdigit():
            continue
        try:
            dev = int(toks[0])
        except ValueError:
            continue
        tpos = None
        for k, tk in enumerate(toks):
            if "\u00b0" in tk or tk.endswith("C"):
                tpos = k
                break
        if tpos is None or tpos + 11 >= len(toks):
            continue
        gpu_pct = _num(toks[tpos + 11])
        vram_pct = _num(toks[tpos + 10])
        if gpu_pct is None:
            continue
        mclk = _num(toks[tpos + 6])
        sclk = _num(toks[tpos + 5])
        dev_rows.append({
            # util is a PERCENT number (0-100) to match the nvidia-smi parser
            # (fetch_gpu stores the raw nvidia-smi utilization.gpu value, which
            # is 0-100). gpu_cols() and line ~3170 both divide by 100.0 for a
            # 0-1 gauge, and print util as "%.0f%" - so storing a 0-1 fraction
            # here made the gauge read u/100 = 0.003x (empty) and the label
            # print 0.03 -> "0%". This was why suzu showed 0-1% while rocm-smi
            # swung 0->100%.
            "util": gpu_pct,
            "c": _num(toks[tpos]),
            "w": _num(toks[tpos + 1]),
            "mhz": (mclk if mclk is not None else sclk),
            "mem": None,
            "memtot": None,
            "vram_pct": vram_pct,
            "dev": dev,
        })

    rows = []
    for dr in dev_rows:
        best, best_d = None, float("inf")
        if dr["vram_pct"] is not None:
            for card in cards:
                if card["taken"] or card["ratio"] is None:
                    continue
                d = abs(card["ratio"] - dr["vram_pct"] / 100.0)
                if d < best_d:
                    best, best_d = card, d
        if best is not None and best_d <= 0.15:
            best["taken"] = True
            dr["mem"] = best["used"]
            dr["memtot"] = best["tot"]
        del dr["vram_pct"]
        del dr["dev"]
        rows.append(dr)
    return rows or None

# ==================== suzu/fetch/info.py
"""Model/server introspection: config.json, cmdline, arch map."""



def server_info(port):
    """Draft model, spec type, k and uptime from the process holding the port.
    tmux and earlyoom lines are skipped: tmux carries an identical command line
    (it is the wrapper) and earlyoom embeds 'llama-server' in its --prefer regex."""
    info = {"draft": None, "spec": None, "k": None, "up": None, "model": None,
            "engine": None}
    try:
        # -ww IS MANDATORY: without it ps truncates args= to $COLUMNS, and this
        # panel sets COLUMNS itself for layout, so "--port N" fell off the end
        # and the header silently lost the draft model and the uptime.
        out = subprocess.run(["ps", "-ww", "-eo", "pid=,etimes=,args="],
                             capture_output=True, text=True, timeout=4).stdout
    except (OSError, subprocess.SubprocessError) as e:
        if os.environ.get("SUZU_DEBUG"):
            sys.stderr.write("suzu: ps failed: %r\n" % (e,))
        return info
    portpat = re.compile(r"(?:--port[= ]\s*|-p\s+)%d(?!\d)" % port)
    best = fallback = None
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        args = parts[2]
        base = os.path.basename(args.split(None, 1)[0])
        if base in ("tmux", "earlyoom", "sh", "bash", "grep"):
            continue
        if not any(b in base for b in ("llama-server", "ds4-server", "vllm", "python")):
            continue
        # "python" also matches the gateway and this very monitor: neither is an
        # engine, and picking one would report ITS uptime as the model's. Exclude
        # suzu by SCRIPT NAME, not path substring - the demo engine lives under a
        # .../suzu/ directory and must still match - plus the gateway if
        # --lanes-url points at one.
        try:
            script = os.path.basename(args.split(None, 1)[1].split()[0])
        except IndexError:
            script = ""
        if script == "suzu.py" or (config.LANES_URL and _lanes_host() in args):
            continue
        if portpat.search(args):
            best = (parts, args, base)
            break
        # The panel normally polls :30001, which is the GATEWAY - the engine
        # itself listens on :30021 and its cmdline never carries the polled port,
        # so the strict match found nothing and the header lost the draft model
        # AND the uptime ("(no draft model)", "--:--", seen 23-08 with a dspark
        # drafter plainly running). The launcher guarantees one engine at a time, so an
        # engine process with the wrong port is still THE engine.
        if fallback is None and any(
                x in args for x in ("llama-server", "ds4-server", "vllm", "sglang", "tensorfold serve")):
            fallback = (parts, args, base)
        # TabbyAPI presents itself as `venv/bin/python .../tabby/main.py --config X`:
        # none of the markers above. Without this branch the glm53x entry's header
        # stayed without engine name and without uptime.
        if fallback is None and "main.py" in args and "tabby" in args:
            fallback = (parts, args, base)
    if not best:
        best = fallback
    if not best:
        if os.environ.get("SUZU_DEBUG"):
            sys.stderr.write("suzu: no process on port %d among %d ps lines\n"
                             % (port, len(out.splitlines())))
        return info
    parts, args, base = best
    try:
        info["up"] = int(parts[1])
    except ValueError:
        pass
    info["pid"] = parts[0]
    info["engine"] = base
    # `python -m vllm.entrypoints...` is not a useful name in the header.
    for nice in ("llama-server", "ds4-server", "vllm", "sglang"):
        if nice in args:
            info["engine"] = nice
            break
    if "tensorfold serve" in args:
        # rank 0 of the TP=2 engine (glm53tf): inside the container, visible from the host's ps
        info["engine"] = "tensorfold"
        m = re.search(r"--drafter[= ]\s*(\S+)", args)
        m_drafter = bool(m) and m.group(1) != "none"
        if m:
            d = re.search(r"models--([^/]+)--([^/]+)/snapshots/([0-9a-f]{7})", m.group(1))
            info["draft"] = "%s/%s@%s" % d.groups() if d else os.path.basename(m.group(1).rstrip("/"))
            info["spec"] = "DFlash2 + MTP + lookup, depth by cost"
        else:
            info["spec"] = "MTP, depth by cost"
        for flag, key in (("--context", "maxlen"), ("--name", "model"), ("--tp", "tp")):
            m = re.search(re.escape(flag) + r"[= ]\s*(\S+)", args)
            if m:
                info[key] = m.group(1)
        if info.get("tp") == "2":
            info["nnodes"] = "2"          # the two ranks live on this box and the rank-1 box
        env = tf_env()
        info["k"] = env.get("GLM53_TF_AUTO_FDRAFTS") or info["k"]
        info["slots"] = env.get("GLM53_TF_BATCH")
        info["kvdtype"] = env.get("GLM53_TF_KV_DTYPE")
        info["quant"] = "EXL3 4bpw + q4 non-expert" if env.get("GLM53_TF_NONEXPERT") == "q4mse" else None
        if "TF_GLM_DFLASH_POLICY" in env:                 # newer kit: other knobs, other names
            pol = env["TF_GLM_DFLASH_POLICY"]
            m = re.match(r"f\D*(\d+)", pol)
            info["k"] = m.group(1) if m else info["k"]
            copy = env.get("TF_GLM_COPY_DRAFTS") == "1"
            info["spec"] = ("DFlash2 %s + copy drafts (up to %s)" % (pol, env.get("TF_GLM_COPY_MAX", "?")) if copy
                            else "DFlash2 %s" % pol) if m_drafter else "MTP"
            m = re.search(r"--parallel[= ]\s*(\d+)", args)
            info["slots"] = m.group(1) if m else info["slots"]
            info["kvdtype"] = env.get("TF_GLM_KV") or info["kvdtype"]
            dense = env.get("TF_GLM_DENSE")
            info["quant"] = "EXL3 4bpw + %s dense" % dense if dense else info["quant"]
        return info
    if "main.py" in args and "tabby" in args:
        info["engine"] = "tabby/exl3"
        # without the environment the header does not even know whether MTP is on
        env = proc_env(parts[0])
        info["spec"] = env.get("TABBY_DRAFT_MODEL_DRAFT_MODE") or info["spec"]
        if info["spec"] and info["spec"] != "disabled":
            info["draft"] = info["draft"] or "native MTP of the pack"
        cache = env.get("TABBY_MODEL_CACHE_MODE")
        if cache:
            info["engine"] += " . cache " + cache
    for flag, key in (("--spec-draft-model", "draft"), ("--spec-type", "spec"),
                      ("--spec-draft-n-max", "k"), ("--model", "model"),
                      ("--served-model-name", "model"),
                      ("--quantization", "quant"), ("--kv-cache-dtype", "kvdtype"),
                      ("--tensor-parallel-size", "tp"), ("--nnodes", "nnodes"),
                      ("--max-num-seqs", "seqs"), ("--max-model-len", "maxlen")):
        if info.get(key):
            continue
        m = re.search(re.escape(flag) + r"[= ]\s*(\S+)", args)
        if m:
            info[key] = m.group(1)
    # vLLM does not take the llama.cpp flags: the whole speculative stack arrives
    # as one JSON blob, so without this the panel says "no draft model" while
    # dspark is demonstrably drafting (spec_decode_* counters moving).
    m = re.search(r"--speculative[-_]config[= ]\s*(\{.*?\})(?:\s+--|\s*$)", args)
    if m:
        try:
            sc = json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            sc = {}
        if isinstance(sc, dict):
            info["draft"] = info["draft"] or sc.get("model")
            info["spec"] = info["spec"] or sc.get("method")
            k = sc.get("num_speculative_tokens")
            info["k"] = info["k"] or (str(k) if k is not None else None)
    return info



def model_config_path():
    """Host path to the running engine's model config.json.

    The layer grid is built from the REAL model topology (num layers, experts,
    DSpark target layers, KV compression ratios) - not a hard-coded table. The
    engine is a docker container; the config lives in the /models mount. Falls
    back to scanning ~/models for the most recent tp1/config.json.
    """
    try:
        names = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                               capture_output=True, text=True, timeout=8).stdout.split()
        # TensorFold (glm53mia, glm53tf: 05-10): `tensorfold serve <model folder>` inside the container, the folder
        # under the HF cache mount. Without this the expert heat map had no topology on TensorFold and vanished
        # once the old tp1/ folders of the vLLM kit were gone from ~/models.
        for n in names:
            try:
                out = subprocess.run(["docker", "inspect", "-f", "{{json .Config.Cmd}}\t{{json .Mounts}}", n],
                                     capture_output=True, text=True, timeout=8).stdout
                cmd_j, mounts_j = out.rstrip("\n").split("\t", 1)
                cmd, mounts = json.loads(cmd_j) or [], json.loads(mounts_j) or []
            except (ValueError, OSError, subprocess.SubprocessError):
                continue
            if "serve" not in cmd or not any("tensorfold" in str(c) for c in cmd[:2]):
                continue
            model = cmd[cmd.index("serve") + 1] if cmd.index("serve") + 1 < len(cmd) else ""
            for mnt in mounts:
                dst = (mnt.get("Destination") or "").rstrip("/")
                # `tensorfold serve /model` names the mount root itself, not a
                # folder inside it - accept both shapes or the layer grid goes
                # dark on every TensorFold box started that way.
                if dst and (model == dst or model.startswith(dst + "/")):
                    rel = "" if model == dst else model[len(dst) + 1:]
                    p = os.path.join(mnt.get("Source") or "", rel, "config.json")
                    if os.path.exists(p):
                        return p
        for n in names:
            if not any(t in n for t in ("vllm", "spark", "sglang")):
                continue
            mounts = subprocess.run(
                ["docker", "inspect", "-f",
                 "{{range .Mounts}}{{.Source}}|{{.Destination}} {{end}}", n],
                capture_output=True, text=True, timeout=8).stdout
            for part in mounts.split():
                if "|" not in part:
                    continue
                src, dst = part.split("|", 1)
                dst = dst.rstrip("/")
                if dst == "/models":
                    p = os.path.join(src, "tp1", "config.json")
                    if os.path.exists(p):
                        return p
    except (OSError, subprocess.SubprocessError):
        pass
    # fallback: the most recent tp1 config in ~/models
    cands = []
    mdir = os.path.expanduser("~/models")
    try:
        for root, dirs, files in os.walk(mdir):
            if root.endswith(os.sep + "tp1") and "config.json" in files:
                cands.append(os.path.join(root, "config.json"))
    except OSError:
        pass
    return max(cands, key=os.path.getmtime) if cands else None



def load_model_arch(path):
    """Extract the per-layer topology from a model config.json.

    Returns None on any failure; the layer grid degrades to a message instead
    of a crash. Everything here is static model structure - the LIVE work
    colours come from the Sample, never from this table.
    """
    try:
        with open(path) as f:
            c = json.load(f)
    except (OSError, ValueError):
        return None
    if isinstance(c.get("text_config"), dict):      # GLM-5.3 (glm5_next): the text model's fields live here
        top, c = c, dict(c["text_config"])
        if not c.get("vision_n_layers"):
            c["vision_n_layers"] = (top.get("vision_config") or {}).get("depth") or 0
        kinds = c.get("mlp_layer_types") or []
        if kinds and not c.get("hybrid_tr3_tail"):
            c["hybrid_tr3_tail"] = {"moe_layers": [i for i, k in enumerate(kinds) if k == "sparse"]}
    n = int(c.get("num_hidden_layers") or 0)
    if n <= 0:
        return None
    comp = c.get("compress_ratios") or []
    comp = {i: int(v) for i, v in enumerate(comp)}
    return {
        "n": n,
        "experts": int(c.get("n_routed_experts") or 0),
        "topk": int(c.get("num_experts_per_tok") or 0),
        "shared": int(c.get("n_shared_experts") or 0),
        "nextn": int(c.get("num_nextn_predict_layers") or 0),
        "vision": int(c.get("vision_n_layers") or 0),
        "target": [int(x) for x in (c.get("dspark_target_layer_ids") or [])],
        "compress": comp,
        "moe_layers": (c.get("hybrid_tr3_tail") or {}).get("moe_layers"),
        "name": os.path.basename(os.path.dirname(os.path.dirname(path))),
    }

# ==================== suzu/fetch/logtail.py
"""Engine log discovery and tail."""



def active_log(llm):
    try:
        out = subprocess.run(["tmux", "list-sessions", "-F", "#{session_name}"],
                             capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.SubprocessError):
        out = ""
    cands = []
    for s in (x for x in out.split() if x.startswith("yz-")):
        p = os.path.join(llm, "yz-logs", "yz-%s.log" % s[3:])
        if os.path.exists(p):
            cands.append(p)
    if not cands:
        d = os.path.join(llm, "yz-logs")
        if os.path.isdir(d):
            cands = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".log")]
    return max(cands, key=os.path.getmtime) if cands else None



def log_mtime(llm):
    """Wall-clock of the server's last log write. Used only to seed the idle
    clock: a panel started five minutes ago must not claim the engine has been
    idle for five minutes when the log says it has been quiet since 06:57."""
    lp = active_log(llm)
    try:
        return os.path.getmtime(lp) if lp else None
    except OSError:
        return None



def tail(path, n):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size, data = f.tell(), b""
            while size > 0 and data.count(b"\n") <= n + 2:
                step = min(8192, size)
                size -= step
                f.seek(size)
                data = f.read(step) + data
        return [ANSI_RE.sub("", l).rstrip() for l in
                data.decode("utf-8", "replace").splitlines()[-n:]]
    except OSError:
        return []

# ==================== suzu/fetch/vllm.py
"""vLLM-specific synthesis: pool gauges, ctx max, synthetic slots."""



_VLLM_POOL = {}



_VLLM_CTXMAX = {}



def vllm_pool(port):
    """KV pool size in tokens, from the boot log ("GPU KV cache size: N tokens").

    Not derivable from /metrics: `vllm:cache_config_info` reports num_gpu_blocks
    12355 x block_size 4 = 49,420 on a real pool of 2,192,815 (measured 21-08),
    because the hybrid cache (128-token sliding-window layers beside the global
    ones) is not that product. `docker logs` costs ~10 s: memoized on container
    id + StartedAt, so a restart invalidates the cache by itself.
    """
    if port in _VLLM_POOL:
        return _VLLM_POOL[port]
    _VLLM_POOL[port] = None
    try:
        names = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                               capture_output=True, text=True, timeout=8).stdout.split()
        name = next((x for x in names
                     if any(t in x for t in ("vllm", "spark", "sglang"))), None)
        if not name:
            return None
        key = subprocess.run(["docker", "inspect", "-f", "{{.Id}}-{{.State.StartedAt}}", name],
                             capture_output=True, text=True, timeout=8).stdout.strip()
        cache = "/tmp/yz-kvpool-" + re.sub(r"[^A-Za-z0-9]", "", key)[:40] if key else None
        if cache and os.path.exists(cache):
            _VLLM_POOL[port] = int(open(cache).read().strip())
            return _VLLM_POOL[port]
        log = subprocess.run(["docker", "logs", "--tail", "4000", name],
                             capture_output=True, text=True, timeout=40)
        hits = re.findall(r"GPU KV cache size: ([\d,]+) tokens", log.stdout + log.stderr)
        if hits:
            _VLLM_POOL[port] = int(hits[-1].replace(",", ""))
            if cache:
                open(cache, "w").write(str(_VLLM_POOL[port]))
    except Exception:
        pass
    return _VLLM_POOL[port]



def vllm_ctxmax(port):
    """max_model_len from /v1/models: on vLLM there is no llamacpp:n_tokens_max."""
    if port in _VLLM_CTXMAX:
        return _VLLM_CTXMAX[port]
    _VLLM_CTXMAX[port] = 0
    txt = http_get("http://127.0.0.1:%d/v1/models" % port, timeout=4, tries=1)
    if txt:
        try:
            _VLLM_CTXMAX[port] = int(json.loads(txt)["data"][0].get("max_model_len") or 0)
        except (ValueError, KeyError, IndexError, TypeError):
            pass
    return _VLLM_CTXMAX[port]



def synth_slots_vllm(mx, port):
    """Synthetic slot for vLLM, which does NOT expose /slots.

    On llama.cpp the /metrics counters are flushed at request end, which is why
    this monitor reads /slots. On vLLM it is the OPPOSITE, measured by sampling
    every 2 s inside a single request (21-08 on dsflash):

        t1 draft=530  acc=298 gen=400      <- not started yet
        t3 draft=725  acc=395 gen=537
        t6 draft=1035 acc=502 gen=706      <- still in flight

    So it is enough to remap /metrics into the shape of a llama.cpp slot and the
    whole TUI works without touching anything else. Two differences to keep in
    mind:
      - `n_decoded` here is a LIFETIME counter, not reset per request: deltas
        stay >= 0, so the "negative delta = new task" branch never fires
        (harmless).
      - prefill is not visible on `vllm:prompt_tokens_total`, credited only at
        request CLOSE (measured: 48 after killing a 900k prefill halfway). The
        only reading that advances during a long prefill is
        `kv_cache_usage_perc` -> progress is that fraction of the pool.
    Without the pool (unreadable log) it returns 0 on ptok/n_ctx: the slot panel
    switches itself off and the rest of the TUI stays valid.
    """
    if not any(k.startswith("vllm:") for k in mx):
        return None
    pool = vllm_pool(port) or 0
    ptok = int(mx.get("vllm:kv_cache_usage_perc", 0.0) * pool)
    q = mx.get("vllm:prefix_cache_queries_total", 0.0)
    h = mx.get("vllm:prefix_cache_hits_total", 0.0)
    return [{
        "next_token": {"n_decoded": int(mx.get("vllm:generation_tokens_total", 0.0))},
        "n_prompt_tokens_processed": ptok,
        "n_prompt_tokens": ptok,
        "n_prompt_tokens_cache": int(ptok * h / q) if q > 0 else 0,
        "n_ctx": int(pool),
        "is_processing": mx.get("vllm:num_requests_running", 0.0) > 0,
    }]

# ==================== suzu/fetch/engine.py
"""Engine sample model and the llama.cpp/vLLM live probes."""



class Sample:
    """One sample: cumulative counters from /metrics + live state from /slots."""

    def __init__(self, mx, slots, t):
        self.t = t
        # tabby BEFORE llama.cpp: synth_tabby deliberately writes `llamacpp:` keys
        # (so the rest of the file does not know it exists), and without this case
        # the panel would present itself as llama.cpp.
        self.fam = ("tabby" if "tabby:records" in mx
                    else "tensorfold" if "tensorfold:records" in mx
                    else "llama.cpp" if any(k.startswith("llamacpp:") for k in mx)
                    else "vllm" if any(k.startswith("vllm:") for k in mx)
                    else "ds4" if any(k.startswith("ds4_") for k in mx) else "?")
        self.gen = g(mx, "llamacpp:tokens_predicted_total",
                     "vllm:generation_tokens_total", "ds4_tokens_decoded_total")
        self.pp = g(mx, "llamacpp:prompt_tokens_total", "vllm:prompt_tokens_total",
                    'ds4_tokens_prefilled_total{kind="computed"}')
        self.ps = g(mx, "llamacpp:prompt_seconds_total",
                    "vllm:request_prefill_time_seconds_sum")
        # vllm:prompt_tokens_total counts the PREFIX-CACHE HITS too - here 58.1 M
        # of 60.5 M lifetime tokens were served from cache and cost no prefill at
        # all. Dividing that by prefill seconds measures the cache, not the
        # engine. local_compute is the subset actually computed, which is what
        # llama.cpp's n_prompt_tokens_processed means, so the two engines' rows
        # finally say the same thing.
        # TensorFold without a drafted-token counter (jayleaton): no acceptance %, only tokens/round
        self.tf_nodraft = self.fam == "tensorfold" and "tensorfold:has_drafted" not in mx
        self.tf_fill = (mx.get("tensorfold:fill_streams"), mx.get("tensorfold:fill_pos"), mx.get("tensorfold:fill_len"))
        _lc = gl(mx, "vllm:prompt_tokens_by_source_total", 'source="local_compute"')
        if _lc is not None:
            self.pp = _lc
        self.draft = g(mx, "llamacpp:spec_decode_num_draft_tokens_total",
                       "vllm:spec_decode_num_draft_tokens_total")
        self.acc = g(mx, "llamacpp:spec_decode_num_accepted_tokens_total",
                     "vllm:spec_decode_num_accepted_tokens_total")
        self.drafts = g(mx, "llamacpp:spec_decode_num_drafts_total",
                        "vllm:spec_decode_num_drafts_total")
        self.pos = []
        for i in range(32):
            k = ('llamacpp:spec_decode_num_accepted_tokens_per_pos_total'
                 '{position="%d"}' % i)
            k2 = ('vllm:spec_decode_num_accepted_tokens_per_pos_total'
                  '{position="%d"}' % i)
            if k in mx:
                self.pos.append(mx[k])
            elif k2 in mx:
                self.pos.append(mx[k2])
            else:
                break
        self.run = g(mx, "llamacpp:requests_processing",
                     "vllm:num_requests_running", "ds4_requests_inflight")
        self.wait = g(mx, "llamacpp:requests_deferred", "vllm:num_requests_waiting")
        # 19-09: with a gateway in front, the engine may HOLD requests the gateway
        # has ALREADY admitted (max_num_seqs, full KV pool, the kit's mixed-prefill
        # `skip`): vLLM counts them for a reason. Without these two the panel said
        # just "2/2 slots" while the human waited 335 s for the first token.
        self.wait_cap = gl(mx, "vllm:num_requests_waiting_by_reason", 'reason="capacity"') or 0.0
        self.wait_def = gl(mx, "vllm:num_requests_waiting_by_reason", 'reason="deferred"') or 0.0
        self.ctxmax = g(mx, "llamacpp:n_tokens_max")
        # totals since the engine started, input included (30-09: the panel showed only the generated ones). On vLLM
        # prompt_tokens_total counts the cached too, like TensorFold's prompt. llama.cpp exposes the same total as
        # tokens_prompt_total - without this fallback the totals box (and the harness rows under it) stayed hidden
        # on every llama.cpp-style engine.
        self.tot_in = mx.get("tensorfold:prompt_total",
                             mx.get("vllm:prompt_tokens_total",
                                    mx.get("llamacpp:tokens_prompt_total")))
        self.tot_cached = mx.get("tensorfold:cached_total",
                                 gl(mx, "vllm:prompt_tokens_by_source_total", 'source="local_cache_hit"'))
        self.nreq = mx.get("tensorfold:records", mx.get("vllm:request_success_total"))
        # TensorFold: the usual all-time totals too, from the request log (they survive restarts)
        self.all_in = mx.get("tensorfold:all_in")
        self.all_out = mx.get("tensorfold:all_out")
        self.all_n = mx.get("tensorfold:all_n")
        self.all_since = mx.get("tensorfold:all_since")
        # live, per-token
        self.dec = self.ppp = self.ptok = self.pcache = self.nctx = 0
        self.busy = False
        self.nslots = 0
        if isinstance(slots, list):
            self.nslots = len(slots)
            for s in slots:
                nt = s.get("next_token") or []
                if isinstance(nt, list) and nt:
                    self.dec += int(nt[0].get("n_decoded", 0) or 0)
                elif isinstance(nt, dict):
                    self.dec += int(nt.get("n_decoded", 0) or 0)
                self.ppp += int(s.get("n_prompt_tokens_processed", 0) or 0)
                self.ptok += int(s.get("n_prompt_tokens", 0) or 0)
                self.pcache += int(s.get("n_prompt_tokens_cache", 0) or 0)
                self.nctx += int(s.get("n_ctx", 0) or 0)
                self.busy = self.busy or bool(s.get("is_processing"))

    def restarted_from(self, prev):
        return (self.pp < prev.pp - 1 or self.gen < prev.gen - 1
                or self.acc < prev.acc - 1)



def fetch_yzq(port):
    """Queue depth from the optional gateway in front of the engine, if the
    polled port IS the gateway.

    Why this exists: the engine runs --max-num-seqs 1 and the gateway forwards
    ONE request at a time, so vllm:num_requests_waiting is structurally 0 no
    matter how many calls are outstanding — everyone else is parked in the
    gateway's lanes and has never been offered to the engine. Reading only
    /metrics, the panel said "running 1 / waiting 0" while four conversations
    were in flight (measured 23-08: gateway stats high=3 + inflight, /metrics
    waiting=0). The stats endpoint is answered by the gateway BEFORE it touches
    the scheduler (handle(), first branch), so polling it costs nothing and can
    never queue behind a generation."""
    txt = http_get("http://127.0.0.1:%d/yzq/stats" % port, timeout=2, tries=1)
    if not txt:
        return None
    try:
        j = json.loads(txt)
    except (json.JSONDecodeError, ValueError):
        return None
    return j if isinstance(j, dict) and "high" in j else None



def yzq_live_sse(yzq):
    """SSE events relayed so far for EVERY request still open at the gateway. Its `inflight` is only the first job: with
    two lanes the other one's tokens were missing from the live estimate and landed all at once when it closed (30-09,
    TensorFold: gen t/s 2145, peak 1079, a rolling average near twice the real one)."""
    q = yzq or {}
    jobs = q.get("running")
    if jobs:
        return sum(int(j.get("sse") or 0) for j in jobs)
    return int((q.get("inflight") or {}).get("sse") or 0)



def fetch_yzq_lanes(port, url=None):
    """The optional gateway's admission state in detail: per-lane queues,
    running slots and the free/parallel breakdown - the lanes endpoint.
    What /metrics cannot say about WHY a request is waiting, this can.
    url=None keeps the legacy same-port probe; an explicit --lanes-url wins."""
    txt = http_get(url if url else "http://127.0.0.1:%d/yzq/lanes" % port,
                   timeout=2, tries=1)
    if not txt:
        return None
    try:
        j = json.loads(txt)
    except (json.JSONDecodeError, ValueError):
        return None
    return j if isinstance(j, dict) else None

# ==================== suzu/fetch/tabby.py
"""TabbyAPI / exllamav3 adapter: /props + gateway + engine log."""



_TABBY_PROPS = {}



_TAB_ZERO = {"gen": 0, "new": 0, "ps": 0.0, "acc": 0, "draft": 0, "n": 0}



_TABBY_LOG = {"path": None, "off": 0, "carry": None, "tot": dict(_TAB_ZERO),
              "last": None}



_TABBY_MONO = {"gen": 0.0}



_TABBY_KFILE = "/tmp/yz-tabby-k"



def _tabby_k0():
    """Last calibrated factor, saved by a previous panel run. Without it the
    FIRST request after each engine start would read at a third of truth."""
    try:
        k = float(open(_TABBY_KFILE).read().strip())
        return k if 0.5 <= k <= 12.0 else 0.0
    except (OSError, ValueError):
        return 0.0



_TABBY_RATIO = {"gen": 0.0, "ev": 0.0, "p_gen": None, "p_ev": None, "p_n": None,
                "saved": _tabby_k0()}



_TS_RE = re.compile(r"^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d")



_TAB_GEN = re.compile(r"(\d+) tokens generated in\s+([\d.]+) seconds")



_TAB_Q = re.compile(r"Queue:\s+([\d.]+) s")



_TAB_P = re.compile(r"Process:\s+(\d+) cached tokens and\s+(\d+) new tokens at\s+([\d.]+) T/s")



_TAB_G = re.compile(r"Generate:\s+([\d.]+) T/s")



_TAB_C = re.compile(r"Context:\s+(\d+) tokens")



_TAB_D = re.compile(r"Draft:\s+(\d+) / (\d+) tokens accepted")



def proc_env(pid):
    """A process environment. TabbyAPI's knobs live THERE, not on the command
    line: the config file deliberately stays silent on the dynamic keys and
    the launcher passes them as TABBY_* (with --config X the other flags are
    ignored, tabby_config.py:141)."""
    env = {}
    try:
        with open("/proc/%s/environ" % pid, "rb") as f:
            for kv in f.read().split(b"\0"):
                if b"=" in kv:
                    a, _, b = kv.partition(b"=")
                    env[a.decode("utf-8", "replace")] = b.decode("utf-8", "replace")
    except OSError:
        pass
    return env



def tabby_env():
    """TabbyAPI's process environment, or {}. The tmux line is skipped: it is
    the wrapper and carries the same command line."""
    try:
        out = subprocess.run(["ps", "-ww", "-eo", "pid=,args="],
                             capture_output=True, text=True, timeout=4).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) < 2:
            continue
        args = parts[1]
        if "main.py" not in args or "tabby" not in args:
            continue
        if os.path.basename(args.split(None, 1)[0]) in ("tmux", "sh", "bash"):
            continue
        env = proc_env(parts[0])
        if env:
            return env
    return {}



def tabby_props(port):
    """n_ctx and slots. Memoized ONLY on success: during the load (30-40 s at
    512k) /props does not answer, and memoizing the hole would have switched the
    tabby branch off for the whole life of the panel.

    ⚠ /props is not reliable as a PRESENCE probe. Measured 29-08: under a long
    prefill TabbyAPI stays mute for up to 20 s, and a panel that concludes "no
    engine" at that moment switches off EXACTLY when the engine works — which is
    the defect this file exists for. So if /props stays silent it falls back to
    the process environment, where the launcher already wrote
    TABBY_MODEL_MAX_SEQ_LEN: not an estimate, it is the same number the server
    received at startup.
    """
    if _TABBY_PROPS.get(port):
        return _TABBY_PROPS[port]
    n, slots = 0, 0
    txt = http_get("http://127.0.0.1:%d/props" % port, timeout=6, tries=2)
    if txt:
        try:
            j = json.loads(txt)
            n = int((j.get("default_generation_settings") or {}).get("n_ctx") or 0)
            slots = int(j.get("total_slots") or 0)
        except (ValueError, TypeError, AttributeError):
            n = 0
    if n <= 0:
        try:
            n = int(tabby_env().get("TABBY_MODEL_MAX_SEQ_LEN") or 0)
        except (ValueError, TypeError):
            n = 0
    if n <= 0:
        return {}
    _TABBY_PROPS[port] = {"n_ctx": n, "slots": slots or 1}
    return _TABBY_PROPS[port]



def tabby_parse(rec):
    """A gathered `Metrics (ID: ...)` record -> dict, or None if it is not one."""
    rec = re.sub(r"\s+", " ", rec)
    mg, mp = _TAB_GEN.search(rec), _TAB_P.search(rec)
    if not mg or not mp:
        return None
    mq, mgg, mc, md = (_TAB_Q.search(rec), _TAB_G.search(rec),
                       _TAB_C.search(rec), _TAB_D.search(rec))
    return {
        "gen": int(mg.group(1)), "sec": float(mg.group(2)),
        "queue": float(mq.group(1)) if mq else 0.0,
        "cached": int(mp.group(1)), "new": int(mp.group(2)),
        "ptps": float(mp.group(3)),
        "gtps": float(mgg.group(1)) if mgg else 0.0,
        "ctx": int(mc.group(1)) if mc else 0,
        "acc": int(md.group(1)) if md else 0,
        "draft": int(md.group(2)) if md else 0,
    }



def tabby_scan(path, maxbytes=262144):
    """(cumulative totals, last record) from the `Metrics (ID: ...)` lines.

    ⚠ The log is written by rich WITHOUT a tty: the console falls back to 80
    columns and EVERY record comes out split over 3-4 lines, sometimes in the
    middle of a field ("Queue:" / "0.01 s"). Continuation lines carry no
    timestamp, and that is how they are recognised; they are rejoined and the
    spaces normalized before reading the numbers. (From today the glm53x entry
    launches with TABBY_LOG_CONSOLE_WIDTH: NEW logs sit on a single line, the
    unwrap is for the ones already written and for a hand-started engine.)

    ⚠ The totals are ACCUMULATED, not recomputed over the file tail. It seemed
    simpler to re-sum the last 256 KB at every tick, but that window SLIDES: as
    soon as the log outgrows it the cumulatives DROP, and a drop here is
    exactly what Sample.restarted_from reads as "engine restarted" -> panel
    history thrown away every few minutes. So only the NEW tail is read
    (offset), and the window only decides where to start the first time. A real
    restart (different file, or shorter than before) deliberately zeroes
    everything.
    """
    st = _TABBY_LOG
    if not path:
        return dict(st["tot"]), st["last"]
    try:
        size = os.path.getsize(path)
    except OSError:
        return dict(st["tot"]), st["last"]
    if st["path"] != path or size < st["off"]:
        # First read or rotated log. Start from the tail: the rest of the file
        # is another engine session's history. The cut falls in the middle of
        # a line, which lacking the timestamp discards itself.
        st.update(path=path, off=max(0, size - maxbytes), carry=None,
                  tot=dict(_TAB_ZERO), last=None)
        _TABBY_MONO["gen"] = 0.0
        _TABBY_PROPS.clear()             # new engine = re-read n_ctx
    if size <= st["off"]:
        return dict(st["tot"]), st["last"]
    try:
        with open(path, "rb") as f:
            f.seek(st["off"])
            data = f.read(size - st["off"])
    except OSError:
        return dict(st["tot"]), st["last"]
    cut = data.rfind(b"\n")              # the half line is read on the next round
    if cut < 0:
        return dict(st["tot"]), st["last"]
    st["off"] += cut + 1
    buf, tot = st["carry"], st["tot"]
    for ln in ANSI_RE.sub("", data[:cut].decode("utf-8", "replace")).splitlines():
        if _TS_RE.match(ln):
            if buf is not None:
                rec = tabby_parse(buf)
                if rec:
                    tot["gen"] += rec["gen"]
                    tot["new"] += rec["new"]
                    tot["acc"] += rec["acc"]
                    tot["draft"] += rec["draft"]
                    tot["n"] += 1
                    if rec["ptps"] > 0:
                        tot["ps"] += rec["new"] / rec["ptps"]
                    st["last"] = rec
                buf = None
            if "Metrics (ID:" in ln:
                buf = ln
        elif buf is not None:
            buf += " " + ln.strip()
    st["carry"] = buf                    # record still open: it closes next round
    return dict(tot), st["last"]



def tabby_scale(tot, ev_total, ev_live):
    """In-flight SSE events -> tokens, using the factor calibrated on closed
    records.

    The baseline (p_gen, p_ev) moves ONLY when a record closes (tot["n"]
    grows): that way the two deltas cover the same set of requests. If it
    were updated every tick, the piece still in flight — events counted,
    tokens not yet written to the log — would drag the ratio down
    continuously.
    """
    r = _TABBY_RATIO
    if r["p_ev"] is None or ev_total < r["p_ev"] or tot["gen"] < (r["p_gen"] or 0):
        # first round, or gateway/engine restarted: restart from the current base
        r.update(gen=0.0, ev=0.0, p_gen=float(tot["gen"]), p_ev=ev_total,
                 p_n=tot["n"])
    elif r["p_n"] is not None and tot["n"] > r["p_n"]:
        r["gen"] += float(tot["gen"]) - r["p_gen"]
        r["ev"] += ev_total - r["p_ev"]
        r["p_gen"], r["p_ev"], r["p_n"] = float(tot["gen"]), ev_total, tot["n"]
    if r["ev"] >= 20 and r["gen"] > 0:
        k = r["gen"] / r["ev"]
        if abs(k - r["saved"]) > 0.05:
            r["saved"] = k
            try:                          # so the next panel starts calibrated
                with open(_TABBY_KFILE, "w") as f:
                    f.write("%.4f" % k)
            except OSError:
                pass
    else:
        k = r["saved"] or 1.0
    return ev_live * k



def synth_tabby(port, llm, yzq):
    """(mx, slots) in llama.cpp shape for TabbyAPI. (None, None) if absent.

    The only counter that advances WITHIN the request is the gateway's, and for
    that reason it plays `tokens_predicted_total`: the log records arrive after
    the fact. The two are summed (closed + in flight) and the sum is made
    monotone by hand, because in the instant between "the client hangs up" and
    "TabbyAPI writes the record" the in-flight piece disappears and a drop
    would be read as an engine restart, wiping the whole panel history.
    """
    props = tabby_props(port)
    if not props:
        return None, None
    tot, last = tabby_scan(active_log(llm))
    inf = (yzq or {}).get("inflight") or {}
    live_tok = tabby_scale(tot, float((yzq or {}).get("sse_total") or 0.0), yzq_live_sse(yzq))
    gen = float(tot["gen"] + live_tok)
    if gen < _TABBY_MONO["gen"]:
        gen = _TABBY_MONO["gen"]
    else:
        _TABBY_MONO["gen"] = gen
    pp, ps, acc, dra = float(tot["new"]), tot["ps"], tot["acc"], tot["draft"]
    mx = {
        "tabby:records": float(tot["n"]),
        "llamacpp:tokens_predicted_total": gen,
        "llamacpp:prompt_tokens_total": pp,
        "llamacpp:prompt_seconds_total": ps,
        "llamacpp:requests_processing": 1.0 if inf else 0.0,
        "llamacpp:requests_deferred": float((yzq or {}).get("high") or 0)
                                      + float((yzq or {}).get("low") or 0),
        "llamacpp:n_tokens_max": float(props["n_ctx"]),
    }
    # The two speculative keys exist ONLY with MTP on: with draft_mode=disabled
    # TabbyAPI writes no `Draft:` and the panel must say "No speculation", not
    # 0% (which would mean "speculates and always gets it wrong").
    if dra > 0:
        mx["llamacpp:spec_decode_num_draft_tokens_total"] = float(dra)
        mx["llamacpp:spec_decode_num_accepted_tokens_total"] = float(acc)
    slots = [{
        "next_token": {"n_decoded": live_tok},
        # 0 ON PURPOSE: there is no prefill progress to sample here, and putting
        # the last closed record's tokens there would read as "prefill done in
        # one tick" -> spikes of thousands of t/s. With 0 the step() branch
        # falls back to prompt_tokens/prompt_seconds, the true ratio because
        # both come from the same records.
        "n_prompt_tokens_processed": 0,
        # ⚠ the denominator is NOT `Context:` alone. On an interrupted request
        # (the gateway preempts the low lane) TabbyAPI logs a context smaller
        # than the tokens it already had in cache, and the prefix bar reached
        # 131% — measured 29-08. cached+new is what the engine really had in
        # hand, so the ratio goes back under 1 without lying.
        "n_prompt_tokens": max((last or {}).get("ctx") or 0,
                               ((last or {}).get("cached") or 0)
                               + ((last or {}).get("new") or 0)),
        "n_prompt_tokens_cache": (last or {}).get("cached") or 0,
        "n_ctx": props["n_ctx"],
        "is_processing": bool(inf),
    }]
    return mx, slots

# ==================== suzu/fetch/tensorfold.py
"""TensorFold adapter: request log, gateway SSE, hermes usage db."""



_TF_MONO = {"gen": 0.0, "comp": 0.0}



_TF_LAST = {"key": None, "rec": None}



def tf_last_record(path=config.TF_REQLOG):
    """The last closed request of the request log, re-read only when the file changes."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (st.st_mtime, st.st_size)
    if _TF_LAST["key"] == key:
        return _TF_LAST["rec"]
    rec = None
    try:
        with open(path, "rb") as f:
            f.seek(max(0, st.st_size - 65536))
            for line in reversed(f.read().splitlines()):
                try:
                    rec = json.loads(line)
                    break
                except (ValueError, UnicodeDecodeError):
                    continue
    except OSError:
        pass
    _TF_LAST["key"], _TF_LAST["rec"] = key, rec
    return rec



_TF_ALL = {"ino": None, "off": 0, "carry": b"", "in": 0, "cached": 0, "out": 0, "n": 0, "since": None}



def tf_lifetime(path=config.TF_REQLOG):
    """Totals over the whole request log, across engine restarts (the /metrics counters restart at every boot).
    Incremental: only the bytes appended since the last call are parsed, so the file can grow for months."""
    a = _TF_ALL
    try:
        st = os.stat(path)
    except OSError:
        return None
    if a["ino"] != st.st_ino or st.st_size < a["off"]:        # new file, or truncated: count again from zero
        a.update(ino=st.st_ino, off=0, carry=b"", **{"in": 0}, cached=0, out=0, n=0, since=None)
    if st.st_size > a["off"]:
        try:
            with open(path, "rb") as f:
                f.seek(a["off"])
                buf = a["carry"] + f.read()
                a["off"] = f.tell()
        except OSError:
            return a
        lines = buf.split(b"\n")
        a["carry"] = lines.pop()                               # a half-written last line waits for the next call
        for line in lines:
            try:
                r = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            a["in"] += int(r.get("prompt") or 0)
            a["cached"] += int(r.get("cached") or 0)
            a["out"] += int(r.get("decode_tokens") or 0)
            a["n"] += 1
            if a["since"] is None:
                a["since"] = r.get("start") or r.get("ts")
    return a



_TF_DELTA = {"reqs": None, "prompt": 0.0, "cached": 0.0, "last": {}}



def _lanes_host():
    """Host[:port] of --lanes-url, so the ps filter can skip a gateway process
    (it is not the engine and its uptime would be reported as the model's)."""
    try:
        from urllib.parse import urlsplit
        netloc = urlsplit(config.LANES_URL or "").netloc
        return netloc.split("@")[-1] if netloc else ""
    except ValueError:
        return ""



_HERMES_USE = {"at": -1e9, "val": None}



_HERMES_DAYS = {"at": -1e9, "val": None}



def hermes_days(path=None, every=600.0):
    """GENERATED tokens per day (local "YYYY-MM-DD" -> output_tokens) across all models, from the assistant's
    session traffic. Usage is not recorded per call but per session x model, with a first and last instant: each row
    is attributed to the days on which that session wrote messages inside its window, in proportion to how many;
    with no messages in the window, to the day of the last one. 90% of rows live less than 2.2 hours (measured
    01-10): almost always a single day. Messages are counted on the idx_messages_session_active index (0.2 s on
    207k rows), every 10 min."""
    now = time.time()
    if now - _HERMES_DAYS["at"] < every:
        return _HERMES_DAYS["val"]
    _HERMES_DAYS["at"] = now
    path = path or config.HERMES_DB
    day = lambda ts: time.strftime("%Y-%m-%d", time.localtime(ts))
    val = None
    try:
        import sqlite3
        con = sqlite3.connect("file:%s?mode=ro" % path, uri=True, timeout=1.0)
        try:
            have = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
            act = {}
            if "messages" in have:
                for sid, d, n in con.execute("select session_id, date(timestamp, 'unixepoch', 'localtime'), count(*) "
                                             "from messages where timestamp is not null group by session_id, 2"):
                    act.setdefault(sid, {})[d] = n
            cols = "session_id, output_tokens, first_seen, last_seen"
            parts = ["select %s from session_model_usage" % cols] if "session_model_usage" in have else []
            if "session_model_usage_v21" in have:
                parts.append("select %s from session_model_usage_v21%s" % (cols, (
                    " where session_id not in (select session_id from session_model_usage)" if parts else "")))
            out = {}
            for sid, o, f, l in (con.execute(" union all ".join(parts)) if parts else []):
                tok = float(o or 0)
                if tok <= 0 or not (f or l):
                    continue
                lo, hi = day(f or l), day(l or f)
                w = {d: n for d, n in act.get(sid, {}).items() if lo <= d <= hi}
                tot = sum(w.values())
                if not tot:
                    w, tot = {hi: 1}, 1
                for d, n in w.items():
                    out[d] = out.get(d, 0.0) + tok * n / tot
            val = out or None
        finally:
            con.close()
    except Exception:
        val = _HERMES_DAYS["val"]
    _HERMES_DAYS["val"] = val
    return val



def hermes_usage(path=None, every=60.0):
    """Every token the assistant harness has recorded, per model, on all the engines it used: the
    session_model_usage table of the state DB (since late July) plus session_model_usage_v21 (the earlier schema,
    20-06 -> 27-07; the two do not overlap). input_tokens is the COMPUTED prompt, cache_read_tokens the one taken
    from cache: the whole prompt is the sum. Only the harness's own traffic (not vault cards, benchmarks, direct
    calls). Read read-only at most once a minute: the file weighs 1.3 GB but the tables ~2,400 rows."""
    now = time.time()
    if now - _HERMES_USE["at"] < every:
        return _HERMES_USE["val"]
    _HERMES_USE["at"] = now
    path = path or config.HERMES_DB
    val = None
    try:
        import sqlite3
        con = sqlite3.connect("file:%s?mode=ro" % path, uri=True, timeout=1.0)
        try:
            cols = "model, api_call_count, input_tokens, cache_read_tokens, output_tokens, reasoning_tokens, first_seen"
            have = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
            parts = ["select %s from session_model_usage" % cols] if "session_model_usage" in have else []
            if "session_model_usage_v21" in have:
                parts.append("select %s from session_model_usage_v21%s" % (cols, (
                    " where session_id not in (select session_id from session_model_usage)" if parts else "")))
            if parts:
                q = ("select model, sum(api_call_count), sum(input_tokens), sum(cache_read_tokens), sum(output_tokens), "
                     "sum(reasoning_tokens), min(first_seen) from (%s) group by model" % " union all ".join(parts))
                models = [(m or "?", int(c or 0), int(i or 0), int(cr or 0), int(o or 0), int(r or 0), f)
                          for m, c, i, cr, o, r, f in con.execute(q)]
                models.sort(key=lambda x: -(x[2] + x[3] + x[4]))
                if models:
                    val = {"models": models, "calls": sum(x[1] for x in models), "in": sum(x[2] for x in models),
                           "cached": sum(x[3] for x in models), "out": sum(x[4] for x in models),
                           "reasoning": sum(x[5] for x in models),
                           "since": min((x[6] for x in models if x[6]), default=None)}
        finally:
            con.close()
    except Exception:
        val = _HERMES_USE["val"]            # a locked or missing db: keep the last good reading
    _HERMES_USE["val"] = val
    return val



_TF_FILL = {"pos": {}, "done": 0.0}



_TF_HEALTH = {"at": None}   # rank 0's /health, read by tf_fill: ok, busy, iteration_s (the TP panel)



def tf_fill(port):
    """Kit image y3: the /health "fill" field says, for every stream in prefill, how many prompt lines are already
    in and how many there are in total. Returns (streams in prefill, lines in, prompt lines) of the longest prompt,
    and sums the lines COMPUTED since the last read into a monotone counter: that is the live prefill, because the
    /metrics prompt counters close at request end. A stream's first read is the base: if it restarts from a held
    state, the cached part is not prefill. Without "fill" (image y2) only the stream count."""
    txt = http_get("http://127.0.0.1:%d/health" % port, timeout=2, tries=1)
    try:
        h = json.loads(txt) if txt else {}
    except ValueError:
        h = {}
    if not isinstance(h, dict):
        h = {}
    n = int((h.get("streams") or {}).get("filling") or 0)
    if h:
        _TF_HEALTH.update(at=time.monotonic(), ok=h.get("ok"), busy=h.get("busy"), fatal=h.get("fatal"),
                          iteration_s=h.get("iteration_s"), fill=isinstance(h.get("fill"), list))
    fill = h.get("fill")
    if not isinstance(fill, list):
        return n, None, None
    seen = {}
    for f in fill:
        key, pos = (f.get("sid"), f.get("prompt")), int(f.get("pos") or 0)
        seen[key] = pos
        if key in _TF_FILL["pos"] and pos > _TF_FILL["pos"][key]:
            _TF_FILL["done"] += pos - _TF_FILL["pos"][key]
    _TF_FILL["pos"] = seen
    if not fill:
        return n, 0, 0
    big = max(fill, key=lambda f: int(f.get("prompt") or 0))
    return n, int(big.get("pos") or 0), int(big.get("prompt") or 0)



def tf_last_from_counters(reqs, prompt, cached):
    """Kit image: no requests.jsonl. Prompt and cached of the last closed request = by how much the counters rose
    when requests_total rose (if two close between two reads, it is their sum)."""
    d = _TF_DELTA
    if d["reqs"] is not None and reqs < d["reqs"]:
        d["last"] = {}                                     # restart: the counters start from zero again
    elif d["reqs"] is not None and reqs > d["reqs"]:
        d["last"] = {"prompt": int(prompt - d["prompt"]), "cached": int(cached - d["cached"])}
    d["reqs"], d["prompt"], d["cached"] = reqs, prompt, cached
    return d["last"]



def synth_tensorfold(mx, port, yzq):
    """(mx, slots) in llama.cpp shape for TensorFold, from its /metrics. (None, None) if it is not TensorFold.
    Two dialects: tensorfold_* (jayleaton, glm53tf) and tensorfold:* + tensorfold_health:* (the newer kit on
    TensorFold v0.5.0, since 01-10), which additionally counts drafted tokens (mtp_drafted/accepted: DFlash2 + copy)."""
    new = "tensorfold_health:requests_total" in mx
    if "tensorfold_requests_total" not in mx and not new:
        return None, None
    if new:
        comp = mx.get("tensorfold_health:completion_tokens_total", mx.get("tensorfold:generation_tokens_total", 0.0))
        reqs = mx.get("tensorfold_health:requests_total", 0.0)
        rounds = mx.get("tensorfold_health:rounds_total", 0.0)
        prompt = mx.get("tensorfold:prompt_tokens_total", 0.0)
        cached = mx.get("tensorfold_health:cached_tokens_total", 0.0)
        inflight = mx.get("tensorfold:requests_running", 0.0)
    else:
        comp = mx.get("tensorfold_completion_tokens_total", 0.0)
        reqs = mx.get("tensorfold_requests_total", 0.0)
        rounds = mx.get("tensorfold_decode_rounds_total", 0.0)
        prompt = mx.get("tensorfold_prompt_tokens_total", 0.0)
        cached = mx.get("tensorfold_cached_tokens_total", 0.0)
        inflight = mx.get("tensorfold_requests_inflight", 0.0)
    tpr = (comp - reqs) / rounds if rounds > 0 else 1.0
    # newer kit: completion_tokens_total already counts the tokens of in-flight replies (verified 01-10: +31 in 10 s
    # with requests and rounds still), so the estimate from the gateway's SSE events would count them twice
    live_tok = 0.0 if new else yzq_live_sse(yzq) * max(1.0, tpr)
    # made monotone by hand as for tabby: when a request closes the in-flight estimate disappears and the true
    # number arrives, and a drop would be read as a restart. A REAL restart lowers the closed counters: there it restarts.
    if comp < _TF_MONO["comp"]:
        _TF_MONO["gen"] = 0.0
    _TF_MONO["comp"] = comp
    gen = comp + live_tok
    if gen < _TF_MONO["gen"]:
        gen = _TF_MONO["gen"]
    else:
        _TF_MONO["gen"] = gen
    ctxmax = (mx.get("tensorfold_health:context_length") if new else None) or vllm_ctxmax(port)
    out = {
        "tensorfold:records": reqs,
        "tensorfold:prompt_total": prompt,
        "tensorfold:cached_total": cached,
    }
    life = None if new else tf_lifetime()       # requests.jsonl is the jayleaton engine's: the newer kit does not write it
    if life and life["n"]:
        out.update({"tensorfold:all_in": float(life["in"]), "tensorfold:all_cached": float(life["cached"]),
                    "tensorfold:all_out": float(life["out"]), "tensorfold:all_n": float(life["n"]),
                    "tensorfold:all_since": float(life["since"] or 0)})
    out.update({
        "llamacpp:tokens_predicted_total": gen,
        # only the COMPUTED tokens, like local_compute on vLLM: the cached cost no prefill
        "llamacpp:prompt_tokens_total": max(0.0, prompt - cached),
        "llamacpp:prompt_seconds_total": mx.get("tensorfold_health:prefill_seconds_total" if new else
                                                "tensorfold_prefill_seconds_total", 0.0),
        "llamacpp:requests_processing": inflight,
        "llamacpp:requests_deferred": float((yzq or {}).get("high") or 0) + float((yzq or {}).get("low") or 0),
        "llamacpp:n_tokens_max": float(ctxmax),
        "llamacpp:spec_decode_num_drafts_total": rounds,
        "llamacpp:spec_decode_num_accepted_tokens_total": max(0.0, comp - reqs - rounds),
    })
    if new and "tensorfold:mtp_drafted_total" in mx:
        out.update({"tensorfold:has_drafted": 1.0,
                    "llamacpp:spec_decode_num_draft_tokens_total": mx["tensorfold:mtp_drafted_total"],
                    "llamacpp:spec_decode_num_accepted_tokens_total": mx.get("tensorfold:mtp_accepted_total", 0.0)})
    rec = tf_last_from_counters(reqs, prompt, cached) if new else (tf_last_record() or {})
    if new:
        fill_n, fill_pos, fill_len = tf_fill(port)
        out["tensorfold:fill_streams"] = float(fill_n)
        if fill_len is not None:
            out["tensorfold:fill_pos"], out["tensorfold:fill_len"] = float(fill_pos), float(fill_len)
    slots = [{
        "next_token": {"n_decoded": int(gen)},
        # 0 ON PURPOSE, as for tabby: prefill is read from the closed counters (prompt/seconds), not from a progress.
        # With the newer kit image y3 instead there is a real progress (tf_fill): the computed lines, monotone
        "n_prompt_tokens_processed": int(_TF_FILL["done"]) if new else 0,
        "n_prompt_tokens": int(rec.get("prompt") or 0),
        "n_prompt_tokens_cache": int(rec.get("cached") or 0),
        "n_ctx": int(ctxmax),
        "is_processing": inflight > 0,
    }]
    return out, slots



def tf_env(names=("glm53-flash-tf", "glm53-tf-r0")):
    """The running rank-0 container's environment (GLM53_TF_* knobs on the jayleaton kit, TF_GLM_* on MiaAI's), via
    docker: the process inside runs as root, so /proc/<pid>/environ is not readable from here."""
    for name in names:
        try:
            out = subprocess.run(["docker", "inspect", "-f", "{{range .Config.Env}}{{println .}}{{end}}", name],
                                 capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        if out.strip():
            return dict(l.split("=", 1) for l in out.splitlines() if "=" in l)
    return {}

# ==================== suzu/fetch/rank1.py
"""Rank-1 (tensor-parallel peer) probe and the RoCE link read."""



class Rank1Feed:
    """A second rank's live sample, from a plain HTTP poller: --rank1-url is a
    Prometheus-textformat endpoint polled every ~2 s. Never blocks the caller;
    unreachable or missing keys degrade to None and the panel says so."""

    STALE = 6.0              # seconds without a sample before the telemetry is called stale

    def __init__(self, url=None):
        import threading
        self.url = url
        self.thread = None
        self.down_since = None
        self.lock = threading.Lock()
        self.last = None     # (monotonic time, gpu dict or None, (MemTotal kB, MemAvailable kB) or None, None)
        self.metrics = None  # (monotonic time, info fields) from the rank's /metrics
        self.want_metrics = False
        self.closed = False

    def start(self):
        import threading
        if self.thread is None:
            self.thread = threading.Thread(target=self._run, name="rank1-feed", daemon=True)
            self.thread.start()

    def stop(self):
        pass

    def close(self):
        """For good: the poll loop ends (a host that reloads this module
        re-imports suzu.py whenever its mtime changes)."""
        self.closed = True

    def _run(self):
        backoff = 2.0
        while not self.closed and self.url:
            t0 = time.monotonic()
            try:
                txt = http_get(self.url, timeout=3, tries=1)
                if txt:
                    self._parse(txt)
            except Exception:
                pass
            if self.closed:
                return
            with self.lock:
                if self.last is None:
                    self.down_since = time.monotonic()
            backoff = 2.0 if time.monotonic() - t0 > 30 else min(30.0, backoff * 2)
            time.sleep(backoff)

    def _parse(self, txt):
        mx = parse_metrics(txt)
        gpu = {"util": mx.get("gpu_utilization", mx.get("gpu_util", mx.get("utilization.gpu"))),
               "mem": mx.get("gpu_mem_used_mb", mx.get("memory.used")),
               "memtot": mx.get("gpu_mem_total_mb", mx.get("memory.total")),
               "w": mx.get("gpu_power_watts", mx.get("power.draw")),
               "c": mx.get("gpu_temp_c", mx.get("temperature.gpu")),
               "mhz": mx.get("gpu_clock_mhz", mx.get("clocks.sm"))}
        if gpu["mem"] is not None and gpu["memtot"]:
            ram = (gpu["memtot"] * 1024.0, (gpu["memtot"] - gpu["mem"]) * 1024.0)
        else:
            ram = None
        if all(v is None for v in gpu.values()) and ram is None:
            return
        with self.lock:
            now = time.monotonic()
            self.down_since = None
            self.last = (now, gpu, ram, None)

    def _metrics_loop(self):
        """The rank's own /metrics (non-TensorFold clusters only)."""
        while not self.closed:
            if self.want_metrics and self.url:
                t0 = time.monotonic()
                txt = http_get(self.url, timeout=3, tries=1)
                info = None
                if txt:
                    info = {"ms": int((time.monotonic() - t0) * 1000)}
                    try:
                        mx = parse_metrics(txt)
                        info.update(run=g(mx, "vllm:num_requests_running"), kv=g(mx, "vllm:kv_cache_usage_perc"),
                                    prompt=g(mx, "vllm:prompt_tokens_total"), gen=g(mx, "vllm:generation_tokens_total"))
                    except Exception:
                        pass
                with self.lock:
                    self.metrics = (time.monotonic(), info)
            time.sleep(2.0)

    def info(self, tf=False):
        """The display dict (up, ms, run, kv, prompt, gen, gpu, ramUsedMb, ramTotalMb) plus age (s since the
        last sample) and telemetry ("live", "stale", "unreachable"). up: True / False only when KNOWN
        (a fresh sample / no sample for 20 s), None while it is not known: stale telemetry is not a dead rank."""
        self.want_metrics = not tf
        info = {"rank": 1, "up": None, "ms": None, "run": None, "kv": None, "prompt": None, "gen": None,
                "gpu": None, "ramUsedMb": None, "ramTotalMb": None, "availMb": None, "age": None,
                "container": None, "telemetry": "connecting"}
        if not self.url:
            info["telemetry"] = "no url"
            return info
        with self.lock:
            last, metrics, down = self.last, self.metrics, self.down_since
        now = time.monotonic()
        if last is not None:
            at, gpu, ram, _cont = last
            info["age"] = now - at
            info["gpu"] = gpu
            if ram and ram[0] and ram[1] is not None:
                info["ramTotalMb"] = ram[0] / 1024.0
                info["availMb"] = ram[1] / 1024.0
                info["ramUsedMb"] = (ram[0] - ram[1]) / 1024.0
            info["telemetry"] = "live" if info["age"] < self.STALE else "stale"
        if down is not None and (last is None or info["age"] >= self.STALE):
            info["telemetry"] = "unreachable"
        fresh = info["telemetry"] == "live"
        if tf:
            if fresh:
                info["up"] = True
            elif down is not None and now - down > 20:
                info["up"] = False
        else:
            if metrics is not None and now - metrics[0] < 10 and metrics[1]:
                info.update(metrics[1])
                info["up"] = True
            elif fresh:
                info["up"] = True          # the endpoint answers; no worker metrics (the panel says "metrics silent")
            elif down is not None and now - down > 20:
                info["up"] = False
        return info



def _rank1_feed(url=None):
    """One feed per PROCESS, not per copy of this module. A reload closes the previous copy's feed."""
    import atexit
    import gc
    old = getattr(sys, "_suzu_rank1_feed", None)
    if old is not None and hasattr(old, "close"):
        try:
            old.close()
        except Exception:
            pass
    for o in gc.get_objects():
        try:
            if type(o).__name__ == "Rank1Feed" and o is not old and not getattr(o, "closed", False):
                if hasattr(o, "close"):
                    o.close()
                else:
                    o.url = None
                    o.stop()
        except Exception:
            pass
    feed = Rank1Feed(url)
    sys._suzu_rank1_feed = feed
    atexit.register(feed.close)
    return feed



_RANK1 = _rank1_feed()



def fetch_rank1(url=None, tf=False):
    """The OTHER half of a tensor-parallel pair: the last sample of the background poller, at once (Rank1Feed). The very
    first call waits up to 3 s for a sample, so `--once` and the first frame show rank 1 instead of "probing"."""
    if _RANK1.thread is None and _RANK1.url:
        _RANK1.start()
        t_end = time.monotonic() + 3.0
        while time.monotonic() < t_end and _RANK1.last is None and _RANK1.down_since is None:
            time.sleep(0.05)
    return _RANK1.info(tf=tf)



_ROCE = {"prev": None, "tx": None, "rx": None, "pk": 0.0, "devs": ()}



def roce_link():
    """The tensor-parallel link as the counters see it: bytes/s out of and into this box's RoCE functions (the CX7;
    port_xmit_data / port_rcv_data count 4-byte words), summed over the ones that ever carried traffic. Rates over the
    last >= 0.5 s; peak-hold decays 2%/read. (tx, rx, peak, device names), rates None before two reads."""
    now = time.monotonic()
    tx = rx = 0
    devs = []
    try:
        names = sorted(os.listdir("/sys/class/infiniband"))
    except OSError:
        names = []
    for d in names:
        base = "/sys/class/infiniband/%s/ports/1/counters/" % d
        try:
            with open(base + "port_xmit_data") as f:
                x = int(f.read())
            with open(base + "port_rcv_data") as f:
                r = int(f.read())
        except (OSError, ValueError):
            continue
        if x or r:
            tx, rx = tx + 4 * x, rx + 4 * r
            devs.append(d)
    prev = _ROCE["prev"]
    if prev is None or now - prev[0] >= 0.5:
        if prev is not None and tx >= prev[1] and rx >= prev[2]:
            dt = now - prev[0]
            _ROCE["tx"], _ROCE["rx"] = (tx - prev[1]) / dt, (rx - prev[2]) / dt
            _ROCE["pk"] = max(_ROCE["pk"] * 0.98, _ROCE["tx"], _ROCE["rx"])
        _ROCE["prev"] = (now, tx, rx)
    _ROCE["devs"] = tuple(devs)
    return _ROCE["tx"], _ROCE["rx"], _ROCE["pk"], _ROCE["devs"]

# ==================== suzu/render/core.py
"""View state: construction, theme cycling, ramps, smoothing, tick."""



def _smooth(prev, target, dt, tau_up, tau_down):
    """Exponential approach, fast up / slow down - fade.rs smooth()."""
    tau = tau_up if target > prev else tau_down
    a = 1.0 - 2.718281828459045 ** (-dt / max(tau, 0.001))
    v = prev + (target - prev) * a
    return max(0.0, min(1.0, v))



def _routed_indices(layer, step, n_experts, k):
    """fade.rs routed_indices(): deterministic LCG spread, same constants."""
    if n_experts == 0:
        return []
    k = max(1, min(k, n_experts))
    M = (1 << 64) - 1
    x = ((layer * 0x9E3779B1) ^ (step * 0x85EBCA6B)) & M
    out, guard = [], 0
    while len(out) < k and guard < k * 8:
        x = (x * 6364136223846793005 + 1442695040888963407) & M
        e = (x >> 33) % n_experts
        if e not in out:
            out.append(e)
        guard += 1
    return out



class CoreMixin:
    WIDE = 118
    NARROW = 76
    FADE_ATTACK = 0.65
    FADE_RELEASE = 0.12

    def __init__(self, port, interval, show_log, theme="neon"):
        self.port, self.interval, self.show_log = port, interval, show_log
        self.t = Theme(theme)
        self.llm = os.environ.get("SUZU_LOG_DIR", os.path.expanduser("~/.local/share/suzu/logs"))
        self.prev = None
        self.h_gen = deque(maxlen=120)
        self.h_pf = deque(maxlen=120)
        self.h_acc = deque(maxlen=120)
        self.h_tau = deque(maxlen=120)
        # Same rolling windows, but as (numerator, denominator) pairs. A mean of
        # per-tick RATES weights a 0,2 s tick like a 4 s one; the average has to be
        # sum(tokens)/sum(seconds) or it is not the rate of anything.
        self.q_gen = deque(maxlen=120)
        self.q_pf = deque(maxlen=120)
        self.q_acc = deque(maxlen=120)
        self.q_tau = deque(maxlen=120)
        self.yzq = None
        # vLLM credits the two halves of the prefill rate on DIFFERENT ticks:
        # measured 23-08, prompt_tokens_total jumped +7294 at 20:27:34 and
        # request_prefill_time_seconds_sum jumped +6,203 s at 20:27:36. Anything
        # that demands both inside one sample therefore never fires and the row
        # reads 0 forever. Bank each half and divide once both have landed.
        self.pend_tok = 0.0
        self.pend_sec = 0.0
        self.info, self.info_at = {}, 0.0
        self.mode = "spec"          # "spec" | "layers" (toggled with g)
        self.view = "full"          # "full" | "perf" (p) | "pipes" (b)
        self.show_layers = False    # L: layer visualizer replaces the log section
        self.arch = None            # cached model topology
        self.arch_path = None
        self.mm_prev = None         # last mm_cache_queries_total (vision detect)
        self.mm_t = 0.0
        self.miss, self.last_frame = 0, None
        self.restarts = 0
        self.w = None
        self.w_at = None
        self.last_prefix = None
        self.act_ts = None
        self.mx_retry = 0.0              # backoff while /metrics is absent (see step)
        self.fade = {}                   # attack/release heat per cell (fade_step)
        self._fade_dt = time.time()
        self.pk_gen = 0.0                # peak-hold gen t/s (VU-style, slow decay)
        self.pk_at = 0.0
        self.pk_pf = 0.0                 # peak-hold prefill t/s (pipes view)
        self.pk_pf_at = 0.0
        self.pk_ch = ""                  # big-digit channel string
        self.fade_expert = []            # per-layer expert heat (fade.rs expert[])
        self.gpu = None                  # nvidia-smi rows (fetch_gpu, cached)
        self.rank1 = None                # rank-1 probe (fetch_rank1)
        self.lanes = None                # /yzq/lanes detail (fetch_yzq_lanes)
        self.memory = None               # /proc memory picture (fetch_memory)
        self.h_ttft = deque(maxlen=120)  # time-to-first-token per request, s



    def cycle_theme(self, step=1):
        i = (THEME_ORDER.index(self.t.name) + step) % len(THEME_ORDER)
        self.t = Theme(THEME_ORDER[i])
        _theme_save(self.t.name)   # remember it for the next run
        return self.t.name



    def ramps(self):
        """The RGB ramp set for the active theme: gold family for the limone
        themes, the upstream ocean ramps otherwise."""
        if self.t.name in PHOSPHOR_FAMILY:
            return HEAT_PHOS, VU_PHOS, FLOW_PHOS, TRACK_BG_PHOS
        if self.t.name in GOLD_FAMILY:
            return HEAT_GOLD, VU_GOLD, FLOW_GOLD, TRACK_BG_GOLD
        return HEAT_COOL, VU_COOL, FLOW_COOL, TRACK_BG_COOL



    def ramp_heat(self, v):
        return _col(_grad(self.ramps()[0], v))



    def ramp_heat_bg(self, v):
        return _bg_col(_grad(self.ramps()[0], v))



    def ramp_vu(self, v):
        return _col(_grad(self.ramps()[1], v))



    def ramp_flow(self, v):
        return _col(_grad(self.ramps()[2], v))



    def ramp_track(self):
        return self.ramps()[3]



    def fade_dt(self):
        """One shared dt per frame - call once, pass to every fade_step."""
        now = time.time()
        dt = now - self._fade_dt if now > self._fade_dt else 0.25
        self._fade_dt = now
        return dt



    def fade_step(self, key, hot, dt):
        """Advance one heat cell; returns its heat 0..1 (0 = cold)."""
        v = self.fade.get(key, 0.0)
        if hot:
            v = min(1.0, v + self.FADE_ATTACK * max(dt, 0.25) * 4)
        else:
            v *= max(0.0, 1.0 - self.FADE_RELEASE * dt)
        if v < 0.02:
            self.fade.pop(key, None)
            return 0.0
        self.fade[key] = v
        return v



    def expert_deltas(self):
        """Real routing from TensorFold's GET /experts (patches/0990: glm53tf, glm53mia y9+): per layer, how many times each expert was
        picked by decode windows since the previous read. None when the engine has no such endpoint (every engine
        but tensorfold: polled only there, so the others' logs do not fill with 404s) or on the first read."""
        if not getattr(self, "tf", False):
            return None
        now = time.time()
        st = getattr(self, "_xs", None)
        if st is None:
            st = self._xs = {"at": 0.0, "prev": None, "delta": None}
        if now - st["at"] < 0.5:
            return st["delta"]
        st["at"] = now
        txt = http_get("http://127.0.0.1:%d/experts" % self.port, timeout=2, tries=1)
        try:
            j = json.loads(txt) if txt else None
        except (json.JSONDecodeError, ValueError):
            j = None
        cnt = j.get("counts") if isinstance(j, dict) and j.get("on") else None
        # on: the engine counts (a second read gives the first delta); why: it has the route but counts nothing
        st["on"] = bool(cnt)
        st["why"] = (j.get("why") or j.get("error")) if isinstance(j, dict) and not cnt else None
        if not cnt:
            st["prev"], st["delta"] = None, None
            return None
        prev, st["prev"] = st["prev"], cnt
        if prev is None or len(prev) != len(cnt):
            st["delta"] = None
            return None
        st["delta"] = [[max(0, a - b) for a, b in zip(r, q)] for r, q in zip(cnt, prev)]
        return st["delta"]



    def fade_experts(self, arch, working, dt, token_step):
        """fade.rs expert heat: routed expert jumps to 1.0, cools with a
        0.6 s exponential. On TensorFold the identities are REAL (GET /experts,
        patches/0990): every expert picked since the last read jumps to its
        share of that layer's busiest one. Elsewhere they come from the same
        deterministic stand-in router as upstream."""
        real = self.expert_deltas()
        n_l = max(arch["n"], len(real)) if real is not None else arch["n"]     # + the MTP layer's row
        n_e = min(arch["experts"], 320 if real is not None else 256)
        if len(self.fade_expert) != n_l or \
                (self.fade_expert and len(self.fade_expert[0]) != n_e):
            self.fade_expert = [[0.0] * n_e for _ in range(n_l)]
        cool = 2.718281828459045 ** (-dt / 0.6)
        for row in self.fade_expert:
            for j in range(n_e):
                row[j] *= cool
        if real is not None:
            for l in range(min(n_l, len(real))):
                d = real[l]
                m = max(d) if d else 0
                if m <= 0:
                    continue
                row = self.fade_expert[l]
                for j in range(min(n_e, len(d))):
                    if d[j] > 0:
                        row[j] = max(row[j], d[j] / m)
        elif working:
            k = arch.get("topk") or 8
            for l in range(n_l):
                row = self.fade_expert[l]
                for e in _routed_indices(l, int(token_step), n_e, k):
                    row[e] = 1.0



    def refresh_info(self):
        if time.time() - self.info_at > 20:
            self.info = server_info(self.port)
            self.info_at = time.time()



    def step(self):
        self.yzq = fetch_yzq(self.port)
        self.lanes = fetch_yzq_lanes(self.port, config.LANES_URL) if config.LANES_URL else None
        self.gpu = fetch_gpu()
        self.memory = fetch_memory(pid=self.info.get("pid"))
        # Rank-1 probe: fetch_rank1 caches for 10 s, so this is polite
        # even every frame. Probed on EVERY engine shape now - on the TP=2
        # cluster it is the rank-1 worker, on a single-node engine it is the
        # second box itself (asked for on 21-42: visible on 1-node too).
        self.rank1 = fetch_rank1(tf=getattr(self, "tf", False))
        mx = slots = None
        now = time.time()
        if now >= self.mx_retry:
            txt = http_get("http://127.0.0.1:%d/metrics" % self.port)
            if txt is None:
                # Silence or 404. On TabbyAPI it is 404 and will stay 404: without
                # this pause the panel would hammer it twice a second and every
                # hit lands in the engine log (seen 29-08: thousands of
                # `GET /metrics 404` lines). Retried every minute, because under
                # :30001 the engine may change while the panel is open.
                self.mx_retry = now + 60
            else:
                self.mx_retry = 0.0
                mx = parse_metrics(txt)
                if "tensorfold_requests_total" in mx or "tensorfold_health:requests_total" in mx:
                    mx, slots = synth_tensorfold(mx, self.port, self.yzq)
                else:
                    slots = fetch_slots(self.port) or synth_slots_vllm(mx, self.port)
        if mx is None:
            mx, slots = synth_tabby(self.port, self.llm, self.yzq)
            if mx is None:
                return None
        cur = Sample(mx, slots, time.time())
        self.tf = cur.fam == "tensorfold"
        if not cur.ctxmax and cur.fam == "vllm":
            cur.ctxmax = vllm_ctxmax(self.port)
        self.refresh_info()
        # cmdline facts arrive with info; refresh_info only re-fetches every
        # 20 s. Same un-gated probe as the head of step(): the panel decides
        # cluster-vs-standalone from nnodes at render time, not here.
        self.rank1 = fetch_rank1(tf=self.tf)

        p = self.prev
        if p is not None and cur.restarted_from(p):
            self.restarts += 1
            p = None
            for h in (self.h_gen, self.h_pf, self.h_acc, self.h_tau,
                      self.q_gen, self.q_pf, self.q_acc, self.q_tau):
                h.clear()
            self.pend_tok = self.pend_sec = 0.0
            self.w = self.w_at = None
        self.prev = cur

        # Last engine activity. Real token deltas are the truth: a slot left in
        # "processing" with nothing decoded is not work. The first frame has no
        # predecessor to diff, so the clock is seeded from the log's mtime.
        if cur.busy or (p is not None and (cur.gen > p.gen or cur.pp > p.pp)):
            self.act_ts = cur.t
        elif self.act_ts is None:
            self.act_ts = log_mtime(self.llm)

        # Time to first token: the poll where a quiet engine turns busy AND
        # starts decoding. Quantised to the interval (1 s), like the repo's
        # TTFT - honest about what a poller can know.
        if p is not None and not p.busy and cur.busy:
            self._ttft_pending = cur.t
        if p is not None and getattr(self, "_ttft_pending", None) \
                and cur.dec > p.dec:
            self.h_ttft.append(max(0.0, cur.t - self._ttft_pending))
            self._ttft_pending = None
        elif p is not None and not cur.busy:
            self._ttft_pending = None

        live = {"gen": 0.0, "pf": 0.0}
        if p is not None:
            dt = max(1e-3, cur.t - p.t)
            # n_decoded restarts at 0 on every request: a negative delta is a new
            # task, not a stall -> drop the window instead of reporting nonsense.
            ddec = cur.dec - p.dec
            if ddec >= 0 and cur.busy:
                live["gen"] = ddec / dt
            dpp, dps = cur.pp - p.pp, cur.ps - p.ps
            pf_pushed = False
            if cur.fam == "vllm":
                # NEVER derive prefill from ppp here. On vLLM /slots is synthetic
                # (synth_slots_vllm) and n_prompt_tokens_processed is really
                # kv_cache_usage_perc * pool — KV OCCUPANCY, not prefill progress.
                # Its delta is the rate the cache fills, which a prefix-cache hit
                # or a fresh 100k conversation makes jump by the whole context in
                # one tick; and it is quantised by the pool (2.19 M tokens here,
                # so one step of the gauge is ~22 k phantom tokens). That is where
                # "prompt t/s 93.550" and "avg 4487" came from (measured 23-08).
                # The counters are the only honest source: prompt_tokens_total is
                # credited at request close, request_prefill_time_seconds_sum is
                # the matching denominator, so their ratio is tokens per second
                # OF PREFILL — which is what the label claims.
                if dpp > 0:
                    self.pend_tok += dpp
                if dps > 1e-3:
                    self.pend_sec += dps
                if self.pend_tok > 0 and self.pend_sec > 1e-3:
                    live["pf"] = self.pend_tok / self.pend_sec
                    self.h_pf.append(live["pf"])
                    self.q_pf.append((self.pend_tok, self.pend_sec))
                    self.pend_tok = self.pend_sec = 0.0
                    pf_pushed = True
            else:
                # llama.cpp: /slots really does advance during the prefill, and
                # that live reading is the whole reason this file exists.
                dppp = cur.ppp - p.ppp
                if dppp > 0:
                    live["pf"] = dppp / dt
                elif dpp > 0 and dps > 1e-3:
                    live["pf"] = dpp / dps

            dd, da, dn = cur.draft - p.draft, cur.acc - p.acc, cur.drafts - p.drafts
            if cur.tf_nodraft and dn > 0:
                self.w = {"alpha": 0.0, "tau": da / dn, "dwin": 0.0, "pos": [], "drafts": dn}
                self.w_at = cur.t
                self.h_tau.append(self.w["tau"])
                self.q_tau.append((da, dn))
            elif dd > 0 and dn > 0:
                self.w = {"alpha": da / dd, "tau": da / dn, "dwin": dd / dn,
                          "pos": [a - b for a, b in zip(cur.pos, p.pos)],
                          "drafts": dn}
                self.w_at = cur.t
                self.h_acc.append(self.w["alpha"] * 100)
                self.h_tau.append(self.w["tau"])
                self.q_acc.append((da, dd))
                self.q_tau.append((da, dn))

            # ONLY ticks that actually decoded. The old gate was `cur.busy`, and
            # on vLLM busy is just num_requests_running > 0 — which is also true
            # for the whole PREFILL, where generation_tokens_total does not move.
            # At 245k ctx that is many seconds of 0.0 pushed into the history per
            # turn, and they dragged "avg gen t/s" down while the panel was
            # showing a healthy live rate. A prefill is not a slow generation.
            if ddec > 0:
                self.h_gen.append(live["gen"])
                self.q_gen.append((ddec, dt))
            if live["pf"] > 0 and not pf_pushed:
                self.h_pf.append(live["pf"])
                self.q_pf.append((dpp, dps) if dps > 1e-3 else (live["pf"] * dt, dt))
            # prefill peak-hold for the pipes view (same decay rule as gen)
            if live["pf"] >= self.pk_pf:
                self.pk_pf, self.pk_pf_at = live["pf"], time.time()
            elif time.time() - self.pk_pf_at > 2.0:
                self.pk_pf = max(live["pf"], self.pk_pf - 0.02 * self.pk_pf)
        return cur, live



    @staticmethod
    def mean(seq):
        return sum(seq) / len(seq) if seq else 0.0



    @staticmethod
    def wmean(seq):
        """sum(numerators)/sum(denominators) over the window - the rate itself,
        not the mean of the per-tick rates."""
        num = sum(a for a, _ in seq)
        den = sum(b for _, b in seq)
        return num / den if den else 0.0

# ==================== suzu/render/widgets.py
"""Drawing primitives: bars, sparks, boxes, gauges, big digits."""



class WidgetsMixin:
    def bar(self, frac, width, color=None, invert=False):
        """Gradient bar: every cell is painted by ITS OWN position on the ramp.

        That is the point. A flat bar says "some amount"; a graded one says
        which band you are in without reading the number, because the colour
        breaks under your eye at the same cutoffs heat() uses. Pass color= to
        force one flat colour (where the hue already means something else).
        invert=True for metrics where high is BAD, e.g. a full kv cache.
        """
        frac = max(0.0, min(1.0, frac))
        width = max(0, width)
        quarters = int(round(frac * width * 4))           # dithered edge: a cell in quarters, ░ ▒ ▓ then █
        fill, edge = divmod(quarters, 4)
        if fill >= width:
            fill, edge = width, 0
        t = self.t
        if t.plain:
            return "█" * fill + (DITHER[edge] if edge else "") + "░" * (width - fill - (1 if edge else 0))
        if color is not None:
            body = color + "█" * fill + (DITHER[edge] if edge else "")
        else:
            out, last = [], None
            for i in range(fill + (1 if edge else 0)):
                # centre of the cell, so a 1-cell bar is not pinned at 0.0
                pos = (i + 0.5) / width
                col = t.heat(1.0 - pos if invert else pos)
                if col != last:
                    out.append(col)
                    last = col
                out.append("█" if i < fill else DITHER[edge])
            body = "".join(out)
        return body + t.frame + "░" * (width - fill - (1 if edge else 0)) + RST



    def spark(self, series, width):
        t = self.t
        if width <= 0:
            return ""
        if not series:
            return ("." * width) if t.plain else (t.frame + "·" * width + t.rst())
        s = list(series)[-width:]
        lo, hi = min(s), max(s)
        span = (hi - lo) or 1.0
        out = []
        for v in s:
            idx = min(len(BLOCKS) - 1, int((v - lo) / span * (len(BLOCKS) - 1) + 0.5))
            ch = BLOCKS[idx]
            # llm-visuals flow ramp by level: violet -> cyan -> white (or the
            # gold family on the limone themes)
            out.append(ch if t.plain
                       else self.ramp_flow(v / hi if hi else 0.0) + ch)
        rest = max(0, width - len(s))
        tailpad = ("." if t.plain else "·") * rest
        return ("".join(out) + t.rst()
                + ((t.frame + tailpad + t.rst()) if rest else ""))



    def box(self, title, body, width, color=None):
        t = self.t
        color = color if color is not None else t.head
        if width < 12:                      # too narrow to frame - print bare
            return [clip(b, width) for b in body]
        lab = " %s%s%s " % (color, title, t.rst())
        lab = clip(lab, max(4, width - 4))
        top = (t.frame + "┌─" + t.rst() + lab + t.frame
               + "─" * max(0, width - vis_len(lab) - 3) + "┐" + t.rst())
        rows = [top]
        for ln in body:
            rows.append(t.frame + "│" + t.rst()
                        + pad(clip(ln, width - 2), width - 2)
                        + t.frame + "│" + t.rst())
        rows.append(t.frame + "└" + "─" * (width - 2) + "┘" + t.rst())
        return rows



    def pct(self, frac, rw=10, good_high=False):
        """good_high: a high prefix-cache rate is GOOD, a full KV cache is BAD."""
        t = self.t
        if frac is None:
            return t.frame + "n/a" + t.rst()
        w = max(4, min(14, rw))
        # the NUMBER keeps the flat verdict colour; the BAR is graded, and on a
        # metric where high is bad the gradient runs the other way round
        col = t.heat(frac) if good_high else t.heat(1.0 - frac)
        return "%s %s%3.0f%%%s" % (self.bar(frac, w, invert=not good_high),
                                   col, frac * 100, t.rst())



    def hrow(self, label, series, sw, fmt, col, note, lw=12, fit=None):
        """fit: the row's inner width; the sparkline then gets exactly what the label, value, range and note leave."""
        t = self.t
        cur = series[-1] if series else 0.0
        rng = ("%s%s-%s%s" % (t.frame, fmt % min(series), fmt % max(series), t.rst())
               if series else t.frame + "-" + t.rst())
        tailtxt = (" %s%s%s" % (t.frame, note, t.rst())) if note else ""
        if fit:
            sw = max(4, fit - lw - 3 - vis_len(fmt % cur) - vis_len(rng) - vis_len(tailtxt))
        return ("%s%-*s%s %s %s%s%s %s%s"
                % (t.label, lw, label, t.rst(), self.spark(series, sw),
                   col + t.bold(), fmt % cur, t.rst(), rng, tailtxt))



    def big_digits(self, value, width, color, fmt="%.1f"):
        """llm-visuals big_digits(): 3-row half-block numerals, glyph shapes
        copied verbatim from src/render.rs. Falls back to plain text when the
        art does not fit the width."""
        txt = fmt % value
        rows = ["", "", ""]
        for i, ch in enumerate(txt):
            g = BIG_GLYPHS.get(ch)
            if g is None:
                continue
            for r in range(3):
                if i > 0:
                    rows[r] += " "
                rows[r] += g[r]
        rows = [r.rstrip() for r in rows]
        if max(len(r) for r in rows) > width:
            return [color + txt + (RST if color else "")]
        end = RST if color else ""
        return [color + r + end for r in rows]



    def gauge(self, frac, width, peak=None):
        """llm-visuals gauge(): half-cell precision, VU ramp by position
        (green -> amber -> red), white peak-hold notch on a visible track."""
        frac = max(0.0, min(1.0, frac))
        t = self.t
        if t.plain:
            return self.bar(frac, width)
        halves = int(round(frac * width * 2))
        peak_idx = (min(width - 1, int(max(0.0, min(1.0, peak)) * width))
                    if peak is not None else None)
        col = self.ramp_track()
        out = []
        for x in range(width):
            pos = (x + 0.5) / max(1, width)
            filled = max(0, min(2, halves - x * 2))
            if peak_idx == x and filled < 2:
                out.append(_col(WHITE_FG) + _bg_col(col) + "▌" + RST)
            elif filled == 2:
                out.append(self.ramp_vu(pos) + "█" + RST)
            elif filled == 1:
                out.append(self.ramp_vu(pos) + _bg_col(col) + "▒" + RST)
            else:
                out.append(_col(col) + "█" + RST)
        return "".join(out)



    @staticmethod
    def pulse_head(i):
        return PULSE[i % len(PULSE)]



    def vmeter(self, frac, hold, height, width):
        """Vertical VU strip, rows top to bottom: eighth-block precision, ramp
        colour by height, white peak-hold line that sits then falls (the
        repo's vmeter, on our ramps)."""
        t = self.t
        height = max(1, height)
        frac = max(0.0, min(1.0, frac))
        hold = max(0.0, min(1.0, hold))
        if t.plain:
            fill = int(round(frac * height))
            return ["█" * width if r >= height - fill else "░" * width
                    for r in range(height)]
        total = int(round(frac * height * 8))
        hold_row = None
        if hold > frac + 1e-3:
            cell = min(height - 1, max(0, int(hold * height * 0.999)))
            hold_row = height - 1 - cell
        rows = []
        for r in range(height):
            from_bottom = height - 1 - r
            e = max(0, min(8, total - from_bottom * 8))
            col = self.ramp_vu((from_bottom + 0.5) / height)
            if hold_row == r and e < 8:
                rows.append(_col(WHITE_FG) + "▔" * width + RST)
            elif e == 8:
                rows.append(col + "█" * width + RST)
            elif e == 0:
                rows.append(" " * width)
            else:
                rows.append(col + BLOCKS[e - 1] * width + RST)
        return rows

# ==================== suzu/render/panels.py
"""Panel builders: cluster, lanes, model strip, experts, heatmap, slots."""



class PanelsMixin:
    MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
    WEEKDAYS = ("Mo", "Tu", "We", "Th", "Fr", "Sa", "Su")

    def cluster_title(self):
        return "tensor-parallel cluster" if self.info.get("nnodes") in ("2", 2) else "gb10 . rank 0 + rank 1"



    def cluster_panel(self, width):
        """Both GB10s in one box, in aligned columns (05-10: "replace gpu . memory . gb10 with that"): this box
        (rank 0 on the TP=2 cluster: GPU on the VU ramp, watts, clock, temperature, unified memory, the engine's rss)
        and rank 1 (from Rank1Feed's HTTP poller: same columns, plus how fresh the sample is), then on the cluster
        the link row: bytes/s through this box's RoCE functions (roce_link: every round's all-gathers, so it moves
        with decode and prefill) and whether the pair is in step (rank 0's /health iteration_s: a stalled pair keeps
        growing it). On a single-node engine the same two box rows, no link, and an explicit UNREACHABLE when rank 1
        is asleep."""
        cluster = self.info.get("nnodes") in ("2", 2)
        t = self.t
        r1 = self.rank1 or {"up": None, "telemetry": "connecting"}
        inner = width - 2
        gw = max(8, min(32, (inner - 118) // 2 + 8))
        narrow = inner < 110

        def name(rank, box):
            return pad("%s%s%s  %s%s%s" % (t.head, rank, t.rst(), t.label, box, t.rst()), 14) if cluster \
                else pad("%s%s%s" % (t.label, box, t.rst()), 6)

        def gpu_cols(gp):
            if not gp:
                return pad("%sGPU%s %s" % (t.label, t.rst(), t.dim() + "n/a" + t.rst()), 4 + gw + 5 + (0 if narrow else 22))
            u = gp.get("util") or 0.0
            core = "%sGPU%s %s %s%3.0f%%%s" % (t.label, t.rst(), self.gauge(u / 100.0, gw), t.value + t.bold(), u, t.rst())
            if narrow:
                return core
            return core + "%s %4.0f W %5.0f MHz %3.0fC%s" % (t.frame, gp.get("w") or 0, gp.get("mhz") or 0,
                                                           gp.get("c") or 0, t.rst())

        def ram_cols(used_mb, total_mb, avail_mb):
            if not total_mb:
                return pad("", 28)
            return pad("   %sRAM%s %3.0f/%3.0fG %s%4.1fG free%s"
                       % (t.label, t.rst(), used_mb / 1024.0, total_mb / 1024.0, t.frame, (avail_mb or 0) / 1024.0,
                          t.rst()), 28)

        tel = r1.get("telemetry")
        if tel == "no url":
            tel_s = ("%srank 1: no url%s" if narrow else "%srank 1: --rank1-url not set%s") % (t.dim(), t.rst())
        elif tel == "live":
            tel_s = "%s%.1f s ago%s" % (t.frame, r1.get("age") or 0.0, t.rst())
        elif tel == "stale":
            tel_s = "%ssample stale %.0f s%s" % (t.heat(0.4), r1.get("age") or 0.0, t.rst())
        elif tel == "unreachable":
            tel_s = "%srank 1 unreachable%s" % (t.heat(0.1), t.rst())
        else:
            tel_s = "%sconnecting...%s" % (t.dim(), t.rst())
        if r1.get("up") is False:
            state = ("%sRANK 1 DOWN - decode will stall%s" if cluster else "%sUNREACHABLE (asleep?)%s") % (t.heat(0.1), t.rst())
        elif r1.get("up") is True:
            cont = r1.get("container") or ""
            # "glm53-flash-tf Up 52 minutes" -> "up 52 minutes": the name says nothing the row does not
            st = cont.split(" ", 1)[1] if " " in cont else ""
            state = "%s%s%s" % (t.heat(0.9), (st[:1].lower() + st[1:]) if st else "up", t.rst())
        else:
            state = ""
        def fit_row(base, *pieces):
            """Append optional columns while the row still fits the box.
            A column that does not fit is dropped whole - never cut mid-word:
            at 78 columns 'rank 1: --rank1-url not set' used to lose its tail
            and read as if the feed had simply gone quiet."""
            row = base
            for p in pieces:
                if not p:
                    continue
                cand = row + "   " + p
                if vis_len(cand) <= inner:
                    row = cand
            return row

        body = []
        g0 = (self.gpu or [None])[0]
        mem = self.memory or {}
        tot, avail = (mem.get("total") or 0) / 1024.0, (mem.get("avail") or 0) / 1024.0
        rss = ("%sengine rss %.1fG%s" % (t.frame, (mem.get("rss") or 0) / 1048576.0, t.rst())) if mem.get("rss") else ""
        r0 = name("rank 0", "this box") + gpu_cols(g0)
        ram0 = ram_cols(tot - avail, tot, avail)
        if vis_len(r0 + ram0) <= inner:
            r0 += ram0
        body.append(fit_row(r0, rss))
        r1row = name("rank 1", "remote") + gpu_cols(r1.get("gpu"))
        ram1 = ram_cols(r1.get("ramUsedMb"), r1.get("ramTotalMb"), r1.get("availMb"))
        if vis_len(r1row + ram1) <= inner:
            r1row += ram1
        body.append(fit_row(r1row, state, tel_s))
        if cluster:
            tx, rx, pk, devs = roce_link()
            if tx is None:
                link = name("link", "CX7 ") + "%sNET%s %smeasuring...%s" % (t.label, t.rst(), t.dim(), t.rst())
            else:
                frac = (max(tx, rx) / pk) if pk else 0.0
                link = (name("link", "CX7 ") + "%sNET%s %s" % (t.label, t.rst(), self.gauge(frac, gw))
                        + "%s %sout %9s/s%s" % (t.rst(), t.value, si_bytes(tx), t.rst()))
                tail = "  in %9s/s%s" % (si_bytes(rx), t.rst())
                if vis_len(link + tail) <= inner:
                    link += tail
                if not narrow:
                    tail = "   %speak %s/s . %s%s" % (t.frame, si_bytes(pk), "+".join(devs) or "no RoCE device", t.rst())
                    if vis_len(link + tail) <= inner:
                        link += tail
            hl = _TF_HEALTH if getattr(self, "tf", False) else {"at": None}
            if hl.get("at") is not None and time.monotonic() - hl["at"] < 10:
                it = hl.get("iteration_s") or 0.0
                if hl.get("ok") is False or hl.get("fatal"):
                    extra = "   %sENGINE FATAL - restart both ranks%s" % (t.heat(0.05), t.rst())
                elif it >= 30:
                    extra = "   %sSTALLED: iteration running %.0f s (ranks out of step?)%s" % (t.heat(0.1), it, t.rst())
                else:
                    extra = "   %sin step%s %s(iteration %.1f s)%s" % (t.heat(0.9), t.rst(), t.frame, it, t.rst())
                if vis_len(link + extra) <= inner:
                    link += extra
            body.append(link)
            if r1.get("run") is not None or r1.get("kv") is not None:      # the vLLM cluster's rank-1 /metrics
                bits = []
                if r1.get("run") is not None:
                    bits.append("req %d" % r1["run"])
                if r1.get("kv") is not None:
                    bits.append("kv %.0f%%" % (r1["kv"] * 100))
                if r1.get("gen") is not None:
                    bits.append("gen %s tok" % human(r1["gen"]))
                if r1.get("prompt") is not None:
                    bits.append("prompt %s tok" % human(r1["prompt"]))
                body.append(pad("", 14) + "%srank 1 worker: %s%s" % (t.frame, " . ".join(bits), t.rst()))
        return [clip(x, inner) for x in body]



    def lanes_panel(self, cur, width):
        """The optional gateway, the thing no upstream dashboard has: the
        two-lane queue that decides what reaches the engine, and the preemption
        machinery that protects the human's lane."""
        q, l = self.yzq, self.lanes
        if not q and not l:
            if config.LANES_URL:
                t = self.t
                return [clip("%sno lanes endpoint%s" % (t.frame, t.rst()), width - 2)]
            return None
        t = self.t
        hi = int((q or {}).get("high") or 0)
        lo = int((q or {}).get("low") or 0)
        par = int((q or {}).get("parallel") or 1)
        run = len((q or {}).get("running") or [])
        rl = [{"alta": "high", "bassa": "low"}.get(r.get("prio"), r.get("prio")) for r in ((q or {}).get("running") or [])]
        inf = ("high %d . low %d" % (rl.count("high"), rl.count("low"))) if rl else "-"   # 05-10: every running request, not the oldest
        free = (l or {}).get("free_slots")
        bits = ["%shigh%s %d queued . running %d/%d (%s)"
                % (t.head, t.rst(), hi, run, par, inf),
                "%slow %s %d queued" % (t.accent, t.rst(), lo)]
        if free is not None:
            bits.append("free slots %s" % free)
        for k, lab in (("beat_refused", "beat refused"),
                       ("beat_kill", "beat killed")):
            v = (q or {}).get(k)
            if v:
                bits.append("%s %s%d%s" % (lab, t.heat(0.3), int(v), t.rst()))
        return [clip(" . ".join(bits), width - 2)]



    def model_strip(self, width):
        """One line of what is actually loaded - the repo's model strip,
        reduced to the one model this box serves. Facts come from the engine
        cmdline, not from a config file that can rot."""
        i = self.info
        if not i.get("model") and not i.get("engine"):
            return None
        bits = []
        model = (i.get("model") or "?").split("/")[-1]
        bits.append(model)
        if i.get("quant"):
            bits.append(str(i["quant"]))
        if i.get("kvdtype"):
            bits.append("kv " + str(i["kvdtype"]))
        if i.get("tp"):
            bits.append("TP %s" % i["tp"])
        if i.get("seqs"):
            bits.append("seqs %s" % i["seqs"])
        if i.get("maxlen"):
            bits.append("ctx %s" % human(float(i["maxlen"])))
        if i.get("spec"):
            bits.append("%s k=%s" % (i["spec"], i.get("k") or "?"))
        t = self.t
        head = "  %s>%s " % (t.head, t.rst())
        line = head
        for n, b in enumerate(bits):
            cand = line + ("" if n == 0 else " . ") + b
            if vis_len(cand + t.rst()) <= width - 2:
                line = cand
            elif n == 0:                       # the model name itself: clip it
                line = cand
            else:
                break
        return [clip(line + t.rst(), width - 2)] if bits else None



    def expert_grid(self, width, rows_avail=6, hot_fn=None):
        """Port of render.rs render_experts(): the expert heatmap.

        One block per group of experts per layer row; block heat = hottest
        expert in the group; two layers share a row through the ▀ half block
        (fg = upper layer, bg = lower layer), exactly as upstream. Colour is
        the HEAT ramp by TIME since routed - white now, cooling to dark.
        Identities come from the same deterministic stand-in router as
        fade.rs (real routing needs the server /experts patch), and the
        title says `simulated`, like the repo.
        """
        t = self.t
        arch = self.model_arch()
        if not arch:
            return []
        n_l = len(self.fade_expert) or arch["n"]                    # fade_experts sized it (MTP row included)
        n_e = len(self.fade_expert[0]) if self.fade_expert and self.fade_expert[0] else min(arch["experts"], 256)
        w = max(8, width - 2)
        cols = max(1, w - 4)                 # 4 cells for the layer label
        per_block = max(1, (n_e + cols - 1) // cols)
        n_blocks = max(1, (n_e + per_block - 1) // per_block)
        bw = max(1, min(4, cols // n_blocks))
        gap = 1 if (bw >= 2 and (bw + 1) * n_blocks <= cols) else 0
        rows_avail = max(1, rows_avail)
        layers_per_slot = 1 if n_l <= rows_avail \
            else max(1, (n_l + 2 * rows_avail - 1) // (2 * rows_avail))
        half = n_l > rows_avail       # was `layers_per_slot > 1`: 46 layers in 23 rows has lps 1 and showed 0-22 only
        n_slots = (n_l + layers_per_slot - 1) // layers_per_slot
        text_rows = (n_slots + 1) // 2 if half else n_slots

        def heat(slot, b):
            l0 = slot * layers_per_slot
            l1 = min(l0 + layers_per_slot, n_l)
            e0 = b * per_block
            e1 = min(e0 + per_block, n_e)
            m = 0.0
            for l in range(l0, l1):
                row = self.fade_expert[l] if l < len(self.fade_expert) else None
                if not row:
                    continue
                for e in range(e0, e1):
                    if e < len(row) and row[e] > m:
                        m = row[e]
            return m

        body = []
        for r in range(min(text_rows, rows_avail)):
            top = r * 2 if half else r
            bot = top + 1
            lab = "%3d " % (top * layers_per_slot)
            cells = [lab] if t.plain else [t.frame + lab + t.rst()]
            for b in range(n_blocks):
                ht = heat(top, b)
                if t.plain:
                    cells.append(" .:-=+*#%@"[min(9, int(ht * 9.999))]
                                 * bw if ht > 0 else " " * bw)
                elif half:
                    hb = heat(bot, b) if bot < n_slots else 0.0
                    cells.append(self.ramp_heat(ht)
                                 + self.ramp_heat_bg(hb) + "▀" * bw + RST)
                else:
                    cells.append(self.ramp_heat_bg(ht) + " " * bw + RST)
                if gap and b + 1 < n_blocks:
                    cells.append(" ")
            body.append(clip("".join(cells), width - 2))
        return body



    def totals_rows(self, cur):
        """Every token the engine has handled since it started: prompt in (and how much of it came from cache),
        generated out, requests. Short units: the box is ~44 columns."""
        t = self.t

        def sh(n):
            n = float(n or 0)
            for div, suf in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
                if n >= div:
                    return "%.1f%s" % (n / div, suf)
            return "%d" % n
        reqs = (" . %d req" % cur.nreq) if cur.nreq is not None else ""
        out = "%s%-13s%s %s%s%s%s%s%s" % (t.label, "tokens out", t.rst(), t.value, sh(cur.gen), t.rst(),
                                          t.frame, reqs, t.rst())
        if cur.tot_in is None:
            return [out]
        cached = (" (%s cached)" % sh(cur.tot_cached)) if cur.tot_cached is not None else ""
        rows = ["%s%-13s%s %s%s%s%s%s%s" % (t.label, "tokens in", t.rst(), t.value, sh(cur.tot_in), t.rst(),
                                            t.frame, cached, t.rst()), out]
        if cur.all_n:
            since = time.strftime("%d-%m", time.localtime(cur.all_since)) if cur.all_since else "?"
            rows.append("%s%-13s%s %s%s in . %s out%s%s . %d req%s"
                        % (t.label, "since " + since, t.rst(), t.value, sh(cur.all_in), sh(cur.all_out), t.rst(),
                           t.frame, cur.all_n, t.rst()))
        rows += self.hermes_rows(sh)
        return rows



    @staticmethod
    def short(n):
        n = float(n or 0)
        for div, suf in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
            if n >= div:
                return ("%.0f%s" if n / div >= 100 else "%.1f%s") % (n / div, suf)
        return "%d" % n



    def heat_stats(self, days, today):
        """The numbers next to the map, stacked: the best days in order, then today, the last 7 days, the mean of
        the active days and the total."""
        import datetime as _dt
        md = lambda d: "%s %s" % (self.MONTHS[int(d[5:7]) - 1], d[8:10])
        top = sorted(((v, d) for d, v in days.items() if v > 0), reverse=True)[:3]
        rows = [("#%d" % (i + 1), v, md(d)) for i, (v, d) in enumerate(top)]
        week = sum(days.get((today - _dt.timedelta(days=i)).isoformat(), 0.0) for i in range(7))
        active = [v for v in days.values() if v > 0]
        rows += [("today", days.get(today.isoformat(), 0.0), ""), ("7 days", week, ""),
                 ("avg/day", sum(active) / len(active) if active else 0.0, "%d active" % len(active)),
                 ("total", sum(active), "since " + md(min(days)))]
        return rows



    def heatmap_rows(self, width, days=None, today=None):
        """Generated tokens per day as squares, like GitHub's contributions: one column a week (Mo-Su top down), the
        months on top over the week holding their 1st, the last column this week. Five levels, dithered and coloured:
        none, then the quartiles of the days with traffic. The stats stand on the right, stacked (below the legend
        when the box is too narrow for both). As many weeks as fit, the most recent."""
        import datetime as _dt
        days = hermes_days() if days is None else days
        if not days:
            return []
        t = self.t
        today = today or _dt.date.today()
        first = min(_dt.date.fromisoformat(d) for d in days)
        mon_now = today - _dt.timedelta(days=today.weekday())
        need = (mon_now - (first - _dt.timedelta(days=first.weekday()))).days // 7 + 1
        stats = self.heat_stats(days, today)
        cells = [("%-7s %6s %s" % (lab, self.short(v), note)).rstrip() for lab, v, note in stats]
        statw = max(len(c) for c in cells)
        inner = width - 2
        GAP = 8                                           # between the grid and the stats, which sit on the right edge
        side = inner >= 3 + 3 * 8 + GAP + statw           # at least 8 weeks next to the stats, else stack them
        W = max(1, min(need, (inner - 3 - ((GAP + statw) if side else 0)) // 3))
        mons = [mon_now - _dt.timedelta(weeks=W - 1 - k) for k in range(W)]
        vals = sorted(v for v in days.values() if v > 0)
        q = [vals[min(len(vals) - 1, int(len(vals) * f))] for f in (0.25, 0.5, 0.75)] if vals else [0, 0, 0]

        def level(v):
            return 0 if v <= 0 else 1 + sum(v > x for x in q)

        def cell(lv):
            # two glyphs a day: a terminal cell is about twice as tall as wide, so two side by side make a square
            ch = DITHER[lv] * 2
            if t.plain:
                return ch
            return (t.frame if lv == 0 else t.heat(lv / 4.0)) + ch + t.rst()

        gw = 3 + 3 * W                                    # the grid's visible width, labels included
        head = [" "] * (gw + 4)                           # a label may run past the last column (this week's month)
        last = -9
        for k, m in enumerate(mons):
            starts = [m + _dt.timedelta(days=i) for i in range(7) if (m + _dt.timedelta(days=i)).day == 1]
            if starts and 3 + 3 * k - last >= 4:
                for j, ch in enumerate(self.MONTHS[starts[0].month - 1]):
                    head[3 + 3 * k + j] = ch
                last = 3 + 3 * k
        rows = [t.label + "".join(head).rstrip() + t.rst()]
        for wd in range(7):
            line = t.label + self.WEEKDAYS[wd] + t.rst() + " "
            for m in mons:
                d = m + _dt.timedelta(days=wd)
                line += (cell(level(days.get(d.isoformat(), 0.0))) if d <= today else "  ") + " "
            if side and wd < len(stats):
                lab, v, note = stats[wd]
                col = t.value if lab != "today" else (t.heat(level(v) / 4.0) if v > 0 else t.frame)
                st = (t.label + "%-7s" % lab + t.rst() + " " + col + "%6s" % self.short(v) + t.rst()
                      + ((" " + t.frame + note + t.rst()) if note else ""))
                line = pad(line, inner - statw - 1) + st  # right-aligned block, one column off the right edge
            rows.append(line.rstrip())
        rows.append("")
        rows.append("%sless%s %s %smore%s" % (t.frame, t.rst(), " ".join(cell(lv) for lv in range(5)), t.frame, t.rst()))
        if not side:
            rows.append("")
            rows += [t.label + c[:8] + t.rst() + t.value + c[8:] + t.rst() for c in cells]
        return rows



    def hermes_rows(self, sh, top=3):
        """The harness total over all the models it used (hermes_usage), and the top ones by tokens."""
        u = hermes_usage()
        if not u:
            return []
        t = self.t
        since = time.strftime("%d-%m", time.localtime(u["since"])) if u["since"] else "?"
        rows = ["%s%-13s%s %s%s in%s %s(%s cached)%s %s. %s out%s"
                % (t.label, "all models", t.rst(), t.value, sh(u["in"] + u["cached"]), t.rst(), t.frame,
                   sh(u["cached"]), t.rst(), t.value, sh(u["out"]), t.rst()),
                "%s%-13s%s %s{:,} calls . %d models . since %s%s".format(u["calls"])
                % (t.label, "", t.rst(), t.frame, len(u["models"]), since, t.rst())]
        for m, _c, i, cr, o, _r, _f in u["models"][:top]:
            rows.append("%s  %-24s%s %s%s in . %s out%s" % (t.frame, m[:24], t.rst(), t.value, sh(i + cr), sh(o), t.rst()))
        return rows



    @staticmethod
    def tf_frac(tau, k=7):
        """Tokens a round as a fraction of a full draft block (k drafts + the bonus token): only a colour scale."""
        return min(1.0, (tau + 1.0) / (max(1, k) + 1.0))



    def fill_row(self, cur):
        """TensorFold (newer kit): the prompts in prefill, with the longest one's progress if /health says so (y3)."""
        t = self.t
        n, pos, ln = cur.tf_fill
        if not n:
            return "%s%-13s%s %s-%s" % (t.label, "prefill", t.rst(), t.frame, t.rst())
        if ln:
            return ("%s%-13s%s %s%s / %s%s  %s%.0f%%%s  %s%d stream%s"
                    % (t.label, "prefill", t.rst(), t.value, human(pos), human(ln), t.rst(),
                       t.heat(pos / ln), 100.0 * pos / ln, t.rst(), t.frame, int(n), t.rst()))
        return "%s%-13s%s %s%d stream%s" % (t.label, "prefill", t.rst(), t.heat(0.7), int(n), t.rst())



    def tf_line(self, tau, bw, k=7):
        t = self.t
        f = self.tf_frac(tau, k)
        return ("%s◆ TOKENS/ROUND%s %s  %s%.2f%s  %stau=%.2f/%d%s"
                % (t.title, t.rst(), self.bar(f, bw), t.heat(f), tau + 1, t.rst(), t.label, tau, k, t.rst()))



    @staticmethod
    def verdict(alpha, tau):
        if alpha >= 0.60:
            return "Excellent", "draft model is well aligned"
        if alpha >= 0.45:
            return "Good", "tau=%.2f free tokens per verify step" % tau
        if alpha >= 0.30:
            return "Fair", "~%.0f%% of drafted compute is thrown away" % ((1 - alpha) * 100)
        if alpha > 0:
            return "Poor", "the draft loses more than it wins - try a lower k"
        return "No speculation", "no draft model, or engine idle since boot"



    def waiting_row(self, cur):
        """What is actually outstanding, which /metrics cannot tell you.

        The engine only ever sees the one request the gateway has let through,
        so vllm:num_requests_waiting is a constant 0 and says nothing about the
        queue. When the gateway answers, count ITS lanes instead and name them:
        high = the conversation you are typing in plus its tool chain,
        low = heartbeat wakes and the background ingest."""
        t = self.t
        q = self.yzq
        if not q:
            return "%s%-13s%s %s%d%s" % (t.label, "waiting", t.rst(),
                                         t.heat(0.3) if cur.wait else t.frame,
                                         cur.wait, t.rst())
        hi, lo = int(q.get("high") or 0), int(q.get("low") or 0)
        n = hi + lo
        # heat: an empty queue is fine, one near YZQ_MAXQ (16) is not
        col = t.heat(1.0 - min(1.0, n / 8.0)) if n else t.frame
        # 31-08: it used to read "high 2 . low 0" and was read as "I have two
        # high sessions open". These are requests IN QUEUE, and two high ones
        # together are usually the same conversation in two calls (the turn +
        # an auxiliary call: compression, title, memory flush). The word
        # "queued" costs 7 columns and removes the ambiguity.
        par = int(q.get("parallel") or 1)
        slots = ("   %s%d/%d slot%s" % (t.frame, len(q.get("running") or []), par, t.rst())
                 if par > 1 else "")
        # 19-09: the gateway queue is NOT the whole story. The engine (vLLM, newer
        # kit) may hold a request the gateway has already admitted: "capacity" =
        # max_num_seqs or a full KV pool, "deferred" = prefill denied while
        # another decodes (GLM53_MIXED_PREFILL_CHUNK=skip). Measured 19-09: 2/2
        # slots at the gateway, engine running 1 / waiting 1, human 335 s to the
        # first token.
        eng = ""
        if cur.wait:
            why = []
            if getattr(cur, "wait_cap", 0):
                why.append("capacity %d" % cur.wait_cap)
            if getattr(cur, "wait_def", 0):
                why.append("deferred %d" % cur.wait_def)
            eng = "%s%-13s%s %s! %d held by the engine (%s)%s" % (
                t.label, "", t.rst(), t.heat(0.2), cur.wait, " . ".join(why) or "?", t.rst())
        row = "%s%-13s%s %s%d%s   %squeued: high %d . low %d%s%s" % (
            t.label, "waiting", t.rst(), col, n, t.rst(), t.frame, hi, lo, t.rst(), slots)
        return (row, eng) if eng else row



    def running_lane(self):
        """Which lane the engine is actually chewing on, or nothing if the
        gateway is not in front of it."""
        q = self.yzq or {}
        # 05-10: with parallel > 1 several requests run at once; "inflight" is only the oldest of them, usually
        # a beat, so a high turn running beside it showed as "low". Any high request running = high.
        lanes = [{"alta": "high", "bassa": "low"}.get(r.get("prio"), r.get("prio")) for r in (q.get("running") or [])]
        if lanes:
            return "high" if "high" in lanes else lanes[0]
        prio = (q.get("inflight") or {}).get("prio")
        return {"alta": "high", "bassa": "low"}.get(prio, prio)      # the gateway names its lanes in Italian



    def slot_panel(self, cur):
        out = {"kv": None, "prefix": None, "ctxtxt": "-", "stale": False}
        if not cur.nslots or not cur.nctx:
            return out
        out["kv"] = min(1.0, cur.ptok / cur.nctx)
        out["prefix"] = (cur.pcache / cur.ptok) if cur.ptok else None
        out["ctxtxt"] = "%s / %s" % (human(cur.ptok), human(cur.nctx))
        return out



    def model_arch(self):
        """Cached per-layer topology of the running engine (or None)."""
        now = time.time()
        if self.arch is None and now >= getattr(self, "arch_retry", 0.0):
            self.arch_path = model_config_path()
            if self.arch_path:
                self.arch = load_model_arch(self.arch_path)
            if self.arch is None:
                self.arch_retry = now + 30.0
        return self.arch



    def _engine_aux(self):
        """(vision_active, weights_offloaded, kv_usage) from one /metrics read."""
        try:
            txt = http_get("http://127.0.0.1:%d/metrics" % self.port,
                           timeout=2, tries=1)
            if not txt:
                return False, False, None
            mx = parse_metrics(txt)
            q = g(mx, "vllm:mm_cache_queries_total")
            now = time.time()
            vact = False
            if q is not None:
                if self.mm_prev is not None and q > self.mm_prev \
                        and now - self.mm_t < 120:
                    vact = True
                self.mm_prev, self.mm_t = q, now
            woff = bool(gl(mx, "vllm:engine_sleep_state",
                           'sleep_state="weights_offloaded"'))
            kv = g(mx, "vllm:kv_cache_usage_perc")
            return vact, woff, kv
        except Exception:
            return False, False, None



    def layer_panel(self, width, free):
        """Layer-activity visualizer: the model's REAL topology as coloured boxes.

        Read-only - it reads /metrics and the engine's config.json and never
        touches the engine, so it cannot perturb the LLM's output. Each layer is
        a coloured box; the colour encodes work: hot = actively computing,
        dim = idle, and the DSpark target layers are painted by their
        speculative-decoding acceptance. This is the panel that sits where the
        log is inside the spec view (toggled with L), not a separate screen.
        """
        t = self.t
        res = self.step()
        if res is None:
            return []
        cur, live = res
        arch = self.model_arch()
        if not arch:
            return [t.dim() + "  no model config found - layer topology unavailable"
                    + t.rst()]
        n = arch["n"]
        target = set(arch["target"])
        comp = arch["compress"]
        working = cur.busy and live["gen"] > 0
        alpha = cur.acc / cur.draft if cur.draft > 0 else 0.0
        # text colour that stays readable on a coloured box: dark on bright bg
        tcol = "" if t.plain else fg(0)
        cells = []
        for i in range(n):
            if i in target:
                heat = alpha if working else 0.0
                mark = t.accent + "★" + t.rst()
            else:
                heat = 0.85 if working else 0.0
                mark = ""
            kv = comp.get(i)
            kvmark = ("~" if kv and kv > 1 else "·" if kv == 0 else "")
            cells.append(self.ramp_heat_bg(heat) + tcol
                         + pad("L%02d%s%s" % (i, kvmark, mark), 5) + RST)
        cw = 5
        per_row = max(1, (width - 4) // cw)
        body = []
        for i in range(0, len(cells), per_row):
            body.append("".join(cells[i:i + per_row]))
        legend = ("  %s  %s working   %s  %s idle   %s★%s DSpark target   "
                  "%s~%s KV-compressed   %s·%s full-KV"
                  % (self.ramp_heat_bg(0.85), t.rst(),
                     self.ramp_heat_bg(0.0), t.rst(),
                     t.accent, t.rst(), t.label, t.rst(), t.label, t.rst()))
        body.append(legend)
        body.append(t.dim()
                     + "  per-expert routing heat needs enable_return_routed_experts"
                     + " (boot flag)" + t.rst())
        if free and len(body) > free:
            body = body[:free]
        return body

# ==================== suzu/render/frames.py
"""Full-screen frame composition: full, perf, pipes, layers, spec views."""



class FramesMixin:
    def frame(self, width, height=None):
        if self.mode == "layers":
            return self.frame_layers(width, height)
        if self.view == "pipes":
            return self.frame_pipes(width, height)
        if self.view == "perf":
            return self.frame_perf(width, height)
        return self.frame_spec(width, height)



    def frame_perf(self, width, height=None):
        """Performance zoom (p): the throughput story at full width - big
        digits, the acceptance line, one very wide history block and the GPU
        row. The repo's `p` view, ported to what this box exposes."""
        t = self.t
        if height is None:
            height = shutil.get_terminal_size((100, 40)).lines
        width = max(20, width)
        res = self.step()
        if res is None:
            return [clip("%s  no engine on :%d%s" % (t.heat(0.0), self.port,
                                                     t.rst()), width)]
        cur, live = res
        gen = live["gen"]
        L = []
        A = L.append
        A(clip("%s◆%s %sPERFORMANCE ZOOM%s   %s( %s )%s   %sp = back%s"
               % (t.title, t.rst(), t.bold(), t.rst(), t.head,
                  self.info.get("engine") or cur.fam, t.rst(),
                  t.accent, t.rst()), width))
        A("")
        if not height or height >= 22:
            if t.plain:
                gcol = ""
            else:
                gcol = self.ramp_flow(min(1.0, gen / max(self.pk_gen, 0.5))) \
                    if gen else t.frame
            for r in self.big_digits(gen, max(18, width // 2 - 4), gcol):
                A(clip(r, width))
            stat = (self.pulse_head(int(time.time() * 2)) + " DECODE"
                    if cur.busy and gen > 0 else "  DECODE")
            bar = self.gauge(min(1.0, gen / max(self.pk_gen, 0.5))
                             if self.pk_gen else 0.0, 14)
            A(clip("%s%s%s %s%*.1f t/s%s  %s  %speak %*.1f%s  %s%s tok total%s"
                   % (t.label, stat, t.rst(), t.value + t.bold(), 6, gen,
                      t.rst(), bar, t.frame, 6, self.pk_gen, t.rst(),
                      t.label, human(cur.gen), t.rst()), width))
            A("")
        alpha = (self.w["alpha"] if self.w else
                 (cur.acc / cur.draft if cur.draft > 0 else 0.0))
        tau = (self.w["tau"] if self.w else
               (cur.acc / cur.drafts if cur.drafts > 0 else 0.0))
        if cur.tf_nodraft:
            A(clip(self.tf_line(tau, min(46, width - 40)), width))
        else:
            A(clip("%s◆ ACCEPTANCE%s %s  %s%.1f%%%s  %stau=%.2f%s"
                   % (t.title, t.rst(), self.bar(alpha, min(46, width - 40)),
                      t.heat(alpha), alpha * 100, t.rst(), t.label, tau, t.rst()),
                   width))
        A("")
        sw = max(10, width - 34)
        for row in self.box("history  (full width)", [
                self.hrow("gen t/s", self.h_gen, sw, "%.1f", t.heat(0.9),
                          "%d tick" % len(self.h_gen), fit=width - 2),
                self.hrow("prompt t/s", self.h_pf, sw, "%.0f", t.heat(0.7), "", fit=width - 2),
                self.hrow("accept %", self.h_acc, sw, "%.0f", t.heat(alpha),
                          "%d req" % len(self.h_acc), fit=width - 2),
                self.hrow("tau", self.h_tau, sw, "%.2f", t.head, "", fit=width - 2),
        ], width, t.head):
            A(clip(row, width))
        cl = self.cluster_panel(width)
        if cl:
            A("")
            for r in self.box(self.cluster_title(), cl, width, t.accent):
                A(clip(r, width))
            ln = self.lanes_panel(cur, width)
            if ln:
                for r in self.box("gateway lanes", ln, width, t.head):
                    A(clip(r, width))
        L = [clip(ln, width) for ln in L]
        if height and len(L) > height:
            L = L[:height]
        self.last_frame = L
        return L



    def frame_pipes(self, width, height=None):
        """Memory-pipeline zoom (b): the hops a token takes on this box, as
        vertical VU meters. The repo's six strips assume split RAM/VRAM; on
        GB10 unified memory they collapse, and what THIS box uniquely gates a
        token with is the gateway lane machinery and the rank-1 link - so those
        get the strips the repo spends on PCIe."""
        t = self.t
        if height is None:
            height = shutil.get_terminal_size((100, 40)).lines
        width = max(40, width)
        res = self.step()
        if res is None:
            return [clip("%s  no engine on :%d%s" % (t.heat(0.0), self.port,
                                                     t.rst()), width)]
        cur, live = res
        gen = live["gen"]
        vh = max(6, min(12, height - 14))
        sl = self.slot_panel(cur)
        kv = sl["kv"] if sl["kv"] is not None else 0.0
        gpu_u = ((self.gpu[0].get("util") or 0.0) / 100.0) if self.gpu else 0.0
        mem_f = (self.memory or {}).get("used_frac")
        dec_f = min(1.0, gen / max(self.pk_gen, 0.5)) if self.pk_gen else 0.0
        pf_f = min(1.0, live["pf"] / max(self.pk_pf, 0.5)) if self.pk_pf else 0.0
        r1 = self.rank1
        link_f = 1.0 if (r1 and r1.get("up")) else (0.0 if r1 else None)
        stages = [
            ("GPU", gpu_u, gpu_u, "%5.0f%%" % (gpu_u * 100)),
            ("MEM", mem_f if mem_f is not None else 0.0,
             mem_f if mem_f is not None else 0.0,
             ("%5.0f%%" % (mem_f * 100)) if mem_f is not None else "  n/a"),
            ("CTX", kv, kv, "%5.0f%%" % (kv * 100)),
            ("DECODE", dec_f, dec_f, "%5.1f" % gen),
            ("PREFILL", pf_f, pf_f, "%5.0f" % live["pf"]),
        ]
        if link_f is not None:
            stages.append(("RANK1", link_f, link_f,
                           "link" if link_f else "down"))
        meters = [self.vmeter(f, h, vh, 5) for _, f, h, _ in stages]
        L = []
        A = L.append
        A(clip("%s◆%s %sTOKEN PIPELINE%s   %sb = back%s"
               % (t.title, t.rst(), t.bold(), t.rst(), t.accent, t.rst()),
               width))
        A("")
        colw = 8
        row = []
        for i in range(vh):
            row = []
            for m in meters:
                row.append(pad(m[i], colw - 2))
            A(clip("".join(row), width))
        lab = []
        for name, _, _, val in stages:
            lab.append(pad("%s%-6s%s%s" % (t.label, name, t.rst(),
                                           t.value + val + t.rst()), colw))
        A(clip("".join(lab), width))
        A("")
        verdict = self.pipe_verdict(cur, gpu_u, kv, r1)
        A(clip("%s* %s%s" % (t.heat(0.8), verdict, t.rst()), width))
        ln = self.lanes_panel(cur, width)
        if ln:
            A("")
            for r in self.box("gateway lanes", ln, width, t.accent):
                A(clip(r, width))
        L = [clip(x, width) for x in L]
        if height and len(L) > height:
            L = L[:height]
        self.last_frame = L
        return L



    def pipe_verdict(self, cur, gpu_u, kv, r1):
        """The repo's bottleneck rule chain, adapted to this box: with the
        weights resident in unified memory there is no PCIe hop, so the
        failure modes that remain are the partner link, the KV pool, compute
        and latency."""
        if r1 is not None and not r1.get("up") \
                and self.info.get("nnodes") in ("2", 2):
            return "rank 1 unreachable - tensor-parallel decode stalls"
        if cur.busy and gpu_u < 0.2:
            return ("GPU idle while decoding - latency-bound "
                    "(small active batch, bandwidth-bound decode)")
        if gpu_u > 0.85:
            return "GPU compute-bound at %.0f%% utilisation" % (gpu_u * 100)
        if kv > 0.9:
            return "KV pool nearly full - prefill will be deferred"
        if cur.busy:
            return "no hop saturated - the gap is per-token latency"
        return "engine idle - nothing to bound"



    def frame_layers(self, width, height=None):
        """Layer-activity panel: the model's REAL topology as a heat grid.

        Read-only - it reads /metrics and the engine's config.json and never
        touches the engine, so it cannot perturb the LLM's output. The grid is
        built from the actual model (num layers, experts, DSpark target layers,
        KV compression ratios). Live "work" colour comes from the same Sample
        the spec panel uses: global generation activity plus the speculative-
        decode acceptance on the DSpark target layers.
        """
        t = self.t
        if height is None:
            height = shutil.get_terminal_size((100, 40)).lines
        width = max(20, width)
        res = self.step()
        if res is None:
            return [clip("%s%s  x no engine on :%d%s" % (t.heat(0.0), t.bold(),
                                                          self.port, t.rst()), width),
                    clip("%s  (g = back to spec, q = quit)%s" % (t.dim(), t.rst()),
                         width)]
        cur, live = res
        fdt = self.fade_dt()
        self.fade_step("gen", live["gen"] > 0, fdt)
        arch = self.model_arch()
        vact, woff, kv_usage = self._engine_aux()

        L = []
        A = L.append

        # ---- header
        title = "MODEL LAYER ACTIVITY"
        eng = self.info.get("engine") or cur.fam
        head = ("%s◆%s %s%s%s%s %s────%s %s( %s )%s   %s◆ layer grid ◆%s"
                % (t.title, t.rst(), t.bold(), t.title, title, t.rst(), t.frame,
                   t.rst(), t.head, eng, t.rst(), t.accent, t.rst()))
        right = ("%s%s%s  %s│%s  %s%s tok%s  %s│%s  %sg=spec%s"
                 % (t.value, fmt_hms(self.info.get("up")), t.rst(), t.frame,
                    t.rst(), t.label, human(cur.gen), t.rst(), t.frame, t.rst(),
                    t.head, t.rst()))
        if vis_len(head) + vis_len(right) + 2 > width:
            head = "%s◆%s %s%s%s%s" % (t.title, t.rst(), t.bold(), t.title,
                                            title, t.rst())
        A(pad(head, max(0, width - vis_len(right))) + right)

        # ---- model strip: what is loaded, straight from the engine cmdline
        strip = self.model_strip(width)
        if strip:
            A(strip[0])

        # ---- engine activity + speculative summary
        if cur.busy:
            act = "%s%s* WORKING%s" % (t.bold(), t.heat(0.9), t.rst())
            if cur.run:
                act += " %s%d req%s" % (t.label, int(cur.run), t.rst())
        else:
            idle = None if self.act_ts is None else max(0.0, time.time() - self.act_ts)
            act = "%s. idle%s %s%s%s" % (t.dim(), t.rst(), t.value,
                                         fmt_hms(idle) if idle is not None else "?",
                                         t.rst())
        A(act)
        alpha = cur.acc / cur.draft if cur.draft > 0 else 0.0
        tau = cur.acc / cur.drafts if cur.drafts > 0 else 0.0
        tgt = ",".join(str(x) for x in (arch["target"] if arch else []))
        spec = ("  spec: draft→L%s  accept %s%.0f%%%s  tau %s%.2f%s"
                % (tgt, t.heat(alpha), alpha * 100, t.rst(), t.label, tau, t.rst()))
        if cur.tf_nodraft:
            spec = "  spec: %s%.2f token/round%s (TensorFold: no drafted-token counter)" % (t.label, tau + 1, t.rst())
        A(spec)
        A("")

        if not arch:
            A(clip("%s  no model config found - layer topology unavailable%s"
                   % (t.dim(), t.rst()), width))
            A(clip("%s  (the layer grid needs the engine's config.json)%s"
                   % (t.dim(), t.rst()), width))
            L = [clip(ln, width) for ln in L]
            if height and len(L) > height:
                L = L[:height]
            self.last_frame = L
            return L

        n = arch["n"]
        target = set(arch["target"])
        comp = arch["compress"]
        working = cur.busy and live["gen"] > 0
        acc = cur.acc / cur.draft if cur.draft > 0 else 0.0

        # ---- transformer layer grid
        cw = 6
        per_row = max(1, (width - 2) // cw)
        cells = []
        for i in range(n):
            if i in target:
                col = t.heat(acc) if working else t.dim()
                mark = t.accent + "★" + t.rst()
            else:
                col = self.ramp_heat(max(0.35, self.fade.get("gen", 0.0))) \
                    if working else t.dim()
                mark = ""
            kv = comp.get(i)
            kvmark = ("·" if kv == 0 else "~" if kv and kv > 1 else "")
            cells.append("%sL%02d%s%s%s" % (col, i, t.rst(), kvmark, mark))
        body = []
        for i in range(0, len(cells), per_row):
            row = "".join(pad(c, cw) for c in cells[i:i + per_row])
            body.append(clip(row, width - 2))
        for r in self.box("transformer layers  (%d · MoE · %d experts, %d active/tok)"
                          % (n, arch["experts"], arch["topk"]), body, width, t.accent):
            A(clip(r, width))
        A("")

        # ---- auxiliary modules: vision / nextn / shared
        aux_body = []
        if arch["vision"]:
            vcol = t.heat(0.9) if vact else t.dim()
            aux_body.append("%s█%s vision encoder  %d layers  %s"
                            % (vcol, t.rst(), arch["vision"],
                               "ACTIVE" if vact else "idle (no image)"))
        if arch["nextn"]:
            aux_body.append("%s█%s nextn / MTP  %d layer"
                            % (t.heat(0.9) if working else t.dim(), t.rst(),
                               arch["nextn"]))
        if arch["shared"]:
            aux_body.append("%s█%s shared experts  %d per layer"
                            % (t.heat(0.6) if working else t.dim(), t.rst(),
                               arch["shared"]))
        for r in self.box("auxiliary modules", aux_body, width, t.head):
            A(clip(r, width))
        A("")

        # ---- engine state line
        sl = self.slot_panel(cur)
        kv = sl["kv"] if sl["kv"] is not None else kv_usage
        ctxtxt = sl["ctxtxt"]
        if ctxtxt == "-" and cur.ctxmax:
            ctxtxt = "up to %s" % human(cur.ctxmax)
        A(clip("%s  kv cache %s%s   weights %s%s%s   ctx %s%s%s"
               % (t.label, self.pct(kv, 8), t.rst(), t.label,
                  "OFFLOADED" if woff else "loaded", t.rst(), t.value,
                  ctxtxt, t.rst()), width))
        A("")

        # ---- legend
        legend = ("  %s█%s working   %s█%s idle   %s★%s DSpark target   "
                  "%s~%s KV-compressed   %s·%s full-KV"
                  % (t.heat(0.9), t.rst(), t.dim(), t.rst(), t.accent, t.rst(),
                     t.label, t.rst(), t.label, t.rst()))
        A(legend)
        A(clip("%s  per-expert routing heat needs enable_return_routed_experts "
               "(boot flag) - not collected here%s" % (t.dim(), t.rst()), width))

        L = [clip(ln, width) for ln in L]
        if height and len(L) > height:
            L = L[:height]
        self.last_frame = L
        return L



    def frame_spec(self, width, height=None):
        t = self.t
        if height is None:
            height = shutil.get_terminal_size((100, 40)).lines
        width = max(20, width)
        res = self.step()
        if res is None:
            self.miss += 1
            age = self.miss * self.interval
            if self.last_frame and age < 30:
                return [clip("%s/metrics silent for %.0fs - panel frozen%s"
                             % (t.heat(0.1), age, t.rst()), width)] + self.last_frame[1:]
            return [clip("%s%s  x no engine on :%d - /metrics and /props silent%s"
                         % (t.heat(0.0), t.bold(), self.port, t.rst()), width),
                    clip("%s  (check the engine is running on this port, or start it)%s"
                         % (t.dim(), t.rst()), width)]
        self.miss = 0
        cur, live = res

        # llm-visuals smoothing: one fade tick per frame + peak-hold on gen
        fdt = self.fade_dt()
        self.fade_step("gen", live["gen"] > 0, fdt)
        self.fade_step("pf", live["pf"] > 0, fdt)
        gen = live["gen"]
        if gen >= self.pk_gen:
            self.pk_gen, self.pk_at = gen, time.time()
        elif time.time() - self.pk_at > 2.0:
            self.pk_gen = max(gen, self.pk_gen - 0.02 * self.pk_gen)

        s_alpha = cur.acc / cur.draft if cur.draft > 0 else 0.0
        s_tau = cur.acc / cur.drafts if cur.drafts > 0 else 0.0
        s_dwin = cur.draft / cur.drafts if cur.drafts > 0 else 0.0
        w = self.w
        alpha = w["alpha"] if w else s_alpha
        tau = w["tau"] if w else s_tau
        dwin = w["dwin"] if w else s_dwin
        posn = w["pos"] if w else cur.pos
        posd = w["drafts"] if w else cur.drafts
        if not any(posn):
            posn, posd = cur.pos, cur.drafts
        src = ("window, %s" % ago(time.time() - self.w_at)) if w else "session"

        k = self.info.get("k")
        try:
            k = int(k)
        except (TypeError, ValueError):
            k = int(round(s_dwin)) or len(cur.pos)

        gen = live["gen"]
        steps = gen / (tau + 1.0) if tau > 0 else 0.0
        acc_ts, draft_ts = steps * tau, steps * dwin

        sl = self.slot_panel(cur)
        if cur.busy and sl["prefix"]:
            self.last_prefix = sl["prefix"]
        elif not cur.busy and self.last_prefix is not None:
            # An idle slot resets n_prompt_tokens_cache to 0 and the panel showed
            # a red 0% on a perfectly healthy engine. Keep the last true value.
            sl["prefix"], sl["stale"] = self.last_prefix, True

        L = []
        A = L.append

        # ---- header (degrades: full title -> short -> just the mark)
        title = ("SPECULATIVE DECODING MONITOR" if width >= 104
                 else "SPEC DECODING" if width >= 80 else "SPEC")
        eng = self.info.get("engine") or cur.fam
        head = ("%s◆%s %s%s%s%s %s────%s %s( %s )%s   %s◆ suzu ◆%s"
                % (t.title, t.rst(), t.bold(), t.title, title, t.rst(), t.frame,
                   t.rst(), t.head, eng, t.rst(), t.accent, t.rst()))
        if width >= 96:
            right = ("%s%s%s  %s│%s  %s%s tok%s  %s│%s  %sk=%d%s  %s%s%s"
                     % (t.value, fmt_hms(self.info.get("up")), t.rst(), t.frame,
                        t.rst(), t.label, human(cur.gen), t.rst(), t.frame, t.rst(),
                        t.heat(0.8), k, t.rst(), t.frame, t.name, t.rst())
                     + ("  " + t.legend() if width >= 112 else ""))
        else:
            right = "%s%s%s %sk=%d%s" % (t.value, fmt_hms(self.info.get("up")),
                                         t.rst(), t.heat(0.8), k, t.rst())
        if vis_len(head) + vis_len(right) + 2 > width:
            head = "%s◆%s %s%s%s%s" % (t.title, t.rst(), t.bold(), t.title,
                                            title, t.rst())
        A(pad(head, max(0, width - vis_len(right))) + right)

        # ---- model strip: what is loaded, straight from the engine cmdline
        strip = self.model_strip(width)
        if strip:
            A(strip[0])

        draft = self.info.get("draft") or "(no draft model)"
        if len(draft) > max(8, width - 24):
            draft = "..." + draft[-(max(8, width - 27)):]
        dline = ("  %s>%s %s%s%s  %s<-%s %s%s%s"
                 % (t.head, t.rst(), t.head, draft, t.rst(), t.frame, t.rst(),
                    t.label, self.info.get("spec") or "-", t.rst()))

        # Engine activity, right of the draft line. WORKING while tokens are
        # actually moving; otherwise how long the engine has been quiet. This is
        # the field that answers "is it chewing on something or has it been
        # sitting there since dawn" without reading the server log by hand.
        if cur.busy:
            act = "%s%s* WORKING%s" % (t.bold(), t.heat(0.9), t.rst())
            if cur.run:
                act += " %s%d req%s" % (t.label, int(cur.run), t.rst())
        else:
            idle = None if self.act_ts is None else max(0.0, time.time() - self.act_ts)
            act = "%s. idle%s %s%s%s" % (t.dim(), t.rst(), t.value,
                                         fmt_hms(idle) if idle is not None else "?",
                                         t.rst())
        if vis_len(dline) + vis_len(act) + 3 <= width:
            dline = pad(dline, max(0, width - vis_len(act) - 1)) + act
        A(dline)
        A("")

        # ---- big decode readout (llm-visuals: tall numerals on the FLOW ramp).
        # btop-style: full digits with room, one stat line on a short terminal.
        stat = (self.pulse_head(int(time.time() * 2)) + " DECODE"
                if cur.busy and gen > 0 else "  DECODE")
        bar = self.gauge(min(1.0, gen / max(self.pk_gen, 0.5))
                         if self.pk_gen else 0.0, 14)
        if width >= 66:
            dstat_line = clip("%s%s%s %s%*.1f t/s%s  %s  %speak %*.1f%s  %s%s tok total%s"
                              % (t.label, stat, t.rst(),
                                 t.value + t.bold(), 6, gen, t.rst(),
                                 bar, t.frame, 6, self.pk_gen, t.rst(),
                                 t.label, human(cur.gen), t.rst()), width)
        else:
            # narrow: the rate and its gauge are the point, peak and total are not
            dstat_line = clip("%s%s%s %s%*.1f t/s%s  %s"
                              % (t.label, stat, t.rst(),
                                 t.value + t.bold(), 6, gen, t.rst(), bar), width)
        if not height or height >= 26:
            if t.plain:
                gcol = t.value if gen else t.frame
            else:
                gcol = self.ramp_flow(min(1.0, gen / max(self.pk_gen, 0.5))) \
                    if gen else t.frame
            for r in self.big_digits(gen, max(18, width // 2 - 4),
                                     gcol if not t.plain else ""):
                A(clip(r, width))
            A(dstat_line)
            A("")
        elif height >= 20:
            A(dstat_line)
            A("")

        # ---- acceptance
        bw = max(10, min(46, width - 50))
        tfam = cur.tf_nodraft
        if tfam:
            line = self.tf_line(tau, bw, k)
            tail = "  %s(%s)%s" % (t.frame, src, t.rst())
            A(line + tail if vis_len(line + tail) <= width else line)
            # The explanation is a sentence, so it must not be cut mid-word at a
            # narrow terminal: pick the longest form that fits instead.
            forms = [
                "* %.2f tokens a round - this TensorFold does not count drafted tokens: no acceptance %%, tau from the decode rounds",
                "* %.2f tokens a round - TensorFold counts no drafted tokens: no acceptance %%, tau from decode rounds",
                "* %.2f tokens a round - no acceptance %% on TensorFold",
                "* %.2f tok/round - no acceptance %%",
            ]
            txt = next((f % (tau + 1) for f in forms if len(f % (tau + 1)) + 4 <= width),
                       forms[-1] % (tau + 1))
            A("%s%s%s" % (t.heat(self.tf_frac(tau, k)), txt, t.rst()))
        else:
            A(clip("%s◆ ACCEPTANCE%s %s  %s%.1f%%%s  %stau=%.2f/%d%s  %s(%s)%s"
                   % (t.title, t.rst(), self.bar(alpha, bw), t.heat(alpha), alpha * 100,
                      t.rst(), t.label, tau, k, t.rst(), t.frame, src, t.rst()), width))
            verdict, why = self.verdict(alpha, tau)
            A(clip("%s* %s%s %s- %s%s" % (t.heat(alpha), verdict, t.rst(), t.dim(),
                                          why, t.rst()), width))
        A("")

        # ---- number grid: 3 columns wide, 1 column narrow. On a very short
        # terminal the engine box matters more than the grid: skip it.
        if not height or height >= 20:
            na = "n/a"
            cells = [("tau acc length", "%.2f" % tau, t.heat(self.tf_frac(tau, k) if tfam else alpha)),
                     ("draft window", na if tfam else "%.1f" % dwin, t.frame if tfam else t.value),
                     ("spec tokens", "%d" % k, t.value),
                     ("accepted t/s", "%.1f" % acc_ts, t.heat(0.9)),
                     ("drafted t/s", na if tfam else "%.1f" % draft_ts, t.frame if tfam else t.label),
                     ("waste ratio", na if tfam else "%.1f%%" % ((1 - alpha) * 100), t.frame if tfam else t.heat(alpha)),
                     ("gen t/s", "%.1f" % gen, t.heat(0.9) if gen else t.frame),
                     ("prompt t/s", "%.0f" % live["pf"],
                      t.heat(0.7) if live["pf"] else t.frame),
                     ("verify step/s", "%.1f" % steps, t.value if steps else t.frame)]
            ncol = 3 if width >= self.NARROW else 1
            cw = (width - 2 * (ncol - 1)) // ncol
            for r in range(0, len(cells), ncol):
                line = ""
                for i in range(ncol):
                    if r + i >= len(cells):
                        break
                    lab, val, col = cells[r + i]
                    line += pad("%s%-14s%s %s%s%s" % (t.label, lab, t.rst(),
                                                      col + t.bold(), val, t.rst()), cw)
                    if i < ncol - 1:
                        line += t.frame + "│ " + t.rst()
                A(clip(line, width))
            A("")

        # ---- engine/session, history, rolling averages
        eng_body = [
            "%s%-13s%s %s" % (t.label, "kv cache", t.rst(), self.pct(sl["kv"], 10)),
            "%s%-13s%s %s%s" % (t.label, "prefix cache", t.rst(),
                                self.pct(sl["prefix"], 10, good_high=True),
                                (t.frame + " last" + t.rst()) if sl["stale"] else ""),
            "%s%-13s%s %s" % (t.label, "ttft last", t.rst(),
                              ("%s%.1f s%s  %s( %d seen )%s"
                               % (t.value, self.h_ttft[-1], t.rst(),
                                  t.frame, len(self.h_ttft), t.rst())
                               if self.h_ttft else
                               t.frame + "-" + t.rst())),
            "%s%-13s%s %s%d%s   %s%s%s%s" % (t.label, "running", t.rst(),
                                             t.heat(0.9) if cur.run else t.frame,
                                             cur.run, t.rst(), t.frame,
                                             "busy" if cur.busy else "idle",
                                             (" . " + self.running_lane())
                                             if self.running_lane() else "", t.rst()),
            *([self.fill_row(cur)] if cur.fam == "tensorfold" else []),
            *((lambda w: list(w) if isinstance(w, tuple) else [w])(self.waiting_row(cur))),
            "%s%-13s%s %s%s%s" % (t.label, "slot ctx", t.rst(), t.value,
                                  sl["ctxtxt"], t.rst()),
            "%s%-13s%s %s%s%s" % (t.label, "ctx max seen", t.rst(), t.label,
                                  human(cur.ctxmax), t.rst()),
            *self.totals_rows(cur),
            "",
            "%s%-13s%s %s%s%s" % (t.label, "accepted", t.rst(), t.value,
                                  human(cur.acc), t.rst()),
            "%s%-13s%s %s%s%s" % (t.label, "drafted", t.rst(), t.value,
                                  "n/a" if cur.tf_nodraft else human(cur.draft), t.rst()),
            ("%s%-13s%s %s%.2f token/round%s  %stau %.2f%s" % (t.label, "session", t.rst(), t.value, s_tau + 1,
                                                             t.rst(), t.frame, s_tau, t.rst())
             if cur.tf_nodraft else
             "%s%-13s%s %s%.1f%%%s  %stau %.2f%s" % (t.label, "session a", t.rst(),
                                                     t.heat(s_alpha), s_alpha * 100,
                                                     t.rst(), t.frame, s_tau, t.rst())),
        ]
        # ---- content-aware grid (19-10: "right sized ... doesn't resize at all").
        # Each column asks for the width its own text needs; three columns only
        # when they can hold their content, otherwise two, otherwise stacked.
        # Slack is shared instead of the fixed 38/32 fractions that squeezed the
        # left box to 47 columns for an 89-column line and clipped it mid-word.
        def _need(body, floor=24):
            return max(floor, max([vis_len(x) for x in body] + [0]) + 4)

        avg_body = [
            "%s%-15s%s %s%.1f%s   %s%-11s%s %s%s%s"
            % (t.label, "avg gen t/s", t.rst(), t.value, self.wmean(self.q_gen),
               t.rst(), t.label, "avg a", t.rst(), t.value,
               "n/a" if cur.tf_nodraft else "%.1f%%" % (self.wmean(self.q_acc) * 100), t.rst()),
            "%s%-15s%s %s%.1f%s   %s%-11s%s %s%.2f%s"
            % (t.label, "avg prompt t/s", t.rst(), t.value, self.wmean(self.q_pf),
               t.rst(), t.label, "avg tau", t.rst(), t.value,
               self.wmean(self.q_tau), t.rst()),
        ]
        # The cluster panel narrows itself below 110 columns, so measure the
        # narrow shape it will actually render in a side column.
        cl_probe = self.cluster_panel(min(78, max(34, width - 4)))
        need_l = _need(eng_body, 34)
        if cl_probe:
            need_l = max(need_l, _need(cl_probe, 34))
        need_m, need_r = 34, 44
        if need_l + need_m + need_r + 4 <= width:
            ncols = 3
        elif need_l + need_r + 2 <= width:
            ncols = 2
        else:
            ncols = 1
        # A short terminal cannot afford the middle chart column (the heatmap is
        # ~20 rows in narrow mode): drop to two columns before the widths are
        # computed, so the right column can hold or shed the chart by its budget.
        if ncols == 3 and height and (height - len(L)) < 22:
            ncols = 2
        if ncols == 3:
            rest = width - 4 - need_l
            extra = max(0, rest - need_m - need_r)
            tot = need_l + need_m + need_r
            lw = need_l + extra * need_l // tot
            mw = need_m + extra * need_m // tot
            rw = width - 4 - lw - mw
        elif ncols == 2:
            lw = min(need_l, width - 2 - need_r)
            rw = width - 2 - lw
            mw = rw if rw >= 44 else width
        else:
            lw = mw = rw = width
        wide = ncols >= 2
        sw = max(4, rw - 36)
        hist_body = [
            self.hrow("gen t/s", self.h_gen, sw, "%.1f", t.heat(0.9),
                      "%d tick" % len(self.h_gen), fit=rw - 2),
            self.hrow("prompt t/s", self.h_pf, sw, "%.0f", t.heat(0.7), "", fit=rw - 2),
            self.hrow("accept %", self.h_acc, sw, "%.0f", t.heat(alpha),
                      "n/a on tensorfold" if cur.tf_nodraft else "%d req" % len(self.h_acc),
                      fit=rw - 2),
            self.hrow("tau", self.h_tau, sw, "%.2f", t.head, "", fit=rw - 2),
        ]
        cl = self.cluster_panel(lw)
        # Height guard (left column): GB10 must ALWAYS sit under engine & session,
        # even on a short terminal where the full engine & session box would push GB10
        # past the L[:height] trim. If eng_body + GB10 won't fit the available rows,
        # drop engine & session rows from the bottom (least-important first: session,
        # drafted, accepted, the trailing blank, ctx max seen, slot ctx, then totals)
        # until it fits. GB10 itself is never trimmed.
        if cl and height:
            avail = height - len(L)
            need = (len(eng_body) + 2) + 1 + (len(cl) + 2)
            while need > avail and len(eng_body) > 5:
                eng_body.pop()
                need -= 1
            if need > avail:
                cl = None       # no room even for the minimum: drop the box whole
        if height and len(eng_body) + 2 > height - len(L):
            # last resort on a tiny terminal: keep kv/prefix/ttft, the three
            # rows that say whether the engine is alive at all
            while len(eng_body) > 3 and len(eng_body) + 2 > height - len(L):
                eng_body.pop()
        left_col = self.box("engine & session", eng_body, lw)
        if cl:
            left_col.append("")
            left_col += self.box(self.cluster_title(), cl, lw, t.accent)
        # Middle column = per-day chart only. History moved down to the right column
        # to stack with rolling averages and per-position. In two-column mode the
        # heatmap joins the right column (it needs ~44 columns, not a whole row).
        heat_body = self.heatmap_rows(mw)
        heat_title = ("generated tokens per day . all models"
                      if vis_len(" generated tokens per day . all models ") <= mw - 4
                      else "generated tokens per day"
                      if vis_len(" generated tokens per day ") <= mw - 4
                      else "tokens/day")
        heat_box = (self.box(heat_title, heat_body, mw)
                    if heat_body else [])
        mid_col = heat_box if ncols == 3 else []
        # per-position acceptance: a small box that wraps one row of cells per the
        # column width it is rendered in. In wide mode it lives in the RIGHT column,
        # under rolling averages; in narrow mode it stays full-width at the bottom.
        def _pos_box(w):
            barw = 8 if w >= 100 else 5 if w >= 60 else 3
            cellw = barw + 10
            per = max(1, (w - 4) // cellw)
            chunks = []
            for i in range(len(posn)):
                v = (posn[i] / posd) if posd else 0.0
                chunks.append("%sp%d%s %s %s%3.0f%%%s"
                              % (t.head, i, t.rst(), self.bar(v, barw),
                                 t.heat(v), v * 100, t.rst()))
            body = []
            for i in range(0, len(chunks), per):
                body.append("".join(pad(c, cellw) for c in chunks[i:i + per]))
            title = next((x for x in ("per-position acceptance  (%s)" % src,
                                      "per-position  (%s)" % src,
                                      "per-position")
                          if vis_len(" " + x + " ") <= w - 4), "per-pos")
            return self.box(title, body, w, t.accent)
        # Side column in visual order: history on top, rolling averages,
        # per-position, then the day chart (when the chart is not the middle
        # column). A btop-style row budget then drops the lowest-priority box
        # whole - chart first, then per-position, then history - instead of
        # L[:height] cutting one mid-box.
        hist_box = self.box("history", hist_body, rw) if hist_body else []
        avg_box = self.box("rolling averages", avg_body, rw)
        pos_box = _pos_box(rw) if posn else []
        heat_in_col = bool(heat_box) and ncols == 2 and mw != width
        heat_below = bool(heat_box) and ncols == 2 and mw == width
        order = []
        if hist_box:
            order.append(("hist", hist_box + [""]))
        order.append(("avg", avg_box))
        if pos_box:
            order.append(("pos", [""] + pos_box))
        if heat_in_col:
            order.append(("heat", [""] + heat_box))
        if ncols == 3 and height and len(mid_col) > height - len(L):
            mid_col = []
        if height:
            avail = height - len(L)
            drops = ("heat", "pos", "hist", "avg") if ncols == 1 else ("heat", "pos", "hist")
            for drop in drops:
                if ncols >= 2:
                    block = max(len(left_col),
                                sum(len(b) for _n, b in order),
                                len(mid_col) if ncols == 3 else 0)
                else:
                    block = len(left_col) + sum(len(b) for _n, b in order)
                if block <= avail:
                    break
                order = [(n, b) for n, b in order if n != drop]
        right_col = [ln for _n, block in order for ln in block]
        if ncols == 3:
            for i in range(max(len(left_col), len(mid_col), len(right_col))):
                a = left_col[i] if i < len(left_col) else ""
                b = mid_col[i] if i < len(mid_col) else ""
                c = right_col[i] if i < len(right_col) else ""
                A(clip(pad(a, lw) + "  " + pad(b, mw) + "  " + c, width))
        elif ncols == 2:
            for i in range(max(len(left_col), len(right_col))):
                a = left_col[i] if i < len(left_col) else ""
                c = right_col[i] if i < len(right_col) else ""
                A(clip(pad(a, lw) + "  " + c, width))
            if heat_below and (not height or (height - len(L))
                               >= max(len(left_col), len(right_col)) + len(heat_box) + 1):
                for r in heat_box:
                    A(clip(r, width))
        else:
            for r in left_col + right_col:
                A(clip(r, width))
        A("")

        # ---- gateway lanes. (The GB10 / cluster node box now lives directly under the
        # engine & session panel; see that section. It shows this box's GPU + RAM and the
        # rank-1 worker + RoCE link on a TP pair, and is the highest-priority box.)
        free_rows = (height - len(L)) if height else 99
        ln = self.lanes_panel(cur, width) if free_rows >= 3 else None
        if ln:
            for r in self.box("gateway lanes", ln, width, t.head):
                A(clip(r, width))

        # ---- expert routing heat (render.rs render_experts port)
        archx = self.model_arch()
        if archx and (not height or height - len(L) >= 5):
            self.fade_experts(archx, cur.busy and gen > 0, fdt,
                              cur.pp + cur.dec)
            # every layer its own half row when there is room (46 on GLM-5.3 with MTP = 23 rows), fewer when not
            natural = (len(self.fade_expert) or archx["n"] + 1) // 2
            avail = (height - len(L) - 3) if height else natural
            rows = self.expert_grid(width, rows_avail=max(2, min(natural, avail)))
            if rows:
                xs = getattr(self, "_xs", None)
                if xs and xs.get("delta") is not None:
                    title = ("expert routing heat  (REAL - TensorFold /experts: decode picks since the last read, "
                             "colour = share of the layer's busiest expert, fading)")
                elif xs and xs.get("on"):
                    title = "expert routing heat  (REAL - TensorFold /experts: waiting for a second read)"
                elif xs and xs.get("why"):
                    title = "expert routing heat  (simulated - /experts is off on this engine: %s)" % xs["why"]
                elif getattr(self, "tf", False):
                    title = ("expert routing heat  (simulated - this TensorFold image has no /experts: "
                             "patches/0990, glm53mia image y9, goes live with the next restart)")
                else:
                    title = ("expert routing heat  (simulated - vllm does not "
                             "export router choices; colour = time since routed)")
                # A box title is never cut mid-word: drop the parenthetical first.
                if vis_len(" " + title + " ") > width - 4:
                    short = title.split("  (")[0]
                    title = short if vis_len(" " + short + " ") <= width - 4 else "expert heat"
                for r in self.box(title, rows, width, t.accent):
                    A(clip(r, width))

        # ---- layer visualizer or log fills whatever rows are left
        if self.show_layers:
            lp = self.layer_panel(width, height - len(L) - 4)
            if lp:
                for r in self.box("layer activity  (L = log, g = full)",
                                  lp, width, t.accent):
                    A(clip(r, width))
        elif self.show_log:
            lp = active_log(self.llm)
            free = height - len(L) - 4
            if lp and free >= 3:
                rows = [t.dim() + clip(x, width - 4) + t.rst()
                        for x in tail(lp, free)]
                for r in self.box(os.path.basename(lp),
                                  rows or [t.dim() + "(empty)" + t.rst()], width, t.head):
                    A(clip(r, width))

        L = [clip(ln, width) for ln in L]
        # Never hand back more rows than the terminal has: the extras scroll the
        # top of the panel off screen and the fixed-position redraw tears.
        if height and len(L) > height:
            L = L[:height]
        self.last_frame = L
        return L

# ==================== suzu/render/view.py
"""The dashboard object.

`View` is assembled from four mixins so each concern has its own file:

  core.py     state: construction, theme cycling, ramps, smoothing, tick()
  widgets.py  drawing primitives: bars, sparks, boxes, gauges, big digits
  panels.py   panel builders: cluster, lanes, model strip, experts, heatmap
  frames.py   full-screen composition: full / perf / pipes / layers / spec

Methods in one mixin call methods from another through `self`, exactly as they
did inside the single class.
"""


class View(WidgetsMixin, PanelsMixin, FramesMixin, CoreMixin):
    """Live view of one engine on one port.

    Nothing is cached per terminal size: every frame is rebuilt from the size
    it is given, so a resize takes effect on the next tick.
    """

# ==================== suzu/cli.py
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
        globals()["_RANK1"] = _rank1_feed(config.RANK1_URL)   # rebuild the process-wide feed with the URL

    view = View(port, interval, show_log, theme)
    if layers:
        view.mode = "layers"
    out = sys.stdout

    if once:
        size = term_size()
        w, h = min(size.columns, 500), size.lines
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


# ==================== entry point
if __name__ == "__main__":
    sys.exit(main())
