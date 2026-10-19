"""Themes: role->colour tables, the 5-step quality ramp, truecolor ramps."""
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

from .ansi import BOLD, DIM, ESC, RST, bg, fg


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

