"""Drawing primitives: bars, sparks, boxes, gauges, big digits."""
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

from ..ansi import RST
from ..fetch.http import g
from ..text import clip, pad, vis_len
from ..themes import BIG_GLYPHS, BLOCKS, DITHER, PULSE, WHITE_FG, _bg_col, _col


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

