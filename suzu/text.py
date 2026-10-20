"""Text measurement and formatting: visible width, pad, clip, units."""
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

from .ansi import RST, SGR_RE


def vis_len(s):
    return len(SGR_RE.sub("", s))



def pad(s, w):
    n = w - vis_len(s)
    return s + " " * n if n > 0 else s



def clip(s, w):
    """Truncate by VISIBLE characters - len() lies on coloured strings. When the
    cut would land inside a word it backs up to the last space that fits and
    drops the separator: a row that has to go loses its tail, not half a word."""
    if w <= 0:
        return ""
    if vis_len(s) <= w:
        return s
    out, seen, i, had, last = [], 0, 0, False, None
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
        if s[i] == " ":
            last = (len(out), seen)
        seen += 1
        i += 1
    if last is not None and last[1] >= max(6, w // 2):
        out = out[:last[0]]
    # A cut can land inside a coloured run, so close it - but only when the
    # string had any escape at all: appending RST unconditionally painted a
    # reset into plain (mono) output and broke the "no ANSI in mono" contract.
    return "".join(out).rstrip(" .") + (RST if had else "")



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

