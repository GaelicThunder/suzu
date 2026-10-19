"""Engine log discovery and tail."""
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

from ..ansi import ANSI_RE


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

