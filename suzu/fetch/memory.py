"""Host memory, process RSS and llama.cpp /slots."""
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

from .http import http_get


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

