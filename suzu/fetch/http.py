"""HTTP + Prometheus text parsing shared by every engine adapter."""
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

