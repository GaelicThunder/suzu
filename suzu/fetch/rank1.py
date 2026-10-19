"""Rank-1 (tensor-parallel peer) probe and the RoCE link read."""
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

from .http import g, http_get, parse_metrics


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

