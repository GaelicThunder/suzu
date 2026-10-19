"""GPU rows: nvidia-smi first, rocm-smi fallback."""
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
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,",
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

