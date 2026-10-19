"""TabbyAPI / exllamav3 adapter: /props + gateway + engine log."""
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
from .engine import yzq_live_sse
from .http import http_get
from .logtail import active_log


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

