"""TensorFold adapter: request log, gateway SSE, hermes usage db."""
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

from .. import config
from .engine import yzq_live_sse
from .http import http_get
from .vllm import vllm_ctxmax


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

