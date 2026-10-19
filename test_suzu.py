#!/usr/bin/env python3
"""Tests for suzu's logic, driven against a synthetic engine.

Nothing here touches :30001. The point is to reproduce the situation that
dictates the tool's design and would otherwise only show up in production:
/metrics frozen mid-generation while /slots keeps advancing.

run: python3 test_suzu.py     (exit 0 = all good)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import suzu as m
import suzu.fetch.engine
import suzu.fetch.http
import suzu.fetch.info
import suzu.fetch.logtail
import suzu.fetch.memory
import suzu.fetch.rank1
import suzu.fetch.tabby
import suzu.fetch.tensorfold
import suzu.fetch.vllm
import suzu.render.core
import suzu.render.frames
import suzu.render.panels

# The split put the code in submodules: `suzu.http_get` is a re-export, not the
# binding core.step() looks up. patch() writes a fake into every module that
# imported that name, so a test patches the behaviour the code actually calls
# instead of only the package facade.
_TARGETS = [m, suzu.fetch.engine, suzu.fetch.http, suzu.fetch.info,
            suzu.fetch.logtail, suzu.fetch.memory, suzu.fetch.rank1,
            suzu.fetch.tabby, suzu.fetch.tensorfold, suzu.fetch.vllm,
            suzu.render.core, suzu.render.frames, suzu.render.panels]


def patch(name, value):
    for mod in _TARGETS:
        if hasattr(mod, name):
            setattr(mod, name, value)


# the theme cycling in test 9 goes through cycle_theme(), which SAVES the theme: pointed at the real file it overwrote
# the one picked with `t` on every run of this suite (30-09). The tests get their own file.
import tempfile as _tf
patch("THEME_STATE", _tf.NamedTemporaryFile(prefix="suzu-theme-", delete=False).name)

STATE = {"dec": 0, "draft": 1000.0, "acc": 440.0, "drafts": 200.0,
         "pos": [150.0, 110.0, 80.0, 57.0, 43.0], "gen": 5000.0, "busy": True}


def fake_metrics():
    s = STATE
    L = ["llamacpp:tokens_predicted_total %f" % s["gen"],
         "llamacpp:prompt_tokens_total 100000",
         "llamacpp:prompt_seconds_total 400",
         "llamacpp:spec_decode_num_draft_tokens_total %f" % s["draft"],
         "llamacpp:spec_decode_num_accepted_tokens_total %f" % s["acc"],
         "llamacpp:spec_decode_num_drafts_total %f" % s["drafts"],
         "llamacpp:requests_processing %d" % (1 if s["busy"] else 0),
         "llamacpp:requests_deferred 0", "llamacpp:n_tokens_max 165117"]
    for i, v in enumerate(s["pos"]):
        L.append('llamacpp:spec_decode_num_accepted_tokens_per_pos_total'
                 '{position="%d"} %f' % (i, v))
    return "\n".join(L)


def fake_slots():
    s = STATE
    return json.dumps([{"id": 0, "n_ctx": 262144, "is_processing": s["busy"],
                        "n_prompt_tokens": 60000, "n_prompt_tokens_processed": 1000,
                        "n_prompt_tokens_cache": 57000,
                        "next_token": [{"n_decoded": s["dec"]}]}])


patch("http_get", lambda url, timeout=4, tries=2: fake_slots() if "/slots" in url else fake_metrics())
patch("server_info", lambda port: {"draft": "/fake/drafter.gguf", "spec": "draft-dspark",
                              "k": "5", "up": 3600, "model": "/fake/m.gguf",
                              "engine": "llama-server"})

# no rank-1 HTTP from the tests: the rank-1 feed is tested on its own in 11c
patch("fetch_rank1", lambda url=None, tf=False: {"up": None, "telemetry": "connecting"})
v = m.View(30001, 1.0, False)
CLOCK = [1000.0]
m.time.time = lambda: CLOCK[0]           # deterministic clock

fails = []


def tick(dt=1.0):
    CLOCK[0] += dt
    return v.step()


def check(name, got, want, tol=1e-6):
    ok = abs(got - want) <= tol
    print("%-44s got=%-10.4f want=%-10.4f %s" % (name, got, want, "OK" if ok else "FAIL"))
    if not ok:
        fails.append(name)


def ok(name, cond, detail=""):
    print("%-44s %s %s" % (name, "OK" if cond else "FAIL", detail))
    if not cond:
        fails.append(name)


print("--- 1) generating: /metrics frozen, /slots climbing at 20 tok/s")
tick()                                   # baseline, no delta
for _ in range(3):
    STATE["dec"] += 20
    cur, live = tick(1.0)
check("live gen t/s (from /slots)", live["gen"], 20.0)
check("history got the 3 real ticks", len(v.h_gen), 3)
check("no spec window closed yet", len(v.h_acc), 0)

print("\n--- 2) request completes: /metrics unfreezes in one jump")
STATE["gen"] += 60
STATE["dec"] = 0
STATE["busy"] = False
STATE["draft"] += 500
STATE["acc"] += 315
STATE["drafts"] += 100
for i, d in enumerate([88, 74, 61, 50, 42]):
    STATE["pos"][i] += d
cur, live = tick(1.0)
check("window alpha = 315/500", v.w["alpha"], 0.63)
check("window tau = 315/100", v.w["tau"], 3.15)
check("window draft window = 500/100", v.w["dwin"], 5.0)
check("window p0 = 88/100", v.w["pos"][0] / v.w["drafts"], 0.88)
check("window p4 = 42/100", v.w["pos"][4] / v.w["drafts"], 0.42)
check("cumulative jump is not read as speed", live["gen"], 0.0)

print("\n--- 3) new request: n_decoded restarts from 0 (negative delta)")
STATE["busy"] = True
STATE["dec"] = 5
cur, live = tick(1.0)
check("positive delta after the reset", live["gen"], 5.0)
STATE["dec"] = 3                          # artificial regression
cur, live = tick(1.0)
check("negative delta dropped, not negated", live["gen"], 0.0)

print("\n--- 4) engine restart: cumulative counters go backwards")
STATE.update(dec=0, draft=10.0, acc=4.0, drafts=2.0, gen=10.0,
             pos=[2.0, 1.0, 1.0, 0.0, 0.0])
cur, live = tick(1.0)
check("history cleared on restart", len(v.h_gen), 0)
check("restart counted", v.restarts, 1)

print("\n--- 5) idle slot must not show prefix cache as 0%")
STATE.update(draft=1000.0, acc=440.0, drafts=200.0,
             pos=[150.0, 110.0, 80.0, 57.0, 43.0], busy=True, dec=10)
tick(1.0)
v.frame(130, 40)
STATE["busy"] = False
lines = v.frame(130, 40)
joined = m.SGR_RE.sub("", "\n".join(lines))
ok("prefix cache keeps its last value", "prefix cache" in joined and " last" in joined)

print("\n--- 6) layout at every width: nothing crosses the border")
for w in (40, 60, 80, 100, 118, 130, 170, 200):
    lines = v.frame(w, 60)
    over = [l for l in lines if m.vis_len(l) > w]
    print("  width=%-4d rows=%-3d overflow=%d"
          % (w, len(lines), len(over)))
    if over:
        fails.append("overflow@%d" % w)

print("\n--- 7) height clamp: never return more rows than the terminal has")
for h in (10, 20, 30, 60):
    lines = v.frame(130, h)
    ok("height=%-3d -> %d rows <= %d" % (h, len(lines), h), len(lines) <= h)

print("\n--- 8) every theme in the cycle is multicolour, and its ramp is readable")
for name in m.THEME_ORDER:
    tv = m.View(30001, 1.0, False, name)
    tv.prev = v.prev
    lines = tv.frame(130, 40)
    body = "\n".join(lines)
    has_ansi = bool(m.SGR_RE.search(body))
    # the contract: 5 DISTINCT steps. A ramp that repeats a colour cannot tell
    # 30% from 70%, which is the whole reason the bars are coloured at all.
    scale = m.THEMES[name]["scale"]
    ok("theme %-7s renders coloured, %d/5 distinct ramp steps"
       % (name, len(set(scale))), bool(lines) and has_ansi and len(set(scale)) == 5)

print("\n--- 8b) mono is out of the cycle but still reachable, and stays plain")
ok("mono not in the cycle", "mono" not in m.THEME_ORDER)
mv = m.View(30001, 1.0, False, "mono")
mv.prev = v.prev
mlines = mv.frame(130, 40)
ok("mono renders with no ANSI at all", bool(mlines)
   and not m.SGR_RE.search("\n".join(mlines)))

print("\n--- 8c) bars are graded, so the colour itself carries the value")
gv = m.View(30001, 1.0, False, "neon")
full = gv.bar(1.0, 20)
cols = m.SGR_RE.findall(full)
ok("full bar walks the whole ramp (%d colour changes)" % len(set(cols)),
   len(set(cols)) >= 5)
half = gv.bar(0.5, 20)
ok("half bar stops partway up the ramp (%d < %d)"
   % (len(set(m.SGR_RE.findall(half))), len(set(cols))),
   len(set(m.SGR_RE.findall(half))) < len(set(cols)))
inv = gv.bar(1.0, 20, invert=True)
seq_up = [c for c in m.SGR_RE.findall(full) if c != m.RST]
seq_dn = [c for c in m.SGR_RE.findall(inv) if c != m.RST]
ok("invert=True runs the ramp backwards", seq_dn[:-1] == seq_up[:-1][::-1],
   "%s vs %s" % (len(seq_dn), len(seq_up)))
flat = gv.bar(1.0, 20, color=m.fg(200))
# the trailing frame colour is for the UNFILLED track, so only count what is
# actually painting a block
paint = [c for c in m.SGR_RE.findall(flat) if c not in (m.RST, gv.t.frame)]
ok("explicit color= still gives a flat bar", len(set(paint)) == 1, str(paint))
ok("legend shows all 5 steps",
   len(set(m.SGR_RE.findall(gv.t.legend()))) >= 5)

print("\n--- 9) theme cycling wraps around")
tv = m.View(30001, 1.0, False, "neon")
seen = [tv.t.name] + [tv.cycle_theme(1) for _ in range(len(m.THEME_ORDER))]
ok("cycle visits every theme and returns", seen[-1] == "neon"
   and set(seen) == set(m.THEME_ORDER), str(seen))

# --------------------------------------------------------------------------
# 10) TabbyAPI/exllamav3: no /metrics, no /slots. The whole branch lives on
#     /props + the gateway stats + the engine log's `Metrics (ID: ...)` lines,
#     which rich writes SPLIT at 80 columns. That exact format is reproduced
#     here, because it is the only thing the parser has to support.
print("\n--- 10) TabbyAPI: /metrics 404, everything rebuilt from log + gateway")

TAB_LOG = "/tmp/test-suzu-tabby.log"
TAB = {"sse_total": 0, "inflight": None}


def tab_rec(gen, sec, cached, new, ptps, gtps, ctx, draft=None):
    """A record as rich writes it at 80 columns: wrapped inside the fields."""
    out = ("2026-08-29 21:44:41.508 INFO:     Metrics (ID: \n"
           "689515c2e0b646a59b79ec82e8781546): %d tokens generated in %s seconds (Queue: \n"
           "0.01 s, Process: %d cached tokens and %d new tokens at %s T/s, \n"
           "Generate: %s T/s, Context: %d tokens" % (gen, sec, cached, new, ptps, gtps, ctx))
    if draft:
        out += ", Draft: %d / %d tokens accepted (%.2f%%)" % (
            draft[0], draft[1], 100.0 * draft[0] / draft[1])
    return out + ") \n"


TAIL = ('2026-08-29 21:44:41.802 WARNING:  Unable to switch model to x\n'
        'because "inline_model_loading" is not True in config.yml.\n')


def tab_write(*recs):
    with open(TAB_LOG, "w") as f:
        f.write('2026-08-29 21:44:30.488 INFO:     127.0.0.1:49520 - "POST /v1/chat/completions \n'
                'HTTP/1.1" 200\n')
        for r in recs:
            f.write(r + TAIL)


def tab_http(url, timeout=4, tries=2):
    if "/metrics" in url:
        return None                       # 404: the case that defines the branch
    if "/props" in url:
        return json.dumps({"total_slots": 4,
                           "default_generation_settings": {"n_ctx": 524288}})
    if "/yzq/stats" in url:
        return json.dumps({"high": 0, "low": 0, "inflight": TAB["inflight"],
                           "sse_total": TAB["sse_total"], "served": 1})
    return None


def tab_reset():
    m._TABBY_PROPS.clear()
    m._TABBY_LOG.update(path=None, off=0, carry=None, tot=dict(m._TAB_ZERO), last=None)
    m._TABBY_MONO["gen"] = 0.0
    m._TABBY_RATIO.update(gen=0.0, ev=0.0, p_gen=None, p_ev=None, p_n=None, saved=0.0)


import os
try:
    os.remove(m._TABBY_KFILE)
except OSError:
    pass
tab_reset()
tab_write(tab_rec(206, "10.8", 137984, 312, "150.72", "23.62", 138296))
patch("http_get", tab_http)
patch("active_log", lambda llm: TAB_LOG)
patch("server_info", lambda port: {"draft": None, "spec": "disabled", "k": None,
                              "up": 900, "model": None, "engine": "tabby/exl3"})
tv = m.View(30001, 1.0, False, "mono")


def tab_tick(dt=1.0):
    CLOCK[0] += dt
    return tv.step()


cur, live = tab_tick()
ok("record split over 4 lines read all the same", m._TABBY_LOG["tot"]["n"] == 1,
   str(m._TABBY_LOG["tot"]))
check("tokens of the record", cur.gen, 206.0)
check("prefill: new tokens, not the cached ones", cur.pp, 312.0)
check("ctx max from /props", cur.ctxmax, 524288.0)
ok("family recognised as tabby", cur.fam == "tabby", cur.fam)
ok("without MTP no acceptance is invented", cur.draft == 0 and cur.acc == 0)
ok("verdict = No speculation", tv.verdict(0.0, 0.0)[0] == "No speculation")
ok("slot ctx from the last record", cur.ptok == 138296 and cur.pcache == 137984,
   "%d / %d" % (cur.ptok, cur.pcache))

# interrupted request: `Context:` smaller than the tokens already in cache.
# Without the max() the prefix bar read 131%, which is not a percentage.
tab_reset()
tab_write(tab_rec(12, "1.0", 34700, 100, "100.0", "23.0", 26502))
tvx = m.View(30001, 1.0, False, "mono")
CLOCK[0] += 1
curx, _ = tvx.step()
slx = tvx.slot_panel(curx)
ok("prefix never over 100% on an interrupted request",
   slx["prefix"] is not None and slx["prefix"] <= 1.0,
   "%.1f%%" % (100.0 * (slx["prefix"] or 0)))
tab_reset()
tab_write(tab_rec(206, "10.8", 137984, 312, "150.72", "23.62", 138296))
tv = m.View(30001, 1.0, False, "mono")
CLOCK[0] += 1
cur, live = tv.step()

print("\n--- 10b) in flight: the gateway's SSE events, not calibrated yet")
TAB.update(sse_total=10, inflight={"id": 1, "prio": "alta", "sse": 10})
cur, live = tab_tick()
check("k=1 until a record closes: 10 events = 10", cur.gen, 216.0)
ok("busy from the gateway, not the engine", cur.busy)

print("\n--- 10c) the record closes: k measures itself (300 tokens / 100 events)")
tab_write(tab_rec(206, "10.8", 137984, 312, "150.72", "23.62", 138296),
          tab_rec(300, "12.3", 0, 28, "107.69", "24.78", 28))
TAB.update(sse_total=100, inflight=None)
cur, live = tab_tick()
check("k misurato", m._TABBY_RATIO["gen"] / m._TABBY_RATIO["ev"], 3.0)
ok("k saved for the next panel",
   os.path.exists(m._TABBY_KFILE) and abs(float(open(m._TABBY_KFILE).read()) - 3.0) < 0.01)

print("\n--- 10d) next request: events are worth 3 tokens each")
TAB.update(sse_total=120, inflight={"id": 2, "prio": "alta", "sse": 20})
cur, live = tab_tick()
check("20 calibrated events = 60 tokens", cur.gen, 506.0 + 60.0)
check("live t/s scaled", live["gen"], 60.0)

print("\n--- 10e) client hangs up: the in-flight piece vanishes, NOT a restart")
r_before = tv.restarts
TAB.update(inflight=None)                 # no new record: the drop is fictitious
cur, live = tab_tick()
ok("no false restart", tv.restarts == r_before, "restarts=%d" % tv.restarts)
check("total kept monotone", cur.gen, 566.0)

print("\n--- 10f) with MTP on the acceptance exists and comes from the log")
tab_reset()
TAB.update(sse_total=0, inflight=None)
tab_write(tab_rec(206, "10.8", 137984, 312, "150.72", "23.62", 138296, draft=(120, 400)))
tv2 = m.View(30001, 1.0, False, "mono")
CLOCK[0] += 1
cur, live = tv2.step()
check("draft from the log", cur.draft, 400.0)
check("accepted from the log", cur.acc, 120.0)
check("alpha = 120/400", cur.acc / cur.draft, 0.30)

print("\n--- 10g) long prefill: /props mute but the process is there -> panel ALIVE")
# The case that counts: under a prefill TabbyAPI may not answer for tens of
# seconds, and switching off exactly then would be the defect this file exists
# to avoid. The fallback is the process environment, where the launcher wrote
# TABBY_MODEL_MAX_SEQ_LEN: not an estimate, the number the server received.
patch("http_get", lambda url, timeout=4, tries=2: (
    None if "/props" in url or "/metrics" in url
    else json.dumps({"high": 0, "low": 0, "inflight": TAB["inflight"],
                     "sse_total": TAB["sse_total"], "served": 1})))
patch("tabby_env", lambda: {"TABBY_MODEL_MAX_SEQ_LEN": "262144"})
m._TABBY_PROPS.clear()
tab_reset()
tab_write(tab_rec(206, "10.8", 137984, 312, "150.72", "23.62", 138296))
tv3 = m.View(30001, 1.0, False, "mono")
CLOCK[0] += 1
r3 = tv3.step()
ok("step() does not give up when /props is mute", r3 is not None)
if r3:
    check("n_ctx from the process environment", r3[0].ctxmax, 262144.0)

print("\n--- 10h) engine really down: no /props, no process -> None")
patch("tabby_env", lambda: {})
m._TABBY_PROPS.clear()
tv4 = m.View(30001, 1.0, False, "mono")
CLOCK[0] += 1
ok("step() says None when nobody is there", tv4.step() is None)


print("\n--- 11) TensorFold: the two /metrics dialects (jayleaton tensorfold_*, newer kit tensorfold:* since 01-10)")
MIA = """tensorfold:requests_running 1
tensorfold:prompt_tokens_total 247397
tensorfold:generation_tokens_total 7936
tensorfold:kv_cache_usage_ratio{pool="0"} 0
tensorfold:mtp_drafted_total 7199
tensorfold:mtp_accepted_total 4274
tensorfold_health:requests_total 6
tensorfold_health:completion_tokens_total 7936
tensorfold_health:cached_tokens_total 156928
tensorfold_health:rounds_total 3656
tensorfold_health:prefill_seconds_total 56.2663
tensorfold_health:streams{state="decoding"} 1
tensorfold_health:context_length 1048576"""
m._TF_DELTA.update(reqs=None, prompt=0.0, cached=0.0, last={})
m._TF_MONO.update(gen=0.0, comp=0.0)
mx, sl = m.synth_tensorfold(m.parse_metrics(MIA), 8888, None)
ok("kit: recognised", mx is not None and mx.get("tensorfold:records") == 6.0)
ok("kit: real drafted tokens", mx.get("llamacpp:spec_decode_num_draft_tokens_total") == 7199.0
   and mx.get("llamacpp:spec_decode_num_accepted_tokens_total") == 4274.0)
ok("kit: round = verify step", mx.get("llamacpp:spec_decode_num_drafts_total") == 3656.0)
ok("kit: ctx from context_length, not HTTP", mx.get("llamacpp:n_tokens_max") == 1048576.0)
ok("kit: computed = prompt - cached", mx.get("llamacpp:prompt_tokens_total") == 247397.0 - 156928.0)
s11 = m.Sample(mx, sl, 0.0)
ok("kit: acceptance in %, not tokens/round", s11.fam == "tensorfold" and not s11.tf_nodraft)
nxt = MIA.replace("requests_total 6", "requests_total 7").replace("prompt_tokens_total 247397", "prompt_tokens_total 334052") \
         .replace("cached_tokens_total 156928", "cached_tokens_total 228200")
mx2, sl2 = m.synth_tensorfold(m.parse_metrics(nxt), 8888, None)
ok("kit: last request from the deltas", sl2[0]["n_prompt_tokens"] == 86655 and sl2[0]["n_prompt_tokens_cache"] == 71272,
   "got %d / %d" % (sl2[0]["n_prompt_tokens"], sl2[0]["n_prompt_tokens_cache"]))
OLD = """tensorfold_requests_total 6
tensorfold_completion_tokens_total 7936
tensorfold_decode_rounds_total 3656
tensorfold_prompt_tokens_total 247397
tensorfold_cached_tokens_total 156928
tensorfold_requests_inflight 0"""
m._TF_MONO.update(gen=0.0, comp=0.0)
patch("vllm_ctxmax", lambda port: 1048576)
patch("tf_last_record", lambda path=None: {})
patch("tf_lifetime", lambda path=None: None)
mx3, sl3 = m.synth_tensorfold(m.parse_metrics(OLD), 8888, None)
ok("jayleaton: accepted = completion - req - rounds", mx3.get("llamacpp:spec_decode_num_accepted_tokens_total") == 7936.0 - 6 - 3656)
ok("jayleaton: no drafted, stays tokens/round", m.Sample(mx3, sl3, 0.0).tf_nodraft)

print("\n--- 11b) newer kit: no double-counting in flight, live prefill from /health \"fill\" (image y3)")
HEALTH = {"v": {"streams": {"decoding": 1, "filling": 1}, "fill": [{"sid": 3, "pos": 20480, "prompt": 138000}]}}
_old_get = m.http_get
patch("http_get", lambda url, timeout=4, tries=2: json.dumps(HEALTH["v"]) if url.endswith("/health") else "")
m._TF_FILL.update(pos={}, done=0.0)
m._TF_MONO.update(gen=0.0, comp=0.0)
yzq = {"running": [{"sse": 500}]}
mxa, sla = m.synth_tensorfold(m.parse_metrics(MIA), 8888, yzq)
ok("kit: gen = completion (already in flight), no SSE", mxa.get("llamacpp:tokens_predicted_total") == 7936.0)
ok("kit: first read of the stream = base", sla[0]["n_prompt_tokens_processed"] == 0)
ok("kit: progress of the longest", mxa.get("tensorfold:fill_pos") == 20480.0 and mxa.get("tensorfold:fill_len") == 138000.0)
HEALTH["v"]["fill"][0]["pos"] = 22528
mxb, slb = m.synth_tensorfold(m.parse_metrics(MIA), 8888, yzq)
ok("kit: lines computed live", slb[0]["n_prompt_tokens_processed"] == 2048, "got %d" % slb[0]["n_prompt_tokens_processed"])
HEALTH["v"] = {"streams": {"decoding": 1, "filling": 1}, "fill": [{"sid": 3, "pos": 90000, "prompt": 200000}]}
mxc, slc = m.synth_tensorfold(m.parse_metrics(MIA), 8888, yzq)
ok("kit: new prompt resumed from cache is not prefill", slc[0]["n_prompt_tokens_processed"] == 2048)
HEALTH["v"] = {"streams": {"decoding": 1, "filling": 1}}
mxd, sld = m.synth_tensorfold(m.parse_metrics(MIA), 8888, yzq)
ok("kit on y2: only the stream count", mxd.get("tensorfold:fill_streams") == 1.0 and "tensorfold:fill_len" not in mxd)
patch("http_get", _old_get)

print("\n--- 11c) TP cluster (05-10): rank 1 from a background HTTP poller, never from the render loop; the link from RoCE")
import os as _os, re as _re, json as _json, types
_f = m.Rank1Feed("http://127.0.0.1:9100/metrics")
ok("feed: nothing yet -> connecting, up unknown", _f.info(tf=True)["telemetry"] == "connecting" and _f.info(tf=True)["up"] is None)
_f._parse("gpu_utilization 95\ngpu_power_watts 56.2\ngpu_temp_c 71\ngpu_clock_mhz 2509\n")
_i = _f.info(tf=True)
ok("feed: sample parsed (GPU row)", _i["gpu"]["util"] == 95.0 and _i["gpu"]["w"] == 56.2
   and _i["gpu"]["c"] == 71.0 and _i["gpu"]["mhz"] == 2509.0, repr(_i))
ok("feed: fresh sample -> up, live", _i["up"] is True and _i["telemetry"] == "live")
_f._parse("gpu_utilization 0\ngpu_power_watts 4.0\ngpu_temp_c 40\ngpu_clock_mhz 208\n")
ok("feed: second sample keeps it live", _f.info(tf=True)["telemetry"] == "live")
_f.last = (m.time.monotonic() - 30, None, None, None)
_i = _f.info(tf=True)
ok("feed: stale telemetry is NOT a dead rank", _i["telemetry"] == "stale" and _i["up"] is None, repr(_i["up"]))
_f.down_since = m.time.monotonic() - 25
ok("feed: unreachable 25 s -> unreachable, up False", _f.info(tf=True)["telemetry"] == "unreachable" and _f.info(tf=True)["up"] is False)
_f.down_since = m.time.monotonic() - 5
ok("feed: unreachable 5 s -> not yet called down", _f.info(tf=True)["up"] is None)
_f0 = m.Rank1Feed(None)
ok("feed: no --rank1-url -> not set, never up", _f0.info(tf=True)["telemetry"] == "no url" and _f0.info(tf=True)["up"] is None)
ok("feed: tf=False asks for rank 1's /metrics", (_f.info(tf=False), _f.want_metrics)[1] is True)
_r = _tf.mkdtemp()
_old_open = open
for _d, (_x, _y) in {"rocep1s0f0": (0, 0), "rocep1s0f1": (1000, 2000)}.items():
    _os.makedirs(_os.path.join(_r, _d, "ports/1/counters"))
    _old_open(_os.path.join(_r, _d, "ports/1/counters/port_xmit_data"), "w").write(str(_x))
    _old_open(_os.path.join(_r, _d, "ports/1/counters/port_rcv_data"), "w").write(str(_y))
m._ROCE.update(prev=None, tx=None, rx=None, pk=0.0, devs=())
_real_listdir = m.os.listdir
m.os.listdir = lambda p: _real_listdir(_r) if p == "/sys/class/infiniband" else _real_listdir(p)
import builtins as _bi
_real_open = _bi.open
_bi.open = lambda p, *a, **k: _real_open(p.replace("/sys/class/infiniband", _r), *a, **k) if isinstance(p, str) else _real_open(p, *a, **k)
try:
    _t0 = m.roce_link()
    m._ROCE["prev"] = (m._ROCE["prev"][0] - 1.0,) + m._ROCE["prev"][1:]
    _real_open(_os.path.join(_r, "rocep1s0f1/ports/1/counters/port_xmit_data"), "w").write(str(1000 + 250))
    _real_open(_os.path.join(_r, "rocep1s0f1/ports/1/counters/port_rcv_data"), "w").write(str(2000 + 500))
    _t1 = m.roce_link()
finally:
    _bi.open = _real_open
    m.os.listdir = _real_listdir
ok("roce: first read has no rate yet", _t0[0] is None)
ok("roce: 4-byte words -> bytes/s, idle functions left out", abs(_t1[0] - 1000.0) < 50 and abs(_t1[1] - 2000.0) < 100
   and _t1[3] == ("rocep1s0f1",), repr(_t1))
_v2 = m.View(30001, 1.0, False)
_v2.info = {"nnodes": 2}
_v2.tf = True
_v2.gpu = [{"util": 94.0, "mem": None, "memtot": None, "w": 42.0, "c": 75.0, "mhz": 2281.0}]
_v2.memory = {"total": 127535356, "avail": 2726297, "used_frac": 0.98}
_f2 = m.Rank1Feed("http://127.0.0.1:9100/metrics")
_f2._parse("gpu_utilization 95\ngpu_power_watts 56.2\ngpu_temp_c 71\ngpu_clock_mhz 2509\n")
_v2.rank1 = _f2.info(tf=True)
m._TF_HEALTH.update(at=m.time.monotonic(), ok=True, iteration_s=0.1)
for _w in (60, 100, 170):
    _rows = _v2.cluster_panel(_w)
    ok("panel at %d cols: rank 0, rank 1, link; inside the box" % _w, len(_rows) == 3 and all(m.vis_len(x) <= _w - 2 for x in _rows))
_txt = m.strip_ansi("\n".join(_v2.cluster_panel(200))) if hasattr(m, "strip_ansi") else _re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(_v2.cluster_panel(200)))
ok("panel: both GPUs, rank 1 up, in step", "94%" in _txt and "95%" in _txt and "in step" in _txt, _txt)
m._TF_HEALTH.update(iteration_s=45.0)
ok("panel: a long iteration says STALLED", "STALLED" in _re.sub(r"\x1b\[[0-9;]*m", "", "\n".join(_v2.cluster_panel(200))))
m._TF_HEALTH.update(at=None)

print("\n--- 11d) layout (05-10): GLM-5.3 topology for the expert heat map, every layer shown, nothing past the border")
_cfg = _tf.NamedTemporaryFile("w", suffix=".json", delete=False)
_json.dump({"architectures": ["Glm5NextForConditionalGeneration"], "vision_config": {"depth": 24},
            "text_config": {"num_hidden_layers": 45, "n_routed_experts": 288, "num_experts_per_tok": 8,
                            "n_shared_experts": 1, "num_nextn_predict_layers": 1,
                            "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 42}}, _cfg)
_cfg.close()
_a = m.load_model_arch(_cfg.name)
ok("arch: text_config read (45 layers, 288 experts, top-8, MTP, vision 24)", _a and _a["n"] == 45 and _a["experts"] == 288
   and _a["topk"] == 8 and _a["nextn"] == 1 and _a["vision"] == 24, repr(_a))
ok("arch: MoE layers from mlp_layer_types", _a and _a["moe_layers"] == list(range(3, 45)))
_v3 = m.View(30001, 1.0, False)
_v3.arch, _v3.tf = _a, True
_v3.fade_expert = [[0.5] * 288 for _ in range(46)]
for _ra, _want in ((23, 23), (12, 12), (46, 46)):
    _g = _v3.expert_grid(200, rows_avail=_ra)
    _labs = [int(_re.sub(r"\x1b\[[0-9;]*m", "", x).split()[0]) for x in _g]
    ok("heat grid: %d rows for 46 layers -> %d rows, the last one reaches layer 44+" % (_ra, _want),
       len(_g) == _want and _labs[-1] >= 44 - (46 // _want), "rows %d labels %s" % (len(_g), _labs[-3:]))
_hr = _v3.hrow("gen t/s", [0.0, 51.2, 25.0], 99, "%.1f", "", "13 tick", fit=80)
ok("history row: fit= makes it exactly the box's inner width", m.vis_len(_hr) == 80, "len %d" % m.vis_len(_hr))
_v3.yzq = {"high": 0, "low": 1, "parallel": 4, "running": [{}]}
_c = types.SimpleNamespace(wait=1, wait_cap=0, wait_def=0)
_wr = _v3.waiting_row(_c)
ok("waiting: the engine's hold goes on a row of its own", isinstance(_wr, tuple) and len(_wr) == 2
   and "held by the engine" in _wr[1] and "held" not in _wr[0])
_os.unlink(_cfg.name)

print("\n--- 12) usage db: tokens of all models (two tables, no overlap)")
import os as _os, sqlite3 as _sq, tempfile as _tmp
_db = _tmp.NamedTemporaryFile(suffix=".db", delete=False).name
_c = _sq.connect(_db)
_cols = "session_id, model, api_call_count, input_tokens, output_tokens, cache_read_tokens, reasoning_tokens, first_seen"
for _t in ("session_model_usage", "session_model_usage_v21"):
    _c.execute("create table %s (%s)" % (_t, _cols))
_c.executemany("insert into session_model_usage values (?,?,?,?,?,?,?,?)",
               [("s1", "GLM", 10, 1000, 50, 4000, 5, 200.0), ("s2", "dsv4", 3, 300, 30, 0, 0, 300.0)])
_c.executemany("insert into session_model_usage_v21 values (?,?,?,?,?,?,?,?)",
               [("s0", "yuzu", 7, 700, 70, 0, 0, 100.0), ("s1", "GLM", 10, 1000, 50, 4000, 5, 200.0)])
_c.commit(); _c.close()
m._HERMES_USE.update(at=-1e9, val=None)
u = m.hermes_usage(path=_db, every=0)
ok("usage: sum without duplicates", u is not None and u["calls"] == 20 and u["in"] == 2000 and u["cached"] == 4000,
   "got %r" % ((u and (u["calls"], u["in"], u["cached"])),))
ok("usage: sorted by tokens, GLM first", u["models"][0][0] == "GLM" and len(u["models"]) == 3)
ok("usage: since first seen", u["since"] == 100.0)
m._HERMES_USE.update(at=-1e9, val=u)
ok("usage: db gone -> keeps the last reading", m.hermes_usage(path="/nonexistent/x.db", every=0) is u)
_os.unlink(_db)

print("\n--- 12b) heatmap: 7 rows Mo-Su, a column a week, months on top, today in the last, stats stacked on the right")
import datetime as _dt
_today = _dt.date(2026, 10, 1)
_days = {(_today - _dt.timedelta(days=i)).isoformat(): float(1e6 * (i % 9 + 1)) for i in range(0, 60)}
_days[(_today - _dt.timedelta(days=5)).isoformat()] = 0.0
_v = m.View(30001, 1.0, False, "mono")
_rows = _v.heatmap_rows(80, days=_days, today=_today)
_grid = lambda r: [c for c in r[3:].split() if len(c) == 2 and c[0] == c[1] and c[0] in m.DITHER]   # a day = 2 glyphs
ok("heatmap: months + 7 days + blank + legend", len(_rows) == 10 and _rows[8] == "" and _rows[9].startswith("less"),
   "got %d" % len(_rows))
ok("heatmap: weekday labels", [r[:2] for r in _rows[1:8]] == list(m.View.WEEKDAYS))
ok("heatmap: a column a week from Mon Aug 3 to this week", len(_grid(_rows[1])) == 9, "got %d" % len(_grid(_rows[1])))
ok("heatmap: no cell after today", all(len(_grid(_rows[1 + wd])) == (9 if wd <= _today.weekday() else 8) for wd in range(7)))
ok("heatmap: months over the week holding their 1st", _rows[0].split() == ["Sep", "Oct"], _rows[0])
ok("heatmap: stats on the right, best day first", "#1" in _rows[1] and "9.0M" in _rows[1] and "today" in _rows[4]
   and "total" in _rows[7], _rows[1])
ok("heatmap: width respected", all(m.vis_len(r) <= 78 for r in _rows))
ok("heatmap: square days (two glyphs) and stats flush right", all(len(c) == 2 for c in _grid(_rows[1]))
   and m.vis_len(_rows[7]) == 77, "len %d" % m.vis_len(_rows[7]))
_narrow = _v.heatmap_rows(15, days=_days, today=_today)
ok("heatmap: narrow -> stats below the legend, most recent weeks", len(_grid(_narrow[1])) == (13 - 3) // 3
   and any(r.startswith("#1") for r in _narrow[10:]))
ok("heatmap: no data -> no box", _v.heatmap_rows(60, days={}, today=_today) == [])
_db = _tmp.NamedTemporaryFile(suffix=".db", delete=False).name
_c = _sq.connect(_db)
_c.execute("create table session_model_usage (%s)" % _cols)
_c.execute("create table messages (session_id, timestamp)")
import time as _time
_t0 = _time.mktime((2026, 9, 1, 12, 0, 0, 0, 0, -1))
_c.execute("insert into session_model_usage values ('a', 'GLM', 1, 5000, 300, 9000, 0, ?)", (_t0,))
_c.execute("alter table session_model_usage add column last_seen")
_c.execute("update session_model_usage set last_seen = ?", (_t0 + 86400,))
_c.executemany("insert into messages values ('a', ?)", [(_t0,), (_t0 + 86400,), (_t0 + 86400 + 60,)])
_c.commit(); _c.close()
m._HERMES_DAYS.update(at=-1e9, val=None)
_hd = m.hermes_days(path=_db, every=0)
ok("days: generated tokens only, split by the session's messages per day", _hd == {"2026-09-01": 100.0, "2026-09-02": 200.0},
   "got %r" % (_hd,))
_os.unlink(_db)

print("\n--- 12c) dithered bars: full cells, a dithered edge, a dithered track")
_b = _v.bar(0.55, 10)
ok("bar: 5 full, one dithered edge, 4 of track", _b == "█████▒░░░░", repr(_b))
ok("bar: empty and full", _v.bar(0.0, 4) == "░░░░" and _v.bar(1.0, 4) == "████")

print("\nRESULT: %s" % ("ALL OK" if not fails else "FAILED: %s" % fails))
sys.exit(1 if fails else 0)
