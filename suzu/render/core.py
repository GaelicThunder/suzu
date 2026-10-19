"""View state: construction, theme cycling, ramps, smoothing, tick."""
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
from ..fetch.engine import Sample, fetch_yzq, fetch_yzq_lanes
from ..fetch.gpu import fetch_gpu
from ..fetch.http import http_get, parse_metrics
from ..fetch.info import server_info
from ..fetch.logtail import log_mtime
from ..fetch.memory import fetch_memory, fetch_slots
from ..fetch.rank1 import fetch_rank1
from ..fetch.tabby import synth_tabby
from ..fetch.tensorfold import synth_tensorfold
from ..fetch.vllm import synth_slots_vllm, vllm_ctxmax
from ..themes import FLOW_COOL, FLOW_GOLD, FLOW_PHOS, GOLD_FAMILY, HEAT_COOL, HEAT_GOLD, HEAT_PHOS, PHOSPHOR_FAMILY, THEME_ORDER, TRACK_BG_COOL, TRACK_BG_GOLD, TRACK_BG_PHOS, Theme, VU_COOL, VU_GOLD, VU_PHOS, _bg_col, _col, _grad, _theme_save


def _smooth(prev, target, dt, tau_up, tau_down):
    """Exponential approach, fast up / slow down - fade.rs smooth()."""
    tau = tau_up if target > prev else tau_down
    a = 1.0 - 2.718281828459045 ** (-dt / max(tau, 0.001))
    v = prev + (target - prev) * a
    return max(0.0, min(1.0, v))



def _routed_indices(layer, step, n_experts, k):
    """fade.rs routed_indices(): deterministic LCG spread, same constants."""
    if n_experts == 0:
        return []
    k = max(1, min(k, n_experts))
    M = (1 << 64) - 1
    x = ((layer * 0x9E3779B1) ^ (step * 0x85EBCA6B)) & M
    out, guard = [], 0
    while len(out) < k and guard < k * 8:
        x = (x * 6364136223846793005 + 1442695040888963407) & M
        e = (x >> 33) % n_experts
        if e not in out:
            out.append(e)
        guard += 1
    return out



class CoreMixin:
    WIDE = 118
    NARROW = 76
    FADE_ATTACK = 0.65
    FADE_RELEASE = 0.12

    def __init__(self, port, interval, show_log, theme="neon"):
        self.port, self.interval, self.show_log = port, interval, show_log
        self.t = Theme(theme)
        self.llm = os.environ.get("SUZU_LOG_DIR", os.path.expanduser("~/.local/share/suzu/logs"))
        self.prev = None
        self.h_gen = deque(maxlen=120)
        self.h_pf = deque(maxlen=120)
        self.h_acc = deque(maxlen=120)
        self.h_tau = deque(maxlen=120)
        # Same rolling windows, but as (numerator, denominator) pairs. A mean of
        # per-tick RATES weights a 0,2 s tick like a 4 s one; the average has to be
        # sum(tokens)/sum(seconds) or it is not the rate of anything.
        self.q_gen = deque(maxlen=120)
        self.q_pf = deque(maxlen=120)
        self.q_acc = deque(maxlen=120)
        self.q_tau = deque(maxlen=120)
        self.yzq = None
        # vLLM credits the two halves of the prefill rate on DIFFERENT ticks:
        # measured 23-08, prompt_tokens_total jumped +7294 at 20:27:34 and
        # request_prefill_time_seconds_sum jumped +6,203 s at 20:27:36. Anything
        # that demands both inside one sample therefore never fires and the row
        # reads 0 forever. Bank each half and divide once both have landed.
        self.pend_tok = 0.0
        self.pend_sec = 0.0
        self.info, self.info_at = {}, 0.0
        self.mode = "spec"          # "spec" | "layers" (toggled with g)
        self.view = "full"          # "full" | "perf" (p) | "pipes" (b)
        self.show_layers = False    # L: layer visualizer replaces the log section
        self.arch = None            # cached model topology
        self.arch_path = None
        self.mm_prev = None         # last mm_cache_queries_total (vision detect)
        self.mm_t = 0.0
        self.miss, self.last_frame = 0, None
        self.restarts = 0
        self.w = None
        self.w_at = None
        self.last_prefix = None
        self.act_ts = None
        self.mx_retry = 0.0              # backoff while /metrics is absent (see step)
        self.fade = {}                   # attack/release heat per cell (fade_step)
        self._fade_dt = time.time()
        self.pk_gen = 0.0                # peak-hold gen t/s (VU-style, slow decay)
        self.pk_at = 0.0
        self.pk_pf = 0.0                 # peak-hold prefill t/s (pipes view)
        self.pk_pf_at = 0.0
        self.pk_ch = ""                  # big-digit channel string
        self.fade_expert = []            # per-layer expert heat (fade.rs expert[])
        self.gpu = None                  # nvidia-smi rows (fetch_gpu, cached)
        self.rank1 = None                # rank-1 probe (fetch_rank1)
        self.lanes = None                # /yzq/lanes detail (fetch_yzq_lanes)
        self.memory = None               # /proc memory picture (fetch_memory)
        self.h_ttft = deque(maxlen=120)  # time-to-first-token per request, s



    def cycle_theme(self, step=1):
        i = (THEME_ORDER.index(self.t.name) + step) % len(THEME_ORDER)
        self.t = Theme(THEME_ORDER[i])
        _theme_save(self.t.name)   # remember it for the next run
        return self.t.name



    def ramps(self):
        """The RGB ramp set for the active theme: gold family for the limone
        themes, the upstream ocean ramps otherwise."""
        if self.t.name in PHOSPHOR_FAMILY:
            return HEAT_PHOS, VU_PHOS, FLOW_PHOS, TRACK_BG_PHOS
        if self.t.name in GOLD_FAMILY:
            return HEAT_GOLD, VU_GOLD, FLOW_GOLD, TRACK_BG_GOLD
        return HEAT_COOL, VU_COOL, FLOW_COOL, TRACK_BG_COOL



    def ramp_heat(self, v):
        return _col(_grad(self.ramps()[0], v))



    def ramp_heat_bg(self, v):
        return _bg_col(_grad(self.ramps()[0], v))



    def ramp_vu(self, v):
        return _col(_grad(self.ramps()[1], v))



    def ramp_flow(self, v):
        return _col(_grad(self.ramps()[2], v))



    def ramp_track(self):
        return self.ramps()[3]



    def fade_dt(self):
        """One shared dt per frame - call once, pass to every fade_step."""
        now = time.time()
        dt = now - self._fade_dt if now > self._fade_dt else 0.25
        self._fade_dt = now
        return dt



    def fade_step(self, key, hot, dt):
        """Advance one heat cell; returns its heat 0..1 (0 = cold)."""
        v = self.fade.get(key, 0.0)
        if hot:
            v = min(1.0, v + self.FADE_ATTACK * max(dt, 0.25) * 4)
        else:
            v *= max(0.0, 1.0 - self.FADE_RELEASE * dt)
        if v < 0.02:
            self.fade.pop(key, None)
            return 0.0
        self.fade[key] = v
        return v



    def expert_deltas(self):
        """Real routing from TensorFold's GET /experts (patches/0990: glm53tf, glm53mia y9+): per layer, how many times each expert was
        picked by decode windows since the previous read. None when the engine has no such endpoint (every engine
        but tensorfold: polled only there, so the others' logs do not fill with 404s) or on the first read."""
        if not getattr(self, "tf", False):
            return None
        now = time.time()
        st = getattr(self, "_xs", None)
        if st is None:
            st = self._xs = {"at": 0.0, "prev": None, "delta": None}
        if now - st["at"] < 0.5:
            return st["delta"]
        st["at"] = now
        txt = http_get("http://127.0.0.1:%d/experts" % self.port, timeout=2, tries=1)
        try:
            j = json.loads(txt) if txt else None
        except (json.JSONDecodeError, ValueError):
            j = None
        cnt = j.get("counts") if isinstance(j, dict) and j.get("on") else None
        # on: the engine counts (a second read gives the first delta); why: it has the route but counts nothing
        st["on"] = bool(cnt)
        st["why"] = (j.get("why") or j.get("error")) if isinstance(j, dict) and not cnt else None
        if not cnt:
            st["prev"], st["delta"] = None, None
            return None
        prev, st["prev"] = st["prev"], cnt
        if prev is None or len(prev) != len(cnt):
            st["delta"] = None
            return None
        st["delta"] = [[max(0, a - b) for a, b in zip(r, q)] for r, q in zip(cnt, prev)]
        return st["delta"]



    def fade_experts(self, arch, working, dt, token_step):
        """fade.rs expert heat: routed expert jumps to 1.0, cools with a
        0.6 s exponential. On TensorFold the identities are REAL (GET /experts,
        patches/0990): every expert picked since the last read jumps to its
        share of that layer's busiest one. Elsewhere they come from the same
        deterministic stand-in router as upstream."""
        real = self.expert_deltas()
        n_l = max(arch["n"], len(real)) if real is not None else arch["n"]     # + the MTP layer's row
        n_e = min(arch["experts"], 320 if real is not None else 256)
        if len(self.fade_expert) != n_l or \
                (self.fade_expert and len(self.fade_expert[0]) != n_e):
            self.fade_expert = [[0.0] * n_e for _ in range(n_l)]
        cool = 2.718281828459045 ** (-dt / 0.6)
        for row in self.fade_expert:
            for j in range(n_e):
                row[j] *= cool
        if real is not None:
            for l in range(min(n_l, len(real))):
                d = real[l]
                m = max(d) if d else 0
                if m <= 0:
                    continue
                row = self.fade_expert[l]
                for j in range(min(n_e, len(d))):
                    if d[j] > 0:
                        row[j] = max(row[j], d[j] / m)
        elif working:
            k = arch.get("topk") or 8
            for l in range(n_l):
                row = self.fade_expert[l]
                for e in _routed_indices(l, int(token_step), n_e, k):
                    row[e] = 1.0



    def refresh_info(self):
        if time.time() - self.info_at > 20:
            self.info = server_info(self.port)
            self.info_at = time.time()



    def step(self):
        self.yzq = fetch_yzq(self.port)
        self.lanes = fetch_yzq_lanes(self.port, config.LANES_URL) if config.LANES_URL else None
        self.gpu = fetch_gpu()
        self.memory = fetch_memory(pid=self.info.get("pid"))
        # Rank-1 probe: fetch_rank1 caches for 10 s, so this is polite
        # even every frame. Probed on EVERY engine shape now - on the TP=2
        # cluster it is the rank-1 worker, on a single-node engine it is the
        # second box itself (asked for on 21-42: visible on 1-node too).
        self.rank1 = fetch_rank1(tf=getattr(self, "tf", False))
        mx = slots = None
        now = time.time()
        if now >= self.mx_retry:
            txt = http_get("http://127.0.0.1:%d/metrics" % self.port)
            if txt is None:
                # Silence or 404. On TabbyAPI it is 404 and will stay 404: without
                # this pause the panel would hammer it twice a second and every
                # hit lands in the engine log (seen 29-08: thousands of
                # `GET /metrics 404` lines). Retried every minute, because under
                # :30001 the engine may change while the panel is open.
                self.mx_retry = now + 60
            else:
                self.mx_retry = 0.0
                mx = parse_metrics(txt)
                if "tensorfold_requests_total" in mx or "tensorfold_health:requests_total" in mx:
                    mx, slots = synth_tensorfold(mx, self.port, self.yzq)
                else:
                    slots = fetch_slots(self.port) or synth_slots_vllm(mx, self.port)
        if mx is None:
            mx, slots = synth_tabby(self.port, self.llm, self.yzq)
            if mx is None:
                return None
        cur = Sample(mx, slots, time.time())
        self.tf = cur.fam == "tensorfold"
        if not cur.ctxmax and cur.fam == "vllm":
            cur.ctxmax = vllm_ctxmax(self.port)
        self.refresh_info()
        # cmdline facts arrive with info; refresh_info only re-fetches every
        # 20 s. Same un-gated probe as the head of step(): the panel decides
        # cluster-vs-standalone from nnodes at render time, not here.
        self.rank1 = fetch_rank1(tf=self.tf)

        p = self.prev
        if p is not None and cur.restarted_from(p):
            self.restarts += 1
            p = None
            for h in (self.h_gen, self.h_pf, self.h_acc, self.h_tau,
                      self.q_gen, self.q_pf, self.q_acc, self.q_tau):
                h.clear()
            self.pend_tok = self.pend_sec = 0.0
            self.w = self.w_at = None
        self.prev = cur

        # Last engine activity. Real token deltas are the truth: a slot left in
        # "processing" with nothing decoded is not work. The first frame has no
        # predecessor to diff, so the clock is seeded from the log's mtime.
        if cur.busy or (p is not None and (cur.gen > p.gen or cur.pp > p.pp)):
            self.act_ts = cur.t
        elif self.act_ts is None:
            self.act_ts = log_mtime(self.llm)

        # Time to first token: the poll where a quiet engine turns busy AND
        # starts decoding. Quantised to the interval (1 s), like the repo's
        # TTFT - honest about what a poller can know.
        if p is not None and not p.busy and cur.busy:
            self._ttft_pending = cur.t
        if p is not None and getattr(self, "_ttft_pending", None) \
                and cur.dec > p.dec:
            self.h_ttft.append(max(0.0, cur.t - self._ttft_pending))
            self._ttft_pending = None
        elif p is not None and not cur.busy:
            self._ttft_pending = None

        live = {"gen": 0.0, "pf": 0.0}
        if p is not None:
            dt = max(1e-3, cur.t - p.t)
            # n_decoded restarts at 0 on every request: a negative delta is a new
            # task, not a stall -> drop the window instead of reporting nonsense.
            ddec = cur.dec - p.dec
            if ddec >= 0 and cur.busy:
                live["gen"] = ddec / dt
            dpp, dps = cur.pp - p.pp, cur.ps - p.ps
            pf_pushed = False
            if cur.fam == "vllm":
                # NEVER derive prefill from ppp here. On vLLM /slots is synthetic
                # (synth_slots_vllm) and n_prompt_tokens_processed is really
                # kv_cache_usage_perc * pool — KV OCCUPANCY, not prefill progress.
                # Its delta is the rate the cache fills, which a prefix-cache hit
                # or a fresh 100k conversation makes jump by the whole context in
                # one tick; and it is quantised by the pool (2.19 M tokens here,
                # so one step of the gauge is ~22 k phantom tokens). That is where
                # "prompt t/s 93.550" and "avg 4487" came from (measured 23-08).
                # The counters are the only honest source: prompt_tokens_total is
                # credited at request close, request_prefill_time_seconds_sum is
                # the matching denominator, so their ratio is tokens per second
                # OF PREFILL — which is what the label claims.
                if dpp > 0:
                    self.pend_tok += dpp
                if dps > 1e-3:
                    self.pend_sec += dps
                if self.pend_tok > 0 and self.pend_sec > 1e-3:
                    live["pf"] = self.pend_tok / self.pend_sec
                    self.h_pf.append(live["pf"])
                    self.q_pf.append((self.pend_tok, self.pend_sec))
                    self.pend_tok = self.pend_sec = 0.0
                    pf_pushed = True
            else:
                # llama.cpp: /slots really does advance during the prefill, and
                # that live reading is the whole reason this file exists.
                dppp = cur.ppp - p.ppp
                if dppp > 0:
                    live["pf"] = dppp / dt
                elif dpp > 0 and dps > 1e-3:
                    live["pf"] = dpp / dps

            dd, da, dn = cur.draft - p.draft, cur.acc - p.acc, cur.drafts - p.drafts
            if cur.tf_nodraft and dn > 0:
                self.w = {"alpha": 0.0, "tau": da / dn, "dwin": 0.0, "pos": [], "drafts": dn}
                self.w_at = cur.t
                self.h_tau.append(self.w["tau"])
                self.q_tau.append((da, dn))
            elif dd > 0 and dn > 0:
                self.w = {"alpha": da / dd, "tau": da / dn, "dwin": dd / dn,
                          "pos": [a - b for a, b in zip(cur.pos, p.pos)],
                          "drafts": dn}
                self.w_at = cur.t
                self.h_acc.append(self.w["alpha"] * 100)
                self.h_tau.append(self.w["tau"])
                self.q_acc.append((da, dd))
                self.q_tau.append((da, dn))

            # ONLY ticks that actually decoded. The old gate was `cur.busy`, and
            # on vLLM busy is just num_requests_running > 0 — which is also true
            # for the whole PREFILL, where generation_tokens_total does not move.
            # At 245k ctx that is many seconds of 0.0 pushed into the history per
            # turn, and they dragged "avg gen t/s" down while the panel was
            # showing a healthy live rate. A prefill is not a slow generation.
            if ddec > 0:
                self.h_gen.append(live["gen"])
                self.q_gen.append((ddec, dt))
            if live["pf"] > 0 and not pf_pushed:
                self.h_pf.append(live["pf"])
                self.q_pf.append((dpp, dps) if dps > 1e-3 else (live["pf"] * dt, dt))
            # prefill peak-hold for the pipes view (same decay rule as gen)
            if live["pf"] >= self.pk_pf:
                self.pk_pf, self.pk_pf_at = live["pf"], time.time()
            elif time.time() - self.pk_pf_at > 2.0:
                self.pk_pf = max(live["pf"], self.pk_pf - 0.02 * self.pk_pf)
        return cur, live



    @staticmethod
    def mean(seq):
        return sum(seq) / len(seq) if seq else 0.0



    @staticmethod
    def wmean(seq):
        """sum(numerators)/sum(denominators) over the window - the rate itself,
        not the mean of the per-tick rates."""
        num = sum(a for a, _ in seq)
        den = sum(b for _, b in seq)
        return num / den if den else 0.0

