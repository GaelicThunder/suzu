"""Panel builders: cluster, lanes, model strip, experts, heatmap, slots."""
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

from ..ansi import RST, fg
from .. import config
from ..fetch.http import g, gl, http_get, parse_metrics
from ..fetch.info import load_model_arch, model_config_path
from ..fetch.rank1 import roce_link
from ..fetch.tensorfold import _TF_HEALTH, hermes_days, hermes_usage
from ..text import clip, human, pad, si_bytes
from ..themes import DITHER


class PanelsMixin:
    MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
    WEEKDAYS = ("Mo", "Tu", "We", "Th", "Fr", "Sa", "Su")

    def cluster_title(self):
        return "tensor-parallel cluster" if self.info.get("nnodes") in ("2", 2) else "gb10 . rank 0 + rank 1"



    def cluster_panel(self, width):
        """Both GB10s in one box, in aligned columns (05-10: "replace gpu . memory . gb10 with that"): this box
        (rank 0 on the TP=2 cluster: GPU on the VU ramp, watts, clock, temperature, unified memory, the engine's rss)
        and rank 1 (from Rank1Feed's HTTP poller: same columns, plus how fresh the sample is), then on the cluster
        the link row: bytes/s through this box's RoCE functions (roce_link: every round's all-gathers, so it moves
        with decode and prefill) and whether the pair is in step (rank 0's /health iteration_s: a stalled pair keeps
        growing it). On a single-node engine the same two box rows, no link, and an explicit UNREACHABLE when rank 1
        is asleep."""
        cluster = self.info.get("nnodes") in ("2", 2)
        t = self.t
        r1 = self.rank1 or {"up": None, "telemetry": "connecting"}
        inner = width - 2
        gw = max(8, min(32, (inner - 118) // 2 + 8))
        narrow = inner < 110

        def name(rank, box):
            return pad("%s%s%s  %s%s%s" % (t.head, rank, t.rst(), t.label, box, t.rst()), 14) if cluster \
                else pad("%s%s%s" % (t.label, box, t.rst()), 6)

        def gpu_cols(gp):
            if not gp:
                return pad("%sGPU%s %s" % (t.label, t.rst(), t.dim() + "n/a" + t.rst()), 4 + gw + 5 + (0 if narrow else 22))
            u = gp.get("util") or 0.0
            core = "%sGPU%s %s %s%3.0f%%%s" % (t.label, t.rst(), self.gauge(u / 100.0, gw), t.value + t.bold(), u, t.rst())
            if narrow:
                return core
            return core + "%s %4.0f W %5.0f MHz %3.0fC%s" % (t.frame, gp.get("w") or 0, gp.get("mhz") or 0,
                                                           gp.get("c") or 0, t.rst())

        def ram_cols(used_mb, total_mb, avail_mb):
            if not total_mb:
                return pad("", 28)
            return pad("   %sRAM%s %3.0f/%3.0fG %s%4.1fG free%s"
                       % (t.label, t.rst(), used_mb / 1024.0, total_mb / 1024.0, t.frame, (avail_mb or 0) / 1024.0,
                          t.rst()), 28)

        tel = r1.get("telemetry")
        if tel == "no url":
            tel_s = "%srank 1: --rank1-url not set%s" % (t.dim(), t.rst())
        elif tel == "live":
            tel_s = "%s%.1f s ago%s" % (t.frame, r1.get("age") or 0.0, t.rst())
        elif tel == "stale":
            tel_s = "%ssample stale %.0f s%s" % (t.heat(0.4), r1.get("age") or 0.0, t.rst())
        elif tel == "unreachable":
            tel_s = "%srank 1 unreachable%s" % (t.heat(0.1), t.rst())
        else:
            tel_s = "%sconnecting...%s" % (t.dim(), t.rst())
        if r1.get("up") is False:
            state = ("%sRANK 1 DOWN - decode will stall%s" if cluster else "%sUNREACHABLE (asleep?)%s") % (t.heat(0.1), t.rst())
        elif r1.get("up") is True:
            cont = r1.get("container") or ""
            # "glm53-flash-tf Up 52 minutes" -> "up 52 minutes": the name says nothing the row does not
            st = cont.split(" ", 1)[1] if " " in cont else ""
            state = "%s%s%s" % (t.heat(0.9), (st[:1].lower() + st[1:]) if st else "up", t.rst())
        else:
            state = ""
        body = []
        g0 = (self.gpu or [None])[0]
        mem = self.memory or {}
        tot, avail = (mem.get("total") or 0) / 1024.0, (mem.get("avail") or 0) / 1024.0
        rss = ("%sengine rss %.1fG%s" % (t.frame, (mem.get("rss") or 0) / 1048576.0, t.rst())) if mem.get("rss") else ""
        body.append(name("rank 0", "this box") + gpu_cols(g0) + ram_cols(tot - avail, tot, avail) + "   " + rss)
        body.append(name("rank 1", "remote") + gpu_cols(r1.get("gpu"))
                    + ram_cols(r1.get("ramUsedMb"), r1.get("ramTotalMb"), r1.get("availMb"))
                    + "   " + "   ".join(x for x in (state, tel_s) if x))
        if cluster:
            tx, rx, pk, devs = roce_link()
            if tx is None:
                link = name("link", "CX7 ") + "%sNET%s %smeasuring...%s" % (t.label, t.rst(), t.dim(), t.rst())
            else:
                frac = (max(tx, rx) / pk) if pk else 0.0
                link = (name("link", "CX7 ") + "%sNET%s %s %sout %9s/s  in %9s/s%s"
                        % (t.label, t.rst(), self.gauge(frac, gw), t.value, si_bytes(tx), si_bytes(rx), t.rst()))
                if not narrow:
                    link += "   %speak %s/s . %s%s" % (t.frame, si_bytes(pk), "+".join(devs) or "no RoCE device", t.rst())
            hl = _TF_HEALTH if getattr(self, "tf", False) else {"at": None}
            if hl.get("at") is not None and time.monotonic() - hl["at"] < 10:
                it = hl.get("iteration_s") or 0.0
                if hl.get("ok") is False or hl.get("fatal"):
                    link += "   %sENGINE FATAL - restart both ranks%s" % (t.heat(0.05), t.rst())
                elif it >= 30:
                    link += "   %sSTALLED: iteration running %.0f s (ranks out of step?)%s" % (t.heat(0.1), it, t.rst())
                else:
                    link += "   %sin step%s %s(iteration %.1f s)%s" % (t.heat(0.9), t.rst(), t.frame, it, t.rst())
            body.append(link)
            if r1.get("run") is not None or r1.get("kv") is not None:      # the vLLM cluster's rank-1 /metrics
                bits = []
                if r1.get("run") is not None:
                    bits.append("req %d" % r1["run"])
                if r1.get("kv") is not None:
                    bits.append("kv %.0f%%" % (r1["kv"] * 100))
                if r1.get("gen") is not None:
                    bits.append("gen %s tok" % human(r1["gen"]))
                if r1.get("prompt") is not None:
                    bits.append("prompt %s tok" % human(r1["prompt"]))
                body.append(pad("", 14) + "%srank 1 worker: %s%s" % (t.frame, " . ".join(bits), t.rst()))
        return [clip(x, inner) for x in body]



    def lanes_panel(self, cur, width):
        """The optional gateway, the thing no upstream dashboard has: the
        two-lane queue that decides what reaches the engine, and the preemption
        machinery that protects the human's lane."""
        q, l = self.yzq, self.lanes
        if not q and not l:
            if config.LANES_URL:
                t = self.t
                return [clip("%sno lanes endpoint%s" % (t.frame, t.rst()), width - 2)]
            return None
        t = self.t
        hi = int((q or {}).get("high") or 0)
        lo = int((q or {}).get("low") or 0)
        par = int((q or {}).get("parallel") or 1)
        run = len((q or {}).get("running") or [])
        rl = [{"alta": "high", "bassa": "low"}.get(r.get("prio"), r.get("prio")) for r in ((q or {}).get("running") or [])]
        inf = ("high %d . low %d" % (rl.count("high"), rl.count("low"))) if rl else "-"   # 05-10: every running request, not the oldest
        free = (l or {}).get("free_slots")
        bits = ["%shigh%s %d queued . running %d/%d (%s)"
                % (t.head, t.rst(), hi, run, par, inf),
                "%slow %s %d queued" % (t.accent, t.rst(), lo)]
        if free is not None:
            bits.append("free slots %s" % free)
        for k, lab in (("beat_refused", "beat refused"),
                       ("beat_kill", "beat killed")):
            v = (q or {}).get(k)
            if v:
                bits.append("%s %s%d%s" % (lab, t.heat(0.3), int(v), t.rst()))
        return [clip(" . ".join(bits), width - 2)]



    def model_strip(self, width):
        """One line of what is actually loaded - the repo's model strip,
        reduced to the one model this box serves. Facts come from the engine
        cmdline, not from a config file that can rot."""
        i = self.info
        if not i.get("model") and not i.get("engine"):
            return None
        bits = []
        model = (i.get("model") or "?").split("/")[-1]
        bits.append(model)
        if i.get("quant"):
            bits.append(str(i["quant"]))
        if i.get("kvdtype"):
            bits.append("kv " + str(i["kvdtype"]))
        if i.get("tp"):
            bits.append("TP %s" % i["tp"])
        if i.get("seqs"):
            bits.append("seqs %s" % i["seqs"])
        if i.get("maxlen"):
            bits.append("ctx %s" % human(float(i["maxlen"])))
        if i.get("spec"):
            bits.append("%s k=%s" % (i["spec"], i.get("k") or "?"))
        t = self.t
        line = "  %s>%s %s%s%s" % (t.head, t.rst(), t.value,
                                   " . ".join(bits), t.rst())
        return [clip(line, width - 2)] if bits else None



    def expert_grid(self, width, rows_avail=6, hot_fn=None):
        """Port of render.rs render_experts(): the expert heatmap.

        One block per group of experts per layer row; block heat = hottest
        expert in the group; two layers share a row through the ▀ half block
        (fg = upper layer, bg = lower layer), exactly as upstream. Colour is
        the HEAT ramp by TIME since routed - white now, cooling to dark.
        Identities come from the same deterministic stand-in router as
        fade.rs (real routing needs the server /experts patch), and the
        title says `simulated`, like the repo.
        """
        t = self.t
        arch = self.model_arch()
        if not arch:
            return []
        n_l = len(self.fade_expert) or arch["n"]                    # fade_experts sized it (MTP row included)
        n_e = len(self.fade_expert[0]) if self.fade_expert and self.fade_expert[0] else min(arch["experts"], 256)
        w = max(8, width - 2)
        cols = max(1, w - 4)                 # 4 cells for the layer label
        per_block = max(1, (n_e + cols - 1) // cols)
        n_blocks = max(1, (n_e + per_block - 1) // per_block)
        bw = max(1, min(4, cols // n_blocks))
        gap = 1 if (bw >= 2 and (bw + 1) * n_blocks <= cols) else 0
        rows_avail = max(1, rows_avail)
        layers_per_slot = 1 if n_l <= rows_avail \
            else max(1, (n_l + 2 * rows_avail - 1) // (2 * rows_avail))
        half = n_l > rows_avail       # was `layers_per_slot > 1`: 46 layers in 23 rows has lps 1 and showed 0-22 only
        n_slots = (n_l + layers_per_slot - 1) // layers_per_slot
        text_rows = (n_slots + 1) // 2 if half else n_slots

        def heat(slot, b):
            l0 = slot * layers_per_slot
            l1 = min(l0 + layers_per_slot, n_l)
            e0 = b * per_block
            e1 = min(e0 + per_block, n_e)
            m = 0.0
            for l in range(l0, l1):
                row = self.fade_expert[l] if l < len(self.fade_expert) else None
                if not row:
                    continue
                for e in range(e0, e1):
                    if e < len(row) and row[e] > m:
                        m = row[e]
            return m

        body = []
        for r in range(min(text_rows, rows_avail)):
            top = r * 2 if half else r
            bot = top + 1
            lab = "%3d " % (top * layers_per_slot)
            cells = [lab] if t.plain else [t.frame + lab + t.rst()]
            for b in range(n_blocks):
                ht = heat(top, b)
                if t.plain:
                    cells.append(" .:-=+*#%@"[min(9, int(ht * 9.999))]
                                 * bw if ht > 0 else " " * bw)
                elif half:
                    hb = heat(bot, b) if bot < n_slots else 0.0
                    cells.append(self.ramp_heat(ht)
                                 + self.ramp_heat_bg(hb) + "▀" * bw + RST)
                else:
                    cells.append(self.ramp_heat_bg(ht) + " " * bw + RST)
                if gap and b + 1 < n_blocks:
                    cells.append(" ")
            body.append(clip("".join(cells), width - 2))
        return body



    def totals_rows(self, cur):
        """Every token the engine has handled since it started: prompt in (and how much of it came from cache),
        generated out, requests. Short units: the box is ~44 columns."""
        t = self.t

        def sh(n):
            n = float(n or 0)
            for div, suf in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
                if n >= div:
                    return "%.1f%s" % (n / div, suf)
            return "%d" % n
        reqs = (" . %d req" % cur.nreq) if cur.nreq is not None else ""
        out = "%s%-13s%s %s%s%s%s%s%s" % (t.label, "tokens out", t.rst(), t.value, sh(cur.gen), t.rst(),
                                          t.frame, reqs, t.rst())
        if cur.tot_in is None:
            return [out]
        cached = (" (%s cached)" % sh(cur.tot_cached)) if cur.tot_cached is not None else ""
        rows = ["%s%-13s%s %s%s%s%s%s%s" % (t.label, "tokens in", t.rst(), t.value, sh(cur.tot_in), t.rst(),
                                            t.frame, cached, t.rst()), out]
        if cur.all_n:
            since = time.strftime("%d-%m", time.localtime(cur.all_since)) if cur.all_since else "?"
            rows.append("%s%-13s%s %s%s in . %s out%s%s . %d req%s"
                        % (t.label, "since " + since, t.rst(), t.value, sh(cur.all_in), sh(cur.all_out), t.rst(),
                           t.frame, cur.all_n, t.rst()))
        rows += self.hermes_rows(sh)
        return rows



    @staticmethod
    def short(n):
        n = float(n or 0)
        for div, suf in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
            if n >= div:
                return ("%.0f%s" if n / div >= 100 else "%.1f%s") % (n / div, suf)
        return "%d" % n



    def heat_stats(self, days, today):
        """The numbers next to the map, stacked: the best days in order, then today, the last 7 days, the mean of
        the active days and the total."""
        import datetime as _dt
        md = lambda d: "%s %s" % (self.MONTHS[int(d[5:7]) - 1], d[8:10])
        top = sorted(((v, d) for d, v in days.items() if v > 0), reverse=True)[:3]
        rows = [("#%d" % (i + 1), v, md(d)) for i, (v, d) in enumerate(top)]
        week = sum(days.get((today - _dt.timedelta(days=i)).isoformat(), 0.0) for i in range(7))
        active = [v for v in days.values() if v > 0]
        rows += [("today", days.get(today.isoformat(), 0.0), ""), ("7 days", week, ""),
                 ("avg/day", sum(active) / len(active) if active else 0.0, "%d active" % len(active)),
                 ("total", sum(active), "since " + md(min(days)))]
        return rows



    def heatmap_rows(self, width, days=None, today=None):
        """Generated tokens per day as squares, like GitHub's contributions: one column a week (Mo-Su top down), the
        months on top over the week holding their 1st, the last column this week. Five levels, dithered and coloured:
        none, then the quartiles of the days with traffic. The stats stand on the right, stacked (below the legend
        when the box is too narrow for both). As many weeks as fit, the most recent."""
        import datetime as _dt
        days = hermes_days() if days is None else days
        if not days:
            return []
        t = self.t
        today = today or _dt.date.today()
        first = min(_dt.date.fromisoformat(d) for d in days)
        mon_now = today - _dt.timedelta(days=today.weekday())
        need = (mon_now - (first - _dt.timedelta(days=first.weekday()))).days // 7 + 1
        stats = self.heat_stats(days, today)
        cells = [("%-7s %6s %s" % (lab, self.short(v), note)).rstrip() for lab, v, note in stats]
        statw = max(len(c) for c in cells)
        inner = width - 2
        GAP = 8                                           # between the grid and the stats, which sit on the right edge
        side = inner >= 3 + 3 * 8 + GAP + statw           # at least 8 weeks next to the stats, else stack them
        W = max(1, min(need, (inner - 3 - ((GAP + statw) if side else 0)) // 3))
        mons = [mon_now - _dt.timedelta(weeks=W - 1 - k) for k in range(W)]
        vals = sorted(v for v in days.values() if v > 0)
        q = [vals[min(len(vals) - 1, int(len(vals) * f))] for f in (0.25, 0.5, 0.75)] if vals else [0, 0, 0]

        def level(v):
            return 0 if v <= 0 else 1 + sum(v > x for x in q)

        def cell(lv):
            # two glyphs a day: a terminal cell is about twice as tall as wide, so two side by side make a square
            ch = DITHER[lv] * 2
            if t.plain:
                return ch
            return (t.frame if lv == 0 else t.heat(lv / 4.0)) + ch + t.rst()

        gw = 3 + 3 * W                                    # the grid's visible width, labels included
        head = [" "] * (gw + 4)                           # a label may run past the last column (this week's month)
        last = -9
        for k, m in enumerate(mons):
            starts = [m + _dt.timedelta(days=i) for i in range(7) if (m + _dt.timedelta(days=i)).day == 1]
            if starts and 3 + 3 * k - last >= 4:
                for j, ch in enumerate(self.MONTHS[starts[0].month - 1]):
                    head[3 + 3 * k + j] = ch
                last = 3 + 3 * k
        rows = [t.label + "".join(head).rstrip() + t.rst()]
        for wd in range(7):
            line = t.label + self.WEEKDAYS[wd] + t.rst() + " "
            for m in mons:
                d = m + _dt.timedelta(days=wd)
                line += (cell(level(days.get(d.isoformat(), 0.0))) if d <= today else "  ") + " "
            if side and wd < len(stats):
                lab, v, note = stats[wd]
                col = t.value if lab != "today" else (t.heat(level(v) / 4.0) if v > 0 else t.frame)
                st = (t.label + "%-7s" % lab + t.rst() + " " + col + "%6s" % self.short(v) + t.rst()
                      + ((" " + t.frame + note + t.rst()) if note else ""))
                line = pad(line, inner - statw - 1) + st  # right-aligned block, one column off the right edge
            rows.append(line.rstrip())
        rows.append("")
        rows.append("%sless%s %s %smore%s" % (t.frame, t.rst(), " ".join(cell(lv) for lv in range(5)), t.frame, t.rst()))
        if not side:
            rows.append("")
            rows += [t.label + c[:8] + t.rst() + t.value + c[8:] + t.rst() for c in cells]
        return rows



    def hermes_rows(self, sh, top=3):
        """The harness total over all the models it used (hermes_usage), and the top ones by tokens."""
        u = hermes_usage()
        if not u:
            return []
        t = self.t
        since = time.strftime("%d-%m", time.localtime(u["since"])) if u["since"] else "?"
        rows = ["%s%-13s%s %s%s in%s %s(%s cached)%s %s. %s out%s"
                % (t.label, "all models", t.rst(), t.value, sh(u["in"] + u["cached"]), t.rst(), t.frame,
                   sh(u["cached"]), t.rst(), t.value, sh(u["out"]), t.rst()),
                "%s%-13s%s %s{:,} calls . %d models . since %s%s".format(u["calls"])
                % (t.label, "", t.rst(), t.frame, len(u["models"]), since, t.rst())]
        for m, _c, i, cr, o, _r, _f in u["models"][:top]:
            rows.append("%s  %-24s%s %s%s in . %s out%s" % (t.frame, m[:24], t.rst(), t.value, sh(i + cr), sh(o), t.rst()))
        return rows



    @staticmethod
    def tf_frac(tau, k=7):
        """Tokens a round as a fraction of a full draft block (k drafts + the bonus token): only a colour scale."""
        return min(1.0, (tau + 1.0) / (max(1, k) + 1.0))



    def fill_row(self, cur):
        """TensorFold (newer kit): the prompts in prefill, with the longest one's progress if /health says so (y3)."""
        t = self.t
        n, pos, ln = cur.tf_fill
        if not n:
            return "%s%-13s%s %s-%s" % (t.label, "prefill", t.rst(), t.frame, t.rst())
        if ln:
            return ("%s%-13s%s %s%s / %s%s  %s%.0f%%%s  %s%d stream%s"
                    % (t.label, "prefill", t.rst(), t.value, human(pos), human(ln), t.rst(),
                       t.heat(pos / ln), 100.0 * pos / ln, t.rst(), t.frame, int(n), t.rst()))
        return "%s%-13s%s %s%d stream%s" % (t.label, "prefill", t.rst(), t.heat(0.7), int(n), t.rst())



    def tf_line(self, tau, bw, k=7):
        t = self.t
        f = self.tf_frac(tau, k)
        return ("%s◆ TOKENS/ROUND%s %s  %s%.2f%s  %stau=%.2f/%d%s"
                % (t.title, t.rst(), self.bar(f, bw), t.heat(f), tau + 1, t.rst(), t.label, tau, k, t.rst()))



    @staticmethod
    def verdict(alpha, tau):
        if alpha >= 0.60:
            return "Excellent", "draft model is well aligned"
        if alpha >= 0.45:
            return "Good", "tau=%.2f free tokens per verify step" % tau
        if alpha >= 0.30:
            return "Fair", "~%.0f%% of drafted compute is thrown away" % ((1 - alpha) * 100)
        if alpha > 0:
            return "Poor", "the draft loses more than it wins - try a lower k"
        return "No speculation", "no draft model, or engine idle since boot"



    def waiting_row(self, cur):
        """What is actually outstanding, which /metrics cannot tell you.

        The engine only ever sees the one request the gateway has let through,
        so vllm:num_requests_waiting is a constant 0 and says nothing about the
        queue. When the gateway answers, count ITS lanes instead and name them:
        high = the conversation you are typing in plus its tool chain,
        low = heartbeat wakes and the background ingest."""
        t = self.t
        q = self.yzq
        if not q:
            return "%s%-13s%s %s%d%s" % (t.label, "waiting", t.rst(),
                                         t.heat(0.3) if cur.wait else t.frame,
                                         cur.wait, t.rst())
        hi, lo = int(q.get("high") or 0), int(q.get("low") or 0)
        n = hi + lo
        # heat: an empty queue is fine, one near YZQ_MAXQ (16) is not
        col = t.heat(1.0 - min(1.0, n / 8.0)) if n else t.frame
        # 31-08: it used to read "high 2 . low 0" and was read as "I have two
        # high sessions open". These are requests IN QUEUE, and two high ones
        # together are usually the same conversation in two calls (the turn +
        # an auxiliary call: compression, title, memory flush). The word
        # "queued" costs 7 columns and removes the ambiguity.
        par = int(q.get("parallel") or 1)
        slots = ("   %s%d/%d slot%s" % (t.frame, len(q.get("running") or []), par, t.rst())
                 if par > 1 else "")
        # 19-09: the gateway queue is NOT the whole story. The engine (vLLM, newer
        # kit) may hold a request the gateway has already admitted: "capacity" =
        # max_num_seqs or a full KV pool, "deferred" = prefill denied while
        # another decodes (GLM53_MIXED_PREFILL_CHUNK=skip). Measured 19-09: 2/2
        # slots at the gateway, engine running 1 / waiting 1, human 335 s to the
        # first token.
        eng = ""
        if cur.wait:
            why = []
            if getattr(cur, "wait_cap", 0):
                why.append("capacity %d" % cur.wait_cap)
            if getattr(cur, "wait_def", 0):
                why.append("deferred %d" % cur.wait_def)
            eng = "%s%-13s%s %s! %d held by the engine (%s)%s" % (
                t.label, "", t.rst(), t.heat(0.2), cur.wait, " . ".join(why) or "?", t.rst())
        row = "%s%-13s%s %s%d%s   %squeued: high %d . low %d%s%s" % (
            t.label, "waiting", t.rst(), col, n, t.rst(), t.frame, hi, lo, t.rst(), slots)
        return (row, eng) if eng else row



    def running_lane(self):
        """Which lane the engine is actually chewing on, or nothing if the
        gateway is not in front of it."""
        q = self.yzq or {}
        # 05-10: with parallel > 1 several requests run at once; "inflight" is only the oldest of them, usually
        # a beat, so a high turn running beside it showed as "low". Any high request running = high.
        lanes = [{"alta": "high", "bassa": "low"}.get(r.get("prio"), r.get("prio")) for r in (q.get("running") or [])]
        if lanes:
            return "high" if "high" in lanes else lanes[0]
        prio = (q.get("inflight") or {}).get("prio")
        return {"alta": "high", "bassa": "low"}.get(prio, prio)      # the gateway names its lanes in Italian



    def slot_panel(self, cur):
        out = {"kv": None, "prefix": None, "ctxtxt": "-", "stale": False}
        if not cur.nslots or not cur.nctx:
            return out
        out["kv"] = min(1.0, cur.ptok / cur.nctx)
        out["prefix"] = (cur.pcache / cur.ptok) if cur.ptok else None
        out["ctxtxt"] = "%s / %s" % (human(cur.ptok), human(cur.nctx))
        return out



    def model_arch(self):
        """Cached per-layer topology of the running engine (or None)."""
        now = time.time()
        if self.arch is None and now >= getattr(self, "arch_retry", 0.0):
            self.arch_path = model_config_path()
            if self.arch_path:
                self.arch = load_model_arch(self.arch_path)
            if self.arch is None:
                self.arch_retry = now + 30.0
        return self.arch



    def _engine_aux(self):
        """(vision_active, weights_offloaded, kv_usage) from one /metrics read."""
        try:
            txt = http_get("http://127.0.0.1:%d/metrics" % self.port,
                           timeout=2, tries=1)
            if not txt:
                return False, False, None
            mx = parse_metrics(txt)
            q = g(mx, "vllm:mm_cache_queries_total")
            now = time.time()
            vact = False
            if q is not None:
                if self.mm_prev is not None and q > self.mm_prev \
                        and now - self.mm_t < 120:
                    vact = True
                self.mm_prev, self.mm_t = q, now
            woff = bool(gl(mx, "vllm:engine_sleep_state",
                           'sleep_state="weights_offloaded"'))
            kv = g(mx, "vllm:kv_cache_usage_perc")
            return vact, woff, kv
        except Exception:
            return False, False, None



    def layer_panel(self, width, free):
        """Layer-activity visualizer: the model's REAL topology as coloured boxes.

        Read-only - it reads /metrics and the engine's config.json and never
        touches the engine, so it cannot perturb the LLM's output. Each layer is
        a coloured box; the colour encodes work: hot = actively computing,
        dim = idle, and the DSpark target layers are painted by their
        speculative-decoding acceptance. This is the panel that sits where the
        log is inside the spec view (toggled with L), not a separate screen.
        """
        t = self.t
        res = self.step()
        if res is None:
            return []
        cur, live = res
        arch = self.model_arch()
        if not arch:
            return [t.dim() + "  no model config found - layer topology unavailable"
                    + t.rst()]
        n = arch["n"]
        target = set(arch["target"])
        comp = arch["compress"]
        working = cur.busy and live["gen"] > 0
        alpha = cur.acc / cur.draft if cur.draft > 0 else 0.0
        # text colour that stays readable on a coloured box: dark on bright bg
        tcol = "" if t.plain else fg(0)
        cells = []
        for i in range(n):
            if i in target:
                heat = alpha if working else 0.0
                mark = t.accent + "★" + t.rst()
            else:
                heat = 0.85 if working else 0.0
                mark = ""
            kv = comp.get(i)
            kvmark = ("~" if kv and kv > 1 else "·" if kv == 0 else "")
            cells.append(self.ramp_heat_bg(heat) + tcol
                         + pad("L%02d%s%s" % (i, kvmark, mark), 5) + RST)
        cw = 5
        per_row = max(1, (width - 4) // cw)
        body = []
        for i in range(0, len(cells), per_row):
            body.append("".join(cells[i:i + per_row]))
        legend = ("  %s  %s working   %s  %s idle   %s★%s DSpark target   "
                  "%s~%s KV-compressed   %s·%s full-KV"
                  % (self.ramp_heat_bg(0.85), t.rst(),
                     self.ramp_heat_bg(0.0), t.rst(),
                     t.accent, t.rst(), t.label, t.rst(), t.label, t.rst()))
        body.append(legend)
        body.append(t.dim()
                     + "  per-expert routing heat needs enable_return_routed_experts"
                     + " (boot flag)" + t.rst())
        if free and len(body) > free:
            body = body[:free]
        return body

