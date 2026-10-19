"""Full-screen frame composition: full, perf, pipes, layers, spec views."""
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

from ..fetch.logtail import active_log, tail
from ..text import ago, clip, fmt_hms, human, pad, vis_len


class FramesMixin:
    def frame(self, width, height=None):
        if self.mode == "layers":
            return self.frame_layers(width, height)
        if self.view == "pipes":
            return self.frame_pipes(width, height)
        if self.view == "perf":
            return self.frame_perf(width, height)
        return self.frame_spec(width, height)



    def frame_perf(self, width, height=None):
        """Performance zoom (p): the throughput story at full width - big
        digits, the acceptance line, one very wide history block and the GPU
        row. The repo's `p` view, ported to what this box exposes."""
        t = self.t
        if height is None:
            height = shutil.get_terminal_size((100, 40)).lines
        width = max(20, width)
        res = self.step()
        if res is None:
            return [clip("%s  no engine on :%d%s" % (t.heat(0.0), self.port,
                                                     t.rst()), width)]
        cur, live = res
        gen = live["gen"]
        L = []
        A = L.append
        A(clip("%s◆%s %sPERFORMANCE ZOOM%s   %s( %s )%s   %sp = back%s"
               % (t.title, t.rst(), t.bold(), t.rst(), t.head,
                  self.info.get("engine") or cur.fam, t.rst(),
                  t.accent, t.rst()), width))
        A("")
        if not height or height >= 22:
            if t.plain:
                gcol = ""
            else:
                gcol = self.ramp_flow(min(1.0, gen / max(self.pk_gen, 0.5))) \
                    if gen else t.frame
            for r in self.big_digits(gen, max(18, width // 2 - 4), gcol):
                A(clip(r, width))
            stat = (self.pulse_head(int(time.time() * 2)) + " DECODE"
                    if cur.busy and gen > 0 else "  DECODE")
            bar = self.gauge(min(1.0, gen / max(self.pk_gen, 0.5))
                             if self.pk_gen else 0.0, 14)
            A(clip("%s%s%s %s%*.1f t/s%s  %s  %speak %*.1f%s  %s%s tok total%s"
                   % (t.label, stat, t.rst(), t.value + t.bold(), 6, gen,
                      t.rst(), bar, t.frame, 6, self.pk_gen, t.rst(),
                      t.label, human(cur.gen), t.rst()), width))
            A("")
        alpha = (self.w["alpha"] if self.w else
                 (cur.acc / cur.draft if cur.draft > 0 else 0.0))
        tau = (self.w["tau"] if self.w else
               (cur.acc / cur.drafts if cur.drafts > 0 else 0.0))
        if cur.tf_nodraft:
            A(clip(self.tf_line(tau, min(46, width - 40)), width))
        else:
            A(clip("%s◆ ACCEPTANCE%s %s  %s%.1f%%%s  %stau=%.2f%s"
                   % (t.title, t.rst(), self.bar(alpha, min(46, width - 40)),
                      t.heat(alpha), alpha * 100, t.rst(), t.label, tau, t.rst()),
                   width))
        A("")
        sw = max(10, width - 34)
        for row in self.box("history  (full width)", [
                self.hrow("gen t/s", self.h_gen, sw, "%.1f", t.heat(0.9),
                          "%d tick" % len(self.h_gen), fit=width - 2),
                self.hrow("prompt t/s", self.h_pf, sw, "%.0f", t.heat(0.7), "", fit=width - 2),
                self.hrow("accept %", self.h_acc, sw, "%.0f", t.heat(alpha),
                          "%d req" % len(self.h_acc), fit=width - 2),
                self.hrow("tau", self.h_tau, sw, "%.2f", t.head, "", fit=width - 2),
        ], width, t.head):
            A(clip(row, width))
        cl = self.cluster_panel(width)
        if cl:
            A("")
            for r in self.box(self.cluster_title(), cl, width, t.accent):
                A(clip(r, width))
            ln = self.lanes_panel(cur, width)
            if ln:
                for r in self.box("gateway lanes", ln, width, t.head):
                    A(clip(r, width))
        L = [clip(ln, width) for ln in L]
        if height and len(L) > height:
            L = L[:height]
        self.last_frame = L
        return L



    def frame_pipes(self, width, height=None):
        """Memory-pipeline zoom (b): the hops a token takes on this box, as
        vertical VU meters. The repo's six strips assume split RAM/VRAM; on
        GB10 unified memory they collapse, and what THIS box uniquely gates a
        token with is the gateway lane machinery and the rank-1 link - so those
        get the strips the repo spends on PCIe."""
        t = self.t
        if height is None:
            height = shutil.get_terminal_size((100, 40)).lines
        width = max(40, width)
        res = self.step()
        if res is None:
            return [clip("%s  no engine on :%d%s" % (t.heat(0.0), self.port,
                                                     t.rst()), width)]
        cur, live = res
        gen = live["gen"]
        vh = max(6, min(12, height - 14))
        sl = self.slot_panel(cur)
        kv = sl["kv"] if sl["kv"] is not None else 0.0
        gpu_u = ((self.gpu[0].get("util") or 0.0) / 100.0) if self.gpu else 0.0
        mem_f = (self.memory or {}).get("used_frac")
        dec_f = min(1.0, gen / max(self.pk_gen, 0.5)) if self.pk_gen else 0.0
        pf_f = min(1.0, live["pf"] / max(self.pk_pf, 0.5)) if self.pk_pf else 0.0
        r1 = self.rank1
        link_f = 1.0 if (r1 and r1.get("up")) else (0.0 if r1 else None)
        stages = [
            ("GPU", gpu_u, gpu_u, "%5.0f%%" % (gpu_u * 100)),
            ("MEM", mem_f if mem_f is not None else 0.0,
             mem_f if mem_f is not None else 0.0,
             ("%5.0f%%" % (mem_f * 100)) if mem_f is not None else "  n/a"),
            ("CTX", kv, kv, "%5.0f%%" % (kv * 100)),
            ("DECODE", dec_f, dec_f, "%5.1f" % gen),
            ("PREFILL", pf_f, pf_f, "%5.0f" % live["pf"]),
        ]
        if link_f is not None:
            stages.append(("RANK1", link_f, link_f,
                           "link" if link_f else "down"))
        meters = [self.vmeter(f, h, vh, 5) for _, f, h, _ in stages]
        L = []
        A = L.append
        A(clip("%s◆%s %sTOKEN PIPELINE%s   %sb = back%s"
               % (t.title, t.rst(), t.bold(), t.rst(), t.accent, t.rst()),
               width))
        A("")
        colw = 8
        row = []
        for i in range(vh):
            row = []
            for m in meters:
                row.append(pad(m[i], colw - 2))
            A(clip("".join(row), width))
        lab = []
        for name, _, _, val in stages:
            lab.append(pad("%s%-6s%s%s" % (t.label, name, t.rst(),
                                           t.value + val + t.rst()), colw))
        A(clip("".join(lab), width))
        A("")
        verdict = self.pipe_verdict(cur, gpu_u, kv, r1)
        A(clip("%s* %s%s" % (t.heat(0.8), verdict, t.rst()), width))
        ln = self.lanes_panel(cur, width)
        if ln:
            A("")
            for r in self.box("gateway lanes", ln, width, t.accent):
                A(clip(r, width))
        L = [clip(x, width) for x in L]
        if height and len(L) > height:
            L = L[:height]
        self.last_frame = L
        return L



    def pipe_verdict(self, cur, gpu_u, kv, r1):
        """The repo's bottleneck rule chain, adapted to this box: with the
        weights resident in unified memory there is no PCIe hop, so the
        failure modes that remain are the partner link, the KV pool, compute
        and latency."""
        if r1 is not None and not r1.get("up") \
                and self.info.get("nnodes") in ("2", 2):
            return "rank 1 unreachable - tensor-parallel decode stalls"
        if cur.busy and gpu_u < 0.2:
            return ("GPU idle while decoding - latency-bound "
                    "(small active batch, bandwidth-bound decode)")
        if gpu_u > 0.85:
            return "GPU compute-bound at %.0f%% utilisation" % (gpu_u * 100)
        if kv > 0.9:
            return "KV pool nearly full - prefill will be deferred"
        if cur.busy:
            return "no hop saturated - the gap is per-token latency"
        return "engine idle - nothing to bound"



    def frame_layers(self, width, height=None):
        """Layer-activity panel: the model's REAL topology as a heat grid.

        Read-only - it reads /metrics and the engine's config.json and never
        touches the engine, so it cannot perturb the LLM's output. The grid is
        built from the actual model (num layers, experts, DSpark target layers,
        KV compression ratios). Live "work" colour comes from the same Sample
        the spec panel uses: global generation activity plus the speculative-
        decode acceptance on the DSpark target layers.
        """
        t = self.t
        if height is None:
            height = shutil.get_terminal_size((100, 40)).lines
        width = max(20, width)
        res = self.step()
        if res is None:
            return [clip("%s%s  x no engine on :%d%s" % (t.heat(0.0), t.bold(),
                                                          self.port, t.rst()), width),
                    clip("%s  (g = back to spec, q = quit)%s" % (t.dim(), t.rst()),
                         width)]
        cur, live = res
        fdt = self.fade_dt()
        self.fade_step("gen", live["gen"] > 0, fdt)
        arch = self.model_arch()
        vact, woff, kv_usage = self._engine_aux()

        L = []
        A = L.append

        # ---- header
        title = "MODEL LAYER ACTIVITY"
        eng = self.info.get("engine") or cur.fam
        head = ("%s◆%s %s%s%s%s %s────%s %s( %s )%s   %s◆ layer grid ◆%s"
                % (t.title, t.rst(), t.bold(), t.title, title, t.rst(), t.frame,
                   t.rst(), t.head, eng, t.rst(), t.accent, t.rst()))
        right = ("%s%s%s  %s│%s  %s%s tok%s  %s│%s  %sg=spec%s"
                 % (t.value, fmt_hms(self.info.get("up")), t.rst(), t.frame,
                    t.rst(), t.label, human(cur.gen), t.rst(), t.frame, t.rst(),
                    t.head, t.rst()))
        if vis_len(head) + vis_len(right) + 2 > width:
            head = "%s◆%s %s%s%s%s" % (t.title, t.rst(), t.bold(), t.title,
                                            title, t.rst())
        A(pad(head, max(0, width - vis_len(right))) + right)

        # ---- model strip: what is loaded, straight from the engine cmdline
        strip = self.model_strip(width)
        if strip:
            A(strip[0])

        # ---- engine activity + speculative summary
        if cur.busy:
            act = "%s%s* WORKING%s" % (t.bold(), t.heat(0.9), t.rst())
            if cur.run:
                act += " %s%d req%s" % (t.label, int(cur.run), t.rst())
        else:
            idle = None if self.act_ts is None else max(0.0, time.time() - self.act_ts)
            act = "%s. idle%s %s%s%s" % (t.dim(), t.rst(), t.value,
                                         fmt_hms(idle) if idle is not None else "?",
                                         t.rst())
        A(act)
        alpha = cur.acc / cur.draft if cur.draft > 0 else 0.0
        tau = cur.acc / cur.drafts if cur.drafts > 0 else 0.0
        tgt = ",".join(str(x) for x in (arch["target"] if arch else []))
        spec = ("  spec: draft→L%s  accept %s%.0f%%%s  tau %s%.2f%s"
                % (tgt, t.heat(alpha), alpha * 100, t.rst(), t.label, tau, t.rst()))
        if cur.tf_nodraft:
            spec = "  spec: %s%.2f token/round%s (TensorFold: no drafted-token counter)" % (t.label, tau + 1, t.rst())
        A(spec)
        A("")

        if not arch:
            A(clip("%s  no model config found - layer topology unavailable%s"
                   % (t.dim(), t.rst()), width))
            A(clip("%s  (the layer grid needs the engine's config.json)%s"
                   % (t.dim(), t.rst()), width))
            L = [clip(ln, width) for ln in L]
            if height and len(L) > height:
                L = L[:height]
            self.last_frame = L
            return L

        n = arch["n"]
        target = set(arch["target"])
        comp = arch["compress"]
        working = cur.busy and live["gen"] > 0
        acc = cur.acc / cur.draft if cur.draft > 0 else 0.0

        # ---- transformer layer grid
        cw = 6
        per_row = max(1, (width - 2) // cw)
        cells = []
        for i in range(n):
            if i in target:
                col = t.heat(acc) if working else t.dim()
                mark = t.accent + "★" + t.rst()
            else:
                col = self.ramp_heat(max(0.35, self.fade.get("gen", 0.0))) \
                    if working else t.dim()
                mark = ""
            kv = comp.get(i)
            kvmark = ("·" if kv == 0 else "~" if kv and kv > 1 else "")
            cells.append("%sL%02d%s%s%s" % (col, i, t.rst(), kvmark, mark))
        body = []
        for i in range(0, len(cells), per_row):
            row = "".join(pad(c, cw) for c in cells[i:i + per_row])
            body.append(clip(row, width - 2))
        for r in self.box("transformer layers  (%d · MoE · %d experts, %d active/tok)"
                          % (n, arch["experts"], arch["topk"]), body, width, t.accent):
            A(clip(r, width))
        A("")

        # ---- auxiliary modules: vision / nextn / shared
        aux_body = []
        if arch["vision"]:
            vcol = t.heat(0.9) if vact else t.dim()
            aux_body.append("%s█%s vision encoder  %d layers  %s"
                            % (vcol, t.rst(), arch["vision"],
                               "ACTIVE" if vact else "idle (no image)"))
        if arch["nextn"]:
            aux_body.append("%s█%s nextn / MTP  %d layer"
                            % (t.heat(0.9) if working else t.dim(), t.rst(),
                               arch["nextn"]))
        if arch["shared"]:
            aux_body.append("%s█%s shared experts  %d per layer"
                            % (t.heat(0.6) if working else t.dim(), t.rst(),
                               arch["shared"]))
        for r in self.box("auxiliary modules", aux_body, width, t.head):
            A(clip(r, width))
        A("")

        # ---- engine state line
        sl = self.slot_panel(cur)
        kv = sl["kv"] if sl["kv"] is not None else kv_usage
        ctxtxt = sl["ctxtxt"]
        if ctxtxt == "-" and cur.ctxmax:
            ctxtxt = "up to %s" % human(cur.ctxmax)
        A(clip("%s  kv cache %s%s   weights %s%s%s   ctx %s%s%s"
               % (t.label, self.pct(kv, 8), t.rst(), t.label,
                  "OFFLOADED" if woff else "loaded", t.rst(), t.value,
                  ctxtxt, t.rst()), width))
        A("")

        # ---- legend
        legend = ("  %s█%s working   %s█%s idle   %s★%s DSpark target   "
                  "%s~%s KV-compressed   %s·%s full-KV"
                  % (t.heat(0.9), t.rst(), t.dim(), t.rst(), t.accent, t.rst(),
                     t.label, t.rst(), t.label, t.rst()))
        A(legend)
        A(clip("%s  per-expert routing heat needs enable_return_routed_experts "
               "(boot flag) - not collected here%s" % (t.dim(), t.rst()), width))

        L = [clip(ln, width) for ln in L]
        if height and len(L) > height:
            L = L[:height]
        self.last_frame = L
        return L



    def frame_spec(self, width, height=None):
        t = self.t
        if height is None:
            height = shutil.get_terminal_size((100, 40)).lines
        width = max(20, width)
        res = self.step()
        if res is None:
            self.miss += 1
            age = self.miss * self.interval
            if self.last_frame and age < 30:
                return [clip("%s/metrics silent for %.0fs - panel frozen%s"
                             % (t.heat(0.1), age, t.rst()), width)] + self.last_frame[1:]
            return [clip("%s%s  x no engine on :%d - /metrics and /props silent%s"
                         % (t.heat(0.0), t.bold(), self.port, t.rst()), width),
                    clip("%s  (check the engine is running on this port, or start it)%s"
                         % (t.dim(), t.rst()), width)]
        self.miss = 0
        cur, live = res

        # llm-visuals smoothing: one fade tick per frame + peak-hold on gen
        fdt = self.fade_dt()
        self.fade_step("gen", live["gen"] > 0, fdt)
        self.fade_step("pf", live["pf"] > 0, fdt)
        gen = live["gen"]
        if gen >= self.pk_gen:
            self.pk_gen, self.pk_at = gen, time.time()
        elif time.time() - self.pk_at > 2.0:
            self.pk_gen = max(gen, self.pk_gen - 0.02 * self.pk_gen)

        s_alpha = cur.acc / cur.draft if cur.draft > 0 else 0.0
        s_tau = cur.acc / cur.drafts if cur.drafts > 0 else 0.0
        s_dwin = cur.draft / cur.drafts if cur.drafts > 0 else 0.0
        w = self.w
        alpha = w["alpha"] if w else s_alpha
        tau = w["tau"] if w else s_tau
        dwin = w["dwin"] if w else s_dwin
        posn = w["pos"] if w else cur.pos
        posd = w["drafts"] if w else cur.drafts
        if not any(posn):
            posn, posd = cur.pos, cur.drafts
        src = ("window, %s" % ago(time.time() - self.w_at)) if w else "session"

        k = self.info.get("k")
        try:
            k = int(k)
        except (TypeError, ValueError):
            k = int(round(s_dwin)) or len(cur.pos)

        gen = live["gen"]
        steps = gen / (tau + 1.0) if tau > 0 else 0.0
        acc_ts, draft_ts = steps * tau, steps * dwin

        sl = self.slot_panel(cur)
        if cur.busy and sl["prefix"]:
            self.last_prefix = sl["prefix"]
        elif not cur.busy and self.last_prefix is not None:
            # An idle slot resets n_prompt_tokens_cache to 0 and the panel showed
            # a red 0% on a perfectly healthy engine. Keep the last true value.
            sl["prefix"], sl["stale"] = self.last_prefix, True

        L = []
        A = L.append

        # ---- header (degrades: full title -> short -> just the mark)
        title = ("SPECULATIVE DECODING MONITOR" if width >= 104
                 else "SPEC DECODING" if width >= 80 else "SPEC")
        eng = self.info.get("engine") or cur.fam
        head = ("%s◆%s %s%s%s%s %s────%s %s( %s )%s   %s◆ suzu ◆%s"
                % (t.title, t.rst(), t.bold(), t.title, title, t.rst(), t.frame,
                   t.rst(), t.head, eng, t.rst(), t.accent, t.rst()))
        if width >= 96:
            right = ("%s%s%s  %s│%s  %s%s tok%s  %s│%s  %sk=%d%s  %s%s%s"
                     % (t.value, fmt_hms(self.info.get("up")), t.rst(), t.frame,
                        t.rst(), t.label, human(cur.gen), t.rst(), t.frame, t.rst(),
                        t.heat(0.8), k, t.rst(), t.frame, t.name, t.rst())
                     + ("  " + t.legend() if width >= 112 else ""))
        else:
            right = "%s%s%s %sk=%d%s" % (t.value, fmt_hms(self.info.get("up")),
                                         t.rst(), t.heat(0.8), k, t.rst())
        if vis_len(head) + vis_len(right) + 2 > width:
            head = "%s◆%s %s%s%s%s" % (t.title, t.rst(), t.bold(), t.title,
                                            title, t.rst())
        A(pad(head, max(0, width - vis_len(right))) + right)

        # ---- model strip: what is loaded, straight from the engine cmdline
        strip = self.model_strip(width)
        if strip:
            A(strip[0])

        draft = self.info.get("draft") or "(no draft model)"
        if len(draft) > max(8, width - 24):
            draft = "..." + draft[-(max(8, width - 27)):]
        dline = ("  %s>%s %s%s%s  %s<-%s %s%s%s"
                 % (t.head, t.rst(), t.head, draft, t.rst(), t.frame, t.rst(),
                    t.label, self.info.get("spec") or "-", t.rst()))

        # Engine activity, right of the draft line. WORKING while tokens are
        # actually moving; otherwise how long the engine has been quiet. This is
        # the field that answers "is it chewing on something or has it been
        # sitting there since dawn" without reading the server log by hand.
        if cur.busy:
            act = "%s%s* WORKING%s" % (t.bold(), t.heat(0.9), t.rst())
            if cur.run:
                act += " %s%d req%s" % (t.label, int(cur.run), t.rst())
        else:
            idle = None if self.act_ts is None else max(0.0, time.time() - self.act_ts)
            act = "%s. idle%s %s%s%s" % (t.dim(), t.rst(), t.value,
                                         fmt_hms(idle) if idle is not None else "?",
                                         t.rst())
        if vis_len(dline) + vis_len(act) + 3 <= width:
            dline = pad(dline, max(0, width - vis_len(act) - 1)) + act
        A(dline)
        A("")

        # ---- big decode readout (llm-visuals: tall numerals on the FLOW ramp)
        if not height or height >= 26:
            if t.plain:
                gcol = t.value if gen else t.frame
            else:
                gcol = self.ramp_flow(min(1.0, gen / max(self.pk_gen, 0.5))) \
                    if gen else t.frame
            for r in self.big_digits(gen, max(18, width // 2 - 4),
                                     gcol if not t.plain else ""):
                A(clip(r, width))
            stat = (self.pulse_head(int(time.time() * 2)) + " DECODE"
                    if cur.busy and gen > 0 else "  DECODE")
            bar = self.gauge(min(1.0, gen / max(self.pk_gen, 0.5))
                             if self.pk_gen else 0.0, 14)
            A(clip("%s%s%s %s%*.1f t/s%s  %s  %speak %*.1f%s  %s%s tok total%s"
                   % (t.label, stat, t.rst(),
                      t.value + t.bold(), 6, gen, t.rst(),
                      bar, t.frame, 6, self.pk_gen, t.rst(),
                      t.label, human(cur.gen), t.rst()), width))
            A("")

        # ---- acceptance
        bw = max(10, min(46, width - 50))
        tfam = cur.tf_nodraft
        if tfam:
            A(clip(self.tf_line(tau, bw, k) + "  %s(%s)%s" % (t.frame, src, t.rst()), width))
            A(clip("%s* %.2f tokens a round%s %s- this TensorFold does not count drafted tokens: no acceptance %%, "
                   "tau from the decode rounds%s" % (t.heat(self.tf_frac(tau, k)), tau + 1, t.rst(), t.dim(), t.rst()),
                   width))
        else:
            A(clip("%s◆ ACCEPTANCE%s %s  %s%.1f%%%s  %stau=%.2f/%d%s  %s(%s)%s"
                   % (t.title, t.rst(), self.bar(alpha, bw), t.heat(alpha), alpha * 100,
                      t.rst(), t.label, tau, k, t.rst(), t.frame, src, t.rst()), width))
            verdict, why = self.verdict(alpha, tau)
            A(clip("%s* %s%s %s- %s%s" % (t.heat(alpha), verdict, t.rst(), t.dim(),
                                          why, t.rst()), width))
        A("")

        # ---- number grid: 3 columns wide, 1 column narrow
        na = "n/a"
        cells = [("tau acc length", "%.2f" % tau, t.heat(self.tf_frac(tau, k) if tfam else alpha)),
                 ("draft window", na if tfam else "%.1f" % dwin, t.frame if tfam else t.value),
                 ("spec tokens", "%d" % k, t.value),
                 ("accepted t/s", "%.1f" % acc_ts, t.heat(0.9)),
                 ("drafted t/s", na if tfam else "%.1f" % draft_ts, t.frame if tfam else t.label),
                 ("waste ratio", na if tfam else "%.1f%%" % ((1 - alpha) * 100), t.frame if tfam else t.heat(alpha)),
                 ("gen t/s", "%.1f" % gen, t.heat(0.9) if gen else t.frame),
                 ("prompt t/s", "%.0f" % live["pf"],
                  t.heat(0.7) if live["pf"] else t.frame),
                 ("verify step/s", "%.1f" % steps, t.value if steps else t.frame)]
        ncol = 3 if width >= self.NARROW else 1
        cw = (width - 2 * (ncol - 1)) // ncol
        for r in range(0, len(cells), ncol):
            line = ""
            for i in range(ncol):
                if r + i >= len(cells):
                    break
                lab, val, col = cells[r + i]
                line += pad("%s%-14s%s %s%s%s" % (t.label, lab, t.rst(),
                                                  col + t.bold(), val, t.rst()), cw)
                if i < ncol - 1:
                    line += t.frame + "│ " + t.rst()
            A(clip(line, width))
        A("")

        # ---- engine/session, history, rolling averages
        eng_body = [
            "%s%-13s%s %s" % (t.label, "kv cache", t.rst(), self.pct(sl["kv"], 10)),
            "%s%-13s%s %s%s" % (t.label, "prefix cache", t.rst(),
                                self.pct(sl["prefix"], 10, good_high=True),
                                (t.frame + " last" + t.rst()) if sl["stale"] else ""),
            "%s%-13s%s %s" % (t.label, "ttft last", t.rst(),
                              ("%s%.1f s%s  %s( %d seen )%s"
                               % (t.value, self.h_ttft[-1], t.rst(),
                                  t.frame, len(self.h_ttft), t.rst())
                               if self.h_ttft else
                               t.frame + "-" + t.rst())),
            "%s%-13s%s %s%d%s   %s%s%s%s" % (t.label, "running", t.rst(),
                                             t.heat(0.9) if cur.run else t.frame,
                                             cur.run, t.rst(), t.frame,
                                             "busy" if cur.busy else "idle",
                                             (" . " + self.running_lane())
                                             if self.running_lane() else "", t.rst()),
            *([self.fill_row(cur)] if cur.fam == "tensorfold" else []),
            *((lambda w: list(w) if isinstance(w, tuple) else [w])(self.waiting_row(cur))),
            "%s%-13s%s %s%s%s" % (t.label, "slot ctx", t.rst(), t.value,
                                  sl["ctxtxt"], t.rst()),
            "%s%-13s%s %s%s%s" % (t.label, "ctx max seen", t.rst(), t.label,
                                  human(cur.ctxmax), t.rst()),
            *self.totals_rows(cur),
            "",
            "%s%-13s%s %s%s%s" % (t.label, "accepted", t.rst(), t.value,
                                  human(cur.acc), t.rst()),
            "%s%-13s%s %s%s%s" % (t.label, "drafted", t.rst(), t.value,
                                  "n/a" if cur.tf_nodraft else human(cur.draft), t.rst()),
            ("%s%-13s%s %s%.2f token/round%s  %stau %.2f%s" % (t.label, "session", t.rst(), t.value, s_tau + 1,
                                                             t.rst(), t.frame, s_tau, t.rst())
             if cur.tf_nodraft else
             "%s%-13s%s %s%.1f%%%s  %stau %.2f%s" % (t.label, "session a", t.rst(),
                                                     t.heat(s_alpha), s_alpha * 100,
                                                     t.rst(), t.frame, s_tau, t.rst())),
        ]
        # ---- content-aware grid (19-10: "right sized ... doesn't resize at all").
        # Each column asks for the width its own text needs; three columns only
        # when they can hold their content, otherwise two, otherwise stacked.
        # Slack is shared instead of the fixed 38/32 fractions that squeezed the
        # left box to 47 columns for an 89-column line and clipped it mid-word.
        def _need(body, floor=24):
            return max(floor, max([vis_len(x) for x in body] + [0]) + 4)

        avg_body = [
            "%s%-15s%s %s%.1f%s   %s%-11s%s %s%s%s"
            % (t.label, "avg gen t/s", t.rst(), t.value, self.wmean(self.q_gen),
               t.rst(), t.label, "avg a", t.rst(), t.value,
               "n/a" if cur.tf_nodraft else "%.1f%%" % (self.wmean(self.q_acc) * 100), t.rst()),
            "%s%-15s%s %s%.1f%s   %s%-11s%s %s%.2f%s"
            % (t.label, "avg prompt t/s", t.rst(), t.value, self.wmean(self.q_pf),
               t.rst(), t.label, "avg tau", t.rst(), t.value,
               self.wmean(self.q_tau), t.rst()),
        ]
        # The cluster panel narrows itself below 110 columns, so measure the
        # narrow shape it will actually render in a side column.
        cl_probe = self.cluster_panel(min(78, max(34, width - 4)))
        need_l = _need(eng_body, 34)
        if cl_probe:
            need_l = max(need_l, _need(cl_probe, 34))
        need_m, need_r = 34, 44
        if need_l + need_m + need_r + 4 <= width:
            ncols = 3
        elif need_l + need_r + 2 <= width:
            ncols = 2
        else:
            ncols = 1
        if ncols == 3:
            rest = width - 4 - need_l
            extra = max(0, rest - need_m - need_r)
            tot = need_l + need_m + need_r
            lw = need_l + extra * need_l // tot
            mw = need_m + extra * need_m // tot
            rw = width - 4 - lw - mw
        elif ncols == 2:
            lw = min(need_l, width - 2 - need_r)
            rw = width - 2 - lw
            mw = rw if rw >= 44 else width
        else:
            lw = mw = rw = width
        wide = ncols >= 2
        sw = max(4, rw - 36)
        hist_body = [
            self.hrow("gen t/s", self.h_gen, sw, "%.1f", t.heat(0.9),
                      "%d tick" % len(self.h_gen), fit=rw - 2),
            self.hrow("prompt t/s", self.h_pf, sw, "%.0f", t.heat(0.7), "", fit=rw - 2),
            self.hrow("accept %", self.h_acc, sw, "%.0f", t.heat(alpha),
                      "n/a on tensorfold" if cur.tf_nodraft else "%d req" % len(self.h_acc),
                      fit=rw - 2),
            self.hrow("tau", self.h_tau, sw, "%.2f", t.head, "", fit=rw - 2),
        ]
        cl = self.cluster_panel(lw)
        # Height guard (left column): GB10 must ALWAYS sit under engine & session,
        # even on a short terminal where the full engine & session box would push GB10
        # past the L[:height] trim. If eng_body + GB10 won't fit the available rows,
        # drop engine & session rows from the bottom (least-important first: session,
        # drafted, accepted, the trailing blank, ctx max seen, slot ctx, then totals)
        # until it fits. GB10 itself is never trimmed.
        if cl and height:
            avail = height - len(L)
            need = (len(eng_body) + 2) + 1 + (len(cl) + 2)
            while need > avail and len(eng_body) > 5:
                eng_body.pop()
                need -= 1
        left_col = self.box("engine & session", eng_body, lw)
        if cl:
            left_col.append("")
            left_col += self.box(self.cluster_title(), cl, lw, t.accent)
        # Middle column = per-day chart only. History moved down to the right column
        # to stack with rolling averages and per-position. In two-column mode the
        # heatmap joins the right column (it needs ~44 columns, not a whole row).
        heat_body = self.heatmap_rows(mw)
        heat_title = ("generated tokens per day . all models" if mw >= 42
                      else "generated tokens per day" if mw >= 28 else "tokens/day")
        heat_box = (self.box(heat_title, heat_body, mw)
                    if heat_body else [])
        mid_col = heat_box if ncols == 3 else []
        # per-position acceptance: a small box that wraps one row of cells per the
        # column width it is rendered in. In wide mode it lives in the RIGHT column,
        # under rolling averages; in narrow mode it stays full-width at the bottom.
        def _pos_box(w):
            barw = 8 if w >= 100 else 5 if w >= 60 else 3
            cellw = barw + 10
            per = max(1, (w - 4) // cellw)
            chunks = []
            for i in range(len(posn)):
                v = (posn[i] / posd) if posd else 0.0
                chunks.append("%sp%d%s %s %s%3.0f%%%s"
                              % (t.head, i, t.rst(), self.bar(v, barw),
                                 t.heat(v), v * 100, t.rst()))
            body = []
            for i in range(0, len(chunks), per):
                body.append("".join(pad(c, cellw) for c in chunks[i:i + per]))
            return self.box("per-position acceptance  (%s)" % src, body, w, t.accent)
        # RIGHT column: rolling averages + per-position are the must-keep boxes
        # (per the user's layout request); history is shown on top only when there
        # is room in the grid, so on a short wide terminal we drop history (the
        # least-priority of the three) rather than rolling or per-position.
        hist_box = self.box("history", hist_body, rw) if hist_body else []
        avg_box = self.box("rolling averages", avg_body, rw)
        pos_box = _pos_box(rw) if (posn and wide) else []
        right_col = list(avg_box)
        if pos_box:
            right_col.append("")
            right_col += pos_box
        if wide and hist_box:
            if not (height and (height - len(L)) < len(right_col) + len(hist_box) + 1):
                right_col = list(hist_box) + [""] + right_col
        if ncols == 2 and heat_box and mw != width:
            # the heatmap is the first thing to go when the block would run past
            # the terminal: it is a day chart, not a live gauge
            if not height or (height - len(L)) >= max(len(left_col), len(right_col)) + len(heat_box) + 1:
                right_col.append("")
                right_col += heat_box
        if ncols == 3:
            for i in range(max(len(left_col), len(mid_col), len(right_col))):
                a = left_col[i] if i < len(left_col) else ""
                b = mid_col[i] if i < len(mid_col) else ""
                c = right_col[i] if i < len(right_col) else ""
                A(clip(pad(a, lw) + "  " + pad(b, mw) + "  " + c, width))
        elif ncols == 2:
            for i in range(max(len(left_col), len(right_col))):
                a = left_col[i] if i < len(left_col) else ""
                c = right_col[i] if i < len(right_col) else ""
                A(clip(pad(a, lw) + "  " + c, width))
            if heat_box and mw == width:
                for r in heat_box:
                    A(clip(r, width))
        else:
            for r in left_col:
                A(clip(r, width))
            if heat_box:
                for r in heat_box:
                    A(clip(r, width))
            for r in right_col:
                A(clip(r, width))
        A("")

        # ---- gateway lanes. (The GB10 / cluster node box now lives directly under the
        # engine & session panel; see that section. It shows this box's GPU + RAM and the
        # rank-1 worker + RoCE link on a TP pair, and is the highest-priority box.)
        free_rows = (height - len(L)) if height else 99
        ln = self.lanes_panel(cur, width) if free_rows >= 3 else None
        if ln:
            for r in self.box("gateway lanes", ln, width, t.head):
                A(clip(r, width))

        # ---- expert routing heat (render.rs render_experts port)
        archx = self.model_arch()
        if archx and (not height or height - len(L) >= 5):
            self.fade_experts(archx, cur.busy and gen > 0, fdt,
                              cur.pp + cur.dec)
            # every layer its own half row when there is room (46 on GLM-5.3 with MTP = 23 rows), fewer when not
            natural = (len(self.fade_expert) or archx["n"] + 1) // 2
            avail = (height - len(L) - 3) if height else natural
            rows = self.expert_grid(width, rows_avail=max(2, min(natural, avail)))
            if rows:
                xs = getattr(self, "_xs", None)
                if xs and xs.get("delta") is not None:
                    title = ("expert routing heat  (REAL - TensorFold /experts: decode picks since the last read, "
                             "colour = share of the layer's busiest expert, fading)")
                elif xs and xs.get("on"):
                    title = "expert routing heat  (REAL - TensorFold /experts: waiting for a second read)"
                elif xs and xs.get("why"):
                    title = "expert routing heat  (simulated - /experts is off on this engine: %s)" % xs["why"]
                elif getattr(self, "tf", False):
                    title = ("expert routing heat  (simulated - this TensorFold image has no /experts: "
                             "patches/0990, glm53mia image y9, goes live with the next restart)")
                else:
                    title = ("expert routing heat  (simulated - vllm does not "
                             "export router choices; colour = time since routed)")
                for r in self.box(title, rows, width, t.accent):
                    A(clip(r, width))

        # ---- layer visualizer or log fills whatever rows are left
        if self.show_layers:
            lp = self.layer_panel(width, height - len(L) - 4)
            if lp:
                for r in self.box("layer activity  (L = log, g = full)",
                                  lp, width, t.accent):
                    A(clip(r, width))
        elif self.show_log:
            lp = active_log(self.llm)
            free = height - len(L) - 4
            if lp and free >= 3:
                rows = [t.dim() + clip(x, width - 4) + t.rst()
                        for x in tail(lp, free)]
                for r in self.box(os.path.basename(lp),
                                  rows or [t.dim() + "(empty)" + t.rst()], width, t.head):
                    A(clip(r, width))

        L = [clip(ln, width) for ln in L]
        # Never hand back more rows than the terminal has: the extras scroll the
        # top of the panel off screen and the fixed-position redraw tears.
        if height and len(L) > height:
            L = L[:height]
        self.last_frame = L
        return L

