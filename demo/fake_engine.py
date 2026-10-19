#!/usr/bin/env python3
"""fake_engine - a synthetic llama.cpp-style server for trying suzu without a GPU.

Serves the endpoints suzu reads (/v1/models, /slots, /metrics, /health and an
optional gateway lanes JSON) and simulates one busy slot running speculative
decoding: prefill, decode with draft/accept counters, idle gaps, next request.
Counters behave like the real thing - /slots advances every poll, /metrics
flushes only at request close (the measurement that shaped suzu's design).

usage: fake_engine.py [--port 8099] [--model NAME] [--draft NAME]
"""

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

AP = argparse.ArgumentParser()
AP.add_argument("--port", type=int, default=8099)
AP.add_argument("--model", default="Qwen3.5-32B-Instruct-Q4_K_M")
AP.add_argument("--draft", "--spec-draft-model", dest="draft",
                default="Qwen3.5-0.8B-DFlash",
                help="draft model name (llama.cpp spells it --spec-draft-model; "
                     "suzu reads the engine cmdline, so pass it the way a real server does)")
AP.add_argument("--spec-type", default="draft-mtp",
                help="speculative method, as llama.cpp's --spec-type")
AP.add_argument("--spec-draft-n-max", dest="spec_k", type=int, default=5,
                help="draft tokens per round (the engine drafts exactly this many)")
AP.add_argument("--ctx", type=int, default=131072)
AP.add_argument("--lanes", action="store_true", help="also serve a /yzq/stats-style lanes JSON")
A = AP.parse_args()

T0 = time.monotonic()


def schedule(k):
    """Deterministic per-request shape: prompt, cached, decode tokens, tps, idle."""
    prompt = 2400 + (k * 737) % 3800
    cached = int(prompt * 0.82)
    out = 180 + (k * 331) % 420
    tps = 24.0 + (k * 13) % 9
    return prompt, cached, out, tps, 2.5 + (k % 3)


PREFILL_S = 1.2


def phase(t):
    """-> (k, phase_name, phase_t, dur, req)."""
    k = 0
    t = t - T0
    while True:
        prompt, cached, out, tps, idle = schedule(k)
        pre = PREFILL_S
        dec = out / tps
        if t < pre:
            return k, "prefill", t, pre, (prompt, cached, out, tps)
        t -= pre
        if t < dec:
            return k, "decode", t, dec, (prompt, cached, out, tps)
        t -= dec
        if t < idle:
            return k, "idle", t, idle, (prompt, cached, out, tps)
        t -= idle
        k += 1


def closed_metrics():
    """Cumulative /metrics counters, advanced only at request close."""
    t = time.monotonic() - T0
    pred = draft_tok = drafts = acc = 0.0
    per_pos = [0.0] * 5
    prompt_tot = 0.0
    k = 0
    while True:
        prompt, cached, out, tps, idle = schedule(k)
        pre, dec = PREFILL_S, out / tps
        span = pre + dec + idle
        if t < span:
            # current request not closed yet
            if t >= pre + dec:  # in idle: this request HAS closed
                pass
            else:
                break
        rounds = max(1, out // 3)
        pred += out
        draft_tok += rounds * 5
        drafts += rounds
        acc += out
        for pos in range(5):
            per_pos[pos] += rounds * (1.0 if pos == 0 else max(0.0, 0.92 - 0.18 * pos))
        prompt_tot += prompt
        t -= span
        k += 1
    return pred, draft_tok, drafts, acc, per_pos, prompt_tot


def slots_json():
    k, name, pt, dur, (prompt, cached, out, tps) = phase(time.monotonic())
    if name == "prefill":
        frac = min(1.0, pt / dur)
        return [{"id": 0, "is_processing": True, "n_ctx": A.ctx,
                 "n_prompt_tokens": prompt, "n_prompt_tokens_processed": int(prompt * frac),
                 "n_prompt_tokens_cache": cached,
                 "next_token": {"n_decoded": 0}}]
    if name == "decode":
        done = int(pt * tps)
        return [{"id": 0, "is_processing": True, "n_ctx": A.ctx,
                 "n_prompt_tokens": prompt, "n_prompt_tokens_processed": prompt,
                 "n_prompt_tokens_cache": cached,
                 "next_token": {"n_decoded": done}}]
    return [{"id": 0, "is_processing": False, "n_ctx": A.ctx,
             "n_prompt_tokens": 0, "n_prompt_tokens_processed": 0,
             "n_prompt_tokens_cache": 0, "next_token": {"n_decoded": 0}}]


def metrics_text():
    pred, draft_tok, drafts, acc, per_pos, prompt_tot = closed_metrics()
    L = ["# TYPE llamacpp:tokens_predicted_total counter",
         "# TYPE llamacpp:tokens_prompt_total counter",
         "# TYPE llamacpp:spec_decode_num_draft_tokens_total counter",
         "# TYPE llamacpp:spec_decode_num_accepted_tokens_total counter",
         "# TYPE llamacpp:spec_decode_num_drafts_total counter"]
    L.append("llamacpp:tokens_predicted_total %g" % pred)
    L.append("llamacpp:tokens_prompt_total %g" % prompt_tot)
    L.append("llamacpp:spec_decode_num_draft_tokens_total %g" % draft_tok)
    L.append("llamacpp:spec_decode_num_accepted_tokens_total %g" % acc)
    L.append("llamacpp:spec_decode_num_drafts_total %g" % drafts)
    for pos, v in enumerate(per_pos):
        L.append('# TYPE llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="%d"} counter' % pos)
        L.append('llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="%d"} %g' % (pos, v))
    return "\n".join(L) + "\n"


LANES = {"parallel": 4, "lane_exclusive": True, "running_high": 1, "running_low": 0,
         "queued_high": 0, "queued_low": 0, "free_slots": 3,
         "hint_high": "[yzq: 3 of 4 lanes free right now]"}


class H(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send(self, body, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        p = urlparse(self.path).path
        if p == "/health":
            self._send('{"status":"ok"}')
        elif p == "/v1/models":
            self._send(json.dumps({"object": "list", "data": [
                {"id": A.model, "object": "model", "owned_by": "fake_engine"}]}))
        elif p == "/slots":
            self._send(json.dumps(slots_json()))
        elif p == "/metrics":
            self._send(metrics_text(), "text/plain; version=0.0.4")
        elif p == "/yzq/stats" and A.lanes:
            self._send(json.dumps(LANES))
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == "__main__":
    print("fake_engine: %s + draft %s (%s) on :%d%s" % (
        A.model, A.draft, A.spec_type, A.port, " (lanes on)" if A.lanes else ""))
    ThreadingHTTPServer(("127.0.0.1", A.port), H).serve_forever()
