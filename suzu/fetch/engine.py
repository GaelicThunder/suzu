"""Engine sample model and the llama.cpp/vLLM live probes."""
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

from .http import g, gl, http_get


class Sample:
    """One sample: cumulative counters from /metrics + live state from /slots."""

    def __init__(self, mx, slots, t):
        self.t = t
        # tabby BEFORE llama.cpp: synth_tabby deliberately writes `llamacpp:` keys
        # (so the rest of the file does not know it exists), and without this case
        # the panel would present itself as llama.cpp.
        self.fam = ("tabby" if "tabby:records" in mx
                    else "tensorfold" if "tensorfold:records" in mx
                    else "llama.cpp" if any(k.startswith("llamacpp:") for k in mx)
                    else "vllm" if any(k.startswith("vllm:") for k in mx)
                    else "ds4" if any(k.startswith("ds4_") for k in mx) else "?")
        self.gen = g(mx, "llamacpp:tokens_predicted_total",
                     "vllm:generation_tokens_total", "ds4_tokens_decoded_total")
        self.pp = g(mx, "llamacpp:prompt_tokens_total", "vllm:prompt_tokens_total",
                    'ds4_tokens_prefilled_total{kind="computed"}')
        self.ps = g(mx, "llamacpp:prompt_seconds_total",
                    "vllm:request_prefill_time_seconds_sum")
        # vllm:prompt_tokens_total counts the PREFIX-CACHE HITS too - here 58.1 M
        # of 60.5 M lifetime tokens were served from cache and cost no prefill at
        # all. Dividing that by prefill seconds measures the cache, not the
        # engine. local_compute is the subset actually computed, which is what
        # llama.cpp's n_prompt_tokens_processed means, so the two engines' rows
        # finally say the same thing.
        # TensorFold without a drafted-token counter (jayleaton): no acceptance %, only tokens/round
        self.tf_nodraft = self.fam == "tensorfold" and "tensorfold:has_drafted" not in mx
        self.tf_fill = (mx.get("tensorfold:fill_streams"), mx.get("tensorfold:fill_pos"), mx.get("tensorfold:fill_len"))
        _lc = gl(mx, "vllm:prompt_tokens_by_source_total", 'source="local_compute"')
        if _lc is not None:
            self.pp = _lc
        self.draft = g(mx, "llamacpp:spec_decode_num_draft_tokens_total",
                       "vllm:spec_decode_num_draft_tokens_total")
        self.acc = g(mx, "llamacpp:spec_decode_num_accepted_tokens_total",
                     "vllm:spec_decode_num_accepted_tokens_total")
        self.drafts = g(mx, "llamacpp:spec_decode_num_drafts_total",
                        "vllm:spec_decode_num_drafts_total")
        self.pos = []
        for i in range(32):
            k = ('llamacpp:spec_decode_num_accepted_tokens_per_pos_total'
                 '{position="%d"}' % i)
            k2 = ('vllm:spec_decode_num_accepted_tokens_per_pos_total'
                  '{position="%d"}' % i)
            if k in mx:
                self.pos.append(mx[k])
            elif k2 in mx:
                self.pos.append(mx[k2])
            else:
                break
        self.run = g(mx, "llamacpp:requests_processing",
                     "vllm:num_requests_running", "ds4_requests_inflight")
        self.wait = g(mx, "llamacpp:requests_deferred", "vllm:num_requests_waiting")
        # 19-09: with a gateway in front, the engine may HOLD requests the gateway
        # has ALREADY admitted (max_num_seqs, full KV pool, the kit's mixed-prefill
        # `skip`): vLLM counts them for a reason. Without these two the panel said
        # just "2/2 slots" while the human waited 335 s for the first token.
        self.wait_cap = gl(mx, "vllm:num_requests_waiting_by_reason", 'reason="capacity"') or 0.0
        self.wait_def = gl(mx, "vllm:num_requests_waiting_by_reason", 'reason="deferred"') or 0.0
        self.ctxmax = g(mx, "llamacpp:n_tokens_max")
        # totals since the engine started, input included (30-09: the panel showed only the generated ones). On vLLM
        # prompt_tokens_total counts the cached too, like TensorFold's prompt. llama.cpp exposes the same total as
        # tokens_prompt_total - without this fallback the totals box (and the harness rows under it) stayed hidden
        # on every llama.cpp-style engine.
        self.tot_in = mx.get("tensorfold:prompt_total",
                             mx.get("vllm:prompt_tokens_total",
                                    mx.get("llamacpp:tokens_prompt_total")))
        self.tot_cached = mx.get("tensorfold:cached_total",
                                 gl(mx, "vllm:prompt_tokens_by_source_total", 'source="local_cache_hit"'))
        self.nreq = mx.get("tensorfold:records", mx.get("vllm:request_success_total"))
        # TensorFold: the usual all-time totals too, from the request log (they survive restarts)
        self.all_in = mx.get("tensorfold:all_in")
        self.all_out = mx.get("tensorfold:all_out")
        self.all_n = mx.get("tensorfold:all_n")
        self.all_since = mx.get("tensorfold:all_since")
        # live, per-token
        self.dec = self.ppp = self.ptok = self.pcache = self.nctx = 0
        self.busy = False
        self.nslots = 0
        if isinstance(slots, list):
            self.nslots = len(slots)
            for s in slots:
                nt = s.get("next_token") or []
                if isinstance(nt, list) and nt:
                    self.dec += int(nt[0].get("n_decoded", 0) or 0)
                elif isinstance(nt, dict):
                    self.dec += int(nt.get("n_decoded", 0) or 0)
                self.ppp += int(s.get("n_prompt_tokens_processed", 0) or 0)
                self.ptok += int(s.get("n_prompt_tokens", 0) or 0)
                self.pcache += int(s.get("n_prompt_tokens_cache", 0) or 0)
                self.nctx += int(s.get("n_ctx", 0) or 0)
                self.busy = self.busy or bool(s.get("is_processing"))

    def restarted_from(self, prev):
        return (self.pp < prev.pp - 1 or self.gen < prev.gen - 1
                or self.acc < prev.acc - 1)



def fetch_yzq(port):
    """Queue depth from the optional gateway in front of the engine, if the
    polled port IS the gateway.

    Why this exists: the engine runs --max-num-seqs 1 and the gateway forwards
    ONE request at a time, so vllm:num_requests_waiting is structurally 0 no
    matter how many calls are outstanding — everyone else is parked in the
    gateway's lanes and has never been offered to the engine. Reading only
    /metrics, the panel said "running 1 / waiting 0" while four conversations
    were in flight (measured 23-08: gateway stats high=3 + inflight, /metrics
    waiting=0). The stats endpoint is answered by the gateway BEFORE it touches
    the scheduler (handle(), first branch), so polling it costs nothing and can
    never queue behind a generation."""
    txt = http_get("http://127.0.0.1:%d/yzq/stats" % port, timeout=2, tries=1)
    if not txt:
        return None
    try:
        j = json.loads(txt)
    except (json.JSONDecodeError, ValueError):
        return None
    return j if isinstance(j, dict) and "high" in j else None



def yzq_live_sse(yzq):
    """SSE events relayed so far for EVERY request still open at the gateway. Its `inflight` is only the first job: with
    two lanes the other one's tokens were missing from the live estimate and landed all at once when it closed (30-09,
    TensorFold: gen t/s 2145, peak 1079, a rolling average near twice the real one)."""
    q = yzq or {}
    jobs = q.get("running")
    if jobs:
        return sum(int(j.get("sse") or 0) for j in jobs)
    return int((q.get("inflight") or {}).get("sse") or 0)



def fetch_yzq_lanes(port, url=None):
    """The optional gateway's admission state in detail: per-lane queues,
    running slots and the free/parallel breakdown - the lanes endpoint.
    What /metrics cannot say about WHY a request is waiting, this can.
    url=None keeps the legacy same-port probe; an explicit --lanes-url wins."""
    txt = http_get(url if url else "http://127.0.0.1:%d/yzq/lanes" % port,
                   timeout=2, tries=1)
    if not txt:
        return None
    try:
        j = json.loads(txt)
    except (json.JSONDecodeError, ValueError):
        return None
    return j if isinstance(j, dict) else None

