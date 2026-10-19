"""vLLM-specific synthesis: pool gauges, ctx max, synthetic slots."""
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

from .http import http_get


_VLLM_POOL = {}



_VLLM_CTXMAX = {}



def vllm_pool(port):
    """KV pool size in tokens, from the boot log ("GPU KV cache size: N tokens").

    Not derivable from /metrics: `vllm:cache_config_info` reports num_gpu_blocks
    12355 x block_size 4 = 49,420 on a real pool of 2,192,815 (measured 21-08),
    because the hybrid cache (128-token sliding-window layers beside the global
    ones) is not that product. `docker logs` costs ~10 s: memoized on container
    id + StartedAt, so a restart invalidates the cache by itself.
    """
    if port in _VLLM_POOL:
        return _VLLM_POOL[port]
    _VLLM_POOL[port] = None
    try:
        names = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                               capture_output=True, text=True, timeout=8).stdout.split()
        name = next((x for x in names
                     if any(t in x for t in ("vllm", "spark", "sglang"))), None)
        if not name:
            return None
        key = subprocess.run(["docker", "inspect", "-f", "{{.Id}}-{{.State.StartedAt}}", name],
                             capture_output=True, text=True, timeout=8).stdout.strip()
        cache = "/tmp/yz-kvpool-" + re.sub(r"[^A-Za-z0-9]", "", key)[:40] if key else None
        if cache and os.path.exists(cache):
            _VLLM_POOL[port] = int(open(cache).read().strip())
            return _VLLM_POOL[port]
        log = subprocess.run(["docker", "logs", "--tail", "4000", name],
                             capture_output=True, text=True, timeout=40)
        hits = re.findall(r"GPU KV cache size: ([\d,]+) tokens", log.stdout + log.stderr)
        if hits:
            _VLLM_POOL[port] = int(hits[-1].replace(",", ""))
            if cache:
                open(cache, "w").write(str(_VLLM_POOL[port]))
    except Exception:
        pass
    return _VLLM_POOL[port]



def vllm_ctxmax(port):
    """max_model_len from /v1/models: on vLLM there is no llamacpp:n_tokens_max."""
    if port in _VLLM_CTXMAX:
        return _VLLM_CTXMAX[port]
    _VLLM_CTXMAX[port] = 0
    txt = http_get("http://127.0.0.1:%d/v1/models" % port, timeout=4, tries=1)
    if txt:
        try:
            _VLLM_CTXMAX[port] = int(json.loads(txt)["data"][0].get("max_model_len") or 0)
        except (ValueError, KeyError, IndexError, TypeError):
            pass
    return _VLLM_CTXMAX[port]



def synth_slots_vllm(mx, port):
    """Synthetic slot for vLLM, which does NOT expose /slots.

    On llama.cpp the /metrics counters are flushed at request end, which is why
    this monitor reads /slots. On vLLM it is the OPPOSITE, measured by sampling
    every 2 s inside a single request (21-08 on dsflash):

        t1 draft=530  acc=298 gen=400      <- not started yet
        t3 draft=725  acc=395 gen=537
        t6 draft=1035 acc=502 gen=706      <- still in flight

    So it is enough to remap /metrics into the shape of a llama.cpp slot and the
    whole TUI works without touching anything else. Two differences to keep in
    mind:
      - `n_decoded` here is a LIFETIME counter, not reset per request: deltas
        stay >= 0, so the "negative delta = new task" branch never fires
        (harmless).
      - prefill is not visible on `vllm:prompt_tokens_total`, credited only at
        request CLOSE (measured: 48 after killing a 900k prefill halfway). The
        only reading that advances during a long prefill is
        `kv_cache_usage_perc` -> progress is that fraction of the pool.
    Without the pool (unreadable log) it returns 0 on ptok/n_ctx: the slot panel
    switches itself off and the rest of the TUI stays valid.
    """
    if not any(k.startswith("vllm:") for k in mx):
        return None
    pool = vllm_pool(port) or 0
    ptok = int(mx.get("vllm:kv_cache_usage_perc", 0.0) * pool)
    q = mx.get("vllm:prefix_cache_queries_total", 0.0)
    h = mx.get("vllm:prefix_cache_hits_total", 0.0)
    return [{
        "next_token": {"n_decoded": int(mx.get("vllm:generation_tokens_total", 0.0))},
        "n_prompt_tokens_processed": ptok,
        "n_prompt_tokens": ptok,
        "n_prompt_tokens_cache": int(ptok * h / q) if q > 0 else 0,
        "n_ctx": int(pool),
        "is_processing": mx.get("vllm:num_requests_running", 0.0) > 0,
    }]

