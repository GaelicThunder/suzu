"""Model/server introspection: config.json, cmdline, arch map."""
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
from .tabby import proc_env
from .tensorfold import _lanes_host, tf_env


def server_info(port):
    """Draft model, spec type, k and uptime from the process holding the port.
    tmux and earlyoom lines are skipped: tmux carries an identical command line
    (it is the wrapper) and earlyoom embeds 'llama-server' in its --prefer regex."""
    info = {"draft": None, "spec": None, "k": None, "up": None, "model": None,
            "engine": None}
    try:
        # -ww IS MANDATORY: without it ps truncates args= to $COLUMNS, and this
        # panel sets COLUMNS itself for layout, so "--port N" fell off the end
        # and the header silently lost the draft model and the uptime.
        out = subprocess.run(["ps", "-ww", "-eo", "pid=,etimes=,args="],
                             capture_output=True, text=True, timeout=4).stdout
    except (OSError, subprocess.SubprocessError) as e:
        if os.environ.get("SUZU_DEBUG"):
            sys.stderr.write("suzu: ps failed: %r\n" % (e,))
        return info
    portpat = re.compile(r"(?:--port[= ]\s*|-p\s+)%d(?!\d)" % port)
    best = fallback = None
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        args = parts[2]
        base = os.path.basename(args.split(None, 1)[0])
        if base in ("tmux", "earlyoom", "sh", "bash", "grep"):
            continue
        if not any(b in base for b in ("llama-server", "ds4-server", "vllm", "python")):
            continue
        # "python" also matches the gateway and this very monitor: neither is an
        # engine, and picking one would report ITS uptime as the model's. Exclude
        # suzu by SCRIPT NAME, not path substring - the demo engine lives under a
        # .../suzu/ directory and must still match - plus the gateway if
        # --lanes-url points at one.
        try:
            script = os.path.basename(args.split(None, 1)[1].split()[0])
        except IndexError:
            script = ""
        if script == "suzu.py" or (config.LANES_URL and _lanes_host() in args):
            continue
        if portpat.search(args):
            best = (parts, args, base)
            break
        # The panel normally polls :30001, which is the GATEWAY - the engine
        # itself listens on :30021 and its cmdline never carries the polled port,
        # so the strict match found nothing and the header lost the draft model
        # AND the uptime ("(no draft model)", "--:--", seen 23-08 with a dspark
        # drafter plainly running). The launcher guarantees one engine at a time, so an
        # engine process with the wrong port is still THE engine.
        if fallback is None and any(
                x in args for x in ("llama-server", "ds4-server", "vllm", "sglang", "tensorfold serve")):
            fallback = (parts, args, base)
        # TabbyAPI presents itself as `venv/bin/python .../tabby/main.py --config X`:
        # none of the markers above. Without this branch the glm53x entry's header
        # stayed without engine name and without uptime.
        if fallback is None and "main.py" in args and "tabby" in args:
            fallback = (parts, args, base)
    if not best:
        best = fallback
    if not best:
        if os.environ.get("SUZU_DEBUG"):
            sys.stderr.write("suzu: no process on port %d among %d ps lines\n"
                             % (port, len(out.splitlines())))
        return info
    parts, args, base = best
    try:
        info["up"] = int(parts[1])
    except ValueError:
        pass
    info["pid"] = parts[0]
    info["engine"] = base
    # `python -m vllm.entrypoints...` is not a useful name in the header.
    for nice in ("llama-server", "ds4-server", "vllm", "sglang"):
        if nice in args:
            info["engine"] = nice
            break
    if "tensorfold serve" in args:
        # rank 0 of the TP=2 engine (glm53tf): inside the container, visible from the host's ps
        info["engine"] = "tensorfold"
        m = re.search(r"--drafter[= ]\s*(\S+)", args)
        m_drafter = bool(m) and m.group(1) != "none"
        if m:
            d = re.search(r"models--([^/]+)--([^/]+)/snapshots/([0-9a-f]{7})", m.group(1))
            info["draft"] = "%s/%s@%s" % d.groups() if d else os.path.basename(m.group(1).rstrip("/"))
            info["spec"] = "DFlash2 + MTP + lookup, depth by cost"
        else:
            info["spec"] = "MTP, depth by cost"
        for flag, key in (("--context", "maxlen"), ("--name", "model"), ("--tp", "tp")):
            m = re.search(re.escape(flag) + r"[= ]\s*(\S+)", args)
            if m:
                info[key] = m.group(1)
        if info.get("tp") == "2":
            info["nnodes"] = "2"          # the two ranks live on this box and the rank-1 box
        env = tf_env()
        info["k"] = env.get("GLM53_TF_AUTO_FDRAFTS") or info["k"]
        info["slots"] = env.get("GLM53_TF_BATCH")
        info["kvdtype"] = env.get("GLM53_TF_KV_DTYPE")
        info["quant"] = "EXL3 4bpw + q4 non-expert" if env.get("GLM53_TF_NONEXPERT") == "q4mse" else None
        if "TF_GLM_DFLASH_POLICY" in env:                 # newer kit: other knobs, other names
            pol = env["TF_GLM_DFLASH_POLICY"]
            m = re.match(r"f\D*(\d+)", pol)
            info["k"] = m.group(1) if m else info["k"]
            copy = env.get("TF_GLM_COPY_DRAFTS") == "1"
            info["spec"] = ("DFlash2 %s + copy drafts (up to %s)" % (pol, env.get("TF_GLM_COPY_MAX", "?")) if copy
                            else "DFlash2 %s" % pol) if m_drafter else "MTP"
            m = re.search(r"--parallel[= ]\s*(\d+)", args)
            info["slots"] = m.group(1) if m else info["slots"]
            info["kvdtype"] = env.get("TF_GLM_KV") or info["kvdtype"]
            dense = env.get("TF_GLM_DENSE")
            info["quant"] = "EXL3 4bpw + %s dense" % dense if dense else info["quant"]
        return info
    if "main.py" in args and "tabby" in args:
        info["engine"] = "tabby/exl3"
        # without the environment the header does not even know whether MTP is on
        env = proc_env(parts[0])
        info["spec"] = env.get("TABBY_DRAFT_MODEL_DRAFT_MODE") or info["spec"]
        if info["spec"] and info["spec"] != "disabled":
            info["draft"] = info["draft"] or "native MTP of the pack"
        cache = env.get("TABBY_MODEL_CACHE_MODE")
        if cache:
            info["engine"] += " . cache " + cache
    for flag, key in (("--spec-draft-model", "draft"), ("--spec-type", "spec"),
                      ("--spec-draft-n-max", "k"), ("--model", "model"),
                      ("--served-model-name", "model"),
                      ("--quantization", "quant"), ("--kv-cache-dtype", "kvdtype"),
                      ("--tensor-parallel-size", "tp"), ("--nnodes", "nnodes"),
                      ("--max-num-seqs", "seqs"), ("--max-model-len", "maxlen")):
        if info.get(key):
            continue
        m = re.search(re.escape(flag) + r"[= ]\s*(\S+)", args)
        if m:
            info[key] = m.group(1)
    # vLLM does not take the llama.cpp flags: the whole speculative stack arrives
    # as one JSON blob, so without this the panel says "no draft model" while
    # dspark is demonstrably drafting (spec_decode_* counters moving).
    m = re.search(r"--speculative[-_]config[= ]\s*(\{.*?\})(?:\s+--|\s*$)", args)
    if m:
        try:
            sc = json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            sc = {}
        if isinstance(sc, dict):
            info["draft"] = info["draft"] or sc.get("model")
            info["spec"] = info["spec"] or sc.get("method")
            k = sc.get("num_speculative_tokens")
            info["k"] = info["k"] or (str(k) if k is not None else None)
    return info



def model_config_path():
    """Host path to the running engine's model config.json.

    The layer grid is built from the REAL model topology (num layers, experts,
    DSpark target layers, KV compression ratios) - not a hard-coded table. The
    engine is a docker container; the config lives in the /models mount. Falls
    back to scanning ~/models for the most recent tp1/config.json.
    """
    try:
        names = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                               capture_output=True, text=True, timeout=8).stdout.split()
        # TensorFold (glm53mia, glm53tf: 05-10): `tensorfold serve <model folder>` inside the container, the folder
        # under the HF cache mount. Without this the expert heat map had no topology on TensorFold and vanished
        # once the old tp1/ folders of the vLLM kit were gone from ~/models.
        for n in names:
            try:
                out = subprocess.run(["docker", "inspect", "-f", "{{json .Config.Cmd}}\t{{json .Mounts}}", n],
                                     capture_output=True, text=True, timeout=8).stdout
                cmd_j, mounts_j = out.rstrip("\n").split("\t", 1)
                cmd, mounts = json.loads(cmd_j) or [], json.loads(mounts_j) or []
            except (ValueError, OSError, subprocess.SubprocessError):
                continue
            if "serve" not in cmd or not any("tensorfold" in str(c) for c in cmd[:2]):
                continue
            model = cmd[cmd.index("serve") + 1] if cmd.index("serve") + 1 < len(cmd) else ""
            for mnt in mounts:
                dst = (mnt.get("Destination") or "").rstrip("/")
                if dst and model.startswith(dst + "/"):
                    p = os.path.join(mnt.get("Source") or "", model[len(dst) + 1:], "config.json")
                    if os.path.exists(p):
                        return p
        for n in names:
            if not any(t in n for t in ("vllm", "spark", "sglang")):
                continue
            mounts = subprocess.run(
                ["docker", "inspect", "-f",
                 "{{range .Mounts}}{{.Source}}|{{.Destination}} {{end}}", n],
                capture_output=True, text=True, timeout=8).stdout
            for part in mounts.split():
                if "|" not in part:
                    continue
                src, dst = part.split("|", 1)
                dst = dst.rstrip("/")
                if dst == "/models":
                    p = os.path.join(src, "tp1", "config.json")
                    if os.path.exists(p):
                        return p
    except (OSError, subprocess.SubprocessError):
        pass
    # fallback: the most recent tp1 config in ~/models
    cands = []
    mdir = os.path.expanduser("~/models")
    try:
        for root, dirs, files in os.walk(mdir):
            if root.endswith(os.sep + "tp1") and "config.json" in files:
                cands.append(os.path.join(root, "config.json"))
    except OSError:
        pass
    return max(cands, key=os.path.getmtime) if cands else None



def load_model_arch(path):
    """Extract the per-layer topology from a model config.json.

    Returns None on any failure; the layer grid degrades to a message instead
    of a crash. Everything here is static model structure - the LIVE work
    colours come from the Sample, never from this table.
    """
    try:
        with open(path) as f:
            c = json.load(f)
    except (OSError, ValueError):
        return None
    if isinstance(c.get("text_config"), dict):      # GLM-5.3 (glm5_next): the text model's fields live here
        top, c = c, dict(c["text_config"])
        if not c.get("vision_n_layers"):
            c["vision_n_layers"] = (top.get("vision_config") or {}).get("depth") or 0
        kinds = c.get("mlp_layer_types") or []
        if kinds and not c.get("hybrid_tr3_tail"):
            c["hybrid_tr3_tail"] = {"moe_layers": [i for i, k in enumerate(kinds) if k == "sparse"]}
    n = int(c.get("num_hidden_layers") or 0)
    if n <= 0:
        return None
    comp = c.get("compress_ratios") or []
    comp = {i: int(v) for i, v in enumerate(comp)}
    return {
        "n": n,
        "experts": int(c.get("n_routed_experts") or 0),
        "topk": int(c.get("num_experts_per_tok") or 0),
        "shared": int(c.get("n_shared_experts") or 0),
        "nextn": int(c.get("num_nextn_predict_layers") or 0),
        "vision": int(c.get("vision_n_layers") or 0),
        "target": [int(x) for x in (c.get("dspark_target_layer_ids") or [])],
        "compress": comp,
        "moe_layers": (c.get("hybrid_tr3_tail") or {}).get("moe_layers"),
        "name": os.path.basename(os.path.dirname(os.path.dirname(path))),
    }

