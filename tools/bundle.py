#!/usr/bin/env python3
"""Build suzu.py - the single-file release - from the suzu/ package.

suzu/ is the source of truth; suzu.py is what people curl. This script inlines
the modules in dependency order into one flat file, dropping the relative
imports and keeping the module bodies as sections. Nothing else is rewritten:
the flat namespace is safe because no two modules define the same top-level
name (checked below at build time).

The two module-qualified names that survive are handled explicitly:
  * `config.X` reads/writes go through a small `config` namespace object, so
    cli.py's runtime overrides are seen by every module;
  * cli.py's `rank1._RANK1 = rank1._rank1_feed(...)` becomes a globals() write,
    because fetch_rank1() reads the module global directly.

usage: python3 tools/bundle.py [--check]
       --check  build in memory and fail if suzu.py on disk is stale
"""
import io
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "suzu.py"

ORDER = [
    "suzu/ansi.py",
    "suzu/text.py",
    "suzu/config.py",
    "suzu/themes.py",
    "suzu/fetch/http.py",
    "suzu/fetch/memory.py",
    "suzu/fetch/gpu.py",
    "suzu/fetch/info.py",
    "suzu/fetch/logtail.py",
    "suzu/fetch/vllm.py",
    "suzu/fetch/engine.py",
    "suzu/fetch/tabby.py",
    "suzu/fetch/tensorfold.py",
    "suzu/fetch/rank1.py",
    "suzu/render/core.py",
    "suzu/render/widgets.py",
    "suzu/render/panels.py",
    "suzu/render/frames.py",
    "suzu/render/view.py",
    "suzu/cli.py",
]

REL_IMPORT = re.compile(r"^\s*from \.+[\w.]* import ")
ABS_IMPORT = re.compile(r"^(?:import \S+|from \S+ import .*)$")
DEF_RE = re.compile(r"^(?:def|class) ([A-Za-z_][A-Za-z0-9_]*)", re.M)

HEADER = '''#!/usr/bin/env python3
"""suzu - a small bell for your LLM engine: live monitor for a
speculative-decoding inference server.

*** GENERATED FILE - edit suzu/ and run tools/bundle.py, not this file. ***

The measurement that dictates this file's whole architecture:
llama.cpp's /metrics counters DO NOT MOVE during a generation. They are
flushed at request completion. Measured: 6.2 s of active slot produced delta
0 on tokens_predicted_total and on every spec_decode_* counter, while the
model was emitting 20.5 tok/s. A 1 s monitor built on /metrics alone would
read "0 tok/s" for minutes under full load. Hence the split:

  live rate       <- /slots  next_token.n_decoded   (advances every token)
  prefill         <- /slots  n_prompt_tokens_processed
  KV/prefix cache <- /slots  (llama.cpp exposes NEITHER on /metrics)
  acceptance, tau, p0..pN <- /metrics, shown WITH THE AGE of the last closed
                             request rather than passed off as instantaneous

Sources, all already exposed - no engine patch required:
  /slots    next_token.n_decoded, n_prompt_tokens{,_processed,_cache}, n_ctx,
            is_processing
  /metrics  spec_decode_num_{draft,accepted}_tokens_total,
            ..._num_drafts_total, ..._accepted_tokens_per_pos_total{position=N}
  ps -ww    server cmdline -> draft model, spec method, k, uptime, plus what
            vLLM adds: quantisation, kv dtype, tensor-parallel size, nnodes,
            max-num-seqs, max model len (the model strip reads these)
  nvidia-smi  GPU utilisation, VRAM, watts, degC, SM clock (the GB10 panel)
  --lanes-url  optional gateway lanes endpoint: the queue you actually sit
            in, which /metrics cannot see (the gateway answers before
            touching the scheduler, so polling it is free)
  --rank1-url  rank 1 of a tensor-parallel pair (Prometheus textformat),
            probed only when the cmdline says --nnodes 2: the cluster is
            only as alive as its quietest rank

See ARCHITECTURE.md for the module map; this file is those modules inlined.
"""
'''

CONFIG_NAMES = ("TF_REQLOG", "HERMES_DB", "LANES_URL", "RANK1_URL")


def build():
    parts = [HEADER]
    seen_defs = {}
    imports = []
    for rel in ORDER:
        src = (ROOT / rel).read_text()
        body_lines = []
        for ln in src.splitlines():
            if REL_IMPORT.match(ln):
                continue
            if ABS_IMPORT.match(ln):
                if ln not in imports:
                    imports.append(ln)
                continue
            body_lines.append(ln)
        body = "\n".join(body_lines)
        for name in DEF_RE.findall(body):
            if name in seen_defs:
                raise SystemExit("bundle: duplicate top-level name %r in %s and %s"
                                 % (name, seen_defs[name], rel))
            seen_defs[name] = rel
        body = body.replace("rank1._RANK1 = rank1._rank1_feed(",
                            'globals()["_RANK1"] = _rank1_feed(')
        body = body.replace("rank1._rank1_feed(", "_rank1_feed(")
        parts.append("\n# %s %s\n" % ("=" * 20, rel))
        parts.append(body.rstrip() + "\n")
        if rel == "suzu/config.py":
            parts.append(
                "\n\nclass _Config:\n    pass\n\n\n"
                "config = _Config()\n"
                "for _k in %r:\n"
                "    setattr(config, _k, globals()[_k])\n"
                "del _k\n" % (CONFIG_NAMES,))
    parts.insert(1, "\n" + "\n".join(imports) + "\n")
    parts.append(
        "\n\n# ==================== entry point\n"
        "if __name__ == \"__main__\":\n"
        "    sys.exit(main())\n")
    return "".join(parts)


def main():
    text = build()
    if "--check" in sys.argv:
        cur = OUT.read_text() if OUT.exists() else ""
        if cur != text:
            print("suzu.py is STALE - run tools/bundle.py")
            return 1
        print("suzu.py is up to date")
        return 0
    OUT.write_text(text)
    OUT.chmod(0o755)
    print("wrote %s (%d bytes)" % (OUT.relative_to(ROOT), len(text)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
