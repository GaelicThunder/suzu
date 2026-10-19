#!/usr/bin/env python3
"""shots - capture the README assets from a live fake engine, reproducibly.

Why: the README screenshots are the honest output of the program, so they are
captured from what suzu actually prints (ANSI in a tmux pane of a known size)
and rasterized by tools/render.py. Nothing here needs a terminal emulator,
a browser or a GUI box.

usage:
  tools/shots.py <outdir> [--w 140] [--h 60] [--theme yuzu]
                          [--model NAME] [--draft NAME] [--tokens-db PATH]
                          [--port 8099] [--gif-frames 30] [--gif-step 0.45]

writes <outdir>/main-2.txt, perf-2.txt, pipe-2.txt, layers-2.txt and the
gif-*.txt frames, then renders hero.png, perf.png, pipeline.png, layers.png
and hero.gif into <outdir>. Copy the ones you want into assets/.
"""
import json
import os
import re
import subprocess
import sys
import time
import urllib.request

SUZU = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSION = "suzu-shots"
plain = lambda t: re.sub(r"\x1b\[[0-9;]*m", "", t)

args = sys.argv[1:]
if not args:
    sys.exit(__doc__)
OUT = args.pop(0)
W, H, THEME = 140, 60, "yuzu"
MODEL, DRAFT = "DeepSeek-V4.1-Flash-1017", "DeepSeek-V4.1-Flash-DSpark-k64"
TOKENS_DB, PORT = None, 8099
GIFN, GIFSTEP = 30, 0.45
i = 0
while i < len(args):
    a = args[i]
    v = args[i + 1] if i + 1 < len(args) else ""
    if a == "--w":
        W = int(v)
    elif a == "--h":
        H = int(v)
    elif a == "--theme":
        THEME = v
    elif a == "--model":
        MODEL = v
    elif a == "--draft":
        DRAFT = v
    elif a == "--tokens-db":
        TOKENS_DB = v
    elif a == "--port":
        PORT = int(v)
    elif a == "--gif-frames":
        GIFN = int(v)
    elif a == "--gif-step":
        GIFSTEP = float(v)
    else:
        sys.exit("unknown option %r" % a)
    i += 2
os.makedirs(OUT, exist_ok=True)


def http(path):
    with urllib.request.urlopen("http://127.0.0.1:%d%s" % (PORT, path), timeout=2) as r:
        return r.read().decode()


def tmux(*a, **kw):
    return subprocess.run(["tmux", "-L", "suzu-shots"] + list(a),
                          capture_output=True, text=True, **kw)


engine = subprocess.Popen(
    [sys.executable, os.path.join(SUZU, "demo", "fake_engine.py"),
     "--port", str(PORT), "--model", MODEL, "--spec-draft-model", DRAFT,
     "--spec-type", "draft-mtp", "--spec-draft-n-max", "5", "--lanes"],
    cwd=SUZU, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(200):
        try:
            if json.loads(http("/health")).get("status") == "ok":
                break
        except Exception:
            pass
        time.sleep(0.05)
    else:
        sys.exit("fake engine did not come up on :%d" % PORT)

    tmux("kill-server")
    cmd = [sys.executable, os.path.join(SUZU, "suzu.py"), "0.3",
           "--port", str(PORT), "--theme", THEME]
    if TOKENS_DB:
        cmd += ["--tokens-db", TOKENS_DB]
    tmux("new-session", "-d", "-s", SESSION, "-x", str(W), "-y", str(H),
         " ".join(cmd), check=True)

    def wait_busy():
        for _ in range(300):
            try:
                s = json.loads(http("/slots"))[0]
                if s.get("is_processing") and s.get("next_token", {}).get("n_decoded", 0) > 20:
                    return True
            except Exception:
                pass
            time.sleep(0.1)
        return False

    def cap(name):
        txt = tmux("capture-pane", "-e", "-p", "-t", SESSION).stdout
        with open(os.path.join(OUT, name + ".txt"), "w", encoding="utf-8") as f:
            f.write(txt)
        p = plain(txt)
        m = re.search(r"DECODE\s+([\d.]+)\s*t/s", p)
        print("%-12s %s dec=%s" % (name, "busy" if "WORKING" in p else "idle",
                                   m.group(1) if m else "?"))
        return txt

    time.sleep(12.0)          # let the first request close: counters + acceptance exist
    wait_busy()
    time.sleep(0.4)
    cap("main-1")
    time.sleep(1.5)
    wait_busy()
    time.sleep(0.4)
    cap("main-2")
    for key, stem in (("p", "perf"), ("b", "pipe"), ("g", "layers")):
        tmux("send-keys", "-t", SESSION, key)
        time.sleep(2.0)
        wait_busy()
        time.sleep(0.4)
        cap(stem + "-1")
        time.sleep(1.0)
        cap(stem + "-2")

    # GIF frames: back to the main view (g and b are toggles, not modes)
    tmux("send-keys", "-t", SESSION, "g")
    time.sleep(0.4)
    tmux("send-keys", "-t", SESSION, "b")
    time.sleep(1.0)
    t0 = time.monotonic()
    for n in range(GIFN):
        target = t0 + n * GIFSTEP
        now = time.monotonic()
        if target > now:
            time.sleep(target - now)
        cap("gif-%02d" % n)
finally:
    tmux("kill-server")
    engine.terminate()
    try:
        engine.wait(timeout=5)
    except subprocess.TimeoutExpired:
        engine.kill()

# render: PNGs at the documented cell size, GIF at a lighter one
render = os.path.join(SUZU, "tools", "render.py")
for stem, png in (("main-2", "hero.png"), ("perf-2", "perf.png"),
                  ("pipe-2", "pipeline.png"), ("layers-2", "layers.png")):
    subprocess.run([sys.executable, render, "png", os.path.join(OUT, stem + ".txt"),
                    os.path.join(OUT, png), "--px", "14", "--pad", "12"], check=True)

rates = []
for n in range(GIFN):
    p = plain(open(os.path.join(OUT, "gif-%02d.txt" % n), encoding="utf-8").read())
    m = re.search(r"DECODE\s+([\d.]+)\s*t/s", p)
    rates.append(float(m.group(1)) if m else 0.0)
best, score = 0, -1.0
for s in range(0, max(1, GIFN - 14)):
    v = sum(rates[s:s + 15])
    if v > score:
        best, score = s, v
manifest = [{"file": os.path.join(OUT, "gif-%02d.txt" % n), "ms": 450}
            for n in range(best, best + 15)]
mpath = os.path.join(OUT, "manifest-hero.json")
json.dump(manifest, open(mpath, "w"), indent=1)
subprocess.run([sys.executable, render, "gif", mpath, os.path.join(OUT, "hero.gif"),
                "--px", "12", "--pad", "12"], check=True)
print("done: %s" % OUT)
