#!/usr/bin/env python3
"""render - turn captured ANSI terminal frames into PNGs / a GIF (PIL only).

Why: suzu draws with raw ANSI SGR, so the honest way to screenshot it is to
capture what it actually prints and rasterize that. No browser, no terminal
emulator screenshot tooling.

modes:
  render.py png  <frame.txt> <out.png> [--font TTF] [--px 15] [--pad 12]
                 [--bg RRGGBB] [--fg RRGGBB] [--scale F] [--cellh N]
  render.py gif  <manifest.json> <out.gif>   # [{"file":..., "ms":...}, ...]

The cell is the font's line box (ascent + descent), exactly as a terminal lays
it out: half-block glyphs (▀ ▄) tile with no seam only at that height, and
suzu's big decode digits are drawn with them. --cellh overrides it (a shorter
cell than the line box seams the blocks - the old theme shots used 14).

SGR supported: 0 reset, 7 reverse, 1/2 bold(dim-ish), 22, 39/49 defaults,
38;5;n / 48;5;n (256), 38;2;r;g;b / 48;2;r;g;b (truecolor). Other escapes
(cursor moves, alt-screen) are stripped - captures of --once frames and
tmux capture-pane -e carry none that matter.
"""

import json
import sys

from PIL import Image, ImageDraw, ImageFont

DEFAULT_FONT = "/home/d00l1n/.local/share/fonts/JetBrainsMonoNerdFont-Regular.ttf"
DEFAULT_BG = (13, 15, 20)
DEFAULT_FG = (204, 204, 204)

ANSI256 = [
    (0, 0, 0), (205, 49, 49), (13, 188, 121), (229, 229, 16), (36, 114, 200),
    (188, 63, 188), (17, 168, 205), (229, 229, 229), (102, 102, 102),
    (241, 76, 76), (35, 209, 139), (245, 245, 67), (59, 142, 234),
    (214, 112, 214), (41, 184, 219), (229, 229, 229)] + [
    # xterm 6x6x6 cube: levels 0, 95, 135, 175, 215, 255 (the old 8+10*n ramp
    # made every cube colour near-black: 231 came out #3a3a3a instead of #fff)
    ((0, 95, 135, 175, 215, 255)[(i - 16) // 36],
     (0, 95, 135, 175, 215, 255)[((i - 16) % 36) // 6],
     (0, 95, 135, 175, 215, 255)[(i - 16) % 6]) for i in range(16, 232)] + [
    (8 + (i - 232) * 10,) * 3 for i in range(232, 256)]


def parse_frame(text, bg, fg):
    rows, cur, cfg, cbg, bold = [], [], fg, bg, False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "\x1b":
            j = text.find("m", i)
            k = text.find("\x1b", i + 1)
            if k != -1 and (j == -1 or k < j):
                i = k  # non-SGR escape: skip it
                continue
            if j == -1:
                break
            params = text[i + 2:j].split(";")
            if params == [""]:
                params = ["0"]
            vals = [int(x) if x else 0 for x in params]
            p, idx = 0, 0
            while idx < len(vals):
                v = vals[idx]
                if v == 0:
                    cfg, cbg, bold = fg, bg, False
                elif v == 1:
                    bold = True
                elif v == 22:
                    bold = False
                elif v == 7:
                    cfg, cbg = cbg, cfg
                elif v == 39:
                    cfg = fg
                elif v == 49:
                    cbg = bg
                elif v in (38, 48):
                    if idx + 1 < len(vals) and vals[idx + 1] == 2:
                        c = tuple(vals[idx + 2:idx + 5])
                        idx += 4
                    elif idx + 1 < len(vals) and vals[idx + 1] == 5:
                        c = ANSI256[vals[idx + 2] % 256]
                        idx += 2
                    else:
                        c = fg
                        idx += 1
                    if v == 38:
                        cfg = c
                    else:
                        cbg = c
                idx += 1
            i = j + 1
            continue
        if ch == "\n":
            rows.append(cur)
            cur = []
        elif ch == "\r":
            pass
        else:
            cur.append((ch, cfg, cbg, bold))
        i += 1
    if cur:
        rows.append(cur)
    return rows


def bright(c):
    return tuple(min(255, int(v * 1.35 + 20)) for v in c)


def render(rows, font_path, px, pad, bg, cellh=None):
    font = ImageFont.truetype(font_path, px)
    adv = max(1, int(round(font.getlength("M"))))
    asc, desc = font.getmetrics()
    # The cell is the font's line box: a terminal draws ▀ as the top half of
    # the em box and ▄ as the bottom half, so only at ascent+descent do the
    # two tile with no seam. The old 2.05*adv cell cut the box and left a
    # black row between every row of blocks - the big digits looked shattered.
    cellh = cellh or int(asc + desc)
    off = (cellh - (asc + desc)) // 2
    cols = max((len(r) for r in rows), default=1)
    W, H = cols * adv + pad * 2, len(rows) * cellh + pad * 2
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    for y, row in enumerate(rows):
        cy = pad + y * cellh
        # per-cell background runs
        x = pad
        runs = []
        for ch, fgc, bgc, bold in row:
            runs.append((x, ch, fgc, bgc, bold))
            x += adv
        # backgrounds first
        for x0, ch, fgc, bgc, bold in runs:
            if bgc != bg:
                d.rectangle([x0, cy, x0 + adv - 1, cy + cellh - 1], fill=bgc)
        for x0, ch, fgc, bgc, bold in runs:
            if ch == " ":
                continue
            col = bright(fgc) if bold else fgc
            d.text((x0, cy + off), ch, font=font, fill=col)
    return img


def main():
    a = sys.argv[1:]
    mode = a.pop(0)
    font = DEFAULT_FONT
    px, pad, scale = 15, 12, 1.0
    cellh = None
    bg, fg = DEFAULT_BG, DEFAULT_FG
    args, opts = [], a
    i = 0
    while i < len(opts):
        if opts[i] == "--font":
            font = opts[i + 1]; i += 2
        elif opts[i] == "--px":
            px = int(opts[i + 1]); i += 2
        elif opts[i] == "--cellh":
            cellh = int(opts[i + 1]); i += 2
        elif opts[i] == "--pad":
            pad = int(opts[i + 1]); i += 2
        elif opts[i] == "--bg":
            bg = tuple(int(opts[i + 1][k:k + 2], 16) for k in (0, 2, 4)); i += 2
        elif opts[i] == "--fg":
            fg = tuple(int(opts[i + 1][k:k + 2], 16) for k in (0, 2, 4)); i += 2
        elif opts[i] == "--scale":
            scale = float(opts[i + 1]); i += 2
        else:
            args.append(opts[i]); i += 1
    if mode == "png":
        text = open(args[0], encoding="utf-8").read()
        img = render(parse_frame(text, bg, fg), font, px, pad, bg, cellh)
        if scale != 1.0:
            img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
        img.save(args[1])
        print("png: %s %dx%d" % (args[1], img.width, img.height))
    elif mode == "gif":
        frames = json.load(open(args[0]))
        imgs, durs = [], []
        for f in frames:
            text = open(f["file"], encoding="utf-8").read()
            imgs.append(render(parse_frame(text, bg, fg), font, px, pad, bg, cellh))
            durs.append(f.get("ms", 400))
        imgs[0].save(args[1], save_all=True, append_images=imgs[1:],
                     duration=durs, loop=0, optimize=True)
        print("gif: %s %d frames" % (args[1], len(imgs)))
    else:
        sys.exit("mode must be png|gif")


if __name__ == "__main__":
    main()
