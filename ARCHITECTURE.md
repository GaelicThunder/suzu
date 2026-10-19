# suzu — architecture

`suzu/` is the source tree. `suzu.py` is the single-file release, **generated**
from it by `tools/bundle.py`; never edit `suzu.py` by hand. `test_suzu.py`
tests the package against a synthetic engine and never touches `:30001`.

    python3 tools/bundle.py      # rebuild suzu.py
    python3 tools/bundle.py --check
    python3 test_suzu.py         # exit 0 = all good

## The measurement that dictates everything

llama.cpp's `/metrics` counters do not move during a generation: they are
flushed at request completion. Measured 6.2 s of an active slot produced delta
0 on `tokens_predicted_total` and on every `spec_decode_*` counter while the
model was emitting 20.5 tok/s. So the panel is split by source of truth:

| what | where | why |
|---|---|---|
| live decode rate | `/slots` `next_token.n_decoded` | advances every token |
| prefill progress | `/slots` `n_prompt_tokens_processed` | live on llama.cpp |
| KV / prefix cache | `/slots` | llama.cpp exposes neither on `/metrics` |
| acceptance, tau, p0..pN | `/metrics` | only closed at request end, shown with age |
| vLLM prefill rate | `prompt_tokens_total / request_prefill_time_seconds_sum` | `/slots` on vLLM is synthetic |

Every engine adapter exists to answer one question honestly: *which of these
numbers is live, which is reconstructed, and which is absent?* Panels label the
difference (`window` vs `session`, `last` on a stale prefix cache, `n/a`).

## Module map

```
suzu/
  __init__.py        re-exports the public names (import suzu == old single file)
  __main__.py        python -m suzu
  cli.py             argv, terminal size, raw mode, keys, SIGWINCH, --once
  config.py          runtime paths/URLs set by the CLI (TF_REQLOG, HERMES_DB, LANES_URL, RANK1_URL)
  ansi.py            SGR helpers, escape constants, ANSI regexes
  text.py            vis_len, pad, clip, human, si_bytes, fmt_hms, ago
  themes.py          THEMES tables, THEME_ORDER, ramp families, Theme, colour mode
  fetch/
    http.py          HTTP GET + Prometheus textformat parsing
    engine.py        Sample (one tick of counters), llama.cpp/vLLM probes
    memory.py        /proc memory, process RSS, llama.cpp /slots
    gpu.py           nvidia-smi rows, rocm-smi fallback
    info.py          server cmdline, model config.json, layer topology
    logtail.py       find and tail the engine's log
    vllm.py          vLLM synthetic slots + ctx max
    tabby.py         TabbyAPI/ExLlamaV3 reconstruction from log + /props
    tensorfold.py    TensorFold /health, counters, /experts, hermes usage
    rank1.py         rank-1 HTTP poller, RoCE link
  render/
    core.py          View state: construction, theme cycle, ramps, smoothing, step()
    widgets.py       drawing primitives: bars, sparks, boxes, gauges, big digits
    panels.py        panel builders: cluster, lanes, model strip, experts, heatmap, slots
    frames.py        full-screen composition: spec / perf / pipes / layers + the grid
    view.py          View = WidgetsMixin + PanelsMixin + FramesMixin + CoreMixin
tools/
  bundle.py          builds suzu.py from the package
```

`View` is one class assembled from four mixins, so each concern has its own
file and methods call each other through `self`, exactly as they did inside the
old 4k-line class. No mixin imports another; `view.py` is the only place that
knows the composition order.

## One tick

`cli.main()` → `view.frame(width, height)` → `frame_spec()`:

1. `step()` fetches `/metrics`, `/slots`, GPU, memory, rank 1, lanes and the
   server info, and returns `(Sample, live)`.
2. `Sample` diffs against the previous one: live gen/prefill rates, the spec
   window (`alpha`, `tau`, per-position), TTFT, restart detection.
3. `frame_spec()` builds rows top-down: header, model strip, big digits,
   acceptance, number grid, the content-aware column grid, lanes, expert heat,
   log/layer panel. Everything is clipped to `width`; the final list is clamped
   to `height`.

Nothing is cached per terminal size: every frame is rebuilt from the size it is
given, so a resize takes effect on the next frame (and `SIGWINCH` redraws at
once instead of waiting out the interval).

## The laws (tests enforce most of them)

1. Live rate comes from `/slots`, never from `/metrics` during a generation.
2. On vLLM, never derive prefill from `n_prompt_tokens_processed` (it is KV
   occupancy): bank `prompt_tokens_total` and `request_prefill_time_seconds_sum`
   across ticks and divide once both have landed.
3. A negative `n_decoded` delta is a new request, not a stall: drop the window.
   Cumulative counters going backwards are a restart: clear every history.
4. An idle slot resets `n_prompt_tokens_cache` to 0: keep the last true prefix
   value and mark it `last` instead of showing a red 0%.
5. Only ticks that actually decoded enter the gen history; a prefill is not a
   slow generation.
6. Every rendered line is `<= width` visible characters. `clip()` closes a
   colour run only when it cut one — plain (mono) output stays ANSI-free.
7. Never return more rows than the terminal has, and never cut a box
   mid-frame: a row budget drops optional boxes whole (chart -> per-position ->
   history, and averages only in the stacked layout), the big digits degrade to
   a stat line below 26 rows, and the number grid is skipped below 20. Test 7b
   counts box tops against bottoms at 4 widths x 7 heights.
8. The grid is content-aware: each column asks for the width its own text needs
   (engine rows, cluster probe, averages line), the layout uses three columns
   only when they fit, else two, else stacked, and slack is shared. No fixed
   38/32 fractions, no silent mid-word cuts.

## Add an engine

1. New file `suzu/fetch/<engine>.py` exposing either
   `fetch_<engine>(port) -> (metrics_dict, slots)` or
   `synth_<engine>(metrics_dict, port, yzq) -> (metrics_dict, slots)`.
   Return `None` when the engine is not there.
2. Detect it in `CoreMixin.step()` where the metrics are parsed (the
   TensorFold branch is the template: a distinctive counter name decides).
3. Add a test section in `test_suzu.py` using synthetic counters (sections 10
   and 11 are the TabbyAPI and TensorFold templates).
4. Do not touch `render/`: panels only read `Sample` fields and the adapter's
   output, never engine-specific code.

## Add a theme

1. Entry in `THEMES` (six role colours: title, accent, head, frame, label,
   value) plus `scale`: five **distinct** colours from bad to good. Test 8
   fails a theme whose ramp repeats a colour — a bar must be its own legend.
2. Append the name to `THEME_ORDER` (mono stays out: it is reachable by name
   and chosen automatically for pipes / `NO_COLOR`).
3. If the theme needs its own RGB ramps for bars/heat/VU/flow, add a family
   set in `themes.py` (`GOLD_FAMILY`, `PHOSPHOR_FAMILY` are the pattern) and a
   branch in `CoreMixin.ramps()`.
4. Check it with `python3 suzu.py --once --theme <name>` against
   `demo/fake_engine.py`.

## Add a panel

1. Builder in `panels.py`: returns a list of strings, each `<= width - 2`
   visible characters (the box adds one space of padding and the borders).
2. Call it from `frames.py` with the column width and wrap it:
   `self.box(title, body, width)`.
3. If it is optional, gate it on the rows left (`height - len(L)`), so a short
   terminal drops it instead of scrolling the top off screen.
4. Titles degrade with width (see `heat_title`); a title longer than the box is
   clipped mid-word.

## Tests

`test_suzu.py` drives the real `View` against a synthetic llama.cpp engine:

1-5   live rate, spec window, request reset, restart, idle prefix cache
6-7   layout at every width (no overflow) and height clamp
8-9   every theme renders coloured with a distinct ramp; cycling wraps
10    TabbyAPI: `/metrics` 404, everything rebuilt from log + gateway
11    TensorFold: counters, `/health` fill, rank-1 feed, RoCE, expert grid
12    usage DB, token heatmap, dithered bars

Patches go through `patch(name, value)` at the top of the file: it writes the
fake into every module that imported that name, because after the split a bare
`suzu.http_get = fake` only changes the package facade, not the binding the
code under test looks up.
