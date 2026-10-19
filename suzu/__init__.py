"""suzu - a small bell for your LLM engine.

The public entry points are re-exported here so `import suzu` behaves like the
old single-file module; the code itself lives in the modules named below.
See ARCHITECTURE.md for the map and how to add an engine or a theme.
"""
import os
import time

from .ansi import ESC, RST, fg, bg, BOLD, DIM, ANSI_RE, SGR_RE
from .config import TF_REQLOG, HERMES_DB, LANES_URL, RANK1_URL
from .text import vis_len, pad, clip, human, si_bytes, fmt_hms, ago
from .themes import THEMES, THEME_ORDER, BLOCKS, DITHER, PULSE, _rgb, _lerp_rgb, _grad, _rgb256, _TRUECOLOR, _COLOR_MODE, set_color_mode, _have_truecolor, _col, HEAT_COOL, HEAT_GOLD, VU_COOL, VU_GOLD, FLOW_COOL, FLOW_GOLD, GOLD_FAMILY, TRACK_BG_COOL, TRACK_BG_GOLD, WHITE_FG, _bg_col, BIG_GLYPHS, BIGFONT, THEME_STATE, _theme_read, _theme_save, Theme
from .fetch.engine import Sample, fetch_yzq, yzq_live_sse, fetch_yzq_lanes
from .fetch.gpu import _GPU_SMI, fetch_gpu, _fetch_rocm_gpu
from .fetch.http import http_get, METRIC_RE, POS_RE, parse_metrics, gl, g
from .fetch.info import server_info, model_config_path, load_model_arch
from .fetch.logtail import active_log, log_mtime, tail
from .fetch.memory import _MEMINFO, fetch_memory, fetch_slots
from .fetch.rank1 import Rank1Feed, _rank1_feed, _RANK1, fetch_rank1, _ROCE, roce_link
from .fetch.tabby import _TABBY_PROPS, _TAB_ZERO, _TABBY_LOG, _TABBY_MONO, _TABBY_KFILE, _tabby_k0, _TABBY_RATIO, _TS_RE, _TAB_GEN, _TAB_Q, _TAB_P, _TAB_G, _TAB_C, _TAB_D, proc_env, tabby_env, tabby_props, tabby_parse, tabby_scan, tabby_scale, synth_tabby
from .fetch.tensorfold import _TF_MONO, _TF_LAST, tf_last_record, _TF_ALL, tf_lifetime, _TF_DELTA, _lanes_host, _HERMES_USE, _HERMES_DAYS, hermes_days, hermes_usage, _TF_FILL, _TF_HEALTH, tf_fill, tf_last_from_counters, synth_tensorfold, tf_env
from .fetch.vllm import _VLLM_POOL, _VLLM_CTXMAX, vllm_pool, vllm_ctxmax, synth_slots_vllm
from .render.view import View
from .cli import main
