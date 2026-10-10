#!/usr/bin/env python3
"""fake_tokens - generate a synthetic token-usage DB for suzu's Token Gen heatmap.

Produces the schema suzu reads (the one the Hermes assistant writes:
session_model_usage / session_model_usage_v21) filled with ~16 weeks of
plausible per-day traffic across a few models, so the GitHub-style
"generated tokens per day" heatmap and the all-models totals render
without a real harness database.

usage: fake_tokens.py [--out PATH] [--days 110]
"""

import argparse
import datetime as dt
import random
import sqlite3
import time

AP = argparse.ArgumentParser()
AP.add_argument("--out", default="demo-tokens.db")
AP.add_argument("--days", type=int, default=110)
A = AP.parse_args()

MODELS = [
    # model, weight of traffic, in-per-out ratio, cache-hit ratio
    ("qwen3.5-32b-instruct", 0.55, 28.0, 0.72),
    ("qwen3.5-14b",          0.30, 22.0, 0.64),
    ("qwen3.5-0.8b-draft",   0.15, 12.0, 0.40),
]

random.seed(42)
today = dt.date.today()
rows = []
sid = 0
for age in range(A.days, -1, -1):
    d = today - dt.timedelta(days=age)
    # weekends lighter, a quiet stretch, gentle growth towards today
    dow = d.weekday()
    base = 1.0 if dow < 5 else 0.35
    if 40 <= age <= 48:
        base *= 0.15  # a holiday
    growth = 0.6 + 0.8 * (1 - age / max(1, A.days))
    daily_out = int(random.uniform(0.4e6, 2.2e6) * base * growth)
    if daily_out < 20000:
        continue
    for m, w, in_ratio, cache in MODELS:
        out = int(daily_out * w * random.uniform(0.8, 1.2))
        inp = int(out * in_ratio * random.uniform(0.9, 1.1))
        cached = int(inp * cache * random.uniform(0.9, 1.0))
        calls = max(3, int(out / random.uniform(9e3, 18e3)))
        # split the day across a few sessions
        for _ in range(random.randint(1, 4)):
            sid += 1
            f = d.strftime("%Y-%m-%d")
            h1 = random.uniform(8, 20)
            h2 = min(23.9, h1 + random.uniform(0.5, 4))
            t1 = time.mktime(time.strptime("%s %02d:%02d" % (f, int(h1), int(h1 % 1 * 60)),
                                           "%Y-%m-%d %H:%M"))
            t2 = time.mktime(time.strptime("%s %02d:%02d" % (f, int(h2), int(h2 % 1 * 60)),
                                           "%Y-%m-%d %H:%M"))
            share = random.uniform(0.2, 0.6)
            rows.append(("sess-%05d" % sid, m, int(calls * share), int(inp * share),
                         int(cached * share), int(out * share), int(out * share * 0.2),
                         t1, t2))

con = sqlite3.connect(A.out)
con.execute("create table if not exists session_model_usage "
            "(session_id text, model text, api_call_count int, input_tokens int, "
            "cache_read_tokens int, output_tokens int, reasoning_tokens int, "
            "first_seen real, last_seen real)")
con.execute("delete from session_model_usage")
con.executemany("insert into session_model_usage values (?,?,?,?,?,?,?,?,?)", rows)
con.commit()
con.close()
print("fake_tokens: %d rows -> %s (%d days, %d models)" % (len(rows), A.out, A.days, len(MODELS)))
