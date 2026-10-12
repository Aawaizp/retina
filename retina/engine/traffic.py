"""RETINA store traffic: how many real shoppers walk in each hour, and how many NPCs we send.

A real store's footfall (shoppers per week) is split into days (weekends busier) and hours (a
real hourly curve), then scaled down by npc_scale: with 0.1, one NPC stands for 10 real people,
so 100 real shoppers at 10 am -> about 10 NPCs. The exact count each day varies a little, like
real arrivals (Poisson), but follows the schedule.

Who walks in: the store has a pool of regular customers (NPCs). Each one plans their next trip
from their real habits. Each day the schedule says how many come; those whose trip is most
due come first (if the day is busier than planned, a few come a little early).

Sources (labels in the store profile):
  shoppers per week   PUBLIC_STATISTIC  FMI Supermarket Facts: 13,583 transactions a week (2018)
  weekend vs weekday  PUBLIC_STATISTIC  USDA ERS / ATUS 2014-17: 16% shop on a weekend day, 13% on a weekday
  hourly curve        PUBLIC_STATISTIC  Instacart order hours (online orders), cut to opening hours
  hourly_counts       REAL_OBSERVED     your own door-counter numbers, if you have them

Run:  python -m retina.engine.traffic      (prints the schedule for a weekday and a Saturday)
"""

import json

import numpy as np
import pandas as pd

from .config import (TRAFFIC, START_DATE, START_WEEKDAY, SHOPPER_PARAMS_FILE, SHOPPER_PROFILES_FILE,
                    STREAM_TRAFFIC, stream)

_cache = {}


def enabled():
    return bool(TRAFFIC) and TRAFFIC.get("enabled", True)


def hour_shape():
    """Share of a day's shoppers who arrive in each hour (24 values, 0 while closed)."""
    if "shape" not in _cache:
        raw = np.zeros(24)
        counts = TRAFFIC.get("hourly_counts")
        source = counts or json.load(open(SHOPPER_PARAMS_FILE, encoding="utf-8"))["orders_by_hour"]
        for h, v in source.items():
            raw[int(h)] = v
        open_h, close_h = TRAFFIC["open_hours"]
        hours = np.arange(24)
        raw[(hours < open_h) | (hours >= close_h)] = 0
        _cache["shape"] = raw / raw.sum()
    return _cache["shape"]


def hour_table():
    """{hour: share} for the open hours - the form sample_from() takes."""
    shape = hour_shape()
    return {str(h): float(v) for h, v in enumerate(shape) if v > 0}


def is_weekend(day):
    return (START_WEEKDAY + day) % 7 >= 5


def real_per_day(day):
    """Real shoppers expected in the store on simulation day `day`."""
    w = TRAFFIC["weekend_vs_weekday"]
    counts = TRAFFIC.get("hourly_counts")
    if counts:                                            # your counts = a typical weekday
        weekday = float(sum(counts.values()))
    else:
        weekday = TRAFFIC["real_shoppers_per_week"] / (5 + 2 * w)
    return weekday * (w if is_weekend(day) else 1.0)


def expected_npcs(day):
    return real_per_day(day) * TRAFFIC["npc_scale"]


def npcs_today(day):
    """How many NPCs walk in today: the schedule plus normal day-to-day luck. Its own dice, so a
    what-if experiment never changes how many people come."""
    return int(stream(STREAM_TRAFFIC, day).poisson(expected_npcs(day)))


def average_daily_npcs():
    return float(np.mean([expected_npcs(d) for d in range(7)]))


def pool_size():
    """Regular customers needed so their usual trips (real gaps between trips) fill the schedule."""
    if SHOPPER_PROFILES_FILE.exists():
        gaps = pd.read_csv(SHOPPER_PROFILES_FILE, usecols=["days_between_orders"])["days_between_orders"]
        trips_per_day = (1 / np.maximum(1, gaps)).mean()
    else:                                       # no profiles file: the real distribution of gaps
        dist = json.load(open(SHOPPER_PARAMS_FILE, encoding="utf-8"))["shopper_profiles"]["avg_days_between_orders"]
        dist = json.loads(dist.replace("'", '"')) if isinstance(dist, str) else dist
        gaps, shares = np.maximum(1, np.array([float(k) for k in dist])), np.array(list(dist.values()), dtype=float)
        trips_per_day = float((shares / shares.sum() / gaps).sum())
    return int(round(average_daily_npcs() / trips_per_day))


def choose_visitors(day, next_visit_day, n):
    """The n NPCs whose trip is most due (ties broken by today's own dice)."""
    key = stream(STREAM_TRAFFIC, day, 1).random(len(next_visit_day))
    return np.sort(np.lexsort((key, next_visit_day))[:n])


def weekday_shares():
    """Share of a week's shoppers on each weekday (0 = Monday) - for the scorecard."""
    w = TRAFFIC["weekend_vs_weekday"]
    raw = np.array([1, 1, 1, 1, 1, w, w], dtype=float)
    return raw / raw.sum()


def summarize_traffic(pool):
    t = TRAFFIC
    scale = t["npc_scale"]
    shape = hour_shape()
    peak = int(shape.argmax())
    wd, we = real_per_day((5 - START_WEEKDAY) % 7 + 2), real_per_day((5 - START_WEEKDAY) % 7)
    source = "your hourly counts" if t.get("hourly_counts") else f"{t['real_shoppers_per_week']:,} a week"
    print(f"\nStore traffic ({source}), open {t['open_hours'][0]}:00-{t['open_hours'][1]}:00, "
          f"1 NPC = {1 / scale:.0f} real shoppers")
    print(f"  weekday ~{wd:,.0f} real -> {wd * scale:.0f} NPCs | weekend day ~{we:,.0f} -> {we * scale:.0f} NPCs | "
          f"busiest hour {peak}:00 ~{we * shape[peak]:.0f} real -> {we * shape[peak] * scale:.0f} NPCs")
    print(f"  NPC pool: {pool:,} regular customers (each shops on their own real habits)")


def main():
    pool = pool_size()
    summarize_traffic(pool)
    shape, scale = hour_shape(), TRAFFIC["npc_scale"]
    monday = (7 - START_WEEKDAY) % 7
    days = {"Weekday": monday, "Saturday": monday + 5}
    print(f"\n{'hour':>6} " + "".join(f"{name + ' real':>15}{'NPCs':>6}" for name in days))
    for h in np.flatnonzero(shape):
        cells = ""
        for d in days.values():
            real = real_per_day(d) * shape[h]
            cells += f"{real:>15.0f}{real * scale:>6.1f}"
        print(f"{h:>4}:00 {cells}  " + "#" * int(round(real * scale)))
    print(f"\nFirst day {pd.Timestamp(START_DATE):%a %d %b %Y}: {npcs_today(0)} NPCs (expected {expected_npcs(0):.0f})")


if __name__ == "__main__":
    main()
