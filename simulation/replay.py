"""RETINA 2D replay: watch one simulated day in the store - people walking the real floor plan.

Records every shopper's steps for ONE day - SHOPPER_ENTER, MOVE_TO_DEPARTMENT, VIEW_PRODUCT,
PURCHASE, SUBSTITUTE, SKIP, STOCKOUT, CHECKOUT, LEAVE - with the route they walked (floor.py),
and writes a web page that replays it. The page only DRAWS: every route, decision, price, minute
and total comes from the simulation.

Run:   python simulation\\replay.py                       (day 33, the normal store)
       python simulation\\replay.py --day 70 --scenario scenarios\\example.json
Open:  data\\simulated\\<store>\\replay_day033.html  (double-click: opens in your browser)
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import *
import shopping
from engine import run_simulation, load_scenario
from floor import floor_plan

DEFAULT_DAY = 33                  # a Saturday in October (store has settled in)
GROUP_NAMES = ["low price sensitivity", "medium price sensitivity", "high price sensitivity"]
PALETTE = ["#7fb77e", "#e8c27a", "#8fb3d9", "#d99a8f", "#b9a3d6", "#7fc4c0", "#d6b38f", "#a9c47f",
           "#c7a0b8", "#9fb0bf", "#d9cf8f", "#8fc7a4", "#c4a77f", "#a3b7d6", "#d6a3a3", "#b0c4a0"]


def build_layout(store):
    """The floor plan (metres) plus a colour per department."""
    plan = floor_plan(store.floor)
    names = sorted(store.dept_names, key=lambda n: -store.departments[n].share)
    plan["colors"] = {n: PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}
    plan["lanes_open"] = [store.floor.lanes_open(h) for h in range(24)]
    return plan


# ================================================================ recording
def record_day(day, scenario=None):
    """Run the simulation up to `day` (or the scenario's length) with the recorder on for that day."""
    days = (scenario.get("days") or SIM_DAYS) if scenario else day + 1
    if not 0 <= day < days:
        raise ValueError(f"--day must be between 0 and {days - 1}")
    shopping.RECORDER.update(day=day, events=[])
    try:
        print(f"Simulating {day + 1} days to reach day {day} (this takes a little while)...")
        r = run_simulation(scenario, days=days, verbose=False)
        events = shopping.RECORDER["events"]
    finally:
        shopping.RECORDER.update(day=None, events=[])
    return r, events


def check_events(r, day, events):
    """The replay must tell the same story as the sales log."""
    s = r.sales[r.sales["day"] == day]
    ev = pd.DataFrame(events)
    units = ev.loc[ev["event"] == "PURCHASE", "qty"].sum()
    paid = ev.loc[ev["event"] == "CHECKOUT", "total"].sum()
    revenue = (s["quantity"] * s["unit_price"]).sum()
    trips = (ev["event"] == "SHOPPER_ENTER").sum()
    ok = units == s["quantity"].sum() and abs(paid - revenue) < 1 and trips == s["trip_id"].nunique() \
        + (ev.loc[ev["event"] == "CHECKOUT", "items"] == 0).sum()
    print(f"  events: {len(ev):,} | shoppers {trips} | units {int(units):,} | paid ${paid:,.2f}")
    print(f"  sales log that day: units {s['quantity'].sum():,} | revenue ${revenue:,.2f}")
    print(f"  CHECK {'PASS' if ok else 'FAIL'}: replay events match the sales log")
    print("  " + " | ".join(f"{k} {v:,}" for k, v in ev["event"].value_counts().items()))
    return ok


def page_data(r, day, events, scenario):
    p = r.store.products
    used = {e["product"] for e in events if e["product"] >= 0} | \
           {e["instead_of"] for e in events if "instead_of" in e}
    changed = []
    if scenario:
        ids = {str(c["product_id"]) for c in scenario.get("changes", [])}
        changed = [int(i) for i in p.index[p["product_id"].astype(str).isin(ids)]]
        used |= set(changed)
    date = pd.Timestamp(START_DATE) + pd.Timedelta(days=day)
    return {"store": PROFILE["name"], "day": day, "date": date.strftime("%a %d %b %Y"),
            "scenario": scenario.get("name") if scenario else None,
            "changed_products": changed,
            "layout": build_layout(r.store),
            "groups": GROUP_NAMES,
            "products": {int(i): str(p["product_name"][i]) for i in sorted(used)},
            "product_aisle": {int(i): int(r.store.floor.product_aisle[i]) for i in sorted(used)},
            "events": events}


def write_page(data, path):
    payload = json.dumps(data, separators=(",", ":"), default=lambda o: o.item() if hasattr(o, "item") else str(o))
    html = PAGE.replace("__DATA__", payload.replace("</", "<\\/"))
    path.write_text(html, encoding="utf-8")


def main():
    args = sys.argv[1:]
    day = int(args[args.index("--day") + 1]) if "--day" in args else DEFAULT_DAY
    scenario = load_scenario(args[args.index("--scenario") + 1]) if "--scenario" in args else None
    r, events = record_day(day, scenario)
    check_events(r, day, events)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = "" if not scenario else "_" + re.sub(r"[^a-z0-9]+", "-", scenario.get("name", "scenario").lower()).strip("-")[:30]
    data = page_data(r, day, events, scenario)
    events_file = OUTPUT_DIR / f"events_day{day:03d}{tag}.json"
    events_file.write_text(json.dumps({k: v for k, v in data.items() if k != "layout"},
                                      default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                           encoding="utf-8")
    page = OUTPUT_DIR / f"replay_day{day:03d}{tag}.html"
    write_page(data, page)
    print(f"\nSaved {events_file.name} (the events) and {page.name} (the replay) to {OUTPUT_DIR}")
    print(f"Open the replay: double-click {page}")


# ================================================================ the web page
PAGE_FILE = Path(__file__).parent / "replay_page.html"     # the page design (HTML + drawing code)
PAGE = PAGE_FILE.read_text(encoding="utf-8")

if __name__ == "__main__":
    main()
