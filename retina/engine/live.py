"""RETINA live: run the store day by day, like a real store - not a whole year in one go.

Run it with the `retina` command (retina/cli.py), which picks the store first:
  retina                       simulate the NEXT day (opens a new store the first time)
  retina --days 7              the next 7 days
  retina --watch               show each hour's batch of NPCs walking in, slowly
  retina --status              where the store is now
  retina --reset               close this store and start a new one
  retina --days 7 --view       also show the NPCs walking in your browser (view.html)
  Another store format:  retina --store supercenter   (or python -m retina.engine.studio to choose)
  New store only:  --scale 0.1 (1 NPC = 10 real shoppers)   --per-week 13583 (real shoppers a week)

Each run carries on where the last one stopped: shelves, stock, prices and every NPC's habits,
pantry and price memory are saved in data/live/<store>/state.pkl. New data is ADDED to:
  receipts.csv   one row per product line (date, hour, trip, NPC, product, quantity, price, promo...)
  trips.csv      one row per trip (arrival, walk, checkout queue, leaving time)
  stockouts.csv  every empty shelf a shopper met, and what they did
  hourly.csv     one row per opening hour: NPCs, real-store equivalent, items, revenue
Running 3 x 10 days gives exactly the same data as one 30-day run (every NPC has fixed dice).
Every row is SIMULATED.
"""

import argparse
import pickle
import shutil
import time

import numpy as np
import pandas as pd

from . import config
from .config import ROOT, PROFILE, TRAFFIC
from . import traffic
from . import shopping
from .engine import Simulation
from .reports import build_sales_log, build_stockout_log

LIVE_DIR = ROOT / "data" / "live" / PROFILE["id"]
STATE = LIVE_DIR / "state.pkl"


def use_run_dir(mind):
    """fast runs keep data/live/<store>/; thinking runs get their own folder (<store>_mix, <store>_llm),
    so engine-only data and thinking-NPC data never mix in one file."""
    global LIVE_DIR, STATE
    LIVE_DIR = ROOT / "data" / "live" / (PROFILE["id"] if mind == "fast" else f"{PROFILE['id']}_{mind}")
    STATE = LIVE_DIR / "state.pkl"


def clock(minutes):
    m = np.asarray(minutes)
    return [f"{int(x // 60):02d}:{int(x % 60):02d}" for x in m]


def open_store(scale, per_week):
    """A new store with its NPC pool. Traffic settings are saved with it."""
    if not traffic.enabled():
        raise SystemExit("This store profile has no 'traffic' section - live mode needs one.")
    if scale:
        TRAFFIC["npc_scale"] = scale
    if per_week:
        TRAFFIC["real_shoppers_per_week"] = per_week
    print(f"Opening a new store: {PROFILE['name']} (building shelves, floor and NPCs)...")
    sim = Simulation(verbose=False)
    sim.traffic_settings = dict(TRAFFIC)
    traffic.summarize_traffic(len(sim.shoppers.shopper_id))
    return sim


def load_store():
    mind = LIVE_DIR.name[len(PROFILE["id"]) + 1:] or "fast"
    reset = (f"This saved store was made by an older RETINA version. Start a new one: "
             f"retina --store {PROFILE['id']} --mind {mind} --reset")
    try:
        with open(STATE, "rb") as f:
            sim = pickle.load(f)
    except (ModuleNotFoundError, AttributeError):      # saved before v2 moved the code into retina/engine
        raise SystemExit(reset)
    if getattr(sim.shoppers, "home", None) is None:
        raise SystemExit(reset)
    config.set_seed(sim.seed)                    # same dice as when the store opened
    TRAFFIC.update(sim.traffic_settings)
    return sim


def save_store(sim):
    # keep only last week's sales (the store's weekly forecast needs them); the rest is on disk
    sim.daily_sales = sim.daily_sales[-7:]
    sim.daily_misses, sim.daily_stock, sim.daily_trips = [], [], []
    sim.waste, sim.effects, sim.memory_stats = [], [], []
    tmp = STATE.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(sim, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(STATE)                           # never leave a half-written state behind


def append(df, name):
    path = LIVE_DIR / name
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def hourly_table(sim, day, sales, trips):
    scale = TRAFFIC["npc_scale"]
    hour = (trips["arrive_min"] // 60).astype(int)
    shape = traffic.hour_shape()
    lines = sales.assign(revenue=sales["quantity"] * sales["unit_price"])
    by_hour = pd.DataFrame({
        "npcs": trips.groupby(hour).size(),
        "items": lines.groupby("hour")["quantity"].sum(),
        "revenue": lines.groupby("hour")["revenue"].sum().round(2),
        "avg_wait_min": trips.groupby(hour)["wait_min"].mean().round(1),
    }).reindex(np.flatnonzero(shape)).fillna(0)
    by_hour[["npcs", "items"]] = by_hour[["npcs", "items"]].astype(int)
    by_hour["planned_npcs"] = (traffic.expected_npcs(day) * shape[by_hour.index]).round(1)
    by_hour["real_equivalent"] = (by_hour["npcs"] / scale).round()
    by_hour.index.name = "hour"
    by_hour = by_hour.reset_index()
    by_hour.insert(0, "date", sim.book.dates[day].date())
    by_hour["data_label"] = "SIMULATED"
    return by_hour


def show_day(sim, day, hourly, watch, pace, trips=None):
    date = sim.book.dates[day]
    scale = TRAFFIC["npc_scale"]
    total = int(hourly["npcs"].sum())
    print(f"\n{date:%a %d %b %Y}  (day {day + 1}) - real store ~{traffic.real_per_day(day):,.0f} shoppers "
          f"-> {total} NPCs (1 NPC = {1 / scale:.0f} people)")
    for r in hourly.itertuples():
        print(f"  {r.hour:02d}:00  {'#' * int(r.npcs):<30} {int(r.npcs):>3} NPCs  {int(r.items):>4} items  "
              f"${r.revenue:>8,.2f}  queue {r.avg_wait_min:.1f} min")
        if watch:
            time.sleep(pace)
    print(f"  day total: {total} trips | {int(hourly['items'].sum()):,} items | ${hourly['revenue'].sum():,.2f}"
          f"  (real-store equivalent ~${hourly['revenue'].sum() / scale:,.0f})")
    if trips is not None and "mission" in trips:
        kinds = trips["mission"].value_counts()
        print(f"  trips: {kinds.get('top_up', 0)} top-up | {kinds.get('routine', 0)} routine | "
              f"{kinds.get('stock_up', 0)} stock-up | {int((trips['spent'] > trips['trip_budget']).sum())} over budget")


def write_view(sim, day, events, first_time=False):
    """view.html = the replay page for the day just simulated. It plays the day, then reloads itself
    to pick up the next day as soon as the simulator has written it."""
    from . import replay
    from types import SimpleNamespace
    data = replay.page_data(SimpleNamespace(store=sim.store), day, events, None)
    data["live"] = True
    path = LIVE_DIR / "view.html"
    replay.write_page(data, path)
    if first_time:
        print(f"  NPCs walking: open {path.relative_to(ROOT)} in your browser (it follows the days by itself)")
        try:
            import webbrowser
            webbrowser.open(path.resolve().as_uri())
        except Exception:
            pass


def status(sim):
    done = sim.day
    print(f"Store: {PROFILE['name']} | NPC pool {len(sim.shoppers.shopper_id):,} | "
          f"1 NPC = {1 / TRAFFIC['npc_scale']:.0f} real shoppers")
    if done:
        print(f"Simulated {done} days: {sim.book.dates[0]:%d %b %Y} to {sim.book.dates[done - 1]:%d %b %Y}. "
              f"Next day: {sim.book.dates[done]:%a %d %b %Y}" if done < sim.days else
              f"Simulated all {done} days.")
    else:
        print("Open, no days simulated yet.")
    for name in ("receipts.csv", "trips.csv", "stockouts.csv", "hourly.csv"):
        path = LIVE_DIR / name
        if path.exists():
            print(f"  {name:<14} {path.stat().st_size / 1e6:6.1f} MB")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the RETINA store day by day.")
    ap.add_argument("--days", type=int, default=1, help="how many days to simulate now (default 1)")
    ap.add_argument("--watch", action="store_true", help="print each hour slowly, as it happens")
    ap.add_argument("--pace", type=float, default=0.4, help="seconds per hour with --watch")
    ap.add_argument("--status", action="store_true", help="show where the store is and stop")
    ap.add_argument("--reset", action="store_true", help="close the saved store and open a new one")
    ap.add_argument("--scale", type=float, help="new store only: NPCs per real shopper (0.1 = 1 in 10)")
    ap.add_argument("--per-week", type=int, help="new store only: real shoppers a week")
    ap.add_argument("--view", action="store_true", help="write view.html: the NPCs walking, day by day")
    ap.add_argument("--wait", type=float, default=0, help="seconds to wait between days (to watch --view)")
    ap.add_argument("--mind", choices=("fast", "mix", "llm"), default="fast",
                    help="fast = engine only; mix = a panel of thinking NPCs; llm = every NPC thinks (small runs)")
    ap.add_argument("--panel", type=int, default=50, help="new mix run only: thinking NPCs (default 50)")
    ap.add_argument("--provider", default="auto", help="LLM teacher: auto (chain), groq, gemini, ollama, cerebras, "
                                                       "or fake (fixed rules, no LLM - for tests)")
    ap.add_argument("--model", help="model name for that provider")
    ap.add_argument("--yes", action="store_true", help="run even if the estimate is above today's free LLM limits")
    ap.add_argument("--report", action="store_true", help="thinking runs: print the panel-vs-crowd report and stop")
    args = ap.parse_args(argv)
    use_run_dir(args.mind)

    if args.reset and LIVE_DIR.exists():
        shutil.rmtree(LIVE_DIR)
        print(f"Closed the old store ({LIVE_DIR.relative_to(ROOT)} removed).")
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    if STATE.exists():
        if args.scale or args.per_week:
            print("Note: --scale / --per-week only apply to a new store (use --reset). Ignored.")
        sim = load_store()
    else:
        sim = open_store(args.scale, args.per_week)
        save_store(sim)
    if args.mind != "fast":
        from retina.mind import run as mind_run
        sim.mind = mind_run.attach(sim, args.mind, LIVE_DIR, args.panel, args.provider, args.model)
    if args.status:
        status(sim)
        return
    if args.report:
        if args.mind == "fast":
            raise SystemExit("--report is for thinking runs (--mind mix or --mind llm).")
        mind_run.finish(sim.mind, LIVE_DIR)
        return

    days = min(args.days, sim.days - sim.day)
    if days <= 0:
        print(f"The price calendar ends after {sim.days} days. Use --reset, or raise simulation.days "
              f"in the store profile.")
        return
    if args.mind != "fast" and not mind_run.check_estimate(sim, days, args.yes):
        return
    t = time.time()
    first = sim.day
    for k in range(days):
        day = sim.day
        if args.view:
            shopping.RECORDER.update(day=day, events=[])     # write down every step of today's shoppers
        try:
            sales, trips = sim.run_day()
            events = list(shopping.RECORDER["events"])
        finally:
            shopping.RECORDER.update(day=None, events=[])
        misses = sim.daily_misses[-1]
        hourly = hourly_table(sim, day, sales, trips)
        show_day(sim, day, hourly, args.watch, args.pace, trips)
        log = build_sales_log(sales, sim.store, sim.book)
        log.insert(log.columns.get_loc("department") + 1, "aisle",
                   sim.store.products["aisle"].to_numpy()[sales["product_idx"].to_numpy()])
        append(log.drop(columns=["cpi_item_code"]), "receipts.csv")
        trips = trips.assign(date=sim.book.dates[day].date(), arrive_time=clock(trips["arrive_min"]),
                             leave_time=clock(trips["leave_min"]), data_label="SIMULATED")
        append(trips.round(2), "trips.csv")
        if len(misses):
            append(build_stockout_log(misses, sim.store, sim.book).assign(data_label="SIMULATED"), "stockouts.csv")
        append(hourly, "hourly.csv")
        if args.mind != "fast":                           # memory first: a newer memory is rolled back
            mind_run.day_done(sim.mind, day, LIVE_DIR)    # on load, an older one could not catch up
        save_store(sim)                                   # after every day: safe to stop at any time
        if args.view:
            write_view(sim, day, events, first_time=(k == 0))
            if args.wait and k < days - 1:
                print(f"  (watching day {day + 1} in the browser - next day in {args.wait:.0f}s; Ctrl+C stops safely)")
                time.sleep(args.wait)
    print(f"\nSaved {days} day(s) ({sim.book.dates[first]:%d %b} to {sim.book.dates[sim.day - 1]:%d %b %Y}) "
          f"in {time.time() - t:.0f}s -> {LIVE_DIR.relative_to(ROOT)}. Run again to continue.")
    if args.mind != "fast":
        mind_run.finish(sim.mind, LIVE_DIR)


if __name__ == "__main__":
    main()
