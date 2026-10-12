"""RETINA studio: choose a store format, then watch its NPCs walk and shop while the data is made.

  python simulation\\studio.py                      menu: pick a store, 7 days, browser view
  python simulation\\studio.py --store supercenter --days 3
  python simulation\\studio.py --compare            one table: every store format side by side

Store formats (stores/*.json) are GENERIC formats sized with public numbers - "supercenter"
(Walmart Supercenter-type size and traffic), "supermarket", "neighborhood market", "convenience"
(7-Eleven-type). They are not copies of any real retailer and predict no real store.

What happens: the chosen store opens (or carries on where it stopped), and every simulated day
  - prints each hour's NPCs, items and sales in this window,
  - adds the receipts, trips and stockouts to data/live/<store>/ (CSV files),
  - writes data/live/<store>/view.html: the floor plan with every NPC walking, picking products
    and queuing - the page plays the day and then loads the next one by itself.
Stop any time with Ctrl+C: everything up to the last finished day is saved.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORES = ROOT / "stores"


def stores():
    out = []
    for f in sorted(STORES.glob("*.json")):
        p = json.loads(f.read_text(encoding="utf-8"))
        t, fm = p.get("traffic", {}), p.get("format", {})
        out.append({"id": p["id"], "name": p["name"], "like": fm.get("like", ""),
                    "per_week": t.get("real_shoppers_per_week"), "hours": t.get("open_hours"),
                    "assortment": fm.get("assortment_share", 1.0), "lanes": p.get("floor", {}).get("checkout_lanes")})
    return out


def menu(options):
    print("\nRETINA studio - choose a store format (all SIMULATED, generic formats)\n")
    for i, o in enumerate(options, 1):
        hours = f"{o['hours'][0]:02d}-{o['hours'][1]:02d}h" if o["hours"] else ""
        print(f"  {i}. {o['name']:<28} {o['per_week'] or 0:>7,} shoppers/week | {hours} | "
              f"{o['assortment']:.0%} of the range | {o['lanes']} lanes")
        if o["like"]:
            print(f"     {o['like']}")
    while True:
        pick = input(f"\nStore number (1-{len(options)}): ").strip()
        if pick.isdigit() and 1 <= int(pick) <= len(options):
            return options[int(pick) - 1]
        print("  type one of the numbers above")


def compare(days):
    """Run every format for a few days (fresh, same seed) and print one comparison table."""
    code = ("import json, engine, traffic\n"
            "from config import PROFILE\n"
            f"sim = engine.Simulation(verbose=False)\n"
            f"while sim.day < {days}: sim.run_day()\n"
            "r = sim.results(); s = r.sales; t = r.trips; f = sim.store.floor\n"
            "rev = float((s.quantity * s.unit_price).sum())\n"
            "dep = sim.store.products['department'].values[s.product_idx]\n"
            "mix = (s.assign(d=dep).groupby('d').quantity.sum() / s.quantity.sum()).sort_values(ascending=False)\n"
            "print(json.dumps({'name': PROFILE['name'], 'npcs': len(sim.shoppers.shopper_id),"
            f" 'trips_day': len(t) / {days}, 'items_trip': float(s.quantity.sum() / max(len(t), 1)),"
            " 'spend_trip': rev / max(len(t), 1),"
            f" 'revenue_day': rev / {days}, 'real_revenue_day': rev / {days} / (traffic.TRAFFIC['npc_scale'] if traffic.enabled() else 1),"
            " 'products': int(sum(len(d.product_idx) for d in sim.store.departments.values())),"
            " 'floor': f'{f.width:.0f} x {f.depth:.0f} m', 'minutes': float(t.minutes_in_store.mean()),"
            " 'wait': float(t.wait_min.mean()), 'top': ', '.join(f'{k} {v:.0%}' for k, v in mix.head(3).items())}))\n")
    rows = []
    for o in stores():
        print(f"  running {o['name']} for {days} days...")
        env = dict(os.environ, RETINA_STORE=o["id"])
        out = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).parent, env=env,
                             capture_output=True, text=True)
        line = [l for l in out.stdout.splitlines() if l.startswith("{")]
        if line:
            rows.append(json.loads(line[-1]))
        else:
            print(out.stderr[-500:])
    print(f"\nStore formats side by side ({days} days each, same seed; SIMULATED)\n")
    print(f"{'store':<28}{'NPCs':>6}{'trips/day':>10}{'items/trip':>11}{'$/trip':>8}{'$/day':>9}"
          f"{'real $/day':>12}{'products':>9}{'min':>6}{'wait':>6}  floor / top departments")
    for r in rows:
        print(f"{r['name']:<28}{r['npcs']:>6}{r['trips_day']:>10.0f}{r['items_trip']:>11.1f}{r['spend_trip']:>8.1f}"
              f"{r['revenue_day']:>9,.0f}{r['real_revenue_day']:>12,.0f}{r['products']:>9,}{r['minutes']:>6.1f}"
              f"{r['wait']:>6.1f}  {r['floor']} | {r['top']}")
    print("\n'real $/day' = NPC sales scaled back up to the real-store traffic (1 NPC = 10 shoppers).")


def main():
    ap = argparse.ArgumentParser(description="Choose a store format and watch it run.")
    ap.add_argument("--store", help="store id (file name in stores/), e.g. supercenter")
    ap.add_argument("--days", type=int, default=7, help="days to run now (default 7)")
    ap.add_argument("--wait", type=float, default=None, help="seconds between days (default: time to watch one day)")
    ap.add_argument("--no-view", action="store_true", help="no browser view, just the data")
    ap.add_argument("--reset", action="store_true", help="start this store again from day 1")
    ap.add_argument("--compare", action="store_true", help="compare all store formats in one table")
    args = ap.parse_args()
    if args.compare:
        compare(min(args.days, 14))
        return
    options = stores()
    chosen = next((o for o in options if o["id"] == args.store), None) if args.store else menu(options)
    if chosen is None:
        sys.exit(f"No store '{args.store}'. Choose one of: {', '.join(o['id'] for o in options)}")
    hours = (chosen["hours"][1] - chosen["hours"][0]) if chosen["hours"] else 16
    wait = args.wait if args.wait is not None else (0 if args.no_view else hours * 60 / 15 + 6)   # page plays 15 min/s
    print(f"\nOpening {chosen['name']} - {args.days} day(s)" + ("" if args.no_view else
          f", one day every ~{wait:.0f}s so you can watch it in the browser"))
    os.environ["RETINA_STORE"] = chosen["id"]        # the store profile every module loads
    sys.path.insert(0, str(Path(__file__).parent))
    import live                                      # imported AFTER choosing: it reads the chosen store
    sys.argv = ["live.py", "--days", str(args.days)] + ([] if args.no_view else ["--view", "--wait", str(wait)]) \
        + (["--reset"] if args.reset else [])
    try:
        live.main()
    except KeyboardInterrupt:
        print("\nStopped. Everything up to the last finished day is saved - run again to continue.")


if __name__ == "__main__":
    main()
