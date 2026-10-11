"""RETINA experiments: does a change really help, and how sure are we?

An experiment runs the store TWICE with the same shoppers and the same dice (same seed):
  baseline = the normal store,   scenario = the store with the scenario's changes.
Because everything else is identical, the difference between the two runs is caused by the
change (not by luck). We repeat this with several seeds (different luck) and report the
average effect plus the range across runs.

Run:   python simulation\\experiment.py scenarios\\example.json
       python simulation\\experiment.py scenarios\\example.json --runs 10 --days 140
Saved: data/experiments/<date-time>-<name>/  (scenario, every run's numbers, summary, report)
Everything here is SIMULATED data from an assumed store - not a forecast of a real store.
"""

import json
import platform
import re
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

import config
from config import *
from engine import run_simulation, load_scenario, Interventions, SEGMENTS
import traffic

EXPERIMENTS_DIR = ROOT / "data" / "experiments"
DEFAULT_RUNS = 5
# Typical grocery price elasticity: Bijmolt, van Heerde & Pieters (2005) meta-analysis, mean -2.6
ELASTICITY_RANGE = (-4.0, -0.5)
SMALL_SALES = 30          # below this many units, results are mostly luck
GROUP_NAMES = {g: name for name, g in SEGMENTS.items() if g is not None}


# ============================================================ measuring
def line_money(r):
    """Revenue and profit (revenue - cost of goods) of every receipt line."""
    s, p = r.sales, r.store.products
    idx, day = s["product_idx"].to_numpy(), s["day"].to_numpy()
    cost = p["unit_cost"].to_numpy()[idx] * r.book.inflation[r.book.month_of_day[day], idx]
    revenue = s["quantity"].to_numpy() * s["unit_price"].to_numpy()
    return revenue, revenue - s["quantity"].to_numpy() * cost


def measure(r, items, window):
    """All the numbers we compare, for one run. window = (first day, last day + 1) of the changes."""
    s, p = r.sales, r.store.products
    day, idx, q = s["day"].to_numpy(), s["product_idx"].to_numpy(), s["quantity"].to_numpy()
    revenue, profit = line_money(r)
    group = r.shoppers.sensitivity_group[s["shopper_id"].to_numpy()]
    miss_day, miss_idx = r.misses["day"].to_numpy(), r.misses["product_idx"].to_numpy()
    effects = pd.DataFrame(r.effects, columns=["day", "shopper", "product_idx", "effect", "n"])
    waste = pd.DataFrame(r.waste, columns=["day", "product_idx", "units"])
    w_idx, w_day = waste["product_idx"].to_numpy(), waste["day"].to_numpy()
    waste_cost = waste["units"].to_numpy() * p["unit_cost"].to_numpy()[w_idx] * \
        r.book.inflation[r.book.month_of_day[w_day], w_idx]

    def between(days, start, end):
        return (days >= start) & (days < end)

    w = between(day, *window)
    w_waste = waste_cost[between(w_day, *window)].sum()
    walk = effects[between(effects["day"], *window) & (effects["effect"] == "walked_away")]
    store = {"trips": s.loc[w, "trip_id"].nunique(),
             "units": q[w].sum(),
             "revenue": revenue[w].sum(),
             "profit (after waste)": profit[w].sum() - w_waste,
             "waste cost": w_waste,
             "out-of-stock moments": between(miss_day, *window).sum(),
             "walk-aways": walk["n"].sum()}

    changes = []
    department = p["department"].to_numpy()
    for it in items:
        cw = between(day, it["start"], it["end"])
        target = cw & (idx == it["idx"])
        aisle = r.store.aisles[r.store.aisle_of[it["idx"]]].product_idx
        in_aisle = cw & np.isin(idx, aisle)
        rest = in_aisle & (idx != it["idx"])
        dept = cw & (department[idx] == department[it["idx"]])
        m = {"units": q[target].sum(),
             "revenue": revenue[target].sum(),
             "profit": profit[target].sum(),
             "rest of aisle units": q[rest].sum(),
             "rest of aisle profit": profit[rest].sum(),
             "department units": q[dept].sum(),
             "department profit": profit[dept].sum(),
             "out-of-stock moments": (between(miss_day, it["start"], it["end"])
                                      & (miss_idx == it["idx"])).sum()}
        for g, name in GROUP_NAMES.items():
            m[f"units, {name}"] = q[target & (group == g)].sum()
        aisle_units = np.bincount(idx[in_aisle], weights=q[in_aisle], minlength=len(p))[aisle]
        changes.append({"metrics": m, "aisle_units": pd.Series(aisle_units, index=aisle)})
    return {"store": store, "changes": changes}


def basket_differences(base, scen, start, end):
    """How many trips and shoppers got a different basket between start and end."""
    def lines(r):
        s = r.sales[(r.sales["day"] >= start) & (r.sales["day"] < end)]
        return s.groupby(["shopper_id", "trip_id", "product_idx"])["quantity"].sum()
    both = pd.concat([lines(base), lines(scen)], axis=1, keys=["a", "b"]).fillna(0)
    changed = both.index[both["a"] != both["b"]]
    return {"trips": both.index.get_level_values(1).nunique(),
            "trips changed": changed.get_level_values(1).nunique(),
            "shoppers changed": changed.get_level_values(0).nunique()}


# ============================================================ summarizing
def summarize(pairs):
    """pairs = [(baseline, scenario), ...] one per run -> average, % change and range across runs."""
    b = np.array([x[0] for x in pairs], dtype=float)
    e = np.array([x[1] for x in pairs], dtype=float)
    diff = e - b
    return {"baseline": b.mean(), "scenario": e.mean(), "change": diff.mean(),
            "change_pct": diff.mean() / b.mean() * 100 if b.mean() else float("nan"),
            "low": diff.min(), "high": diff.max(),          # smallest and largest change in one run
            "same_direction": bool((diff > 0).all() or (diff < 0).all() or (diff == 0).all())}


def describe(item, products):
    name = products["product_name"][item["idx"]]
    if item["type"] == "price":
        what = f"price {(item['mult'] - 1) * 100:+.0f}%"
    elif item["type"] == "promotion":
        who = "everyone" if item["group"] is None else GROUP_NAMES[item["group"]]
        what = f"promotion {(1 - item['mult']) * 100:.0f}% off for {who}"
    elif item["type"] == "remove":
        what = "taken off the shelves"
    else:
        what = f"safety stock {item['safety_days']:g} days (normal {SAFETY_DAYS:g})"
    return f"{what} - {name}"


def check_change(item, m):
    """Does the store react the way research says it should? Returns (text, passed)."""
    units = m["units"]
    if item["type"] == "price":
        price_pct = (item["mult"] - 1) * 100
        e = units["change_pct"] / price_pct
        ok = ELASTICITY_RANGE[0] <= e <= ELASTICITY_RANGE[1] and units["same_direction"]
        return (f"price elasticity {e:.1f} (research: usually {ELASTICITY_RANGE[0]:g} to "
                f"{ELASTICITY_RANGE[1]:g}; higher price -> fewer units)", ok)
    if item["type"] == "promotion":
        lift = units["scenario"] / units["baseline"] if units["baseline"] else float("inf")
        if item["group"] is None:
            return (f"promotion lift {lift:.1f}x (research: often 2x or more)",
                    lift > 1.2 and units["same_direction"])
        target = m[f"units, {GROUP_NAMES[item['group']]}"]
        mine = target["change_pct"]
        others = [m[f"units, {n}"]["change_pct"] for g, n in GROUP_NAMES.items() if g != item["group"]]
        if not np.isfinite(mine):
            return "targeted group normally buys none of this product - too few sales to judge", False
        others_txt = ", ".join(f"{x:+.0f}%" if np.isfinite(x) else "n/a" for x in others)
        return (f"targeted group units {mine:+.0f}% vs other groups {others_txt} "
                f"(only the targeted group should react)", mine > max(others) and target["same_direction"])
    if item["type"] == "remove":
        ok = units["scenario"] == 0 and m["rest of aisle units"]["change"] > 0
        return "no sales once removed, similar products in the aisle pick up sales", ok
    oos = m["out-of-stock moments"]["change"]
    lower = item["safety_days"] < SAFETY_DAYS
    return (f"less safety stock -> more empty shelves" if lower else
            f"more safety stock -> fewer empty shelves", (oos > 0) if lower else (oos <= 0))


# ============================================================ main
def run_experiment(scenario, runs=DEFAULT_RUNS, days=None):
    days = days or scenario.get("days") or SIM_DAYS
    first_seed = config.SEED
    seeds = [first_seed + k for k in range(runs)]
    per_run, items, store_obj = [], None, None
    print(f"Experiment: {scenario.get('name', 'unnamed')} | {runs} runs x {days} days "
          f"(seeds {seeds[0]}-{seeds[-1]})")
    for k, seed in enumerate(seeds, start=1):
        t = time.time()
        base = run_simulation(None, seed=seed, days=days, verbose=False)
        scen = run_simulation(scenario, seed=seed, days=days, verbose=False)
        if items is None:
            store_obj = scen.store
            items = Interventions(scenario, store_obj, days).items
            window = (min(i["start"] for i in items), max(i["end"] for i in items))
        per_run.append({"seed": seed, "baseline": measure(base, items, window),
                        "scenario": measure(scen, items, window),
                        "before": basket_differences(base, scen, 0, window[0]),
                        "during": basket_differences(base, scen, *window)})
        print(f"  run {k}/{runs} (seed {seed}) done in {time.time() - t:.0f}s")
    config.set_seed(first_seed)
    return days, seeds, items, window, store_obj, per_run


def report(scenario, days, seeds, items, window, store_obj, per_run):
    """Average effect and range across runs, with checks. Returns (text lines, summary dict)."""
    p = store_obj.products
    dates = pd.Timestamp(START_DATE) + pd.to_timedelta([window[0], window[1] - 1], unit="D")
    out, summary = [], {"changes": [], "store": {}, "checks": []}
    say = out.append

    def table(names, getter):
        say(f"    {'measure':<28}{'baseline':>12}{'scenario':>12}{'change':>9}   change in each run (lowest to highest)")
        rows = {}
        for name in names:
            sm = summarize([(getter(r["baseline"])[name], getter(r["scenario"])[name]) for r in per_run])
            rows[name] = sm
            money = "$" if any(w in name for w in ("revenue", "profit", "cost")) else ""
            fmt = lambda v: f"{money}{v:,.0f}"
            signed = lambda v: f"{'-' if v < 0 else '+'}{money}{abs(v):,.0f}"
            pct = f"{sm['change_pct']:+.1f}%" if np.isfinite(sm["change_pct"]) else "-"
            rng = f"{signed(sm['low'])} to {signed(sm['high'])}"
            sure = "" if sm["change"] == 0 else ("  same direction every run" if sm["same_direction"]
                                                 else "  NOT SURE (direction differs between runs)")
            say(f"    {name:<28}{fmt(sm['baseline']):>12}{fmt(sm['scenario']):>12}{pct:>9}   {rng}{sure}")
        return rows

    for n, item in enumerate(items):
        start = (pd.Timestamp(START_DATE) + pd.Timedelta(days=item["start"])).strftime("%d %b %Y")
        say(f"\nChange {n + 1}: {describe(item, p)}")
        say(f"  from {start} for {item['end'] - item['start']} days (numbers = totals over those days, "
            f"average of {len(seeds)} runs)")
        names = list(per_run[0]["baseline"]["changes"][n]["metrics"])
        m = table(names, lambda run: run["changes"][n]["metrics"])
        # Where did the units go (or come from)? Products in the same aisle that changed most
        diff = pd.concat([r["scenario"]["changes"][n]["aisle_units"].sub(
                              r["baseline"]["changes"][n]["aisle_units"], fill_value=0)
                          for r in per_run], axis=1).fillna(0).mean(axis=1).drop(item["idx"], errors="ignore")
        top = diff[diff.abs() >= 0.5].abs().sort_values(ascending=False).index[:3]
        movers = [(p["product_name"][j], diff[j]) for j in top]
        if movers:
            say("    biggest movers in the same aisle: " +
                "; ".join(f"{name[:30]} {d:+.0f} units" for name, d in movers))
        if m["units"]["baseline"] < SMALL_SALES:
            say(f"    NOTE: only {m['units']['baseline']:.0f} units normally sold in this period - "
                f"small numbers jump around; use more runs (--runs 10) or a longer change")
        text, ok = check_change(item, m)
        say(f"    CHECK {'PASS' if ok else 'WARN'}: {text}")
        summary["checks"].append({"change": n + 1, "check": text, "passed": bool(ok)})
        summary["changes"].append({"change": describe(item, p), "start_day": item["start"],
                                   "end_day": item["end"], "metrics": m,
                                   "aisle_movers": [{"product": a, "units": float(b)} for a, b in movers]})

    say(f"\nWhole store, {dates[0]:%d %b %Y} to {dates[1]:%d %b %Y} (all changes together)")
    summary["store"] = table(list(per_run[0]["baseline"]["store"]), lambda run: run["store"])
    during = [r["during"] for r in per_run]
    shoppers = np.mean([d["shoppers changed"] for d in during])
    trips = np.mean([d["trips changed"] / d["trips"] for d in during]) * 100
    pool = traffic.pool_size() if traffic.enabled() else N_SHOPPERS
    say(f"    shoppers affected: {shoppers:,.0f} of {pool:,} ({shoppers / pool * 100:.0f}%) | "
        f"trips with a different basket: {trips:.1f}%")
    same_before = all(r["before"]["trips changed"] == 0 for r in per_run)
    text = "before the first change, baseline and scenario are identical (same shoppers, same dice)"
    say(f"    CHECK {'PASS' if same_before else 'WARN'}: {text}")
    summary["checks"].append({"change": "store", "check": text, "passed": same_before})
    summary["store"]["shoppers affected"] = float(shoppers)
    summary["store"]["trips changed %"] = float(trips)
    passed = sum(c["passed"] for c in summary["checks"])
    say(f"\n{passed}/{len(summary['checks'])} checks passed. SIMULATED results for an assumed store "
        f"({PROFILE['name']}), not a forecast of a real store.")
    return out, summary


def save(scenario, days, seeds, per_run, lines, summary):
    slug = re.sub(r"[^a-z0-9]+", "-", scenario.get("name", "experiment").lower()).strip("-")[:40]
    folder = EXPERIMENTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}-{slug}"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in per_run:
        for part in ("baseline", "scenario"):
            for name, v in r[part]["store"].items():
                rows.append((r["seed"], part, "store", name, round(float(v), 2)))
            for n, c in enumerate(r[part]["changes"], start=1):
                for name, v in c["metrics"].items():
                    rows.append((r["seed"], part, f"change {n}", name, round(float(v), 2)))
    pd.DataFrame(rows, columns=["seed", "run", "scope", "measure", "value"]).to_csv(
        folder / "runs.csv", index=False)
    info = {"name": scenario.get("name"), "created": datetime.now().isoformat(timespec="seconds"),
            "data_label": "SIMULATED", "runs": len(seeds), "seeds": seeds, "days": days,
            "store_profile": PROFILE, "summary": summary,
            "computer": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "system": platform.platform()},
            "note": "Same seeds give the same results on this computer; another operating "
                    "system may give slightly different numbers."}
    with open(folder / "summary.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    with open(folder / "scenario.json", "w", encoding="utf-8") as f:
        json.dump(scenario, f, indent=2)
    (folder / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    return folder


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    args = sys.argv[2:]
    runs = int(args[args.index("--runs") + 1]) if "--runs" in args else DEFAULT_RUNS
    days = int(args[args.index("--days") + 1]) if "--days" in args else None
    scenario = load_scenario(sys.argv[1])
    days, seeds, items, window, store_obj, per_run = run_experiment(scenario, runs, days)
    lines, summary = report(scenario, days, seeds, items, window, store_obj, per_run)
    print("\n".join(lines))
    folder = save(scenario, days, seeds, per_run, lines, summary)
    print(f"\nSaved to {folder}")


if __name__ == "__main__":
    main()
