"""RETINA engine: run the simulation for a SCENARIO - a list of what-if changes.
The normal year is simply a scenario with no changes, so baseline and experiments run through
exactly the same code.

A scenario is a small JSON file (see scenarios/), for example:
  {"name": "Mineral water +20%",
   "changes": [{"type": "price", "product_id": "...", "change_pct": 20, "start": "2025-11-03"}]}
Change types:
  price      change_pct (e.g. 20 = 20% dearer, -10 = 10% cheaper)
  promotion  discount_pct (e.g. 20), optional "segment": all | low_sensitivity |
             medium_sensitivity | high_sensitivity  (who sees the deal)
  remove     the product is taken off the shelves
  inventory  safety_days for this product (the store's stock buffer)
Every change has "product_id" and optional "start" / "end" dates (end = not included).

Run:  python simulation\\engine.py scenarios\\example.json
"""

import json
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from config import *
from world import *
from shopping import *
from reports import summarize_prices
from floor import build_floor, summarize_floor
import traffic
from households import assign_households, summarize_households

CHANGE_TYPES = ("price", "promotion", "remove", "inventory")
SEGMENTS = {"all": None, "low_sensitivity": 0, "medium_sensitivity": 1, "high_sensitivity": 2}


# ============================================================= scenario
def load_scenario(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def to_day(date, default):
    """'2025-11-03' -> simulation day number (0 = START_DATE)."""
    return default if date is None else (pd.Timestamp(date) - pd.Timestamp(START_DATE)).days


class Interventions:
    """A scenario's changes, checked and turned into product rows and simulation days."""

    def __init__(self, scenario, store, days):
        scenario = scenario or {}
        row_of = pd.Series(store.products.index.to_numpy(), index=store.products["product_id"].to_numpy())
        on_shelves = set(np.concatenate([a.product_idx for a in store.aisles.values()]).tolist())
        self.items = []
        for n, change in enumerate(scenario.get("changes", []), start=1):
            kind = change.get("type")
            if kind not in CHANGE_TYPES:
                raise ValueError(f"Change {n}: type must be one of {', '.join(CHANGE_TYPES)}")
            product_id = str(change.get("product_id"))
            if product_id not in row_of.index:
                raise ValueError(f"Change {n}: unknown product_id {product_id}")
            idx = int(row_of[product_id])
            if idx not in on_shelves:
                raise ValueError(f"Change {n}: product {product_id} is not on this store's shelves")
            start, end = to_day(change.get("start"), 0), to_day(change.get("end"), days)
            if not 0 <= start < end <= days:
                raise ValueError(f"Change {n}: dates must fall inside the {days}-day simulation")
            item = {"type": kind, "idx": idx, "start": start, "end": end}
            if kind == "price":
                pct = float(change["change_pct"])
                if pct <= -100:
                    raise ValueError(f"Change {n}: change_pct must be above -100")
                item["mult"] = 1 + pct / 100
            elif kind == "promotion":
                pct = float(change["discount_pct"])
                if not 0 < pct < 100:
                    raise ValueError(f"Change {n}: discount_pct must be between 0 and 100")
                segment = change.get("segment", "all")
                if segment not in SEGMENTS:
                    raise ValueError(f"Change {n}: segment must be one of {', '.join(SEGMENTS)}")
                item["mult"], item["group"] = 1 - pct / 100, SEGMENTS[segment]
            elif kind == "inventory":
                item["safety_days"] = float(change["safety_days"])
                if item["safety_days"] < 0:
                    raise ValueError(f"Change {n}: safety_days cannot be negative")
            self.items.append(item)
        self.change_days = {i["start"] for i in self.items} | {i["end"] for i in self.items}

    def active(self, day, kind):
        return [i for i in self.items if i["type"] == kind and i["start"] <= day < i["end"]]

    def available(self, n_products, day):
        """False for products taken off the shelves today."""
        ok = np.ones(n_products, dtype=bool)
        for item in self.active(day, "remove"):
            ok[item["idx"]] = False
        return ok


def shelf_prices(day, store, book, iv, n_groups):
    """Prices and promo flags for each price-sensitivity group (a targeted promotion is only
    seen by one group). Also returns the normal price and the public shelf price."""
    prices, regular, promo = prices_on(day, store, book)          # normal calendar
    change = np.ones(len(prices))
    for item in iv.active(day, "price"):
        change[item["idx"]] *= item["mult"]
    shelf, promo = prices.astype(float).copy(), promo.copy()
    for item in iv.active(day, "promotion"):
        if item["group"] is None:                                  # a promotion for everyone
            shelf[item["idx"]] = regular[item["idx"]] * item["mult"]
            promo[item["idx"]] = True
    shelf = (shelf * change).round(2)
    prices_g, promo_g = [], []
    for g in range(n_groups):
        price, flag = shelf.copy(), promo.copy()
        for item in iv.active(day, "promotion"):
            if item["group"] == g:                                 # a deal only this group sees
                price[item["idx"]] = round(regular[item["idx"]] * item["mult"] * change[item["idx"]], 2)
                flag[item["idx"]] = True
        prices_g.append(price)
        promo_g.append(flag)
    return prices_g, regular, promo_g, shelf


# ================================================================= run
@dataclass
class Results:
    store: object
    shoppers: object
    book: object
    inv: object
    params: dict
    sales: pd.DataFrame
    misses: pd.DataFrame
    stock_stats: pd.DataFrame
    waste: list
    effects: list
    memory_stats: list
    trips: pd.DataFrame = None      # one row per shopping trip: times, walking, checkout queue


class Simulation:
    """The store, its shoppers and the calendar. run_day() moves the store on by one day, so a
    run can stop and carry on later (live.py) and still give exactly the same data as one long run
    (every shopper and every day has its own fixed dice)."""

    def __init__(self, scenario=None, seed=None, days=None, verbose=True):
        if seed is not None:
            config.set_seed(seed)
        self.seed, self.scenario, self.verbose = config.SEED, scenario, verbose
        self.days = days or SIM_DAYS
        quiet = verbose is False
        catalog, cpi, params, departments = load_inputs()
        store = build_store(catalog, departments)
        if traffic.enabled():
            n = traffic.pool_size()                         # enough regulars to fill the schedule
        elif N_SHOPPERS == "auto":
            raise ValueError("n_shoppers 'auto' needs a traffic section in the store profile")
        else:
            n = N_SHOPPERS
        shoppers = create_shoppers(params, n, stream(STREAM_SHOPPERS))
        if HOUSEHOLDS:
            assign_households(shoppers, stream(STREAM_SHOPPERS, 2))   # household, income, food budget
        book = build_price_book(store, cpi, self.days, stream(STREAM_PROMOS))
        if verbose:
            summarize_store(store)
            summarize_shoppers(shoppers, params)
            if HOUSEHOLDS:
                summarize_households(shoppers)
            summarize_prices(store, book)
        with_output(quiet, plan_assortment, store, shoppers)
        with_output(quiet, build_aisles, store)
        store.partners = with_output(quiet, build_partners, store)
        store.aisle_partners = with_output(quiet, build_aisle_partners, store)
        seed_history(store, shoppers, stream(STREAM_SHOPPERS, 1))   # NPCs arrive with past purchases
        store.floor = build_floor(store, shoppers, params)      # where every aisle sits, how people walk
        if traffic.enabled():
            store.floor.hour_share = traffic.hour_shape()        # checkout lanes follow the schedule
        if verbose:
            summarize_floor(store.floor)
            if traffic.enabled():
                traffic.summarize_traffic(n)
        self.store, self.shoppers, self.book, self.params = store, shoppers, book, params
        self.inv = build_inventory(store, shoppers, stream(STREAM_STOCK))
        self.iv = Interventions(scenario, store, self.days)
        self.group_sensitivity = [shoppers.price_sensitivity[shoppers.sensitivity_group == g].mean()
                                  for g in range(SENSITIVITY_GROUPS)]
        self.day = 0                                             # next day to simulate
        self.memory = self.last_prices = self.tables = None
        self.daily_sales, self.daily_misses, self.daily_stock = [], [], []
        self.waste, self.effects, self.memory_stats, self.daily_trips = [], [], [], []

    def run_day(self):
        """Simulate the next day. Returns that day's receipt lines and trips."""
        day, store, inv, iv = self.day, self.store, self.inv, self.iv
        if day >= self.days:
            raise ValueError(f"The price calendar ends after {self.days} days")
        prices_g, regular, promo_g, shelf = shelf_prices(day, store, self.book, iv, SENSITIVITY_GROUPS)
        if day % 7 == 0:   # weekly: shoppers update what they remember paying
            if day == 0:
                self.memory = regular.copy()                     # start: remember the regular price
            else:
                self.memory = (1 - REF_PRICE_MEMORY) * self.memory + REF_PRICE_MEMORY * self.last_prices
            self.last_prices = shelf
            ratio = shelf / self.memory
            self.memory_stats.append((day, (ratio > 1.01).mean(), (ratio < 0.99).mean()))
        if day % 7 == 0 or day in iv.change_days:          # rebuild pick tables
            self.tables = choice_tables(store, [p / regular for p in prices_g],
                                        [p / self.memory for p in prices_g], self.group_sensitivity,
                                        iv.available(len(regular), day))
        if day % 7 == 0 and day > 0:   # weekly: the store re-forecasts from last week's sales
            last_week = pd.concat(self.daily_sales[-7:])
            update_forecast(inv, np.bincount(last_week["product_idx"], weights=last_week["quantity"],
                                             minlength=len(inv.stock)))
        apply_store_changes(day, iv, inv)
        spoiled = remove_spoiled(day, inv)                  # morning: throw away expired stock
        self.waste.extend((day, product, units) for product, units in spoiled.items())
        receive_deliveries(day, inv)
        for item in iv.active(day, "remove"):              # delisted: nothing left on the shelf
            inv.stock[item["idx"]] = 0
        sales_today, misses_today, stock_today, effects_today, trips_today = simulate_day(
            day, store, self.shoppers, self.params, prices_g, regular, promo_g, self.tables, inv)
        self.daily_trips.append(trips_today)
        self.effects.extend(effects_today)
        self.daily_sales.append(sales_today)
        self.daily_misses.append(misses_today)
        self.daily_stock.append(stock_today)
        self.day += 1
        return sales_today, trips_today

    def results(self):
        return Results(self.store, self.shoppers, self.book, self.inv, self.params,
                       pd.concat(self.daily_sales, ignore_index=True),
                       pd.concat(self.daily_misses, ignore_index=True), pd.DataFrame(self.daily_stock),
                       self.waste, self.effects, self.memory_stats,
                       pd.concat(self.daily_trips, ignore_index=True))


def run_simulation(scenario=None, seed=None, days=None, verbose=True):
    """Build the store and shoppers, apply the scenario's changes on their dates, simulate."""
    sim = Simulation(scenario, seed, days, verbose)
    if verbose:
        print(f"\nSimulating {sim.days} days" + (f" - scenario: {scenario.get('name')}" if scenario else "") + "...")
    while sim.day < sim.days:
        sim.run_day()
        if verbose and sim.day % 30 == 0:
            print(f"  day {sim.day}/{sim.days}", end="\r")
    if verbose:
        print()
    return sim.results()


def with_output(quiet, function, *args):
    """Run a setup step, hiding its printout when quiet."""
    if not quiet:
        return function(*args)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        return function(*args)


def apply_store_changes(day, iv, inv):
    """Remove products / change safety stock on the days a scenario says."""
    for item in iv.items:
        if item["type"] == "remove" and day == item["start"]:
            inv.sellable[item["idx"]] = False             # no more orders
            inv.batches.pop(item["idx"], None)
        if item["type"] == "remove" and day == item["end"]:
            inv.sellable[item["idx"]] = True              # back on the shelves
        if item["type"] == "inventory" and day in (item["start"], item["end"]):
            inv.safety_days[item["idx"]] = item["safety_days"] if day == item["start"] else SAFETY_DAYS
            set_order_levels(inv)


# ================================================================= demo
def main():
    scenario = load_scenario(sys.argv[1])
    results = run_simulation(scenario, days=scenario.get("days"), verbose=False)
    p, s = results.store.products, results.sales
    print(f"Scenario: {scenario['name']} ({len(s):,} receipt lines)")
    for item in Interventions(scenario, results.store, scenario.get("days") or SIM_DAYS).items:
        rows = s[s["product_idx"] == item["idx"]]
        before = rows[rows["day"] < item["start"]]["quantity"].sum() / max(item["start"], 1) * 7
        during = rows[(rows["day"] >= item["start"]) & (rows["day"] < item["end"])]["quantity"].sum()
        during = during / max(item["end"] - item["start"], 1) * 7
        print(f"  {item['type']:<9} {p['product_name'][item['idx']][:40]:<40} units/week before "
              f"{before:5.1f} | during {during:5.1f}")
    print("Next step (Step 6) compares this with a baseline run and gives ranges.")


if __name__ == "__main__":
    main()