"""RETINA Mart: agent-based supermarket simulator. Run this file.

Everything this program generates is SIMULATED data, not real sales.
Inputs come from the data layer built by loaders/loader.py (see data/data_sources.json).

config.py   - settings and assumptions
world.py    - store, shoppers, prices, stock
shopping.py - how NPCs shop, one day in the store
reports.py  - logs, summaries, scorecard, saved files
"""

import numpy as np
import pandas as pd

from config import *
from world import *
from shopping import *
from reports import *
from engine import run_simulation
from floor import summarize_walking, save_walking

# =================================================================== main
def run_pipeline(seed=None, days=None, save=True):
    """The normal year (no changes): simulate, print every summary and the scorecard, save files.
    Returns everything export.py needs."""
    r = run_simulation(seed=seed, days=days)
    store, shoppers, book, params = r.store, r.shoppers, r.book, r.params
    sales, misses, stock_stats, waste, effects = r.sales, r.misses, r.stock_stats, r.waste, r.effects
    memory_stats = r.memory_stats

    log = build_sales_log(sales, store, book)
    stockouts = build_stockout_log(misses, store, book)
    waste_log = build_waste_log(waste, store, book)
    money = money_summary(log, waste_log)
    promo_log = build_promo_log(effects, store, book)
    pair_stats = pair_report(sales, store)
    summarize_year(log)
    promo_lift(sales, store, book)
    summarize_promo(promo_log, sales, store, book)
    summarize_memory(memory_stats, promo_log)
    summarize_pairs(pair_stats, promo_log, log)
    summarize_stock(stockouts, stock_stats, log)
    summarize_money(money, log)
    summarize_walking(r.trips, store.floor)
    summarize_needs(log, r.trips, promo_log, shoppers)
    checks = scorecard(log, params, store, stock_stats, stockouts, money, promo_log, pair_stats)
    if save:
        save_outputs(log, store, shoppers, book, stockouts, waste_log, promo_log)
        save_walking(r.trips, store.floor)
    return {"results": r, "log": log, "stockouts": stockouts, "waste": waste_log, "money": money,
            "promo_log": promo_log, "scorecard": checks}


def main():
    run_pipeline()


if __name__ == "__main__":
    main()
