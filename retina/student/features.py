"""Features of one shelf decision: one row per option the NPC could pick (sold-out options are not
choosable and are left out). Used the SAME way in training, in collect mode and in the live mind.

  NPC      price sensitivity, household size, kids, income fifth, weekly food budget, re-buy rate
  item     why it is on the list (need / repeat / new / partner), trip type
  option   own price change log(price / regular), price level vs the other options, deal, how deep,
           bought before (and how often), the NPC's usual, the engine's pick, a best seller,
           base popularity, price vs what the NPC paid last time, sold out last time
  mixes    deal x sensitivity, price change x sensitivity, usual x re-buy rate (for the logit;
           the tree model finds such mixes by itself)
The option's position (A, B, C...) is NOT a feature: the student cannot have a position bias.
"""

import math

import numpy as np

ROUTES = ("need", "repeat", "new", "partner")
MISSIONS = ("top_up", "routine", "stock_up")
FEATURES = ["price_change", "price_level", "deal", "deal_depth", "bought_before", "times_log", "usual",
            "engine_pick", "best_seller", "popularity_log", "vs_last_paid", "sold_out_before",
            "sensitivity", "household", "kids", "income", "budget_log", "rebuy_rate",
            "route_need", "route_repeat", "route_new", "route_partner",
            "mission_top_up", "mission_routine", "mission_stock_up",
            "deal_x_sensitivity", "change_x_sensitivity", "usual_x_rebuy"]


def npc_numbers(shoppers, s):
    """The NPC's numeric traits (stored with every label)."""
    return {"sensitivity": float(shoppers.price_sensitivity[s]),
            "household": int(shoppers.household_size[s]) if shoppers.household_size is not None else 2,
            "kids": int(shoppers.kids[s]) if shoppers.kids is not None else 0,
            "income": int(shoppers.income_quintile[s]) if shoppers.income_quintile is not None else 2,
            "budget": float(shoppers.weekly_budget[s]) if shoppers.weekly_budget is not None else 150.0,
            "rebuy": float(shoppers.reorder_prob[s])}


def choosable(options):
    return [o for o in options if o["in_stock"]]


def matrix(npc, route, mission, options, engine_product):
    """Feature matrix (options x FEATURES, float32) for the choosable options, and those options."""
    opts = choosable(options)
    if not opts:
        return np.zeros((0, len(FEATURES)), dtype=np.float32), opts
    prices = np.array([max(o["price"], 0.01) for o in opts])
    level = np.log(prices) - np.log(prices).mean()
    rows = []
    for o, lv in zip(opts, level):
        change = math.log(max(o["price"], 0.01) / max(o["regular"], 0.01))
        last = o.get("last_price")
        vs_last = math.log(max(o["price"], 0.01) / last) if last else 0.0
        bought = 1.0 if o["times"] > 0 else 0.0
        usual = 1.0 if o.get("usual") else 0.0
        sens = npc["sensitivity"]
        rows.append([change, lv, float(o["deal"]), o.get("off", 0.0) / 100, bought, math.log1p(o["times"]), usual,
                     1.0 if o["product"] == engine_product else 0.0, 1.0 if o.get("top") else 0.0,
                     math.log(max(o.get("pop", 0.0), 1e-6)), vs_last, float(o.get("sold_out", 0) > 0),
                     sens, npc["household"], npc["kids"], npc["income"], math.log(max(npc["budget"], 1.0)),
                     npc["rebuy"]]
                    + [1.0 if route == r else 0.0 for r in ROUTES]
                    + [1.0 if mission == m else 0.0 for m in MISSIONS]
                    + [float(o["deal"]) * sens, change * sens, usual * npc["rebuy"]])
    return np.asarray(rows, dtype=np.float32), opts
