"""RETINA trip needs from Instacart PRIOR orders (32 million order lines).

Two things real shoppers show, that make NPCs shop from NEEDS:
  1. Trips: the longer since the last trip, the bigger the basket (the cupboard is emptier).
     Measured inside each shopper (their own big vs small trips), so it is not just "big families
     shop more". Also how much one shopper's basket size swings from trip to trip, and the share
     of small top-up / normal / big stock-up trips.
  2. How long things last at home: per aisle, how many trips (and days) pass before a shopper buys
     the SAME product again (milk soon, spices much later).

Writes data/processed/instacart_trip_stats.json and data/processed/instacart_aisle_cycles.csv.
Only "prior" orders are read; "train" orders stay held out for simulation/realism.py.
PUBLIC_STATISTIC (Instacart Market Basket data, non-commercial use only).

Run once:  python loaders\\trip_needs.py      (about 1-2 minutes, needs ~2 GB of memory)
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
INSTACART = DATA / "raw" / "instacart"
STATS_FILE = DATA / "processed" / "instacart_trip_stats.json"
CYCLES_FILE = DATA / "processed" / "instacart_aisle_cycles.csv"
TOP_UP, STOCK_UP = 0.5, 1.5        # trip size vs the shopper's own average: <= 0.5 small, >= 1.5 big


def read_orders():
    o = pd.read_csv(INSTACART / "orders.csv", usecols=["order_id", "user_id", "eval_set", "order_number",
                                                       "days_since_prior_order"])
    o = o[o["eval_set"] == "prior"].drop(columns="eval_set").sort_values(["user_id", "order_number"])
    o["day"] = o["days_since_prior_order"].fillna(0).groupby(o["user_id"]).cumsum()   # day of each order
    return o


def read_lines():
    parts = []
    for chunk in pd.read_csv(INSTACART / "order_products__prior.csv", usecols=["order_id", "product_id"],
                             dtype="int32", chunksize=8_000_000):
        parts.append(chunk.to_numpy())
        print(f"  {sum(len(p) for p in parts):,} order lines", end="\r")
    lines = np.concatenate(parts)
    return lines[:, 0], lines[:, 1]


def trip_stats(o):
    """Basket size vs days since the last trip, inside each shopper."""
    o = o[o["size"] > 0].copy()
    user_mean = o.groupby("user_id")["size"].transform("mean")
    o["ratio"] = o["size"] / user_mean
    o["n"] = o.groupby("user_id")["size"].transform("size")
    # elasticity: log size vs log days, both measured against the shopper's own average
    g = o[(o["order_number"] > 1) & (o["days_since_prior_order"].between(1, 29))].copy()
    g = g[g.groupby("user_id")["size"].transform("size") >= 3]
    x = np.log(g["days_since_prior_order"]) - np.log(g["days_since_prior_order"]).groupby(g["user_id"]).transform("mean")
    y = np.log(g["size"]) - np.log(g["size"]).groupby(g["user_id"]).transform("mean")
    beta = float((x * y).sum() / (x * x).sum())
    later = o[o["order_number"] > 1]
    bins = pd.cut(later["days_since_prior_order"], [0, 3, 7, 14, 29, 30], labels=["1-3", "4-7", "8-14", "15-29", "30+"],
                  include_lowest=True)
    by_gap = later.groupby(bins, observed=True)["ratio"].mean().round(3).to_dict()
    # how much one shopper's trips swing: extra variance beyond a Poisson count (negative binomial)
    many = o[o["n"] >= 5]
    stats = many.groupby("user_id")["size"].agg(["mean", "var"])
    excess = float(((stats["var"] - stats["mean"]) / stats["mean"] ** 2).clip(lower=0).median())
    shares = {"top_up": float((many["ratio"] <= TOP_UP).mean()),
              "routine": float(((many["ratio"] > TOP_UP) & (many["ratio"] < STOCK_UP)).mean()),
              "stock_up": float((many["ratio"] >= STOCK_UP).mean())}
    return {"basket_vs_days_elasticity": round(beta, 4),
            "basket_vs_days_by_gap": {k: float(v) for k, v in by_gap.items()},
            "basket_extra_variance": round(excess, 4),
            "trip_type_shares": {k: round(v, 4) for k, v in shares.items()},
            "trip_type_rule": f"basket size vs the shopper's own average: <= {TOP_UP} top_up, >= {STOCK_UP} stock_up",
            "orders_used": int(len(g)), "data_label": "PUBLIC_STATISTIC",
            "source": "Instacart prior orders (online grocery, 2017), non-commercial use"}


def aisle_cycles(order_id, product_id, o):
    """Trips and days until the same shopper buys the same product again, per aisle."""
    products = pd.read_csv(INSTACART / "products.csv", usecols=["product_id", "aisle_id"])
    aisles = pd.read_csv(INSTACART / "aisles.csv")
    aisle_of = np.zeros(products["product_id"].max() + 1, dtype=np.int16)
    aisle_of[products["product_id"].to_numpy()] = products["aisle_id"].to_numpy()
    n = o["order_id"].max() + 1
    user_of, num_of, day_of = (np.zeros(n, dtype=np.int32) for _ in range(3))
    user_of[o["order_id"]], num_of[o["order_id"]] = o["user_id"], o["order_number"]
    day_of[o["order_id"]] = o["day"]
    key = (user_of[order_id].astype(np.int64) * 50_000 + product_id) * 128 + num_of[order_id]
    order = np.argsort(key, kind="stable")
    del key
    oid = order_id[order]
    pid = product_id[order]
    del order
    same = (pid[1:] == pid[:-1]) & (user_of[oid[1:]] == user_of[oid[:-1]])
    trips = (num_of[oid[1:]] - num_of[oid[:-1]])[same]
    days = (day_of[oid[1:]] - day_of[oid[:-1]])[same]
    aisle = aisle_of[pid[1:]][same]
    df = pd.DataFrame({"aisle_id": aisle, "trips": trips, "days": days, "next": trips == 1})
    out = df.groupby("aisle_id").agg(rebuys=("trips", "size"), trips_mean=("trips", "mean"),
                                     trips_median=("trips", "median"), days_mean=("days", "mean"),
                                     days_median=("days", "median"), next_trip_share=("next", "mean"))
    # Without waiting for a re-buy: of everything bought on a trip, the share bought again on the
    # NEXT trip, and within the next 3 trips (only trips that have that many trips after them)
    last_num = np.zeros(user_of.max() + 1, dtype=np.int32)
    np.maximum.at(last_num, o["user_id"].to_numpy(), o["order_number"].to_numpy())
    left = (last_num[user_of[oid]] - num_of[oid])               # trips this shopper made afterwards
    a_all = aisle_of[pid]
    gap_next = np.full(len(oid), 10 ** 6, dtype=np.int32)
    gap_next[:-1][same] = trips
    n_aisles = aisle_of.max() + 1
    has1, has3 = left >= 1, left >= 3
    next1 = np.bincount(a_all[has1], weights=(gap_next[has1] == 1), minlength=n_aisles) / \
        np.maximum(np.bincount(a_all[has1], minlength=n_aisles), 1)
    within3 = np.bincount(a_all[has3], weights=(gap_next[has3] <= 3), minlength=n_aisles) / \
        np.maximum(np.bincount(a_all[has3], minlength=n_aisles), 1)
    out["bought_again_next_trip"] = next1[out.index.to_numpy()]
    out["bought_again_within_3_trips"] = within3[out.index.to_numpy()]
    out = aisles.merge(out.reset_index(), on="aisle_id")
    out = out.round({"trips_mean": 3, "days_mean": 2, "next_trip_share": 4, "bought_again_next_trip": 4,
                     "bought_again_within_3_trips": 4})
    out["data_label"] = "PUBLIC_STATISTIC"
    return out.drop(columns="aisle_id")


def main():
    t = time.time()
    print("Reading Instacart prior orders (one time, a minute or two)...")
    o = read_orders()
    order_id, product_id = read_lines()
    o["size"] = np.bincount(order_id, minlength=o["order_id"].max() + 1)[o["order_id"]]
    stats = trip_stats(o)
    STATS_FILE.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"\nTrips: basket size grows with days since the last trip (elasticity {stats['basket_vs_days_elasticity']:+.2f})")
    print("  basket vs the shopper's own average, by days since last trip: " +
          ", ".join(f"{k} days {v:.2f}x" for k, v in stats["basket_vs_days_by_gap"].items()))
    s = stats["trip_type_shares"]
    print(f"  trip types: top-up {s['top_up']:.0%} | routine {s['routine']:.0%} | stock-up {s['stock_up']:.0%} "
          f"| extra variance {stats['basket_extra_variance']:.3f}")
    cycles = aisle_cycles(order_id, product_id, o)
    cycles.to_csv(CYCLES_FILE, index=False)
    big = cycles[cycles["rebuys"] > 50_000].sort_values("bought_again_next_trip", ascending=False)
    print(f"Aisle cycles: {len(cycles)} aisles | share bought again on the very next trip:")
    print("  most:  " + ", ".join(f"{r.aisle} {r.bought_again_next_trip:.0%}" for r in big.head(4).itertuples()))
    print("  least: " + ", ".join(f"{r.aisle} {r.bought_again_next_trip:.0%}" for r in big.tail(3).itertuples()))
    print(f"Saved {STATS_FILE.name} and {CYCLES_FILE.name} in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
