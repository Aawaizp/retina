"""RETINA real shopper profiles from Instacart PRIOR orders.

Each real shopper's habits TOGETHER: average basket size, average days between orders, and how
often they re-buy something they bought before. Habits are linked in real life (people who shop
often re-buy much more), so the simulator copies one real shopper's whole profile per NPC
instead of drawing each habit separately.
Re-buy rate counts orders 2, 3, ... only (a first order cannot contain re-buys) - the same thing
an NPC does once it has a shopping history.

Writes data/processed/instacart_shopper_profiles.csv. PUBLIC_STATISTIC (Instacart, non-commercial).
Only "prior" orders are read; "train" orders stay held out for simulation/realism.py.

Run once:  python loaders\\shopper_profiles.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
INSTACART = DATA / "raw" / "instacart"
OUT = DATA / "processed" / "instacart_shopper_profiles.csv"


def main():
    t = time.time()
    orders = pd.read_csv(INSTACART / "orders.csv", usecols=["order_id", "user_id", "eval_set", "order_number",
                                                            "days_since_prior_order"])
    orders = orders[orders["eval_set"] == "prior"].drop(columns="eval_set")
    print("Reading Instacart prior order lines (one time, about a minute)...")
    per_order = []
    for chunk in pd.read_csv(INSTACART / "order_products__prior.csv", usecols=["order_id", "reordered"],
                             dtype="int32", chunksize=8_000_000):
        per_order.append(chunk.groupby("order_id")["reordered"].agg(["size", "sum"]))
    per_order = pd.concat(per_order).groupby(level=0).sum()      # orders split across chunks
    o = orders.merge(per_order, left_on="order_id", right_index=True)
    later = o[o["order_number"] > 1]                              # orders with a history
    users = pd.DataFrame({
        "basket_size": o.groupby("user_id")["size"].mean(),
        "days_between_orders": later.groupby("user_id")["days_since_prior_order"].mean(),
        "rebuy_rate": later.groupby("user_id")["sum"].sum() / later.groupby("user_id")["size"].sum(),
        "orders": o.groupby("user_id").size(),
    }).dropna()
    users = users.round({"basket_size": 2, "days_between_orders": 2, "rebuy_rate": 4})
    users["data_label"] = "PUBLIC_STATISTIC"
    users.to_csv(OUT, index=True)
    item_weighted = (users["rebuy_rate"] * users["basket_size"] * users["orders"]).sum() / \
        (users["basket_size"] * users["orders"]).sum()
    corr = users[["days_between_orders", "rebuy_rate"]].corr(method="spearman").iloc[0, 1]
    print(f"Saved {len(users):,} real shopper profiles to {OUT.name} in {time.time() - t:.0f}s")
    print(f"  re-buy rate: average shopper {users['rebuy_rate'].mean():.0%}, per item {item_weighted:.0%} | "
          f"link with days between orders: {corr:+.2f} (frequent shoppers re-buy more)")


if __name__ == "__main__":
    main()
