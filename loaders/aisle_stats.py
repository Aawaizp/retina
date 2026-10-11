"""RETINA real aisle statistics from Instacart PRIOR orders (32 million order lines).

Writes two small files the simulator uses:
  data/processed/instacart_aisle_stats.csv  - per aisle: order lines, re-bought lines (the shopper
                                              had bought that product before), orders containing
                                              it, and its share of its department (how dairy
                                              splits into milk, yogurt, eggs ...)
  data/processed/instacart_aisle_pairs.csv  - per pair of aisles: orders containing both
                                              (pasta + pasta sauce, frozen meals + frozen pizza)
Only "prior" orders are read: the "train" orders stay held out for simulation/realism.py.
PUBLIC_STATISTIC (Instacart Market Basket data, non-commercial use only).

Run once:  python loaders\\aisle_stats.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
INSTACART = DATA / "raw" / "instacart"
STATS_FILE = DATA / "processed" / "instacart_aisle_stats.csv"
PAIRS_FILE = DATA / "processed" / "instacart_aisle_pairs.csv"
CHUNK_ROWS = 4_000_000
MIN_PAIR_ORDERS = 50          # keep aisle pairs seen together in at least 50 real orders


def within_order_pairs(order_ids, aisle):
    """All (a, b) aisle pairs inside each order; rows sorted by order, one row per (order, aisle)."""
    start = np.r_[0, np.flatnonzero(np.diff(order_ids)) + 1]
    size = np.diff(np.r_[start, len(order_ids)])
    end = np.repeat(start + size, size)
    row = np.arange(len(order_ids))
    later = end - row - 1
    left = np.repeat(aisle, later)
    first = np.repeat(row + 1, later)
    offset = np.arange(later.sum()) - np.repeat(np.cumsum(later) - later, later)
    return left, aisle[first + offset]


def count_chunk(order_ids, aisle, reordered, n, lines, rebought, orders, pairs):
    lines += np.bincount(aisle, minlength=n)
    rebought += np.bincount(aisle, weights=reordered, minlength=n).astype(np.int64)
    key = np.unique(order_ids.astype(np.int64) * n + aisle)        # one row per (order, aisle)
    o, a = key // n, (key % n).astype(np.int64)
    orders += np.bincount(a, minlength=n)
    x, y = within_order_pairs(o, a)
    pairs += np.bincount(x * n + y, minlength=n * n)
    return len(np.unique(o))


def main():
    t = time.time()
    products = pd.read_csv(INSTACART / "products.csv", usecols=["product_id", "aisle_id"])
    aisles = pd.read_csv(INSTACART / "aisles.csv").merge(
        pd.read_csv(INSTACART / "products.csv", usecols=["aisle_id", "department_id"]).drop_duplicates("aisle_id"),
        on="aisle_id").merge(pd.read_csv(INSTACART / "departments.csv"), on="department_id")
    aisles = aisles.sort_values("aisle_id").reset_index(drop=True)
    n = len(aisles)
    position = np.full(products["aisle_id"].max() + 1, -1)
    position[aisles["aisle_id"].to_numpy()] = np.arange(n)
    aisle_of_product = np.zeros(products["product_id"].max() + 1, dtype=np.int64)
    aisle_of_product[products["product_id"].to_numpy()] = position[products["aisle_id"].to_numpy()]

    lines, orders, pairs = np.zeros(n, np.int64), np.zeros(n, np.int64), np.zeros(n * n, np.int64)
    rebought = np.zeros(n, np.int64)
    total_orders, carry, done = 0, None, 0
    print("Counting aisles in Instacart prior orders (one time, a minute or two)...")
    for chunk in pd.read_csv(INSTACART / "order_products__prior.csv", usecols=["order_id", "product_id", "reordered"],
                             dtype="int32", chunksize=CHUNK_ROWS):
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        last = chunk["order_id"].iat[-1]                          # last order may continue in next chunk
        carry, chunk = chunk[chunk["order_id"] == last], chunk[chunk["order_id"] != last]
        total_orders += count_chunk(chunk["order_id"].to_numpy(), aisle_of_product[chunk["product_id"].to_numpy()],
                                    chunk["reordered"].to_numpy(), n, lines, rebought, orders, pairs)
        done += len(chunk)
        print(f"  {done:,} order lines", end="\r")
    if carry is not None and len(carry):
        total_orders += count_chunk(carry["order_id"].to_numpy(), aisle_of_product[carry["product_id"].to_numpy()],
                                    carry["reordered"].to_numpy(), n, lines, rebought, orders, pairs)

    stats = aisles[["aisle", "department"]].assign(lines=lines, rebought=rebought, orders=orders)
    stats["rebuy_rate"] = (stats["rebought"] / stats["lines"].clip(lower=1)).round(4)
    stats["share_of_department"] = (stats["lines"] / stats.groupby("department")["lines"].transform("sum")).round(5)
    stats["orders_total"] = total_orders
    stats["data_label"] = "PUBLIC_STATISTIC"
    stats.to_csv(STATS_FILE, index=False)

    m = pairs.reshape(n, n)
    a, b = np.nonzero(m >= MIN_PAIR_ORDERS)
    names = aisles["aisle"].to_numpy()
    pd.DataFrame({"aisle_a": names[a], "aisle_b": names[b], "together_orders": m[a, b]}).to_csv(PAIRS_FILE, index=False)
    print(f"\nSaved {STATS_FILE.name} ({n} aisles, {total_orders:,} orders) and {PAIRS_FILE.name} "
          f"({len(a):,} aisle pairs) in {time.time() - t:.0f}s")
    top = stats.nlargest(5, "lines")
    print("Biggest aisles: " + ", ".join(f"{r.aisle} ({r.share_of_department:.0%} of {r.department})"
                                         for r in top.itertuples()))
    big = stats[stats["lines"] > 100_000]
    print("Most re-bought: " + ", ".join(f"{r.aisle} {r.rebuy_rate:.0%}" for r in big.nlargest(4, "rebuy_rate").itertuples())
          + " | least: " + ", ".join(f"{r.aisle} {r.rebuy_rate:.0%}" for r in big.nsmallest(3, "rebuy_rate").itertuples()))


if __name__ == "__main__":
    main()
