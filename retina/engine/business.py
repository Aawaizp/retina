"""Business-ready sales data from a live run: the tables a retailer's analysts actually work with.

A live run (retina --mind student|fast ...) writes simulation logs (receipts.csv, trips.csv, ...). After
each run, build() turns them into run_dir/business/:

  sales_transactions.csv   one row per product line sold (POS data): price, discount, cost, profit
  baskets.csv              one row per receipt (transaction header): customer, times, items, total
  daily_product_sales.csv  every carried product x completed day, including zero-sale days: units,
                           revenue, price, promotion offer, closing stock and lost demand
  inventory_daily.csv      product x day with a stock MOVEMENT: opening stock, delivered, wasted, sold,
                           closing stock, on order, out of stock, lost sales (no row = stock unchanged)
                           Stock counts are at simulated NPC scale, not weighted store inventory.
  daily_store_kpis.csv     one row per day: customers, sales, discounts, cost, profit, margin, basket,
                           promotions, stock-outs, lost sales, waste
  products.csv             product master: id, name, brand, category, sub-category, size, price, cost
  customers.csv            customers who shopped in the period: household, income band, price
                           sensitivity, visits, spend (the simulated behaviour - no real person)
  promotions.csv           promotion calendar: product, week, discount
  DATA_DICTIONARY.md, DATASET_CARD.md

Every row is SIMULATED (unit costs ESTIMATED). Nothing here is invented after the run: every number
comes from what the simulated shoppers and the simulated store did.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import csvsafe
from .config import PROFILE

IDS = {"product_id": str, "trip_id": str, "substitute_id": str}
INVENTORY_FILE = "inventory_moves.csv"
STOCK_FILE = "stock_daily.csv"


# ------------------------------------------------------------- during the run
def inventory_moves(sim, day, stock_before, delivered, waste_from, sales, misses):
    """One day's stock movements per product (only products where something happened)."""
    n = len(stock_before)
    sold = np.bincount(sales["product_idx"].to_numpy(), weights=sales["quantity"].to_numpy(), minlength=n)
    wasted = np.zeros(n)
    for d, product, units in sim.waste[waste_from:]:
        wasted[product] += units
    lost = np.zeros(n)
    if len(misses):
        no_sub = misses["substitute_idx"].to_numpy() < 0
        np.add.at(lost, misses["product_idx"].to_numpy()[no_sub], 1)
    closing = sim.inv.stock.astype(float)
    moved = sim.inv.sellable & ((sold > 0) | (delivered > 0) | (wasted > 0) | (lost > 0)
                                | ((closing <= 0) & (stock_before > 0)))
    idx = np.flatnonzero(moved)
    p = sim.store.products
    return pd.DataFrame({
        "date": sim.book.dates[day].date(), "product_id": p["product_id"].to_numpy()[idx],
        "opening_stock": stock_before[idx].astype(int), "delivered_units": delivered[idx].astype(int),
        "wasted_units": wasted[idx].astype(int), "sold_units": sold[idx].astype(int),
        "closing_stock": np.maximum(closing[idx], 0).astype(int), "on_order_units": sim.inv.on_order[idx].astype(int),
        "out_of_stock": closing[idx] <= 0, "lost_sales_units": lost[idx].astype(int),
        "reorder_point": np.round(sim.inv.reorder_point[idx], 1), "store_forecast_daily": np.round(sim.inv.forecast[idx], 2),
    })


def stock_snapshot(sim, day):
    """The whole store's stock at closing (one row): units, value at cost, empty shelves."""
    inv, p = sim.inv, sim.store.products
    sell = inv.sellable
    units = np.maximum(inv.stock[sell], 0)
    return pd.DataFrame([{"date": sim.book.dates[day].date(), "stock_units": int(units.sum()),
                          "stock_value_at_cost": round(float((units * p["unit_cost"].to_numpy()[sell]).sum()), 2),
                          "products_out_of_stock": int((inv.stock[sell] <= 0).sum()),
                          "products_carried": int(sell.sum())}])


def due_today(inv, day):
    """Units arriving this morning, per product (read BEFORE the day runs)."""
    delivered = np.zeros(len(inv.stock))
    for products, units in inv.deliveries.get(day, []):
        np.add.at(delivered, products, units)
    return delivered


REASONS = {"need": "ran out at home", "repeat": "regular buy", "new": "new item on the list",
           "partner": "goes with another item", "impulse": "impulse (saw a deal on the shelf)"}


def decided_by(r):
    """Who chose this product: the student model (thinking NPC), an LLM, an autopilot routine buy, or a
    store rule (impulse buy of a deal, nothing choosable on the shelf, or a fast engine run)."""
    if "mind" not in r:                                     # a fast run: no thinking NPCs at all
        return pd.Series("statistical engine", index=r.index)
    mind = r["mind"]
    out = mind.map({"student": "student model", "llm": "LLM", "autopilot": "autopilot (routine buy)",
                    "llm_fallback": "store rule (LLM unavailable)"}).fillna("store rule (nothing to choose)")
    out[(mind == "engine") & (r["reason"] == "impulse")] = "store rule (impulse buy)"
    return out


def npc_cards(run_dir):
    """{customer_id: card} from the run's memory (thinking runs): each NPC's role and habits."""
    path = Path(run_dir) / "memory.sqlite"
    if not path.exists():
        return {}
    import sqlite3
    db = sqlite3.connect(path)
    try:
        return {f"C{int(npc):05d}": json.loads(card) for npc, card in db.execute("SELECT npc, card FROM cards")}
    except sqlite3.Error:
        return {}
    finally:
        db.close()


def role(card):
    return card["persona"]["shopping_style"].split(" - ")[0] if card else None


def population_scope(sample_weight):
    """Scope of unprefixed counts, sales and stock in a business export."""
    return "SIMULATED_NPC_SAMPLE" if sample_weight > 1 else "SIMULATED_FULL_RUN"


def write_dense_daily_sales(path, sparse, moves, sim, store_id, sample_weight, capped_days):
    """Write one carried-SKU/day chunk at a time so a one-year run stays memory bounded.

    The sparse receipt/stockout aggregation is joined to every carried SKU. Closing stock on
    unchanged days is reconstructed from the final inventory and the logged daily movements.
    """
    from .engine import shelf_prices

    p = sim.store.products
    carried = np.flatnonzero(sim.inv.sellable)
    ids = p["product_id"].iloc[carried].astype(str).to_numpy()
    order = np.argsort(ids, kind="stable")
    carried, ids = carried[order], ids[order]
    base = pd.DataFrame({"store_id": store_id, "product_id": ids,
                         "product_name": p["product_name"].iloc[carried].to_numpy(),
                         "category": p["department"].iloc[carried].to_numpy()})
    positions = pd.Series(np.arange(len(carried)), index=ids)
    dates = pd.DatetimeIndex(sim.book.dates[:sim.day]).strftime("%Y-%m-%d").tolist()
    movement_by_day = {str(day): group for day, group in moves.groupby("date")} if len(moves) else {}
    sales_columns = sparse.drop(columns=["date", "store_id", "product_name", "category"], errors="ignore")
    empty_sales = sales_columns.iloc[0:0].set_index("product_id")
    sales_by_day = {str(day): group.drop(columns=["date", "store_id", "product_name", "category"],
                                         errors="ignore").set_index("product_id")
                    for day, group in sparse.groupby("date")}

    # Walk the movement log backwards once to recover opening stock on the first exported day.
    current_stock = np.maximum(sim.inv.stock[carried], 0).astype(int).copy()
    for day in reversed(dates):
        movement = movement_by_day.get(day)
        if movement is not None:
            loc = positions.loc[movement["product_id"].astype(str)].to_numpy(dtype=int)
            current_stock[loc] = movement["opening_stock"].to_numpy(dtype=int)

    scope = population_scope(sample_weight)
    row_count, schema = 0, None
    for day_index, day in enumerate(dates):
        movement = movement_by_day.get(day)
        if movement is not None:
            loc = positions.loc[movement["product_id"].astype(str)].to_numpy(dtype=int)
            current_stock[loc] = movement["closing_stock"].to_numpy(dtype=int)
        dense = base.join(sales_by_day.get(day, empty_sales), on="product_id")
        dense.insert(0, "date", day)
        _, regular, promo_by_group, shelf = shelf_prices(day_index, sim.store, sim.book,
                                                        sim.iv, len(sim.group_sensitivity))
        for column in ("units_sold", "transactions", "lost_sales_units"):
            dense[column] = dense[column].fillna(0).astype(int)
        for column in ("gross_sales", "discount_amount", "net_sales", "gross_profit", "lost_sales_value"):
            dense[column] = dense[column].fillna(0).round(2)
        dense["avg_unit_price"] = dense["avg_unit_price"].round(2)  # blank when nothing sold
        dense["regular_price"] = np.round(regular[carried], 2)
        dense["shelf_price"] = np.round(shelf[carried], 2)  # public price; targeted offers may differ
        dense["on_promotion"] = dense["on_promotion"].fillna(False).astype(bool)
        dense["promotion_offer_active"] = np.any(np.asarray(promo_by_group)[:, carried], axis=0)
        dense["closing_stock"] = current_stock.copy()
        dense["out_of_stock"] = current_stock <= 0
        capped = day in capped_days
        dense["estimated_store_units_sold"] = (dense["units_sold"] * sample_weight).round(2) if not capped else np.nan
        dense["estimated_store_net_sales"] = (dense["net_sales"] * sample_weight).round(2) if not capped else np.nan
        dense["sample_weight"] = sample_weight
        dense["estimate_status"] = "ARRIVAL_CAP_REACHED" if capped else "EXPANDED_NPC_SAMPLE"
        dense["population_scope"] = scope
        dense["stock_scope"] = scope
        dense["data_label"] = "SIMULATED"
        csvsafe.to_csv(dense, path, index=False, mode="w" if row_count == 0 else "a",
                       header=row_count == 0)
        row_count += len(dense)
        if schema is None:
            schema = dense.iloc[0:0].copy()
    return row_count, schema


# ------------------------------------------------------------- after the run
def build(run_dir, sim, say=print):
    run_dir = Path(run_dir)
    receipts_file = run_dir / "receipts.csv"
    if not receipts_file.exists():
        return None
    out = run_dir / "business"
    out.mkdir(exist_ok=True)
    store_id, store_name = PROFILE["id"], PROFILE["name"]
    p = sim.store.products

    r = pd.read_csv(receipts_file, dtype=IDS)
    trips = pd.read_csv(run_dir / "trips.csv", dtype=IDS)
    stock = pd.read_csv(run_dir / "stockouts.csv", dtype=IDS) if (run_dir / "stockouts.csv").exists() else pd.DataFrame()
    moves = pd.read_csv(run_dir / INVENTORY_FILE, dtype=IDS) if (run_dir / INVENTORY_FILE).exists() else pd.DataFrame()
    limit = sim.traffic_settings.get("npc_limit")
    if limit:
        from . import traffic

        # Recreate the deterministic draw before the cap and before a configured start time
        # filters out early visitors. Paid trips alone cannot reveal whether arrivals were cut.
        capped_days = {str(sim.book.dates[i].date()) for i in range(sim.day)
                       if traffic.uncapped_npcs_today(i) >= int(limit)}
    else:
        capped_days = set()

    info = p.set_index("product_id")
    size = info["pack_size"] if "pack_size" in info else pd.Series(dtype=object)
    leave = trips.set_index("trip_id")["leave_time"]
    arrive = trips.set_index("trip_id")["arrive_time"]

    # 1. sales_transactions: POS lines
    lines = pd.DataFrame({
        "transaction_id": r["trip_id"], "line_number": r.groupby("trip_id").cumcount() + 1,
        "date": r["date"], "checkout_time": r["trip_id"].map(leave), "store_id": store_id,
        "customer_id": r["shopper_id"].map(lambda s: f"C{int(s):05d}"),
        "product_id": r["product_id"], "product_name": r["product_name"],
        "brand": r["product_id"].map(info["brand"]).fillna(""), "category": r["department"], "subcategory": r["aisle"],
        "quantity": r["quantity"], "unit_price": r["unit_price"], "regular_price": r["regular_price"],
    })
    lines["gross_sales"] = (lines["regular_price"] * lines["quantity"]).round(2)
    lines["discount_amount"] = ((lines["regular_price"] - lines["unit_price"]).clip(lower=0) * lines["quantity"]).round(2)
    lines["net_sales"] = r["line_total"].round(2)
    lines["unit_cost"] = r["unit_cost"]
    lines["cost_of_goods"] = (r["unit_cost"] * r["quantity"]).round(2)
    lines["gross_profit"] = (lines["net_sales"] - lines["cost_of_goods"]).round(2)
    lines["on_promotion"] = r["on_promo"].astype(bool)
    lines["promotion_type"] = r["discount_type"].fillna("")
    lines["repeat_purchase"] = r["is_repeat"].astype(bool)
    lines["purchase_reason"] = r["reason"].map(REASONS).fillna(r["reason"])
    lines["decided_by"] = decided_by(r)
    sample_weight = round(1 / float(sim.traffic_settings.get("npc_scale", 1.0)), 4)
    lines["sample_weight"] = sample_weight
    lines["population_scope"] = population_scope(sample_weight)
    lines["data_label"] = "SIMULATED"

    # 2. baskets: one row per receipt
    t = trips.copy()
    agg = lines.groupby("transaction_id").agg(lines_=("product_id", "size"), units=("quantity", "sum"),
                                               gross_sales=("gross_sales", "sum"), discount=("discount_amount", "sum"),
                                               net_sales=("net_sales", "sum"), gross_profit=("gross_profit", "sum"),
                                               promo_units=("on_promotion", "sum"))
    baskets = pd.DataFrame({
        "transaction_id": t["trip_id"], "date": t["date"], "store_id": store_id,
        "customer_id": t["shopper_id"].map(lambda s: f"C{int(s):05d}"),
        "arrival_time": t["arrive_time"], "checkout_time": t["leave_time"],
        "minutes_in_store": t["minutes_in_store"].round(1), "checkout_wait_min": t["wait_min"].round(1),
        "walk_metres": t["walk_m"].round(0), "shelves_visited": t["shelves_visited"], "aisles_passed": t["aisles_seen"],
        "trip_type": t["mission"]}).set_index("transaction_id").join(agg, how="inner").reset_index()
    baskets = baskets.rename(columns={"lines_": "lines"})
    for c in ("gross_sales", "discount", "net_sales", "gross_profit"):
        baskets[c] = baskets[c].round(2)
    baskets = baskets.rename(columns={"discount": "discount_amount", "promo_units": "lines_on_promotion"})
    if "list_planned_by" in t:                                   # student runs: who wrote the shopping list
        baskets["list_planned_by"] = baskets["transaction_id"].map(
            t.set_index("trip_id")["list_planned_by"]).fillna("store rule")
    baskets["sample_weight"] = sample_weight
    baskets["population_scope"] = population_scope(sample_weight)
    baskets["data_label"] = "SIMULATED"

    # 3. daily_product_sales: product x day with sales or lost demand
    day_sales = lines.groupby(["date", "product_id"]).agg(
        units_sold=("quantity", "sum"), transactions=("transaction_id", "nunique"),
        gross_sales=("gross_sales", "sum"), discount_amount=("discount_amount", "sum"), net_sales=("net_sales", "sum"),
        gross_profit=("gross_profit", "sum"), avg_unit_price=("unit_price", "mean"),
        regular_price=("regular_price", "max"), on_promotion=("on_promotion", "max"))
    if len(stock):
        lost = (stock[stock["lost_sale"].astype(bool)].assign(date=lambda x: pd.to_datetime(x["date"]).dt.strftime("%Y-%m-%d"))
                .groupby(["date", "product_id"]).agg(lost_sales_units=("lost_sale", "size"),
                                                      lost_sales_value=("unit_price", "sum")))
        day_sales = day_sales.join(lost, how="outer")
    else:
        day_sales["lost_sales_units"], day_sales["lost_sales_value"] = 0, 0.0
    day_sales = day_sales.reset_index()
    for c in ("units_sold", "transactions", "lost_sales_units"):
        day_sales[c] = day_sales[c].fillna(0).astype(int)
    for c in ("gross_sales", "discount_amount", "net_sales", "gross_profit", "lost_sales_value"):
        day_sales[c] = day_sales[c].fillna(0.0).round(2)
    day_sales["avg_unit_price"] = day_sales["avg_unit_price"].round(2)
    day_sales["on_promotion"] = day_sales["on_promotion"].fillna(False).astype(bool)
    day_sales["estimated_store_units_sold"] = (day_sales["units_sold"] * sample_weight).round(2)
    day_sales["estimated_store_net_sales"] = (day_sales["net_sales"] * sample_weight).round(2)
    if len(moves):
        close = moves.set_index(["date", "product_id"])[["closing_stock", "out_of_stock"]]
        day_sales = day_sales.join(close, on=["date", "product_id"])
        day_sales["out_of_stock"] = day_sales["out_of_stock"].fillna(False).astype(bool)
    day_sales.insert(1, "store_id", store_id)
    day_sales.insert(3, "product_name", day_sales["product_id"].map(info["product_name"]))
    day_sales.insert(4, "category", day_sales["product_id"].map(info["department"]))
    day_sales["data_label"] = "SIMULATED"
    day_sales = day_sales.sort_values(["date", "product_id"])

    # 4. inventory_daily: stock movements
    inventory = moves.copy()
    if len(inventory):
        inventory.insert(1, "store_id", store_id)
        inventory.insert(3, "product_name", inventory["product_id"].map(info["product_name"]))
        cost = inventory["product_id"].map(info["unit_cost"])
        inventory["wasted_cost"] = (inventory["wasted_units"] * cost).round(2)
        inventory["stock_value_at_cost"] = (inventory["closing_stock"] * cost).round(2)
        inventory["sample_weight"] = sample_weight
        inventory["stock_scope"] = population_scope(sample_weight)
        inventory["data_label"] = "SIMULATED"

    # 5. daily_store_kpis
    kpi = lines.groupby("date").agg(transactions=("transaction_id", "nunique"), customers=("customer_id", "nunique"),
                                    units_sold=("quantity", "sum"), gross_sales=("gross_sales", "sum"),
                                    discount_amount=("discount_amount", "sum"), net_sales=("net_sales", "sum"),
                                    cost_of_goods=("cost_of_goods", "sum"), gross_profit=("gross_profit", "sum"))
    completed_dates = pd.DatetimeIndex(sim.book.dates[:sim.day]).strftime("%Y-%m-%d")
    kpi = kpi.reindex(completed_dates)
    for column in ("transactions", "customers", "units_sold"):
        kpi[column] = kpi[column].fillna(0).astype(int)
    for column in ("gross_sales", "discount_amount", "net_sales", "cost_of_goods", "gross_profit"):
        kpi[column] = kpi[column].fillna(0).round(2)
    kpi.index.name = "date"
    promo_sales = lines[lines["on_promotion"]].groupby("date")["net_sales"].sum()
    kpi["gross_margin_pct"] = (100 * kpi["gross_profit"] / kpi["net_sales"]).round(2)
    kpi["avg_basket_value"] = (kpi["net_sales"] / kpi["transactions"]).round(2)
    kpi["units_per_basket"] = (kpi["units_sold"] / kpi["transactions"]).round(2)
    kpi["promotion_sales_share_pct"] = (100 * promo_sales.reindex(kpi.index).fillna(0)
                                         / kpi["net_sales"].replace(0, np.nan)).fillna(0).round(2)
    kpi["stockout_events"] = 0
    kpi["lost_sales_value"] = 0.0
    kpi["waste_cost"] = 0.0
    if len(stock):
        s = stock.assign(date=lambda x: pd.to_datetime(x["date"]).dt.strftime("%Y-%m-%d"))
        kpi["stockout_events"] = s.groupby("date").size().reindex(kpi.index).fillna(0).astype(int)
        kpi["lost_sales_value"] = (s[s["lost_sale"].astype(bool)].groupby("date")["unit_price"].sum()
                                   .reindex(kpi.index).fillna(0).round(2))
    if len(inventory):
        kpi["waste_cost"] = inventory.groupby("date")["wasted_cost"].sum().reindex(kpi.index).fillna(0).round(2)
    if (run_dir / STOCK_FILE).exists():                    # the WHOLE store's stock at closing
        snap = pd.read_csv(run_dir / STOCK_FILE).drop_duplicates("date", keep="last").set_index("date")
        for c in ("stock_units", "stock_value_at_cost", "products_out_of_stock", "products_carried"):
            kpi[c] = snap[c].reindex(kpi.index)
    kpi["estimated_store_transactions"] = (kpi["transactions"] * sample_weight).round(2)
    kpi["estimated_store_net_sales"] = (kpi["net_sales"] * sample_weight).round(2)
    if capped_days:
        kpi.loc[kpi.index.isin(capped_days), ["estimated_store_transactions", "estimated_store_net_sales"]] = np.nan
    kpi["estimate_status"] = np.where(kpi.index.isin(capped_days), "ARRIVAL_CAP_REACHED", "EXPANDED_NPC_SAMPLE")
    kpi["sample_weight"] = sample_weight
    kpi["population_scope"] = population_scope(sample_weight)
    kpi["stock_scope"] = population_scope(sample_weight)
    kpi = kpi.round(2).reset_index()
    kpi.insert(1, "weekday", pd.to_datetime(kpi["date"]).dt.day_name())
    kpi.insert(2, "store_id", store_id)
    kpi["data_label"] = "SIMULATED"

    # 6. products: the store's product master (products it sells)
    sold_here = sim.inv.sellable
    prod = pd.DataFrame({
        "product_id": p["product_id"][sold_here], "product_name": p["product_name"][sold_here],
        "brand": p["brand"][sold_here].fillna(""), "category": p["department"][sold_here],
        "subcategory": p["aisle"][sold_here],
        "pack_size": (p["pack_size"][sold_here] if "pack_size" in p else ""),
        "regular_price": p["regular_price"][sold_here].round(2), "unit_cost": p["unit_cost"][sold_here].round(2),
        "promotion_eligible": p["promo_eligible"][sold_here].astype(bool) if "promo_eligible" in p else False,
        "shelf_life_days": sim.inv.shelf_life[sold_here]})
    prod["cost_label"], prod["data_label"] = "ESTIMATED", "SIMULATED"

    # 7. customers: who shopped in the period, with what they did
    sh = sim.shoppers
    ids = np.unique(r["shopper_id"].to_numpy())
    spend = lines.groupby("customer_id").agg(first_visit=("date", "min"), last_visit=("date", "max"),
                                             visits=("transaction_id", "nunique"), units=("quantity", "sum"),
                                             total_spend=("net_sales", "sum"))
    fav = lines.groupby(["customer_id", "category"])["net_sales"].sum().reset_index() \
        .sort_values("net_sales").drop_duplicates("customer_id", keep="last").set_index("customer_id")["category"]
    sens = sh.price_sensitivity[ids]
    cut = np.quantile(sh.price_sensitivity, [1 / 3, 2 / 3])
    cust = pd.DataFrame({"customer_id": [f"C{int(s):05d}" for s in ids],
                         "household_size": sh.household_size[ids] if sh.household_size is not None else np.nan,
                         "children": sh.kids[ids] if sh.kids is not None else np.nan,
                         "income_band": (sh.income_quintile[ids] + 1) if sh.income_quintile is not None else np.nan,
                         "weekly_food_budget": np.round(sh.weekly_budget[ids], 2) if sh.weekly_budget is not None else np.nan,
                         "price_sensitivity": np.where(sens < cut[0], "low", np.where(sens < cut[1], "medium", "high"))})
    cust = cust.set_index("customer_id").join(spend).join(fav.rename("favourite_category")).reset_index()
    cust["total_spend"] = cust["total_spend"].round(2)
    cust["avg_basket_value"] = (cust["total_spend"] / cust["visits"]).round(2)
    cards = npc_cards(run_dir)
    if cards:
        cust["age"] = cust["customer_id"].map(lambda c: cards.get(c, {}).get("facts", {}).get("age"))
        cust["shopper_role"] = cust["customer_id"].map(lambda c: role(cards.get(c)))
    cust["data_label"] = "SIMULATED"
    profiles = pd.DataFrame([{
        "customer_id": cid, "name": c["facts"]["name"], "age": c["facts"]["age"], "shopper_role": role(c),
        "shopping_style": c["persona"]["shopping_style"], "bio": c["persona"]["bio"],
        "cares_about": "; ".join(c["persona"].get("cares_about", [])), "quote": c["persona"].get("quote", ""),
        "habit_products": "; ".join(c["facts"].get("habit_products", [])),
        "favourite_day": c["facts"].get("favorite_day"), "days_between_trips": c["facts"].get("days_between_trips"),
        "data_label": "SIMULATED"} for cid, c in sorted(cards.items())])

    # 8. promotions: the calendar for the weeks of this run
    book = sim.book
    # The offer calendar covers every completed store day, including a week with no receipts.
    weeks = range(int(book.week_of_day[0]), int(book.week_of_day[sim.day - 1]) + 1) if sim.day else []
    rows = []
    for w in weeks:
        on = np.flatnonzero(book.on_promo[w] & sold_here)
        start = book.dates[min(w * 7, len(book.dates) - 1)]
        for i in on:
            ratio = float(p["promo_ratio"].iat[i])
            rows.append({"promotion_id": f"PR{w:03d}-{p['product_id'].iat[i]}", "product_id": p["product_id"].iat[i],
                         "product_name": p["product_name"].iat[i], "category": p["department"].iat[i],
                         "start_date": start.date(), "end_date": (start + pd.Timedelta(days=6)).date(),
                         "discount_pct": round(100 * (1 - ratio), 1),
                         "regular_price": round(float(p["regular_price"].iat[i]), 2),
                         "promotion_price": round(float(p["regular_price"].iat[i]) * ratio, 2)})
    promos = pd.DataFrame(rows)
    if len(promos):
        promos["data_label"] = "SIMULATED"

    daily_count, daily_schema = write_dense_daily_sales(out / "daily_product_sales.csv", day_sales,
                                                         moves, sim, store_id, sample_weight, capped_days)
    tables = {"sales_transactions": lines, "baskets": baskets, "daily_product_sales": daily_schema,
              "inventory_daily": inventory, "daily_store_kpis": kpi, "products": prod, "customers": cust,
              "promotions": promos}
    if len(profiles):
        tables["npc_profiles"] = profiles
    for name, df in tables.items():
        if name != "daily_product_sales":
            csvsafe.to_csv(df, out / f"{name}.csv", index=False)
    row_counts = {name: len(df) for name, df in tables.items()}
    row_counts["daily_product_sales"] = daily_count
    (out / "DATA_DICTIONARY.md").write_text(dictionary(tables, row_counts), encoding="utf-8")
    (out / "DATASET_CARD.md").write_text(card(tables, store_name, run_dir, sim, row_counts), encoding="utf-8")
    say(f"Business data: {', '.join(f'{k} {n:,}' for k, n in row_counts.items())} rows -> {out.name}/")
    return row_counts


# ------------------------------------------------------------- documentation
COLUMNS = {
    "transaction_id": "receipt id (one shopping trip)", "line_number": "line on the receipt",
    "date": "calendar date (YYYY-MM-DD)", "checkout_time": "time the customer paid (HH:MM)",
    "arrival_time": "time the customer came in", "store_id": "store (generic format, not a real retailer)",
    "customer_id": "loyalty-card style customer id (simulated person)", "product_id": "product barcode / SKU",
    "product_name": "product name", "brand": "brand", "category": "department", "subcategory": "aisle / sub-category",
    "quantity": "units on this line", "unit_price": "price paid per unit (after promotion)",
    "regular_price": "shelf price without promotion; present even on zero-sale SKU-days",
    "shelf_price": "public shelf price on the date (targeted offers to a shopper group may differ)",
    "gross_sales": "regular_price x quantity",
    "discount_amount": "money off from promotions", "net_sales": "what the customer paid",
    "unit_cost": "store's cost per unit (ESTIMATED)", "cost_of_goods": "unit_cost x quantity",
    "gross_profit": "net_sales - cost_of_goods", "on_promotion": "sold on a promotion; in daily sales, at least one sold unit was on promotion",
    "promotion_offer_active": "at least one shopper group had a promotion offer for this SKU on this date, even if zero units sold",
    "promotion_type": "kind of promotion", "repeat_purchase": "customer had bought this product before",
    "purchase_reason": "why the item was bought: ran out at home, regular buy, new item, goes with another item, impulse",
    "decided_by": "who chose the product: student model (the thinking NPC), autopilot, LLM, or a store rule",
    "minutes_in_store": "time in the store", "checkout_wait_min": "waiting time at the till",
    "trip_type": "top_up / routine / stock_up", "lines": "different products on the receipt",
    "list_planned_by": "who wrote the shopping list: list student (the NPC brain) or a store rule (student runs only)",
    "units": "units on the receipt", "lines_on_promotion": "receipt lines on promotion",
    "units_sold": "units sold by simulated NPCs", "transactions": "paid simulated NPC receipts, not expanded store transactions",
    "avg_unit_price": "average price paid; blank if no units sold",
    "lost_sales_units": "units customers wanted but the shelf was empty and they bought nothing instead",
    "lost_sales_value": "value of those lost units", "closing_stock": "simulated NPC-scale units on the shelf after closing",
    "out_of_stock": "shelf empty at closing", "opening_stock": "units on the shelf before opening",
    "delivered_units": "units delivered that morning", "wasted_units": "units thrown away (expired)",
    "sold_units": "units sold", "on_order_units": "units ordered, not yet delivered",
    "reorder_point": "stock level that triggers an order", "store_forecast_daily": "the store's own demand forecast",
    "wasted_cost": "cost of wasted units", "stock_value_at_cost": "NPC-scale closing stock x unit_cost (in daily_store_kpis: the "
    "whole store)", "weekday": "day name", "stock_units": "units on all shelves at closing",
    "products_out_of_stock": "products with an empty shelf at closing", "products_carried": "products the store sells",
    "customers": "different customers", "gross_margin_pct": "gross_profit / net_sales x 100",
    "avg_basket_value": "net_sales per receipt", "units_per_basket": "units per receipt",
    "promotion_sales_share_pct": "share of net sales on promotion", "stockout_events": "times a customer met an empty shelf",
    "waste_cost": "cost of expired stock thrown away", "pack_size": "pack size", "promotion_eligible": "can be promoted",
    "shelf_life_days": "days before fresh stock expires (0 = not perishable)", "cost_label": "how unit_cost was made",
    "household_size": "people in the household", "children": "children in the household",
    "income_band": "income fifth, 1 (lowest) to 5", "weekly_food_budget": "usual weekly food budget",
    "price_sensitivity": "low / medium / high", "first_visit": "first visit in the period",
    "last_visit": "last visit in the period", "visits": "trips in the period", "total_spend": "net sales in the period",
    "favourite_category": "category with the most spend", "promotion_id": "promotion id", "start_date": "first day",
    "end_date": "last day", "discount_pct": "percent off", "promotion_price": "price during the promotion",
    "walk_metres": "metres walked in the store", "shelves_visited": "shelves stopped at",
    "aisles_passed": "aisles walked through", "age": "age (ESTIMATED to fit the household)",
    "shopper_role": "the NPC's role: Deal hunter, Steady regular or Convenience first",
    "name": "made-up first name and initial (SIMULATED person)", "shopping_style": "role with a short description",
    "bio": "short persona written from the NPC's facts", "cares_about": "what matters to them in the store",
    "quote": "something they might say", "habit_products": "products they buy again and again",
    "favourite_day": "day they like to shop", "days_between_trips": "usual days between trips",
    "data_label": "SIMULATED - made by the RETINA simulation, not real sales",
    "sample_weight": "number of expected shoppers represented by one simulated NPC; does not create extra receipts or stock",
    "estimate_status": "EXPANDED_NPC_SAMPLE, or ARRIVAL_CAP_REACHED when full-store estimates are blank because arrivals may have been clipped",
    "population_scope": "scope of unprefixed sales and receipt counts: simulated NPC sample or full simulated run",
    "stock_scope": "scope of all stock counts and values: simulated NPC sample or full simulated run, never weighted",
    "estimated_store_units_sold": "sample units sold multiplied by sample_weight; model estimate, not observed units",
    "estimated_store_transactions": "sample receipts multiplied by sample_weight; model estimate, not observed receipts",
    "estimated_store_net_sales": "sample net sales multiplied by sample_weight; model estimate, not observed sales",
}


def dictionary(tables, row_counts=None):
    out = ["# Data dictionary", "", "Every value is SIMULATED (unit costs ESTIMATED). Money in USD.",
           "Unprefixed transaction, sales and stock values describe the simulated NPC population. "
           "Only `estimated_store_*` values expand by `sample_weight`; stock values are never expanded.", ""]
    for name, df in tables.items():
        count = row_counts[name] if row_counts is not None else len(df)
        out += [f"## {name}.csv ({count:,} rows)", "", "| column | type | meaning |", "|---|---|---|"]
        for c in df.columns:
            kind = "number" if pd.api.types.is_numeric_dtype(df[c]) and df[c].dtype != bool else \
                ("true/false" if df[c].dtype == bool else "text")
            out.append(f"| {c} | {kind} | {COLUMNS.get(c, '')} |")
        out.append("")
    return "\n".join(out)


def card(tables, store_name, run_dir, sim, row_counts=None):
    k = tables["daily_store_kpis"]
    days = len(k)
    first, last = (k["date"].min(), k["date"].max()) if days else ("-", "-")
    mind = getattr(getattr(sim, "mind", None), "mode", None) or ("student" if run_dir.name.endswith("_student") else "fast")
    has_checkout_student = getattr(getattr(sim, "mind", None), "ender", None) is not None
    has_browse_student = getattr(getattr(sim, "mind", None), "lister", None) is not None
    who = {"student": "customers used RETINA's local student models for goals and shelf choices, with a "
                      "rule fallback where a model could not answer; no LLM was called during the live trip",
           "mix": "a panel of customers decided with the student / an LLM; the rest with the statistical engine",
           "fast": "every customer was simulated by the statistical engine (habits calibrated on public data)"}
    net = k["net_sales"].sum() if days else 0
    custom = PROFILE.get("custom_store")
    store_note = (f"- Store: {store_name} (`{PROFILE['id']}`), configured from user-provided store details. "
                  "It is not a record of this store's actual customers or sales." if custom else
                  f"- Store: {store_name} (`{PROFILE['id']}`), a generic format - never a real retailer.")
    if custom and PROFILE.get("catalog_kind") == "only":
        product_note = ("- Product names, brands, categories and prices were USER_PROVIDED in the store's CSV; "
                        "RETINA did not verify them against the retailer. Unit costs are ESTIMATED from category margins.")
    elif custom:
        product_note = ("- Products and prices use a generic US catalog, not this retailer's actual range. "
                        "Their match to this store is ESTIMATED; unit costs are ESTIMATED from category margins.")
    else:
        product_note = ("- Product names, brands and sizes come from Open Food Facts and prices from Open Prices "
                        "(ODbL 1.0); unit costs are ESTIMATED from category margins.")
    inputs_note = ("- Store-provided inputs: " +
                   ("product range and prices, " if PROFILE.get("catalog_kind") == "only" else "") +
                   "opening hours, visitor volume, checkout lanes; hourly visitor counts if supplied. "
                   "Other behavior and layout settings are ESTIMATED from a generic store profile.")
    limit = sim.traffic_settings.get("npc_limit")
    limit_note = (["- This run used an NPC arrival cap. On days that reached it, "
                   "`estimated_store_*` values are blank and `estimate_status` is "
                   "`ARRIVAL_CAP_REACHED`: multiplying sampled receipts might understate intended volume."]
                  if limit else [])
    return "\n".join([
        f"# RETINA sales dataset - {store_name}", "",
        "**SIMULATED retail sales data** - not real sales, no real customers. Made on this PC by RETINA "
        f"({datetime.now(timezone.utc):%Y-%m-%d}).", "",
        store_note,
        f"- Period: {first} to {last} ({days} days). Net sales ${net:,.0f}, "
        f"{int(k['transactions'].sum()) if days else 0:,} receipts.",
        f"- Random seed: {sim.seed}. Replaying it also requires the same store inputs, models and code version.",
        f"- Shopper sample weight: {1 / float(sim.traffic_settings.get('npc_scale', 1.0)):g} expected shoppers per NPC. "
        "Raw receipt rows are individual simulated NPC trips; estimated_store_* columns expand aggregate totals. "
        "They do not create additional independent transactions.",
        "- Unprefixed sales and receipt counts are from simulated NPCs. Stock units, waste and value "
        "are also at the NPC simulation scale; they are never expanded to a full-store inventory estimate.",
        *limit_note,
        f"- Customers: {who.get(mind, who['fast'])}.",
        "- The store: opening hours and traffic by hour, prices with inflation and weekly promotions, (s, S) "
        "stock ordering and late suppliers, expiry of fresh food, checkout queues.",
        *([inputs_note] if custom else []), "",
        "## Files", *[f"- `{n}.csv` - {(row_counts[n] if row_counts is not None else len(df)):,} rows"
                      for n, df in tables.items()],
        "- `DATA_DICTIONARY.md` - every column explained.", "",
        "## Good for", "- sales and profit analysis, category and brand performance, basket / market-basket analysis;",
        "- demand forecasting (daily_product_sales has every carried SKU on every completed day, "
        "including zero sales, offers and closing stock);",
        "- inventory: stock-outs, waste, reorder policies (inventory_daily records only stock movements); "
        "promotion effects (promotions);",
        "- customer analytics: frequency, spend, segments (customers + baskets).", "",
        "## Honest limits",
        product_note,
        "- Shopper habits are calibrated on public statistics (Instacart 2017 aggregates: non-commercial use).",
        *(["- Early checkout choices use a local model trained on online order lengths. Physical store "
           "departures were not observed; the in-store probability scale is ESTIMATED."]
          if has_checkout_student else []),
        *(["- Unplanned browse stops rescore visible, previously unselected products with the list student. "
           "The chance of stopping to browse is ESTIMATED; physical browsing was not observed."]
          if has_browse_student else []),
        "- No payments, returns or online orders are simulated.",
        "- inventory_daily has a row only when a product's stock moved; on other days it was unchanged.", ""])
