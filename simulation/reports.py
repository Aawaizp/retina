"""RETINA Mart reports: logs, summaries, the scorecard and saving the output files."""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import *
from world import *
import traffic

def summarize_prices(store, book):
    p = store.products
    eligible = p["promo_eligible"].to_numpy()
    print(f"\nPrices: {book.dates[0]:%d %b %Y} to {book.dates[-1]:%d %b %Y}")
    print(f"  on promotion in week 1: {book.on_promo[0].sum():,} products "
          f"({book.on_promo[0][eligible].mean() * 100:.0f}% of promo-eligible)")
    print(f"  average promo discount: {(1 - p.loc[eligible, 'promo_ratio']).mean() * 100:.0f}%")
    last = book.inflation[-1]
    print(f"  price change vs observed price by last month (median): {(np.median(last) - 1) * 100:+.1f}%")
    by_dept = pd.Series(last).groupby(p["department"].to_numpy()).median()
    print("  by department:", {k: f"{(v - 1) * 100:+.1f}%" for k, v in by_dept.round(4).items()})


def promo_lift(sales, store, book):
    """Real-world check: do promoted products sell more? Units per product-week, on vs off promo."""
    eligible = store.products["promo_eligible"].to_numpy()
    weeks = sales["day"] // 7
    n_weeks = weeks.nunique()
    on = book.on_promo[:n_weeks][:, eligible]
    units = sales[eligible[sales["product_idx"]]].groupby("on_promo")["quantity"].sum()
    per_on = units.get(True, 0) / on.sum()
    per_off = units.get(False, 0) / (~on).sum()
    print(f"  promotion lift: promoted products sold {per_on / per_off:.1f}x more per week "
          f"than when not promoted (research: promotions often 2x+)")


def summarize_sales(sales, store):
    p = store.products
    sales = sales.assign(line_total=sales["quantity"] * sales["unit_price"],
                         product_name=p["product_name"].to_numpy()[sales["product_idx"]],
                         department=p["department"].to_numpy()[sales["product_idx"]])
    print(f"\nSimulated {sales['day'].nunique()} days (SIMULATED data)")
    daily = sales.groupby("day").agg(trips=("trip_id", "nunique"), items=("quantity", "sum"),
                                     revenue=("line_total", "sum"))
    print(daily.round(2).to_string())
    trips = sales.groupby("trip_id")["quantity"].sum()
    print(f"\n  items per trip: average {trips.mean():.1f} | median {trips.median():.0f}")
    print(f"  average basket value: ${sales.groupby('trip_id')['line_total'].sum().mean():.2f}")
    print(f"  repeat purchases: {sales['is_repeat'].mean() * 100:.0f}% of lines "
          f"(starts at 0 - NPCs have no favorites on day 0)")
    print(f"  lines bought on promotion: {sales['on_promo'].mean() * 100:.0f}%")
    print("\n  top 5 products:")
    top = sales.groupby("product_name")["quantity"].sum().nlargest(5)
    print(top.to_string())
    print("\n  department share of items (simulated vs real):")
    sim_share = sales.groupby("department")["quantity"].sum() / sales["quantity"].sum()
    real_share = pd.Series(dict(zip(store.dept_names, store.dept_shares)))
    comp = pd.DataFrame({"simulated %": sim_share * 100, "real %": real_share * 100}).fillna(0)
    print(comp.sort_values("real %", ascending=False).head(6).round(1).to_string())
    example = sales[sales["trip_id"] == sales["trip_id"].iloc[0]]
    print(f"\n  example receipt {example['trip_id'].iloc[0]} (hour {example['hour'].iloc[0]}:00):")
    for _, r in example.iterrows():
        print(f"    {r['quantity']} x {r['product_name'][:40]:<40} ${r['line_total']:6.2f}")
    print(f"    {'TOTAL':>44} ${example['line_total'].sum():6.2f}")

# ================================================================= output
def build_sales_log(sales, store, book):
    """Readable sales log: one row per product line on a receipt. Clearly labelled SIMULATED."""
    p = store.products
    idx = sales["product_idx"].to_numpy()
    log = pd.DataFrame({
        "date": book.dates[sales["day"].to_numpy()],
        "day": sales["day"],
        "hour": sales["hour"],
        "trip_id": sales["trip_id"],
        "shopper_id": sales["shopper_id"],
        "product_id": p["product_id"].to_numpy()[idx],
        "product_name": p["product_name"].to_numpy()[idx],
        "department": p["department"].to_numpy()[idx],
        "cpi_item_code": p["cpi_item_code"].to_numpy()[idx],
        "quantity": sales["quantity"],
        "unit_price": sales["unit_price"],
        "regular_price": sales["regular_price"],
        "on_promo": sales["on_promo"],
        "discount_type": np.where(sales["on_promo"], p["discount_type"].to_numpy()[idx], None),
        "is_repeat": sales["is_repeat"],
        "mission": sales["mission"],          # top_up / routine / stock_up (trip size vs the NPC's usual)
        "reason": sales["reason"],            # need (ran out at home) / partner / new / impulse
        
    })
    month = book.month_of_day[sales["day"].to_numpy()]
    log["unit_cost"] = (p["unit_cost"].to_numpy()[idx] * book.inflation[month, idx]).round(2)
    log["line_total"] = (log["quantity"] * log["unit_price"]).round(2)
    log["data_label"] = "SIMULATED"
    return log


def save_outputs(log, store, shoppers, book, stockouts, waste_log, promo_log):
    """Sales log + 'truth' files (the hidden settings behind the data) + run information."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log.to_parquet(OUTPUT_DIR / "sales_log.parquet", index=False)

    # The store's shelves: every product it carries, with prices, cost and data labels
    carried = np.sort(np.concatenate([d.product_idx for d in store.departments.values()]))
    columns = ["product_id", "product_name", "brand", "department","aisle", "aisle_confidence", "regular_price",
                "promo_eligible", "promo_ratio", "unit_cost", "popularity"] + CATALOG_LABEL_COLUMNS
    shelves = store.products.loc[carried, columns].assign(cost_label="ESTIMATED")
    shelves.to_parquet(OUTPUT_DIR / "store_catalog.parquet", index=False)

    truth_shoppers = pd.DataFrame({
        "shopper_id": shoppers.shopper_id,
        "basket_mean": shoppers.basket_mean,
        "gap_mean": shoppers.gap_mean,
        "reorder_prob": shoppers.reorder_prob,
        "price_sensitivity": shoppers.price_sensitivity.round(3),
        "sensitivity_group": shoppers.sensitivity_group,
        "favorite_weekday": shoppers.favorite_weekday,
    })
    if shoppers.household_size is not None:
        truth_shoppers = truth_shoppers.assign(household_size=shoppers.household_size, kids=shoppers.kids,
                                               income_quintile=shoppers.income_quintile + 1,
                                               weekly_food_budget=shoppers.weekly_budget)
    truth_shoppers.to_parquet(OUTPUT_DIR / "truth_shoppers.parquet", index=False)

    week, prod = np.nonzero(book.on_promo)
    p = store.products
    truth_promos = pd.DataFrame({
        "week": week,
        "week_start": book.dates[np.minimum(week * 7, len(book.dates) - 1)],
        "product_id": p["product_id"].to_numpy()[prod],
        "promo_ratio": p["promo_ratio"].to_numpy()[prod].round(3),
    })
    truth_promos.to_parquet(OUTPUT_DIR / "truth_promotions.parquet", index=False)
    stockouts.to_parquet(OUTPUT_DIR / "truth_stockouts.parquet", index=False)
    waste_log.to_parquet(OUTPUT_DIR / "store_waste.parquet", index=False)
    promo_log.to_parquet(OUTPUT_DIR / "truth_promo_effects.parquet", index=False)

    run_info = {
        "label": "SIMULATED DATA - RETINA Mart v3, not real sales",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED, "n_shoppers": int(len(shoppers.shopper_id)), "start_date": START_DATE, "days": SIM_DAYS,
        "npc_scale": TRAFFIC["npc_scale"] if traffic.enabled() else None,   # 1 NPC = 1/scale real shoppers
        "store_profile": PROFILE,   # every setting with its data label and source
        "data_labels": DATA_LABELS,
        "rows": int(len(log)),
        "inputs": "data/processed (see data/data_sources.json for sources and licences)",
    }
    with open(OUTPUT_DIR / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR}:")
    for f in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {f.name:<26} {f.stat().st_size / 1e6:6.1f} MB")


def build_stockout_log(misses, store, book):
    """Every time a shopper found an empty shelf: what they wanted, what they did, and the
    sale the store lost. A real store never sees lost sales - here they are the 'truth'."""
    p = store.products
    idx = misses["product_idx"].to_numpy()
    sub = misses["substitute_idx"].to_numpy()
    day = misses["day"].to_numpy()
    regular = p["regular_price"].to_numpy()[idx] * book.inflation[book.month_of_day[day], idx]
    promo = book.on_promo[book.week_of_day[day], idx]
    price = (regular * np.where(promo, p["promo_ratio"].to_numpy()[idx], 1.0)).round(2)
    has_sub = sub >= 0
    return pd.DataFrame({
        "date": book.dates[day],
        "day": day,
        "hour": misses["hour"],
        "shopper_id": misses["shopper_id"],
        "product_id": p["product_id"].to_numpy()[idx],
        "product_name": p["product_name"].to_numpy()[idx],
        "department": p["department"].to_numpy()[idx],
        "unit_price": price,
        "on_promo": promo,
        "reaction": misses["reaction"],
        "substitute_id": np.where(has_sub, p["product_id"].to_numpy()[np.maximum(sub, 0)], None),
        "substitute_name": np.where(has_sub, p["product_name"].to_numpy()[np.maximum(sub, 0)], None),
        "lost_sale": ~has_sub,
    })


def summarize_stock(stockouts, stock_stats, log):
    print("\nStock (v2): textbook (s, S) ordering, 2-day deliveries, imperfect forecasts")
    print(f"  shelves empty at closing: {stock_stats['empty_share'].mean() * 100:.1f}% of products "
          f"(low: in a small store most of the 20,000+ slow sellers rarely run out)")
    print(f"    promo-eligible products: {stock_stats['empty_share_promo'].mean() * 100:.1f}% when on "
          f"promotion vs {stock_stats['empty_share_no_promo'].mean() * 100:.1f}% when not")
    wanted = log["quantity"].sum() + stockouts["lost_sale"].sum()
    print(f"  shoppers found an empty shelf {len(stockouts):,} times "
          f"({len(stockouts) / wanted * 100:.1f}% of items they wanted; research: ~8%)")
    share = stockouts["reaction"].value_counts(normalize=True)
    print("  what they did (simulated vs research):")
    for reaction, real in OOS_REACTIONS.items():
        print(f"    {reaction:<12} {share.get(reaction, 0) * 100:5.1f}%   {real * 100:4.0f}%")
    lost = stockouts.loc[stockouts["lost_sale"], "unit_price"].sum()
    print(f"  lost sales: ${lost:,.0f} = {lost / (log['line_total'].sum() + lost) * 100:.1f}% of "
          f"possible revenue (research: ~4%)")
    top = stockouts.groupby("product_name").size().nlargest(3)
    print("  most often out of stock:", ", ".join(f"{n[:30]} ({c})" for n, c in top.items()))

def build_waste_log(waste, store, book):
    """Fresh units thrown away: a real store records this (it is part of 'shrink')."""
    p = store.products
    waste = pd.DataFrame(waste, columns=["day", "product_idx", "units"])
    idx = waste["product_idx"].to_numpy()
    day = waste["day"].to_numpy()
    cost = p["unit_cost"].to_numpy()[idx] * book.inflation[book.month_of_day[day], idx]
    return pd.DataFrame({
        "date": book.dates[day],
        "day": day,
        "product_id": p["product_id"].to_numpy()[idx],
        "product_name": p["product_name"].to_numpy()[idx],
        "department": p["department"].to_numpy()[idx],
        "units": waste["units"],
        "unit_cost": cost.round(2),
        "total_cost": (waste["units"] * cost).round(2),
    })


def money_summary(log, waste_log):
    """Store profit and loss view: revenue, cost of goods sold, waste, gross margin."""
    revenue = log["line_total"].sum()
    cogs = (log["quantity"] * log["unit_cost"]).sum()
    waste_cost = waste_log["total_cost"].sum()
    margin = (revenue - cogs - waste_cost) / revenue
    sold = log.groupby("department")["quantity"].sum()
    thrown = waste_log.groupby("department")["units"].sum()
    spoil = (thrown / (sold.reindex(thrown.index) + thrown)).dropna()
    return {"revenue": revenue, "cogs": cogs, "waste_cost": waste_cost, "margin": margin,
            "spoil_by_dept": spoil}


def summarize_money(money, log):
    print("\nMoney (v2): cost = regular price x (1 - department margin), fresh items spoil")
    print(f"  revenue            ${money['revenue']:>12,.0f}")
    print(f"  cost of goods sold ${money['cogs']:>12,.0f}")
    print(f"  thrown away        ${money['waste_cost']:>12,.0f}")
    print(f"  gross margin       {money['margin'] * 100:>12.1f}%   (Kroger 2025: 22.9%, "
          f"Albertsons FY2025: 27.2%)")
    promo = log[log["on_promo"]]
    promo_margin = 1 - (promo["quantity"] * promo["unit_cost"]).sum() / promo["line_total"].sum()
    print(f"  margin on promoted lines: {promo_margin * 100:.1f}% (discounts eat margin)")
    print("  share of fresh units thrown away (research: produce ~5-13%, meat ~13%):")
    for dept, share in money["spoil_by_dept"].sort_values(ascending=False).items():
        print(f"    {dept:<14} {share * 100:5.1f}%")

def build_promo_log(effects, store, book):
    """Why promoted units sold (the hidden truth behind the promotion bump), plus the
    'had_at_home' skips that cause the dip after a promotion."""
    p = store.products
    e = pd.DataFrame(effects, columns=["day", "shopper_id", "product_idx", "effect", "units"])
    idx = e["product_idx"].to_numpy()
    return pd.DataFrame({
        "date": book.dates[e["day"].to_numpy()],
        "day": e["day"],
        "shopper_id": e["shopper_id"],
        "product_id": p["product_id"].to_numpy()[idx],
        "product_name": p["product_name"].to_numpy()[idx],
        "department": p["department"].to_numpy()[idx],
        "effect": e["effect"],
        "units": e["units"],
    })


def promo_split(promo_log):
    """Share of the promotion bump from each source (research: about 1/3 each)."""
    units = promo_log.groupby("effect")["units"].sum()
    bump = units.reindex(["brand_switch", "stock_up", "impulse"]).fillna(0)
    return bump / bump.sum(), units


def summarize_promo(promo_log, sales, store, book):
    split, units = promo_split(promo_log)
    print("\nPromotions (v3): where the extra promo sales come from (research: ~1/3 each)")
    names = {"brand_switch": "brand switching", "stock_up": "stocking up (borrowed from later)",
             "impulse": "extra unplanned buying"}
    for effect, share in split.items():
        print(f"  {names[effect]:<36} {share * 100:5.1f}%  ({int(units.get(effect, 0)):,} units)")
    print(f"  skipped later because they had some at home: {int(-units.get('had_at_home', 0)):,} items")

    # Post-promotion dip: units per product-week in the week AFTER a promotion vs normal weeks
    eligible = np.flatnonzero(store.products["promo_eligible"].to_numpy())
    weeks = (sales["day"] // 7).to_numpy()
    n_weeks = weeks.max() + 1
    grid = np.zeros((n_weeks, len(store.products)))
    np.add.at(grid, (weeks, sales["product_idx"].to_numpy()), sales["quantity"].to_numpy())
    on = book.on_promo[:n_weeks]
    after = np.zeros_like(on)
    after[1:] = on[:-1] & ~on[1:]
    normal = ~on & ~after
    g, a, n = grid[:, eligible], after[:, eligible], normal[:, eligible]
    dip = g[a].mean() / g[n].mean()
    print(f"  week after a promotion: {dip:.2f}x normal sales (below 1 = post-promotion dip)")



def summarize_memory(memory_stats, promo_log):
    m = pd.DataFrame(memory_stats, columns=["day", "feels_expensive", "feels_cheap"])
    print("\nPrice memory (v3): shoppers compare prices with what they remember")
    print(f"  average week: {m['feels_expensive'].mean() * 100:.1f}% of products feel more expensive "
          f"than remembered, {m['feels_cheap'].mean() * 100:.1f}% feel cheaper")
    counts = promo_log["effect"].value_counts()
    print(f"  (v4) usual items swapped for a similar product because they got pricier: "
          f"{counts.get('price_switch', 0):,}")
    print(f"  (v4) items not bought because the whole aisle felt too expensive (walk-away): "
          f"{counts.get('walked_away', 0):,}")


def pair_report(sales, store):
    """Do simulated baskets show the real pairs? Lift = how many times more often two products
    share a basket than by chance. Measured for pairs seen together in at least 10 trips."""
    trips = sales[["trip_id", "product_idx"]].drop_duplicates()
    rows = [(min(a, b), max(a, b), l) for a, (idx, _, lift) in store.partners.items()
            for b, l in zip(idx, lift)]
    pairs = pd.DataFrame(rows, columns=["a", "b", "real_lift"]).drop_duplicates(["a", "b"])
    both = (trips.rename(columns={"product_idx": "a"}).merge(pairs, on="a")
            .merge(trips.rename(columns={"product_idx": "b"}), on=["trip_id", "b"]))
    together = both.groupby(["a", "b"]).size().rename("together").reset_index()
    pairs = pairs.merge(together, on=["a", "b"])
    pairs = pairs[pairs["together"] >= 10].copy()
    count = trips["product_idx"].value_counts()
    n_trips = trips["trip_id"].nunique()
    pairs["sim_lift"] = (pairs["together"] * n_trips
                         / (pairs["a"].map(count).to_numpy() * pairs["b"].map(count).to_numpy()))
    return {"pairs": len(pairs),
            "corr": np.corrcoef(np.log(pairs["real_lift"]), np.log(pairs["sim_lift"]))[0, 1],
            "above_chance": (pairs["sim_lift"] > 1).mean(),
            "median_real": pairs["real_lift"].median(), "median_sim": pairs["sim_lift"].median()}


def summarize_pairs(pair_stats, promo_log, log):
    picked = promo_log.loc[promo_log["effect"] == "bought_together", "units"].sum()
    print("\nBought together (v3): real Instacart pairs (chips -> dip, pasta -> pasta of another shape)")
    print(f"  items picked because they go with the previous item: {picked:,} "
          f"({picked / log['quantity'].sum() * 100:.1f}% of items)")
    print(f"  {pair_stats['pairs']:,} real pairs measurable in simulated baskets: "
          f"{pair_stats['above_chance'] * 100:.0f}% bought together more than chance")
    print(f"  pair strength vs real (correlation of lifts): {pair_stats['corr']:.2f} | median lift "
          f"{pair_stats['median_sim']:.1f} simulated vs {pair_stats['median_real']:.1f} real (weaker)")

# ============================================================== scorecard
def scorecard(log, params, store, stock_stats, stockouts, money, promo_log, pair_stats):   
    """Compare the simulated year with real Instacart patterns.
    'input' = we fed this pattern in, so matching only proves the code works.
    'emergent' = not fed in directly; matching is real evidence the model is realistic.
    'tuned' = we adjusted assumptions until it matched, so it is NOT evidence either."""
    def mean_of(dist):
        return sum(float(k) * v for k, v in dist.items())

    rows = []
    def check(name, kind, sim, real, ok):
        rows.append((name, kind, sim, real, "PASS" if ok else "FAIL"))

    trips = log.groupby("trip_id")["quantity"].sum()
    real_basket = mean_of(params["basket_size"])
    check("items per trip (average)", "input", f"{trips.mean():.1f}", f"{real_basket:.1f}",
          abs(trips.mean() - real_basket) / real_basket < 0.15)

       # Planned items only: promotion extras (stocking up, impulse buys) are checked separately
    extras = promo_log[promo_log["effect"].isin(["stock_up", "impulse"])]
    planned = log.groupby("department")["quantity"].sum().sub(
        extras.groupby("department")["units"].sum(), fill_value=0)
    sim_dept = planned / planned.sum()

    real_dept = pd.Series(dict(zip(store.dept_names, store.dept_shares)))
    gap = (sim_dept.reindex(real_dept.index).fillna(0) - real_dept).abs().max()
    check("department shares (largest gap)", "input", f"{gap * 100:.1f} pts", "0", gap < 0.02)

    hours = log.drop_duplicates("trip_id")["hour"].value_counts(normalize=True)
    if traffic.enabled():                                 # the store's traffic schedule (open hours)
        real_hours = pd.Series(traffic.hour_shape())
    else:
        real_hours = pd.Series(params["orders_by_hour"]).rename(index=int)
    hour_gap = (hours.reindex(real_hours.index).fillna(0) - real_hours).abs().max()
    check("trips by hour (largest gap)", "input", f"{hour_gap * 100:.1f} pts", "0", hour_gap < 0.01)

    second_half = log[log["day"] >= (log["day"].max() + 1) // 2]   # works for any run length
    real_reorder = params["reorder_rate"]
    sim_reorder = second_half["is_repeat"].mean()
    check("repeat purchases (items, 2nd half)", "emergent", f"{sim_reorder * 100:.0f}%",
          f"{real_reorder * 100:.0f}%", abs(sim_reorder - real_reorder) < 0.10)

    units = log.groupby("product_id")["quantity"].sum().sort_values(ascending=False)
    carried = sum(len(d.product_idx) for d in store.departments.values())   # on the shelves
    top20 = units.head(max(1, int(carried * 0.2))).sum() / units.sum()
    check("top 20% of products = share of sales", "emergent", f"{top20 * 100:.0f}%", "~80%",
          0.70 <= top20 <= 0.90)

    trip_days = log.drop_duplicates("trip_id")[["shopper_id", "day"]].sort_values(["shopper_id", "day"])
    gaps = trip_days.groupby("shopper_id")["day"].diff().dropna()
    real_gap = mean_of(params["days_between_orders"])
    check("days between trips (average)", "emergent", f"{gaps.mean():.1f}", f"{real_gap:.1f}",
          abs(gaps.mean() - real_gap) / real_gap < 0.25)

    dow = log.drop_duplicates("trip_id")["date"].dt.dayofweek.value_counts(normalize=True)
    if traffic.enabled():                                 # weekends busier: ATUS, fed in by the schedule
        real_dow = dict(enumerate(traffic.weekday_shares()))
        kind = "input"
    else:
        real_dow, kind = params["orders_by_day_of_week"], "emergent"
    real_spread = max(real_dow.values()) - min(real_dow.values())
    sim_spread = dow.max() - dow.min()
    check("busy vs quiet weekday (spread)", kind, f"{sim_spread * 100:.1f} pts",
          f"{real_spread * 100:.1f} pts", abs(sim_spread - real_spread) < 0.02)
    
    wanted = log["quantity"].sum() + stockouts["lost_sale"].sum()
    missed = len(stockouts) / wanted
    check("items wanted but out of stock", "tuned", f"{missed * 100:.1f}%", "~8%",
            abs(missed - 0.08) < 0.04)
    promo_empty = stock_stats["empty_share_promo"].mean()
    other_empty = stock_stats["empty_share_no_promo"].mean()
    check("more stockouts when on promotion", "emergent",
            f"{promo_empty / max(other_empty, 1e-9):.1f}x", "higher", promo_empty > other_empty)
    lost = stockouts.loc[stockouts["lost_sale"], "unit_price"].sum()
    lost_share = lost / (log["line_total"].sum() + lost)
    check("sales lost to empty shelves", "emergent", f"{lost_share * 100:.1f}%", "~4%",
            abs(lost_share - 0.04) < 0.02)
    check("gross margin (after promos and waste)", "emergent", f"{money['margin'] * 100:.1f}%",
             "23-27%", 0.20 <= money["margin"] <= 0.30)
    produce_spoil = money["spoil_by_dept"].get("produce", 0)
    check("produce thrown away", "emergent", f"{produce_spoil * 100:.1f}%", "5-13%",
             0.04 <= produce_spoil <= 0.14)
    split, _ = promo_split(promo_log)
    check("promo bump split (switch/stock/extra)", "tuned",
             "/".join(f"{v * 100:.0f}" for v in split), "33/33/33", split.between(0.2, 0.45).all())
    check("real pairs bought together (lift corr.)", "input", f"{pair_stats['corr']:.2f}", "high",
             pair_stats["corr"] >= 0.7 and pair_stats["above_chance"] >= 0.9)
    real = real_shopping_stats()              # real receipts and price tags (Open Prices)
    if real:
        u = real["units_per_receipt_line"]
        lo, hi = sorted([u["usd"]["share"]["1"], u["all"]["share"]["1"]])
        one = (log["quantity"] == 1).mean()
        check("lines with 1 unit (real receipts)", "emergent", f"{one * 100:.0f}%",
              f"{lo * 100:.0f}-{hi * 100:.0f}%", lo - 0.03 <= one <= hi + 0.03)
        d = real["discount_depth"]
        promo = log[log["on_promo"]]
        depth = (1 - promo["unit_price"] / promo["regular_price"]).median()
        lo, hi = sorted([d["usd"]["0.5"], d["all"]["0.5"]])
        check("promo discount, median (real tags)", "input", f"{depth * 100:.0f}%",
              f"{lo * 100:.0f}-{hi * 100:.0f}%", lo - 0.05 <= depth <= hi + 0.05)
    if "mission" in log and TRIP_STATS_FILE.exists():
        real_types = json.loads(TRIP_STATS_FILE.read_text(encoding="utf-8"))["trip_type_shares"]
        sim_types = log.drop_duplicates("trip_id")["mission"].value_counts(normalize=True)
        gap = max(abs(sim_types.get(k, 0) - v) for k, v in real_types.items())
        check("trip types top-up/routine/stock-up", "input",
              "/".join(f"{sim_types.get(k, 0) * 100:.0f}" for k in real_types),
              "/".join(f"{v * 100:.0f}" for v in real_types.values()), gap < 0.03)
    if AISLE_CYCLES_FILE.exists():
        sim_next, real_next = rebuy_next_trip(log, store)
        check("same product again next trip", "tuned", f"{sim_next * 100:.0f}%", f"{real_next * 100:.0f}%",
              abs(sim_next - real_next) < 0.05)
    print("\nSCORECARD - simulated year vs real data (Instacart orders, Open Prices receipts)")
    print(f"  {'check':<38} {'type':<9} {'simulated':>10} {'real':>10}  result")
    for name, kind, sim, real, res in rows:
        print(f"  {name:<38} {kind:<9} {sim:>10} {real:>10}  {res}")
    passed = sum(r[4] == "PASS" for r in rows)
    print(f"  {passed}/{len(rows)} passed. FAILs show what to improve next - that is the point.")
    return rows


def summarize_needs(log, trips, promo_log, shoppers):
    """Step 14: why NPCs shop - trip types, why each item was bought, the household budget."""
    if "mission" not in log:
        return
    t = trips.drop_duplicates("trip_id")
    types = t["mission"].value_counts(normalize=True)
    print("\nNeeds (v6): NPCs shop for a household, from what ran out at home, within a food budget")
    print("  trip types: " + " | ".join(f"{k.replace('_', '-')} {types.get(k, 0):.0%}" for k in ("top_up", "routine", "stock_up"))
          + "   (real Instacart: top-up 14% | routine 72% | stock-up 14%)")
    by_type = t.groupby("mission")["items"].mean()
    print("  items per trip: " + " | ".join(f"{k.replace('_', '-')} {by_type.get(k, 0):.1f}" for k in ("top_up", "routine", "stock_up")))
    why = log["reason"].value_counts(normalize=True)
    names = {"need": "ran out at home", "repeat": "usual item, buying ahead", "new": "new or different choice",
             "partner": "goes with another item", "impulse": "deal seen on the way"}
    print("  why each item was bought: " + " | ".join(f"{names[k]} {why.get(k, 0):.0%}" for k in names))
    over = promo_log[promo_log["effect"] == "over_budget"]
    hit = (t["spent"] > t["trip_budget"]).mean()
    print(f"  budget: {over['units'].sum():,} extra items left on the shelf because the week's food money was spent "
          f"({over['units'].sum() / max(log['quantity'].sum(), 1):.1%} of items); {hit:.0%} of trips went over on needed items")
    if shoppers.income_quintile is not None:
        q = pd.Series(shoppers.income_quintile, index=shoppers.shopper_id)
        spend = t.assign(q=t["shopper_id"].map(q)).groupby("q")["spent"].mean()
        print(f"  average spend per trip: lowest income fifth ${spend.get(0, 0):.0f} | highest ${spend.get(4, 0):.0f}")


def rebuy_next_trip(log, store, warmup=60):
    """Of the products a shopper buys on a trip, the share they buy AGAIN on their next trip
    (trips that have a next trip; after a warm-up). Real figure = Instacart's rate per aisle,
    weighted by how often each aisle appears in the simulated trips, so both use the same mix."""
    trips = log.drop_duplicates("trip_id")[["trip_id", "shopper_id", "day"]].sort_values(["shopper_id", "day"])
    trips["n"] = trips.groupby("shopper_id").cumcount()
    trips["last"] = trips.groupby("shopper_id")["n"].transform("max")
    lines = log[["trip_id", "shopper_id", "product_id", "day"]].merge(trips[["trip_id", "n", "last"]], on="trip_id")
    lines = lines[lines["day"] >= warmup]
    nxt = lines[["shopper_id", "product_id", "n"]].assign(n=lines["n"] - 1, again=True)
    lines = lines.merge(nxt, on=["shopper_id", "product_id", "n"], how="left")
    lines = lines[lines["n"] < lines["last"]]
    aisle = pd.Series(store.products["aisle"].to_numpy(), index=store.products["product_id"].to_numpy())
    lines["aisle"] = lines["product_id"].map(aisle[~aisle.index.duplicated()])
    real = pd.read_csv(AISLE_CYCLES_FILE).set_index("aisle")["bought_again_next_trip"]
    lines = lines[lines["aisle"].isin(real.index)]
    return float(lines["again"].fillna(False).mean()), float(real[lines["aisle"]].mean())


def real_shopping_stats():
    """Real units per receipt line and discount depths (built by loaders/clean_catalog.py)."""
    path = PROCESSED_DIR / "real_shopping_stats.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def summarize_year(log):
    log = log.assign(month=log["date"].dt.to_period("M"))
    monthly = log.groupby("month").agg(trips=("trip_id", "nunique"), items=("quantity", "sum"),
                                       revenue=("line_total", "sum"),
                                       promo_share=("on_promo", "mean"))
    monthly["avg_price"] = (log.groupby("month")["unit_price"].mean()).round(2)
    print(f"\nSimulated year (SIMULATED data): {log['date'].min():%d %b %Y} to {log['date'].max():%d %b %Y}")
    print(monthly.round({"revenue": 0, "promo_share": 2}).to_string())
    print(f"  total: {log['trip_id'].nunique():,} trips | {log['quantity'].sum():,} items | "
          f"${log['line_total'].sum():,.0f} revenue")