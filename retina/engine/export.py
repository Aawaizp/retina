"""RETINA dataset export: one command -> a ready-to-use synthetic retail dataset.

  python -m retina.engine.export                       one year, seed 42 -> data/export/retina_<store>_seed42/
  python -m retina.engine.export --seed 7 --days 180   another dataset (different luck, same store)
  python -m retina.engine.export --csv                  also write CSV (gzip) next to Parquet
  python -m retina.engine.export --no-realism           skip the realism test (needs the Instacart files)

What you get (one folder):
  transactions   one row per product line on a receipt (the main table)
  trips          one row per shopping trip: times, walk, queue, trip type, budget, spend
  products       the store's shelves: every product it carries, with prices and data labels
  customers      the NPC customers: habits, household, income, budget (all SIMULATED people)
  daily_store    one row per day: trips, items, revenue, gross profit, empty shelves, waste
  hourly_traffic one row per day and opening hour: trips (and the real-store equivalent)
  truth/         what a real store can NOT see: lost sales at empty shelves, why promoted units
                 sold, the promotion calendar, waste - for testing forecasting / causal models
  DATASET_CARD.md, data_dictionary.csv / DATA_DICTIONARY.md, scorecard.csv, realism.json,
  checks.json (row counts and totals that must agree), run_info.json
Every row is SIMULATED. Inputs and their labels: see DATASET_CARD.md.
"""

import argparse
import json
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config, csvsafe
from .config import ROOT, PROFILE, DATA_LABELS, CATALOG_LABEL_COLUMNS
from . import traffic

EXPORT_DIR = ROOT / "data" / "export"
VERSION = "1.0"

# ============================================================ data dictionary
# table -> (what one row is, {column: (description, label)})
S, P, R, E = "SIMULATED", "PUBLIC_STATISTIC", "REAL_OBSERVED", "ESTIMATED"
DICTIONARY = {
    "transactions": ("one product line on one receipt", {
        "date": ("calendar date of the trip", S), "day": ("simulation day (0 = first day)", S),
        "hour": ("hour the shopper walked in (0-23)", S), "trip_id": ("trip (receipt) id, joins to trips", S),
        "shopper_id": ("NPC customer id, joins to customers", S),
        "product_id": ("product barcode/id, joins to products", R), "product_name": ("product name", R),
        "brand": ("brand, when known", R), "department": ("store department", E), "aisle": ("aisle (product group)", E),
        "cpi_item_code": ("BLS CPI item the product's price follows", E),
        "quantity": ("units bought", S), "unit_price": ("price paid per unit that day", S),
        "regular_price": ("normal shelf price that day (before any promotion)", S),
        "on_promo": ("True if the product was on promotion for this shopper", S),
        "discount_type": ("kind of promotion (from real price tags)", R),
        "is_repeat": ("True if this shopper had bought the product before", S),
        "mission": ("trip type: top_up / routine / stock_up (size vs the shopper's usual)", S),
        "reason": ("why it was bought: need (ran out at home) / repeat / new / partner / impulse", S),
        "unit_cost": ("store's cost per unit (regular price x department margin, inflated)", E),
        "line_total": ("quantity x unit_price", S), "gross_profit": ("line_total - quantity x unit_cost", S),
        "data_label": ("always SIMULATED", S)}),
    "trips": ("one shopping trip", {
        "trip_id": ("trip id", S), "date": ("date", S), "day": ("simulation day", S), "shopper_id": ("customer id", S),
        "arrive_time": ("time they walked in (HH:MM)", S), "leave_time": ("time they left (HH:MM)", S),
        "minutes_in_store": ("minutes from entrance to exit", S), "walk_m": ("metres walked on the floor plan", S),
        "shelves_visited": ("shelf stops on their route", S), "aisles_seen": ("aisles walked past", S),
        "items": ("units bought", S), "lines": ("different products bought", S),
        "planned_items": ("items on their list before skips and stockouts", S),
        "mission": ("trip type: top_up / routine / stock_up", S),
        "trip_budget": ("food money available for this trip (household weekly budget based)", S),
        "spend": ("amount paid", S), "lane": ("checkout lane", S), "wait_min": ("minutes waiting at checkout", S),
        "data_label": ("always SIMULATED", S)}),
    "products": ("one product the store carries", {
        "product_id": ("product barcode/id", R), "product_name": ("name", R), "brand": ("brand", R),
        "department": ("department", E), "aisle": ("aisle (product group)", E),
        "aisle_confidence": ("how sure the aisle assignment is", E),
        "regular_price": ("observed shelf price (Open Prices), before inflation over the year", R),
        "promo_eligible": ("can go on promotion", E), "promo_ratio": ("price on promotion / regular price", R),
        "unit_cost": ("store's cost per unit", E), "popularity": ("relative demand weight (Instacart orders when matched)", P),
        "price_label": ("label of the price field", R), "product_facts_label": ("label of name/brand", R),
        "popularity_label": ("label of popularity", E), "department_label": ("label of department", E),
        "cpi_category_label": ("label of the CPI category", E), "aisle_label": ("label of the aisle", E),
        "cost_label": ("label of unit_cost", E)}),
    "customers": ("one NPC customer (a SIMULATED person - not a real customer)", {
        "shopper_id": ("customer id", S), "basket_mean": ("usual items per trip (copied from one real Instacart shopper)", P),
        "gap_mean": ("usual days between trips (same real shopper)", P), "reorder_prob": ("re-buy rate (same real shopper)", P),
        "price_sensitivity": ("1.0 = average reaction to price", E), "sensitivity_group": ("0 low / 1 medium / 2 high", E),
        "favorite_weekday": ("0 = Monday ... 6 = Sunday", P), "household_size": ("people they shop for", P),
        "kids": ("children at home", E), "income_quintile": ("1 = lowest fifth ... 5 = highest", P),
        "weekly_food_budget": ("food-at-home budget per week, dollars", P), "data_label": ("always SIMULATED", S)}),
    "daily_store": ("one day in the store", {
        "date": ("date", S), "day": ("simulation day", S), "trips": ("shopping trips", S), "items": ("units sold", S),
        "revenue": ("sales, dollars", S), "gross_profit": ("revenue - cost of goods sold", S),
        "promo_share": ("share of receipt lines on promotion", S),
        "empty_shelf_share": ("share of products with no stock at closing", S),
        "stockout_moments": ("times a shopper found an empty shelf", S), "lost_sales": ("$ lost to empty shelves (truth)", S),
        "waste_cost": ("cost of fresh stock thrown away", S),
        "expected_real_shoppers": ("real-store footfall the day is scaled from (traffic schedule)", P),
        "data_label": ("always SIMULATED", S)}),
    "hourly_traffic": ("one opening hour on one day", {
        "date": ("date", S), "hour": ("hour", S), "trips": ("NPC trips starting that hour", S),
        "real_equivalent": ("trips / npc_scale = the real-store number it stands for", S),
        "data_label": ("always SIMULATED", S)}),
    "truth/lost_sales": ("one moment a shopper found an empty shelf (a real store never sees this)", {
        "date": ("date", S), "hour": ("hour", S), "shopper_id": ("customer id", S), "product_id": ("wanted product", R),
        "product_name": ("product name", R), "department": ("department", E),
        "unit_price": ("its price that day", S), "on_promo": ("True if it was on promotion", S),
        "reaction": ("other_store / other_brand / same_brand / delay / skip (Gruen et al. 2002 shares)", P),
        "substitute_id": ("product bought instead, if any", S), "lost_sale": ("True = the store lost the sale", S)}),
    "truth/promo_effects": ("one behaviour event: why a unit sold or was skipped", {
        "date": ("date", S), "day": ("simulation day", S), "shopper_id": ("customer id", S), "product_id": ("product", R),
        "product_name": ("product name", R), "department": ("department", E),
        "effect": ("brand_switch / stock_up / impulse / bought_together / had_at_home / price_switch / walked_away / over_budget", S),
        "units": ("units affected", S)}),
    "truth/promotions": ("one product on promotion in one week (the store's plan)", {
        "week": ("week number", S), "week_start": ("first day of that week", S), "product_id": ("product", R),
        "promo_ratio": ("promo price / regular price", R)}),
    "truth/waste": ("fresh stock thrown away on one day", {
        "date": ("date", S), "day": ("simulation day", S), "product_id": ("product", R),
        "product_name": ("product name", R), "department": ("department", E), "units": ("units thrown away", S),
        "unit_cost": ("cost per unit", E), "total_cost": ("units x unit cost", S)}),
}


def clock(minutes):
    m = np.asarray(minutes, dtype=float)
    return [f"{int(x // 60):02d}:{int(x % 60):02d}" for x in m]


# ============================================================ tables
def build_tables(run):
    r = run["results"]
    store, book, shoppers = r.store, r.book, r.shoppers
    p = store.products
    log = run["log"].copy()
    idx = r.sales["product_idx"].to_numpy()
    log.insert(log.columns.get_loc("department") + 1, "aisle", p["aisle"].to_numpy()[idx])
    log.insert(log.columns.get_loc("product_name") + 1, "brand", p["brand"].to_numpy()[idx])
    log["gross_profit"] = (log["line_total"] - log["quantity"] * log["unit_cost"]).round(2)
    log = log[[c for c in log.columns if c != "data_label"] + ["data_label"]]

    t = r.trips.copy()
    spend = log.groupby("trip_id")["line_total"].sum().round(2)
    trips = pd.DataFrame({
        "trip_id": t["trip_id"], "date": book.dates[t["day"].to_numpy()].date, "day": t["day"],
        "shopper_id": t["shopper_id"], "arrive_time": clock(t["arrive_min"]), "leave_time": clock(t["leave_min"]),
        "minutes_in_store": t["minutes_in_store"].round(1), "walk_m": t["walk_m"].round(1),
        "shelves_visited": t["shelves_visited"], "aisles_seen": t["aisles_seen"], "items": t["items"],
        "lines": t["lines"]})
    for col in ("planned_items", "mission", "trip_budget"):
        if col in t:
            trips[col] = t[col]
    trips["spend"] = trips["trip_id"].map(spend).fillna(0.0)
    trips["lane"], trips["wait_min"] = t["lane"], t["wait_min"].round(2)
    trips["data_label"] = "SIMULATED"

    carried = np.sort(np.concatenate([d.product_idx for d in store.departments.values()]))
    cols = ["product_id", "product_name", "brand", "department", "aisle", "aisle_confidence", "regular_price",
            "promo_eligible", "promo_ratio", "unit_cost", "popularity"] + CATALOG_LABEL_COLUMNS
    products = p.loc[carried, [c for c in cols if c in p]].assign(cost_label="ESTIMATED").reset_index(drop=True)

    customers = pd.DataFrame({"shopper_id": shoppers.shopper_id, "basket_mean": shoppers.basket_mean.round(2),
                              "gap_mean": shoppers.gap_mean.round(2), "reorder_prob": shoppers.reorder_prob.round(4),
                              "price_sensitivity": shoppers.price_sensitivity.round(3),
                              "sensitivity_group": shoppers.sensitivity_group,
                              "favorite_weekday": shoppers.favorite_weekday})
    if shoppers.household_size is not None:
        customers["household_size"] = shoppers.household_size
        customers["kids"] = shoppers.kids
        customers["income_quintile"] = shoppers.income_quintile + 1
        customers["weekly_food_budget"] = shoppers.weekly_budget
    customers["data_label"] = "SIMULATED"

    stock = run["stockouts"]
    waste = run["waste"]
    days = pd.DataFrame({"day": np.arange(len(book.dates))})
    by = log.groupby("day").agg(trips=("trip_id", "nunique"), items=("quantity", "sum"), revenue=("line_total", "sum"),
                                gross_profit=("gross_profit", "sum"), promo_share=("on_promo", "mean"))
    daily = days.join(by, on="day")
    daily = daily[daily["day"] < r.stock_stats["day"].max() + 1].copy()
    daily.insert(0, "date", book.dates[daily["day"].to_numpy()].date)
    daily = daily.merge(r.stock_stats[["day", "empty_share"]].rename(columns={"empty_share": "empty_shelf_share"}),
                        on="day", how="left")
    lost_value = stock.assign(v=stock["unit_price"] * stock["lost_sale"]).groupby("day")["v"].sum()
    daily["stockout_moments"] = daily["day"].map(stock.groupby("day").size()).fillna(0).astype(int)
    daily["lost_sales"] = daily["day"].map(lost_value).fillna(0.0).round(2)
    daily["waste_cost"] = daily["day"].map(waste.groupby("day")["total_cost"].sum()).fillna(0.0).round(2)
    if traffic.enabled():
        daily["expected_real_shoppers"] = [round(traffic.real_per_day(d)) for d in daily["day"]]
    daily = daily.round({"revenue": 2, "gross_profit": 2, "promo_share": 4, "empty_shelf_share": 4})
    daily["data_label"] = "SIMULATED"

    hour = (t["arrive_min"] // 60).astype(int)
    hourly = t.assign(hour=hour).groupby(["day", "hour"]).size().rename("trips").reset_index()
    hourly.insert(0, "date", book.dates[hourly["day"].to_numpy()].date)
    scale = config.TRAFFIC["npc_scale"] if traffic.enabled() else 1.0
    hourly["real_equivalent"] = (hourly["trips"] / scale).round().astype(int)
    hourly = hourly.drop(columns="day").assign(data_label="SIMULATED")

    week, prod = np.nonzero(book.on_promo)
    truth = {
        "lost_sales": stock[["date", "hour", "shopper_id", "product_id", "product_name", "department", "unit_price",
                             "on_promo", "reaction", "substitute_id", "lost_sale"]],
        "promo_effects": run["promo_log"],
        "promotions": pd.DataFrame({"week": week, "week_start": book.dates[np.minimum(week * 7, len(book.dates) - 1)],
                                    "product_id": p["product_id"].to_numpy()[prod],
                                    "promo_ratio": p["promo_ratio"].to_numpy()[prod].round(3)}),
        "waste": waste,
    }
    return {"transactions": log, "trips": trips, "products": products, "customers": customers,
            "daily_store": daily, "hourly_traffic": hourly}, truth


# ============================================================ checks
def run_checks(tables, truth, run):
    tx, trips, daily = tables["transactions"], tables["trips"], tables["daily_store"]
    checks = {
        "transactions revenue = sum of trip spend": abs(tx["line_total"].sum() - trips["spend"].sum()) < 1.0,
        "transactions revenue = sum of daily revenue": abs(tx["line_total"].sum() - daily["revenue"].sum()) < 1.0,
        "every transaction trip exists in trips": bool(tx["trip_id"].isin(trips["trip_id"]).all()),
        "every product sold is in products": bool(tx["product_id"].isin(tables["products"]["product_id"]).all()),
        "every shopper is in customers": bool(tx["shopper_id"].isin(tables["customers"]["shopper_id"]).all()),
        "line_total = quantity x unit_price": bool(np.allclose(tx["line_total"], (tx["quantity"] * tx["unit_price"]).round(2), atol=0.011)),
        "no negative prices or quantities": bool((tx["unit_price"] > 0).all() and (tx["quantity"] > 0).all()),
        "promo price <= regular price": bool((tx.loc[tx["on_promo"], "unit_price"] <= tx.loc[tx["on_promo"], "regular_price"] + 0.01).all()),
        "trip ids unique": bool(trips["trip_id"].is_unique),
    }
    return {k: bool(v) for k, v in checks.items()}


# ============================================================ writing
def write_table(df, path, csv):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    for c in df.columns:                                      # dates without a time part
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.date
    df.to_parquet(path.with_suffix(".parquet"), index=False)
    if csv:
        csvsafe.to_csv(df, path.with_suffix(".csv.gz"), index=False, compression="gzip")


def dictionary_rows(tables, truth):
    rows = []
    all_tables = {**tables, **{f"truth/{k}": v for k, v in truth.items()}}
    for name, df in all_tables.items():
        grain, cols = DICTIONARY.get(name, ("", {}))
        for c in df.columns:
            desc, label = cols.get(c, ("", "SIMULATED"))
            rows.append({"table": name, "row_is": grain, "column": c, "type": str(df[c].dtype),
                         "description": desc, "data_label": label,
                         "example": str(df[c].dropna().iloc[0])[:40] if df[c].notna().any() else ""})
    return pd.DataFrame(rows)


def dictionary_md(d, tables, truth):
    sizes = {**{k: len(v) for k, v in tables.items()}, **{f"truth/{k}": len(v) for k, v in truth.items()}}
    out = ["# RETINA data dictionary", "", "Data labels: " + "; ".join(f"**{k}** = {v}" for k, v in DATA_LABELS.items()), ""]
    for name, g in d.groupby("table", sort=False):
        out += [f"## {name}  ({sizes[name]:,} rows)", f"One row = {g['row_is'].iat[0]}.", "",
                "| column | type | description | label | example |", "|---|---|---|---|---|"]
        out += [f"| {r.column} | {r.type} | {r.description} | {r.data_label} | {r.example} |" for r in g.itertuples()]
        out.append("")
    return "\n".join(out)


def dataset_card(name, tables, truth, run, realism, checks, args):
    tx, daily = tables["transactions"], tables["daily_store"]
    money = run["money"]
    sc = run["scorecard"] or []
    passed = sum(r[4] == "PASS" for r in sc)
    judged = sum(r[4] != "N/A" for r in sc)
    first, last = tx["date"].min(), tx["date"].max()
    lines = [
        f"# {name}", "",
        "**Synthetic supermarket transactions** generated by RETINA (Retail Intelligence & Agent Simulation): "
        f"NPC shoppers with real habits shop a virtual {PROFILE['name'].lower()} stocked with real catalog products "
        "and real shelf prices. **Every row is SIMULATED** - no real customer is in this data.", "",
        "## At a glance",
        f"- Period: {first:%d %b %Y} to {last:%d %b %Y} ({len(daily)} days), seed {config.SEED}",
        f"- {tx['trip_id'].nunique():,} trips | {len(tx):,} receipt lines | {int(tx['quantity'].sum()):,} units | "
        f"${tx['line_total'].sum():,.0f} revenue | gross margin {money['margin']:.1%}",
        f"- {len(tables['customers']):,} NPC customers | {len(tables['products']):,} products on the shelves",
        f"- Store traffic follows a real footfall schedule scaled by npc_scale = "
        f"{config.TRAFFIC['npc_scale'] if traffic.enabled() else 1} (1 NPC = {1 / config.TRAFFIC['npc_scale']:.0f} real shoppers)"
        if traffic.enabled() else "- Store traffic: every NPC whose trip is due comes in",
        "", "## Tables",
        "| table | rows | one row = |", "|---|---|---|"]
    for k, v in {**tables, **{f"truth/{k}": v for k, v in truth.items()}}.items():
        lines.append(f"| {k} | {len(v):,} | {DICTIONARY.get(k, ('',))[0]} |")
    lines += ["", "Files: Parquet (and `.csv.gz` with `--csv`). Column meanings and labels: `DATA_DICTIONARY.md`.", "",
              "## How it was made",
              "- **Products and prices**: real US shelf prices (Open Prices) and product facts (Open Food Facts); "
              "prices move monthly with the real BLS CPI for each category; weekly promotions.",
              "- **Shoppers**: each NPC copies one real Instacart shopper's habits (basket size, days between trips, "
              "re-buy rate); households, income and food budgets from BLS/EIA statistics.",
              "- **Behaviour**: needs (products run out at home), habits, shopping missions (real aisle pairs), "
              "price memory and elasticity, promotions (brand switch / stock up / impulse), empty-shelf reactions "
              "(Gruen et al. 2002), budgets, walking a floor plan and queuing at checkout.",
              "- **Store**: (s, S) re-ordering with forecast errors and late suppliers, spoilage, margins.", ""]
    if config.NOT_SOLD_HERE:
        lines += ["- **Custom product range**: this store sells products UPLOADED by the user "
                  f"({'only those products' if config.CATALOG_KIND == 'only' else 'added to the US catalog'}). "
                  "Their names, prices, deals and aisles are labelled USER_PROVIDED (not checked against real stores). "
                  "NPCs still want what real shoppers want; a want in an aisle the store does not sell is recorded "
                  "as 'not sold here' (trips: not_sold_here, not_sold_aisles) - lost demand, never a substitute.", ""]
    lines += [f"## Quality: scorecard {passed}/{judged}" + (f" ({len(sc) - judged} N/A for a custom range)"
                                                         if judged < len(sc) else ""),
              "| check | type | simulated | real | result |", "|---|---|---|---|---|"]
    lines += [f"| {c[0]} | {c[1]} | {c[2]} | {c[3]} | {c[4]} |" for c in sc]
    lines += ["", "'input' = fed in (proves the code works), 'emergent' = not fed in (real evidence), "
              "'tuned' = adjusted to match (not evidence)."]
    if realism:
        c, s, t, d = realism["tell_apart"], realism["sim_to_real"], realism["bought_together"], realism["distributions"]
        lines += ["", "## Realism vs real baskets the simulator never saw (Instacart, held-out half)",
                  f"- Can a classifier tell simulated from real baskets? AUC **{c['auc']}** (0.5 = cannot, 1.0 = always) - {realism['verdict']['tell_apart']}",
                  f"- Train on RETINA, test on real: **{s['ratio']:.0%}** of the quality of training on real data - {realism['verdict']['sim_to_real']}",
                  f"- Bought-together patterns (aisle lift correlation): **{t['log_lift_corr']}** - {realism['verdict']['bought_together']}",
                  f"- Aisle mix difference {d['aisle_mix_tvd']:.1%} | repeat items {d['repeat_share']['simulated']:.0%} vs real {d['repeat_share']['real']:.0%}",
                  "- Known giveaways: " + ", ".join(f"{k} {v:+.0%}" for k, v in c["too_common_in_simulation"].items()) + " too common"]
    lines += ["", "## Consistency checks (all must pass)"] + [f"- {'PASS' if v else 'FAIL'}: {k}" for k, v in checks.items()]
    lines += ["", "## Good uses",
              "- Learning and testing: demand forecasting, basket analysis, recommendation, promotion uplift, "
              "stockout / lost-sales estimation (the truth/ folder has the answers), store dashboards, SQL practice.",
              "- Comparing methods on data where the true effects are known.", "",
              "## Limits (honest)",
              "- Shopper habits come from Instacart (online orders, 2017); a physical store may differ.",
              "- 59% of products have ESTIMATED popularity (no real order counts); the assortment mixes retailers and years.",
              "- No seasonality or holidays yet; walking and checkout times are ESTIMATED.",
              "- Same product bought again on the next trip: 13% vs real 28% (re-buy timing is too weak).",
              "- Not a forecast of any real store or retailer.", "",
              "## Licences of the inputs",
              "- Product and price facts: Open Prices / Open Food Facts - ODbL 1.0 (credit Open Food Facts; "
              "the products table is a derived database and stays ODbL).",
              "- Shopper habit statistics: Instacart Online Grocery Shopping Dataset 2017 - non-commercial use only "
              "(raw Instacart files are NOT included).",
              "- CPI, consumer expenditure, household size: U.S. BLS / EIA - public domain.",
              "- **Use this dataset for learning, research and portfolios (non-commercial).**", "",
              f"Made with RETINA {VERSION} on {datetime.now(timezone.utc):%Y-%m-%d} "
              f"(command: `python -m retina.engine.export {' '.join(sys.argv[1:])}`)."]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Export a RETINA synthetic retail dataset.")
    ap.add_argument("--seed", type=int, default=None, help="random seed (default: the store profile's)")
    ap.add_argument("--days", type=int, default=None, help="days to simulate (default: the store profile's)")
    ap.add_argument("--name", help="folder name (default retina_<store>_seed<seed>)")
    ap.add_argument("--csv", action="store_true", help="also write .csv.gz files")
    ap.add_argument("--no-realism", action="store_true", help="skip the realism test")
    args = ap.parse_args(argv)
    if args.seed is not None:
        config.set_seed(args.seed)
    from .retina_mart import run_pipeline
    run = run_pipeline(seed=args.seed, days=args.days, save=True)
    name = args.name or f"retina_{config.RUN_ID}_seed{config.SEED}" + (f"_{args.days}d" if args.days else "")
    out = EXPORT_DIR / name
    print(f"\nExporting to {out.relative_to(ROOT)} ...")
    tables, truth = build_tables(run)
    for k, v in tables.items():
        write_table(v, out / k, args.csv)
    for k, v in truth.items():
        write_table(v, out / "truth" / k, args.csv)
    realism = None
    if config.NOT_SOLD_HERE and not args.no_realism:
        print("\nRealism test: N/A for a custom product range (it compares baskets with real Instacart "
              "baskets of the full US range).")
    elif not args.no_realism:
        try:
            from . import realism as realism_mod
            print("\nRealism test (held-out real baskets)...")
            realism = realism_mod.main(final=True)
        except Exception as error:                            # e.g. Instacart files not on this PC
            print(f"Realism test skipped: {str(error)[:120]}")
    checks = run_checks(tables, truth, run)
    sc = pd.DataFrame(run["scorecard"] or [], columns=["check", "type", "simulated", "real", "result"])
    csvsafe.to_csv(sc, out / "scorecard.csv", index=False)
    if realism:
        (out / "realism.json").write_text(json.dumps(realism, indent=1, default=float), encoding="utf-8")
    (out / "checks.json").write_text(json.dumps(checks, indent=1), encoding="utf-8")
    d = dictionary_rows(tables, truth)
    csvsafe.to_csv(d, out / "data_dictionary.csv", index=False)
    (out / "DATA_DICTIONARY.md").write_text(dictionary_md(d, tables, truth), encoding="utf-8")
    (out / "DATASET_CARD.md").write_text(dataset_card(name, tables, truth, run, realism, checks, args), encoding="utf-8")
    info = {"name": name, "retina_version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seed": config.SEED, "days": len(tables["daily_store"]), "store_profile": PROFILE,
            "python": platform.python_version(), "rows": {k: len(v) for k, v in tables.items()},
            "label": "SIMULATED DATA - not real sales"}
    (out / "run_info.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"\nDataset ready: {out.relative_to(ROOT)}  ({size:.0f} MB)")
    for k, v in tables.items():
        print(f"  {k:<15} {len(v):>9,} rows")
    for k, v in truth.items():
        print(f"  truth/{k:<9} {len(v):>9,} rows")
    print("  checks: " + ("all passed" if all(checks.values()) else
                          "FAILED: " + ", ".join(k for k, v in checks.items() if not v)))
    print("  read DATASET_CARD.md first, then DATA_DICTIONARY.md")


if __name__ == "__main__":
    main()
