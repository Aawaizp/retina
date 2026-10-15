"""retina web-data: the small data files the website reads (web/public/data, web/public/model).

  retina web-data                      RETINA Supercenter, 91 days, every NPC run by the student
  retina web-data --days 7 --no-student                a quick try (engine only)

ONE store, RETINA Supercenter, run by the student (the list student plans every shopping list, the
shelf student picks every product). The website files are built from the run's company-style sales
tables (business/: sales_transactions, baskets, daily_store_kpis) and are AGGREGATED and SIMULATED:
no row per shopper. The simulation runs with RETINA_NO_PROFILES=1, exactly like a fresh clone of the
repository (and like GitHub Actions), so the shoppers never come from the private per-shopper file.

Files (Parquet, read in the browser by DuckDB-WASM), in data/retina_supercenter/:
  daily.parquet        one row per day: trips, items, revenue, profit, deal share, empty shelves, lost sales, waste
  hourly.parquet       weekday x hour: average trips
  departments.parquet  department x week: revenue, items
  top_products.parquet the 25 best sellers: revenue, units, deal share
  baskets.parquet      trip size, time in store and spend as histograms
Plus scorecard.json (the run next to real public numbers), whatif.json (example shelves for the
student model), model/ (student.onnx, student.json, MODEL_CARD.md) and manifest.json.
"""

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from retina.release import ROOT, STUDENT, git_commit, run

WEB = ROOT / "web" / "public"
STORE = "retina_supercenter"
LIST_STUDENT = ROOT / "models" / "list_student" / "list_student.onnx"
PROCESSED = ROOT / "data" / "processed"
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def histogram(values, edges, name):
    counts, _ = np.histogram(values, bins=edges)
    return pd.DataFrame({"measure": name, "low": edges[:-1], "high": edges[1:], "share": counts / max(len(values), 1)})


def store_files(business, out):
    """Aggregate a run's business tables into the website's small files."""
    kpi = pd.read_csv(business / "daily_store_kpis.csv")
    daily = pd.DataFrame({
        "date": kpi["date"], "trips": kpi["transactions"], "items": kpi["units_sold"], "revenue": kpi["net_sales"],
        "gross_profit": kpi["gross_profit"], "promo_share": (kpi["promotion_sales_share_pct"] / 100).round(4),
        "empty_shelf_share": (kpi["products_out_of_stock"] / kpi["products_carried"]).round(4),
        "stockout_moments": kpi["stockout_events"], "lost_sales": kpi["lost_sales_value"], "waste_cost": kpi["waste_cost"]})
    b = pd.read_csv(business / "baskets.csv", usecols=["date", "arrival_time", "minutes_in_store", "units", "lines",
                                                       "net_sales", "trip_type"])
    b["weekday"] = pd.to_datetime(b["date"]).dt.dayofweek
    b["hour"] = b["arrival_time"].str.slice(0, 2).astype(int)
    days_per_weekday = b.groupby("weekday")["date"].nunique()
    hourly = (b.groupby(["weekday", "hour"]).size() / days_per_weekday).rename("trips").reset_index()
    hourly["weekday_name"] = hourly["weekday"].map(dict(enumerate(WEEKDAYS)))
    hourly["trips"] = hourly["trips"].round(2)
    tx = pd.read_csv(business / "sales_transactions.csv",
                     usecols=["date", "product_id", "product_name", "brand", "category", "quantity", "net_sales",
                              "on_promotion"], dtype={"product_id": str})
    tx["week"] = pd.to_datetime(tx["date"]).dt.to_period("W-SUN").dt.start_time.dt.strftime("%Y-%m-%d")
    departments = (tx.groupby(["category", "week"]).agg(revenue=("net_sales", "sum"), items=("quantity", "sum"))
                   .round(2).reset_index().rename(columns={"category": "department"}))
    top = (tx.fillna({"brand": ""}).groupby(["product_id", "product_name", "brand", "category"])
           .agg(revenue=("net_sales", "sum"), units=("quantity", "sum"), deal_share=("on_promotion", "mean"))
           .sort_values("revenue", ascending=False).head(25).round(3).reset_index()
           .rename(columns={"category": "department"}))
    baskets = pd.concat([histogram(b["units"], np.array([0, 1, 3, 5, 8, 12, 16, 20, 30, 50, 1000]), "items"),
                         histogram(b["minutes_in_store"], np.array([0, 5, 10, 15, 20, 30, 45, 60, 1000]), "minutes"),
                         histogram(b["net_sales"], np.array([0, 10, 25, 50, 75, 100, 150, 200, 1e6]), "spend")])
    out.mkdir(parents=True, exist_ok=True)
    for name, df in (("daily", daily), ("hourly", hourly), ("departments", departments), ("top_products", top),
                     ("baskets", baskets)):
        df.assign(data_label="SIMULATED").to_parquet(out / f"{name}.parquet", index=False)
    return {"days": len(daily), "trips": int(daily["trips"].sum()), "revenue": round(float(daily["revenue"].sum()), 2),
            "items": int(daily["items"].sum()), "deal_share": round(float(tx["on_promotion"].mean()), 4),
            "avg_basket": round(float(b["net_sales"].mean()), 2),
            "minutes_in_store": round(float(b["minutes_in_store"].mean()), 1)}


def real_numbers():
    """Public benchmarks from data/processed (aggregates only); {} where a file is missing (fresh clone)."""
    real = {}
    aisles = PROCESSED / "instacart_aisle_stats.csv"
    if aisles.exists():
        a = pd.read_csv(aisles)
        real["lines_per_basket"] = round(float(a["lines"].sum() / a["orders_total"].iloc[0]), 2)
        real["repeat_share"] = round(float(a["rebought"].sum() / a["lines"].sum()), 3)
        real["departments"] = (a.groupby("department")["lines"].sum() / a["lines"].sum()).round(4).to_dict()
    trips = PROCESSED / "instacart_trip_stats.json"
    if trips.exists():
        real["trip_types"] = json.loads(trips.read_text(encoding="utf-8"))["trip_type_shares"]
    shop = PROCESSED / "real_shopping_stats.json"
    if shop.exists():
        real["units_per_line"] = json.loads(shop.read_text(encoding="utf-8"))["units_per_receipt_line"]["usd"]["mean"]
    return real


def scorecard(run_dir, days):
    """The student-run store next to real public numbers (shoppers from the second half of the run, when
    everyone has a shopping history - day 1 has no re-buys by definition)."""
    business = run_dir / "business"
    s = pd.read_csv(business / "sales_transactions.csv", usecols=["date", "category", "quantity", "repeat_purchase",
                                                                  "on_promotion", "decided_by"])
    b = pd.read_csv(business / "baskets.csv")
    late = s[pd.to_datetime(s["date"]) >= pd.to_datetime(s["date"]).min() + pd.Timedelta(days=days // 2)]
    real, price = real_numbers(), {}
    if (run_dir / "report.json").exists():
        price = json.loads((run_dir / "report.json").read_text(encoding="utf-8")).get("price") or {}
    mix = (s.groupby("category").size() / len(s)).round(4)
    trip = b["trip_type"].value_counts(normalize=True).round(3).to_dict()
    rows = [
        {"measure": "Products per basket", "ours": round(float(b["lines"].mean()), 2), "real": real.get("lines_per_basket"),
         "source": "Instacart orders (online grocery, 2017)", "label": "PUBLIC_STATISTIC"},
        {"measure": "Re-buy share (product bought before)", "ours": round(float(late["repeat_purchase"].mean()), 3),
         "real": real.get("repeat_share"), "source": "Instacart reorder rate", "label": "PUBLIC_STATISTIC",
         "format": "pct"},
        {"measure": "Units per receipt line", "ours": round(float(s["quantity"].mean()), 2), "real": real.get("units_per_line"),
         "source": "Open Prices receipts (US)", "label": "REAL_OBSERVED"},
        {"measure": "Units bought on promotion", "ours": round(float((s["quantity"] * s["on_promotion"]).sum()
                                                                    / s["quantity"].sum()), 3),
         "real": "24-31%", "source": "Nielsen 2019-2020, Bernstein 2022", "label": "PUBLIC_STATISTIC", "format": "pct"},
        {"measure": "Own-price elasticity", "ours": price.get("llm_elasticity"), "real": -2.5,
         "source": "grocery research (about -2.5)", "label": "PUBLIC_STATISTIC"},
    ]
    for kind in ("routine", "stock_up", "top_up"):
        rows.append({"measure": f"Trips: {kind.replace('_', '-')}", "ours": trip.get(kind),
                     "real": (real.get("trip_types") or {}).get(kind), "source": "Instacart, basket vs own average",
                     "label": "PUBLIC_STATISTIC", "format": "pct"})
    for dept in ("produce", "dairy eggs", "snacks", "beverages", "frozen", "household", "personal care", "babies"):
        rows.append({"measure": f"Lines: {dept}", "ours": float(mix.get(dept, 0.0)),
                     "real": (real.get("departments") or {}).get(dept), "source": "Instacart department shares",
                     "label": "PUBLIC_STATISTIC", "format": "pct"})
    return {"data_label": "SIMULATED", "days": days, "store": "RETINA Supercenter",
            "baskets": len(b), "lines": len(s),
            "decided_by_student": round(float((s["decided_by"] == "student model").mean()), 3),
            "lists_by_list_student": round(float((b.get("list_planned_by") == "list student").mean()), 3)
            if "list_planned_by" in b else 0.0,
            "rows": rows,
            "notes": ["Re-buy share uses the second half of the run (on day 1 nobody has bought anything yet).",
                      "Household and personal care are a little low: a known limit (the list student has no "
                      "department input; the gap is about 1 point of sales)."]}


def whatif_shelves(folder, n_aisles=8, per_aisle=5):
    """Example shelves for the what-if page: popular aisles, their 5 most popular products."""
    p = pd.read_parquet(folder / "products.parquet")
    p = p[p["regular_price"] > 0].copy()
    sizes = p.groupby("aisle")["popularity"].sum().sort_values(ascending=False)
    shelves = []
    for aisle in sizes.index:
        top = p[p["aisle"] == aisle].sort_values("popularity", ascending=False).head(per_aisle)
        if len(top) < per_aisle:
            continue
        total = float(sizes[aisle]) or 1.0                  # the model's "pop" = share of the aisle
        shelves.append({"aisle": aisle, "department": str(top["department"].iloc[0]),
                        "products": [{"id": str(r.product_id), "name": str(r.product_name), "brand": str(r.brand or ""),
                                      "regular": round(float(r.regular_price), 2), "pop": round(float(r.popularity) / total, 6)}
                                     for r in top.itertuples()]})
        if len(shelves) == n_aisles:
            break
    return shelves


def web_data(days=91, out=WEB, student=True, say=print):
    out = Path(out)
    data = out / "data"
    student = student and STUDENT.exists() and LIST_STUDENT.exists()
    mind = "student" if student else "fast"
    manifest = {"built": datetime.now(timezone.utc).isoformat(timespec="seconds"), "commit": git_commit(),
                "days": days, "data_label": "SIMULATED", "mind": mind, "stores": {},
                "note": "one store run by the student model; aggregates only; built without the private "
                        "Instacart per-shopper profiles (like a fresh clone)"}
    with tempfile.TemporaryDirectory(prefix="retina_web_") as tmp:
        tmp = Path(tmp)
        env = dict(os.environ, RETINA_EXPORT_DIR=str(tmp / "export"), RETINA_LIVE_ROOT=str(tmp / "live"),
                   RETINA_CACHE_DIR=str(tmp / "cache"), RETINA_NO_PROFILES="1", PYTHONIOENCODING="utf-8")
        say(f"Simulating RETINA Supercenter ({days} days, mind: {mind})...")
        run(["--store", STORE, "--mind", mind, "--days", str(days), "--reset", "--yes"], env, say)
        run_dir = tmp / "live" / (STORE if mind == "fast" else f"{STORE}_{mind}")
        for old in data.iterdir() if data.exists() else []:     # the older multi-store files
            if old.is_dir():
                shutil.rmtree(old)
            elif old.name == "student.json":
                old.unlink()
        manifest["stores"][STORE] = store_files(run_dir / "business", data / STORE)
        if student:
            (data / "scorecard.json").write_text(json.dumps(scorecard(run_dir, days), indent=1), encoding="utf-8")
            model = out / "model"
            model.mkdir(parents=True, exist_ok=True)
            for f in ("student.onnx", "student.json", "MODEL_CARD.md"):
                shutil.copy2(STUDENT.parent / f, model / f)
        say("Example shelves for the what-if page (1 day)...")
        run(["export", "--store", STORE, "--days", "1", "--name", "web_shelves", "--no-realism"], env, say)
        (data / "whatif.json").write_text(json.dumps({"data_label": "SIMULATED",
                                                      "shelves": whatif_shelves(tmp / "export" / "web_shelves")},
                                                     indent=1), encoding="utf-8")
    (data / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file() and ("data" in f.parts or "model" in f.parts))
    say(f"Website data ready: {out} ({size / 1e6:.1f} MB)")
    return manifest
