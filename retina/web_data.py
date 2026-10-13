"""retina web-data: the small data files the website reads (web/public/data, web/public/model).

  retina web-data                      4 store formats x 90 days + the student panel + the what-if model
  retina web-data --days 14 --stores convenience        a quick try

Everything is AGGREGATED and SIMULATED: no row per shopper, nothing from Instacart per shopper. The
simulation runs with RETINA_NO_PROFILES=1, exactly like a fresh clone of the repository (and like
GitHub Actions), so even the simulated shoppers never come from the private per-shopper file.

Files per store (Parquet, read in the browser by DuckDB-WASM):
  daily.parquet        one row per day: trips, items, revenue, profit, deal share, empty shelves, lost sales, waste
  hourly.parquet       weekday x hour: average trips
  departments.parquet  department x week: revenue, items
  top_products.parquet the 25 best sellers: revenue, units, deal share
  baskets.parquet      trip size and time in store as histograms
Plus student.json (thinking panel vs engine crowd), whatif.json (example shelves for the student
model), model/ (student.onnx, student.json, MODEL_CARD.md) and manifest.json (what was built, how).
"""

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from retina.release import ROOT, STORES, STUDENT, git_commit, run

WEB = ROOT / "web" / "public"
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def histogram(values, edges, name):
    counts, _ = np.histogram(values, bins=edges)
    return pd.DataFrame({"measure": name, "low": edges[:-1], "high": edges[1:], "share": counts / max(len(values), 1)})


def store_files(folder, out):
    """Aggregate one exported store (Parquet tables) into the website's small files."""
    daily = pd.read_parquet(folder / "daily_store.parquet")
    keep = [c for c in ("date", "trips", "items", "revenue", "gross_profit", "promo_share", "empty_shelf_share",
                        "stockout_moments", "lost_sales", "waste_cost") if c in daily]
    daily = daily[keep].assign(date=lambda d: pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d"))
    hourly = pd.read_parquet(folder / "hourly_traffic.parquet")
    hourly["weekday"] = pd.to_datetime(hourly["date"]).dt.dayofweek
    days_per_weekday = hourly.groupby("weekday")["date"].nunique()
    hourly = (hourly.groupby(["weekday", "hour"])["trips"].sum() / days_per_weekday).rename("trips").reset_index()
    hourly["weekday_name"] = hourly["weekday"].map(dict(enumerate(WEEKDAYS)))
    hourly["trips"] = hourly["trips"].round(2)
    tx = pd.read_parquet(folder / "transactions.parquet",
                         columns=["date", "product_id", "product_name", "brand", "department", "quantity",
                                  "line_total", "on_promo"])
    tx["week"] = pd.to_datetime(tx["date"]).dt.to_period("W-SUN").dt.start_time.dt.strftime("%Y-%m-%d")
    departments = (tx.groupby(["department", "week"]).agg(revenue=("line_total", "sum"), items=("quantity", "sum"))
                   .round(2).reset_index())
    top = (tx.groupby(["product_id", "product_name", "brand", "department"])
           .agg(revenue=("line_total", "sum"), units=("quantity", "sum"), deal_share=("on_promo", "mean"))
           .sort_values("revenue", ascending=False).head(25).round(3).reset_index())
    trips = pd.read_parquet(folder / "trips.parquet", columns=["items", "minutes_in_store", "spend"])
    baskets = pd.concat([histogram(trips["items"], np.array([0, 1, 3, 5, 8, 12, 16, 20, 30, 50, 1000]), "items"),
                         histogram(trips["minutes_in_store"], np.array([0, 5, 10, 15, 20, 30, 45, 60, 1000]), "minutes"),
                         histogram(trips["spend"], np.array([0, 10, 25, 50, 75, 100, 150, 200, 1e6]), "spend")])
    out.mkdir(parents=True, exist_ok=True)
    for name, df in (("daily", daily), ("hourly", hourly), ("departments", departments), ("top_products", top),
                     ("baskets", baskets)):
        df.assign(data_label="SIMULATED").to_parquet(out / f"{name}.parquet", index=False)
    return {"days": len(daily), "trips": int(daily["trips"].sum()), "revenue": round(float(daily["revenue"].sum()), 2),
            "items": int(daily["items"].sum()), "deal_share": round(float(tx["on_promo"].mean()), 4),
            "avg_basket": round(float(trips["spend"].mean()), 2), "minutes_in_store": round(float(trips["minutes_in_store"].mean()), 1)}


def student_summary(run_dir):
    """Thinking panel (student) vs engine crowd from a student-only mix run."""
    r = pd.read_csv(run_dir / "receipts.csv", usecols=["trip_id", "shopper_id", "quantity", "line_total", "on_promo",
                                                        "is_repeat", "mind"])
    panel = set(r.loc[r["mind"] != "engine", "shopper_id"])
    r["group"] = np.where(r["shopper_id"].isin(panel), "thinking panel (student)", "engine crowd")
    groups = {}
    for name, g in r.groupby("group"):
        trips = g["trip_id"].nunique()
        groups[name] = {"trips": int(trips), "lines": len(g), "repeat_share": round(float(g["is_repeat"].mean()), 3),
                        "units_on_deal": round(float((g["quantity"] * g["on_promo"]).sum() / g["quantity"].sum()), 3),
                        "items_per_trip": round(float(g["quantity"].sum() / trips), 1),
                        "spend_per_trip": round(float(g["line_total"].sum() / trips), 2)}
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8")) if (run_dir / "report.json").exists() else {}
    d = pd.read_csv(run_dir / "decisions.csv", usecols=["mind"])
    return {"groups": groups, "price": report.get("price"), "minds": d["mind"].value_counts().to_dict(),
            "real": {"repeat_share": "~60% (Instacart reorder rate, PUBLIC_STATISTIC)",
                     "units_on_deal": "~24-31% (Nielsen 2019-2020, Bernstein 2022; PUBLIC_STATISTIC)",
                     "elasticity": "about -2.5 (grocery research, PUBLIC_STATISTIC)"}}


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
                        "products": [{"id": int(r.product_id), "name": str(r.product_name), "brand": str(r.brand or ""),
                                      "regular": round(float(r.regular_price), 2), "pop": round(float(r.popularity) / total, 6)}
                                     for r in top.itertuples()]})
        if len(shelves) == n_aisles:
            break
    return shelves


def web_data(days=90, stores=None, out=WEB, student=True, say=print):
    stores = stores or STORES
    out = Path(out)
    data = out / "data"
    manifest = {"built": datetime.now(timezone.utc).isoformat(timespec="seconds"), "commit": git_commit(),
                "days": days, "data_label": "SIMULATED", "stores": {},
                "note": "aggregates only; built without the private Instacart per-shopper profiles (like a fresh clone)"}
    with tempfile.TemporaryDirectory(prefix="retina_web_") as tmp:
        tmp = Path(tmp)
        env = dict(os.environ, RETINA_EXPORT_DIR=str(tmp / "export"), RETINA_LIVE_ROOT=str(tmp / "live"),
                   RETINA_CACHE_DIR=str(tmp / "cache"), RETINA_NO_PROFILES="1", PYTHONIOENCODING="utf-8")
        for store in stores:
            say(f"Simulating {store} ({days} days)...")
            name = f"web_{store}"
            run(["export", "--store", store, "--days", str(days), "--name", name, "--no-realism"], env, say)
            folder = tmp / "export" / name
            manifest["stores"][store] = store_files(folder, data / store)
            if store == "generic_supermarket":
                (data / "whatif.json").write_text(json.dumps({"data_label": "SIMULATED", "shelves": whatif_shelves(folder)},
                                                             indent=1), encoding="utf-8")
        if student and STUDENT.exists():
            say(f"Student panel: supermarket, {days} days, no LLM...")
            run(["--store", "supermarket", "--mind", "mix", "--student", "only", "--days", str(days), "--reset",
                 "--yes"], env, say)
            summary = student_summary(tmp / "live" / "generic_supermarket_mix")
            (data / "student.json").write_text(json.dumps({"days": days, "data_label": "SIMULATED", **summary},
                                                          indent=1), encoding="utf-8")
            model = out / "model"
            model.mkdir(parents=True, exist_ok=True)
            for f in ("student.onnx", "student.json", "MODEL_CARD.md"):
                shutil.copy2(STUDENT.parent / f, model / f)
            manifest["student"] = True
    (data / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file() and ("data" in f.parts or "model" in f.parts))
    say(f"Website data ready: {out} ({size / 1e6:.1f} MB)")
    return manifest
