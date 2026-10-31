"""Business-ready sales data (retina/engine/business.py): every run ends with tables a retailer's
analysts use, and they agree with each other to the cent."""

import io
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from fastapi import HTTPException

ROOT = Path(__file__).resolve().parents[1]
IDS = {"product_id": str, "transaction_id": str}
TABLES = ["sales_transactions", "baskets", "daily_product_sales", "inventory_daily", "daily_store_kpis",
          "products", "customers", "promotions"]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    live = tmp_path_factory.mktemp("live")
    env = dict(os.environ, RETINA_LIVE_ROOT=str(live), RETINA_CACHE_DIR=str(live / "cache"), PYTHONIOENCODING="utf-8")
    out = subprocess.run([sys.executable, "-m", "retina", "--store", "convenience", "--days", "3", "--mind", "fast"], cwd=ROOT, env=env,
                         capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert out.returncode == 0, out.stdout[-800:] + out.stderr[-800:]
    folder = live / "convenience" / "business"
    return live, {t: pd.read_csv(folder / f"{t}.csv", dtype=IDS) for t in TABLES}, folder


def test_every_table_is_there_and_labelled(run):
    _, t, folder = run
    for name, df in t.items():
        assert len(df) > 0, name
        assert (df["data_label"] == "SIMULATED").all(), name
    assert (folder / "DATA_DICTIONARY.md").exists() and "SIMULATED" in (folder / "DATASET_CARD.md").read_text(encoding="utf-8")
    assert len(t["daily_store_kpis"]) == 3
    assert (t["sales_transactions"]["sample_weight"] == 10).all()
    kpi = t["daily_store_kpis"]
    assert (kpi["estimated_store_net_sales"] == (kpi["net_sales"] * 10).round(2)).all()
    assert (kpi["sample_weight"] == 10).all()
    assert (kpi["population_scope"] == "SIMULATED_NPC_SAMPLE").all()
    assert (kpi["stock_scope"] == "SIMULATED_NPC_SAMPLE").all()
    assert (kpi["estimate_status"] == "EXPANDED_NPC_SAMPLE").all()
    s = t["sales_transactions"]
    assert set(s["decided_by"]) == {"statistical engine"}             # a fast run: no thinking NPCs
    assert set(s["purchase_reason"]) <= {"ran out at home", "regular buy", "new item on the list",
                                         "goes with another item", "impulse (saw a deal on the shelf)"}


def test_daily_sales_is_a_complete_carried_sku_day_panel(run):
    _, tables, folder = run
    daily, products, kpi = tables["daily_product_sales"], tables["products"], tables["daily_store_kpis"]
    assert len(daily) == len(products) * len(kpi)
    assert not daily.duplicated(["date", "product_id"]).any()
    assert set(daily["date"]) == set(kpi["date"])
    assert set(daily["product_id"]) == set(products["product_id"])
    zero = daily[daily["units_sold"] == 0]
    assert len(zero) > 0 and (zero["net_sales"] == 0).all()
    assert zero["avg_unit_price"].isna().all()
    assert (daily["regular_price"] > 0).all() and (daily["shelf_price"] > 0).all()
    assert daily["closing_stock"].notna().all()
    assert (daily["out_of_stock"] == (daily["closing_stock"] <= 0)).all()
    assert (daily["sample_weight"] == 10).all()
    assert (daily["stock_scope"] == "SIMULATED_NPC_SAMPLE").all()
    inventory = tables["inventory_daily"]
    assert (inventory["stock_scope"] == "SIMULATED_NPC_SAMPLE").all()
    assert (inventory["sample_weight"] == 10).all()
    doc = (folder / "DATASET_CARD.md").read_text(encoding="utf-8")
    assert "Stock units, waste and value" in doc and "including zero sales" in doc


def test_tables_agree_to_the_cent(run):
    _, t, _ = run
    net = round(t["sales_transactions"]["net_sales"].sum(), 2)
    for name, col in (("baskets", "net_sales"), ("daily_product_sales", "net_sales"),
                      ("daily_store_kpis", "net_sales"), ("customers", "total_spend")):
        assert abs(t[name][col].sum() - net) < 0.05, name
    units = t["sales_transactions"]["quantity"].sum()
    assert t["daily_product_sales"]["units_sold"].sum() == units == t["inventory_daily"]["sold_units"].sum()
    s = t["sales_transactions"]
    assert ((s["net_sales"] - s["cost_of_goods"] - s["gross_profit"]).abs() <= 0.011).all()
    assert (s["discount_amount"] >= 0).all() and (s["net_sales"] <= s["gross_sales"] + 0.01).all()


def test_stock_balances_every_day(run):
    _, t, _ = run
    i = t["inventory_daily"]
    balance = i["opening_stock"] - i["wasted_units"] + i["delivered_units"] - i["sold_units"] - i["closing_stock"]
    assert (balance == 0).mean() > 0.999          # (a product delisted that day is the only exception)
    assert (i["closing_stock"] >= 0).all()
    margin = t["daily_store_kpis"]["gross_margin_pct"]
    assert ((margin > 5) & (margin < 50)).all()
    k = t["daily_store_kpis"]
    assert (k["stock_value_at_cost"] > 1000).all() and (k["products_out_of_stock"] <= k["products_carried"]).all()


def test_app_offers_the_business_tables_first(run, monkeypatch):
    live, _, _ = run
    monkeypatch.setenv("RETINA_LIVE_ROOT", str(live))
    from retina.app import server
    listed = next(r for r in server.runs() if r["name"] == "convenience")
    assert listed["business"][:2] == ["sales_transactions.csv", "baskets.csv"]
    assert Path(server.download_business("convenience", "sales_transactions.csv").path).read_text(
        encoding="utf-8").startswith("transaction_id,")
    with pytest.raises(HTTPException) as error:
        server.download_business("convenience", "state.pkl")
    assert error.value.status_code == 404
    monkeypatch.setattr(server, "StreamingResponse", lambda buffer, **kwargs: buffer.getvalue())
    with zipfile.ZipFile(io.BytesIO(server.download_zip("convenience"))) as z:
        names = z.namelist()
    assert "convenience/sales_transactions.csv" in names and "convenience/DATA_DICTIONARY.md" in names
    assert "convenience/VALIDATION.md" in names and "convenience/VALIDATION.json" in names
    assert "convenience/extra/receipts.csv" in names


def test_capped_realtime_day_has_no_expanded_estimate_after_start_time(tmp_path):
    live = tmp_path / "live"
    env = dict(os.environ, RETINA_LIVE_ROOT=str(live), RETINA_CACHE_DIR=str(tmp_path / "cache"),
               PYTHONIOENCODING="utf-8")
    out = subprocess.run([sys.executable, "-m", "retina", "--store", "convenience", "--days", "1",
                          "--mind", "fast", "--realtime", "--npc-limit", "20", "--start-time", "10:00",
                          "--seed", "42", "--yes"], cwd=ROOT, env=env, capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=120)
    assert out.returncode == 0, out.stdout[-800:] + out.stderr[-800:]
    folder = live / "convenience"
    trips = pd.read_csv(folder / "trips.csv")
    assert (trips["arrive_min"] >= 10 * 60).all()
    kpi = pd.read_csv(folder / "business" / "daily_store_kpis.csv")
    assert kpi["estimate_status"].iloc[0] == "ARRIVAL_CAP_REACHED"
    assert pd.isna(kpi["estimated_store_transactions"].iloc[0])
    assert pd.isna(kpi["estimated_store_net_sales"].iloc[0])
    import json
    report = json.loads((folder / "business" / "VALIDATION.json").read_text(encoding="utf-8"))
    assert report["integrity"]["status"] == "pass"
    assert report["sampling"]["arrival_capped_days"] == 1
