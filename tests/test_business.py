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
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
IDS = {"product_id": str, "transaction_id": str}
TABLES = ["sales_transactions", "baskets", "daily_product_sales", "inventory_daily", "daily_store_kpis",
          "products", "customers", "promotions"]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    live = tmp_path_factory.mktemp("live")
    env = dict(os.environ, RETINA_LIVE_ROOT=str(live), RETINA_CACHE_DIR=str(live / "cache"), PYTHONIOENCODING="utf-8")
    out = subprocess.run([sys.executable, "-m", "retina", "--store", "convenience", "--days", "3"], cwd=ROOT, env=env,
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
    s = t["sales_transactions"]
    assert set(s["decided_by"]) == {"statistical engine"}             # a fast run: no thinking NPCs
    assert set(s["purchase_reason"]) <= {"ran out at home", "regular buy", "new item on the list",
                                         "goes with another item", "impulse (saw a deal on the shelf)"}


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
    c = TestClient(server.app)
    listed = next(r for r in c.get("/runs").json() if r["name"] == "convenience")
    assert listed["business"][:2] == ["sales_transactions.csv", "baskets.csv"]
    assert c.get("/runs/convenience/business/sales_transactions.csv").text.startswith("transaction_id,")
    assert c.get("/runs/convenience/business/state.pkl").status_code == 404
    with zipfile.ZipFile(io.BytesIO(c.get("/runs/convenience/zip").content)) as z:
        names = z.namelist()
    assert "convenience/sales_transactions.csv" in names and "convenience/DATA_DICTIONARY.md" in names
    assert "convenience/extra/receipts.csv" in names
