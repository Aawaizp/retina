"""Your own products (v2 Phase 3): reading, checking, converting, safe CSVs and a store that sells
only uploaded products. Nothing is written into data/ - temp folders only."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from retina.engine import csvsafe, reports
from retina.products import aisles as A, reader
from retina.products.check import check_file
from retina.products.convert import to_frame
from retina.products.packsize import parse_pack

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "products_200.csv"
HEADER = "name,brand,category,price,pack_size\n"


def write(tmp_path, text, name="p.csv", encoding="utf-8"):
    path = tmp_path / name
    path.write_bytes(text.encode(encoding))
    return path


# ------------------------------------------------------------------ reading
def test_reads_bom_quotes_commas_and_semicolons(tmp_path):
    p = write(tmp_path, '﻿' + HEADER + '"Cookies, Chocolate Chip",Acme,cookies cakes,3.99,13 oz\n'
              '"12"" Pizza",Acme,frozen pizza,6.49,1 ct\n')
    _, rows, info = reader.read_products(p)
    assert info["encoding"] == "UTF-8" and [r["name"] for _, r in rows] == ["Cookies, Chocolate Chip", '12" Pizza']
    p = write(tmp_path, "Name;Brand;Category;Price;Pack Size\nCafé Latte;Acme;coffee;4.99;250 g\n", "w.csv", "cp1252")
    columns, rows, info = reader.read_products(p)
    assert info["encoding"] == "Windows-1252" and info["separator"] == "semicolon"
    assert rows[0][1]["name"] == "Café Latte" and "pack_size" in columns


def test_row_limit_and_wrong_files(tmp_path, monkeypatch):
    monkeypatch.setattr(reader, "MAX_ROWS", 3)
    p = write(tmp_path, HEADER + "".join(f"P{i},B,milk,1.00,1 l\n" for i in range(4)))
    with pytest.raises(reader.UploadError, match="More than 3"):
        reader.read_products(p)
    with pytest.raises(reader.UploadError, match="CSV UTF-8"):
        reader.read_products(write(tmp_path, "x", "p.xlsx"))
    with pytest.raises(reader.UploadError, match="empty"):
        reader.read_products(write(tmp_path, "  \n", "e.csv"))


def test_pack_sizes():
    cases = {"500 g": "500 g", "500grams": "500 g", "12 x 330 ml": "12 x 330 ml", "12x330ml": "12 x 330 ml",
             "1,5 L": "1.5 l", "16 fl. oz": "16 fl oz", "pack of 6": "6 ct", "1 dozen": "12 ct",
             "1 lb 11.2 oz": "27.2 oz", "7 oz, 198 g": "7 oz", "40 pk. 16.9oz": "40 x 16.9 oz"}
    for text, tidy in cases.items():
        assert parse_pack(text)["text"] == tidy, text
    assert parse_pack("big bag") is None


# ------------------------------------------------------------------ checking
def test_did_you_mean_is_only_a_suggestion(tmp_path):
    p = write(tmp_path, HEADER + "Digestive Biscuits,Acme,biscuits,2.49,400 g\n")
    c = check_file(p)
    assert not c.ok and "did you mean 'cookies cakes'" in c.errors[0][1] and not c.products
    p = write(tmp_path, "name,brand,category,aisle,price,pack_size\nDigestive Biscuits,Acme,biscuits,Cookies & Cakes,2.49,400 g\n")
    c = check_file(p)                                              # the uploader confirms with an aisle column
    assert c.ok and c.products[0]["aisle"] == "cookies cakes" and c.products[0]["department"] == "snacks"
    c = check_file(write(tmp_path, HEADER + "Potato Chips,Acme,snacks,2.49,8 oz\n"))
    assert "is a whole department" in c.errors[0][1] and "chips pretzels" in c.errors[0][1]


def test_prices_usd_only_and_row_problems(tmp_path):
    p = write(tmp_path, "name,brand,category,price,pack_size,promo_price\n"
              "Milk,Acme,milk,₹60,1 l,\n"           # rupees
              "Milk 2,Acme,milk,\"3,49\",1 l,\n"     # decimal comma
              ",Acme,milk,3.49,1 l,\n"              # no name
              "Milk 3,Acme,milk,3.49,,\n"           # no pack size
              "Milk 4,Acme,milk,$1,299.00,1 l,0.99\n"   # unquoted comma shifts cells -> price '$1' + warning
              "Yogurt,Acme,yogurt,5.00,32 oz,1.00\n"  # 80% off -> capped at 50%
              "Yogurt,Acme,yogurt,5.00,32 oz,\n")   # same product again
    c = check_file(p)
    rows = {r: m for r, m in c.errors}
    assert "not in US dollars" in rows[2] and "comma for decimals" in rows[3]
    assert "name is empty" in rows[4] and "pack_size is empty" in rows[5]
    warns = " | ".join(m for _, m in c.warnings)
    assert "capped at 50%" in warns and "same product as row 7" in warns and "more cells than the header" in warns
    c = check_file(p, skip_bad_rows=True)
    assert c.ok and [x["name"] for x in c.products] == ["Milk 4", "Yogurt"]
    assert c.products[1]["promo_price"] == 2.5


def test_missing_columns(tmp_path):
    c = check_file(write(tmp_path, "name,price\nMilk,3.49\n"))
    assert not c.ok and "missing column(s): brand, pack_size, category (or aisle)" in c.errors[0][1]


def test_file_problems_give_a_useful_hint(tmp_path):
    from retina.products.check import report_text
    text = report_text(check_file(tmp_path / "my_products.csv"))
    assert "retina products --template" in text and "--skip-bad-rows" not in text
    text = report_text(check_file(write(tmp_path, HEADER + "Milk,Acme,milk,abc,1 l\n")))
    assert "--skip-bad-rows" in text                               # a bad ROW: skipping it is an option


# ------------------------------------------------------------- safe CSVs
def test_formula_cells_are_neutralised(tmp_path):
    df = pd.DataFrame({"name": ["=HYPERLINK(\"x\")", "+1 Cola", "-5 Gum", "@cmd", "Milk"], "price": [-1.5, 1, 2, 3, 4]})
    path = tmp_path / "out.csv"
    csvsafe.to_csv(df, path, index=False)
    back = pd.read_csv(path)
    assert list(back["name"]) == ["'=HYPERLINK(\"x\")", "'+1 Cola", "'-5 Gum", "'@cmd", "Milk"]
    assert back["price"].iloc[0] == -1.5                           # numbers are never touched


# --------------------------------------------------- scorecard for custom stores
def test_scorecard_mix_checks_are_not_applicable(pipeline, monkeypatch):
    r = pipeline["results"]
    args = (pipeline["log"], r.params, r.store, r.stock_stats, pipeline["stockouts"], pipeline["money"],
            pipeline["promo_log"], reports.pair_report(r.sales, r.store))
    normal = reports.scorecard(*args)
    monkeypatch.setattr(reports, "NOT_SOLD_HERE", True)
    custom = reports.scorecard(*args)
    na = [row[0] for row in custom if row[4] == "N/A"]
    assert "department shares (largest gap)" in na and "real pairs bought together (lift corr.)" in na
    assert not any(row[4] == "N/A" for row in normal) and len(custom) == len(normal)


# -------------------------------------------------------- end to end (200 products)
RUN = """
import json
from retina.engine.engine import Simulation
sim = Simulation(verbose=False)
for _ in range(DAYS): sim.run_day()
r = sim.results()
p = sim.store.products
sold = p["product_id"].to_numpy()[r.sales["product_idx"].to_numpy()]
sellable = set(int(i) for i in sim.inv.sellable.nonzero()[0])
history = {q for c in sim.shoppers.counts for q in c}
print(json.dumps({"lines": len(r.sales), "uploaded_only": bool(all(str(x).startswith("U-") for x in sold)),
                  "not_sold": int(r.trips["not_sold_here"].sum()) if "not_sold_here" in r.trips else -1,
                  "history_ok": history <= sellable, "carried": len(sellable), "products": len(p),
                  "aisles_sold": sorted(set(p["aisle"].to_numpy()[r.sales["product_idx"].to_numpy()]))}))
"""


@pytest.fixture(scope="module")
def uploaded(tmp_path_factory):
    folder = tmp_path_factory.mktemp("upload")
    check = check_file(FIXTURE)
    assert check.ok and len(check.products) == 200
    csvsafe.to_csv(to_frame(check.products, FIXTURE.name), folder / "catalog.csv", index=False)
    return folder / "catalog.csv", {p["aisle"] for p in check.products}


def run_store(catalog, store, kind="only", days=2, strict=False):
    """strict: any RuntimeWarning (e.g. an average of nothing) fails the run."""
    env = dict(os.environ, RETINA_STORE=store, RETINA_CATALOG=str(catalog), RETINA_CATALOG_KIND=kind,
               RETINA_CATALOG_TAG="test")
    flags = ["-W", "error::RuntimeWarning"] if strict else []
    out = subprocess.run([sys.executable, *flags, "-c", RUN.replace("DAYS", str(days))], cwd=ROOT, env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[-1500:]
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("store", ["generic_supermarket", "supercenter", "neighborhood_market", "convenience"])
def test_200_uploaded_products_run_end_to_end(uploaded, store):
    catalog, aisles = uploaded
    r = run_store(catalog, store)
    assert r["lines"] > 0 and r["uploaded_only"]
    assert r["products"] == 200 if store in ("generic_supermarket", "supercenter") else 0 < r["products"] < 200
    #                          (small formats keep only each aisle's best sellers: 35% / 8% of a range)
    assert set(r["aisles_sold"]) <= aisles                           # nothing from an aisle it does not sell
    assert r["not_sold"] > 0                                         # wants elsewhere: lost demand, recorded
    assert r["history_ok"]                                           # NPC history only from what it sells


def test_tiny_store_runs_without_warnings(tmp_path):
    """The 6-product template: days without any deal used to print 'Mean of empty slice'."""
    from retina.products.convert import write_template
    example, _ = write_template(tmp_path)
    check = check_file(example)
    csvsafe.to_csv(to_frame(check.products, example.name), tmp_path / "catalog.csv", index=False)
    r = run_store(tmp_path / "catalog.csv", "generic_supermarket", days=3, strict=True)
    assert r["products"] == 6 and r["lines"] > 0 and r["uploaded_only"]


def test_add_mode_keeps_the_us_catalog(uploaded, tmp_path):
    catalog, _ = uploaded
    us = pd.read_csv(A.US_CATALOG, dtype={"product_id": str}, low_memory=False)
    both = pd.concat([us, pd.read_csv(catalog, dtype={"product_id": str})], ignore_index=True)
    both.to_csv(tmp_path / "both.csv", index=False)
    r = run_store(tmp_path / "both.csv", "generic_supermarket", kind="add", days=1)
    assert r["products"] == len(us) + 200 and r["lines"] > 0 and r["history_ok"]
