"""Checked products -> the engine's catalog format, saved in data/uploads/<tag>/.

  catalog.csv        same columns as the US catalog, every value labelled:
                     USER_PROVIDED = from the upload (name, brand, price, deal, aisle, pack size,
                     popularity if given); ESTIMATED = our defaults (popularity = the aisle's US
                     median when not given, CPI category from the aisle)
  check_report.txt   the full check (every warning)
  upload.json        where it came from (file name, fingerprint, mode, counts)
mode "only": the store sells just these products; "add": the US catalog plus these.
The tag (folder name) = file name + a fingerprint of its content (+ "-add"), so an edited file
gets a new folder and a new run - old data is never mixed with new products.
"""

import hashlib
import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from retina.engine.csvsafe import to_csv
from retina.products import aisles as A
from retina.products.check import report_text

ROOT = Path(__file__).resolve().parents[2]
UPLOADS = ROOT / "data" / "uploads"
U, E = "USER_PROVIDED", "ESTIMATED"
AISLE_STATS = ROOT / "data" / "processed" / "instacart_aisle_stats.csv"


def tag_for(path, mode):
    data = Path(path).read_bytes()
    stem = re.sub(r"[^a-z0-9]+", "-", Path(path).stem.lower()).strip("-")[:30] or "products"
    return f"{stem}-{hashlib.md5(data).hexdigest()[:6]}" + ("-add" if mode == "add" else "")


def coverage(products):
    """Share of what real shoppers buy (Instacart order lines) that falls in the uploaded aisles."""
    if not AISLE_STATS.exists():
        return None
    stats = pd.read_csv(AISLE_STATS).set_index("aisle")["lines"]
    aisles = {p["aisle"] for p in products}
    return float(stats[stats.index.isin(aisles)].sum() / stats.sum())


def to_frame(products, file_name):
    table = A.aisle_table()
    us_median = float(table["median_popularity"].median())
    given = [p["popularity"] for p in products if p["popularity"]]
    scale = us_median / float(np.median(given)) if given else 1.0
    rows = []
    for i, p in enumerate(products, 1):
        a = table.loc[p["aisle"]]
        promo = p["promo_price"]
        rows.append({
            "product_id": f"U-{p['product_id']}" if p["product_id"] else f"U-{i:05d}",
            "product_name": p["name"], "brand": p["brand"] or None, "category": p["category"] or p["aisle"],
            "subcategory": None, "price": promo if promo else p["price"], "original_price": p["price"],
            "discount_percentage": round((1 - promo / p["price"]) * 100, 1) if promo else 0.0,
            "currency": "USD", "availability": None, "rating": None, "review_count": None,
            "product_url": None, "image_url": None, "source": f"upload:{file_name}",
            "built_at": date.today().isoformat(), "price_date": None,
            "discount_type": "SALE" if promo else None,
            "cpi_item_code": a["cpi_item_code"], "cpi_item_name": a["cpi_item_name"],
            "cpi_mapped_by": "aisle", "instacart_match": None, "match_score": None,
            "popularity": round(p["popularity"] * scale, 2) if p["popularity"] else float(a["median_popularity"]),
            "popularity_source": "upload" if p["popularity"] else "aisle_median_us_catalog",
            "department": p["department"], "department_source": "upload_aisle",
            "price_label": U, "product_facts_label": U,
            "popularity_label": U if p["popularity"] else E,
            "department_label": U, "cpi_category_label": E,
            "aisle": p["aisle"], "aisle_confidence": 1.0, "aisle_source": "upload", "aisle_label": U,
            "real_category": None, "department_check": None, "pack_size": p["pack_size"],
            "promo_label": U if promo else "none (no deal price given)",
            "shelf_life_days": p["shelf_life_days"],
        })
    return pd.DataFrame(rows)


def save(check, path, mode):
    """Write the catalog for the engine. Returns (catalog path, tag, summary dict)."""
    if mode not in ("only", "add"):
        raise ValueError("mode must be 'only' or 'add'")
    tag = tag_for(path, mode)
    out = UPLOADS / tag
    out.mkdir(parents=True, exist_ok=True)
    frame = to_frame(check.products, Path(path).name)
    if mode == "add":
        us = pd.read_csv(A.US_CATALOG, dtype={"product_id": str}, low_memory=False)
        frame = pd.concat([us, frame], ignore_index=True)
    to_csv(frame, out / "catalog.csv", index=False)
    (out / "check_report.txt").write_text(report_text(check, max_lines=10**9) + "\n", encoding="utf-8")
    summary = {"file": Path(path).name, "tag": tag, "mode": mode, "uploaded_products": len(check.products),
               "catalog_products": len(frame), "aisles": len({p["aisle"] for p in check.products}),
               "departments": len({p["department"] for p in check.products}),
               "demand_covered": coverage(check.products), "warnings": len(check.warnings),
               "skipped_rows": check.dropped, "checked": date.today().isoformat()}
    (out / "upload.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return out / "catalog.csv", tag, summary


TEMPLATE_ROWS = [
    # name, brand, category, price, pack_size, promo_price, popularity  (EXAMPLES, not real products)
    ("Whole Milk", "Example Dairy", "milk", "3.49", "1 gal", "", "120"),
    ("Greek Yogurt, Plain", "Example Dairy", "yogurt", "5.99", "32 oz", "4.99", "80"),
    ("Sparkling Water Lime", "Example Drinks", "water seltzer sparkling water", "4.99", "8 x 12 fl oz", "", "60"),
    ("Chocolate Chip Cookies", "Example Bakery", "cookies cakes", "3.99", "13 oz", "2.99", "45"),
    ("Basmati Rice", "Example Pantry", "grains rice dried goods", "6.49", "2 lb", "", "30"),
    ("Toothpaste Mint", "Example Care", "oral hygiene", "2.99", "4 oz", "", "15"),
]


def write_template(folder):
    """products_template.csv (example rows) + aisles.csv (the 124 aisles to choose from)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cols = ["name", "brand", "category", "price", "pack_size", "promo_price", "popularity"]
    to_csv(pd.DataFrame(TEMPLATE_ROWS, columns=cols), folder / "products_template.csv", index=False)
    t = A.aisle_table().reset_index()[["aisle", "department", "products", "median_price"]]
    t = t.rename(columns={"products": "us_catalog_products", "median_price": "typical_price_usd"})
    to_csv(t, folder / "aisles.csv", index=False)
    return folder / "products_template.csv", folder / "aisles.csv"
