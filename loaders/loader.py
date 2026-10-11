"""RETINA data loaders: download, clean and cache the public datasets RETINA Mart uses.

Sources (full details, licences and download dates are written to data/data_sources.json):
- Open Prices: Open Food Facts, https://prices.openfoodfacts.org  (ODbL)
- Open Food Facts / Open Beauty Facts / Open Products Facts: https://world.openfoodfacts.org/data  (ODbL)
- US CPI: U.S. Bureau of Labor Statistics, https://www.bls.gov/cpi/  (public domain)
- Instacart: "The Instacart Online Grocery Shopping Dataset 2017" (non-commercial use only)
- Online Retail II: Chen, D. (2019). UCI Machine Learning Repository.
  https://doi.org/10.24432/C5CG6D  (CC BY 4.0)
- India CPI: Ministry of Statistics and Programme Implementation (MoSPI), https://api.mospi.gov.in
"""

import csv
import json
import os
import re
import ssl
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# =============================================================== settings
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DATA_SOURCES_FILE = DATA_DIR / "data_sources.json"

# --- UCI Online Retail II (reference only: UK wholesale, not used for shopper behaviour)
ONLINE_RETAIL_FILE = RAW_DIR / "online_retail_II.xlsx"
TRANSACTIONS_FILE = PROCESSED_DIR / "transactions.csv"

# --- Product catalog: Open Prices + Open Food Facts family
OPEN_PRICES_FILE = RAW_DIR / "prices.parquet"
CATALOG_COUNTRY = "US"  # ISO country code; change to "FR" for France
CATALOG_FILE = PROCESSED_DIR / f"catalog_{CATALOG_COUNTRY.lower()}.csv"

# Product-detail databases, searched in this order (food first).
# name: (big export file, small cached subset, product page URL)
DETAIL_SOURCES = {
    "openfoodfacts": (RAW_DIR / "en.openfoodfacts.org.products.csv.gz",
                      PROCESSED_DIR / "off_products_subset.csv",
                      "https://world.openfoodfacts.org/product/"),
    "openbeautyfacts": (RAW_DIR / "en.openbeautyfacts.org.products.csv.gz",
                        PROCESSED_DIR / "obf_subset.csv",
                        "https://world.openbeautyfacts.org/product/"),
    "openproductsfacts": (RAW_DIR / "en.openproductsfacts.org.products.csv.gz",
                          PROCESSED_DIR / "opf_subset.csv",
                          "https://world.openproductsfacts.org/product/"),
}
OFF_COLUMNS = [
    "code", "product_name", "brands", "categories_en", "main_category_en",
    "quantity", "image_url", "countries_en",
]
CATALOG_FIELDS = [
    "product_id", "product_name", "brand", "category", "subcategory",
    "price", "original_price", "discount_percentage", "currency",
    "availability", "rating", "review_count", "product_url",
    "image_url", "source", "built_at", "price_date", "discount_type",
]

# Found in the data audit: items that are not groceries, and how to spot price typos
NON_GROCERY_PATTERN = (r"gaming|desktop|\bps5\b|playstation|\bconsole\b|vitamix|midea"
                       r"|1080v\d+|sonicare|\blaptop\b|\btv\b|\bmonitor\b|\bblankets\b")  # plural only: keeps "pigs in a blanket"
PRICE_OUTLIER_RATIO = 30  # price 30x above (or below) its category median = likely typo

# --- India CPI (MoSPI API) - kept for a future Indian store, not used by RETINA Mart
MOSPI_API = "https://api.mospi.gov.in"
CPI_FILE = PROCESSED_DIR / "cpi_india.csv"
CPI_STATE_CODE = "1"   # 1 = All India (from MoSPI's filter list; 20 = Maharashtra)
CPI_PAGE_SIZE = 100    # MoSPI allows 10-200 records per page

# --- US CPI (BLS files)
BLS_BASE = "https://download.bls.gov/pub/time.series/cu/"
BLS_FILES = ["cu.data.0.Current", "cu.item"]
BLS_CONTACT = os.environ.get("BLS_CONTACT", "RETINA student project")  # BLS asks scripts for contact info: $env:BLS_CONTACT = "your email"
US_CPI_FILE = PROCESSED_DIR / "cpi_us.csv"

# Keyword rules: product text -> BLS CPI item code. First match wins, so specific rules
# come before general ones.
CPI_RULES = [
    (r"pet food|dog food|cat food|\bpets?\b", "SERB01"),          # Pets and pet products
    (r"detergent|cleaning|cleaner|disinfect|\bbleach\b|laundry|fabric softener|dish soap"
     r"|dishwasher|paper towel|toilet paper|bathroom tissue|facial tissue|trash bag"
     r"|garbage bag|aluminum foil|plastic wrap|\bsponges?\b(?!\W*(cakes?|fingers?))|air freshener|household", "SEHN"),
    (r"shampoo|conditioner|soap|toothpaste|tooth ?brush|floss|mouthwash|deodorant"
     r"|cosmetic|skin care|hair care|hair ?colou?r|body wash|lotion|sunscreen"
     r"|lip balm|razor|shaving|make-?up|acne|sanitizer"
     # over-the-counter medicine: Instacart files it under "personal care"
     r"|ibuprofen|acetaminophen|advil|tylenol|antacid|\btums\b|zzzquil|pain relief"
     r"|gas relief|melatonin|multivitamin", "SEGB"),               # Personal care
    (r"alcoholic|\bwines?\b|\bbeers?\b|spirits|liquor|\bciders?\b|vodka|whisk|champagne"
     r"|prosecco|cremant|tequila", "SEFW"),                        # Alcohol at home
    (r"coffee|\bteas?\b|cocoa|espresso|decaf|french roast|dark roast|medium roast|sumatra", "SEFP"),
    (r"juice|soda|soft drink|carbonated|energy drink|sports drink|\bwaters?\b|beverage"
     r"|\bdrinks?\b", "SEFN"),                                     # Juices and drinks
    (r"\beggs?\b", "SEFH"),                                        # Eggs
    (r"peanut butter|nut butter|\boils?\b|margarine|\bfats?\b", "SEFS"),  # Fats and oils
    (r"ice cream|dair(y|ies)|\bmilks?\b|cheese|yog(h)?urt|\bbutter\b|\bcream\b", "SEFJ"),  # Dairy
    (r"\bbeef\b|\bveal\b", "SEFC"),
    (r"\bpork\b|\bham\b|bacon", "SEFD"),
    (r"chicken|turkey|poultry", "SEFF"),
    (r"\bfish|seafood|tuna|salmon|shrimp", "SEFG"),
    (r"\bmeats?\b|sausage|deli|charcuterie|jerky", "SEFE"),       # Other meats
    (r"sauce|condiment|dressing|spice|season|ketchup|mustard|mayonnaise"
     r"|\bsalt\b|cinnamon|\bherbs?\b|parsley|basil|oregano|vanilla", "SEFT"),  # Other foods
    (r"canned|frozen (fruits|vegetables)|dried fruit|pickle", "SEFM"),  # Processed fruits & veg
    (r"breakfast cereal|granola|\boats?\b|oatmeal", "SEFA02"),
    (r"\brice\b|pasta|noodle|macaroni|spaghetti|angel hair", "SEFA03"),
    (r"flour|baking mix", "SEFA01"),
    (r"\bbreads?\b|bagel|tortilla", "SEFB01"),
    (r"cake|cookie|biscuit|pastr|\bpies?\b|muffin|donut", "SEFB03"),
    (r"cand(y|ies)|chocolate|confectioner|sweets|\bsugars?\b|honey|syrup|\bjams?\b"
     r"|gumm(y|ies)|caramel|licorice|marshmallow", "SEFR"),
    (r"\bfruits?\b|apple|banana|berr|grape|orange|\bpears?\b|plum|citrus|melon", "SEFK"),
    (r"vegetable|potato|onion|lettuce|salad|tomato|carrot|\bpeppers?\b", "SEFL"),
    (r"cereal", "SEFA"),
    (r"\bbeans?\b|lentil|chickpea|hummus|tofu", "SEFM"),           # Processed veg / legumes
    (r"snack|chip|crisp|cracker|pretzel|popcorn|\bnuts?\b|almond|peanut|cashew|seeds?\b"
     r"|soup|broth|bouillon|\bstock\b|meal|frozen|baby food|dessert|pizza|sandwich|burrito"
     r"|ramen|salsa|\bdips?\b|spread|protein bar|granola bar|\bgum\b|mints?\b|pudding", "SEFT"),
]
NON_FOOD_CPI_CODES = {"SERB01", "SEHN", "SEGB"}
# Products found only in a non-food database get that database's category by default
CPI_FALLBACK_BY_SOURCE = {"openbeautyfacts": "SEGB", "openproductsfacts": "SEHN"}
GENERIC_CATEGORIES = {"Plant-based foods and beverages"}  # too broad to tell us anything

# --- Shopper behaviour (Instacart) and store departments
INSTACART_DIR = RAW_DIR / "instacart"
SHOPPER_PARAMS_FILE = PROCESSED_DIR / "shopper_params.json"
DEPARTMENTS_FILE = PROCESSED_DIR / "store_departments.json"
MIN_PRODUCTS_PER_DEPARTMENT = 20
PRODUCT_ORDERS_FILE = PROCESSED_DIR / "instacart_product_orders.csv"
MATCH_MIN_SIMILARITY = 0.8   # name similarity needed to trust a match (0-1)
MATCH_CANDIDATES = 5         # look at the 5 most similar names, take the first in a valid department
PAIRS_FILE = PROCESSED_DIR / "instacart_pairs.csv"
PAIR_MIN_ORDERS = 100     # a pair must appear together in at least 100 real orders to be kept
PAIR_TOP_PARTNERS = 5     # keep each product's 5 strongest partners

# Instacart department -> CPI categories in our catalog (approximate: aisles != CPI groups)
DEPARTMENT_TO_CPI = {
    "produce":         ["SEFK", "SEFL"],
    "dairy eggs":      ["SEFJ", "SEFH"],
    "snacks":          ["SEFT", "SEFR", "SEFB03"],
    "beverages":       ["SEFN", "SEFP"],
    "frozen":          ["SEFT", "SEFM"],
    "pantry":          ["SEFS", "SEFA01", "SEFT", "SEFR"],
    "bakery":          ["SEFB01", "SEFB03"],
    "canned goods":    ["SEFM", "SEFT"],
    "deli":            ["SEFE", "SEFT"],
    "dry goods pasta": ["SEFA03", "SEFA"],
    "breakfast":       ["SEFA02", "SEFA"],
    "meat seafood":    ["SEFC", "SEFD", "SEFF", "SEFG", "SEFE"],
    "international":   ["SEFT", "SEFA03"],
    "alcohol":         ["SEFW"],
    "household":       ["SEHN"],
    "personal care":   ["SEGB"],
    "pets":            ["SERB01"],
    "babies":          [],
    "missing":         ["SAF11"],   # unclassified products go to the unclassified departments
    "other":           ["SAF11"],
    "bulk":            ["SAF11"],
}

# Every product gets ONE store department (fixes CPI categories shared by many departments).
# Order of evidence: Instacart's own department (matched products) -> category keywords
# (first match wins) -> product-name keywords -> default for its CPI category.
DEPARTMENT_RULES = [
    (r"pet food|dog food|cat food", "pets"),
    (r"ice cream|frozen|popsicle|sorbet|gelato|mochi", "frozen"),
    (r"alcoholic|\bwines?\b|\bbeers?\b|spirits|liquor|\bciders?\b|champagne|vodka|whisk", "alcohol"),
    (r"beverage|\bdrinks?\b|juice|soda|\bwaters?\b|coffee|\bteas?\b|cocoa|kombucha", "beverages"),
    (r"dair|cheese|yog(h)?urt|\bmilks?\b|\bbutter\b|\bcream\b|\beggs?\b", "dairy eggs"),
    (r"breakfast|cereal|granola|\boats?\b|oatmeal|pancake|waffle", "breakfast"),
    (r"\bbreads?\b|bakery|bagel|tortilla|\bbuns?\b|muffin|croissant", "bakery"),
    (r"snack|\bchips?\b|crisp|cracker|popcorn|pretzel|cand(y|ies)|chocolate|confection"
     r"|cookie|biscuit|\bbars?\b|\bnuts?\b|almond|peanut|cashew|jerky", "snacks"),
    (r"canned|soup|broth|\bbeans?\b|\btuna\b", "canned goods"),
    (r"pasta|noodle|\brice\b|grain|macaroni|spaghetti|quinoa", "dry goods pasta"),
    (r"\bmeats?\b|beef|pork|chicken|turkey|poultry|\bfish\b|seafood|shrimp|salmon", "meat seafood"),
    (r"deli|salami|\bham\b|sausage|prepared|\bmeals?\b|sandwich|hummus|dip", "deli"),
    (r"fresh|fruit|vegetable|produce|salad|lettuce|spinach|kale|broccoli|avocado|apple|banana"
     r"|berr|grape|orange|lemon|onion|potato|tomato|carrot|pepper|mushroom|herb", "produce"),
    (r"sauce|condiment|spice|season|\boils?\b|vinegar|baking|flour|sugar|syrup|honey"
     r"|spread|\bjams?\b|ketchup|mustard|mayonnaise|salt", "pantry"),
]
CPI_DEFAULT_DEPARTMENT = {
    "SEFK": "produce", "SEFL": "produce", "SEFJ": "dairy eggs", "SEFH": "dairy eggs",
    "SEFR": "snacks", "SEFB03": "snacks", "SEFN": "beverages", "SEFP": "beverages",
    "SEFT": "pantry", "SEFM": "canned goods", "SEFS": "pantry", "SEFA01": "pantry",
    "SEFB01": "bakery", "SEFE": "deli", "SEFA03": "dry goods pasta", "SEFA": "dry goods pasta",
    "SEFA02": "breakfast", "SEFC": "meat seafood", "SEFD": "meat seafood", "SEFF": "meat seafood",
    "SEFG": "meat seafood", "SEFW": "alcohol", "SEHN": "household", "SEGB": "personal care",
    "SERB01": "pets", "SAF11": "missing",
}
# --- Every dataset's origin, for data/data_sources.json
DATA_SOURCES = [
    {"name": "Open Prices", "files": ["raw/prices.parquet"],
     "url": "https://huggingface.co/datasets/openfoodfacts/open-prices",
     "license": "ODbL 1.0 - credit Open Prices; modified databases must stay ODbL",
     "used_for": "US shelf prices, discounts and price dates"},
    {"name": "Open Food Facts", "files": ["raw/en.openfoodfacts.org.products.csv.gz"],
     "url": "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz",
     "license": "ODbL 1.0 (database); product images CC BY-SA",
     "used_for": "food product names, brands, categories, images"},
    {"name": "Open Beauty Facts", "files": ["raw/en.openbeautyfacts.org.products.csv.gz"],
     "url": "https://static.openbeautyfacts.org/data/en.openbeautyfacts.org.products.csv.gz",
     "license": "ODbL 1.0 (database); product images CC BY-SA",
     "used_for": "personal care product details"},
    {"name": "Open Products Facts", "files": ["raw/en.openproductsfacts.org.products.csv.gz"],
     "url": "https://static.openproductsfacts.org/data/en.openproductsfacts.org.products.csv.gz",
     "license": "ODbL 1.0 (database); product images CC BY-SA",
     "used_for": "household product details"},
    {"name": "US CPI (Bureau of Labor Statistics)", "files": ["raw/cu.data.0.Current", "raw/cu.item"],
     "url": "https://download.bls.gov/pub/time.series/cu/",
     "license": "Public domain (U.S. government work); citation requested",
     "used_for": "monthly price changes per product category"},
    {"name": "Instacart Online Grocery Shopping Dataset 2017",
     "files": ["raw/instacart/orders.csv", "raw/instacart/order_products__prior.csv"],
     "url": "https://www.kaggle.com/c/instacart-market-basket-analysis",
     "license": "Non-commercial use only; do not redistribute the raw files",
     "used_for": "shopper behaviour: basket size, visit gaps, reorders, departments"},
    {"name": "UCI Online Retail II", "files": ["raw/online_retail_II.xlsx"],
     "url": "https://archive.ics.uci.edu/dataset/502/online+retail+ii",
     "license": "CC BY 4.0",
     "used_for": "reference only (UK wholesale); NOT used for shopper behaviour"},
    {"name": "India CPI, base 2024 (MoSPI)", "files": ["processed/cpi_india.csv"],
     "url": "https://api.mospi.gov.in/api/cpi/getCPIData",
     "license": "Government of India data - credit MoSPI; exact API terms not verified",
     "used_for": "kept for a future Indian store; NOT used by RETINA Mart"},
    {"name": "webscraper.io e-commerce test site", "files": ["raw/products.csv"],
     "url": "https://webscraper.io/test-sites/e-commerce/static",
     "license": "Public practice site built for scraping exercises",
     "used_for": "learning module only (scraping/scraper.py)"},
]


# ================================================= UCI Online Retail II
def load_online_retail():
    """Read both sheets (2009-10 and 2010-11) into one DataFrame."""
    sheets = pd.read_excel(
        ONLINE_RETAIL_FILE,
        sheet_name=None,  # None = read every sheet
        dtype={"Invoice": str, "StockCode": str},
    )
    return pd.concat(sheets.values(), ignore_index=True)


def _drop(df, keep_mask, reason):
    """Keep rows where keep_mask is True and print how many were removed."""
    removed = len(df) - int(keep_mask.sum())
    print(f"  removed {removed:>8,} rows: {reason}")
    return df[keep_mask]


def clean_online_retail(df):
    """Keep only real, completed product sales with a known customer."""
    print("\nCleaning Online Retail II")
    df = _drop(df, ~df["Invoice"].str.startswith("C", na=False), "cancelled invoices")
    df = _drop(df, df["Quantity"] > 0, "zero or negative quantity")
    df = _drop(df, df["Price"] > 0, "zero or negative price")
    df = _drop(df, df["Customer ID"].notna(), "missing customer ID")
    df = _drop(df, df["StockCode"].str.match(r"^\d{5}", na=False),
               "non-product codes (postage, fees, adjustments)")
    df = _drop(df, df["Description"].notna(), "missing description")
    df = _drop(df, ~df.duplicated(), "exact duplicate rows")

    df = df.rename(columns={
        "Invoice": "invoice",
        "StockCode": "product_id",
        "Description": "product_name",
        "Quantity": "quantity",
        "InvoiceDate": "invoice_date",
        "Price": "unit_price",
        "Customer ID": "customer_id",
        "Country": "country",
    })
    df["customer_id"] = df["customer_id"].astype(int)
    df["product_name"] = df["product_name"].str.strip()
    df["line_total"] = (df["quantity"] * df["unit_price"]).round(2)
    df["currency"] = "GBP"
    return df.reset_index(drop=True)


def build_transactions():
    print("Reading Excel file (takes 1-3 minutes)...")
    raw = load_online_retail()
    print(f"Raw rows: {len(raw):,}")

    clean = clean_online_retail(raw)
    TRANSACTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(TRANSACTIONS_FILE, index=False)

    print(f"\nClean rows: {len(clean):,}")
    print(
        "Customers:", clean["customer_id"].nunique(),
        "| Products:", clean["product_id"].nunique(),
        "| Invoices:", clean["invoice"].nunique(),
    )
    print(f"Total revenue: {clean['line_total'].sum():,.0f} GBP")
    print("Saved to:", TRANSACTIONS_FILE)


# ====================================================== Open Prices
def load_open_prices():
    df = pd.read_parquet(OPEN_PRICES_FILE)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")  # mixed/empty -> datetime/NaT
    for col in ["price", "price_without_discount"]:
        df[col] = df[col].astype("float64")  # Decimal -> normal number (empty -> NaN)
    return df


def local_barcodes(prices):
    """All barcodes with a price in the catalog country."""
    local = prices[prices["location_osm_address_country_code"].eq(CATALOG_COUNTRY)
                   & prices["product_code"].notna()]
    return set(local["product_code"].astype(str))


# ============================== product details (Open Food Facts family)
def extract_open_facts(codes, source_file, subset_file):
    """Scan an Open Facts export in chunks and keep only products whose barcode is in codes."""
    matches = []
    reader = pd.read_csv(
        source_file,
        sep="\t",
        usecols=lambda c: c in OFF_COLUMNS,  # read only the columns we need
        dtype=str,
        chunksize=200_000,
        quoting=csv.QUOTE_NONE,
        on_bad_lines="skip",
        encoding_errors="replace",
    )
    total = 0
    for i, chunk in enumerate(reader, start=1):
        found = chunk[chunk["code"].isin(codes)]
        matches.append(found)
        total += len(found)
        print(f"  scanned {i * 200_000:>9,} rows | matched {total:,}", end="\r")
    print()

    df = pd.concat(matches, ignore_index=True).drop_duplicates("code")
    subset_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(subset_file, index=False)
    return df


def load_product_details(codes):
    """Find details for each barcode: Open Food Facts first, then the sister databases
    for barcodes still missing. Each database is scanned once and cached."""
    remaining = set(codes)
    parts = []
    print(f"\nProduct details for {len(codes):,} {CATALOG_COUNTRY} barcodes:")
    for name, (source_file, subset_file, _) in DETAIL_SOURCES.items():
        if subset_file.exists():
            found = pd.read_csv(subset_file, dtype=str)
        elif source_file.exists():
            print(f"  scanning {name} (one time only)...")
            found = extract_open_facts(remaining, source_file, subset_file)
        else:
            print(f"  {name}: not downloaded ({source_file.name}) - skipped")
            continue
        found = found[found["code"].isin(remaining)].copy()
        found["details_source"] = name
        remaining -= set(found["code"])
        parts.append(found)
        print(f"  {name:<18} matched {len(found):>6,}")
    print(f"  {'no details found':<18} {len(remaining):>13,}")
    if not parts:
        return pd.DataFrame(columns=OFF_COLUMNS + ["details_source"])
    return pd.concat(parts, ignore_index=True)


# ============================================================ catalog
def first_item(text):
    """'Plant-based foods,Fruits,Plums' -> 'Plant-based foods'."""
    if not isinstance(text, str) or not text.strip():
        return None
    return text.split(",")[0].strip()


def last_item(text):
    """'Plant-based foods,Fruits,Plums' -> 'Plums'."""
    if not isinstance(text, str) or not text.strip():
        return None
    return text.split(",")[-1].strip()


def build_catalog(prices, details):
    """One row per product: latest price + product details, in CATALOG_FIELDS order."""
    local = prices[
        prices["location_osm_address_country_code"].eq(CATALOG_COUNTRY)
        & prices["product_code"].notna()
        & (prices["price"] > 0)
    ].copy()
    currency = local["currency"].mode()[0]           # main currency, e.g. USD
    local = local[local["currency"].eq(currency)]
    local["product_code"] = local["product_code"].astype(str)

    # Latest price per product (rows without a date sort first, so dated rows win)
    latest = local.sort_values("date", na_position="first").groupby("product_code").tail(1)

    df = latest.merge(details, left_on="product_code", right_on="code",
                      how="left", suffixes=("_op", "")).reset_index(drop=True)

    # Discount: only trust it when the "before" price is higher than the price paid
    without = pd.to_numeric(df["price_without_discount"], errors="coerce")
    has_discount = df["price_is_discounted"].eq(True) & (without > df["price"])
    original = df["price"].where(~has_discount, without)

    # Where the product details came from, and its page on that database
    url_base = df["details_source"].map({k: v[2] for k, v in DETAIL_SOURCES.items()})
    source = ("open_prices+" + df["details_source"]).where(df["details_source"].notna(), "open_prices")

    catalog = pd.DataFrame({
        "product_id": df["product_code"],
        "product_name": df["product_name"].fillna(df["product_name_op"]),
        "brand": df["brands"].map(first_item),
        "category": df["categories_en"].map(first_item),
        "subcategory": df["main_category_en"].fillna(df["categories_en"].map(last_item)),
        "price": df["price"].round(2),
        "original_price": original.round(2),         # regular shelf price (use this as base)
        "discount_percentage": ((original - df["price"]) / original * 100).round(1),
        "currency": currency,
        "availability": None,        # unknown: stock is simulated in RETINA Mart
        "rating": None,              # no clearly licensed source
        "review_count": None,        # no clearly licensed source
        "product_url": url_base + df["product_code"],  # empty when no database has the product
        "image_url": df["image_url"],
        "source": source,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),  # when this file was built
        "price_date": df["date"].dt.date,                                       # when the price was seen
        "discount_type": df["discount_type"].replace("", None).where(has_discount),
    })
    for col in ["category", "subcategory"]:
        catalog[col] = catalog[col].replace("Undefined", None)  # OFF placeholder -> missing
    catalog = catalog[catalog["product_name"].fillna("").str.strip().ne("")]  # drop empty/blank names
    return catalog[CATALOG_FIELDS].reset_index(drop=True)


def clean_catalog(catalog):
    """Remove non-grocery items and price typos found in the audit."""
    text = catalog["product_name"].fillna("") + " | " + catalog["subcategory"].fillna("")
    non_grocery = text.str.contains(NON_GROCERY_PATTERN, case=False)

    # Compare each price with the typical (median) price of its CPI category
    median = catalog.groupby("cpi_item_code")["price"].transform("median")
    ratio = catalog["price"] / median
    typo = ~non_grocery & ((ratio > PRICE_OUTLIER_RATIO) | (ratio < 1 / PRICE_OUTLIER_RATIO))

    removed = catalog[non_grocery | typo]
    print(f"\nRemoved {non_grocery.sum()} non-grocery items and {typo.sum()} price outliers:")
    print(removed[["product_name", "price", "cpi_item_name"]].to_string(index=False))
    return catalog[~(non_grocery | typo)].reset_index(drop=True)


def summarize_catalog(cat):
    discounted = cat["discount_percentage"] > 0
    print(f"\nCatalog: {len(cat):,} products")
    print(f"  discounted: {discounted.sum():,} "
          f"(avg {cat.loc[discounted, 'discount_percentage'].mean():.1f}% off)")
    print(f"  brand known: {cat['brand'].notna().mean() * 100:.0f}% | "
          f"category known: {cat['category'].notna().mean() * 100:.0f}%")
    print(f"  price range: {cat['price'].min()} - {cat['price'].max()} {cat['currency'].iloc[0]}")
    print("  sources:", cat["source"].value_counts().to_dict())
    print("\nTop 10 categories:")
    print(cat["category"].value_counts().head(10).to_string())
    print("\nMost expensive 5 (check for price typos):")
    print(cat.nlargest(5, "price")[["product_name", "brand", "price", "original_price"]]
          .to_string(index=False))
    print("\nDiscount types:", cat["discount_type"].value_counts().to_dict())
    print("Price dates:", cat["price_date"].min(), "to", cat["price_date"].max())


# =============================================== India CPI (MoSPI API)
class LegacySSLAdapter(HTTPAdapter):
    """Allow the old TLS renegotiation that api.mospi.gov.in still uses.
    Certificate verification stays ON. Mounted only for MoSPI, not other sites."""

    def init_poolmanager(self, *args, **kwargs):
        ctx = ssl.create_default_context()
        ctx.options |= getattr(ssl, "OP_LEGACY_SERVER_CONNECT", 0x4)
        kwargs["ssl_context"] = ctx
        return super().init_poolmanager(*args, **kwargs)


def mospi_session():
    session = requests.Session()
    session.headers.update({"User-Agent": "RETINA/0.1 (learning project)"})
    session.mount(MOSPI_API, LegacySSLAdapter())
    return session


def fetch_cpi():
    """Download every CPI (base 2024) record for one state, page by page."""
    session = mospi_session()
    url = MOSPI_API + "/api/cpi/getCPIData"
    rows, page, total_pages = [], 1, 1

    while page <= total_pages:
        params = {
            "base_year": "2024", "series": "Current", "Format": "JSON",
            "state_code": CPI_STATE_CODE, "limit": str(CPI_PAGE_SIZE), "page": str(page),
        }
        r = session.get(url, params=params, timeout=60)
        body = r.json()
        if r.status_code != 200:
            raise RuntimeError(f"MoSPI error on page {page}: {body}")

        rows.extend(body.get("data", []))
        total_pages = body["meta_data"]["totalPages"]
        print(f"  page {page}/{total_pages} | rows so far: {len(rows):,}", end="\r")
        page += 1
        time.sleep(0.5)  # be polite to the government server
    print()
    return pd.DataFrame(rows)


def clean_cpi(df):
    """Numbers as numbers, a real date column, and the category level of each row."""
    df["index"] = pd.to_numeric(df["index"], errors="coerce")
    df["inflation"] = pd.to_numeric(df["inflation"], errors="coerce")
    df["date"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"],
                                format="%Y-%B", errors="coerce")

    # Most specific category present in each row: division > group > class > sub_class > item
    df["level"] = "division"
    for lv in ["group", "class", "sub_class", "item"]:
        df.loc[df[lv].notna(), "level"] = lv
    df["category_name"] = (df["item"].fillna(df["sub_class"]).fillna(df["class"])
                           .fillna(df["group"]).fillna(df["division"]))

    keep = ["date", "state", "sector", "level", "category_name", "division", "group",
            "class", "sub_class", "item", "code", "index", "inflation"]
    return df[keep].sort_values(["date", "sector", "level"]).reset_index(drop=True)


def build_cpi():
    if CPI_FILE.exists():
        print("\nIndia CPI already downloaded:", CPI_FILE)
        cpi = pd.read_csv(CPI_FILE, parse_dates=["date"])
    else:
        print("\nDownloading India CPI (MoSPI, base 2024)...")
        cpi = clean_cpi(fetch_cpi())
        CPI_FILE.parent.mkdir(parents=True, exist_ok=True)
        cpi.to_csv(CPI_FILE, index=False)
        print("Saved to:", CPI_FILE)

    print(f"India CPI rows: {len(cpi):,} | months: {cpi['date'].min():%b %Y} to {cpi['date'].max():%b %Y}")
    latest = cpi[cpi["date"].eq(cpi["date"].max())]
    general = latest[latest["category_name"].eq("CPI (General)")]
    print("Latest general CPI:\n", general[["sector", "index", "inflation"]].to_string(index=False))


# ================================================== US CPI (BLS files)
def download_bls():
    """Download the BLS CPI files once (BLS asks robots to include contact info)."""
    headers = {"User-Agent": BLS_CONTACT}
    for name in BLS_FILES:
        path = RAW_DIR / name
        if path.exists():
            continue
        print(f"  downloading {name} ...")
        r = requests.get(BLS_BASE + name, headers=headers, timeout=120)
        r.raise_for_status()
        path.write_bytes(r.content)
        time.sleep(1)  # be polite


def read_bls(name):
    """BLS files are tab-separated with padded spaces, so strip everything."""
    df = pd.read_csv(RAW_DIR / name, sep="\t", dtype=str)
    df.columns = df.columns.str.strip()
    return df.apply(lambda col: col.str.strip())


def add_yoy_inflation(cpi):
    """Year-over-year inflation: match each month with the SAME month last year by date."""
    prev = cpi[["item_code", "date", "index"]].copy()
    prev["date"] = prev["date"] + pd.DateOffset(years=1)
    cpi = cpi.drop(columns="inflation", errors="ignore").merge(
        prev.rename(columns={"index": "index_prev"}), on=["item_code", "date"], how="left")
    cpi["inflation"] = ((cpi["index"] / cpi["index_prev"] - 1) * 100).round(2)
    return cpi.drop(columns="index_prev")


def fill_cpi_gaps(cpi):
    """Fill missing months (Oct 2025, US shutdown) by straight-line interpolation between
    the months before and after, flag them, then recalculate inflation."""
    cpi = cpi.sort_values(["item_code", "date"]).reset_index(drop=True)
    was_missing = cpi["index"].isna()
    cpi["index"] = (cpi.groupby("item_code")["index"]
                    .transform(lambda s: s.interpolate(limit_area="inside")))
    newly_filled = was_missing & cpi["index"].notna()

    previous_flag = cpi["index_interpolated"] if "index_interpolated" in cpi.columns else False
    cpi["index_interpolated"] = (newly_filled | previous_flag) & cpi["index"].notna()
    cpi = add_yoy_inflation(cpi)
    print(f"CPI gaps: {int(newly_filled.sum())} filled now | "
          f"{int(cpi['index_interpolated'].sum())} interpolated in total | "
          f"{int(cpi['index'].isna().sum())} left empty (series start/end, no month to draw from)")
    return cpi


def build_us_cpi():
    if US_CPI_FILE.exists():
        print("\nUS CPI already processed:", US_CPI_FILE)
        cpi = pd.read_csv(US_CPI_FILE, parse_dates=["date"])
    else:
        print("\nPreparing US CPI (BLS)...")
        download_bls()
        data = read_bls("cu.data.0.Current")
        items = read_bls("cu.item")

        # CUUR0000 = CPI, Unadjusted, Regular, US city average. Monthly periods only (M13 = annual avg).
        data = data[data["series_id"].str.startswith("CUUR0000")
                    & data["period"].str.match(r"^M(0[1-9]|1[0-2])$")].copy()
        data["item_code"] = data["series_id"].str[8:]
        data["date"] = pd.to_datetime(data["year"] + "-" + data["period"].str[1:] + "-01")
        data["index"] = pd.to_numeric(data["value"], errors="coerce")
        data = data[data["date"].dt.year >= 2015]

        cpi = data.merge(items[["item_code", "item_name", "display_level"]],
                         on="item_code", how="left")
        cpi = cpi[["date", "item_code", "item_name", "display_level", "index"]]

    cpi = fill_cpi_gaps(cpi)
    cpi = cpi.sort_values(["item_code", "date"]).reset_index(drop=True)
    US_CPI_FILE.parent.mkdir(parents=True, exist_ok=True)
    cpi.to_csv(US_CPI_FILE, index=False)

    print(f"US CPI rows: {len(cpi):,} | items: {cpi['item_code'].nunique()} | "
          f"months: {cpi['date'].min():%b %Y} to {cpi['date'].max():%b %Y}")
    latest = cpi[cpi["date"].eq(cpi["date"].max())]
    key = latest[latest["item_code"].isin(["SA0", "SAF11", "SEFJ", "SEFR", "SEFN"])]
    print("Latest inflation (key categories):")
    print(key[["item_code", "item_name", "index", "inflation"]].to_string(index=False))
    return cpi


# ======================================== catalog -> US CPI mapping
def match_cpi_rule(text, allowed=None):
    """First CPI rule whose pattern appears in text (optionally only codes in `allowed`)."""
    for pattern, code in CPI_RULES:
        if allowed is not None and code not in allowed:
            continue
        if re.search(pattern, text, flags=re.IGNORECASE):
            return code
    return None


def add_cpi_mapping(catalog, us_cpi):
    """Give every product a BLS CPI item code: subcategory -> category -> name -> fallback."""
    codes, sources = [], []
    rows = zip(catalog["subcategory"], catalog["category"], catalog["product_name"], catalog["source"])
    for sub, cat, name, source in rows:
        sub = sub if isinstance(sub, str) else ""
        cat = cat if isinstance(cat, str) and cat not in GENERIC_CATEGORIES else ""
        name = name if isinstance(name, str) else ""
        details_source = source.replace("open_prices+", "") if isinstance(source, str) else ""
        source_default = CPI_FALLBACK_BY_SOURCE.get(details_source)
        # Products from a non-food database: try non-food rules first; a food rule only
        # counts if no non-food rule matched (a few foods are filed in the wrong database)
        passes = [NON_FOOD_CPI_CODES, None] if source_default else [None]

        code, how = None, None
        for allowed in passes:
            for label, text in (("subcategory", sub), ("category", cat), ("product_name", name)):
                code = match_cpi_rule(text, allowed) if text else None
                if code:
                    how = label
                    break
            if code:
                break
        if code is None and source_default:
            code, how = source_default, "fallback_source"
        if code is None:
            # RETINA Mart is a grocery store: unmatched products are most likely food
            code, how = "SAF11", ("fallback_food" if (sub or cat) else "fallback_unknown")
        codes.append(code)
        sources.append(how)

    names = us_cpi.drop_duplicates("item_code").set_index("item_code")["item_name"]
    catalog["cpi_item_code"] = codes
    catalog["cpi_item_name"] = catalog["cpi_item_code"].map(names)
    catalog["cpi_mapped_by"] = sources

    missing = set(codes) - set(names.index)
    if missing:
        print("WARNING: CPI codes not found in BLS data:", missing)
    return catalog


def summarize_mapping(catalog):
    print("\nCPI mapping - matched by (%):")
    print((catalog["cpi_mapped_by"].value_counts(normalize=True) * 100).round(1).to_string())
    print("\nTop 10 CPI categories:")
    print(catalog["cpi_item_name"].value_counts().head(10).to_string())
    non_food = catalog[catalog["cpi_item_code"].isin(NON_FOOD_CPI_CODES)]
    print("Non-food categories:", non_food["cpi_item_name"].value_counts().to_dict())
    print("\nRandom sample (check if these make sense):")
    sample = catalog.sample(10, random_state=1)
    print(sample[["product_name", "subcategory", "cpi_item_name"]].to_string(index=False))


# ======================================== shopper behaviour (Instacart)
def load_instacart():
    """Read the Instacart files with small number types to save memory."""
    orders = pd.read_csv(INSTACART_DIR / "orders.csv", dtype={
        "order_id": "int32", "user_id": "int32", "order_number": "int16",
        "order_dow": "int8", "order_hour_of_day": "int8",
        "days_since_prior_order": "float32", "eval_set": "category"})
    items = pd.read_csv(INSTACART_DIR / "order_products__prior.csv", dtype={
        "order_id": "int32", "product_id": "int32",
        "add_to_cart_order": "int16", "reordered": "int8"})
    products = pd.read_csv(INSTACART_DIR / "products.csv")
    departments = pd.read_csv(INSTACART_DIR / "departments.csv")
    return orders, items, products, departments


def distribution(series, max_value=None):
    """Turn a column into {value: share}, e.g. basket size {"8": 0.07, ...}. Shares add up to 1."""
    s = series.dropna()
    if max_value is not None:
        s = s.clip(upper=max_value)  # e.g. baskets of 50+ items all counted as "50"
    shares = s.round().astype(int).value_counts(normalize=True).sort_index()
    return {str(k): round(float(v), 5) for k, v in shares.items()}


def shopper_profiles(orders, items):
    """How shoppers DIFFER from each other: each shopper's own averages.
    The simulator gives every NPC its own habits drawn from these distributions."""
    per_order = items.groupby("order_id").agg(basket=("product_id", "size"),
                                              reorders=("reordered", "sum"))
    shop = orders[["order_id", "user_id", "days_since_prior_order"]].merge(
        per_order, left_on="order_id", right_index=True)  # orders that have item data
    per_user = shop.groupby("user_id").agg(
        avg_basket=("basket", "mean"),
        avg_gap=("days_since_prior_order", "mean"),
        items=("basket", "sum"),
        reorders=("reorders", "sum"),
    )
    per_user["reorder_pct"] = per_user["reorders"] / per_user["items"] * 100
    corr = per_user[["avg_basket", "avg_gap"]].corr(method="spearman").iloc[0, 1]
    return {
        "note": "Per-shopper averages (one value per real shopper). Draw one value per NPC.",
        "n_shoppers": int(len(per_user)),
        "avg_basket_size": distribution(per_user["avg_basket"], max_value=50),
        "avg_days_between_orders": distribution(per_user["avg_gap"]),
        "reorder_pct": distribution((per_user["reorder_pct"] / 5).round() * 5),  # 5% steps
        "orders_per_shopper": distribution(orders.groupby("user_id").size()),
        "corr_basket_vs_gap": round(float(corr), 3),
    }


def build_shopper_params():
    """Summarise Instacart into a small JSON file the simulator can load instantly."""
    params = None
    if SHOPPER_PARAMS_FILE.exists():
        with open(SHOPPER_PARAMS_FILE, encoding="utf-8") as f:
            params = json.load(f)
        if "shopper_profiles" in params:
            print("\nShopper params already saved:", SHOPPER_PARAMS_FILE)
        else:
            params = None  # older file without shopper differences -> rebuild once

    if params is None:
        print("\nReading Instacart (about 1-2 minutes)...")
        orders, items, products, departments = load_instacart()
        basket = items.groupby("order_id").size()
        dept = (items[["product_id"]]
                .merge(products[["product_id", "department_id"]], on="product_id")
                .merge(departments, on="department_id")["department"]
                .value_counts(normalize=True))

        params = {
            "source": "Instacart Online Grocery Shopping Dataset 2017 (non-commercial use)",
            "notes": ["Online grocery orders from 2017.",
                      "days_between_orders: '30' means 30 or more days (Instacart caps it).",
                      "basket_size: '50' means 50 or more items (capped by RETINA).",
                      "order_dow: Instacart never documented which day is 0."],
            "n_orders": int(len(orders)),
            "n_shoppers": int(orders["user_id"].nunique()),
            "reorder_rate": round(float(items["reordered"].mean()), 4),
            "basket_size": distribution(basket, max_value=50),
            "days_between_orders": distribution(orders["days_since_prior_order"]),
            "orders_by_day_of_week": distribution(orders["order_dow"]),
            "orders_by_hour": distribution(orders["order_hour_of_day"]),
            "department_share": {k: round(float(v), 5) for k, v in dept.items()},
            "shopper_profiles": shopper_profiles(orders, items),
        }
        SHOPPER_PARAMS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SHOPPER_PARAMS_FILE, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)

    avg_basket = sum(int(k) * v for k, v in params["basket_size"].items())
    prof = params["shopper_profiles"]
    print(f"Shopper params: avg basket {avg_basket:.1f} | reorder rate {params['reorder_rate']:.0%} | "
          f"{len(params['department_share'])} departments")
    print(f"Shopper profiles: {prof['n_shoppers']:,} shoppers | "
          f"basket vs visit-gap correlation {prof['corr_basket_vs_gap']:+.2f}")
    return params


def build_departments(catalog, shopper_params):
    """Store departments: real Instacart shares + the products assigned to each department.
    Departments with too few products are dropped and the shares rescaled."""
    counts = catalog["department"].value_counts()
    kept, dropped = {}, {}
    for dept, share in shopper_params["department_share"].items():
        n_products = int(counts.get(dept, 0))
        if n_products >= MIN_PRODUCTS_PER_DEPARTMENT:
            kept[dept] = {"instacart_share": share, "n_products": n_products}
        else:
            dropped[dept] = {"instacart_share": share, "n_products": n_products}

    total = sum(d["instacart_share"] for d in kept.values())
    for d in kept.values():
        d["share"] = round(d["instacart_share"] / total, 5)  # rescale so shares add up to 1

    unreachable = catalog.loc[~catalog["department"].isin(kept), "department"]
    result = {
        "note": "Each product belongs to one department (catalog column 'department'). "
                "Shares are real Instacart department shares, rescaled after dropping "
                "departments with too few products.",
        "departments": kept,
        "dropped": dropped,
    }
    with open(DEPARTMENTS_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nStore departments: {len(kept)} kept, {len(dropped)} dropped "
          f"({sum(d['instacart_share'] for d in dropped.values()) * 100:.1f}% of Instacart items)")
    for dept, d in sorted(kept.items(), key=lambda x: -x[1]["share"]):
        print(f"  {dept:<16} share {d['share'] * 100:5.1f}% | products {d['n_products']:,}")
    print("  dropped:", ", ".join(f"{k} ({v['n_products']} products)" for k, v in dropped.items()))
    print(f"Products in dropped departments: {len(unreachable)}", unreachable.value_counts().to_dict())
    return result

# ================================ products bought together (Instacart)
def order_pairs(order_ids, idx):
    """All pairs of products inside each order. Rows must be sorted by order.
    Example: an order with products [a, b, c] gives pairs (a,b), (a,c), (b,c)."""
    start = np.r_[0, np.flatnonzero(np.diff(order_ids)) + 1]          # first row of each order
    size = np.diff(np.r_[start, len(order_ids)])
    end = np.repeat(start + size, size)                                # end of each row's order
    row = np.arange(len(order_ids))
    later = end - row - 1                                              # partners after this row
    left = np.repeat(idx, later)
    first = np.repeat(row + 1, later)
    offset = np.arange(later.sum()) - np.repeat(np.cumsum(later) - later, later)
    right = idx[first + offset]
    return np.minimum(left, right), np.maximum(left, right)


def build_product_pairs(catalog):
    """Which products are bought together in real Instacart orders. Only Instacart products
    matched to our catalog are counted. For each product we keep its strongest partners:
    how many orders had both, and the lift (how many times more often than by chance).
    Reads the 32-million-row file in chunks. Cached in a small file (one time only)."""
    if PAIRS_FILE.exists():
        return pd.read_csv(PAIRS_FILE)
    print("\nCounting products bought together in Instacart orders (one time only, a few minutes)...")
    products = pd.read_csv(INSTACART_DIR / "products.csv")
    products = products[products["product_name"].isin(catalog["instacart_match"].dropna())]
    n = len(products)
    position = pd.Series(np.arange(n), index=products["product_id"])
    pair_counts = np.zeros(n * n, dtype=np.int32)       # pair (a, b) stored at a * n + b
    item_counts = np.zeros(n, dtype=np.int64)
    carry, done = None, 0                                # last order of a chunk continues in the next
    reader = pd.read_csv(INSTACART_DIR / "order_products__prior.csv", usecols=["order_id", "product_id"],
                         dtype="int32", chunksize=5_000_000)
    for chunk in reader:
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        last_order = chunk["order_id"].iloc[-1]
        carry = chunk[chunk["order_id"] == last_order]
        chunk = chunk[chunk["order_id"] != last_order]
        idx = chunk["product_id"].map(position)
        keep = idx.notna().to_numpy()
        order_ids, idx = chunk["order_id"].to_numpy()[keep], idx.to_numpy()[keep].astype(np.int64)
        item_counts += np.bincount(idx, minlength=n)
        a, b = order_pairs(order_ids, idx)
        ids, counts = np.unique(a * n + b, return_counts=True)
        pair_counts[ids] += counts.astype(np.int32)
        done += len(chunk)
        print(f"  {done:,} rows done", end="\r")
    if carry is not None and len(carry):
        idx = carry["product_id"].map(position).dropna().to_numpy().astype(np.int64)
        item_counts += np.bincount(idx, minlength=n)
        a, b = order_pairs(np.zeros(len(idx), dtype=np.int32), idx)
        np.add.at(pair_counts, a * n + b, 1)

    orders = pd.read_csv(INSTACART_DIR / "orders.csv", usecols=["eval_set"])
    n_orders = int((orders["eval_set"] == "prior").sum())
    ids = np.flatnonzero(pair_counts >= PAIR_MIN_ORDERS)
    a, b, together = ids // n, ids % n, pair_counts[ids]
    names = products["product_name"].to_numpy()
    both = pd.DataFrame({                                # each pair in both directions
        "product": np.r_[names[a], names[b]],
        "partner": np.r_[names[b], names[a]],
        "together_orders": np.r_[together, together],
        "product_orders": np.r_[item_counts[a], item_counts[b]],
        "partner_orders": np.r_[item_counts[b], item_counts[a]],
    })
    both["p_partner_given_product"] = both["together_orders"] / both["product_orders"]
    both["p_partner"] = both["partner_orders"] / n_orders
    both["lift"] = both["p_partner_given_product"] / both["p_partner"]
    pairs = (both[both["lift"] > 1].sort_values("lift", ascending=False)
             .groupby("product").head(PAIR_TOP_PARTNERS).sort_values(["product", "lift"],
                                                                      ascending=[True, False]))
    pairs = pairs.round({"p_partner_given_product": 5, "p_partner": 6, "lift": 3})
    pairs.to_csv(PAIRS_FILE, index=False)
    print(f"\nSaved {len(pairs):,} product pairs ({pairs['product'].nunique():,} products, "
          f"{n_orders:,} real orders) to {PAIRS_FILE}")
    top = pairs[pairs["together_orders"] >= 1000].nlargest(5, "lift")
    print(top[["product", "partner", "together_orders", "lift"]].to_string(index=False))
    return pairs

# ===================================== product popularity (Instacart)
def build_product_orders():
    """How many times each Instacart product was bought (real counts). Cached in a small file."""
    if PRODUCT_ORDERS_FILE.exists():
        return pd.read_csv(PRODUCT_ORDERS_FILE)
    print("\nCounting Instacart orders per product (one time only, about 30 seconds)...")
    items = pd.read_csv(INSTACART_DIR / "order_products__prior.csv",
                        usecols=["product_id"], dtype={"product_id": "int32"})
    products = pd.read_csv(INSTACART_DIR / "products.csv")
    departments = pd.read_csv(INSTACART_DIR / "departments.csv")
    products = products.merge(departments, on="department_id")
    products["orders"] = products["product_id"].map(items["product_id"].value_counts()).fillna(0).astype(int)
    products = products[["product_id", "product_name", "department", "orders"]]
    products.to_csv(PRODUCT_ORDERS_FILE, index=False)
    return products


def normalize_name(text):
    """'Organic Baby-Spinach, 5oz' -> 'organic baby spinach 5oz' (for name matching)."""
    text = re.sub(r"[^a-z0-9 ]", " ", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()


def add_popularity(catalog):
    """Give every product a popularity weight.
    1) Match its name to an Instacart product (same department) and borrow that product's
       real order count. If several catalog products match the same Instacart product,
       its orders are split between them.
    2) Products without a safe match get the median weight of their CPI category (estimate)."""
    orders = build_product_orders()

    # Character n-grams make "Bran Flakes" and "Bran Flakes Cereal" look similar
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2)
    ic_matrix = vectorizer.fit_transform(orders["product_name"].map(normalize_name))
    cat_matrix = vectorizer.transform(catalog["product_name"].map(normalize_name))
    finder = NearestNeighbors(n_neighbors=MATCH_CANDIDATES, metric="cosine").fit(ic_matrix)
    distances, indices = finder.kneighbors(cat_matrix)

    # Which Instacart departments are allowed for each CPI code
    allowed = {}
    for dept, codes in DEPARTMENT_TO_CPI.items():
        for code in codes:
            allowed.setdefault(code, set()).add(dept)

    ic_depts = orders["department"].to_numpy()
    match_idx, match_score = [], []
    for row, code in enumerate(catalog["cpi_item_code"]):
        ok_depts = allowed.get(code, set())
        idx, score = -1, 0.0
        for k in range(MATCH_CANDIDATES):   # best candidate in a compatible department
            j, sim = indices[row, k], 1 - distances[row, k]
            if sim < MATCH_MIN_SIMILARITY:
                break
            if code == "SAF11" or ic_depts[j] in ok_depts:  # SAF11 = unknown category
                idx, score = j, sim
                break
        match_idx.append(idx)
        match_score.append(round(float(score), 3))

    catalog["instacart_match"] = [orders["product_name"].iat[j] if j >= 0 else None for j in match_idx]
    catalog["match_score"] = match_score
    matched = pd.Series(match_idx, index=catalog.index) >= 0

    # Real orders (+1 so never-ordered products can still sell), split among duplicates
    ic_orders = pd.Series([orders["orders"].iat[j] if j >= 0 else 0 for j in match_idx], index=catalog.index)
    sharing = pd.Series(match_idx, index=catalog.index).map(pd.Series(match_idx).value_counts())
    catalog["popularity"] = ((ic_orders + 1) / sharing).where(matched)

    # Estimate for unmatched products: median of matched products in the same CPI category
    category_median = catalog.groupby("cpi_item_code")["popularity"].transform("median")
    overall_median = catalog["popularity"].median()
    catalog["popularity"] = catalog["popularity"].fillna(category_median).fillna(overall_median).round(2)
    catalog["popularity_source"] = matched.map({True: "instacart_match", False: "category_median"})

    print(f"\nPopularity: {matched.mean() * 100:.0f}% matched to real Instacart products "
          f"(similarity >= {MATCH_MIN_SIMILARITY}), rest estimated from category median")
    top = catalog.nlargest(10, "popularity")
    print(top[["product_name", "instacart_match", "popularity"]].to_string(index=False))
    share_top20 = (catalog["popularity"].sort_values(ascending=False)
                   .head(int(len(catalog) * 0.2)).sum() / catalog["popularity"].sum())
    print(f"Top 20% of products = {share_top20 * 100:.0f}% of expected sales")
    return catalog

def assign_departments(catalog):
    """Give every product exactly ONE store department."""
    orders = build_product_orders()
    instacart_dept = orders.drop_duplicates("product_name").set_index("product_name")["department"]

    def match_rules(text):
        for pattern, dept in DEPARTMENT_RULES:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return dept
        return None

    departments, sources = [], []
    rows = zip(catalog["instacart_match"], catalog["cpi_item_code"],
               catalog["subcategory"], catalog["category"], catalog["product_name"])
    for match, code, sub, cat, name in rows:
        if isinstance(match, str) and match in instacart_dept.index:
            dept, how = instacart_dept[match], "instacart"
        elif code in NON_FOOD_CPI_CODES:                  # personal care / household / pets
            dept, how = CPI_DEFAULT_DEPARTMENT[code], "cpi_default"
        else:
            text = " ".join(t for t in (sub, cat) if isinstance(t, str))
            dept = match_rules(text) if text else None
            how = "category_keywords"
            if dept is None and isinstance(name, str):
                dept, how = match_rules(name), "name_keywords"
            if dept is None:
                dept, how = CPI_DEFAULT_DEPARTMENT.get(code, "missing"), "cpi_default"
        departments.append(dept)
        sources.append(how)

    catalog["department"] = departments
    catalog["department_source"] = sources
    print("\nDepartments assigned by (%):",
          (catalog["department_source"].value_counts(normalize=True) * 100).round(1).to_dict())
    return catalog

# ======================================= aisles = substitute groups (Instacart)
AISLE_NEIGHBORS = 10   # unmatched products: the 10 most similar Instacart names vote on the aisle
# Instacart departments our catalog does not really have: name matching put human food there
# (e.g. dried squid snacks -> "dog food"). Products are only voted into the other departments.
EXCLUDED_AISLE_DEPARTMENTS = {"pets", "babies", "other", "bulk", "missing"}


def add_aisles(catalog):
    """Give every product an Instacart AISLE (134 clean groups like 'soft drinks', 'chips pretzels')
    and set its department from that aisle. Aisles are the substitute groups the simulator uses.
    1) Products already name-matched to Instacart get that product's real aisle.
    2) The rest: the 10 most similar Instacart product names vote (similarity-weighted).
       Tested on matched products with their match hidden: right department 82% of the time
       (the old keyword rules: 56%). So these stay ESTIMATED, with a confidence score."""
    products = pd.read_csv(INSTACART_DIR / "products.csv")
    products = (products.merge(pd.read_csv(INSTACART_DIR / "aisles.csv"), on="aisle_id")
                .merge(pd.read_csv(INSTACART_DIR / "departments.csv"), on="department_id"))
    products = products[~products["department"].isin(EXCLUDED_AISLE_DEPARTMENTS)]
    products = products.drop_duplicates("product_name")
    aisle_of = products.set_index("product_name")["aisle"]   # matches in excluded departments get voted
    dept_of_aisle = products.drop_duplicates("aisle").set_index("aisle")["department"]

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2)
    ic_matrix = vectorizer.fit_transform(products["product_name"].map(normalize_name))
    text = (catalog["product_name"].fillna("") + " " + catalog["subcategory"].fillna("")).map(normalize_name)
    finder = NearestNeighbors(n_neighbors=AISLE_NEIGHBORS, metric="cosine").fit(ic_matrix)
    distances, indices = finder.kneighbors(vectorizer.transform(text))
    ic_aisles = products["aisle"].to_numpy()

    aisles, confidence, sources = [], [], []
    for row, match in enumerate(catalog["instacart_match"]):
        if isinstance(match, str) and match in aisle_of.index:
            aisles.append(aisle_of[match]); confidence.append(1.0); sources.append("instacart_match")
            continue
        votes = {}
        for k in range(AISLE_NEIGHBORS):
            aisle = ic_aisles[indices[row, k]]
            votes[aisle] = votes.get(aisle, 0.0) + (1 - distances[row, k])
        best = max(votes, key=votes.get)
        total = sum(votes.values())
        aisles.append(best)
        confidence.append(round(votes[best] / total, 3) if total > 0 else 0.0)
        sources.append("name_similarity")

    catalog["aisle"] = aisles
    catalog["aisle_confidence"] = confidence
    catalog["aisle_source"] = sources
    catalog["department"] = catalog["aisle"].map(dept_of_aisle)
    catalog["department_source"] = np.where(catalog["aisle_source"] == "instacart_match",
                                            "instacart", "aisle_similarity")
    print(f"\nAisles: {catalog['aisle'].nunique()} used | "
          + ", ".join(f"{k}: {v * 100:.0f}%" for k, v in catalog["aisle_source"].value_counts(normalize=True).items())
          + f" | median confidence of estimated: "
          f"{catalog.loc[catalog['aisle_source'] == 'name_similarity', 'aisle_confidence'].median():.2f}")
    return catalog


# ============================================================ data labels
# RETINA rule: never present estimated data as real. Every dataset and catalog field gets one:
#   REAL_OBSERVED    - recorded directly (a shelf price seen in a store, a product's name)
#   PUBLIC_STATISTIC - a published figure from an official body or study (CPI, survey results)
#   ESTIMATED        - worked out by us (name matching, keyword rules, assumptions)
#   SIMULATED        - produced by a simulation or a demo website
SOURCE_LABELS = {
    "Open Prices": "REAL_OBSERVED",
    "Open Food Facts": "REAL_OBSERVED",
    "Open Beauty Facts": "REAL_OBSERVED",
    "Open Products Facts": "REAL_OBSERVED",
    "US CPI (Bureau of Labor Statistics)": "PUBLIC_STATISTIC",
    "Instacart Online Grocery Shopping Dataset 2017": "REAL_OBSERVED",   # used only as aggregates
    "UCI Online Retail II": "REAL_OBSERVED",
    "India CPI, base 2024 (MoSPI)": "PUBLIC_STATISTIC",
    "webscraper.io e-commerce test site": "SIMULATED",                  # a demo shop, fake products
}


def add_data_labels(catalog):
    """One label column per group of catalog fields, saying where those values come from."""
    catalog = catalog.copy()
    catalog["price_label"] = "REAL_OBSERVED"          # prices people recorded in real US stores
    catalog["product_facts_label"] = "REAL_OBSERVED"  # names, brands, categories (Open Prices / Facts)
    catalog["popularity_label"] = "ESTIMATED"         # borrowed from a name-matched Instacart product,
                                                      # or a category median (see popularity_source)
    catalog["department_label"] = "ESTIMATED"         # name matching / keyword rules (department_source)
    catalog["cpi_category_label"] = "ESTIMATED"       # keyword rules map products to BLS categories
    catalog["aisle_label"] = "ESTIMATED"              # Instacart aisle via name match / similar names
    
    return catalog

# ==================================================== data sources file


def write_data_sources():
    """Record every dataset's source, licence, URL and download date in one file."""
    entries = []
    for src in DATA_SOURCES:
        paths = [DATA_DIR / f for f in src["files"]]
        found = [p for p in paths if p.exists()]
        downloaded = (datetime.fromtimestamp(min(p.stat().st_mtime for p in found)).date().isoformat()
                      if found else None)
        entries.append({**src, "data_label": SOURCE_LABELS.get(src["name"], "ESTIMATED"),
                        "downloaded": downloaded, "files_present": len(found) == len(paths)})    
    with open(DATA_SOURCES_FILE, "w", encoding="utf-8") as f:
        json.dump({"written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "sources": entries}, f, indent=2)
    print(f"\nData sources recorded: {len(entries)} -> {DATA_SOURCES_FILE.name}")


def print_data_summary():
    """Final checklist of RETINA's data layer."""
    files = [CATALOG_FILE, US_CPI_FILE, SHOPPER_PARAMS_FILE, DEPARTMENTS_FILE,
             DATA_SOURCES_FILE, TRANSACTIONS_FILE, CPI_FILE]
    print("\n========== RETINA DATA LAYER ==========")
    for f in files:
        status = f"{f.stat().st_size / 1e6:7.1f} MB" if f.exists() else "  MISSING"
        print(f"  {f.name:<24} {status}")
    print("========================================")


# ================================================================ main
def main():
    # Reference datasets (not used by RETINA Mart, kept and cached)
    if TRANSACTIONS_FILE.exists():
        print("UCI transactions already processed:", TRANSACTIONS_FILE)
    else:
        build_transactions()
    build_cpi()

    # RETINA Mart inputs
    us_cpi = build_us_cpi()

    prices = load_open_prices()
    details = load_product_details(local_barcodes(prices))
    catalog = build_catalog(prices, details)
    catalog = add_cpi_mapping(catalog, us_cpi)
    catalog = clean_catalog(catalog)
    catalog = add_popularity(catalog)
    catalog = assign_departments(catalog)
    catalog = add_aisles(catalog)
    catalog = add_data_labels(catalog)
    CATALOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(CATALOG_FILE, index=False)
    summarize_catalog(catalog)
    summarize_mapping(catalog)
    print("\nSaved to:", CATALOG_FILE)

    params = build_shopper_params()
    build_departments(catalog, params)
    build_product_pairs(catalog)
    write_data_sources()
    print_data_summary()


if __name__ == "__main__":
    main()
