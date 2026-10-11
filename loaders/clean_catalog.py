"""RETINA catalog clean-up: every product a real, correctly placed supermarket item.

Reads data/processed/catalog_us.csv (built by loader.py) and the real product facts we collected
(Open Food Facts: pack size and category; Open Prices: price records and receipts) and writes
data/processed/catalog_us_clean.csv - the catalog the simulator uses.

1. Category check. A product borrows its popularity and department from the Instacart product
   its NAME matched. The name can mislead: Melona "Banana" is an ice-cream bar, not a fresh
   banana. When the product's real category (Open Food Facts), or the usual category of its
   brand, points to a different department, the match is dropped: popularity falls back to an
   estimate and the department and aisle follow the real category.
2. Pack size from Open Food Facts goes into the name ("Diet Coke, 12 fl oz"): a single can and a
   12-pack are different products at different prices.
3. Exact duplicates (same name, brand and pack size) are merged into one product.
4. Names not written in English letters (230 products from Asian grocery stores) are left out
   of the US generic store.
5. Promotions: clearance prices ("expires soon") are not promotions, and discounts deeper than
   real promotions (Open Prices: 95% of discounts are 50% or less) are capped at 50%.
Also writes data/processed/real_shopping_stats.json (units per receipt line, discount depths
from real Open Prices receipts and price tags) for the simulator's scorecard.

Run:  python loaders\\clean_catalog.py
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

DATA = Path(__file__).resolve().parent.parent / "data"
PROCESSED, RAW = DATA / "processed", DATA / "raw"
CATALOG_IN = PROCESSED / "catalog_us.csv"
CATALOG_OUT = PROCESSED / "catalog_us_clean.csv"
STATS_OUT = PROCESSED / "real_shopping_stats.json"
FACT_FILES = ["off_products_subset.csv", "opf_subset.csv", "obf_subset.csv"]
INSTACART = RAW / "instacart"
OPEN_PRICES = RAW / "prices.parquet"
MAX_DISCOUNT = 50.0            # % - real promotions: 95% are 50% off or less (Open Prices)
NOT_PROMOTIONS = {"EXPIRES_SOON", "SECOND_HAND"}
OFF_DUMP = RAW / "en.openfoodfacts.org.products.csv.gz"     # the full Open Food Facts export
BRAND_KINDS = PROCESSED / "off_brand_kinds.csv"              # cache: each brand's usual product kind
BRAND_MIN_PRODUCTS, BRAND_MIN_SHARE = 2, 0.8   # brand evidence: 2+ categorised products, 80% one kind
AISLE_NEIGHBORS = 10

# Real Open Food Facts categories that leave NO doubt about what a product is, the department(s)
# such a product may sit in, and where it goes otherwise. Only these override a name match:
# product categories are noisy (olives filed as "salted snacks"), so softer hints are ignored.
STRONG_KINDS = [
    ("ice cream",   r"ice creams?\b|frozen desserts|ice pops?|popsicles|sorbets?|gelatos?|frozen yogh?urts?",
     {"frozen"}, "frozen"),
    ("frozen food", r"frozen foods|frozen meals|frozen pizzas|frozen vegetables|frozen fruits|frozen fries",
     {"frozen"}, "frozen"),
    ("soft drink",  r"carbonated drinks|\bsodas\b|soft drinks|energy drinks|sports drinks|fruit juices"
                    r"|juices and nectars|bottled waters|mineral waters|spring waters|iced teas|colas",
     {"beverages"}, "beverages"),
    ("alcohol",     r"(?<!ginger )(?<!root )\bbeers\b|\bwines\b|spirits|liqueurs|hard ciders|(?<!non-)alcoholic beverages",
     {"alcohol"}, "alcohol"),
    ("cereal",      r"breakfast cereals", {"breakfast"}, "breakfast"),
    ("sweets/chips", r"\bcandies\b|confectioneries|chocolate candies|bonbons|gummies|chewing gum"
                     r"|chips and fries|\bcrisps\b|potato crisps|tortilla chips|popcorn|pretzels",
     {"snacks", "international"}, "snacks"),
    ("bread",       r"\bbreads\b|sliced breads|\bbagels\b|\btortillas\b|\bbaguettes\b",
     {"bakery", "frozen", "international"}, "bakery"),
    ("pet food",    r"dog foods?|cat foods?|pet foods?", set(), None),     # not sold in this store
]


def product_kind(categories):
    """(kind, allowed departments, target department) from real categories, or None."""
    if not isinstance(categories, str):
        return None
    text = categories.lower()
    for kind, pattern, allowed, target in STRONG_KINDS:
        if re.search(pattern, text):
            return kind, allowed, target
    return None


def normalize(text):
    text = re.sub(r"[^a-z0-9 ]", " ", str(text).lower())
    return re.sub(r"\s+", " ", text).strip()


def clean_size(text):
    """'9.75 OZ (276g)' -> '9.75 oz'; keeps short, readable pack sizes only."""
    if not isinstance(text, str):
        return None
    t = re.sub(r"\s+", " ", text.strip())
    t = re.split(r"[(\[]", t)[0].strip().rstrip(",;")
    if not re.search(r"\d", t) or len(t) > 24:
        return None
    t = t.lower()
    return f"{t} ct" if re.fullmatch(r"\d+", t) else t     # a bare number is a count: '4' -> '4 ct'


# ================================================================ steps
def load_facts():
    facts = pd.concat([pd.read_csv(PROCESSED / f, dtype=str) for f in FACT_FILES if (PROCESSED / f).exists()])
    return facts.drop_duplicates("code").set_index("code")[["quantity", "brands", "categories_en"]]


def instacart_products():
    p = pd.read_csv(INSTACART / "products.csv")
    p = p.merge(pd.read_csv(INSTACART / "aisles.csv"), on="aisle_id").merge(
        pd.read_csv(INSTACART / "departments.csv"), on="department_id")
    return p.drop_duplicates("product_name")


def brand_kinds(brands):
    """Each brand's usual kind of product, from ALL its products in Open Food Facts (one-time scan
    of the full export, a few minutes; saved to off_brand_kinds.csv)."""
    if BRAND_KINDS.exists():
        return pd.read_csv(BRAND_KINDS, dtype={"brand": str}, keep_default_na=False)
    if not OFF_DUMP.exists():
        print(f"   (no {OFF_DUMP.name}: brand evidence skipped)")
        return pd.DataFrame(columns=["brand", "kind", "kind_products", "categorised_products"])
    import gzip
    print(f"   scanning {OFF_DUMP.name} for brand categories (one time, a few minutes)...")
    total, kinds, n = {}, {}, 0
    with gzip.open(OFF_DUMP, "rb") as f:
        head = f.readline().decode().rstrip("\n").split("\t")
        ib, ic = head.index("brands"), head.index("categories_en")
        last = max(ib, ic)
        for line in f:
            n += 1
            if n % 1_000_000 == 0:
                print(f"   {n:,} products read", end="\r")
            parts = line.split(b"\t", last + 1)
            if len(parts) <= last or not parts[ib] or not parts[ic]:
                continue
            for b in parts[ib].decode("utf-8", "replace").lower().split(",")[:2]:
                b = b.strip()
                if b in brands:
                    total[b] = total.get(b, 0) + 1
                    k = product_kind(parts[ic].decode("utf-8", "replace"))
                    if k:
                        kinds.setdefault(b, {}).setdefault(k[0], 0)
                        kinds[b][k[0]] += 1
    rows = []
    for b, t in total.items():
        k, c = max(kinds.get(b, {"": 0}).items(), key=lambda kv: kv[1])
        rows.append((b, k, c, t))
    table = pd.DataFrame(rows, columns=["brand", "kind", "kind_products", "categorised_products"])
    table.to_csv(BRAND_KINDS, index=False)
    print(f"\n   {n:,} Open Food Facts products read, {len(table):,} of our brands found")
    return table


def check_categories(cat, facts, ic):
    """Find products whose real category (or, without one, their brand's) shows they cannot be
    in the department their name match gave them - ice cream filed as fresh fruit, soda as
    produce - and move them to the right department."""
    cat["real_category"] = cat["product_id"].map(facts["categories_en"])
    kinds = cat["real_category"].map(product_kind)
    # Brand evidence for products without a category: most of the brand's products in Open Food
    # Facts are one strong kind (every Melona product is a frozen dessert)
    brand = cat["brand"].fillna("").str.lower().str.strip()
    table = brand_kinds(set(brand[brand != ""]))
    ok = (table["kind"] != "") & (table["categorised_products"] >= BRAND_MIN_PRODUCTS) & \
         (table["kind_products"] >= BRAND_MIN_SHARE * table["categorised_products"])
    brand_kind = dict(zip(table.loc[ok, "brand"], table.loc[ok, "kind"]))
    lookup = {k: (k, a, t) for k, _, a, t in STRONG_KINDS}
    no_cat = cat["real_category"].isna()
    kinds = kinds.where(~no_cat, brand.map(lambda b: lookup.get(brand_kind.get(b))))
    how = np.where(no_cat, "brand_category", "real_category")
    wrong = kinds.map(lambda k: k is not None) & pd.Series(
        [k is not None and d not in k[1] for k, d in zip(kinds, cat["department"])], index=cat.index)
    target = kinds.map(lambda k: k[2] if k else None)
    print(f"\n1. Category check: {cat['real_category'].notna().mean():.0%} of products have a real category")
    print(f"   {wrong.sum():,} products cannot be in their department "
          f"({(wrong & cat['instacart_match'].notna()).sum():,} borrowed a wrong Instacart match):")
    flows = pd.DataFrame({"was": cat["department"], "kind": kinds.map(lambda k: k[0] if k else None)})[wrong]
    print("   " + " | ".join(f"{k} in {w}: {n}" for (w, k), n in flows.value_counts().head(8).items()))
    for r in cat[wrong].nlargest(6, "popularity").itertuples():
        print(f"     {str(r.product_name)[:26]:<26} {str(r.brand)[:12]:<12} was {r.department:<10} "
              f"-> {target[r.Index] or 'removed'} ({kinds[r.Index][0]}, {how[cat.index.get_loc(r.Index)]})")

    # Pet food: not sold here. Others: drop the wrong match, new department and aisle
    pets = wrong & target.isna()
    fix = cat.index[wrong & ~pets]
    old_match = cat.loc[fix, "instacart_match"].notna().to_numpy()
    cat.loc[fix, "department"] = target[fix]
    cat.loc[fix, "department_source"] = pd.Series(how, index=cat.index)[fix]
    cat.loc[fix[old_match], "instacart_match"] = None
    cat.loc[fix[old_match], "match_score"] = 0.0
    assign_aisles(cat, fix, ic)
    est = cat["popularity_source"] == "category_median"
    dept_median = cat[est].groupby("department")["popularity"].median()
    dropped = fix[old_match]
    cat.loc[dropped, "popularity"] = cat.loc[dropped, "department"].map(dept_median).fillna(cat.loc[est, "popularity"].median())
    cat.loc[dropped, "popularity_source"] = "category_median"
    cat["department_check"] = np.where(cat.index.isin(fix), "moved_by_" + pd.Series(how, index=cat.index),
                                       np.where(kinds.notna(), "confirmed", "no_strong_evidence"))
    print(f"   moved {len(fix):,}, pet food removed {pets.sum():,}")
    return cat[~pets].copy()


def assign_aisles(cat, rows, ic):
    """Aisle for moved products: the most similar Instacart products IN THE NEW DEPARTMENT vote."""
    for dept, idx in cat.loc[rows].groupby("department").groups.items():
        pool = ic[ic["department"] == dept]
        if pool.empty:
            continue
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=1)
        m = vec.fit_transform(pool["product_name"].map(normalize))
        k = min(AISLE_NEIGHBORS, len(pool))
        finder = NearestNeighbors(n_neighbors=k, metric="cosine").fit(m)
        text = (cat.loc[idx, "product_name"].fillna("") + " " + cat.loc[idx, "real_category"].fillna("")).map(normalize)
        dist, ind = finder.kneighbors(vec.transform(text))
        aisles = pool["aisle"].to_numpy()
        for row, d, i in zip(idx, dist, ind):
            votes = {}
            for dd, ii in zip(d, i):
                votes[aisles[ii]] = votes.get(aisles[ii], 0) + (1 - dd)
            best = max(votes, key=votes.get)
            cat.at[row, "aisle"] = best
            cat.at[row, "aisle_confidence"] = round(votes[best] / max(sum(votes.values()), 1e-9), 3)
            cat.at[row, "aisle_source"] = "real_category"


def add_pack_sizes(cat, facts):
    size = cat["product_id"].map(facts["quantity"]).map(clean_size)
    has = size.notna() & ~cat.apply(lambda r: isinstance(size[r.name], str)
                                    and size[r.name].split()[0] in str(r.product_name).lower(), axis=1)
    cat["pack_size"] = size
    cat.loc[has, "product_name"] = cat.loc[has, "product_name"].str.strip() + ", " + size[has]
    print(f"\n2. Pack sizes: {size.notna().mean():.0%} of products have a real pack size "
          f"(e.g. {', '.join(cat.loc[has, 'product_name'].head(3))})")
    return cat


def merge_duplicates(cat):
    key = cat["product_name"].map(normalize) + "|" + cat["brand"].fillna("").map(normalize)
    cat = cat.assign(_key=key).sort_values("price_date", ascending=False)
    popularity = cat.groupby("_key")["popularity"].sum()        # matched duplicates had split it
    n_before = len(cat)
    cat = cat.drop_duplicates("_key").copy()
    cat["popularity"] = cat["_key"].map(popularity).round(2)
    print(f"\n3. Exact duplicates merged: {n_before - len(cat):,} rows (same name, brand and pack size)")
    return cat.drop(columns="_key").sort_index()


def drop_unreadable(cat):
    bad = cat["product_name"].fillna("").str.contains(r"[^\x00-\x7FÀ-ɏ’–—®™]")
    print(f"\n4. Names not in English letters left out of the US store: {bad.sum():,}")
    return cat[~bad].copy()


def fix_promotions(cat):
    disc = pd.to_numeric(cat["discount_percentage"], errors="coerce").fillna(0)
    regular = pd.to_numeric(cat["original_price"], errors="coerce")
    clearance = cat["discount_type"].isin(NOT_PROMOTIONS) & (disc > 0)
    too_deep = (disc > MAX_DISCOUNT) & ~clearance
    disc = disc.where(~clearance, 0.0).clip(upper=MAX_DISCOUNT)
    cat["discount_percentage"] = disc.round(1)
    cat["price"] = (regular * (1 - disc / 100)).round(2)
    cat["promo_label"] = np.where(too_deep, f"ESTIMATED - observed discount capped at {MAX_DISCOUNT:.0f}%",
                                  np.where(clearance, "REAL_OBSERVED clearance - not used as a promotion",
                                           "REAL_OBSERVED"))
    print(f"\n5. Promotions: {clearance.sum():,} clearance prices removed, {too_deep.sum():,} discounts "
          f"capped at {MAX_DISCOUNT:.0f}% | promo-eligible products: {(disc > 0).sum():,}")
    return cat


def real_shopping_stats():
    """Units per receipt line and discount depths, from real Open Prices records."""
    p = pd.read_parquet(OPEN_PRICES)
    p["price"] = pd.to_numeric(p["price"].astype(str), errors="coerce")
    p["was"] = pd.to_numeric(p["price_without_discount"].astype(str), errors="coerce")
    r = p[(p["proof_type"] == "RECEIPT") & (p["type"] == "PRODUCT") & p["receipt_quantity"].notna()]
    q = r["receipt_quantity"]
    r = r[(q - q.round()).abs() < 1e-6]

    def units(df):
        u = df["receipt_quantity"].round().astype(int).clip(upper=5)
        return {"lines": int(len(df)), "receipts": int(df["proof_id"].nunique()),
                "share": {str(k): round(float(v), 4) for k, v in u.value_counts(normalize=True).sort_index().items()},
                "mean": round(float(df["receipt_quantity"].mean()), 3)}

    d = p[(p["price_is_discounted"] == True) & (p["was"] > p["price"])]
    depth = lambda df: {str(k): round(float(v), 3) for k, v in
                        (1 - df["price"] / df["was"]).quantile([.1, .25, .5, .75, .9, .95]).items()}
    stats = {"source": "Open Prices (open-prices.openfoodfacts.org) - REAL_OBSERVED receipts and price tags",
             "units_per_receipt_line": {"all": units(r), "usd": units(r[r["currency"] == "USD"])},
             "discount_depth": {"all": depth(d), "usd": depth(d[d["currency"] == "USD"]),
                                "records": int(len(d))}}
    STATS_OUT.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    u = stats["units_per_receipt_line"]
    print(f"\nReal shopping stats saved ({STATS_OUT.name}): 1 unit per receipt line "
          f"{u['all']['share']['1']:.0%} (US {u['usd']['share']['1']:.0%}), median discount "
          f"{stats['discount_depth']['all']['0.5']:.0%} (US {stats['discount_depth']['usd']['0.5']:.0%})")


def main():
    cat = pd.read_csv(CATALOG_IN, dtype={"product_id": str})
    print(f"Catalog: {len(cat):,} products ({CATALOG_IN.name})")
    facts, ic = load_facts(), instacart_products()
    cat = check_categories(cat, facts, ic)
    cat = add_pack_sizes(cat, facts)
    cat = merge_duplicates(cat)
    cat = drop_unreadable(cat)
    cat = fix_promotions(cat)
    cat.to_csv(CATALOG_OUT, index=False)
    print(f"\nSaved {len(cat):,} products to {CATALOG_OUT.name}")
    real_shopping_stats()


if __name__ == "__main__":
    main()
