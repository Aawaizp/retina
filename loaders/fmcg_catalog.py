"""RETINA FMCG catalog: the product range of the generic RETINA Supercenter.

Real products of real US brands, in the right aisles, with real prices where they exist:

  fresh food    produce, meat & seafood, bakery, deli - from data/processed/catalog_us_clean.csv (real
                Open Prices prices), minus retailer own-brands and minus packaged items that were placed
                in a fresh aisle by a name match (a frozen "diced avocado" is not a fresh avocado)
  packaged food Open Food Facts US products of established brands (20+ US products in Open Food Facts),
                with their real names, sizes and categories; the aisle is predicted by a classifier
                trained on the 14,000 products whose aisle we already know (accuracy printed on a
                held-out part); low-confidence predictions are left out
  household,    Open Products Facts / Open Beauty Facts (and Open Food Facts for baby and pet food) US
  personal care, products, placed in aisles by clear category words (laundry, toothpaste, diapers ...)
  baby, pets
  no retailer   Trader Joe's, Kirkland, Great Value, Whole Foods 365, ... are left out: RETINA stores
  own-brands    are generic and never copy a real retailer

Prices: REAL_OBSERVED when the product has Open Prices records (USD); otherwise ESTIMATED = the median
price per ounce / fluid ounce / count of real-priced products in the same aisle x the product's size
(price_basis says how many real prices the estimate rests on).
Popularity inside an aisle (which product a shopper picks; how much the AISLE sells comes from real
Instacart shares and does not change here): fresh food keeps its Instacart-based popularity; other
products get an ESTIMATED weight from the brand's presence (number of its US products in Open Food
Facts); where a PUBLISHED brand share exists (data/reference/brand_shares.csv) the brand gets exactly
that share of its aisle (PUBLIC_STATISTIC, source in popularity_source).

Run:  python loaders\\fmcg_catalog.py          (first run also extracts the US products from the raw
                                                 Open Food / Products / Beauty Facts exports: ~10 min)
Writes data/processed/catalog_fmcg.csv, store_departments_fmcg.json, catalog_fmcg_report.json.
"""

import hashlib
import json
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from retina.products.aisles import aisle_table          # noqa: E402
from retina.products.packsize import parse_pack         # noqa: E402

RAW, PROCESSED = ROOT / "data" / "raw", ROOT / "data" / "processed"
CATALOG_IN = PROCESSED / "catalog_us_clean.csv"
CATALOG_OUT = PROCESSED / "catalog_fmcg.csv"
DEPTS_IN, DEPTS_OUT = PROCESSED / "store_departments.json", PROCESSED / "store_departments_fmcg.json"
REPORT_OUT = PROCESSED / "catalog_fmcg_report.json"
SHARES = ROOT / "data" / "reference" / "brand_shares.csv"
OFF_US = PROCESSED / "off_us_products.parquet"
NONFOOD_US = PROCESSED / "opf_obf_us_products.parquet"
OPEN_PRICES = RAW / "prices.parquet"
AISLE_STATS = PROCESSED / "instacart_aisle_stats.csv"

FRESH_DEPTS = {"produce", "meat seafood", "bakery", "deli", "alcohol"}   # from the old catalog (real prices)
FRESH_AISLES = {"fresh fruits", "fresh vegetables", "packaged produce", "fresh herbs"}
BRAND_MIN_PRODUCTS = 20          # an "established" brand: 20+ US products in Open Food Facts
PER_AISLE = 200                  # packaged products per aisle (the most present brands first)
PER_BRAND = 15                   # ... but at most this many of one brand: an aisle keeps its real variety
MIN_CONFIDENCE = 0.5             # classifier probability needed to place a product in an aisle
MIN_SIZE_POINTS = 8              # real prices (with a size) needed to fit an aisle's price-size curve
COLUMNS = ["code", "product_name", "brands", "brand_owner", "quantity", "categories_en", "countries_tags",
           "unique_scans_n"]

OWN_BRANDS = [
    r"trader joe", r"kirkland", r"\b365\b", r"whole foods", r"signature (select|kitchens|farms|cafe)",
    r"^o organics", r"^open nature", r"^lucerne", r"great value", r"^equate", r"^marketside", r"sam'?s choice",
    r"member'?s mark", r"parent'?s choice", r"^kroger", r"simple truth", r"private selection", r"good ?& ?gather",
    r"market pantry", r"^up ?& ?up", r"^favorite day", r"^wegmans", r"^publix", r"^h-?e-?b\b", r"hill country fare",
    r"^aldi", r"^clancy'?s", r"^simply nature", r"^specially selected", r"^millville", r"^baker'?s corner",
    r"^friendly farms", r"^countryside creamery", r"^happy farms", r"^happy belly", r"^amazon", r"^safeway",
    r"waterfront bistro", r"^food lion", r"^meijer", r"^hy-?vee", r"^stop ?& ?shop", r"^shoprite", r"bowl ?& ?basket",
    r"wholesome pantry", r"^lidl", r"^fit ?& ?active", r"^great taste", r"^first street", r"^western family",
    r"^essential everyday", r"^food club", r"^best choice", r"^our family", r"^weis\b", r"^harris teeter",
    r"^giant\b", r"^h ?mart", r"^winco", r"^sprouts", r"^natural grocers", r"^fresh market", r"^tops\b",
    r"^raley'?s", r"^schnucks", r"^bettergoods", r"^trader darwin", r"^mainstays", r"^ol'? roy", r"^special kitty",
    r"^spring valley", r"^equaline", r"^cvs", r"^walgreens", r"^rite aid", r"^target", r"^kirkland",
    r"^365 everyday", r"^nature'?s promise", r"^wild harvest", r"^full circle", r"^shurfine", r"^valu time",
    r"^hannaford", r"^market basket", r"^clover valley", r"^dollar general", r"^family dollar", r"^smart ?& ?final",
    r"^great choice", r"^always save", r"^food 4 less", r"^ralphs", r"^vons", r"^albertsons", r"^jewel",
    r"^acme\b", r"^shaw'?s", r"^price chopper", r"^market 32", r"^lowes foods", r"^ingles", r"^save mart",
    r"^stater bros", r"^winn-?dixie", r"^southern grove", r"^bi-?lo", r"^heb\b", r"^central market",
    r"^food town", r"^big y\b", r"^price rite", r"^refreshe", r"^nice!?$", r"^food lion", r"^shop ?rite",
    r"^best yet", r"^iga\b", r"^kroger", r"^p\$\$t", r"^sunny select", r"^lady lee", r"^always fresh",
    r"^culinary circle", r"^our specialty", r"^taste of inspirations", r"^fresh & easy", r"^save-?a-?lot",
]
OWN_RE = re.compile("|".join(OWN_BRANDS), re.IGNORECASE)

# Non-food aisles by clear words in the product's category or name (first match wins: specific first)
RULES = [
    ("diapers wipes", r"\bdiapers?\b|nappies|baby wipes|training pants|pull-?ups"),
    ("baby food formula", r"baby foods?\b|infant formula|baby formula|baby cereal|for babies|toddler (snack|food)"),
    ("baby bath body care", r"baby (shampoo|wash|lotion|bath|oil|powder)"),
    ("dog food care", r"\bdog[- ]foods?\b|dog treats|for dogs\b|\bdog (shampoo|wash|care)|\bpaws?\b.*\bdog"),
    ("cat food care", r"\bcat[- ]foods?\b|cat treats|cat litter|for cats\b"),
    ("laundry", r"laundry|fabric (softener|conditioner)|dryer sheets|stain remover|washing powder|washing liquid"),
    ("dish detergents", r"dishwash|dish ?soap|dishwasher|washing-up"),
    ("paper goods", r"toilet (paper|tissue)|bath(room)? tissue|paper towels?|facial tissues?|paper napkins"),
    ("trash bags liners", r"trash bags?|garbage bags?|bin liners?"),
    ("food storage", r"food storage|storage bags|sandwich bags|zipper bags|aluminum foil|plastic wrap|cling film"),
    ("air fresheners candles", r"air fresheners?|candles?"),
    ("cleaning products", r"cleaners?\b|cleaning|disinfect|bleach|wipes for surfaces|surface spray"),
    ("oral hygiene", r"toothpastes?|toothbrush|mouthwash|dental floss|oral care|whitening strips"),
    ("deodorants", r"deodorants?|antiperspirants?"),
    ("shave needs", r"shav(e|ing)|razors?\b"),
    ("hair care", r"shampoos?|conditioners?|hair (spray|gel|care|styling|color|colour|mask|oil)"),
    ("body lotions soap", r"body (wash|lotion|cream|butter)|shower gels?"),
    ("soap", r"\bsoaps?\b|hand wash|hand soap"),
    ("feminine care", r"tampons?|sanitary (pads?|towels?)|feminine|panty ?liners?"),
    ("facial care", r"face (wash|cream|mask|serum)|facial|cleansers?\b|micellar"),
    ("skin care", r"skin ?care|moisturi[sz]er|sunscreens?|sun protection|lip balm"),
    ("beauty", r"make-?up|lipsticks?|mascara|nail polish|foundation|eyeliner|eye ?shadow|cosmetics?"),
    ("first aid", r"bandages?|first aid|antiseptic"),
    ("cold flu allergy", r"\bcold\b|\bflu\b|allerg|cough"),
    ("muscles joints pain relief", r"pain relie|ibuprofen|acetaminophen|aspirin"),
    ("vitamins supplements", r"vitamins?|supplements?|multivitamin"),
]
NONFOOD_DEPTS = {"household", "personal care", "babies", "pets"}
FOOD_RULE_AISLES = {"diapers wipes", "baby food formula", "dog food care", "cat food care"}
FOOD_RULES = [r for r in RULES if r[0] in FOOD_RULE_AISLES]   # the only rules used on FOOD products
                                                               # (a "cold" rule would catch cold brew coffee)


# ================================================================= extract (cached)
def extract(files, out, say):
    if out.exists():
        return pd.read_parquet(out)
    parts = []
    for name in files:
        say(f"  reading {name} (US products only)...")
        head = pd.read_csv(RAW / name, sep="\t", nrows=0, quoting=3).columns
        cols = [c for c in COLUMNS if c in head]
        for chunk in pd.read_csv(RAW / name, sep="\t", usecols=cols, dtype=str, chunksize=200_000,
                                 on_bad_lines="skip", quoting=3, encoding_errors="replace"):
            us = chunk[chunk["countries_tags"].fillna("").str.contains("united-states")]
            parts.append(us.assign(source=name.split(".")[1]))
    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(out, index=False)
    return df


# ================================================================= helpers
def first_brand(text):
    return str(text).split(",")[0].strip() if isinstance(text, str) and text.strip() else ""


def own_brand(brand):
    return bool(OWN_RE.search(brand)) if brand else False


USDA_UNITS = [(r"\bOZA\b|\bONZ\b", "oz"), (r"\bFOZ\b", "fl oz"), (r"\bGRM\b", "g"), (r"\bMLT\b", "ml"),
              (r"\bLTR\b", "l"), (r"\bLBR\b", "lb"), (r"\bKGM\b", "kg"), (r"\b(EA|EACH|CT|PCS?)\b", "ct")]


def usda_units(text):
    """USDA unit codes ('12 OZA', '355 MLT') -> words our size parser knows ('12 oz', '355 ml')."""
    if not isinstance(text, str):
        return text
    for code, word in USDA_UNITS:
        text = re.sub(code, word, text, flags=re.IGNORECASE)
    return text


def unit_amount(pack):
    """(kind, amount) for price per unit: weight in oz, volume in fl oz, count in ct."""
    if not pack:
        return None, None
    u, a = pack["unit"], pack["amount"] * pack["count"]
    to_oz = {"oz": 1, "lb": 16, "g": 1 / 28.3495, "kg": 35.274}
    to_floz = {"fl oz": 1, "ml": 1 / 29.5735, "l": 33.814, "gal": 128, "qt": 32, "pt": 16}
    if u in to_oz:
        return "oz", a * to_oz[u]
    if u in to_floz:
        return "fl oz", a * to_floz[u]
    if u == "ct":
        return "ct", a
    return None, None


def display_name(brand, name, pack):
    """A till-receipt style name: brand first (if missing), size last (if missing)."""
    name = re.sub(r"\s+", " ", str(name)).strip()
    key = re.sub(r"[^a-z0-9]", "", brand.lower())
    if brand and key and key[:6] not in re.sub(r"[^a-z0-9]", "", name.lower()):
        name = f"{brand} {name}"
    if pack and pack["text"] and not re.search(r"\d\s*(oz|fl|ml|l\b|g\b|kg|lb|ct|count|pack|pk)", name.lower()):
        name = f"{name}, {pack['text']}"
    return name


def stable_unit(code, salt):
    """A fixed number in [0, 1) from the product code (same catalog every build)."""
    return int(hashlib.md5(f"{salt}:{code}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


# ================================================================= aisles
def train_aisle_classifier(train, say):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    text = (train["categories_en"].fillna("") + " | " + train["product_name"].fillna("")).str.lower()
    x_tr, x_te, y_tr, y_te = train_test_split(text, train["aisle"], test_size=0.2, random_state=7,
                                              stratify=None)
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=80_000, sublinear_tf=True)
    model = LogisticRegression(max_iter=2000, C=8.0)
    model.fit(vec.fit_transform(x_tr), y_tr)
    proba = model.predict_proba(vec.transform(x_te))
    pred = model.classes_[proba.argmax(1)]
    conf = proba.max(1)
    keep = conf >= MIN_CONFIDENCE
    acc_all = float((pred == y_te.to_numpy()).mean())
    acc_kept = float((pred[keep] == y_te.to_numpy()[keep]).mean())
    say(f"  aisle classifier: {len(train):,} known products, held-out accuracy {acc_all:.1%} "
        f"(kept {keep.mean():.0%} with confidence >= {MIN_CONFIDENCE}: accuracy {acc_kept:.1%})")
    model.fit(vec.fit_transform(text), train["aisle"])       # final model on everything
    return vec, model, {"held_out_accuracy": round(acc_all, 4), "accuracy_when_confident": round(acc_kept, 4),
                        "kept_share": round(float(keep.mean()), 4)}


def rule_aisle(text, rules=RULES):
    for aisle, pattern in rules:
        if re.search(pattern, text):
            return aisle
    return None


# ================================================================= prices
def open_prices():
    pr = pd.read_parquet(OPEN_PRICES, columns=["product_code", "price", "price_is_discounted",
                                               "price_without_discount", "currency", "date"])
    pr = pr[(pr["currency"] == "USD") & pr["product_code"].notna() & (pr["price"] > 0)]
    regular = np.where(pr["price_is_discounted"].fillna(False) & (pr["price_without_discount"] > 0),
                       pr["price_without_discount"], pr["price"])
    pr = pr.assign(regular=regular, disc=pr["price_is_discounted"].fillna(False).astype(bool))
    g = pr.groupby("product_code")
    return pd.DataFrame({"regular": g["regular"].median().round(2), "records": g.size(),
                         "promo": g["disc"].any(), "price_date": g["date"].max()})


def size_points(priced):
    """(aisle, department, unit kind, log size, log regular price) of every real-priced product with a size."""
    rows = []
    for r in priced.itertuples():
        kind, amount = unit_amount(parse_pack(r.pack_size) if isinstance(r.pack_size, str) else None)
        if kind and amount and amount > 0 and r.regular > 0:
            rows.append((r.aisle, r.department, kind, np.log(amount), np.log(r.regular)))
    return pd.DataFrame(rows, columns=["aisle", "department", "kind", "log_size", "log_price"])


def fit_curve(g):
    """log(price) = a + b log(size): bigger packs cost more, but less per unit (b between 0.2 and 1)."""
    if len(g) < MIN_SIZE_POINTS or g["log_size"].std() < 0.1:
        return None
    b, a = np.polyfit(g["log_size"], g["log_price"], 1)
    return float(a), float(np.clip(b, 0.2, 1.0)), len(g)


def size_curves(points):
    return ({k: c for k, g in points.groupby(["aisle", "kind"]) if (c := fit_curve(g))},
            {k: c for k, g in points.groupby(["department", "kind"]) if (c := fit_curve(g))})


def estimate_price(aisle, dept, pack, curves, aisle_median):
    by_aisle, by_dept = curves
    kind, amount = unit_amount(pack)
    if kind and amount:
        for where, key, table in (("aisle", (aisle, kind), by_aisle), ("department", (dept, kind), by_dept)):
            if key in table:
                a, b, n = table[key]
                est = float(np.exp(a + b * np.log(amount)))
                est = float(np.clip(est, 0.2 * aisle_median, 6 * aisle_median))
                return round(est, 2), f"size curve from {n} real prices ({where}, per {kind})"
    return round(float(aisle_median), 2), "aisle median price (no usable size)"


def estimate_quality(points):
    """How good the estimate is: fit on 80% of the real prices, estimate the other 20%."""
    test = points.sample(frac=0.2, random_state=7)
    curves = size_curves(points.drop(test.index))
    med = points.groupby("aisle")["log_price"].median()
    err = []
    for r in test.itertuples():
        c = curves[0].get((r.aisle, r.kind)) or curves[1].get((r.department, r.kind))
        est = np.exp(c[0] + c[1] * r.log_size) if c else np.exp(med.get(r.aisle, np.log(4.0)))
        err.append(abs(est - np.exp(r.log_price)) / np.exp(r.log_price))
    err = np.array(err)
    return {"held_out_prices": len(err), "median_error": round(float(np.median(err)), 3),
            "within_30pct": round(float((err <= 0.3).mean()), 3), "within_50pct": round(float((err <= 0.5).mean()), 3)}


# ================================================================= shares
def apply_shares(cat, shares, say):
    cat = cat.copy()
    report = []
    for aisle, rows in shares.groupby("aisle", sort=False):
        in_aisle = cat.index[cat["aisle"] == aisle]
        if not len(in_aisle):
            report.append({"aisle": aisle, "status": "aisle not in the catalog"})
            continue
        total = cat.loc[in_aisle, "popularity"].sum()
        words = (cat.loc[in_aisle, "brand"].fillna("") + " | " + cat.loc[in_aisle, "product_name"]).str.lower()
        taken = pd.Series(False, index=in_aisle)
        targets = []
        for r in rows.sort_values("share_pct", key=lambda s: s.astype(float)).itertuples():
            subset = words.str.contains(r.subset_pattern, regex=True) if r.subset_pattern else pd.Series(True, index=in_aisle)
            hit = words.str.contains(r.brand_pattern, regex=True) & subset & ~taken
            if not hit.any():
                report.append({"aisle": aisle, "group": r.group, "status": "no products of this brand"})
                continue
            taken |= hit
            base = cat.loc[in_aisle[subset.to_numpy()], "popularity"].sum()
            targets.append((r, hit, float(r.share_pct) / 100 * base))
        anchored = sum(t for _, _, t in targets)
        rest = ~taken
        rest_now = cat.loc[in_aisle[rest.to_numpy()], "popularity"].sum()
        if not targets or anchored >= total or rest_now <= 0:
            continue
        for r, hit, target in targets:
            idx = in_aisle[hit.to_numpy()]
            now = cat.loc[idx, "popularity"].sum()
            cat.loc[idx, "popularity"] *= target / now
            cat.loc[idx, "popularity_source"] = f"published brand share {r.share_pct}% ({r.metric}, {r.year}; {r.source_url})"
            cat.loc[idx, "popularity_label"] = "PUBLIC_STATISTIC"
            report.append({"aisle": aisle, "group": r.group, "products": int(hit.sum()), "share_pct": float(r.share_pct),
                           "was_pct": round(100 * now / total, 2), "year": r.year, "confidence": r.confidence})
        idx = in_aisle[rest.to_numpy()]
        cat.loc[idx, "popularity"] *= (total - anchored) / rest_now
        say(f"  {aisle}: " + ", ".join(f"{r.group} {r.share_pct}%" for r, _, _ in targets))
    return cat, report


# ================================================================= build
def name_with_brand(name, brand):
    name = re.sub(r"\s+", " ", str(name)).strip()
    if not isinstance(brand, str) or not brand.strip():
        return name
    key = re.sub(r"[^a-z0-9]", "", brand.lower())
    return name if key and key[:6] in re.sub(r"[^a-z0-9]", "", name.lower()) else f"{brand.strip()} {name}"


def main(say=print):
    say("1. Products")
    old = pd.read_csv(CATALOG_IN, dtype={"product_id": str}, low_memory=False)
    table = aisle_table()
    dept_of = table["department"].to_dict()
    stats = pd.read_csv(AISLE_STATS)
    for _, r in stats[stats["department"].isin(["babies", "pets"])].iterrows():
        dept_of[r["aisle"]] = r["department"]

    # --- the real-purchase catalog (Open Prices: products bought in US stores, real prices and sizes)
    old["brand"] = old["brand"].fillna("").astype(str).str.strip()
    own = old["brand"].map(own_brand)
    say(f"  real-purchase catalog: {len(old):,} products; retailer own-brands left out: {int(own.sum()):,} "
        f"(e.g. {', '.join(old.loc[own, 'brand'].value_counts().head(5).index)})")
    base = old[~own].copy()
    misfiled = base["aisle"].isin(FRESH_AISLES) & base["product_name"].str.contains(
        r"frozen|canned|dried|juice|jam|chips|bar\b|puree|sauce|diced avocado", case=False, regex=True)
    nonfood_old = base["department"].isin(NONFOOD_DEPTS)
    nf_rule = (base["product_name"].fillna("") + " | " + base["real_category"].fillna("")).str.lower().map(rule_aisle)
    food_like = base["real_category"].fillna("").str.contains(
        r"\b(food|foods|snacks?|beverages?|noodles?|meals?|sauces?|drinks?|candies|cereals?)\b", case=False, regex=True)
    ruled = base["aisle"].isin({a for a, _ in RULES})
    misfiled |= nonfood_old & ((ruled & (nf_rule != base["aisle"]))    # a ruled aisle: the rule must agree
                               | (~ruled & food_like))                 # other household aisles: no food
    words = base["product_name"].fillna("").str.split().str.len()
    misfiled |= ~base["department"].isin(FRESH_DEPTS) & (base["brand"] == "") & (words <= 2)   # "Soda"
    misfiled |= base["product_name"].str.contains(r"^KS ?\d|kirkland|great value", case=False, regex=True)
    say(f"  misfiled products left out: {int(misfiled.sum())} (e.g. "
        f"{', '.join(base.loc[misfiled, 'product_name'].head(4))})")
    base = base[~misfiled].copy()
    base["product_name"] = [name_with_brand(n, b) for n, b in zip(base["product_name"], base["brand"])]
    base["brand_owner"], base["price_basis"] = None, "Open Prices (real purchase)"

    # --- household, personal care, baby (Open Products / Beauty Facts) + baby food (Open Food Facts)
    nonfood = extract(["en.openproductsfacts.org.products.csv.gz", "en.openbeautyfacts.org.products.csv.gz"],
                      NONFOOD_US, say)
    off = extract(["en.openfoodfacts.org.products.csv.gz"], OFF_US, say)
    food_brands = off["brands"].map(first_brand).str.lower().value_counts()
    nf = nonfood.assign(brand=nonfood["brands"].map(first_brand))
    nf = nf[(nf["brand"] != "") & nf["product_name"].notna() & ~nf["brand"].map(own_brand)]
    nf_brands = nf["brand"].str.lower().value_counts()
    food_brand = nf["brand"].str.lower().map(lambda b: food_brands.get(b, 0) >= 5 and food_brands.get(b, 0) > nf_brands.get(b, 0))
    nf = nf[~food_brand]                                           # chewing gum is not a baby wipe
    nf["aisle"] = (nf["categories_en"].fillna("") + " | " + nf["product_name"]).str.lower().map(rule_aisle)
    baby = off.assign(brand=off["brands"].map(first_brand))
    baby = baby[(baby["brand"] != "") & baby["product_name"].notna() & ~baby["brand"].map(own_brand)]
    baby["aisle"] = (baby["categories_en"].fillna("") + " | " + baby["product_name"]).str.lower().map(
        lambda t: rule_aisle(t, FOOD_RULES))
    new = pd.concat([nf, baby[baby["aisle"].notna()]], ignore_index=True)
    new = new[new["aisle"].notna() & ~new["code"].isin(base["product_id"])].drop_duplicates("code")
    new["department"] = new["aisle"].map(dept_of)
    new = new[new["department"].notna()]
    new["pack"] = new["quantity"].map(lambda q: parse_pack(usda_units(q)) if isinstance(q, str) else None)
    new["name"] = [display_name(b, n, p) for b, n, p in zip(new["brand"], new["product_name"], new["pack"])]
    new = new[~new.duplicated(["brand", "name"])]
    new = new[new.groupby(["aisle", new["brand"].str.lower()]).cumcount() < PER_BRAND]
    say(f"  added household / personal care / baby products: {len(new):,}")

    say("2. Prices of the added products")
    points = size_points(old[old["original_price"] > 0].rename(columns={"original_price": "regular"}))
    curves = size_curves(points)
    price_quality = estimate_quality(points)
    say(f"  price estimate check ({price_quality['held_out_prices']:,} real prices held out): median error "
        f"{price_quality['median_error']:.0%}, within 30% {price_quality['within_30pct']:.0%}, within 50% "
        f"{price_quality['within_50pct']:.0%}")
    op = open_prices()
    disc_old = old["discount_percentage"].fillna(0)
    promo_rate = (disc_old > 0).groupby(old["aisle"]).mean()
    promo_depth = disc_old[disc_old > 0].groupby(old["aisle"]).median()
    dept_rate = (disc_old > 0).groupby(old["department"]).mean()
    all_depth = float(disc_old[disc_old > 0].median())
    rows = []
    for r in new.itertuples():
        a = table.loc[r.aisle] if r.aisle in table.index else None
        aisle_median = float(a["median_price"]) if a is not None else float(table["median_price"].median())
        if r.code in op.index:
            regular = float(op.loc[r.code, "regular"])
            label, pdate, basis = "REAL_OBSERVED", op.loc[r.code, "price_date"], \
                f"Open Prices, {int(op.loc[r.code, 'records'])} record(s)"
        else:
            regular, basis = estimate_price(r.aisle, r.department, r.pack, curves, aisle_median)
            label, pdate = "ESTIMATED", None
        disc, dtype_ = 0.0, None
        if stable_unit(r.code, "promo") < promo_rate.get(r.aisle, dept_rate.get(r.department, 0.0)):
            disc, dtype_ = float(promo_depth.get(r.aisle, all_depth)), "SALE"   # like its aisle (real rates)
        same = old[old["aisle"] == r.aisle]["popularity"]
        rows.append({
            "product_id": r.code, "product_name": r.name, "brand": r.brand, "brand_owner": None,
            "category": r.department, "subcategory": r.aisle, "price": round(regular * (1 - disc / 100), 2),
            "original_price": round(regular, 2), "discount_percentage": disc, "currency": "USD",
            "availability": None, "rating": None, "review_count": None, "product_url": None, "image_url": None,
            "source": r.source, "built_at": date.today().isoformat(), "price_date": pdate, "discount_type": dtype_,
            "cpi_item_code": a["cpi_item_code"] if a is not None else "SA0",
            "cpi_item_name": a["cpi_item_name"] if a is not None else "All items", "cpi_mapped_by": "aisle",
            "instacart_match": None, "match_score": None,
            "popularity": float(same.median()) if len(same) else float(table["median_popularity"].median()),
            "popularity_source": "median of the aisle's real-purchase products - ESTIMATED",
            "department": r.department, "department_source": "aisle", "price_label": label, "price_basis": basis,
            "product_facts_label": "REAL_OBSERVED", "popularity_label": "ESTIMATED", "department_label": "ESTIMATED",
            "cpi_category_label": "ESTIMATED", "aisle": r.aisle, "aisle_confidence": 1.0,
            "aisle_source": "category words", "aisle_label": "ESTIMATED",
            "real_category": str(r.categories_en).split(",")[-1].strip() if isinstance(r.categories_en, str) else None,
            "department_check": None, "pack_size": r.pack["text"] if r.pack else None,
            "promo_label": "ESTIMATED" if disc else "none",
        })
    added = pd.DataFrame(rows)
    say(f"  added products: real price {int((added['price_label'] == 'REAL_OBSERVED').sum()):,}, "
        f"ESTIMATED {int((added['price_label'] == 'ESTIMATED').sum()):,}")
    cat = pd.concat([base, added], ignore_index=True)
    cat["popularity"] = cat["popularity"].astype(float)

    say("3. Published brand shares")
    anchors = pd.read_csv(SHARES, dtype=str, keep_default_na=False)
    cat, report = apply_shares(cat, anchors, say)

    say("4. Departments")
    d = json.loads(DEPTS_IN.read_text(encoding="utf-8"))
    kept = dict(d["departments"])
    for dept, info in d["dropped"].items():
        if int((cat["department"] == dept).sum()) >= 10:
            kept[dept] = {"instacart_share": info["instacart_share"]}
    total = sum(v["instacart_share"] for v in kept.values())
    for dept, v in kept.items():
        v["n_products"] = int((cat["department"] == dept).sum())
        v["share"] = round(v["instacart_share"] / total, 5)
    DEPTS_OUT.write_text(json.dumps({"note": "Real Instacart department shares, rescaled over the departments the "
                                             "RETINA Supercenter sells (babies included when it has products).",
                                     "departments": kept}, indent=2), encoding="utf-8")
    cat = cat[cat["department"].isin(kept)].reset_index(drop=True)
    say("  " + ", ".join(f"{k} {v['share']:.1%} ({v['n_products']:,})" for k, v in
                         sorted(kept.items(), key=lambda kv: -kv[1]["share"])))

    cat.to_csv(CATALOG_OUT, index=False)
    summary = {"products": len(cat), "real_purchase_products": len(base), "added_products": len(added),
               "real_prices": int((cat["price_label"] == "REAL_OBSERVED").sum()),
               "estimated_prices": int((cat["price_label"] == "ESTIMATED").sum()),
               "own_brands_removed": int(own.sum()), "misfiled_removed": int(misfiled.sum()),
               "price_estimate": price_quality, "anchors": report,
               "top_brands": cat.loc[cat["brand"].fillna("") != "", "brand"].value_counts().head(25).to_dict()}
    REPORT_OUT.write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    say(f"Catalog: {len(cat):,} products -> {CATALOG_OUT.relative_to(ROOT)} (report: {REPORT_OUT.name})")
    return cat, summary


if __name__ == "__main__":
    main()
