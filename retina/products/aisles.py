"""The 124 aisles (in 16 departments) every product must belong to.

They come from RETINA's US catalog (data/processed/catalog_us_clean.csv): an uploaded product
says which aisle it belongs to, and the engine uses the aisle for everything it knows from real
shoppers - how often the aisle is bought, re-bought, and with what. Per aisle we also keep the
typical price and popularity of its US products (for checks and honest defaults) and the
CPI category its products follow for inflation.
"""

import difflib
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
US_CATALOG = ROOT / "data" / "processed" / "catalog_us_clean.csv"


def norm(text):
    """Comparable form of an aisle or category name: lower case, no '&' / 'and' (the aisle names
    drop them: 'Cookies & Cakes' = 'cookies cakes'), single spaces."""
    words = re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split()
    return " ".join(w for w in words if w != "and")


@lru_cache(maxsize=1)
def aisle_table():
    """One row per aisle: department, products, median price, median popularity, CPI code."""
    c = pd.read_csv(US_CATALOG, usecols=["department", "aisle", "original_price", "popularity",
                                         "cpi_item_code", "cpi_item_name"], low_memory=False)
    mode = lambda s: s.mode().iat[0] if s.notna().any() else None
    t = c.groupby("aisle").agg(department=("department", mode), products=("aisle", "size"),
                               median_price=("original_price", "median"), median_popularity=("popularity", "median"),
                               cpi_item_code=("cpi_item_code", mode), cpi_item_name=("cpi_item_name", mode))
    t["key"] = [norm(a) for a in t.index]
    return t.sort_values(["department", "products"], ascending=[True, False])


def find(name):
    """Exact aisle for a name (case, spaces and '&' do not matter), else None."""
    t = aisle_table()
    hit = t.index[t["key"] == norm(name)]
    return hit[0] if len(hit) else None


# Everyday words -> aisles, for SUGGESTIONS only (the uploader confirms by writing the aisle name)
HINTS = {
    "biscuit": ["cookies cakes", "crackers"], "cookie": ["cookies cakes"], "crisp": ["chips pretzels"],
    "soda": ["soft drinks"], "cola": ["soft drinks"], "pop": ["soft drinks"], "drink": ["soft drinks", "juice nectars"],
    "candy": ["candy chocolate"], "sweet": ["candy chocolate"], "chocolate": ["candy chocolate"],
    "toothpaste": ["oral hygiene"], "toothbrush": ["oral hygiene"], "mouthwash": ["oral hygiene"],
    "shampoo": ["hair care"], "conditioner": ["hair care"], "detergent": ["laundry", "dish detergents"],
    "rice": ["grains rice dried goods"], "flour": ["baking ingredients"], "sugar": ["baking ingredients"],
    "lentil": ["grains rice dried goods"], "dal": ["grains rice dried goods", "indian foods"],
    "bean": ["canned meals beans", "grains rice dried goods"], "oil": ["oils vinegars"], "salt": ["spices seasonings"],
    "spice": ["spices seasonings"], "masala": ["spices seasonings", "indian foods"], "noodle": ["instant foods", "dry pasta"],
    "pasta": ["dry pasta", "fresh pasta"], "sauce": ["pasta sauce", "condiments"], "ketchup": ["condiments"],
    "jam": ["spreads"], "peanut": ["spreads", "nuts seeds dried fruit"], "ghee": ["butter", "indian foods"],
    "curd": ["yogurt"], "paneer": ["other creams cheeses", "indian foods"], "cheese": ["packaged cheese"],
    "chicken": ["packaged poultry", "poultry counter"], "beef": ["packaged meat", "meat counter"],
    "fish": ["packaged seafood", "seafood counter"], "fruit": ["fresh fruits"], "vegetable": ["fresh vegetables"],
    "veg": ["fresh vegetables"], "juice": ["juice nectars"], "water": ["water seltzer sparkling water"],
    "tissue": ["paper goods"], "toilet": ["paper goods"], "napkin": ["paper goods"], "diaper": ["more household"],
    "oat": ["hot cereal pancake mixes", "cereal"], "cornflake": ["cereal"], "ice": ["ice cream ice"],
    "frozen": ["frozen meals", "frozen produce"], "battery": ["more household"], "razor": ["shave needs"],
}


def singular(word):
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def suggest(name, n=3):
    """Close aisle names for a category that is not an aisle - a SUGGESTION, never applied by itself."""
    t = aisle_table()
    key = norm(name)
    if not key:
        return []
    words = {singular(w) for w in key.split()}
    scored = []
    for aisle, k in zip(t.index, t["key"]):
        aisle_words = {singular(v) for v in k.split()}
        ratio = difflib.SequenceMatcher(None, key, k).ratio()
        overlap = len(words & aisle_words) / max(len(words), 1)
        scored.append((max(ratio, overlap), aisle))
    scored.sort(reverse=True)
    out = [a for score, a in scored if score >= 0.85]                  # nearly the same name
    for w in sorted(words):
        out += [a for a in HINTS.get(w, []) if a not in out]           # everyday words
    out += [a for score, a in scored if score >= 0.6 and a not in out]
    return out[:n]


def department_aisles(name):
    """If `name` is a DEPARTMENT (e.g. 'snacks'), its aisles - so the message can list them."""
    t = aisle_table()
    rows = t[[norm(d) == norm(name) for d in t["department"]]]
    return list(rows.index)
