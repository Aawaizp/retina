"""RETINA settings: file paths plus the STORE PROFILE.

Every assumption lives in a store profile file: stores/<id>.json (with a data label and source
for each setting). This file loads the chosen profile and keeps the same setting names the
rest of the code uses. Choose a store (PowerShell):   $env:RETINA_STORE = "generic_supermarket"
"""

import json
import os
from pathlib import Path
import numpy as np
import pandas as pd

# ================================================================ paths
ROOT = Path(__file__).resolve().parents[2]          # the project folder (retina/engine/config.py)
STORES_DIR = ROOT / "stores"


def shown(path):
    """A path for messages: short inside the project, in full outside it (RETINA_*_DIR overrides)."""
    try:
        return Path(path).relative_to(ROOT)
    except ValueError:
        return path
PROCESSED_DIR = ROOT / "data" / "processed"
US_CPI_FILE = PROCESSED_DIR / "cpi_us.csv"
SHOPPER_PARAMS_FILE = PROCESSED_DIR / "shopper_params.json"
DEPARTMENTS_FILE = PROCESSED_DIR / "store_departments.json"
PAIRS_FILE = PROCESSED_DIR / "instacart_pairs.csv"          # real "bought together" pairs
AISLE_STATS_FILE = PROCESSED_DIR / "instacart_aisle_stats.csv"   # real aisle shares (loaders/aisle_stats.py)
AISLE_PAIRS_FILE = PROCESSED_DIR / "instacart_aisle_pairs.csv"   # real aisles bought together
SHOPPER_PROFILES_FILE = PROCESSED_DIR / "instacart_shopper_profiles.csv"   # real habits, together
if os.environ.get("RETINA_NO_PROFILES"):     # build like a fresh clone: never use the private per-shopper
    SHOPPER_PROFILES_FILE = PROCESSED_DIR / "__private_profiles_switched_off__.csv"   # file (public builds)
TRIP_STATS_FILE = PROCESSED_DIR / "instacart_trip_stats.json"    # trip size vs days since last trip (loaders/trip_needs.py)
AISLE_CYCLES_FILE = PROCESSED_DIR / "instacart_aisle_cycles.csv"  # trips until the same product is bought again


# ======================================================== store profile
def load_profile(store_id):
    """Read stores/<store_id>.json and check that every required section is there."""
    path = STORES_DIR / f"{store_id}.json"
    if not path.exists():
        available = ", ".join(p.stem for p in STORES_DIR.glob("*.json"))
        raise FileNotFoundError(f"No store profile '{store_id}'. Available: {available}")
    with open(path, encoding="utf-8") as f:
        profile = json.load(f)
    required = ["id", "name", "catalog_file", "simulation", "shoppers", "pricing",
                "promotion_response", "operations", "out_of_stock_reactions", "labels"]
    missing = [k for k in required if k not in profile]
    if missing:
        raise ValueError(f"Store profile {path.name} is missing: {', '.join(missing)}")
    return profile


PROFILE = load_profile(os.environ.get("RETINA_STORE", "generic_supermarket"))
BUILTIN_CATALOG_FILE = ROOT / PROFILE["catalog_file"]           # the US catalog (23,011 products)
# Uploaded products (retina --products my.csv; retina/products/): the command converts the upload and
# sets these BEFORE the engine is imported. builtin = the US catalog; only = just the uploaded products;
# add = the US catalog plus the uploaded ones.
CATALOG_KIND = os.environ.get("RETINA_CATALOG_KIND", "builtin")
if CATALOG_KIND not in ("builtin", "only", "add"):
    raise ValueError(f"RETINA_CATALOG_KIND must be builtin, only or add (not '{CATALOG_KIND}')")
CATALOG_FILE = Path(os.environ["RETINA_CATALOG"]) if CATALOG_KIND != "builtin" else BUILTIN_CATALOG_FILE
CATALOG_TAG = os.environ.get("RETINA_CATALOG_TAG", "") if CATALOG_KIND != "builtin" else ""
# Custom stores: NPCs still WANT what real shoppers want (all aisles of the US catalog); a want in an
# aisle the store does not sell is recorded as "not sold here" (lost demand), never swapped.
NOT_SOLD_HERE = CATALOG_KIND != "builtin"
RUN_ID = PROFILE["id"] + (f"+{CATALOG_TAG}" if CATALOG_TAG else "")   # store (+ uploaded products)
OUTPUT_DIR = ROOT / "data" / "simulated" / RUN_ID           # each store keeps its own results

# ============================================== settings from the profile
_sim, _shop, _price = PROFILE["simulation"], PROFILE["shoppers"], PROFILE["pricing"]
_promo, _ops = PROFILE["promotion_response"], PROFILE["operations"]

SEED = _sim["seed"]                        # same seed = same simulation every run
START_DATE = _sim["start_date"]
SIM_DAYS = _sim["days"]
N_SHOPPERS = _sim["n_shoppers"]          # a number, or "auto" = sized from the traffic schedule
TRAFFIC = PROFILE.get("traffic")         # real footfall -> NPCs per hour (traffic.py)
HOUSEHOLDS = PROFILE.get("households")   # household size, children, income, food budget (households.py)
FORMAT = PROFILE.get("format", {})       # store format: assortment size, basket size, department mix
ASSORTMENT_SHARE = FORMAT.get("assortment_share", 1.0)   # share of each aisle's products on the shelves
BASKET_SCALE = FORMAT.get("basket_scale", 1.0)           # trip size vs the Instacart habits (small stores: < 1)
DEPARTMENT_WEIGHTS = FORMAT.get("department_weights", {}) # x department share (convenience: more drinks)
SENSITIVITY_GROUPS = _sim["sensitivity_groups"]

PRICE_SENSITIVITY_SPREAD = _shop["price_sensitivity_spread"]
WEEKDAY_HABIT = _shop["weekday_habit"]
INSTACART_DAY0 = _shop["instacart_day0"]
AISLE_AFFINITY = _shop.get("aisle_affinity", 0.0)       # share of new picks that follow a shopping mission
MISSION_SHARPNESS = _shop.get("mission_sharpness", 1.0) # how strongly missions favour real partner aisles
HISTORY_TRIPS = _shop.get("history_trips", 0)          # past trips' worth of purchases each NPC starts with
_needs = _shop.get("needs", {})
NEEDS_ON = bool(_needs) and AISLE_CYCLES_FILE.exists() and TRIP_STATS_FILE.exists()
HOME_FACTOR = _needs.get("home_factor", 1.0)    # how long a product lasts at home, x real re-buy cycle
NEED_TRIES = _needs.get("need_tries", 4)        # usual items checked to find one that ran out
HABIT_POWER = _needs.get("habit_power", 1.0) if NEEDS_ON else 1.0   # >1: items bought often become habits
REPEAT_BOOST = _needs.get("repeat_boost", 1.0) if NEEDS_ON else 1.0   # re-buy chance x this (TUNED)
BUDGET_ON = _needs.get("budget", False)         # NPCs stop adding extras when the trip budget is spent
START_WEEKDAY = pd.Timestamp(START_DATE).dayofweek       # weekday of simulation day 0

PROMO_WEEKLY_PROB = _price["promo_weekly_prob"]
PRICE_ELASTICITY = _price["price_elasticity"]
REF_PRICE_MEMORY = _price["ref_price_memory"]
LOSS_AVERSION = _price["loss_aversion"]
CATEGORY_RESPONSE = _price["category_response"]
REFERENCE_SHARE = _price["reference_share"]

STOCKPILE_RATE = _promo["stockpile_rate"]
IMPULSE_RATE = _promo["impulse_rate"]

LEAD_TIME_DAYS = _ops["lead_time_days"]
SAFETY_DAYS = _ops["safety_days"]
ORDER_DAYS = _ops["order_days"]
FORECAST_ERROR = _ops["forecast_error"]
FORECAST_LEARNING = _ops["forecast_learning"]
SUPPLIER_LATE_PROB = _ops["supplier_late_prob"]
SUPPLIER_LATE_DAYS = _ops["supplier_late_days"]
FRESH_MIN_SALES = _ops["fresh_min_sales"]
SHELF_LIFE_DAYS = _ops["shelf_life_days"]
GROSS_MARGIN = _ops["gross_margin"]
DEFAULT_MARGIN = _ops["default_margin"]

OOS_REACTIONS = PROFILE["out_of_stock_reactions"]


# ========================================================== data labels
# RETINA rule: never present estimated data as real. Every value carries one of these labels.
DATA_LABELS = {
    "REAL_OBSERVED": "recorded directly (a shelf price seen in a store, a product's name)",
    "PUBLIC_STATISTIC": "a published figure from an official body or study (CPI, survey results)",
    "ESTIMATED": "worked out by us (name matching, keyword rules, assumptions)",
    "SIMULATED": "produced by the RETINA simulation",
    "USER_PROVIDED": "given by the person who uploaded the products (retina --products); not checked against real stores",
}
CATALOG_LABEL_COLUMNS = ["price_label", "product_facts_label", "popularity_label",
                         "department_label", "cpi_category_label","aisle_label"]


# ===================================================== random streams
# Common random numbers: every part of the simulation rolls its OWN dice, made from the seed
# plus a fixed key. A shopper's dice for a given day are always the same - so when an experiment
# changes one product, shoppers who never meet that change behave EXACTLY as in the baseline.
STREAM_SHOPPERS, STREAM_PROMOS, STREAM_STOCK, STREAM_DELIVERIES, STREAM_TRIPS = 1, 2, 3, 4, 5
STREAM_TRAFFIC = 7                       # how many walk in today (6 = STREAM_WALK in floor.py)


def set_seed(seed):
    """Change the seed (for repeated runs with different luck). Every stream uses it."""
    global SEED
    SEED = seed

def stream(*keys):
    """Random generator for one purpose, e.g. stream(STREAM_TRIPS, day, shopper)."""
    return np.random.default_rng([SEED, *keys])