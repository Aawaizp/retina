"""The FMCG catalog builder (loaders/fmcg_catalog.py): the rules that keep the store real and generic.
No raw data needed - small examples only."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "loaders"))
import fmcg_catalog as F                                                      # noqa: E402


def test_retailer_own_brands_are_recognised():
    for brand in ["Trader Joe's", "Kirkland Signature", "365 Whole Foods Market", "Great Value", "Signature Select",
                  "Food Town Stores Inc.", "Hannaford", "Clover Valley"]:
        assert F.own_brand(brand), brand
    for brand in ["Coca-Cola", "Lay's", "Tide", "Chobani", "General Mills", "Pampers", "Colgate"]:
        assert not F.own_brand(brand), brand


def test_usda_units_and_brand_in_name():
    assert F.parse_pack(F.usda_units("12 OZA"))["text"] == "12 oz"
    assert F.parse_pack(F.usda_units("355 MLT"))["text"] == "355 ml"
    assert F.name_with_brand("Soda", "Fanta") == "Fanta Soda"
    assert F.name_with_brand("Lay's Classic Potato Chips", "Lay's") == "Lay's Classic Potato Chips"


def test_aisle_rules_do_not_mix_food_and_household():
    assert F.rule_aisle("fabric conditioner | lenor fabric conditioner") == "laundry"
    assert F.rule_aisle("toothpastes | crest 3d white") == "oral hygiene"
    assert F.rule_aisle("baby wipes | huggies natural care") == "diapers wipes"
    assert F.rule_aisle("hot dog buns | hot dog buns", F.FOOD_RULES) is None            # not dog food
    assert F.rule_aisle("coffees | cold brew coffee", F.FOOD_RULES) is None              # not cold & flu


def test_published_share_is_applied_and_aisle_total_kept():
    cat = pd.DataFrame({"aisle": ["soft drinks"] * 4, "brand": ["Coca-Cola", "Pepsi", "Goya", "Fanta"],
                        "product_name": ["Coca-Cola Classic", "Pepsi Cola", "Goya Soda", "Fanta Orange"],
                        "popularity": [1.0, 1.0, 5.0, 3.0], "popularity_source": "x", "popularity_label": "ESTIMATED"})
    shares = pd.DataFrame([{"aisle": "soft drinks", "group": "Coca-Cola", "brand_pattern": r"^(coca[- ]?cola|coke) \|",
                            "share_pct": "19.2", "metric": "m", "year": "2023", "confidence": "good", "source_url": "u",
                            "note": "", "subset_pattern": ""}])
    out, report = F.apply_shares(cat, shares, say=lambda *a: None)
    assert abs(out["popularity"].sum() - cat["popularity"].sum()) < 1e-9                 # aisle total unchanged
    assert abs(out.loc[0, "popularity"] / out["popularity"].sum() - 0.192) < 1e-9
    assert out.loc[0, "popularity_label"] == "PUBLIC_STATISTIC" and report[0]["group"] == "Coca-Cola"
    assert np.allclose(out.loc[2, "popularity"] / out.loc[3, "popularity"], 5 / 3)       # the rest keep their ratio


def test_price_estimate_follows_size_and_is_labelled():
    rng = np.random.default_rng(1)
    sizes = rng.uniform(4, 64, 60)
    pts = pd.DataFrame({"aisle": "juice nectars", "department": "beverages", "kind": "fl oz",
                        "log_size": np.log(sizes), "log_price": np.log(0.5 * sizes ** 0.7)})
    curves = F.size_curves(pts)
    small, basis = F.estimate_price("juice nectars", "beverages", {"count": 1, "amount": 12.0, "unit": "fl oz",
                                                                    "text": "12 fl oz"}, curves, 4.0)
    big, _ = F.estimate_price("juice nectars", "beverages", {"count": 1, "amount": 64.0, "unit": "fl oz",
                                                              "text": "64 fl oz"}, curves, 4.0)
    assert small < big < small * 64 / 12                                                  # bigger: dearer, cheaper per oz
    assert "size curve" in basis
