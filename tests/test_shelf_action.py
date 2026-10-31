"""The experimental shelf policy must respect what an NPC can see and pay for."""

import json

import numpy as np

from retina.student.shelf_action import FEATURES, MODEL_FILE, ShelfActionPolicy


NPC = {"sensitivity": 1.1, "household": 2, "kids": 0, "income": 3,
       "budget": 120.0, "rebuy": 0.6}


class AvailableProduct:
    def choose(self, npc, route, mission, options, wanted, rng):
        found = next((o for o in options if o["in_stock"]), None)
        return found, 0.8 if found else 0.0, []


class FixedDraw:
    def __init__(self, value):
        self.value = value

    def random(self):
        return self.value


def offers(wanted_stock=True, substitute_stock=True):
    return [
        {"product": 1, "price": 4.0, "regular": 4.0, "in_stock": wanted_stock,
         "deal": False, "times": 2, "usual": True},
        {"product": 2, "price": 2.0, "regular": 2.5, "in_stock": substitute_stock,
         "deal": True, "times": 0, "usual": False},
    ]


def policy(tmp_path):
    # A self-contained unqualified artifact: normal live runs must gate it.
    artifact = {"version": 1, "features": FEATURES, "qualified": False,
                "mean": [0.0] * len(FEATURES), "scale": [1.0] * len(FEATURES),
                "coef": [0.0] * len(FEATURES), "intercept": 0.0}
    (tmp_path / MODEL_FILE).write_text(json.dumps(artifact), encoding="utf-8")
    assert ShelfActionPolicy.experimental_available(tmp_path)
    assert not ShelfActionPolicy.available(tmp_path)
    return ShelfActionPolicy(tmp_path, product_model=AvailableProduct())


def test_empty_and_unaffordable_shelves_force_skip(tmp_path):
    model = policy(tmp_path)
    for options, remaining in ((offers(False, False), 100.0), (offers(), 1.0)):
        action = model.choose(NPC, "need", "routine", options, 1, remaining, 2, FixedDraw(0.99))
        assert action["action"] == "skip"
        assert action["product"] == -1
        assert action["qty"] == 0
        assert action["constraint"] == "stock_or_budget"


def test_student_can_skip_at_a_shelf(tmp_path):
    model = policy(tmp_path)
    model.coef = np.zeros_like(model.coef)
    model.intercept = 30.0
    action = model.choose(NPC, "new", "routine", offers(), 1, 100.0, 1, FixedDraw(0.5))
    assert action["action"] == "skip"
    assert action["reason"] == "model_skip"


def test_sold_out_wanted_item_substitutes_and_quantity_fits_budget(tmp_path):
    model = policy(tmp_path)
    model.coef = np.zeros_like(model.coef)
    model.intercept = -30.0
    action = model.choose(NPC, "need", "routine", offers(False, True), 1, 5.0, 4, FixedDraw(0.5))
    assert action["action"] == "buy"
    assert action["product"] == 2
    assert action["qty"] == 2
    assert action["constraint"] == "stock_or_budget"
