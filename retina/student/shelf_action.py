"""Local shelf action policy: decide SKIP vs a student-selected product.

This policy learns the SKIP action from teacher labels. The existing shelf
student chooses a product when the action is BUY. Quantity is the list
student's planned quantity, checked against live stock and remaining money.
Store constraints remain hard checks; no LLM call happens during a run.
"""

import json
import math
import os
from pathlib import Path

import numpy as np

from retina.student.model import Student


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = (Path(os.environ["RETINA_SHELF_ACTION_DIR"]) if os.environ.get("RETINA_SHELF_ACTION_DIR")
             else ROOT / "models" / "shelf_action")
MODEL_FILE = "shelf_action.json"

FEATURES = [
    "sensitivity_log", "household", "kids", "income", "budget_log", "rebuy",
    "route_need", "route_repeat", "route_new", "route_partner",
    "mission_top_up", "mission_routine", "mission_stock_up",
    "options_count", "available_count", "sold_out_share", "wanted_available",
    "wanted_price_log", "wanted_price_vs_budget", "wanted_price_change",
    "cheapest_price_log", "cheapest_price_vs_budget", "mean_price_log",
    "max_price_change", "min_price_change", "any_deal", "deal_share",
    "usual_available", "bought_before_share", "max_times_log",
    "new_x_wanted_unavailable", "need_x_wanted_unavailable",
]


def feature_vector(npc, route, mission, options, wanted):
    """One scenario vector using only facts available before the shelf action."""
    if not options:
        return np.zeros(len(FEATURES), dtype=np.float64)
    offered = list(options)
    available = [o for o in offered if o.get("in_stock")]
    price = lambda o: max(0.01, float(o.get("price") or 0.01))
    regular = lambda o: max(0.01, float(o.get("regular") or price(o)))
    budget = max(1.0, float(npc.get("budget") or 150.0))
    prices = [price(o) for o in available]
    cheapest = min(prices, default=0.01)
    wanted_option = next((o for o in offered if int(o["product"]) == int(wanted)), offered[0])
    wanted_available = bool(wanted_option.get("in_stock"))
    changes = [math.log(price(o) / regular(o)) for o in available]
    count = max(1, len(offered))
    present = max(1, len(available))
    values = [
        math.log1p(max(0.0, float(npc.get("sensitivity") or 1.0))),
        float(npc.get("household") or 2), float(npc.get("kids") or 0),
        float(npc.get("income") or 2), math.log(budget), float(npc.get("rebuy") or 0),
        *[float(route == r) for r in ("need", "repeat", "new", "partner")],
        *[float(mission == m) for m in ("top_up", "routine", "stock_up")],
        float(len(offered)), float(len(available)), 1 - len(available) / count,
        float(wanted_available), math.log(price(wanted_option)), price(wanted_option) / budget,
        math.log(price(wanted_option) / regular(wanted_option)),
        math.log(cheapest), cheapest / budget,
        float(np.mean(np.log(prices))) if prices else 0.0,
        max(changes, default=0.0), min(changes, default=0.0),
        float(any(o.get("deal") for o in available)),
        sum(bool(o.get("deal")) for o in available) / present,
        float(any(o.get("usual") for o in available)),
        sum(int(o.get("times") or 0) > 0 for o in available) / present,
        math.log1p(max((int(o.get("times") or 0) for o in available), default=0)),
        float(route == "new" and not wanted_available),
        float(route == "need" and not wanted_available),
    ]
    if len(values) != len(FEATURES):
        raise RuntimeError("shelf action feature width changed")
    return np.asarray(values, dtype=np.float64)


class ShelfActionPolicy:
    """Cheap trained skip head plus the existing local product chooser."""

    def __init__(self, folder=MODEL_DIR, product_model=None):
        folder = Path(folder)
        self.meta = json.loads((folder / MODEL_FILE).read_text(encoding="utf-8"))
        if self.meta.get("features") != FEATURES or self.meta.get("version") != 1:
            raise ValueError("shelf action features changed; retrain the model")
        self.product_model = product_model or Student()
        self.mean = np.asarray(self.meta["mean"], dtype=np.float64)
        self.scale = np.asarray(self.meta["scale"], dtype=np.float64)
        self.coef = np.asarray(self.meta["coef"], dtype=np.float64)
        self.intercept = float(self.meta["intercept"])
        self.qualified = bool(self.meta.get("qualified"))

    @staticmethod
    def experimental_available(folder=MODEL_DIR):
        return (Path(folder) / MODEL_FILE).is_file()

    @staticmethod
    def available(folder=MODEL_DIR):
        path = Path(folder) / MODEL_FILE
        if not path.is_file():
            return False
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
            return bool(meta.get("qualified") and meta.get("version") == 1 and meta.get("features") == FEATURES)
        except (OSError, ValueError):
            return False

    def skip_probability(self, npc, route, mission, options, wanted):
        x = (feature_vector(npc, route, mission, options, wanted) - self.mean) / self.scale
        z = float(np.clip(x @ self.coef + self.intercept, -30.0, 30.0))
        return 1.0 / (1.0 + math.exp(-z))

    def choose(self, npc, route, mission, options, wanted, remaining_budget, desired_qty, rng):
        """Return a live NPC proposal; store stock and money remain hard constraints.

        `remaining_budget=None` means no budget limit. An RNG draws from learned
        probabilities; without one, the deterministic most likely action is used.
        The engine checks exact stock counts before removing units.
        """
        offers = list(options)
        affordable = [o for o in offers if o["in_stock"] and
                      (remaining_budget is None or float(o["price"]) <= remaining_budget + 1e-9)]
        if not affordable:
            return {"action": "skip", "product": -1, "qty": 0, "skip_probability": 1.0,
                    "confidence": 1.0, "reason": "no_affordable_stock", "constraint": "stock_or_budget"}
        p_skip = self.skip_probability(npc, route, mission, offers, wanted)
        if (rng.random() < p_skip) if rng is not None else (p_skip >= 0.5):
            return {"action": "skip", "product": -1, "qty": 0, "skip_probability": p_skip,
                    "confidence": p_skip, "reason": "model_skip", "constraint": None}
        eligible = [dict(o, in_stock=o in affordable) for o in offers]
        choice, product_conf, _ = self.product_model.choose(npc, route, mission, eligible, wanted, rng)
        if choice is None:
            return {"action": "skip", "product": -1, "qty": 0, "skip_probability": p_skip,
                    "confidence": 1.0, "reason": "no_available_option", "constraint": "stock_or_budget"}
        product = int(choice["product"])
        desired = max(1, int(desired_qty or 1))
        quantity = desired
        if remaining_budget is not None:
            quantity = min(quantity, max(0, int((remaining_budget + 1e-9) // max(float(choice["price"]), 0.01))))
        if quantity < 1:
            return {"action": "skip", "product": -1, "qty": 0, "skip_probability": p_skip,
                    "confidence": 1.0, "reason": "no_affordable_stock", "constraint": "stock_or_budget"}
        return {"action": "buy", "product": product, "qty": quantity, "skip_probability": p_skip,
                "confidence": (1 - p_skip) * product_conf, "reason": "model_buy",
                "constraint": "stock_or_budget" if quantity < desired else None}
