"""NPC decisions for the shared-clock store.

Both brains receive the shopper's current state. They return choices; the store
agent validates those choices and applies inventory, walking and payment effects.
The rule brain is local and cheap. The student brain delegates choices to trained
models. A qualified shelf action model can choose buy or skip at the shelf; otherwise
the existing store response rules still handle price walk-aways and empty shelves.
Each action reports the model proposal separately from any rule that overrides it.
"""

from .config import CATEGORY_RESPONSE
from .shopping import pick_product
from .world import choose_from_shelf


class RuleBrain:
    label = "rule"

    def __init__(self, agent):
        self.agent = agent

    def next_aisle(self, candidates):
        a = self.agent
        return min(candidates, key=lambda aisle: a.floor._dist(a.pos, aisle))

    def checkout_chance(self):
        return 0.0

    def browse_offer(self, visible):
        return None

    def choose_shelf(self, item):
        a = self.agent
        wanted, route, _, _, u = a.plan[item][:5]
        wanted = int(wanted)
        aisle_name = a.store.aisle_of[wanted]
        choice = wanted
        effect = None
        boost = a.group["boost"][wanted]
        if route != "new" and u[5] < 1 - boost:
            if a.group["cum"][aisle_name] is None:
                return self._skip("aisle_removed", wanted, "assortment", constraint="assortment")
            choice = int(pick_product(a.store.aisles[aisle_name], a.group["cum"][aisle_name], u[6]))
            if choice != wanted:
                effect = "delisted_switch" if boost == 0 else "price_switch"
        if u[7] > min(1.0, a.group["inclusive"][aisle_name] ** CATEGORY_RESPONSE):
            return self._skip("too_expensive", choice, "price", constraint="price")
        return self._stock_choice(choice, wanted, u, effect)

    @staticmethod
    def _skip(reason, product, why_type, brain="rule", proposed_by=None, constraint=None):
        return {"product": -1, "requested": int(product), "mind": brain,
                "proposed_by": proposed_by or brain, "constraint": constraint,
                "why_type": why_type, "why": reason,
                "effect": "walked_away" if reason in {"too_expensive", "aisle_removed"} else None,
                "stockout_reaction": None}

    def _stock_choice(self, choice, wanted, u, effect=None, brain="rule", why_type="habit", why="planned item"):
        proposed_by = brain
        product, reaction = choose_from_shelf(choice, self.agent.store, self.agent.inv, u[8], u[9])
        if reaction is not None:
            why_type, why = "stockout", f"shelf empty: {reaction}"
            brain = "engine" if proposed_by != "rule" else "rule"
        elif effect is not None:
            why_type, why = "substitute", effect
        return {"product": int(product), "requested": int(choice), "mind": brain,
                "proposed_by": proposed_by, "constraint": "stockout" if reaction is not None else None,
                "why_type": why_type, "why": why, "effect": effect,
                "stockout_reaction": reaction}


class StudentBrain(RuleBrain):
    label = "student"

    def __init__(self, agent, mind, decisions):
        super().__init__(agent)
        self.mind = mind
        self.decisions = decisions

    def next_aisle(self, candidates):
        a = self.agent
        if hasattr(self.mind, "next_aisle"):
            return self.mind.next_aisle(a.shopper, a.trip, candidates, a.pos,
                                        a.spent, a.prices, a.promo)
        return super().next_aisle(candidates)

    def checkout_chance(self):
        a = self.agent
        if hasattr(self.mind, "checkout_chance"):
            return self.mind.checkout_chance(a.shopper, a.day, len(a.basket))
        return 0.0

    def browse_offer(self, visible):
        a = self.agent
        if hasattr(self.mind, "browse_offer"):
            return self.mind.browse_offer(a.shopper, a.day, a.trip, visible,
                                          a.spent, a.prices, a.promo, a.basket)
        return None

    def choose_shelf(self, item):
        a = self.agent
        if getattr(self.mind, "shelf_action", None) is not None:
            wanted = int(a.plan[item][0])
            decision = self.decisions.act(item, a.spent, a.budget)
            product = int(decision["product"])
            empty = a.inv.stock[wanted] <= 0
            reaction = None
            if empty:
                if product < 0:
                    reaction = "skip"
                else:
                    same_brand = a.inv.brand[product] == a.inv.brand[wanted] and a.inv.brand[wanted] != ""
                    reaction = "same_brand" if same_brand else "other_brand"
            return {"product": product, "requested": wanted, "mind": "student",
                    "proposed_by": "student", "constraint": decision.get("constraint"),
                    "why_type": "skip" if product < 0 else "substitute" if product != wanted else "choice",
                    "why": decision.get("reason", "local NPC shelf action"),
                    "effect": None, "stockout_reaction": reaction,
                    "qty": int(decision["qty"]), "shelf_policy": True,
                    "confidence": float(decision.get("confidence", 0.0)),
                    "skip_probability": float(decision.get("skip_probability", 0.0))}
        decision = self.decisions.get(item) if self.decisions is not None else None
        if decision is None:
            return super().choose_shelf(item)
        if decision.get("mind") not in ("student", "llm"):
            action = super().choose_shelf(item)
            action.update(mind="engine", proposed_by="engine")
            return action
        wanted, _, _, _, u = a.plan[item][:5]
        wanted = int(wanted)
        choice = int(decision["product"])
        if choice < 0:
            return self._skip("mind_skip", wanted, decision.get("why_type", "skip"), decision["mind"])
        if int(a.floor.product_aisle[choice]) != int(a.floor.product_aisle[wanted]):
            raise ValueError("the NPC brain chose a product outside the observed aisle")
        if decision["mind"] == "student":
            aisle_name = a.store.aisle_of[wanted]
            if u[7] > min(1.0, a.group["inclusive"][aisle_name] ** CATEGORY_RESPONSE):
                return self._skip("too_expensive", choice, "price", "rule", "student", "price")
        return self._stock_choice(choice, wanted, u, brain=decision["mind"],
                                  why_type=decision.get("why_type", "choice"),
                                  why=decision.get("why", "student choice"))
