"""A shopper's needs for one trip, derived from their persistent household state.

The brain may choose products and routes, while these requirements keep its goal grounded
in what the shopper has run out of, the size of the trip and the money available.
"""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class NPCRequirements:
    day: int
    days_since_trip: int
    household_size: int
    children: int
    mission: str
    target_items: int
    budget: float
    due_products: tuple[int, ...]
    essential_products: tuple[int, ...]

    @classmethod
    def for_trip(cls, shoppers, shopper, day, target_items, mission, budget, needs_enabled=True):
        """Choose a small set of overdue usual purchases as this trip's essentials."""
        last = int(shoppers.last_trip_day[shopper])
        since = int(day - last if last >= 0 else shoppers.gap_mean[shopper])
        home = shoppers.home[shopper] or {}
        pantry = shoppers.pantry[shopper] or {}
        counts = shoppers.counts[shopper] or {}
        due = sorted({int(p) for p in shoppers.favorites[shopper]
                      if needs_enabled and home.get(p, -1) <= day and pantry.get(p, -1) <= day},
                     key=lambda p: (-counts.get(p, 0), p))
        # Keep space for discovery, deals and products that go together.
        essential_count = min(len(due), max(1, min(3, (target_items + 4) // 5)))
        household = shoppers.household_size
        children = shoppers.kids
        return cls(int(day), since, int(household[shopper]) if household is not None else 1,
                   int(children[shopper]) if children is not None else 0, mission,
                   int(target_items), float(budget), tuple(due), tuple(due[:essential_count]))

    def public(self, store):
        """Small, JSON-safe explanation for the live store and receipts."""
        names = store.products["product_name"]
        return {"household_size": self.household_size, "children": self.children,
                "days_since_trip": self.days_since_trip, "target_items": self.target_items,
                "budget": round(self.budget, 2) if isfinite(self.budget) else None,
                "due_count": len(self.due_products),
                "essentials": [str(names.iat[p]) for p in self.essential_products]}

    def bought_count(self, basket):
        return sum(p in basket for p in self.essential_products)
