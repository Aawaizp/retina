"""An NPC's in-store trip: choose a destination, observe its shelf, act, then choose again.

The list, product and basket-completion models make local decisions. Store mechanics
enforce walking distance, inventory, budget and checkout. Seeded draws still represent
arrival variation, model probabilities and shopper responses to price or empty shelves.
No remote LLM is called during a trip.
"""

import numpy as np

from .config import HOME_FACTOR, IMPULSE_RATE, NEEDS_ON, STOCKPILE_RATE, stream
from .floor import Walk
from .npc_brain import RuleBrain, StudentBrain
from .shopping import poisson_from_u, record
from .world import take_unit

STREAM_TRIP_END = 35
STREAM_BROWSE = 36
STREAM_BROWSE_ITEM = 37


class NPCAgent:
    def __init__(self, day, shopper, store, shoppers, tables, inv, promo, prices, rng, trip, mind, decisions):
        self.day, self.shopper = day, int(shopper)
        self.store, self.shoppers, self.tables, self.inv = store, shoppers, tables, inv
        self.promo, self.prices, self.rng, self.trip = promo, prices, rng, trip
        self.mind, self.decisions = mind, decisions
        self.brain = StudentBrain(self, mind, decisions) if mind is not None else RuleBrain(self)
        self.floor = store.floor
        self.plan = trip["plan"]
        self.budget = trip["budget"]
        self.group = tables[int(shoppers.sensitivity_group[shopper])]
        self.pantry, self.home = shoppers.pantry[shopper], shoppers.home[shopper]
        self.items_by_aisle = {}
        for i, entry in enumerate(self.plan):
            aisle = int(self.floor.product_aisle[int(entry[0])])
            self.items_by_aisle.setdefault(aisle, []).append(i)
        self.remaining = set(self.items_by_aisle)
        self.requirements = trip["requirements"]
        self.essential_aisles = {int(self.floor.product_aisle[p]) for p in self.requirements.essential_products
                                 if any(int(entry[0]) == p for entry in self.plan)}
        self.pos = self.floor.entrance
        self.stops, self.legs, self.points, self.cum_m = [], [], [self.pos], []
        self.passed = set()
        self.metres = self.spent = 0.0
        self.basket, self.misses, self.routes, self.minds = {}, [], {}, {}
        self.effects = list(trip["effects"])
        self.picks = 0
        self.early_checkout = None
        self.browse_count = 0
        self.max_browses = min(2, int(np.ceil(IMPULSE_RATE))) if IMPULSE_RATE > 0 else 0
        self.last_decisions = []

    def next_move(self, minute, per_metre):
        """The next aisle is chosen now from the remaining goals and the current basket."""
        pending_essentials = self.remaining & self.essential_aisles
        if self.remaining and not pending_essentials and self.stops and self.basket:
            # For calibrated trips the NPC has a stated shopping need. The
            # online-order stop model may end a trip early, but only after most
            # of that need has made it into the basket.
            calibrated = self.mind is not None and getattr(self.mind, "trip_size", None) is not None
            enough = not calibrated or len(self.basket) >= 0.8 * self.requirements.target_items
            p = self.brain.checkout_chance() if enough else 0.0
            if stream(STREAM_TRIP_END, self.day, self.shopper, len(self.stops)).random() < p:
                self.early_checkout = {"brain": self.brain.label, "reason": "basket_complete",
                                       "probability": round(float(p), 4), "unvisited_aisles": len(self.remaining)}
                record(self.day, self.shopper, "MIND_CHECKOUT", brain=self.brain.label, lines=len(self.basket),
                       probability=round(float(p), 4), unvisited_aisles=len(self.remaining),
                       t=round(float(minute), 2), x=round(float(self.pos[0]), 1),
                       y=round(float(self.pos[1]), 1))
                for aisle in sorted(self.remaining):
                    for i in self.items_by_aisle[aisle]:
                        record(self.day, self.shopper, "SKIP", int(self.plan[i][0]), reason="left_early",
                               t=round(float(minute), 2), x=round(float(self.pos[0]), 1),
                               y=round(float(self.pos[1]), 1))
                self.remaining.clear()
        if self.remaining:
            # Complete the household's depleted staples before browsing or deciding
            # that the basket is done. The trained brain still chooses among these aisles.
            candidates = sorted(pending_essentials or self.remaining)
            target = self.brain.next_aisle(candidates)
            if target not in candidates:
                raise ValueError("the NPC brain chose an aisle outside its remaining goals")
            browse = None
            if not pending_essentials and self.stops and self.browse_count < self.max_browses:
                direct = self.floor._dist(self.pos, target)
                _, seen, _ = self.floor.leg(self.pos, self.floor._spots[target])
                visible = [a for a in dict.fromkeys(seen) if a not in self.items_by_aisle
                           and a not in self.stops and self.group["cum"][self.floor.aisle_names[a]] is not None
                           and self.floor._dist(self.pos, a) + self.floor._dist(self.floor._spots[a], target)
                           <= direct + 12]
                offer = self.brain.browse_offer(visible)
                if offer is not None and offer["aisle"] in visible and stream(
                        STREAM_BROWSE, self.day, self.shopper, len(self.stops)).random() < offer["probability"]:
                    browse = offer
                    target = offer["aisle"]
                    product = int(offer["product"])
                    i = len(self.plan)
                    self.plan.append((product, "impulse", False, False,
                                      stream(STREAM_BROWSE_ITEM, self.day, self.shopper, i).random(11)))
                    self.items_by_aisle[target] = [i]
                    self.browse_count += 1
                    record(self.day, self.shopper, "MIND_BROWSE", product, aisle=self.floor.aisle_names[target],
                           brain=self.brain.label, probability=round(float(offer["probability"]), 4),
                           t=round(float(minute), 2), x=round(float(self.pos[0]), 1),
                           y=round(float(self.pos[1]), 1))
            if browse is None:
                self.remaining.remove(target)
            self.stops.append(target)
            stop = len(self.stops) - 1
            point = self.floor._spots[target]
            kind = "shelf"
        else:
            browse = None
            target, stop, point, kind = None, None, self.floor._checkout, "checkout"
        metres, seen, corners = self.floor.leg(self.pos, point, with_points=True)
        self.metres += metres
        self.cum_m.append(self.metres)
        self.legs.append(seen)
        self.passed.update(seen)
        if target is not None:
            self.passed.add(target)
        path, t, prev = [], float(minute), self.pos
        for corner in corners:
            t += (abs(corner[0] - prev[0]) + abs(corner[1] - prev[1])) * per_metre
            path.append([round(t, 2), round(corner[0], 1), round(corner[1], 1)])
            prev = corner
        self.points.extend(corners)
        self.pos = point
        brain = self.brain.label
        return {"kind": kind, "minute": float(minute) + metres * per_metre,
                "aisle": target, "stop": stop, "path": path, "brain": brain,
                "decision": self.early_checkout if kind == "checkout" else None,
                "browse": browse}

    def shop(self, aisle):
        """Products and stock are inspected only when this person reaches the shelf."""
        stop = len(self.stops) - 1
        record(self.day, self.shopper, "MOVE_TO_DEPARTMENT", department=self.floor.aisle_dept[aisle],
               aisle=self.floor.aisle_names[aisle], stop=stop)
        items = self.items_by_aisle[aisle]
        self.last_decisions = []
        for i in items:
            self._shop_item(i, stop)
        return len(items)

    def _shop_item(self, i, stop):
        s, day = self.shopper, self.day
        wanted, route, switched, together, u = self.plan[i][:5]
        wanted = int(wanted)
        want_qty = self.plan[i][5] if len(self.plan[i]) > 5 else None
        self.picks += 1
        record(day, s, "VIEW_PRODUCT", wanted, route=route, stop=stop)
        action = self.brain.choose_shelf(i)
        product = action["product"]
        reaction = action["stockout_reaction"]
        chosen = action["requested"]
        reason = action["why"]
        action_mind = action["mind"]
        constraint = action.get("constraint")
        policy = bool(action.get("shelf_policy"))
        qty = max(1, int(action.get("qty", 1))) if product >= 0 else 0
        if policy and product >= 0:
            # The NPC proposes the purchase; the world accepts only units that
            # physically exist and fit the money left in this trip.
            available = int(self.inv.stock[product])
            if available < 1:
                product, reason, action_mind, constraint, qty = -1, "out_of_stock", "rule", "stockout", 0
                reaction = reaction or "skip"
            else:
                if qty > available:
                    qty, constraint = available, "stockout"
                if np.isfinite(self.budget):
                    affordable = max(0, int((self.budget - self.spent + 1e-9) // max(float(self.prices[product]), .01)))
                    if affordable < 1:
                        self.effects.append((s, product, "over_budget", 1))
                        product, reason, action_mind, constraint, qty = -1, "over_budget", "rule", "budget", 0
                    elif qty > affordable:
                        self.effects.append((s, product, "over_budget", qty - affordable))
                        qty, constraint = affordable, "budget"
        elif product >= 0 and route != "need" and self.spent + self.prices[product] > self.budget:
            self.effects.append((s, product, "over_budget", 1))
            product, reason, qty = -1, "over_budget", 0
            action_mind, constraint = "rule", "budget"
        live_decision = {"item": int(i), "wanted": wanted, "product": int(product),
                         "mind": action_mind,
                         "proposed_by": action.get("proposed_by", action["mind"]),
                         "constraint": constraint, "why": reason,
                         "action": "pick" if product >= 0 else "skip", "qty": qty,
                         "stock_before": int(self.inv.stock[product]) if product >= 0 else 0}
        if policy:
            live_decision["skip_probability"] = action.get("skip_probability")
            live_decision["confidence"] = action.get("confidence")
        self.last_decisions.append(live_decision)
        if action["effect"] is not None:
            self.effects.append((s, chosen, action["effect"], 1))
        if reaction is not None:
            self.misses.append((s, chosen, reaction, product))
            record(day, s, "STOCKOUT", chosen, reaction=reaction, stop=stop)
        if product < 0:
            record(day, s, "SKIP", wanted, reason=reason, stop=stop)
            return
        product = int(product)
        if self.inv.stock[product] <= 0:
            raise ValueError("the NPC tried to take a product that is no longer on the shelf")
        if product != wanted:
            why = "out_of_stock" if reaction else (action["effect"] or "mind_choice")
            record(day, s, "SUBSTITUTE", product, instead_of=wanted, reason=why, stop=stop)
            switched = False
        if policy:
            for _ in range(qty):
                take_unit(product, self.inv)
        else:
            take_unit(product, self.inv)
            qty = 1
        self.basket[product] = self.basket.get(product, 0) + qty
        self.routes.setdefault(product, route)
        self.minds.setdefault(product, dict(action, mind=action_mind, qty=qty, constraint=constraint, why=reason))
        self.spent += self.prices[product] * qty
        if route == "impulse":
            self.effects.append((s, product, "impulse", 1))
        if together and product == wanted:
            self.effects.append((s, product, "bought_together", 1))
        if policy and self.promo[product] and qty > 1:
            self.pantry[product] = day + int((qty - 1) * self.shoppers.gap_mean[s])
            self.effects.append((s, product, "stock_up", qty - 1))
        elif not policy and want_qty and want_qty > 1:
            extra = min(want_qty - 1, int(self.inv.stock[product]))
            if self.budget < np.inf:
                extra = min(extra, max(0, int((self.budget - self.spent) // max(self.prices[product], 0.01))))
            if extra > 0:
                self.spent += extra * self.prices[product]
                for _ in range(extra):
                    take_unit(product, self.inv)
                self.basket[product] += extra
                qty += extra
        elif not policy and self.promo[product]:
            if switched and product == wanted:
                self.effects.append((s, product, "brand_switch", 1))
            extra = min(poisson_from_u(STOCKPILE_RATE * self.shoppers.price_sensitivity[s], u[10]),
                        self.inv.stock[product])
            if extra > 0 and self.budget < np.inf:
                extra = min(extra, max(0, int((self.budget - self.spent) // max(self.prices[product], 0.01))))
            if extra > 0:
                self.spent += extra * self.prices[product]
                for _ in range(extra):
                    take_unit(product, self.inv)
                self.basket[product] += extra
                self.pantry[product] = day + int(extra * self.shoppers.gap_mean[s])
                self.effects.append((s, product, "stock_up", extra))
                qty += extra
        live_decision["qty"] = qty
        self.minds[product]["qty"] = self.basket[product]
        record(day, s, "PURCHASE", product, qty=qty, promo=bool(self.promo[product]), route=route, stop=stop)

    def finish(self):
        if NEEDS_ON:
            gap = self.shoppers.gap_mean[self.shopper]
            for (product, qty), j in zip(self.basket.items(), self.rng.random(len(self.basket))):
                lasts = HOME_FACTOR * self.store.product_cycle[product] * gap * qty * (0.5 + j)
                self.home[product] = max(self.home.get(product, -1), self.day + int(lasts))
        stop_of_item = np.array([self.stops.index(int(self.floor.product_aisle[int(e[0])]))
                                 for e in self.plan if int(self.floor.product_aisle[int(e[0])]) in self.stops],
                                dtype=int)
        walk = Walk(np.array(self.stops, dtype=int), np.argsort(stop_of_item, kind="stable"),
                    stop_of_item, np.array(self.cum_m), self.legs,
                    np.array(sorted(self.passed), dtype=int), self.points)
        walk.picks = self.picks
        info = {"mission": self.trip["mission"], "routes": self.routes, "budget": self.budget,
                "spent": self.spent, "planned": self.trip["planned"], "minds": self.minds,
                "not_sold": self.trip.get("not_sold", []), "list_by": self.trip.get("list_by"),
                "requirements": self.requirements}
        return self.basket, self.misses, self.effects, walk, info


# Existing callers importing StudentAgent keep working while realtime uses the
# same NPC action loop for trained and rule brains.
StudentAgent = NPCAgent
