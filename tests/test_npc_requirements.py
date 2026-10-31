"""Household requirements must affect the NPC's choices, not just its labels."""

import numpy as np

from retina.engine.engine import Simulation, shelf_prices
from retina.engine.config import SENSITIVITY_GROUPS
from retina.engine.shopping import make_list
from retina.engine.student_agent import StudentAgent
from retina.engine.world import add_favorites


def test_depleted_staple_is_planned_and_visited_before_checkout():
    sim = Simulation(days=2, verbose=False)
    sim.run_day()  # initialize today's pick tables and shopper history
    day, shopper = sim.day, 0
    store, shoppers = sim.store, sim.shoppers
    products = [int(a.product_idx[0]) for a in store.aisles.values()]
    staple, extra = next((a, b) for a in products for b in products
                         if store.aisle_of[a] != store.aisle_of[b])
    add_favorites(store, shoppers, shopper, {staple: 1})
    shoppers.favorites[shopper] = [staple]
    shoppers.home[shopper][staple] = day - 1
    shoppers.pantry[shopper][staple] = day - 1
    shoppers.reorder_prob[shopper] = 0.0  # need still matters without a habit dice roll
    prices_g, regular, promo_g, _ = shelf_prices(day, store, sim.book, sim.iv, SENSITIVITY_GROUPS)
    group = int(shoppers.sensitivity_group[shopper])
    trip = make_list(shopper, day, store, shoppers, sim.tables, promo_g[group],
                     np.random.default_rng(81), prices_g[group])
    assert staple in trip["requirements"].essential_products
    assert trip["plan"][0][:2] == (staple, "need")

    # A local brain that wants to check out immediately must still visit the
    # depleted staple. It only sees required aisles while that goal is pending.
    class Brain:
        def __init__(self):
            self.checkout_calls = 0
            self.offered_aisles = None

        def checkout_chance(self, *args):
            self.checkout_calls += 1
            return 1.0

        def next_aisle(self, shopper, trip, aisles, pos, spent, prices, promo):
            self.offered_aisles = aisles
            return aisles[0]

    brain = Brain()
    u = np.random.default_rng(2).random(11)
    trip["plan"] = [(extra, "new", False, False, u), (staple, "need", False, False, u)]
    agent = StudentAgent(day, shopper, store, shoppers, sim.tables, sim.inv, promo_g[group],
                         prices_g[group], np.random.default_rng(3), trip, brain, None)
    agent.stops.append(int(store.floor.product_aisle[extra]))
    agent.basket[extra] = 1
    move = agent.next_move(600, 1 / store.floor.settings["walk_speed_m_per_s"] / 60)
    assert move["aisle"] == int(store.floor.product_aisle[staple])
    assert brain.offered_aisles == [move["aisle"]]
    assert brain.checkout_calls == 0


def test_student_brain_gets_requirements_without_a_rule_generated_list(monkeypatch):
    from retina.engine import engine as E, realtime as RT

    sim = Simulation(days=1, verbose=False)
    product = int(next(iter(sim.store.aisles.values())).product_idx[0])

    class Brain:
        mode = "student"
        lister = object()

        def __init__(self):
            self.today = {"rows": []}
            self.empty_lists_seen = 0

        def start_trip(self, day, shopper, trip, ctx):
            assert trip["plan"] == []
            assert trip["requirements"].target_items >= 1
            self.empty_lists_seen += 1
            trip.update(plan=[(product, "new", False, False, np.random.default_rng(shopper).random(11))],
                        planned=1, list_by="test student brain")

        def at_shelf(self, *args):
            return None

        def after_trip(self, *args):
            pass

    sim.mind = Brain()
    events = []
    sim.on_event = lambda kind, minute, data: events.append((kind, data))
    monkeypatch.setattr(RT, "make_list", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("rule brain generated a student list")))
    monkeypatch.setattr(E, "ENGINE", "classic")
    _, trips = sim.run_day()
    assert sim.mind.empty_lists_seen == len(trips) > 0
    assert (trips["list_planned_by"] == "test student brain").all()
    assert all(data["list_brain"] == "student" for kind, data in events if kind == "enter")
