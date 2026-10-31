"""The real-time engine (retina/engine/realtime.py): every shopper on one clock, receipts written when
they pay. Small store (the supermarket), no LLM."""

import numpy as np
import pytest

from retina.engine import engine as E
from retina.engine.engine import Simulation


@pytest.fixture(scope="module")
def day_run():
    sim = Simulation(verbose=False)
    while sim.day < 3:                                         # a few classic days: shoppers with a history
        sim.run_day()
    events = []
    sim.on_event = lambda kind, minute, data: events.append((kind, minute, data))
    old, E.ENGINE = E.ENGINE, "realtime"
    try:
        sales, trips = sim.run_day()
    finally:
        E.ENGINE = old
    return sim, sales, trips, events


def test_receipts_match_the_trips(day_run):
    sim, sales, trips, events = day_run
    assert len(trips) > 50 and trips["trip_id"].is_unique
    units = sales.groupby("trip_id")["quantity"].sum()
    paid = trips.set_index("trip_id")["items"]
    assert (units.reindex(paid.index, fill_value=0) == paid).all()
    assert not sales.duplicated(["trip_id", "product_idx"]).any()           # one line per product
    assert (sim.inv.stock >= 0).all()


def test_time_only_moves_forward(day_run):
    _, _, trips, events = day_run
    minutes = [m for _, m, _ in events]
    assert minutes == sorted(minutes)                                      # one clock for everyone
    assert (trips["ready_min"] >= trips["arrive_min"]).all()
    assert (trips["leave_min"] >= trips["ready_min"] + trips["wait_min"] - 1e-9).all()
    assert (trips["wait_min"] >= 0).all()


def test_receipt_rows_are_written_when_shoppers_pay(day_run):
    _, sales, trips, events = day_run
    pays = [d for kind, _, d in events if kind == "pay"]
    enters = [d for kind, _, d in events if kind == "enter"]
    assert len(enters) == len(trips) == len(pays)
    lines = {(p["trip_id"], x["product"]): x["qty"] for p in pays for x in p["lines"]}
    assert lines == {(r.trip_id, r.product_idx): r.quantity for r in sales.itertuples()}
    order = [p["trip_id"] for p in pays if p["lines"]]                      # empty baskets have no sale rows
    assert list(dict.fromkeys(sales["trip_id"])) == order


def test_live_trip_reports_requirements(day_run):
    _, _, trips, events = day_run
    enters = [d for kind, _, d in events if kind == "enter"]
    pays = [d for kind, _, d in events if kind == "pay"]
    assert enters and all("target_items" in d["requirements"] for d in enters)
    assert all(d["essential_items_bought"] <= d["essential_items"] for d in pays)
    assert (trips["essential_items_bought"] <= trips["essential_items"]).all()


def test_shoppers_share_the_shelves_in_time_order(day_run):
    """Someone who reaches the checkout later never pays earlier at the same lane (first come, first served)."""
    _, _, trips, _ = day_run
    for _, lane in trips.groupby("lane"):
        lane = lane.sort_values("ready_min")
        start = lane["ready_min"] + lane["wait_min"]
        assert np.all(np.diff(start.to_numpy()) >= -1e-9)


def test_the_shelf_brain_decides_at_the_shelf():
    """The student's shelf decisions are made when the shopper stands there (lazily), and an empty set of
    decisions still counts as a brain (an empty dict would make the engine use its own rules)."""
    from retina.mind.mind import Mind
    m = Mind.__new__(Mind)
    asked = []
    m.decide_item = lambda day, s, k, trip, ctx: asked.append(k) or {"mind": "student", "product": 100 + k}
    shelf = m.at_shelf(5, 7, {"plan": []}, {})
    assert shelf and not asked                                  # a brain, nothing decided before the shelves
    assert shelf.get(2)["product"] == 102 and asked == [2]
    assert shelf.get(2)["product"] == 102 and asked == [2]      # decided once


def test_student_decisions_are_streamed_at_the_shelf():
    class ShelfMind:
        mode = "student"

        def __init__(self):
            self.today = {"rows": []}
            self.routes = []

        def start_trip(self, day, shopper, trip, ctx):
            return trip

        def at_shelf(self, day, shopper, trip, ctx):
            mind = self

            class Choices(dict):
                def __bool__(self):
                    return True

                def get(self, item, default=None):
                    if item not in self:
                        product = int(trip["plan"][item][0])
                        mind.today["rows"].append({"item": item, "mind": "student", "product": product,
                                                   "why": "test student choice", "minute": ctx["minute"][shopper]})
                        self[item] = {"mind": "student", "product": product, "why_type": "habit",
                                      "why": "test student choice"}
                    return self[item]

            return Choices()

        def after_trip(self, *args):
            pass

        def next_aisle(self, shopper, trip, aisles, pos, spent, prices, promo):
            self.routes.append((shopper, tuple(aisles), spent))
            return max(aisles) if len(self.routes) % 2 else min(aisles)

    sim = Simulation(days=1, verbose=False)
    sim.mind = ShelfMind()
    events = []
    sim.on_event = lambda kind, minute, data: events.append((kind, minute, data))
    old, E.ENGINE = E.ENGINE, "realtime"
    try:
        sim.run_day()
    finally:
        E.ENGINE = old
    choices = [(minute, data["shopper"], d) for kind, minute, data in events if kind == "shelf"
               for d in data["decisions"]]
    assert choices and len(choices) == len(sim.mind.today["rows"])
    assert sorted(d["minute"] for d in sim.mind.today["rows"]) == sorted(round(t, 2) for t, _, _ in choices)
    assert any(len(remaining) > 1 for _, remaining, _ in sim.mind.routes)
    assert len([1 for kind, _, _ in events if kind == "move"]) >= len([
        1 for kind, _, _ in events if kind == "enter"])
    # Store price, stock and budget rules may override a student shelf proposal.
    assert all(d["mind"] in {"student", "engine", "rule"} and d["name"] for _, _, d in choices)
    assert any(d["mind"] == "student" for _, _, d in choices)
    enters = {data["shopper"]: minute for kind, minute, data in events if kind == "enter"}
    pays = {data["shopper"]: minute for kind, minute, data in events if kind == "pay"}
    assert all(enters[shopper] < minute < pays[shopper] for minute, shopper, _ in choices)


@pytest.mark.parametrize("limit,expected_qty", [("budget", 2), ("stock", 1)])
def test_shelf_action_proposal_is_applied_at_live_shelf(limit, expected_qty, monkeypatch):
    """A local action policy acts at the shelf; the world clips units and the receipt agrees."""
    from retina.engine import traffic
    from retina.engine.shopping import RECORDER
    from retina.mind.mind import Mind

    sim = Simulation(days=1, verbose=False)
    monkeypatch.setattr(traffic, "enabled", lambda: False)
    sim.shoppers.next_visit_day[:] = 1
    sim.shoppers.next_visit_day[0] = 0
    mind = Mind.__new__(Mind)
    mind.sim, mind.mode, mind.lister, mind.ender = sim, "student", None, None
    mind.today = {"rows": []}

    class Memory:
        def recall(self, *args):
            return {}

        def add_decisions(self, rows):
            pass

    class Policy:
        meta = {"version": 1}

        def __init__(self):
            self.calls = []
            self.product = None

        def choose(self, npc, route, mission, options, wanted, remaining_budget, desired_qty, rng):
            self.calls.append((self.product, remaining_budget, desired_qty,
                               next(o["in_stock"] for o in options if o["product"] == self.product)))
            return {"action": "buy", "product": self.product, "qty": 4,
                    "skip_probability": 0.1, "confidence": 0.9, "reason": "model_buy"}

    mind.memory, mind.shelf_action = Memory(), Policy()
    before = {}

    def start_trip(day, shopper, trip, ctx):
        group = int(sim.shoppers.sensitivity_group[shopper])
        prices, promo, table = ctx["prices_g"][group], ctx["promo_g"][group], ctx["tables"][group]
        for aisle in sim.store.aisles.values():
            for wanted in aisle.product_idx:
                wanted = int(wanted)
                if sim.inv.stock[wanted] < 1:
                    continue
                _, options = mind.options(shopper, 0, day, wanted, table, prices, promo, ctx["regular"])
                alternate = next((o for o in options if o["product"] != wanted and o["in_stock"]), None)
                if alternate is not None:
                    product = int(alternate["product"])
                    mind.shelf_action.product = product
                    sim.inv.stock[product] = 4 if limit == "budget" else 1
                    before["product"], before["stock"] = product, int(sim.inv.stock[product])
                    price = float(prices[product])
                    trip.update(plan=[(wanted, "new", False, False, np.zeros(11), 4)], planned=1,
                                budget=2.5 * price if limit == "budget" else 10 * price)
                    return trip
        raise AssertionError("test store needs two stocked products in an aisle")

    mind.start_trip = start_trip
    mind.after_trip = lambda *args: None
    sim.mind = mind
    events, shelf_stock = [], []

    def on_event(kind, minute, data):
        events.append((kind, minute, data))
        if kind == "shelf":
            shelf_stock.append(int(sim.inv.stock[before["product"]]))

    sim.on_event = on_event
    old, old_recorder = E.ENGINE, dict(RECORDER)
    E.ENGINE = "realtime"
    try:
        sales, _ = sim.run_day()
    finally:
        E.ENGINE = old
        RECORDER.update(old_recorder)

    product = before["product"]
    shelf = [(minute, d) for kind, minute, data in events if kind == "shelf"
             for d in data["decisions"] if d["product"] == product]
    assert len(mind.shelf_action.calls) == len(shelf) == len(shelf_stock) == 1
    assert mind.shelf_action.calls[0][2:] == (4, True)
    assert mind.today["rows"][0]["minute"] == round(shelf[0][0], 2)
    assert shelf[0][1]["mind"] == "student" and shelf[0][1]["qty"] == expected_qty
    assert shelf[0][1]["proposed_by"] == "student"
    assert shelf[0][1]["constraint"] == ("budget" if limit == "budget" else "stockout")
    assert shelf_stock[0] == before["stock"] - expected_qty
    assert sales.loc[sales["product_idx"] == product, "quantity"].tolist() == [expected_qty]
    paid = [line for kind, _, data in events if kind == "pay" for line in data["lines"]
            if line["product"] == product]
    assert len(paid) == 1 and paid[0]["qty"] == expected_qty


def test_streamed_replay_does_not_leak_into_another_run(monkeypatch):
    """An unrecorded run of the same day starts with its own empty event log."""
    from retina.engine import traffic
    from retina.engine.shopping import RECORDER

    first = Simulation(days=1, verbose=False)
    second = Simulation(days=1, verbose=False)
    monkeypatch.setattr(traffic, "enabled", lambda: False)
    for sim in (first, second):
        sim.shoppers.next_visit_day[:] = 1
        sim.shoppers.next_visit_day[0] = 0
    first.on_event = lambda *args: None
    old_engine, old_recorder = E.ENGINE, dict(RECORDER)
    try:
        E.ENGINE = "realtime"
        first.run_day()
        assert RECORDER["day"] == 0 and RECORDER["events"]
        second.run_day()
        assert RECORDER["day"] is None and RECORDER["events"] == []
    finally:
        E.ENGINE = old_engine
        RECORDER.update(old_recorder)


def test_student_can_choose_checkout_before_remaining_aisles():
    """A checkout choice ends a live route without breaking its replay or its receipt."""
    from retina.engine.shopping import RECORDER

    class CheckoutMind:
        mode = "student"

        def __init__(self):
            self.today = {"rows": []}

        def start_trip(self, day, shopper, trip, ctx):
            return trip

        def at_shelf(self, day, shopper, trip, ctx):
            return None

        def checkout_chance(self, shopper, day, lines):
            return 1.0

        def after_trip(self, *args):
            pass

    sim = Simulation(days=1, verbose=False)
    sim.mind = CheckoutMind()
    events = []
    sim.on_event = lambda kind, minute, data: events.append((kind, minute, data))
    old_engine, old_recorder = E.ENGINE, dict(RECORDER)
    try:
        E.ENGINE = "realtime"
        sales, trips = sim.run_day()
        recorded = list(RECORDER["events"])
    finally:
        E.ENGINE = old_engine
        RECORDER.update(old_recorder)
    early = [(minute, data) for kind, minute, data in events
             if kind == "move" and data.get("decision")]
    assert early
    assert all(x["to"] == "checkout" and x["brain"] == "student"
               and x["decision"]["unvisited_aisles"] > 0 for _, x in early)
    early_shoppers = {x["shopper"] for _, x in early}
    skipped = [e for e in recorded if e["event"] == "SKIP" and e.get("reason") == "left_early"]
    checkout_decisions = [e for e in recorded if e["event"] == "MIND_CHECKOUT"]
    assert skipped and {e["shopper"] for e in skipped} <= early_shoppers
    assert len(checkout_decisions) == len(early)
    assert all(e["t"] > trips.set_index("shopper_id").loc[e["shopper"], "arrive_min"] for e in skipped)
    assert (sim.inv.stock >= 0).all()
    paid = trips.set_index("trip_id")["items"]
    assert (sales.groupby("trip_id")["quantity"].sum().reindex(paid.index, fill_value=0) == paid).all()


def test_student_can_browse_a_visible_unplanned_shelf():
    """A new item is decided at its own shelf, appears in the walk, and reaches the receipt."""
    from retina.engine.shopping import RECORDER

    class BrowseMind:
        mode = "student"

        def __init__(self, sim):
            self.sim = sim
            self.today = {"rows": []}

        def start_trip(self, day, shopper, trip, ctx):
            return trip

        def at_shelf(self, day, shopper, trip, ctx):
            mind = self

            class Choices(dict):
                def __bool__(self):
                    return True

                def get(self, item, default=None):
                    if item not in self:
                        product = int(trip["plan"][item][0])
                        mind.today["rows"].append({"item": item, "mind": "student", "product": product,
                                                   "why": "test shelf choice"})
                        self[item] = {"mind": "student", "product": product,
                                      "why_type": "try_new", "why": "test shelf choice"}
                    return self[item]

            return Choices()

        def checkout_chance(self, shopper, day, lines):
            return 0.0

        def browse_offer(self, shopper, day, trip, visible, spent, prices, promo, in_basket):
            floor, store = self.sim.store.floor, self.sim.store
            for aisle in visible:
                for product in store.aisles[floor.aisle_names[aisle]].product_idx:
                    if self.sim.inv.stock[product] > 0 and int(product) not in in_basket:
                        return {"aisle": aisle, "product": int(product), "probability": 1.0, "deal": False}
            return None

        def after_trip(self, *args):
            pass

    sim = Simulation(days=1, verbose=False)
    sim.mind = BrowseMind(sim)
    events = []
    sim.on_event = lambda kind, minute, data: events.append((kind, minute, data))
    old_engine, old_recorder = E.ENGINE, dict(RECORDER)
    try:
        E.ENGINE = "realtime"
        sales, trips = sim.run_day()
        recorded = list(RECORDER["events"])
    finally:
        E.ENGINE = old_engine
        RECORDER.update(old_recorder)
    browse = [d for kind, _, d in events if kind == "move" and d.get("browse")]
    assert browse and all(d["brain"] == "student" for d in browse)
    assert sum(e["event"] == "MIND_BROWSE" for e in recorded) == len(browse)
    bought = sales[sales["reason"] == "impulse"]
    assert len(bought) > 0 and (bought["mind"] == "student").all()
    assert set(bought["shopper_id"]) <= {d["shopper"] for d in browse}
    assert (trips["shelves_visited"] >= 1).all() and (sim.inv.stock >= 0).all()


def test_the_live_stream_tells_the_whole_story(day_run):
    """The NPC chooses each move live; shelves carry actions and pay carries receipt lines."""
    _, sales, _, events = day_run
    assert events[0][0] == "day" and events[0][2]["date"]
    enter = next(d for k, _, d in events if k == "enter")
    assert len(enter["path"]) == 1 and enter["mission"] in ("top_up", "routine", "stock_up")
    moves = [d for k, _, d in events if k == "move"]
    assert moves and all(d["brain"] == "rule" and d["path"] for d in moves)
    choices = [x for k, _, d in events if k == "shelf" for x in d["decisions"]]
    assert choices and all(x["mind"] == "rule" and x["action"] in {"pick", "skip"} for x in choices)
    assert all(x["stock_before"] > 0 for x in choices if x["action"] == "pick")
    shelf = [x for k, _, d in events if k == "shelf" for x in d["events"]]
    assert shelf and all(x["name"] for x in shelf if x["product"] >= 0)
    assert all("price" in x for x in shelf if x["event"] == "PURCHASE")
    pays = [d for k, _, d in events if k == "pay"]
    assert all(x["name"] and x["qty"] > 0 for p in pays for x in p["lines"])
    assert sum(len(p["lines"]) for p in pays) == len(sales)
    assert {k for k, _, _ in events} == {"day", "enter", "move", "shelf", "checkout", "pay", "leave", "day_end"}


def test_live_stream_file_and_pacing(tmp_path, monkeypatch):
    import json
    import time
    from types import SimpleNamespace
    from retina.engine import live
    monkeypatch.setattr(live.LiveStream, "MINUTES_PER_SECOND", 600.0)
    stream = live.LiveStream(tmp_path, SimpleNamespace(store=None))
    stream("enter", 360.0, {"shopper": 1, "path": [[360.0, 1.0, 2.0]]})
    assert json.loads((tmp_path / "stream.ndjson").read_text(encoding="utf-8").splitlines()[0])["k"] == "enter"
    stream("shelf", 370.0, {"shopper": 1, "decisions": [{"mind": "student", "product": 2}], "events": []})
    visible = [json.loads(x) for x in (tmp_path / "stream.ndjson").read_text(encoding="utf-8").splitlines()]
    assert [x["k"] for x in visible] == ["enter", "shelf"]
    assert visible[-1]["decisions"][0]["mind"] == "student"
    stream("pay", 380.5, {"shopper": 1, "lines": []})
    lines = [json.loads(x) for x in (tmp_path / "stream.ndjson").read_text(encoding="utf-8").splitlines()]
    assert [x["k"] for x in lines] == ["enter", "shelf", "pay"] and lines[2]["t"] == 380.5
    import csv
    with open(tmp_path / "npc_actions.csv", encoding="utf-8", newline="") as f:
        actions = list(csv.DictReader(f))
    with open(tmp_path / "visits.csv", encoding="utf-8", newline="") as f:
        visits = list(csv.DictReader(f))
    assert len(actions) == 1 and actions[0]["action"] == "pick"
    assert actions[0]["data_label"] == "SIMULATED" and actions[0]["chosen_product_id"] == "2"
    assert len(visits) == 1 and visits[0]["paid_at"] == "06:20:30"
    stream.checked = 0
    t0 = time.time()
    stream.pace(0, 360.0)                                            # day starts: no wait
    stream.pace(0, 960.0)                                            # 600 minutes later -> ~1 second
    assert 0.7 < time.time() - t0 < 2.5
    (tmp_path / "nowait").touch()
    stream.checked = 0
    t0 = time.time()
    stream.pace(0, 5000.0)                                           # don't wait: full speed
    assert time.time() - t0 < 0.5
    stream.close()


def test_live_pace_does_not_sleep_for_negative_time(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from retina.engine import live

    stream = live.LiveStream(tmp_path, SimpleNamespace(store=None))
    stream.day, stream.base = 0, (0.0, 0.0)
    stream.MINUTES_PER_SECOND = 1.0
    ticks = iter((0.9, 1.1))  # the due time passes while the loop is running
    slept = []
    monkeypatch.setattr(live.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(live.time, "sleep", slept.append)

    stream.pace(0, 1.0)

    assert len(slept) == 1 and 0 < slept[0] <= 0.1
    stream.close()


def test_checkout_csv_is_appended_at_payment(tmp_path):
    import csv
    from types import SimpleNamespace
    from retina.engine import live

    stream = live.LiveStream(tmp_path / "view", SimpleNamespace(store=None))
    stream.date = "2025-09-01"
    stream("pay", 620.5, {"shopper": 7, "trip_id": "D000-S0007", "customer": "C00007",
                           "lines": [{"product": 3, "product_id": "P3", "name": "=unsafe name",
                                      "qty": 2, "price": 3.50, "promo": True}]})
    with open(tmp_path / "checkout_events.csv", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["product_name"] == "'=unsafe name" and rows[0]["line_total"] == "7.0"
    assert rows[0]["data_label"] == "SIMULATED" and rows[0]["pay_minute"] == "620.5"
    assert rows[0]["list_brain"] == "rule" and rows[0]["shelf_brain"] == "rule"
    stream.close()


def test_live_clock_starts_at_selected_time_and_exports_npc_actions(tmp_path, monkeypatch):
    import csv
    import json
    from retina.engine import config, live

    monkeypatch.setattr(config, "LIVE_START_MINUTE", 9 * 60)
    monkeypatch.setitem(config.TRAFFIC, "npc_limit", 12)
    monkeypatch.setattr(live.LiveStream, "MINUTES_PER_SECOND", 1_000_000.0)
    sim = Simulation(days=1, verbose=False)
    watcher = live.LiveStream(tmp_path / "view", sim)
    sim.on_event = watcher
    old, E.ENGINE = E.ENGINE, "realtime"
    try:
        sales, trips = sim.run_day()
    finally:
        E.ENGINE = old
        watcher.close()
    events = [json.loads(line) for line in (tmp_path / "view" / "stream.ndjson").read_text(encoding="utf-8").splitlines()]
    assert events[0]["k"] == "day" and events[0]["t"] == 540
    assert events[-1]["k"] == "day_end"
    enters = [event for event in events if event["k"] == "enter"]
    assert 0 < len(enters) <= 12 and all(event["t"] >= 540 for event in enters)
    with open(tmp_path / "sales.csv", encoding="utf-8", newline="") as f:
        paid = list(csv.DictReader(f))
    with open(tmp_path / "visits.csv", encoding="utf-8", newline="") as f:
        visits = list(csv.DictReader(f))
    with open(tmp_path / "npc_actions.csv", encoding="utf-8", newline="") as f:
        actions = list(csv.DictReader(f))
    assert len(paid) == len(sales) and len(visits) == len(trips) == len(enters)
    assert actions and {a["action"] for a in actions} >= {"move", "pick"}
    assert all(a["data_label"] == "SIMULATED" for a in actions)
    assert all(a["chosen_product_name"] for a in actions if a["action"] == "pick")
    assert all(v["paid_at"] >= v["entered_at"] for v in visits)
