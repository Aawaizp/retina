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
    order = [p["trip_id"] for p in pays]                                   # receipts in paying order
    assert list(dict.fromkeys(sales["trip_id"])) == order


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


def test_the_live_stream_tells_the_whole_story(day_run):
    """enter carries the route; shelves carry what was done (with names); pay carries the receipt lines."""
    _, sales, _, events = day_run
    assert events[0][0] == "day" and events[0][2]["date"]
    enter = next(d for k, _, d in events if k == "enter")
    assert len(enter["path"]) >= 3 and enter["mission"] in ("top_up", "routine", "stock_up")
    shelf = [x for k, _, d in events if k == "shelf" for x in d["events"]]
    assert shelf and all(x["name"] for x in shelf if x["product"] >= 0)
    assert all("price" in x for x in shelf if x["event"] == "PURCHASE")
    pays = [d for k, _, d in events if k == "pay"]
    assert all(x["name"] and x["qty"] > 0 for p in pays for x in p["lines"])
    assert sum(len(p["lines"]) for p in pays) == len(sales)
    assert {k for k, _, _ in events} == {"day", "enter", "shelf", "checkout", "pay", "leave"}


def test_live_stream_file_and_pacing(tmp_path, monkeypatch):
    import json
    import time
    from types import SimpleNamespace
    from retina.engine import live
    stream = live.LiveStream(tmp_path, SimpleNamespace(store=None))
    stream("enter", 360.0, {"shopper": 1, "path": [[360.0, 1.0, 2.0]]})
    stream("pay", 380.5, {"shopper": 1, "lines": []})
    lines = [json.loads(x) for x in (tmp_path / "stream.ndjson").read_text(encoding="utf-8").splitlines()]
    assert [x["k"] for x in lines] == ["enter", "pay"] and lines[1]["t"] == 380.5
    (tmp_path / "speed.txt").write_text("600", encoding="utf-8")     # 600 store minutes per second
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
