"""List scenarios and teacher lists (retina/student/listplan.py): real candidates from the NPC's own state,
answers checked, a paste pack round trip. Small store run, no LLM."""

import json

import pytest

from retina.engine.engine import Simulation, shelf_prices
from retina.engine.config import SENSITIVITY_GROUPS, stream
from retina.student import listplan as L


@pytest.fixture(scope="module")
def pool(tmp_path_factory):
    return L.build_pool(tmp_path_factory.mktemp("lists") / "pool.jsonl.gz", days=(14,), npcs=24, say=lambda *a: None)


def test_candidates_are_real_and_from_the_npcs_state():
    sim = Simulation(verbose=False)
    while sim.day < 14:
        sim.run_day()
    prices, regular, promo, _ = shelf_prices(14, sim.store, sim.book, sim.iv, SENSITIVITY_GROUPS)
    s = next(s for s in range(len(sim.shoppers.shopper_id)) if len(sim.shoppers.counts[s] or {}) >= 5)
    c = L.candidates(sim, s, 14, prices[0], regular, promo[0], stream(L.STREAM_LIST, 14, s))
    assert 8 <= len(c) <= L.MAX_CANDIDATES and len({x["product"] for x in c}) == len(c)
    names = sim.store.products["product_name"]
    for x in c:
        assert sim.inv.sellable[x["product"]] and x["name"] == L.clean(names.iat[x["product"]])
        assert x["times"] == (sim.shoppers.counts[s] or {}).get(x["product"], 0)
        assert x["source"] in L.PER_SOURCE and (x["source"] != "usual" or x["times"] > 0)
    again = L.candidates(sim, s, 14, prices[0], regular, promo[0], stream(L.STREAM_LIST, 14, s))
    assert [x["product"] for x in again] == [x["product"] for x in c]          # same dice, same scenario


def test_list_student_trains_exports_and_plans_npc_lists(pool, tmp_path):
    from retina.engine.config import stream as dice
    from retina.mind.mind import Mind
    from retina.student.listmodel import ListStudent
    from retina.student.listtrain import train
    labels = L.ListLabels(tmp_path / "labels.sqlite")
    big = L.build_pool(tmp_path / "pool2.jsonl.gz", days=(14, 21, 28), npcs=60, say=lambda *a: None)
    L.collect(len(big), provider="fake", labels=labels, pool=big, say=lambda *a: None)
    meta = train(labels.rows(), big, out=tmp_path / "model", say=lambda *a: None, min_lists=30)
    assert meta["test"]["auc"] > 0.9 and abs(meta["test"]["student"]["items_per_list"] -
                                              meta["test"]["teacher"]["items_per_list"]) < 3
    ls = ListStudent(tmp_path / "model")                               # ONNX: same choices for the same dice
    a = [(c["product"], q) for c, q in ls.plan(big[0], dice(1, 2, 3))]
    assert a == [(c["product"], q) for c, q in ls.plan(big[0], dice(1, 2, 3))] and all(1 <= q <= 6 for _, q in a)
    sim = Simulation(verbose=False)                                    # the NPC's list, planned by the mind
    while sim.day < 14:
        sim.run_day()
    prices, regular, promo, _ = shelf_prices(14, sim.store, sim.book, sim.iv, SENSITIVITY_GROUPS)
    m = Mind.__new__(Mind)
    m.sim, m.lister = sim, ls
    trip = {"budget": float("inf"), "plan": []}
    s = next(s for s in range(len(sim.shoppers.shopper_id)) if len(sim.shoppers.counts[s] or {}) >= 5)
    m._plan_list(s, 14, {"prices_g": prices, "regular": regular, "promo_g": promo}, trip)
    assert trip["list_by"] == "list student" and len(trip["plan"]) == trip["planned"]
    assert all(len(item) == 6 and 1 <= item[5] <= 6 and item[1] in ("need", "repeat", "partner", "new")
               for item in trip["plan"])


def test_reply_is_checked(pool):
    group = pool[:2]
    a, b = group[0]["candidates"], group[1]["candidates"]
    reply = json.dumps([{"s": 1, "n": 1, "q": 2}, {"s": 1, "n": 99, "q": 1}, {"s": 3, "n": 1, "q": 1},
                        {"s": 2, "n": 2, "q": 40}, {"s": "x", "n": 1}])
    got = L.parse_reply("Sure! " + reply, group)
    assert got == {1: {a[0]["product"]: 2}, 2: {b[1]["product"]: 6}}             # no number 99, no shopper 3, q <= 6
    assert L.parse_reply("no json here", group) == {}
    gemini = '```json\n[{"s": "S1", "n": 1, "q": 1}, {"s": "s2", "n": "2", "q": 1}]\n```'   # real Gemini style
    assert L.parse_reply(gemini, group) == {1: {a[0]["product"]: 1}, 2: {b[1]["product"]: 1}}


def test_collect_and_paste_round_trip(pool, tmp_path):
    labels = L.ListLabels(tmp_path / "labels.sqlite")
    assert L.collect(8, provider="fake", labels=labels, pool=pool, say=lambda *a: None) >= 8
    assert {r["teacher"] for r in labels.rows()} == {"fake/rules-v1"}
    folder = tmp_path / "paste"
    n = L.paste_pack(8, folder, labels=labels, pool=pool, say=lambda *a: None)
    meta = json.loads((folder / "pack.json").read_text(encoding="utf-8"))
    by_id = {t["id"]: t for t in pool}
    assert n == 2 and "Use ONLY the numbered products" in (folder / "message_01.txt").read_text(encoding="utf-8")
    ids = meta["messages"][0]["ids"]
    (folder / "reply_01.txt").write_text(json.dumps([{"s": k, "n": 1, "q": 1} for k in range(1, len(ids) + 1)]),
                                         encoding="utf-8")
    read, missing = L.paste_read("chatgpt/test", folder, labels=labels, pool=pool, say=lambda *a: None)
    assert read == len(ids) and missing == [2]
    r = next(r for r in labels.rows("chatgpt/test") if r["id"] == ids[0])
    assert r["list"] == {str(by_id[ids[0]]["candidates"][0]["product"]): 1}
