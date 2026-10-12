"""The mind layer (v2 Phase 2) - no API key, no internet, nothing written outside a temp folder.

Fake teachers stand in for the LLM: FakeTeacher (fixed rules), a random one (to prove the cache
makes reruns identical) and a broken one (to prove the engine takes over)."""

import pickle

import numpy as np
import pandas as pd
import pytest

from retina.engine.engine import Simulation
from retina.engine.llm import Pacer
from retina.memory.store import MemoryStore
from retina.mind import mind as M
from retina.mind.teacher import parse_answers

DAYS, PANEL = 4, 25


@pytest.fixture(scope="module")
def fresh():
    """A new store, frozen as bytes: every test thaws its own copy (building it takes a few seconds)."""
    return pickle.dumps(Simulation(verbose=False))


def run(fresh, run_dir, cache, chunks=(DAYS,), teacher=None):
    """Run DAYS days with a mix mind, stopping and reloading (pickle + new Mind) after each chunk."""
    sim, sales, done = pickle.loads(fresh), [], 0
    for n in chunks:
        sim.mind = M.Mind(sim, "mix", run_dir, PANEL, provider="fake", cache_file=cache, say=lambda *a: None)
        if teacher is not None:
            sim.mind.teacher = teacher
        for _ in range(n):
            day = sim.day
            sales.append(sim.run_day()[0])
            sim.mind.end_day(day)
        done += n
        sim.mind.close()
        sim = pickle.loads(pickle.dumps(sim))          # like live.py saving and loading the store
    memory = MemoryStore(run_dir / "memory.sqlite")
    return sim, pd.concat(sales, ignore_index=True), memory


class RandomTeacher:
    """Answers at random - two runs only agree if the second one reads the first one's cache."""
    name, offline = "random", False

    def __init__(self, seed):
        self.rng, self.calls = np.random.default_rng(seed), 0

    def ask(self, batch):
        self.calls += 1
        out = {}
        for n, b in enumerate(batch, 1):
            for i, it in enumerate(b["items"], 1):
                ok = [o for o in it["options"] if o["in_stock"]]
                if ok:
                    o = ok[int(self.rng.integers(len(ok)))]
                    out[(n, i)] = {"choice": o["letter"], "product": o["product"], "why_type": "other", "why": "random"}
        return out, "random", "test", None


class BrokenTeacher:
    name, offline, calls = "broken", False, 0

    def ask(self, batch):
        return {}, None, None, "no provider could answer"


def test_mix_run_stop_continue_equals_one_run(fresh, tmp_path):
    cache = tmp_path / "cache.sqlite"
    _, one, mem1 = run(fresh, tmp_path / "one", cache)
    _, split, mem2 = run(fresh, tmp_path / "split", cache, chunks=(2, 2))
    assert one.equals(split)
    d1, d2 = pd.DataFrame(mem1.decisions()), pd.DataFrame(mem2.decisions())
    cols = [c for c in d1.columns if c != "cached"]                # the 2nd run reads the cache
    assert d1[cols].equals(d2[cols])
    assert {"mind", "why_type", "why"} <= set(one.columns)


def test_cache_makes_a_rerun_identical(fresh, tmp_path):
    cache = tmp_path / "cache.sqlite"
    first = RandomTeacher(1)
    _, a, _ = run(fresh, tmp_path / "a", cache, teacher=first)
    second = RandomTeacher(2)                                      # would answer differently...
    _, b, _ = run(fresh, tmp_path / "b", cache, teacher=second)
    assert first.calls > 0 and second.calls == 0                   # ...but is never asked
    assert a.equals(b)


def test_panel_memory_and_decision_log(fresh, tmp_path):
    sim, sales, memory = run(fresh, tmp_path / "run", tmp_path / "cache.sqlite")
    panel = set(memory.panel())
    assert len(panel) == PANEL
    thinking = sales[sales["shopper_id"].isin(panel)]
    assert set(sales.loc[~sales["shopper_id"].isin(panel), "mind"]) == {"engine"}   # the crowd is the engine
    assert thinking["mind"].isin(["llm", "autopilot", "engine", "llm_fallback"]).all()
    d = pd.DataFrame(memory.decisions())
    assert set(d["npc"]) <= panel and len(d) > 0
    assert (d.loc[d["mind"] == "autopilot", "trigger"] == "routine").all()
    assert (d.loc[d["mind"] == "llm", "trigger"] != "routine").all()   # every LLM choice is logged with why
    llm_lines = thinking[thinking["mind"] == "llm"]
    assert len(llm_lines) > 0 and (llm_lines["why"] != "").all()
    # a returning NPC: same card, and their memory holds their earlier trips
    npc = thinking.groupby("shopper_id")["day"].nunique().idxmax()
    card = memory.card(npc)
    assert card["facts"]["shopper_id"] == npc
    trips = thinking.loc[thinking["shopper_id"] == npc, "day"].nunique()
    assert len(memory.last_trips(npc, DAYS)) == min(trips, 2)     # last_trips gives the latest 2


def test_broken_teacher_falls_back_to_the_engine(fresh, tmp_path):
    _, sales, memory = run(fresh, tmp_path / "run", tmp_path / "cache.sqlite", teacher=BrokenTeacher())
    d = pd.DataFrame(memory.decisions())
    asked = d[d["trigger"] != "routine"]
    assert len(asked) > 0 and (asked["mind"] == "llm_fallback").all() and (asked["valid"] == 0).all()
    assert (d["product"] == d["engine_product"]).all()             # the engine's own pick stands
    assert len(sales) > 0


def test_big_batches_are_split_under_the_token_cap(fresh, tmp_path, monkeypatch):
    class Counting(RandomTeacher):
        def ask(self, batch):
            self.sizes.append(len(batch))
            return super().ask(batch)
    teacher = Counting(3)
    teacher.sizes = []
    monkeypatch.setattr(M, "MAX_REQUEST_TOKENS", 1)                 # nothing fits two NPCs
    run(fresh, tmp_path / "run", tmp_path / "cache.sqlite", teacher=teacher)
    assert teacher.sizes and set(teacher.sizes) == {1}              # one NPC per request


def test_memory_rollback_forgets_an_unsaved_day(tmp_path):
    m = MemoryStore(tmp_path / "memory.sqlite")
    m.add_card(7, {"facts": {"name": "Test N."}, "persona": {}}, 0)
    m.add_trip(7, 0, 9, "routine", 5, 20.0, 50.0, "milk")
    m.add_purchases([(7, 0, 42, "Milk", "milk", 1, 2.5, 2.5, 0, "llm", "habit", "usual")])
    m.commit(0)
    m.add_trip(7, 1, 9, "routine", 3, 10.0, 50.0, "bread")        # day 1 never saved
    m.rollback_to(1)
    assert len(m.last_trips(7, 5)) == 1
    assert m.recall(7, [42], 5)[42]["last_price"] == 2.5
    assert m.card(7)["facts"]["name"] == "Test N."


def test_parse_answers_keeps_only_valid_choices():
    batch = [{"items": [{"options": [{"letter": "A", "product": 1, "in_stock": True},
                                     {"letter": "B", "product": 2, "in_stock": False}]},
                        {"options": [{"letter": "A", "product": 3, "in_stock": True}]}]}]
    text = ('Here: [{"shopper": 1, "item": 1, "choice": "B"}, {"shopper": 1, "item": 2, "choice": "a", '
            '"reason_type": "Deal", "why": "cheap"}, {"shopper": 2, "item": 1, "choice": "A"}]')
    out = parse_answers(text, batch)
    assert (1, 1) not in out                                       # B is sold out
    assert out[(1, 2)] == {"choice": "A", "product": 3, "why_type": "deal", "why": "cheap"}
    assert (2, 1) not in out                                       # no shopper 2 in this batch
    assert parse_answers("no json at all", batch) == {}


def test_pacer_waits_under_per_minute_limits(tmp_path, monkeypatch):
    now = [1_000_000.0]
    slept = []

    def sleep(s):
        slept.append(s)
        now[0] += s
    monkeypatch.setenv("RETINA_LIMITS_GROQ", "rpm=2,tpm=1000,rpd=3,tpd=none")
    p = Pacer(tmp_path / "usage.json", clock=lambda: now[0], sleep=sleep)
    for _ in range(2):
        p.wait("groq", 100, say=lambda *a: None)
        p.spent("groq", 100)
    assert not slept                                               # 2 requests fit in a minute
    p.wait("groq", 100, say=lambda *a: None)                       # the 3rd waits for the window
    p.spent("groq", 100)
    assert sum(slept) >= 59
    with pytest.raises(RuntimeError, match="daily limit"):         # rpd=3 used up
        p.wait("groq", 100, say=lambda *a: None)
    monkeypatch.setenv("RETINA_LIMITS_GROQ", "rpm=100,tpm=250,rpd=100")
    q = Pacer(tmp_path / "usage2.json", clock=lambda: now[0], sleep=sleep)
    slept.clear()
    q.wait("groq", 200, say=lambda *a: None)
    q.wait("groq", 200, say=lambda *a: None)                       # 400 tokens > 250 a minute: waits
    assert sum(slept) >= 59
