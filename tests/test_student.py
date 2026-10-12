"""The student model (v2 Phase 4) - no API key, no internet, temp folders only.

FakeTeacher (fixed rules) labels a small situation pool; the student must learn it, export to ONNX
with the same answers, and run inside a mix store. Collect mode must stop at 80% of the free day
and never keep labels from another model than the one teacher."""

import json
import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from retina.engine import llm
from retina.engine.engine import Simulation
from retina.mind import mind as M
from retina.student import collect as C
from retina.student.features import FEATURES, matrix
from retina.student.labels import LabelStore
from retina.student.train import train


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    folder = tmp_path_factory.mktemp("student")
    pool = C.build_pool(folder / "pool.jsonl.gz", days=(3,), panel=60, say=lambda *a: None)
    store = LabelStore(folder / "labels.sqlite")
    C.collect(400, provider="fake", store=store, pool=pool, say=lambda *a: None)
    C.collect(30, repeat=1, provider="fake", store=store, pool=pool, say=lambda *a: None)
    meta = train(store, "fake", "rules-v1", out=folder / "model", say=lambda *a: None, min_sets=50)
    return folder, pool, store, meta


def test_student_learns_the_teacher_and_onnx_matches(trained):
    folder, _, store, meta = trained
    r = meta["test"]
    best = r[meta["model"]]
    assert best["agreement"] > r["random"]["agreement"] + 0.3 and best["agreement"] > r["engine"]["agreement"]
    assert meta["onnx_max_diff"] < 1e-3
    assert meta["self_agreement"]["rate"] == 1.0 and meta["self_agreement"]["items"] == 30   # fixed rules
    assert json.loads((folder / "model" / "student.json").read_text())["features"] == FEATURES
    assert set(meta["position_bias"]) <= set("ABCDE")
    assert {"repeat_share", "deal_share", "elasticity"} <= set(best)
    assert (folder / "model" / "MODEL_CARD.md").read_text(encoding="utf-8").startswith("# RETINA student model")


def test_position_is_not_a_feature(trained):
    _, pool, _, _ = trained
    it, t = pool[0]["items"][0], pool[0]
    X1, o1 = matrix(t["npc_x"], it["route"], t["mission"], it["options"], it["engine_product"])
    swapped = [dict(o, letter=l) for o, l in zip(it["options"], reversed([o["letter"] for o in it["options"]]))]
    X2, _ = matrix(t["npc_x"], it["route"], t["mission"], swapped, it["engine_product"])
    assert np.array_equal(X1, X2)


def test_mix_run_with_student_only_needs_no_llm(trained, tmp_path):
    folder, _, _, _ = trained
    sim = Simulation(verbose=False)
    sim.mind = M.Mind(sim, "mix", tmp_path / "run", 25, provider="auto", cache_file=tmp_path / "c.sqlite",
                      say=lambda *a: None, student="only", student_dir=folder / "model")
    sim.mind.teacher = None                                         # any LLM call would crash
    for _ in range(3):
        day = sim.day
        sales = sim.run_day()[0]
        sim.mind.end_day(day)
    d = pd.DataFrame(sim.mind.memory.decisions())
    assert set(d["mind"]) <= {"autopilot", "student", "engine"} and (d["mind"] == "student").any()
    assert (d.loc[d["mind"] == "engine", "why"] == "nothing in stock to choose").all()   # all sold out
    assert "student" in set(sales["mind"]) or len(sales) > 0
    sim.mind.close()


class Never:
    name, offline, calls = "never", False, 0

    def ask(self, batch):
        raise AssertionError("the teacher must not be asked")


class OtherModel:
    name, offline = "other", False

    def ask(self, batch):
        return {(1, 1): {"choice": "A", "product": batch[0]["items"][0]["options"][0]["product"],
                         "why_type": "habit", "why": "x"}}, "gemini", "gemini-x", None


def test_collect_stops_at_80_percent_of_the_free_day(trained, tmp_path, monkeypatch):
    _, pool, _, _ = trained
    monkeypatch.setenv("RETINA_LIMITS_GROQ", "rpm=1000,tpm=1000000,rpd=100,tpd=100000")
    usage = tmp_path / "usage.json"
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    usage.write_text(json.dumps({"day": today, "providers": {"groq": {"requests": 10, "tokens": 79_000}}}))
    monkeypatch.setattr(llm, "PACER", llm.Pacer(usage))
    store = LabelStore(tmp_path / "labels.sqlite")
    said = []
    got = C.collect(100, provider="groq", store=store, pool=pool, say=said.append, teacher=Never())
    assert got == 0 and "80% stop" in said[-1] and "run again tomorrow" in said[-1]


def test_collect_keeps_only_the_one_teacher(trained, tmp_path, monkeypatch):
    _, pool, _, _ = trained
    monkeypatch.setattr(llm, "PACER", llm.Pacer(tmp_path / "usage.json"))
    store = LabelStore(tmp_path / "labels.sqlite")
    said = []
    C.collect(100, provider="groq", store=store, pool=pool[:3], say=said.append, teacher=OtherModel())
    assert store.rows(0, valid_only=False) == [] and "labels not saved" in said[-1]
