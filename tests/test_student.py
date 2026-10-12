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


class TooLarge:
    """Like Groq's free tier: refuses (413) any request with more than one trip in it."""
    name, offline, failures, sizes = "groq-like", False, 0, []

    def ask(self, batch):
        self.sizes.append(len(batch))
        if len(batch) > 1:
            self.failures += 1
            return {}, None, None, "groq answered with error 413: Request too large"
        a = {(1, i): {"choice": it["options"][0]["letter"] if it["options"][0]["in_stock"] else "SKIP",
                      "product": it["options"][0]["product"] if it["options"][0]["in_stock"] else -1,
                      "why_type": "habit", "why": "x"} for i, it in enumerate(batch[0]["items"], 1)}
        return a, "groq", "openai/gpt-oss-120b", None


def test_too_large_requests_are_split_not_fatal(trained, tmp_path, monkeypatch):
    _, pool, _, _ = trained
    monkeypatch.setattr(llm, "PACER", llm.Pacer(tmp_path / "usage.json"))
    store, teacher = LabelStore(tmp_path / "labels.sqlite"), TooLarge()
    items = sum(len(t["items"]) for t in pool[:6])
    C.collect(10_000, provider="groq", store=store, pool=pool[:6], say=lambda *a: None, teacher=teacher)
    assert len(store.rows(0, valid_only=False)) == items and max(teacher.sizes) > 1   # every trip labelled


class Unreadable:
    name, offline = "groq-like", False

    def ask(self, batch):
        return {}, "groq", "openai/gpt-oss-120b", None      # answered, but nothing could be read


def test_unreadable_reply_is_asked_again_later(trained, tmp_path, monkeypatch):
    _, pool, _, _ = trained
    monkeypatch.setattr(llm, "PACER", llm.Pacer(tmp_path / "usage.json"))
    store = LabelStore(tmp_path / "labels.sqlite")
    C.collect(100, provider="groq", store=store, pool=pool[:3], say=lambda *a: None, teacher=Unreadable())
    assert store.rows(0, valid_only=False) == []                  # nothing marked as done


class SecondTeacher:
    """Picks the LAST in-stock option - so it agrees with the fixed-rule teacher only sometimes."""
    name, offline = "gemini-like", False

    def ask(self, batch):
        out = {}
        for n, b in enumerate(batch, 1):
            for i, it in enumerate(b["items"], 1):
                o = [x for x in it["options"] if x["in_stock"]][-1]
                out[(n, i)] = {"choice": o["letter"], "product": o["product"], "why_type": "other", "why": "x"}
        return out, "gemini", "gemini-test", None


def test_second_teacher_comparison_keeps_both_labels(trained, tmp_path, monkeypatch):
    _, pool, _, _ = trained
    from retina.student import labels as L
    monkeypatch.setattr(llm, "PACER", llm.Pacer(tmp_path / "usage.json"))
    monkeypatch.setattr(C, "TEACHER", ("fake", "rules-v1"))          # the "base" teacher for this test
    monkeypatch.setitem(llm.PROVIDERS, "gemini", dict(llm.PROVIDERS["gemini"], model="gemini-test"))
    store = LabelStore(tmp_path / "labels.sqlite")
    C.collect(60, provider="fake", store=store, pool=pool, say=lambda *a: None)
    out = C.compare("gemini", 40, store=store, pool=pool, say=lambda *a: None, teacher=SecondTeacher())
    base, other = store.rows(0, "fake", "rules-v1"), store.rows(0, "gemini", "gemini-test")
    assert len(other) == 40 and {r["id"] for r in other} <= {r["id"] for r in base}    # same situations
    assert {r["teacher"] for r in other} == {"gemini/gemini-test"}                     # teacher on every label
    assert len(base) >= 60                                                             # base labels kept
    assert out["situations"] == 40 and 0 <= out["agreement"] < 1
    assert {"repeat_share", "deal_share"} <= set(out["fake"]) and "pending" in out["decision"]


def test_export_then_import_another_teacher(trained, tmp_path, monkeypatch):
    _, pool, _, _ = trained
    monkeypatch.setattr(C, "TEACHER", ("fake", "rules-v1"))
    store = LabelStore(tmp_path / "labels.sqlite")
    C.collect(40, provider="fake", store=store, pool=pool, say=lambda *a: None)
    out = tmp_path / "export.jsonl"
    assert C.export_situations(out, 30, "labelled", store=store, pool=pool, say=lambda *a: None) == 30
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    items = [it for line in lines for it in line["items"]]
    labelled = [r["id"] for r in store.rows(0, "fake", "rules-v1")][:30]
    assert sorted(it["id"] for it in items) == sorted(labelled)                       # the same 30 situations
    assert all("Reply as ONE JSON list" in line["prompt"] and "choice" not in json.dumps(line["items"]) for line in lines)
    # answers: good, wrong letter, sold out, unknown id, no teacher, claims the training teacher, twice, raw reply
    first, second = items[0], items[1]
    ok = next(o["letter"] for o in first["options"] if o["in_stock"])
    gone = next((o["letter"] for o in second["options"] if not o["in_stock"]), "Z")
    rows = [{"id": first["id"], "teacher": "human/me", "choice": ok, "reason_type": "price", "why": "cheap"},
            {"id": second["id"], "teacher": "human/me", "choice": "Q"},
            {"id": second["id"], "teacher": "human/me", "choice": gone},
            {"id": "c-nope", "teacher": "human/me", "choice": "A"},
            {"id": first["id"], "choice": "A"},
            {"id": first["id"], "teacher": "fake/rules-v1", "choice": ok},
            {"id": first["id"], "teacher": "human/me", "choice": ok}]
    raw = dict(lines[-1], teacher="other/llm-x",
               reply=json.dumps([{"shopper": 1, "item": i, "choice": "SKIP", "reason_type": "budget", "why": "x"}
                                 for i in range(1, len(lines[-1]["items"]) + 1)]))
    answers = tmp_path / "answers.jsonl"
    answers.write_text("\n".join(json.dumps(r) for r in rows + [raw]) + "\nnot json\n", encoding="utf-8")
    n, problems = C.import_labels(answers, store=store, pool=pool, say=lambda *a: None)
    assert n == 1 + len(lines[-1]["items"]) and len(problems) == 7
    me = store.rows(0, "human", "me")
    assert len(me) == 1 and me[0]["teacher"] == "human/me" and me[0]["source"] == "import"
    assert store.rows(0, "other", "llm-x")[0]["choice"] == -1                         # SKIP kept as a SKIP
    report = C.compare_report(store, "other", "llm-x", say=lambda *a: None)
    assert report["situations"] == len(lines[-1]["items"])


def test_collect_keeps_only_the_one_teacher(trained, tmp_path, monkeypatch):
    _, pool, _, _ = trained
    monkeypatch.setattr(llm, "PACER", llm.Pacer(tmp_path / "usage.json"))
    store = LabelStore(tmp_path / "labels.sqlite")
    said = []
    C.collect(100, provider="groq", store=store, pool=pool[:3], say=said.append, teacher=OtherModel())
    assert store.rows(0, valid_only=False) == [] and "labels not saved" in said[-1]
