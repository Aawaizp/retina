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


def test_student_choices_are_what_the_shopper_buys(trained, tmp_path):
    """The student's pick must reach the basket (a bug let only 'llm' decisions change purchases)."""
    folder, _, _, _ = trained
    sim = Simulation(verbose=False)
    sim.mind = M.Mind(sim, "mix", tmp_path / "run", 25, provider="auto", cache_file=tmp_path / "c.sqlite",
                      say=lambda *a: None, student="only", student_dir=folder / "model")
    sim.mind.teacher = None
    bought = []
    for _ in range(4):
        day = sim.day
        sales = sim.run_day()[0]
        sim.mind.end_day(day)
        bought.append(sales[sales["mind"] == "student"])
    sales = pd.concat(bought)
    d = pd.DataFrame(sim.mind.memory.decisions())
    own = d[(d["mind"] == "student") & (d["product"] >= 0) & (d["product"] != d["engine_product"])]
    assert len(own) > 0
    have = set(zip(sales["day"], sales["shopper_id"], sales["product_idx"]))
    found = [(r.day, r.npc, r.product) in have for r in own.itertuples()]
    assert np.mean(found) > 0.8          # the rest: sold out at the shelf or over budget
    sim.mind.close()


def student_run(folder, run_dir, days):
    sim = Simulation(verbose=False)
    sim.mind = M.Mind(sim, "student", run_dir, cache_file=run_dir / "c.sqlite", say=lambda *a: None,
                      student_dir=folder / "model")
    sim.mind.teacher = None                                           # any LLM call would crash
    sales = []
    for _ in range(days):
        day = sim.day
        sales.append(sim.run_day()[0])
        sim.mind.end_day(day)
    d = pd.DataFrame(sim.mind.memory.decisions())
    sim.mind.close()
    return pd.concat(sales), d


def test_mind_student_every_npc_decides_without_llm(trained, tmp_path):
    folder, _, _, _ = trained
    sales, d = student_run(folder, tmp_path / "a", 3)
    assert set(sales["mind"]) <= {"autopilot", "student", "engine"} and (sales["mind"] == "student").any()
    assert (sales["mind"] != "engine").mean() > 0.9                   # nearly every line: a thinking NPC decided
    assert set(d["mind"]) <= {"autopilot", "student", "engine"}       # never an LLM
    again, _ = student_run(folder, tmp_path / "b", 3)
    assert sales.reset_index(drop=True).equals(again.reset_index(drop=True))   # same seed, same data


def test_period_choices_and_sold_out_options_left_out_of_the_price_measure():
    from retina.cli import PERIODS, build_parser
    from retina.mind.report import option_sets
    assert PERIODS == {"1w": 7, "2w": 14, "1m": 30, "3m": 91, "6m": 182, "1y": 365}
    assert build_parser().parse_args(["--period", "1y", "--mind", "student"]).period == "1y"
    opt = lambda p, price, stock: {"product": p, "price": price, "regular": 2.0, "times": 0, "w": 1.0, "pop": 1.0,
                                   "in_stock": stock}
    sets, _ = option_sets([{"mind": "student", "product": 1, "options": [opt(1, 2.0, True), opt(2, 1.0, False),
                                                                         opt(3, 2.2, True)]}])
    assert len(sets) == 1 and len(sets[0][0]) == 2                    # the sold-out deal is not an option


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
    assert all("ONE JSON list" in line["prompt"] and line["prompt_version"] == 3
               and "choice" not in json.dumps(line["items"]) for line in lines)
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


def test_consensus_weighting(trained, tmp_path, monkeypatch):
    folder, pool, store, _ = trained
    monkeypatch.setitem(llm.PROVIDERS, "gemini", dict(llm.PROVIDERS["gemini"], model="gemini-test"))
    monkeypatch.setattr(C, "TEACHER", ("fake", "rules-v1"))
    C.compare("gemini", 200, store=store, pool=pool, say=lambda *a: None, teacher=SecondTeacher())
    meta = train(store, "fake", "rules-v1", out=tmp_path / "m", say=lambda *a: None, min_sets=50,
                 consensus="gemini/gemini-test", consensus_weight=3.0)
    w = meta["consensus_weighting"]
    assert w["agree"] + w["disagree"] > 0 and w["agree"] + w["disagree"] + w["not_seen"] == meta["sets"]["train"]
    assert "Consensus weighting" in (tmp_path / "m" / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert meta["onnx_max_diff"] < 1e-3


def test_paste_pack_round_trip(trained, tmp_path, monkeypatch):
    _, pool, _, _ = trained
    monkeypatch.setattr(C, "TEACHER", ("fake", "rules-v1"))
    store = LabelStore(tmp_path / "labels.sqlite")
    C.collect(30, provider="fake", store=store, pool=pool, say=lambda *a: None)
    folder = tmp_path / "paste"
    msgs, items = C.paste_pack(folder, same_as="fake/rules-v1", per_message=2, store=store, pool=pool, say=lambda *a: None)
    assert msgs >= 2 and items == len(store.rows(0, "fake", "rules-v1"))
    meta = json.loads((folder / "pack.json").read_text(encoding="utf-8"))
    by_id = {it["id"]: it for t in pool for it in t["items"]}
    assert "Role-play" in (folder / "message_01.txt").read_text(encoding="utf-8")
    for m in meta["messages"][:-1]:                                   # last reply left empty
        reply = []
        for s, t in enumerate(m["trips"], 1):
            for i, sid in enumerate(t["ids"], 1):
                o = [x for x in by_id[sid]["options"] if x["in_stock"]]
                reply.append({"s": s, "i": i, "c": o[0]["letter"] if o else "SKIP", "r": "habit", "w": "usual"})
        reply[0]["c"] = "Q"                                           # one bad letter
        (folder / f"reply_{m['message']:02d}.txt").write_text("Here you go:\n" + json.dumps(reply), encoding="utf-8")
    said = []
    n, missing, unread = C.paste_read(folder, "chatgpt/test", store=store, pool=pool, say=said.append)
    last = meta["messages"][-1]["message"]
    expected = sum(len(t["ids"]) for m in meta["messages"][:-1] for t in m["trips"]) - (len(meta["messages"]) - 1)
    assert n == expected and missing == [last] and len(unread) == len(meta["messages"]) - 1
    assert {r["teacher"] for r in store.rows(0, "chatgpt", "test")} == {"chatgpt/test"}
    n2, _, _ = C.paste_read(folder, "chatgpt/test", store=store, pool=pool, say=lambda *a: None)
    assert n2 == 0                                                    # nothing imported twice


def test_extra_teachers_add_training_sets_only(trained, tmp_path, monkeypatch):
    folder, pool, store, base_meta = trained
    monkeypatch.setitem(llm.PROVIDERS, "gemini", dict(llm.PROVIDERS["gemini"], model="gemini-test"))
    monkeypatch.setattr(C, "TEACHER", ("fake", "rules-v1"))
    C.compare("gemini", 200, store=store, pool=pool, say=lambda *a: None, teacher=SecondTeacher())
    meta = train(store, "fake", "rules-v1", out=tmp_path / "m", say=lambda *a: None, min_sets=50,
                 teachers=["gemini/gemini-test"])
    added = meta["extra_teachers"]["gemini/gemini-test"]
    assert added > 0 and meta["sets"]["train"] == base_meta["sets"]["train"] + added
    assert meta["sets"]["val"] == base_meta["sets"]["val"] and meta["sets"]["test"] == base_meta["sets"]["test"]
    assert "Extra teachers in training only" in (tmp_path / "m" / "MODEL_CARD.md").read_text(encoding="utf-8")


def test_student_draws_by_probability_and_is_repeatable(trained):
    folder, pool, _, meta = trained
    from retina.engine.config import stream
    from retina.student.model import Student
    st = Student(folder / "model")
    assert meta["pick"] == "draw" and st.pick == "draw"
    t = next(t for t in pool for it in t["items"] if sum(o["in_stock"] for o in it["options"]) >= 3)
    it = next(it for it in t["items"] if sum(o["in_stock"] for o in it["options"]) >= 3)
    args = (t["npc_x"], it["route"], t["mission"], it["options"], it["engine_product"])
    picks = [st.choose(*args, rng=stream(12, 1, 2, n))[0]["product"] for n in range(300)]
    top, conf, p = st.choose(*args)                                   # no dice -> the most likely option
    assert conf == max(p) and top["product"] == max(set(picks), key=picks.count) or max(p) < 0.6
    if max(p) < 0.95:
        assert len(set(picks)) > 1                                    # not always the same product
    again = [st.choose(*args, rng=stream(12, 1, 2, n))[0]["product"] for n in range(300)]
    assert picks == again                                             # same dice -> same choices
    st.pick = "top"
    assert {st.choose(*args, rng=stream(12, 1, 2, n))[0]["product"] for n in range(20)} == {top["product"]}


def test_train_on_all_teachers(trained, tmp_path, monkeypatch):
    folder, pool, store, _ = trained
    monkeypatch.setitem(llm.PROVIDERS, "gemini", dict(llm.PROVIDERS["gemini"], model="gemini-test"))
    monkeypatch.setattr(C, "TEACHER", ("fake", "rules-v1"))
    C.compare("gemini", 200, store=store, pool=pool, say=lambda *a: None, teacher=SecondTeacher())
    meta = train(store, "all", None, out=tmp_path / "m", say=lambda *a: None, min_sets=50)
    assert set(meta["labels_by_teacher"]) >= {"fake/rules-v1", "gemini/gemini-test"}
    assert "fake/rules-v1" in meta["test_by_teacher"] and set(meta["test_by_teacher"]) <= set(meta["labels_by_teacher"])
    assert "repeat_share_natural" in meta["test"]["conditional logit"]
    card = (tmp_path / "m" / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "Teachers: **all**" in card and meta["onnx_max_diff"] < 1e-3


def test_family_balance_exclude_and_realism_choice(trained, tmp_path, monkeypatch):
    folder, pool, store, _ = trained
    from retina.student.train import balance_by_family, family_of, realism_gap
    assert [family_of(t) for t in ("groq/openai/gpt-oss-120b", "chatgpt/GPT-5", "claude-code/opus-5-5",
                                   "gemini/gemini-3.5-flash-lite", "fake/rules-v1")] == \
        ["openai", "openai", "anthropic", "google", "fake"]
    sets = [{"teacher": t, "weight": 1.0}
            for t in ["groq/x"] * 6 + ["claude/y"] * 2]
    balance_by_family(sets)
    assert abs(sum(d["weight"] for d in sets if d["teacher"] == "groq/x")
               - sum(d["weight"] for d in sets if d["teacher"] == "claude/y")) < 1e-9
    assert realism_gap({"repeat_share_natural_drawn": 0.6, "elasticity": -2.5}) == 0
    monkeypatch.setitem(llm.PROVIDERS, "gemini", dict(llm.PROVIDERS["gemini"], model="gemini-test"))
    monkeypatch.setattr(C, "TEACHER", ("fake", "rules-v1"))
    C.compare("gemini", 200, store=store, pool=pool, say=lambda *a: None, teacher=SecondTeacher())
    meta = train(store, "all", None, out=tmp_path / "m", say=lambda *a: None, min_sets=50,
                 balance="family", choose="realism")
    assert set(meta["balance"]) == {"fake", "google"} and meta["choose"] == "realism"
    gaps = {k: meta["test"][k]["val_realism_gap"] for k in ("conditional logit", "LightGBM ranker")}
    assert gaps[meta["model"]] == min(gaps.values())                  # chosen by realism on validation
    left = train(store, "all", None, out=tmp_path / "m2", say=lambda *a: None, min_sets=50, exclude=("google",))
    assert "gemini/gemini-test" not in left["labels_by_teacher"] and left["excluded"] == ["google"]


def test_price_correction_reaches_the_target_and_onnx_keeps_it(trained, tmp_path):
    folder, pool, store, base = trained
    meta = train(store, "fake", "rules-v1", out=tmp_path / "m", say=lambda *a: None, min_sets=50, price_target=-2.5)
    b = meta["price_boost"]
    assert b["label"] == "ESTIMATED" and b["coef"] >= 0
    if b["coef"] > 0:                                                  # the fake teacher may already be there
        assert abs(b["calibration_elasticity"]["after"] - (-2.5)) < 0.1
        assert b["test_after"]["elasticity"] < b["test_before"]["elasticity"]
    assert meta["onnx_max_diff"] < 1e-3                                # the correction is inside the ONNX file
    assert "Price correction (ESTIMATED)" in (tmp_path / "m" / "MODEL_CARD.md").read_text(encoding="utf-8")


def test_compact_prompt_is_shorter_and_old_labels_keep_their_version(trained, tmp_path, monkeypatch):
    from retina.mind.teacher import PROMPT_VERSION, batch_prompt, parse_answers, request_tokens
    _, pool, _, _ = trained
    t = pool[0]
    old = request_tokens([npc_block := C.blocks_of([t], 2)[0]], 2)
    new = request_tokens(C.blocks_of([t], PROMPT_VERSION), PROMPT_VERSION)
    assert PROMPT_VERSION == 3 and new < 0.8 * old
    short = '[{"s":1,"i":1,"c":"%s","r":"price","w":"cheaper"}]' % t["items"][0]["options"][0]["letter"]
    if t["items"][0]["options"][0]["in_stock"]:
        assert parse_answers(short, [{"items": t["items"]}])[(1, 1)]["why_type"] == "price"   # short keys read
    # labels made with prompt v2 are repeated in v2 (the self-agreement must not mix two prompts)
    store = LabelStore(tmp_path / "labels.sqlite")
    C.collect(30, provider="fake", store=store, pool=pool, say=lambda *a: None)
    old_rows = store.rows(0, "fake", "rules-v1")
    for r in old_rows:
        r["prompt_version"] = 2
    store.add(old_rows)

    class Spy(C.FakeTeacher):
        versions = []

        def ask(self, batch):
            self.versions.append(batch[0]["version"])
            return super().ask(batch)
    spy = Spy()
    C.collect(10, repeat=1, provider="fake", store=store, pool=pool, say=lambda *a: None, teacher=spy)
    assert spy.versions and set(spy.versions) == {2}
    assert {r["prompt_version"] for r in store.rows(1, "fake", "rules-v1")} == {2}


def test_collect_keeps_only_the_one_teacher(trained, tmp_path, monkeypatch):
    _, pool, _, _ = trained
    monkeypatch.setattr(llm, "PACER", llm.Pacer(tmp_path / "usage.json"))
    store = LabelStore(tmp_path / "labels.sqlite")
    said = []
    C.collect(100, provider="groq", store=store, pool=pool[:3], say=said.append, teacher=OtherModel())
    assert store.rows(0, valid_only=False) == [] and "labels not saved" in said[-1]
