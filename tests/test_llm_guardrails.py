"""LLM guardrails work without any API key: JSON reading, the persona fact check, choice checks."""

from retina.engine import llm
from retina.engine import personas
from retina.engine import focus


def test_json_from_ignores_extra_text():
    assert llm.json_from('Sure! [{"item": 1}] and then [2]') == [{"item": 1}]
    assert llm.json_from('```json\n[{"a": "x]"}]\n```') == [{"a": "x]"}]


FACTS = {"kids": 0, "household_size": 2, "name": "Jordan D.", "days_between_trips": 25.0}


def persona(bio, style="Plans ahead"):
    return {"bio": bio, "shopping_style": style, "cares_about": ["price"], "quote": "Hello"}


def test_fact_check_accepts_a_good_persona():
    assert personas.check(FACTS, persona("Jordan D., 71, shops about once a month for two.")) is None


def test_fact_check_rejects_invented_facts():
    assert personas.check(FACTS, persona("Jordan D. buys snacks for the kids.")) is not None
    assert personas.check(FACTS, persona("Jordan D. lives with two others.")) is not None
    assert personas.check(FACTS, persona("Jordan D. shops weekly.")) is not None
    assert personas.check(FACTS, persona("A shopper who likes deals.")) is not None      # no name
    assert personas.check(FACTS, persona("Jordan D. is careful.", style="Biweekly shopper")) is not None


def test_tidy_fixes_spacing():
    assert personas.tidy("familiar,fits.She") == "familiar, fits. She"


ITEMS = [{"options": [{"letter": "A", "in_stock": True}, {"letter": "B", "in_stock": False}]},
         {"options": [{"letter": "A", "in_stock": True}]}]


def test_focus_choices_only_valid_options():
    text = '[{"item": 1, "choice": "B"}, {"item": 2, "choice": "A", "qty": 9, "reason": "ok"}, {"item": 7, "choice": "A"}]'
    out = focus.parse_choices(text, ITEMS)
    assert 1 not in out                       # B is sold out
    assert out[2]["choice"] == "A" and out[2]["qty"] == 4      # quantity capped
    assert 7 not in out                       # no such item


def test_auto_chain_skips_providers_without_keys(monkeypatch):
    for key in ("GROQ_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    assert not llm.ready("groq") and not llm.ready("gemini")
    assert llm.ready("ollama")                # needs no key
