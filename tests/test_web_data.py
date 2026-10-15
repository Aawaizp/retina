"""retina web-data: the website's files are small, aggregated and SIMULATED - never a row per shopper,
and built without the private Instacart per-shopper profiles (like a fresh clone)."""

import json

import pandas as pd

from retina import web_data as W


def test_web_data_is_aggregated_and_simulated(tmp_path, monkeypatch):
    calls = []
    real_run = W.run

    def spy(args, env, say):
        calls.append(env.get("RETINA_NO_PROFILES"))
        return real_run(args, env, say)

    monkeypatch.setattr(W, "run", spy)
    manifest = W.web_data(days=3, out=tmp_path, student=False, say=lambda *a: None)
    assert calls and set(calls) == {"1"}                          # every simulation ran without the profiles
    data = tmp_path / "data"
    assert manifest["data_label"] == "SIMULATED" and manifest["stores"][W.STORE]["days"] == 3
    assert json.loads((data / "manifest.json").read_text(encoding="utf-8"))["stores"]
    assert json.loads((data / "whatif.json").read_text(encoding="utf-8"))["shelves"]
    files = sorted((data / W.STORE).glob("*.parquet"))
    assert [f.stem for f in files] == ["baskets", "daily", "departments", "hourly", "top_products"]
    for f in files:
        df = pd.read_parquet(f)
        assert (df["data_label"] == "SIMULATED").all()
        assert not {"shopper_id", "trip_id", "customer_id", "transaction_id"} & set(df.columns)   # no per-shopper rows
        assert len(df) <= 400
    assert sum(f.stat().st_size for f in files) < 200_000


def test_scorecard_puts_the_run_next_to_real_numbers(tmp_path):
    import os
    env = dict(os.environ, RETINA_LIVE_ROOT=str(tmp_path), RETINA_NO_PROFILES="1", PYTHONIOENCODING="utf-8")
    W.run(["--store", W.STORE, "--days", "4", "--reset"], env, print)
    card = W.scorecard(tmp_path / W.STORE, 4)
    rows = {r["measure"]: r for r in card["rows"]}
    assert card["data_label"] == "SIMULATED" and card["baskets"] > 0
    assert 1 < rows["Products per basket"]["ours"] < 40 and 0 <= rows["Re-buy share (product bought before)"]["ours"] <= 1
    assert all(r["label"] in ("PUBLIC_STATISTIC", "REAL_OBSERVED") and r["source"] for r in card["rows"])
    json.dumps(card)                                               # plain JSON for the website


def test_switch_turns_the_private_profiles_off(monkeypatch):
    import importlib

    from retina.engine import config
    monkeypatch.setenv("RETINA_NO_PROFILES", "1")
    try:
        assert not importlib.reload(config).SHOPPER_PROFILES_FILE.exists()
    finally:
        monkeypatch.delenv("RETINA_NO_PROFILES")
        importlib.reload(config)
