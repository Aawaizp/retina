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
    manifest = W.web_data(days=3, stores=["convenience"], out=tmp_path, student=False, say=lambda *a: None)
    assert calls and set(calls) == {"1"}                          # every simulation ran without the profiles
    data = tmp_path / "data"
    assert manifest["data_label"] == "SIMULATED" and manifest["stores"]["convenience"]["days"] == 3
    assert json.loads((data / "manifest.json").read_text(encoding="utf-8"))["stores"]
    files = sorted((data / "convenience").glob("*.parquet"))
    assert [f.stem for f in files] == ["baskets", "daily", "departments", "hourly", "top_products"]
    for f in files:
        df = pd.read_parquet(f)
        assert (df["data_label"] == "SIMULATED").all()
        assert not {"shopper_id", "trip_id", "customer_id"} & set(df.columns)   # no per-shopper rows
        assert len(df) <= 400
    assert sum(f.stat().st_size for f in files) < 200_000


def test_switch_turns_the_private_profiles_off(monkeypatch):
    import importlib

    from retina.engine import config
    monkeypatch.setenv("RETINA_NO_PROFILES", "1")
    try:
        assert not importlib.reload(config).SHOPPER_PROFILES_FILE.exists()
    finally:
        monkeypatch.delenv("RETINA_NO_PROFILES")
        importlib.reload(config)
