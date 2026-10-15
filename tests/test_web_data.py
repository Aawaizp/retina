"""retina web-data: the website plays recorded store days (NPC routes, choices, receipts) - SIMULATED,
built without the private Instacart per-shopper profiles (like a fresh clone)."""

import json

from retina import web_data as W


def test_web_data_records_store_days_for_the_3d_page(tmp_path, monkeypatch):
    calls = []
    real_run = W.run

    def spy(args, env, say):
        calls.append((env.get("RETINA_NO_PROFILES"), env.get("RETINA_NO_BROWSER"), "--view" in args))
        return real_run(args, env, say)

    monkeypatch.setattr(W, "run", spy)
    (tmp_path / "data" / "old_store").mkdir(parents=True)                 # older dashboard files go
    manifest = W.web_data(days=2, keep=1, out=tmp_path, student=False, say=lambda *a: None)
    assert calls == [("1", "1", True)]                            # no profiles, no browser, walks recorded
    data = tmp_path / "data"
    assert sorted(f.name for f in data.iterdir()) == ["day_001.json", "manifest.json"]
    assert manifest["days"] == ["day_001.json"] and manifest["data_label"] == "SIMULATED"
    day = json.loads((data / "day_001.json").read_text(encoding="utf-8"))
    assert day["layout"]["shelves"] and day["live"] is True
    kinds = {e["event"] for e in day["events"]}
    assert {"SHOPPER_ENTER", "PURCHASE", "CHECKOUT"} <= kinds
    enter = next(e for e in day["events"] if e["event"] == "SHOPPER_ENTER")
    assert enter["path"] and enter["mission"] in ("top_up", "routine", "stock_up")


def test_switch_turns_the_private_profiles_off(monkeypatch):
    import importlib

    from retina.engine import config
    monkeypatch.setenv("RETINA_NO_PROFILES", "1")
    try:
        assert not importlib.reload(config).SHOPPER_PROFILES_FILE.exists()
    finally:
        monkeypatch.delenv("RETINA_NO_PROFILES")
        importlib.reload(config)
