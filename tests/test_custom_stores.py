"""A supplied store can run with live student decisions and produce its own sales."""

import json
import time
import csv
from collections import Counter
from pathlib import Path

import pytest
from fastapi import HTTPException


def paid_lines(run):
    with open(run / "receipts.csv", encoding="utf-8", newline="") as f:
        receipts = list(csv.DictReader(f))
    with open(run / "checkout_events.csv", encoding="utf-8", newline="") as f:
        checkout = list(csv.DictReader(f))
    key = lambda row: (row["trip_id"], row["product_id"], int(row["quantity"]))
    assert receipts and Counter(map(key, receipts)) == Counter(map(key, checkout))
    assert all(row["data_label"] == "SIMULATED" for row in checkout)
    assert all(row["list_brain"] in {"rule", "student"} and row["shelf_brain"] for row in checkout)


def test_custom_store_can_align_simulated_dates_with_reference_data(tmp_path, monkeypatch):
    from retina.app.custom_stores import StoreSetup, create_store

    monkeypatch.setenv("RETINA_CUSTOM_STORES", str(tmp_path / "stores"))
    fields = {"name": "Date Aligned Market", "open_hour": 8, "close_hour": 20,
              "shoppers_per_week": 70, "checkout_lanes": 2, "start_date": "2026-04-06"}
    result = create_store(StoreSetup(**fields))
    profile = json.loads((tmp_path / "stores" / result["id"] / "profile.json").read_text(encoding="utf-8"))
    assert profile["simulation"]["start_date"] == "2026-04-06"
    assert "USER_PROVIDED" in profile["labels"]["simulation.start_date"]
    with pytest.raises(ValueError, match="real date"):
        create_store(StoreSetup(**{**fields, "start_date": "2026-02-30"}))


def test_builtin_rule_npcs_use_realtime_and_stream_paid_csv(tmp_path, monkeypatch):
    from retina.app import server

    monkeypatch.setenv("RETINA_LIVE_ROOT", str(tmp_path / "live"))
    server.runner = server.Runner()
    assert server.store_profile("convenience")[1].get("engine") != "realtime"
    server.start_run(server.RunRequest(store="convenience", days=1, mind="fast", reset=True, view=False))
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        state = server.current()
        if not state["running"]:
            break
        time.sleep(0.2)
    assert state["exit_code"] == 0, "\n".join(state["output"][-25:])
    assert state["realtime"]
    run = tmp_path / "live" / "convenience"
    paid_lines(run)
    assert not (run / "view" / "stream.ndjson").exists()


def test_custom_store_setup_and_student_run(tmp_path, monkeypatch):
    import onnx
    from onnx import TensorProto, helper
    from retina.app import server
    from retina.products import convert
    from retina.products.check import check_file
    from retina.student.features import FEATURES

    assert check_file(Path(__file__).parent.parent / "web" / "public" / "store_products_template.csv").ok

    # A tiny test-only shelf brain; no owner model or LLM is used.
    student = tmp_path / "student"
    student.mkdir()
    weights = [1.0 if name == "usual" else 0.0 for name in FEATURES]
    model = helper.make_model(helper.make_graph(
        [helper.make_node("MatMul", ["x", "weights"], ["scores"])], "test-student",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, len(FEATURES)])],
        [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [None, 1])],
        [helper.make_tensor("weights", TensorProto.FLOAT, [len(FEATURES), 1], weights)]),
        opset_imports=[helper.make_operatorsetid("", 13)])
    model.ir_version = 9
    onnx.save(model, student / "student.onnx")
    (student / "student.json").write_text(json.dumps({"features": FEATURES, "temperature": 1.0,
                                                      "threshold": 0.0, "pick": "top",
                                                      "model": "test-only", "version": "test"}), encoding="utf-8")

    monkeypatch.setenv("RETINA_CUSTOM_STORES", str(tmp_path / "stores"))
    monkeypatch.setenv("RETINA_LIVE_ROOT", str(tmp_path / "live"))
    monkeypatch.setenv("RETINA_NO_PROFILES", "1")
    monkeypatch.setenv("RETINA_STUDENT_DIR", str(student))
    monkeypatch.setenv("RETINA_LIST_DIR", str(tmp_path / "no-list-model"))
    monkeypatch.setattr(convert, "UPLOADS", tmp_path / "uploads")
    server.runner = server.Runner()
    products = (Path(__file__).parent / "fixtures" / "products_200.csv").read_text(encoding="utf-8")
    setup = {"name": "Test Market", "city": "Austin", "state": "TX", "store_type": "supermarket",
             "open_hour": 8, "close_hour": 20, "shoppers_per_week": 70, "checkout_lanes": 2,
             "products_csv": products, "hourly_counts": {"8": 2, "12": 4, "17": 4}}
    generic = server.add_custom_store(server.StoreSetup(**{**setup, "name": "Estimated Market", "products_csv": ""}))
    generic_profile = json.loads((tmp_path / "stores" / generic["id"] / "profile.json").read_text(encoding="utf-8"))
    assert generic["products"] is None and generic_profile.get("catalog_kind", "builtin") == "builtin"
    assert "ESTIMATED" in generic_profile["labels"]["catalog_file"]
    with pytest.raises(HTTPException) as bad:
        server.add_custom_store(server.StoreSetup(**{**setup, "close_hour": 7}))
    assert bad.value.status_code == 422
    created = server.add_custom_store(server.StoreSetup(**setup))
    store_id = created["id"]
    assert created["products"] == 200
    profile = json.loads((tmp_path / "stores" / store_id / "profile.json").read_text(encoding="utf-8"))
    assert profile["engine"] == "realtime" and profile["catalog_kind"] == "only"
    assert profile["traffic"]["hourly_counts"] == setup["hourly_counts"]
    assert "USER_PROVIDED" in profile["labels"]["catalog_file"]
    assert any(s["id"] == store_id and s["custom"] for s in server.stores())

    started = server.start_run(server.RunRequest(store=store_id, days=1, mind="student", reset=True, view=False))
    assert started["started"]
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        state = server.current()
        if not state["running"]:
            break
        time.sleep(0.25)
    assert state["exit_code"] == 0, "\n".join(state["output"][-25:])
    assert state["realtime"] and state["store_title"] == "Test Market"
    run = tmp_path / "live" / f"{store_id}_student"
    receipts = (run / "receipts.csv").read_text(encoding="utf-8")
    assert ",student," in receipts and ",llm," not in receipts
    paid_lines(run)
    assert (run / "business" / "sales_transactions.csv").exists()
    card = (run / "business" / "DATASET_CARD.md").read_text(encoding="utf-8")
    assert "USER_PROVIDED" in card and "not a record of this store's actual customers" in card
