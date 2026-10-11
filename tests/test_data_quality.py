"""The data itself: receipts add up, prices make sense, tables join."""

import numpy as np

from export import run_checks


def test_export_consistency_checks_pass(pipeline, tables):
    t, truth = tables
    checks = run_checks(t, truth, pipeline)
    failed = [k for k, ok in checks.items() if not ok]
    assert not failed, failed


def test_receipt_lines(tables):
    tx = tables[0]["transactions"]
    assert (tx["quantity"] >= 1).all()
    assert (tx["unit_price"] > 0).all()
    assert np.allclose(tx["line_total"], (tx["quantity"] * tx["unit_price"]).round(2), atol=0.011)
    promo = tx[tx["on_promo"]]
    assert (promo["unit_price"] <= promo["regular_price"] + 0.01).all()
    assert set(tx["data_label"]) == {"SIMULATED"}


def test_every_line_has_a_reason_and_trip_type(tables):
    tx = tables[0]["transactions"]
    assert set(tx["reason"]) <= {"need", "repeat", "new", "partner", "impulse"}
    assert set(tx["mission"]) <= {"top_up", "routine", "stock_up"}


def test_trip_times_are_in_order(tables):
    trips = tables[0]["trips"]
    assert (trips["minutes_in_store"] > 0).all()
    assert (trips["arrive_time"] < trips["leave_time"]).mean() > 0.99   # (a trip after 23:00 can wrap)


def test_scorecard_runs(pipeline):
    rows = pipeline["scorecard"]
    assert len(rows) >= 16
    assert all(r[4] in ("PASS", "FAIL") for r in rows)
