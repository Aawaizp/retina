"""The validation report keeps accounting checks separate from observed-store evidence."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from retina.engine.validation import generate_report, validate_reference_file


ROOT = Path(__file__).resolve().parents[1]


def _write(folder, name, columns, rows):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def business(tmp_path):
    folder = tmp_path / "business"
    _write(folder, "products", ["product_id"], [{"product_id": "A"}, {"product_id": "B"}])
    _write(folder, "customers", ["customer_id"], [{"customer_id": "C1"}])
    _write(folder, "baskets", ["transaction_id", "date", "customer_id", "net_sales", "sample_weight"], [
        {"transaction_id": "T1", "date": "2025-09-01", "customer_id": "C1", "net_sales": "2.50", "sample_weight": 10},
        {"transaction_id": "T2", "date": "2025-09-02", "customer_id": "C1", "net_sales": "3.00", "sample_weight": 10},
    ])
    _write(folder, "sales_transactions", ["transaction_id", "line_number", "date", "customer_id", "product_id",
                                        "quantity", "net_sales"], [
        {"transaction_id": "T1", "line_number": 1, "date": "2025-09-01", "customer_id": "C1", "product_id": "A",
         "quantity": 1, "net_sales": "2.50"},
        {"transaction_id": "T2", "line_number": 1, "date": "2025-09-02", "customer_id": "C1", "product_id": "B",
         "quantity": 1, "net_sales": "3.00"},
    ])
    _write(folder, "daily_product_sales", ["date", "product_id", "units_sold", "net_sales",
                                           "estimated_store_units_sold", "estimated_store_net_sales"], [
        {"date": "2025-09-01", "product_id": "A", "units_sold": 1, "net_sales": "2.50",
         "estimated_store_units_sold": 10, "estimated_store_net_sales": "25.00"},
        {"date": "2025-09-01", "product_id": "B", "units_sold": 0, "net_sales": "0.00",
         "estimated_store_units_sold": 0, "estimated_store_net_sales": "0.00"},
        {"date": "2025-09-02", "product_id": "A", "units_sold": 0, "net_sales": "0.00",
         "estimated_store_units_sold": 0, "estimated_store_net_sales": "0.00"},
        {"date": "2025-09-02", "product_id": "B", "units_sold": 1, "net_sales": "3.00",
         "estimated_store_units_sold": 10, "estimated_store_net_sales": "30.00"},
    ])
    _write(folder, "daily_store_kpis", ["date", "transactions", "net_sales",
                                         "estimated_store_transactions", "estimated_store_net_sales"], [
        {"date": "2025-09-01", "transactions": 1, "net_sales": "2.50",
         "estimated_store_transactions": 10, "estimated_store_net_sales": "25.00"},
        {"date": "2025-09-02", "transactions": 1, "net_sales": "3.00",
         "estimated_store_transactions": 10, "estimated_store_net_sales": "30.00"},
    ])
    _write(folder, "inventory_daily", ["date", "product_id", "opening_stock", "delivered_units",
                                        "wasted_units", "sold_units", "closing_stock"], [
        {"date": "2025-09-01", "product_id": "A", "opening_stock": 4, "delivered_units": 0,
         "wasted_units": 0, "sold_units": 1, "closing_stock": 3},
        {"date": "2025-09-02", "product_id": "B", "opening_stock": 6, "delivered_units": 0,
         "wasted_units": 0, "sold_units": 1, "closing_stock": 5},
    ])
    return folder


def test_integrity_does_not_pretend_to_measure_realism(business):
    report = generate_report(business)
    assert report["integrity"]["status"] == "pass"
    assert report["realism"]["status"] == "unverified"
    assert report["sampling"]["sample_weight"] == 10
    assert "No real store reference" in (business / "VALIDATION.md").read_text(encoding="utf-8")
    assert json.loads((business / "VALIDATION.json").read_text(encoding="utf-8")) == report


def test_reference_compares_full_store_estimates_with_exact_keys(business, tmp_path):
    reference = tmp_path / "reference"
    _write(reference, "daily_product_sales", ["date", "product_id", "units_sold", "net_sales"], [
        {"date": "2025-09-01", "product_id": "A", "units_sold": 10, "net_sales": "25.00"},
        {"date": "2025-09-01", "product_id": "B", "units_sold": 0, "net_sales": "0.00"},
        {"date": "2025-09-02", "product_id": "A", "units_sold": 0, "net_sales": "0.00"},
        {"date": "2025-09-02", "product_id": "B", "units_sold": 10, "net_sales": "30.00"},
        {"date": "2025-09-03", "product_id": "A", "units_sold": 2, "net_sales": "5.00"},
    ])
    _write(reference, "daily_store_kpis", ["date", "transactions", "net_sales"], [
        {"date": "2025-09-01", "transactions": 10, "net_sales": "25.00"},
        {"date": "2025-09-02", "transactions": 10, "net_sales": "30.00"},
    ])
    report = generate_report(business, reference)
    assert report["realism"]["status"] == "evaluated"
    compare = {c["name"]: c for c in report["realism"]["comparisons"]}
    assert compare["SKU-day units sold"]["matched_rows"] == 4
    assert compare["SKU-day units sold"]["coverage_pct"] == 80
    assert compare["SKU-day units sold"]["wape_pct"] == 0
    assert compare["SKU-day units sold"]["estimate_column"] == "estimated_store_units_sold"
    assert compare["Daily transactions"]["matched_rows"] == 2
    assert compare["Daily transactions"]["wape_pct"] == 0


def test_no_date_overlap_is_reported_not_shifted(business, tmp_path):
    reference = tmp_path / "reference"
    _write(reference, "daily_store_kpis", ["date", "transactions"], [
        {"date": "2026-09-01", "transactions": 10},
    ])
    report = generate_report(business, reference)
    assert report["realism"]["status"] == "no_overlap"
    assert report["realism"]["comparisons"][0]["mae"] is None
    assert report["realism"]["comparisons"][0]["coverage_pct"] == 0


def test_bad_reference_is_rejected_before_overwriting_report(business, tmp_path):
    original = generate_report(business)
    before = (business / "VALIDATION.json").read_text(encoding="utf-8")
    reference = tmp_path / "reference"
    _write(reference, "daily_product_sales", ["date", "product_id", "units_sold"], [
        {"date": "2025-09-01", "product_id": "A", "units_sold": 2},
        {"date": "2025-09-01", "product_id": "A", "units_sold": 2},
    ])
    with pytest.raises(ValueError, match="duplicate key"):
        validate_reference_file(reference / "daily_product_sales.csv", "daily_product_sales")
    with pytest.raises(ValueError, match="duplicate key"):
        generate_report(business, reference)
    assert (business / "VALIDATION.json").read_text(encoding="utf-8") == before
    assert original["realism"]["status"] == "unverified"


def test_missing_zero_sale_sku_day_fails_integrity(business):
    path = business / "daily_product_sales.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    _write(business, "daily_product_sales", list(rows[0]), rows[:-1])
    report = generate_report(business)
    assert report["integrity"]["status"] == "fail"
    check = next(c for c in report["integrity"]["checks"] if c["name"] == "complete product-by-day panel")
    assert check["failures"] > 0


def test_capped_day_has_no_full_store_comparison(business, tmp_path):
    reference = tmp_path / "reference"
    _write(reference, "daily_store_kpis", ["date", "transactions"], [
        {"date": "2025-09-01", "transactions": 10},
        {"date": "2025-09-02", "transactions": 10},
    ])
    path = business / "daily_store_kpis.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[1]["estimated_store_transactions"] = ""
    _write(business, "daily_store_kpis", list(rows[0]), rows)
    report = generate_report(business, reference)
    measure = next(c for c in report["realism"]["comparisons"] if c["name"] == "Daily transactions")
    assert measure["matched_rows"] == 1
    assert measure["coverage_pct"] == 50


def test_uploaded_reference_must_be_utf8(business, tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "daily_store_kpis.csv").write_bytes(b"date,transactions\n2025-09-01,10\n\xff")
    with pytest.raises(ValueError, match="UTF-8"):
        validate_reference_file(reference / "daily_store_kpis.csv", "daily_store_kpis")


def test_weighted_estimate_and_inventory_drift_fail_integrity(business):
    daily = business / "daily_product_sales.csv"
    with daily.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["estimated_store_units_sold"] = "999"
    _write(business, "daily_product_sales", list(rows[0]), rows)
    inventory = business / "inventory_daily.csv"
    with inventory.open(encoding="utf-8", newline="") as handle:
        movements = list(csv.DictReader(handle))
    movements[0]["sold_units"] = "0"
    movements[0]["closing_stock"] = "4"  # still balances, but no longer agrees with POS
    _write(business, "inventory_daily", list(movements[0]), movements)

    report = generate_report(business)
    failed = {check["name"] for check in report["integrity"]["checks"] if check["status"] == "fail"}
    assert "SKU-day estimates match sample weight" in failed
    assert "inventory sold units equal POS lines" in failed


def test_local_reference_upload_updates_report_and_rolls_back_bad_input(business, monkeypatch):
    from fastapi import HTTPException
    from retina.app import server

    monkeypatch.setenv("RETINA_LIVE_ROOT", str(business.parent.parent))
    run = business.parent.name
    upload = server.ValidationUpload(kind="daily_store_kpis",
                                     csv_text="date,transactions,net_sales\n2025-09-01,10,25.00\n2025-09-02,10,30.00\n")
    report = server.compare_with_store_data(run, upload)
    assert report["realism"]["status"] == "evaluated"
    reference = business.parent / "reference" / "daily_store_kpis.csv"
    assert reference.is_file() and "VALIDATION.json" in server.BUSINESS_FILES
    before_reference = reference.read_bytes()
    before_report = (business / "VALIDATION.json").read_bytes()

    bad = server.ValidationUpload(kind="daily_store_kpis",
                                  csv_text="date,transactions\n2025-09-01,10\n2025-09-01,11\n")
    with pytest.raises(HTTPException) as error:
        server.compare_with_store_data(run, bad)
    assert error.value.status_code == 422
    assert reference.read_bytes() == before_reference
    assert (business / "VALIDATION.json").read_bytes() == before_report


def test_validate_cli_generates_report_for_explicit_run_folder(business):
    result = subprocess.run([sys.executable, "-m", "retina", "validate", "--run", str(business.parent)],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "integrity pass" in result.stdout
    assert "real-store comparison unverified" in result.stdout
    assert (business / "VALIDATION.md").is_file()
