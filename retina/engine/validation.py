"""Validate a RETINA business export and, optionally, compare it with real store totals.

The optional reference is a directory containing one or both of:

* daily_product_sales.csv: date,product_id,units_sold[,net_sales]
* daily_store_kpis.csv: date,transactions[,net_sales]

Reference rows describe *full-store observed* totals. RETINA's raw receipts are sampled when
sample_weight > 1, so comparisons use estimated_store_* values, never raw sampled counts.
Only exact matching dates and product IDs are compared. No reference means no realism claim.

The dense product-by-day export is read row by row. The reference is sorted with a temporary
on-disk SQLite table, keeping annual catalogs out of process memory.
"""

from __future__ import annotations

import csv
import json
import math
import sqlite3
from collections import defaultdict
from contextlib import closing
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from tempfile import TemporaryDirectory


REFERENCE_COLUMNS = {
    "daily_product_sales": ("date", "product_id", "units_sold"),
    "daily_store_kpis": ("date", "transactions"),
}
CENT = Decimal("0.01")


def _rows(path):
    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise ValueError(f"Cannot read {path.name}: {exc}") from exc
    with handle:
        reader = csv.DictReader(handle)
        try:
            if not reader.fieldnames:
                raise ValueError(f"{path.name} has no CSV header")
            for line, row in enumerate(reader, 2):
                if None in row:
                    raise ValueError(f"{path.name} line {line} has extra cells")
                yield line, row
        except UnicodeError as exc:
            raise ValueError(f"{path.name} must be UTF-8 CSV") from exc


def _header(path):
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return csv.DictReader(handle).fieldnames or []
    except UnicodeError as exc:
        raise ValueError(f"{path.name} must be UTF-8 CSV") from exc
    except OSError as exc:
        raise ValueError(f"Cannot read {path.name}: {exc}") from exc


def _date(value, file, line):
    from datetime import date

    if len(value or "") != 10:
        raise ValueError(f"{file} line {line}: date must be YYYY-MM-DD")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{file} line {line}: invalid date {value!r}") from exc
    if parsed.isoformat() != value:
        raise ValueError(f"{file} line {line}: date must be YYYY-MM-DD")
    return value


def _number(value, file, line, column):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{file} line {line}: {column} must be a number") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{file} line {line}: {column} must be finite and nonnegative")
    return result


def _integer(value, file, line, column):
    number = _number(value, file, line, column)
    if not number.is_integer():
        raise ValueError(f"{file} line {line}: {column} must be a whole number")
    return int(number)


def _cents(value, file, line, column):
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError) as exc:
        raise ValueError(f"{file} line {line}: {column} must be money") from exc
    if not number.is_finite() or number < 0:
        raise ValueError(f"{file} line {line}: {column} must be finite and nonnegative")
    return int((number / CENT).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def _require(path, required):
    missing = set(required) - set(_header(path))
    if missing:
        raise ValueError(f"{path.name} missing columns: {', '.join(sorted(missing))}")


def _reference_into_db(path, kind, db):
    """Validate every reference row before any report file is replaced."""
    if kind not in REFERENCE_COLUMNS:
        raise ValueError(f"Unknown reference table: {kind}")
    _require(path, REFERENCE_COLUMNS[kind])
    columns = set(_header(path))
    table = "reference_sku" if kind == "daily_product_sales" else "reference_kpi"
    if kind == "daily_product_sales":
        db.execute("CREATE TABLE reference_sku (date TEXT, product_id TEXT, units REAL, net REAL, "
                   "PRIMARY KEY (date, product_id)) WITHOUT ROWID")
    else:
        db.execute("CREATE TABLE reference_kpi (date TEXT PRIMARY KEY, transactions REAL, net REAL) WITHOUT ROWID")
    count, first, last = 0, None, None
    for line, row in _rows(path):
        day = _date(row["date"], path.name, line)
        net = _number(row["net_sales"], path.name, line, "net_sales") if "net_sales" in columns else None
        if kind == "daily_product_sales":
            product = row["product_id"]
            if not product:
                raise ValueError(f"{path.name} line {line}: product_id is required")
            units = _integer(row["units_sold"], path.name, line, "units_sold")
            values = (day, product, units, net)
            statement = "INSERT INTO reference_sku VALUES (?,?,?,?)"
        else:
            transactions = _integer(row["transactions"], path.name, line, "transactions")
            values = (day, transactions, net)
            statement = "INSERT INTO reference_kpi VALUES (?,?,?)"
        try:
            db.execute(statement, values)
        except sqlite3.IntegrityError as exc:
            key = f"{day}/{row['product_id']}" if kind == "daily_product_sales" else day
            raise ValueError(f"{path.name} line {line}: duplicate key {key}") from exc
        count += 1
        first = day if first is None or day < first else first
        last = day if last is None or day > last else last
    if count == 0:
        raise ValueError(f"{path.name} has no data rows")
    db.commit()
    return {"file": path.name, "rows": count, "first_date": first, "last_date": last,
            "has_net_sales": "net_sales" in columns}


def validate_reference_file(path, kind):
    """Check an uploaded reference CSV; raises ValueError with a user-readable reason."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Reference file does not exist: {path.name}")
    with TemporaryDirectory(prefix="retina-reference-", dir=path.parent) as temp:
        with closing(sqlite3.connect(str(Path(temp) / "reference.sqlite"))) as db:
            return _reference_into_db(path, kind, db)


def _check(checks, name, failures, detail="", rows=None):
    item = {"name": name, "status": "pass" if failures == 0 else "fail", "failures": int(failures)}
    if rows is not None:
        item["rows_checked"] = int(rows)
    if detail:
        item["detail"] = detail
    checks.append(item)


def _compare_accumulator():
    return {"rows": 0, "absolute_error": 0.0, "signed_error": 0.0, "actual_total": 0.0,
            "estimated_total": 0.0}


def _add_metric(acc, actual, estimated):
    acc["rows"] += 1
    acc["absolute_error"] += abs(estimated - actual)
    acc["signed_error"] += estimated - actual
    acc["actual_total"] += actual
    acc["estimated_total"] += estimated


def _finish_metric(name, acc, reference_rows, estimate_column, unit):
    rows = acc["rows"]
    actual = acc["actual_total"]
    return {"name": name, "unit": unit, "estimate_column": estimate_column,
            "matched_rows": rows, "reference_rows": reference_rows,
            "coverage_pct": round(100 * rows / reference_rows, 2) if reference_rows else 0.0,
            "actual_total_on_matched_rows": round(actual, 2),
            "estimated_total_on_matched_rows": round(acc["estimated_total"], 2),
            "mae": round(acc["absolute_error"] / rows, 4) if rows else None,
            "wape_pct": round(100 * acc["absolute_error"] / actual, 2) if actual else None,
            "bias_pct": round(100 * acc["signed_error"] / actual, 2) if actual else None}


def _format_number(value):
    return "n/a" if value is None else f"{value:,.2f}"


def _store_estimate(row, estimate_column, raw_value, weight, file, line):
    """A capped day has no defensible full-store expansion; exclude it from comparisons."""
    if estimate_column in row:
        value = row[estimate_column]
        if value is None or value == "":
            return None
        try:
            return _number(value, file, line, estimate_column)
        except ValueError:
            return None
    return raw_value if weight == 1 else None


def _estimate_issue(row, column, raw_value, weight, file, line):
    """Check an expanded value without letting a bad synthetic row abort the report."""
    if weight is None:
        return 0
    if column not in row:
        return int(weight != 1)
    value = row[column]
    if row.get("estimate_status") == "ARRIVAL_CAP_REACHED":
        return int(value not in (None, ""))
    if value in (None, ""):
        return 1
    try:
        estimate = _number(value, file, line, column)
    except ValueError:
        return 1
    return int(abs(estimate - round(raw_value * weight, 2)) > 0.011)


def _markdown(report):
    integrity, realism = report["integrity"], report["realism"]
    weight = report["sampling"]["sample_weight"]
    stock_scope = report["sampling"].get("stock_scope") or "unspecified"
    capped_days = report["sampling"].get("arrival_capped_days", 0)
    lines = ["# RETINA validation report", "", "All exported shoppers and sales are simulated.", "",
             f"**Integrity:** {integrity['status'].upper()}  ",
             f"**Real-store comparison:** {realism['status'].upper()}", "",
             "## Sampling", "",
             f"One exported receipt represents one NPC trip. Sample weight: **{_format_number(weight)}**.",
             "Real-store comparisons use `estimated_store_*` totals; raw receipt rows remain a sample.", "",
             f"Inventory counts and values are **{stock_scope}** and are never expanded. "
             f"Arrival-capped days: **{capped_days:,}**.", "",
             "## Integrity checks", "", "| Check | Result | Issues |", "|---|---|---:|"]
    for check in integrity["checks"]:
        lines.append(f"| {check['name']} | {check['status']} | {check['failures']:,} |")
    lines += ["", "## Observed store comparison", "", realism["note"], ""]
    if report["reference"]["files"]:
        lines += ["References are matched by exact date and product ID. Missing matches are excluded "
                  "from error metrics and shown as coverage. Days without a defensible full-store "
                  "estimate (for example an NPC arrival cap) also reduce coverage.", "",
                  "| Measure | Matched / reference rows | Coverage | MAE | WAPE | Bias |", "|---|---:|---:|---:|---:|---:|"]
        for c in realism["comparisons"]:
            lines.append(f"| {c['name']} | {c['matched_rows']:,} / {c['reference_rows']:,} | "
                         f"{_format_number(c['coverage_pct'])}% | {_format_number(c['mae'])} | "
                         f"{_format_number(c['wape_pct'])}% | {_format_number(c['bias_pct'])}% |")
    lines += ["", "A single run does not provide an uncertainty interval. Compare several runs with "
              "different seeds before using these estimates for decisions.", ""]
    return "\n".join(lines)


def generate_report(business_dir, reference_dir=None):
    """Write VALIDATION.json and VALIDATION.md and return their structured contents.

    Reference files are checked first, so a malformed upload cannot replace an existing report.
    A missing reference leaves realism unverified. Structural checks never imply human realism.
    """
    business_dir = Path(business_dir)
    if not business_dir.is_dir():
        raise ValueError(f"Business export directory does not exist: {business_dir}")
    references = {}
    if reference_dir is not None:
        reference_dir = Path(reference_dir)
        if not reference_dir.is_dir():
            raise ValueError(f"Reference directory does not exist: {reference_dir}")
        for kind in REFERENCE_COLUMNS:
            path = reference_dir / f"{kind}.csv"
            if path.exists():
                references[kind] = path
        if not references:
            raise ValueError("Reference directory needs daily_product_sales.csv or daily_store_kpis.csv")

    # All reference input validation happens before output files are touched.
    with TemporaryDirectory(prefix="retina-validation-", dir=business_dir) as temp:
        with closing(sqlite3.connect(str(Path(temp) / "validation.sqlite"))) as db:
            db.execute("PRAGMA journal_mode=OFF")
            db.execute("PRAGMA synchronous=OFF")
            reference_meta = {kind: _reference_into_db(path, kind, db) for kind, path in references.items()}
            report = _analyze(business_dir, db, reference_meta)
    json_path = business_dir / "VALIDATION.json"
    md_path = business_dir / "VALIDATION.md"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    md_path.write_text(_markdown(report), encoding="utf-8")
    return report


def _analyze(folder, db, reference_meta):
    checks, loaded = [], {}
    required = {
        "products": ("product_id",),
        "customers": ("customer_id",),
        "baskets": ("transaction_id", "date", "customer_id", "net_sales", "sample_weight"),
        "sales_transactions": ("transaction_id", "line_number", "date", "customer_id", "product_id", "quantity", "net_sales"),
        "daily_product_sales": ("date", "product_id", "units_sold", "net_sales"),
        "daily_store_kpis": ("date", "transactions", "net_sales"),
        "inventory_daily": ("date", "product_id", "opening_stock", "delivered_units", "wasted_units", "sold_units", "closing_stock"),
    }
    for name, columns in required.items():
        path = folder / f"{name}.csv"
        present = path.is_file()
        missing = set(columns) - set(_header(path)) if present else set(columns)
        _check(checks, f"{name} columns", int(not present or bool(missing)),
               f"Missing: {', '.join(sorted(missing))}" if missing else "")
        loaded[name] = path if present and not missing else None
    if any(path is None for path in loaded.values()):
        return {"version": 1, "sampling": {"sample_weight": None, "basis": "unavailable"},
                "reference": {"files": reference_meta},
                "integrity": {"status": "fail", "checks": checks},
                "realism": {"status": "unverified", "comparisons": [],
                            "note": "Export files are incomplete; real-store metrics were not computed."}}

    products, customers = set(), set()
    duplicate_products = duplicate_customers = 0
    for _, row in _rows(loaded["products"]):
        key = row["product_id"]
        duplicate_products += key in products or not key
        products.add(key)
    for _, row in _rows(loaded["customers"]):
        key = row["customer_id"]
        duplicate_customers += key in customers or not key
        customers.add(key)
    _check(checks, "unique product IDs", duplicate_products, rows=len(products))
    _check(checks, "unique customer IDs", duplicate_customers, rows=len(customers))
    _check(checks, "nonempty product and customer masters", int(not products or not customers))

    db.execute("CREATE TABLE basket (transaction_id TEXT PRIMARY KEY, date TEXT, customer_id TEXT, cents INTEGER)")
    db.execute("CREATE TABLE receipt (transaction_id TEXT PRIMARY KEY, date TEXT, customer_id TEXT, "
               "cents INTEGER, units INTEGER, lines INTEGER, context_errors INTEGER)")
    db.execute("CREATE TABLE line_key (transaction_id TEXT, line_number INTEGER, PRIMARY KEY (transaction_id, line_number)) WITHOUT ROWID")
    db.execute("CREATE TABLE line_sku (date TEXT, product_id TEXT, units INTEGER, "
               "PRIMARY KEY (date, product_id)) WITHOUT ROWID")
    basket_day_cents, basket_day_count = defaultdict(int), defaultdict(int)
    basket_duplicates = basket_customer_missing = basket_invalid = 0
    weights, basket_rows = set(), 0
    for line, row in _rows(loaded["baskets"]):
        basket_rows += 1
        try:
            day = _date(row["date"], "baskets.csv", line)
            cents = _cents(row["net_sales"], "baskets.csv", line, "net_sales")
            weight = _number(row["sample_weight"], "baskets.csv", line, "sample_weight")
            if weight <= 0:
                raise ValueError("sample_weight must be positive")
        except ValueError:
            basket_invalid += 1
            continue
        weights.add(weight)
        basket_customer_missing += row["customer_id"] not in customers
        try:
            db.execute("INSERT INTO basket VALUES (?,?,?,?)", (row["transaction_id"], day, row["customer_id"], cents))
        except sqlite3.IntegrityError:
            basket_duplicates += 1
            continue
        basket_day_cents[day] += cents
        basket_day_count[day] += 1
    _check(checks, "unique, valid baskets", basket_duplicates + basket_invalid, rows=basket_rows)
    _check(checks, "nonempty paid baskets", int(basket_rows == 0))
    _check(checks, "basket customer references", basket_customer_missing, rows=basket_rows)
    _check(checks, "one positive sample weight", int(len(weights) != 1), rows=basket_rows)
    weight = next(iter(weights)) if len(weights) == 1 else None

    line_product_missing = line_customer_missing = line_duplicates = line_invalid = 0
    line_rows = 0
    line_day_cents, line_day_units = defaultdict(int), defaultdict(int)
    for line, row in _rows(loaded["sales_transactions"]):
        line_rows += 1
        try:
            day = _date(row["date"], "sales_transactions.csv", line)
            cents = _cents(row["net_sales"], "sales_transactions.csv", line, "net_sales")
            units = _integer(row["quantity"], "sales_transactions.csv", line, "quantity")
            line_number = _integer(row["line_number"], "sales_transactions.csv", line, "line_number")
            if units == 0 or line_number == 0:
                raise ValueError("quantity and line number must be positive")
        except ValueError:
            line_invalid += 1
            continue
        line_product_missing += row["product_id"] not in products
        line_customer_missing += row["customer_id"] not in customers
        try:
            db.execute("INSERT INTO line_key VALUES (?,?)", (row["transaction_id"], line_number))
        except sqlite3.IntegrityError:
            line_duplicates += 1
        db.execute("INSERT INTO receipt VALUES (?,?,?,?,?,1,0) ON CONFLICT(transaction_id) DO UPDATE SET "
                   "cents=cents+excluded.cents, units=units+excluded.units, lines=lines+1, "
                   "context_errors=context_errors+(date<>excluded.date OR customer_id<>excluded.customer_id)",
                   (row["transaction_id"], day, row["customer_id"], cents, units))
        db.execute("INSERT INTO line_sku VALUES (?,?,?) ON CONFLICT(date,product_id) DO UPDATE SET "
                   "units=units+excluded.units", (day, row["product_id"], units))
        line_day_cents[day] += cents
        line_day_units[day] += units
    db.commit()
    _check(checks, "valid, unique POS lines", line_invalid + line_duplicates, rows=line_rows)
    _check(checks, "nonempty POS lines", int(line_rows == 0))
    _check(checks, "POS product references", line_product_missing, rows=line_rows)
    _check(checks, "POS customer references", line_customer_missing, rows=line_rows)
    orphan_lines = db.execute("SELECT COUNT(*) FROM receipt r LEFT JOIN basket b ON r.transaction_id=b.transaction_id "
                              "WHERE b.transaction_id IS NULL").fetchone()[0]
    basket_without_lines = db.execute("SELECT COUNT(*) FROM basket b LEFT JOIN receipt r ON b.transaction_id=r.transaction_id "
                                      "WHERE r.transaction_id IS NULL").fetchone()[0]
    wrong_basket_total = db.execute("SELECT COUNT(*) FROM receipt r JOIN basket b ON r.transaction_id=b.transaction_id "
                                    "WHERE r.cents<>b.cents OR r.date<>b.date OR r.customer_id<>b.customer_id "
                                    "OR r.context_errors<>0").fetchone()[0]
    _check(checks, "POS lines match paid baskets", orphan_lines + basket_without_lines + wrong_basket_total,
           rows=basket_rows)

    sku_ref = db.execute("SELECT date,product_id,units,net FROM reference_sku ORDER BY date,product_id") if "daily_product_sales" in reference_meta else None
    current_ref = next(sku_ref, None) if sku_ref is not None else None
    sku_acc = _compare_accumulator()
    sku_sales_acc = _compare_accumulator()
    sku_matched = 0
    sku_days, sku_units, sku_cents = defaultdict(int), defaultdict(int), defaultdict(int)
    sku_invalid = sku_foreign = sku_order_errors = sku_expansion_errors = 0
    sku_rows, prior_key = 0, None
    sku_columns = set(_header(loaded["daily_product_sales"]))
    sku_estimate_available = "estimated_store_units_sold" in sku_columns or weight == 1
    sku_net_available = "estimated_store_net_sales" in sku_columns or weight == 1
    for line, row in _rows(loaded["daily_product_sales"]):
        sku_rows += 1
        try:
            day = _date(row["date"], "daily_product_sales.csv", line)
            units = _integer(row["units_sold"], "daily_product_sales.csv", line, "units_sold")
            cents = _cents(row["net_sales"], "daily_product_sales.csv", line, "net_sales")
        except ValueError:
            sku_invalid += 1
            continue
        key = (day, row["product_id"])
        sku_order_errors += prior_key is not None and key <= prior_key
        prior_key = key
        sku_foreign += row["product_id"] not in products
        sku_days[day] += 1
        sku_units[day] += units
        sku_cents[day] += cents
        sku_expansion_errors += _estimate_issue(row, "estimated_store_units_sold", units, weight,
                                                 "daily_product_sales.csv", line)
        sku_expansion_errors += _estimate_issue(row, "estimated_store_net_sales", cents / 100, weight,
                                                 "daily_product_sales.csv", line)
        if sku_ref is not None:
            while current_ref is not None and current_ref[:2] < key:
                current_ref = next(sku_ref, None)
            if current_ref is not None and current_ref[:2] == key:
                sku_matched += 1
                if sku_estimate_available:
                    estimate = _store_estimate(row, "estimated_store_units_sold", units, weight,
                                               "daily_product_sales.csv", line)
                    if estimate is not None:
                        _add_metric(sku_acc, current_ref[2], estimate)
                if current_ref[3] is not None and sku_net_available:
                    estimate = _store_estimate(row, "estimated_store_net_sales", cents / 100, weight,
                                               "daily_product_sales.csv", line)
                    if estimate is not None:
                        _add_metric(sku_sales_acc, current_ref[3], estimate)
                current_ref = next(sku_ref, None)
    _check(checks, "valid, sorted, unique SKU-day rows", sku_invalid + sku_order_errors, rows=sku_rows)
    _check(checks, "SKU-day product references", sku_foreign, rows=sku_rows)
    _check(checks, "SKU-day estimates match sample weight", sku_expansion_errors, rows=sku_rows)

    kpi_columns = set(_header(loaded["daily_store_kpis"]))
    kpi_day_count, kpi_day_cents, kpi_estimates = {}, {}, {}
    kpi_invalid = kpi_duplicates = kpi_total_errors = kpi_expansion_errors = 0
    stock_scopes, capped_days = set(), 0
    kpi_rows = 0
    for line, row in _rows(loaded["daily_store_kpis"]):
        kpi_rows += 1
        try:
            day = _date(row["date"], "daily_store_kpis.csv", line)
            transactions = _integer(row["transactions"], "daily_store_kpis.csv", line, "transactions")
            cents = _cents(row["net_sales"], "daily_store_kpis.csv", line, "net_sales")
        except ValueError:
            kpi_invalid += 1
            continue
        if day in kpi_day_count:
            kpi_duplicates += 1
            continue
        kpi_day_count[day], kpi_day_cents[day] = transactions, cents
        kpi_total_errors += transactions != basket_day_count[day] or cents != line_day_cents[day]
        kpi_expansion_errors += _estimate_issue(row, "estimated_store_transactions", transactions, weight,
                                                 "daily_store_kpis.csv", line)
        kpi_expansion_errors += _estimate_issue(row, "estimated_store_net_sales", cents / 100, weight,
                                                 "daily_store_kpis.csv", line)
        kpi_estimates[day] = row
        if row.get("stock_scope"):
            stock_scopes.add(row["stock_scope"])
        capped_days += row.get("estimate_status") == "ARRIVAL_CAP_REACHED"
    _check(checks, "unique, valid daily KPIs", kpi_invalid + kpi_duplicates, rows=kpi_rows)
    _check(checks, "nonempty daily KPIs", int(kpi_rows == 0))
    _check(checks, "daily KPI totals equal paid receipts", kpi_total_errors + len(set(line_day_cents) - set(kpi_day_count)),
           rows=kpi_rows)
    _check(checks, "daily estimates match sample weight", kpi_expansion_errors, rows=kpi_rows)
    sku_date_errors = len(set(sku_days) - set(kpi_day_count))
    expected_sku_rows = len(products) * len(kpi_day_count)
    sku_panel_errors = abs(sku_rows - expected_sku_rows) + sku_date_errors
    _check(checks, "complete product-by-day panel", sku_panel_errors,
           f"Expected {expected_sku_rows:,} rows: {len(products):,} products x {len(kpi_day_count):,} dates.", sku_rows)
    sku_total_errors = sum(sku_units[d] != line_day_units[d] or sku_cents[d] != line_day_cents[d]
                           for d in set(sku_days) | set(line_day_cents))
    _check(checks, "SKU-day totals equal POS lines", sku_total_errors, rows=len(sku_days))

    db.execute("CREATE TABLE inventory_sold (date TEXT, product_id TEXT, units INTEGER, "
               "PRIMARY KEY (date, product_id)) WITHOUT ROWID")
    inventory_rows = balance_errors = inventory_invalid = inventory_foreign = inventory_duplicates = 0
    for line, row in _rows(loaded["inventory_daily"]):
        inventory_rows += 1
        try:
            day = _date(row["date"], "inventory_daily.csv", line)
            values = {c: _integer(row[c], "inventory_daily.csv", line, c) for c in
                      ("opening_stock", "delivered_units", "wasted_units", "sold_units", "closing_stock")}
        except ValueError:
            inventory_invalid += 1
            continue
        inventory_foreign += row["product_id"] not in products
        balance_errors += (values["opening_stock"] + values["delivered_units"]
                           - values["wasted_units"] - values["sold_units"] != values["closing_stock"])
        try:
            db.execute("INSERT INTO inventory_sold VALUES (?,?,?)",
                       (day, row["product_id"], values["sold_units"]))
        except sqlite3.IntegrityError:
            inventory_duplicates += 1
    _check(checks, "nonnegative, balanced inventory movements", balance_errors + inventory_invalid + inventory_duplicates,
           rows=inventory_rows)
    _check(checks, "inventory product references", inventory_foreign, rows=inventory_rows)
    stock_sales_errors = db.execute(
        "SELECT COUNT(*) FROM ("
        " SELECT l.date,l.product_id FROM line_sku l LEFT JOIN inventory_sold i USING(date,product_id)"
        " WHERE i.units IS NULL OR i.units<>l.units"
        " UNION ALL"
        " SELECT i.date,i.product_id FROM inventory_sold i LEFT JOIN line_sku l USING(date,product_id)"
        " WHERE l.units IS NULL AND i.units<>0)"
    ).fetchone()[0]
    _check(checks, "inventory sold units equal POS lines", stock_sales_errors, rows=inventory_rows)

    comparisons = []
    if "daily_product_sales" in reference_meta:
        n = reference_meta["daily_product_sales"]["rows"]
        if sku_estimate_available:
            comparisons.append(_finish_metric("SKU-day units sold", sku_acc, n,
                                              "estimated_store_units_sold" if "estimated_store_units_sold" in sku_columns else "units_sold", "units"))
        if reference_meta["daily_product_sales"]["has_net_sales"] and sku_net_available:
            comparisons.append(_finish_metric("SKU-day net sales", sku_sales_acc, n,
                                              "estimated_store_net_sales" if "estimated_store_net_sales" in sku_columns else "net_sales", "USD"))
    if "daily_store_kpis" in reference_meta:
        ref_rows = db.execute("SELECT date,transactions,net FROM reference_kpi").fetchall()
        kpi_tx_acc, kpi_net_acc = _compare_accumulator(), _compare_accumulator()
        for day, actual_tx, actual_net in ref_rows:
            if day not in kpi_estimates:
                continue
            row = kpi_estimates[day]
            if "estimated_store_transactions" in kpi_columns or weight == 1:
                estimate = _store_estimate(row, "estimated_store_transactions", float(row["transactions"]), weight,
                                           "daily_store_kpis.csv", 0)
                if estimate is not None:
                    _add_metric(kpi_tx_acc, actual_tx, estimate)
            if actual_net is not None and ("estimated_store_net_sales" in kpi_columns or weight == 1):
                estimate = _store_estimate(row, "estimated_store_net_sales", float(row["net_sales"]), weight,
                                           "daily_store_kpis.csv", 0)
                if estimate is not None:
                    _add_metric(kpi_net_acc, actual_net, estimate)
        n = reference_meta["daily_store_kpis"]["rows"]
        if "estimated_store_transactions" in kpi_columns or weight == 1:
            comparisons.append(_finish_metric("Daily transactions", kpi_tx_acc, n,
                                              "estimated_store_transactions" if "estimated_store_transactions" in kpi_columns else "transactions", "receipts"))
        if reference_meta["daily_store_kpis"]["has_net_sales"] and ("estimated_store_net_sales" in kpi_columns or weight == 1):
            comparisons.append(_finish_metric("Daily net sales", kpi_net_acc, n,
                                              "estimated_store_net_sales" if "estimated_store_net_sales" in kpi_columns else "net_sales", "USD"))

    if not reference_meta:
        realism_status = "unverified"
        note = "No real store reference was supplied. Accounting checks cannot establish shopper realism."
    elif not comparisons:
        realism_status = "unverified"
        note = "No comparable full-store estimate is available for the supplied reference."
    elif all(c["matched_rows"] == 0 for c in comparisons):
        realism_status = "no_overlap"
        note = "No exact matching dates and product IDs were found. No realism error was calculated."
    else:
        realism_status = "evaluated"
        note = "Errors describe only matched reference rows. Coverage must be checked before interpreting them."
    integrity_status = "pass" if all(c["status"] == "pass" for c in checks) else "fail"
    return {"version": 1,
            "sampling": {"sample_weight": weight,
                         "basis": "unknown" if weight is None else
                         ("weighted NPC sample" if weight > 1 else "one NPC per simulated receipt"),
                         "stock_scope": next(iter(stock_scopes)) if len(stock_scopes) == 1 else None,
                         "arrival_capped_days": capped_days},
            "reference": {"files": reference_meta},
            "integrity": {"status": integrity_status, "checks": checks},
            "realism": {"status": realism_status, "comparisons": comparisons, "note": note}}
