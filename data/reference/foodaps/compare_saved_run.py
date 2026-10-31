r"""Compare one synthetic run with public USDA FoodAPS grocery purchases.

Usage from the repository root:
    .venv\Scripts\python.exe data\reference\foodaps\compare_saved_run.py

The ZIPs are downloaded from the USDA URL in README.md and stay local. This
script writes aggregate diagnostics only; it never changes the saved run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FOODAPS_ZIP = HERE / "csv-data-files.zip"
CPI_FILE = ROOT / "data" / "raw" / "cu.data.0.Current"
DEFAULT_RUN = ROOT / "data" / "live" / "retina_supercenter_student"
FOOD_CATEGORIES = {
    "produce", "dairy eggs", "snacks", "beverages", "frozen", "bakery",
    "pantry", "dry goods pasta", "deli", "breakfast", "canned goods",
    "meat seafood", "international",
}
PLACE_TYPES = {"121": "super store", "122": "supermarket"}
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


def _csv_in_zip(archive: ZipFile, name: str):
    # USDA's CSV release contains Windows-1252 descriptions.
    with archive.open(name) as source:
        yield from csv.DictReader(io.TextIOWrapper(source, encoding="cp1252", newline=""))


def _rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        yield from csv.DictReader(source)


def _number(value: str) -> float | None:
    if value in ("", ".", ".V", "-996", None):
        return None
    try:
        n = float(value)
    except (TypeError, ValueError):
        return None
    return n if math.isfinite(n) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpi(path: Path):
    values = {}
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        for raw in csv.DictReader(source, delimiter="\t"):
            row = {k.strip(): (v or "").strip() for k, v in raw.items()}
            if (row["series_id"] == "CUUR0000SAF11"
                    and re.fullmatch(r"M\d\d", row["period"])
                    and _number(row["value"]) is not None):
                values[f"{row['year']}-{row['period'][1:]}"] = float(row["value"])
    return values


def _weight_stats(samples: list[tuple[float, float]]):
    """Event-weighted mean and nearest-rank quantiles of a positive-weight sample."""
    pairs = sorted((v, w) for v, w in samples if math.isfinite(v) and math.isfinite(w) and w > 0)
    if not pairs:
        return None
    total_weight = sum(w for _, w in pairs)
    mean = sum(v * w for v, w in pairs) / total_weight
    quantiles = {}
    for q in QUANTILES:
        threshold = total_weight * q
        cumulative = 0.0
        for value, weight in pairs:
            cumulative += weight
            if cumulative >= threshold:
                quantiles[f"p{round(q * 100)}"] = round(value, 2)
                break
    return {"n": len(pairs), "mean": round(mean, 2), **quantiles}


def _weighted_share(samples: list[tuple[float, float]], predicate):
    total = sum(weight for _, weight in samples)
    return round(100 * sum(weight for value, weight in samples if predicate(value)) / total, 2)


def _line_key(item: dict):
    barcode = item["barcode"].strip()
    if _usable_barcode(barcode):
        return "barcode:" + barcode
    # This is an approximation: two products can share a generic receipt name.
    desc = " ".join(item["itemdesc"].upper().split())
    return "description:" + desc if desc else "item:" + item["itemnum"]


def _usable_barcode(barcode: str):
    return bool(barcode and barcode not in (".V", ".", "-996"))


def _observed(archive_path: Path, cpi: dict, target_month: str):
    target_index = cpi[target_month]
    with ZipFile(archive_path) as archive:
        household_weight = {
            row["hhnum"]: value
            for row in _csv_in_zip(archive, "faps_household_puf.csv")
            if (value := _number(row["hhwgt"])) is not None and value > 0
        }
        all_events = list(_csv_in_zip(archive, "faps_fahevent_puf.csv"))
        selected = defaultdict(dict)
        for row in all_events:
            place = row["placetype"]
            paid = _number(row["totalpaid"])
            if (
                place in PLACE_TYPES and row["free"] == "0"
                and row["receiptobserved"] == "1" and paid is not None and paid > 0
                and row["hhnum"] in household_weight
            ):
                month = datetime.strptime(row["date"], "%m/%d/%Y").strftime("%Y-%m")
                if month not in cpi:
                    raise ValueError(f"Missing BLS CPI for {month}")
                selected[place][row["eventid"]] = {
                    "weight": household_weight[row["hhnum"]], "month": month,
                    "items": [], "hhnum": row["hhnum"],
                }
        event_to_place = {
            event_id: place for place, events in selected.items() for event_id in events
        }
        for item in _csv_in_zip(archive, "faps_fahitem_puf.csv"):
            place = event_to_place.get(item["eventid"])
            if place is not None:
                selected[place][item["eventid"]]["items"].append(item)

    result = {}
    for place, events in selected.items():
        baskets = []
        line_quantities = []
        missing_expenses = 0
        negative_expenses = 0
        missing_barcodes = 0
        missing_items = 0
        for event in events.values():
            if not event["items"]:
                missing_items += 1
                continue
            lines = defaultdict(float)
            food_expense = 0.0
            complete_spend = True
            for item in event["items"]:
                if not _usable_barcode(item["barcode"].strip()):
                    missing_barcodes += 1
                qty = _number(item["quantity"])
                if qty is None or qty <= 0:
                    raise ValueError("USDA item has missing or invalid quantity")
                lines[_line_key(item)] += qty
                expense = _number(item["totitemexp"])
                if expense is None:
                    missing_expenses += 1
                    complete_spend = False
                else:
                    food_expense += expense
                    if expense < 0:
                        negative_expenses += 1
            weight = event["weight"]
            line_quantities.extend((qty, weight) for qty in lines.values())
            baskets.append({
                "lines": len(lines), "raw_item_rows": len(event["items"]),
                "units": sum(lines.values()), "weight": weight,
                "food_expense_2025": (
                    food_expense * target_index / cpi[event["month"]] if complete_spend else None
                ),
            })
        if not baskets:
            raise ValueError(f"No usable USDA baskets for {place}")
        result[place] = {
            "place_type": PLACE_TYPES[place],
            "all_place_events": sum(e["placetype"] == place for e in all_events),
            "selected_events": len(events),
            "events_with_item_rows": len(baskets),
            "unique_households": len({e["hhnum"] for e in events.values()}),
            "item_rows": sum(len(e["items"]) for e in events.values()),
            "item_rows_without_barcode": missing_barcodes,
            "events_missing_item_rows": missing_items,
            "item_rows_missing_expense": missing_expenses,
            "item_rows_negative_expense": negative_expenses,
            "complete_expense_events": sum(b["food_expense_2025"] is not None for b in baskets),
            "lines_per_basket": _weight_stats([(b["lines"], b["weight"]) for b in baskets]),
            "raw_item_rows_per_basket": _weight_stats([
                (b["raw_item_rows"], b["weight"]) for b in baskets
            ]),
            "units_per_basket": _weight_stats([(b["units"], b["weight"]) for b in baskets]),
            "one_to_three_line_basket_share_pct": _weighted_share(
                [(b["lines"], b["weight"]) for b in baskets], lambda n: 1 <= n <= 3
            ),
            "one_to_three_raw_item_row_basket_share_pct": _weighted_share(
                [(b["raw_item_rows"], b["weight"]) for b in baskets], lambda n: 1 <= n <= 3
            ),
            "one_to_three_unit_basket_share_pct": _weighted_share(
                [(b["units"], b["weight"]) for b in baskets], lambda n: 1 <= n <= 3
            ),
            "twenty_five_plus_unit_basket_share_pct": _weighted_share(
                [(b["units"], b["weight"]) for b in baskets], lambda n: n >= 25
            ),
            "food_spend_target_month_usd": _weight_stats([
                (b["food_expense_2025"], b["weight"]) for b in baskets
                if b["food_expense_2025"] is not None
            ]),
            "single_unit_line_share_pct": round(100 * sum(w for q, w in line_quantities if q == 1) /
                                                 sum(w for _, w in line_quantities), 2),
        }
    return {"all_events": len(all_events), "by_place": result}


def _run_calendar(run: Path):
    dates = {datetime.fromisoformat(row["date"]).date().isoformat()
             for row in _rows(run / "business" / "baskets.csv")}
    return sorted(dates), sorted({day[:7] for day in dates})


def _synthetic(run: Path, cpi: dict, target_month: str):
    baskets = {}
    months = set()
    days = set()
    for row in _rows(run / "business" / "baskets.csv"):
        month = row["date"][:7]
        if month not in cpi:
            raise ValueError(f"Missing BLS CPI for synthetic basket month {month}")
        baskets[row["transaction_id"]] = {"lines": defaultdict(float), "spend": 0.0, "month": month}
        months.add(month)
        days.add(row["date"])
    if not baskets:
        raise ValueError("Synthetic run has no paid baskets")
    all_lines = 0
    food_lines = 0
    excluded = defaultdict(int)
    for row in _rows(run / "business" / "sales_transactions.csv"):
        all_lines += 1
        category = row["category"]
        if category not in FOOD_CATEGORIES:
            excluded[category] += 1
            continue
        basket = baskets[row["transaction_id"]]
        basket["lines"][row["product_id"]] += float(row["quantity"])
        basket["spend"] += float(row["net_sales"]) * cpi[target_month] / cpi[basket["month"]]
        food_lines += 1
    line_quantities = [q for b in baskets.values() for q in b["lines"].values()]
    values = list(baskets.values())
    return {
        "paid_npc_baskets": len(baskets),
        "basket_months": sorted(months),
        "basket_days": len(days),
        "all_pos_lines": all_lines,
        "food_pos_lines": food_lines,
        "excluded_pos_lines_by_category": dict(sorted(excluded.items())),
        "baskets_without_food": sum(not b["lines"] for b in values),
        "lines_per_basket": _weight_stats([(len(b["lines"]), 1) for b in values]),
        "units_per_basket": _weight_stats([(sum(b["lines"].values()), 1) for b in values]),
        "one_to_three_line_basket_share_pct": _weighted_share(
            [(len(b["lines"]), 1) for b in values], lambda n: 1 <= n <= 3
        ),
        "one_to_three_unit_basket_share_pct": _weighted_share(
            [(sum(b["lines"].values()), 1) for b in values], lambda n: 1 <= n <= 3
        ),
        "twenty_five_plus_unit_basket_share_pct": _weighted_share(
            [(sum(b["lines"].values()), 1) for b in values], lambda n: n >= 25
        ),
        "food_spend_target_month_usd": _weight_stats([(b["spend"], 1) for b in values]),
        "single_unit_line_share_pct": round(100 * sum(q == 1 for q in line_quantities) /
                                             len(line_quantities), 2),
    }


def _comparison(synthetic: dict, observed: dict):
    metrics = {}
    for metric in ("lines_per_basket", "units_per_basket", "food_spend_target_month_usd"):
        metrics[metric] = {}
        for stat in ("mean", "p10", "p25", "p50", "p75", "p90"):
            ref = observed[metric][stat]
            sim = synthetic[metric][stat]
            metrics[metric][stat] = {
                "observed": ref, "synthetic": sim,
                "difference": round(sim - ref, 2),
                "relative_error_pct": round(100 * (sim - ref) / ref, 1) if ref else None,
            }
    for metric in ("single_unit_line_share_pct", "one_to_three_line_basket_share_pct",
                   "one_to_three_unit_basket_share_pct", "twenty_five_plus_unit_basket_share_pct"):
        metrics[metric] = {
            "observed": observed[metric], "synthetic": synthetic[metric],
            "difference_percentage_points": round(synthetic[metric] - observed[metric], 2),
        }
    return metrics


def _markdown(result: dict):
    obs = result["observed"]["by_place"]["121"]
    sens = result["observed"]["by_place"]["122"]
    sim = result["synthetic"]
    cmp = result["primary_comparison"]
    source = result["source"]
    rows = [
        "# Public grocery purchase benchmark",
        "",
        "The observed records are USDA FoodAPS food purchases from 2012–13. The simulated records are one saved RETINA supercenter run. This is a diagnostic comparison, not validation of a particular retailer or of walking and shelf decisions.",
        "",
        f"**Sample:** {obs['events_with_item_rows']:,} paid USDA super-store acquisition events with an observed itemized receipt and item records, from {obs['unique_households']:,} households; {sim['paid_npc_baskets']:,} paid NPC baskets. FoodAPS household survey weights are applied per event. The NPC sample weight of 10 is not applied to per-basket metrics.",
        "",
        "| Food per basket | FoodAPS super store | NPC run | NPC minus FoodAPS |",
        "|---|---:|---:|---:|",
    ]
    for metric, label in (("lines_per_basket", "SKU lines"), ("units_per_basket", "Units"),
                          ("food_spend_target_month_usd", f"Food spend, roughly CPI adjusted to {source['target_month']} USD")):
        for key, suffix in (("mean", "mean"), ("p10", "p10"), ("p25", "p25"),
                            ("p50", "median"), ("p75", "p75"), ("p90", "p90")):
            c = cmp[metric][key]
            rows.append(f"| {label}, {suffix} | {c['observed']:.2f} | {c['synthetic']:.2f} | {c['difference']:+.2f} ({c['relative_error_pct']:+.1f}%) |")
    rows.extend(["", "| Distribution share | FoodAPS super store | NPC run | Difference |",
                 "|---|---:|---:|---:|"])
    for metric, label in (("one_to_three_line_basket_share_pct", "1–3 food SKU lines"),
                          ("one_to_three_unit_basket_share_pct", "1–3 food units"),
                          ("twenty_five_plus_unit_basket_share_pct", "25+ food units"),
                          ("single_unit_line_share_pct", "single-unit food SKU lines")):
        c = cmp[metric]
        rows.append(f"| {label} | {c['observed']:.2f}% | {c['synthetic']:.2f}% | {c['difference_percentage_points']:+.2f} pp |")
    rows.extend([
        "", "## Sensitivity and coverage", "",
        f"USDA supermarkets (`PLACETYPE=122`, {sens['events_with_item_rows']:,} events) have mean/median {sens['lines_per_basket']['mean']:.2f}/{sens['lines_per_basket']['p50']:.0f} food SKU lines and {sens['units_per_basket']['mean']:.2f}/{sens['units_per_basket']['p50']:.0f} food units. The small-basket finding is similar to super stores.",
        f"The primary USDA filter selected {obs['selected_events']:,} of {obs['all_place_events']:,} super-store events; all selected events had item rows. Food spend used {obs['complete_expense_events']:,} complete-price events; {obs['item_rows_missing_expense']} of {obs['item_rows']:,} item rows lacked expenditure. {obs['item_rows_negative_expense']} item rows had a negative net expense and were retained. {obs['item_rows_without_barcode']:,} item rows ({100 * obs['item_rows_without_barcode'] / obs['item_rows']:.1f}%) had no usable barcode and were grouped by description. RETINA excluded {sim['all_pos_lines'] - sim['food_pos_lines']} of {sim['all_pos_lines']:,} lines from nonfood, baby, pet and alcohol categories; {sim['baskets_without_food']} NPC baskets had no food lines.",
        f"Without barcode/description grouping, FoodAPS has {obs['raw_item_rows_per_basket']['mean']:.2f} raw item records per basket (median {obs['raw_item_rows_per_basket']['p50']:.0f}; 1–3-row share {obs['one_to_three_raw_item_row_basket_share_pct']:.2f}%). This is a sensitivity check, since repeated units can occupy separate rows and raw row counts are not SKU lines.",
        "", "## Method", "",
        "The USDA filter is `PLACETYPE=121`, `FREE=0`, `RECEIPTOBSERVED=1`, `TOTALPAID>0`, a positive `HHWGT`, and at least one item row. An observed itemized receipt does not imply every item record came only from that receipt; USDA also integrated scanner and diary records. A USDA SKU line groups repeated item records by barcode within an event; rows without a barcode group by normalized description. Units sum `QUANTITY`. This line grouping is approximate. The synthetic filter includes every paid NPC basket and food categories listed in the JSON report.",
        f"Food spending sums `TOTITEMEXP` for USDA events with all item expenses present. Each 2012–13 event is multiplied by the [BLS food-at-home CPI](https://download.bls.gov/pub/time.series/cu/cu.data.0.Current) (`CUUR0000SAF11`) ratio to {source['target_month']} (index {source['target_index']}). Synthetic food `NET_SALES` is adjusted from each simulated basket month to the same target month. This is only a rough price-level comparison: store assortment, geography, household mix, product sizes and catalog price dates differ.",
        "", "## Limits", "",
        f"FoodAPS is a household survey, not observed sales from this simulated retailer. Filtering to purchases with observed itemized receipts selects a subset, so these event quantiles should not be treated as nationally representative. HHWGT alone does not provide survey-design confidence intervals. FoodAPS covers food acquisitions and provides neither in-store paths nor rejected shelf offers or stockouts. This saved NPC run covers {sim['basket_days']} simulated {'day' if sim['basket_days'] == 1 else 'days'}; the comparison does not establish uncertainty across runs. Purchase-only records cannot teach a shelf SKIP decision directly.",
        "", "## Sources and reproduction", "",
        f"- [USDA FoodAPS public-use CSV]({source['foodaps_url']}) — local ZIP SHA256 `{source['foodaps_sha256']}`.",
        f"- [USDA food-at-home event codebook]({source['foodaps_codebook_url']}) and [item codebook]({source['foodaps_item_codebook_url']}).",
        f"- [BLS CPI data]({source['bls_url']}) — local file SHA256 `{source['bls_file_sha256']}`.",
        "- From the repository root, run `.venv\\Scripts\\python.exe data\\reference\\foodaps\\compare_saved_run.py`. The ignored USDA ZIP and existing BLS file are required; the command writes only this benchmark JSON and Markdown report.",
        "", "```powershell",
        "Invoke-WebRequest -Uri 'https://www.ers.usda.gov/media/5594/csv-data-files.zip?v=20317' -OutFile 'data\\reference\\foodaps\\csv-data-files.zip'",
        ".\\.venv\\Scripts\\python.exe data\\reference\\foodaps\\compare_saved_run.py",
        "```",
        "",
    ])
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--foodaps-zip", type=Path, default=FOODAPS_ZIP)
    parser.add_argument("--cpi", type=Path, default=CPI_FILE)
    parser.add_argument("--target-month", help="CPI target in YYYY-MM; inferred when the run has one month")
    parser.add_argument("--output", type=Path, default=HERE / "retina_supercenter_benchmark.json")
    args = parser.parse_args()
    cpi = _cpi(args.cpi)
    _, months = _run_calendar(args.run)
    if args.target_month:
        target_month = args.target_month
    elif len(months) == 1:
        target_month = months[0]
    else:
        parser.error("The run spans multiple months; supply --target-month YYYY-MM")
    if target_month not in cpi:
        parser.error(f"BLS food-at-home CPI has no index for {target_month}")
    synthetic = _synthetic(args.run, cpi, target_month)
    observed = _observed(args.foodaps_zip, cpi, target_month)
    comparison = _comparison(synthetic, observed["by_place"]["121"])
    result = {
        "benchmark": "USDA FoodAPS paid super-store food acquisitions with observed itemized receipts and item records",
        "source": {
            "foodaps_url": "https://www.ers.usda.gov/media/5594/csv-data-files.zip?v=20317",
            "foodaps_sha256": _sha256(args.foodaps_zip),
            "foodaps_codebook_url": "https://www.ers.usda.gov/sites/default/files/images/3_fah-events-codebook-puf.pdf",
            "foodaps_item_codebook_url": "https://www.ers.usda.gov/sites/default/files/images/4_fah-items-codebook-puf.pdf",
            "bls_url": "https://download.bls.gov/pub/time.series/cu/cu.data.0.Current",
            "bls_file_sha256": _sha256(args.cpi),
            "bls_series": "CUUR0000SAF11 (CPI-U food at home, US city average, not seasonally adjusted)",
            "target_month": target_month,
            "target_index": cpi[target_month],
            "synthetic_run": args.run.resolve().relative_to(ROOT).as_posix() if args.run.resolve().is_relative_to(ROOT)
            else str(args.run),
        },
        "method": {
            "observed_filters": "PLACETYPE=121 primary (122 sensitivity); FREE=0; RECEIPTOBSERVED=1; TOTALPAID>0; valid household weight and at least one item row",
            "observed_weight": "HHWGT applied to each acquisition event and its item lines",
            "observed_line_definition": "same-event barcode, or normalized description where barcode absent; approximation of distinct SKU receipt lines",
            "observed_raw_row_sensitivity": "ungrouped item record counts are reported for sensitivity; repeated units may appear on separate rows",
            "observed_units_definition": "sum of USDA item QUANTITY, including repeated records",
            "observed_spend_definition": f"sum of food-item TOTITEMEXP for complete-price events; scaled by monthly BLS food-at-home CPI to {target_month} USD",
            "synthetic_filter": "all paid NPC baskets; count only listed food categories for items and spend",
            "synthetic_spend_definition": f"sum of food line NET_SALES, scaled from each basket month by BLS food-at-home CPI to {target_month} USD",
            "synthetic_food_categories": sorted(FOOD_CATEGORIES),
            "sampling_note": "1 NPC=10 real people is a traffic scale; per-receipt comparisons use unexpanded NPC baskets",
            "limitations": [
                "FoodAPS is a 2012-13 household survey, not a transaction feed from the simulated store or Walmart.",
                "The observed-receipt filter selects a subset; household weights alone do not supply survey-design confidence intervals or make this event subset nationally representative.",
                "USDA strips nonfood items; synthetic nonfood categories are excluded. Generic receipt descriptions can merge distinct observed products.",
                "FoodAPS food item expenditure may miss some items; spend is based only on complete-price events and CPI scaling is approximate.",
                "FoodAPS has no within-store walking path, shelf offers, stockouts, or rejected items; those NPC decisions cannot be validated here.",
                f"One {synthetic['paid_npc_baskets']}-NPC saved run has sampling variation; these errors are diagnostics, not certified realism or a calibration target for a specific store.",
            ],
        },
        "observed": observed,
        "synthetic": synthetic,
        "primary_comparison": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(_markdown(result), encoding="utf-8")
    print(args.output)
    print("super-store events:", observed["by_place"]["121"]["events_with_item_rows"])
    print("NPC baskets:", synthetic["paid_npc_baskets"])
    for metric in comparison:
        if metric.endswith("_pct"):
            print(metric, comparison[metric])
        else:
            print(metric, "mean", comparison[metric]["mean"], "median", comparison[metric]["p50"])


if __name__ == "__main__":
    main()
