"""Fit a public FoodAPS super-store basket-size prior; hold out whole households."""

import csv
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from .trip_size import DEFAULT_FILE, ROOT


ARCHIVE = ROOT / "data" / "reference" / "foodaps" / "csv-data-files.zip"


def rows(archive, name):
    with archive.open(name) as stream:
        yield from csv.DictReader(io.TextIOWrapper(stream, encoding="cp1252", newline=""))


def quantile(values, weights, point):
    order = np.argsort(values)
    index = np.searchsorted(np.cumsum(weights[order]), point * weights.sum())
    return int(values[order[min(index, len(order) - 1)]])


def fit(archive_path=ARCHIVE, output=DEFAULT_FILE):
    with ZipFile(archive_path) as archive:
        weights = {r["hhnum"]: float(r["hhwgt"]) for r in rows(archive, "faps_household_puf.csv")
                   if r["hhwgt"] not in ("", ".") and float(r["hhwgt"]) > 0}
        events = {r["eventid"]: (r["hhnum"], weights[r["hhnum"]])
                  for r in rows(archive, "faps_fahevent_puf.csv")
                  if r["placetype"] == "121" and r["free"] == "0"
                  and r["receiptobserved"] == "1" and r["hhnum"] in weights
                  and r["totalpaid"] not in ("", ".") and float(r["totalpaid"]) > 0}
        lines = defaultdict(set)
        for r in rows(archive, "faps_fahitem_puf.csv"):
            event = r["eventid"]
            if event not in events:
                continue
            barcode = r["barcode"].strip()
            key = ("barcode", barcode) if barcode and barcode not in (".V", ".", "-996") else (
                "description", " ".join(r["itemdesc"].upper().split()) or r["itemnum"])
            lines[event].add(key)
    samples = []
    for event, distinct in lines.items():
        hh, weight = events[event]
        split = "train" if hashlib.sha256(hh.encode()).digest()[0] < 179 else "test"
        samples.append((split, min(len(distinct), 30), weight))
    stats = {}
    for split in ("train", "test"):
        group = [(n, w) for part, n, w in samples if part == split]
        counts = np.asarray([n for n, _ in group], dtype=int)
        survey_weights = np.asarray([w for _, w in group], dtype=float)
        stats[split] = {"events": len(group), "households": len({events[e][0] for e in lines
                            if ("train" if hashlib.sha256(events[e][0].encode()).digest()[0] < 179 else "test") == split}),
                        "mean": round(float(np.average(counts, weights=survey_weights)), 3),
                        "median": quantile(counts, survey_weights, .5),
                        "small_share": round(float(survey_weights[counts <= 3].sum() / survey_weights.sum()), 5),
                        "large_share": round(float(survey_weights[counts >= 25].sum() / survey_weights.sum()), 5)}
    train = [(n, w) for part, n, w in samples if part == "train"]
    values = np.arange(1, 31)
    mass = np.asarray([sum(w for n, w in train if n == value) for value in values])
    probabilities = mass / mass.sum()
    # The independent households verify the observed tail pattern before runtime testing.
    qualified = stats["test"]["events"] >= 500 and stats["test"]["small_share"] > .15 and stats["test"]["large_share"] > .04
    meta = {"version": 1, "qualified": bool(qualified), "source": "USDA FoodAPS 2012-13 public-use, paid receipt-observed super-store events",
            "split": "SHA256 household ID, 70/30 approximate; whole households held out",
            "counts": values.tolist(), "probabilities": probabilities.tolist(), "held_out": stats,
            "limit": "Food basket size prior only; not Walmart-specific, not a model of shelf choices"}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


if __name__ == "__main__":
    print(json.dumps(fit()["held_out"], indent=2))
