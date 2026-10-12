"""Every store format in stores/ opens and runs, and the formats differ the way they should."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODE = ("import json, engine\n"
        "sim = engine.Simulation(verbose=False)\n"
        "sim.run_day(); sim.run_day()\n"
        "r = sim.results()\n"
        "print(json.dumps({'trips': len(r.trips), 'items': float(r.sales.quantity.sum() / len(r.trips)),"
        " 'products': int(sum(len(d.product_idx) for d in sim.store.departments.values()))}))\n")


def run(store):
    out = subprocess.run([sys.executable, "-c", CODE], cwd=ROOT / "simulation", capture_output=True, text=True,
                         env=dict(os.environ, RETINA_STORE=store))
    assert out.returncode == 0, out.stderr[-800:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_every_store_format_runs_and_differs():
    stores = sorted(p.stem for p in (ROOT / "stores").glob("*.json"))
    results = {s: run(s) for s in stores}
    for s, r in results.items():
        assert r["trips"] > 0 and r["items"] >= 1, s
    if {"convenience", "generic_supermarket", "supercenter"} <= set(results):
        c, m, b = results["convenience"], results["generic_supermarket"], results["supercenter"]
        assert c["items"] < m["items"] and c["products"] < m["products"]   # small trips, small range
        assert b["trips"] > m["trips"]                                      # the big box is busier
