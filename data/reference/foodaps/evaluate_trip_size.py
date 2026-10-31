"""Run isolated NPC days with and without the FoodAPS trip-size prior.

Saved live runs are never read or changed. Set RETINA_STORE before importing RETINA.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["RETINA_STORE"] = "retina_supercenter"
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from retina.engine.engine import Simulation  # noqa: E402
from retina.engine import traffic  # noqa: E402
from retina.mind.mind import Mind  # noqa: E402
from retina.student.trip_size import DEFAULT_FILE  # noqa: E402


FOOD = {"produce", "dairy eggs", "snacks", "beverages", "frozen", "bakery",
        "pantry", "dry goods pasta", "deli", "breakfast", "canned goods",
        "meat seafood", "international"}


def run(use_prior, days):
    os.environ["RETINA_TRIP_SIZE_CALIBRATION"] = "1" if use_prior else "0"
    # One NPC represents ten visitors, matching the intended live-store scale.
    traffic.TRAFFIC["npc_scale"] = 0.1
    with tempfile.TemporaryDirectory(prefix="retina_trip_size_", dir=ROOT / "data" / "reference" / "foodaps") as directory:
        path = Path(directory)
        sim = Simulation(days=days, verbose=False)
        mind = Mind(sim, "student", path, provider="fake", cache_file=path / "cache.sqlite", say=lambda _: None)
        sim.mind = mind
        baskets = []
        planned = []
        total_lines = []
        essentials = []
        big_plans = []
        try:
            for _ in range(days):
                sales, trips = sim.run_day()
                food_idx = sim.store.products["department"].isin(FOOD).to_numpy()
                food_sales = sales[sales["product_idx"].map(lambda idx: food_idx[int(idx)])]
                food_lines = food_sales.groupby("trip_id")["product_idx"].nunique().to_dict()
                for trip_id in trips["trip_id"]:
                    baskets.append(int(food_lines.get(trip_id, 0)))
                planned.extend(int(x) for x in trips["planned_items"])
                total_lines.extend(int(x) for x in trips["lines"])
                essentials.extend(int(x) for x in trips["essential_items"])
                big_plans.extend((int(row.planned_items), int(row.lines), float(row.spent), float(row.trip_budget))
                                 for row in trips.itertuples(index=False) if int(row.planned_items) >= 25)
        finally:
            mind.close()
    values = sorted(baskets)
    n = len(values)
    return {"paid_baskets": n, "mean_food_lines": round(sum(values) / n, 3),
            "median_food_lines": values[(n - 1) // 2],
            "small_share": round(sum(1 <= x <= 3 for x in values) / n, 5),
            "large_share": round(sum(x >= 25 for x in values) / n, 5),
            "zero_food_share": round(sum(x == 0 for x in values) / n, 5),
            "planned_mean": round(sum(planned) / n, 3),
            "planned_small_share": round(sum(x <= 3 for x in planned) / n, 5),
            "planned_large_share": round(sum(x >= 25 for x in planned) / n, 5),
            "total_lines_mean": round(sum(total_lines) / n, 3),
            "essentials_mean": round(sum(essentials) / n, 3),
            "large_plans": len(big_plans),
            "large_plan_paid_lines_mean": round(sum(x[1] for x in big_plans) / len(big_plans), 3) if big_plans else None,
            "large_plan_spent_over_budget_mean": round(sum(x[2] / x[3] for x in big_plans if x[3] > 0) / len(big_plans), 3) if big_plans else None,
            "histogram": [values.count(x) for x in range(31)]}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=2)
    args = parser.parse_args()
    target = json.loads(DEFAULT_FILE.read_text(encoding="utf-8"))["held_out"]["test"]
    baseline = run(False, args.days)
    print("BASELINE", json.dumps(baseline), flush=True)
    candidate = run(True, args.days)
    print("CANDIDATE", json.dumps(candidate), flush=True)
    print("HELD_OUT", json.dumps(target), flush=True)
