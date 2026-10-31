"""Basket completion student, trained on real Instacart order lengths.

The training labels say whether an online order ended after its kth distinct product.
They do not reveal a shopper's physical route or whether an intended item was abandoned.
"""

import json
import os
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.environ["RETINA_TRIP_END_DIR"]) if os.environ.get("RETINA_TRIP_END_DIR") else ROOT / "models" / "trip_end_student"
RAW = ROOT / "data" / "raw" / "instacart"
FEATURES = ["basket_lines_over_usual", "usual_basket_lines", "basket_lines", "days_since_prior_order"]


def features(lines, usual, since):
    usual = max(float(usual), 1.0)
    return np.array([[float(lines) / usual, usual, float(lines), max(float(since), 0.0)]], dtype=np.float32)


class TripEndStudent:
    def __init__(self, folder=MODEL_DIR):
        import onnxruntime as rt

        folder = Path(folder)
        self.meta = json.loads((folder / "trip_end.json").read_text(encoding="utf-8"))
        if self.meta["features"] != FEATURES:
            raise ValueError("trip_end.json features differ from this RETINA version; retrain with retina trip-end --train")
        self.session = rt.InferenceSession(str(folder / "trip_end.onnx"))
        self.input = self.session.get_inputs()[0].name

    @staticmethod
    def available(folder=MODEL_DIR):
        folder = Path(folder)
        return (folder / "trip_end.onnx").is_file() and (folder / "trip_end.json").is_file()

    def probability(self, lines, usual, since):
        if lines < 1:
            return 0.0
        out = self.session.run(None, {self.input: features(lines, usual, since)})[1]
        p = np.asarray(out)
        return float(p[0, 1] if p.ndim == 2 else p[0][1])


def order_lengths(raw=RAW):
    """Count distinct order lines without holding the large order-products CSV in memory."""
    import pandas as pd

    orders = pd.read_csv(raw / "orders.csv", usecols=["order_id", "user_id", "eval_set", "order_number",
                                                      "days_since_prior_order"],
                         dtype={"order_id": "int32", "user_id": "int32", "order_number": "int16",
                                "eval_set": "category", "days_since_prior_order": "float32"})
    counts = np.zeros(int(orders.order_id.max()) + 1, dtype=np.int32)
    for chunk in pd.read_csv(raw / "order_products__prior.csv", usecols=["order_id"],
                             dtype={"order_id": "int32"}, chunksize=1_000_000):
        counts += np.bincount(chunk.order_id.to_numpy(), minlength=len(counts)).astype(np.int32)
    orders = orders[orders.eval_set == "prior"].copy()
    orders["lines"] = counts[orders.order_id.to_numpy()]
    orders.sort_values(["user_id", "order_number"], inplace=True)
    prior = orders.groupby("user_id", sort=False)["lines"]
    orders["previous_orders"] = prior.cumcount().to_numpy()
    orders["usual"] = ((prior.cumsum() - orders.lines) / orders.previous_orders.replace(0, np.nan)).to_numpy()
    return orders[(orders.previous_orders >= 2) & orders.days_since_prior_order.notna()
                  & orders.lines.between(1, 60)].copy()


def examples(orders):
    """One risk-set row at each observed prefix; only the final prefix is a stop."""
    sizes = orders.lines.to_numpy(dtype=np.int32)
    repeats = np.repeat(np.arange(len(orders)), sizes)
    k = np.arange(int(sizes.sum()), dtype=np.int32) - np.repeat(np.cumsum(sizes) - sizes, sizes) + 1
    usual = np.maximum(orders.usual.to_numpy(dtype=np.float32)[repeats], 1.0)
    since = orders.days_since_prior_order.to_numpy(dtype=np.float32)[repeats]
    X = np.column_stack((k / usual, usual, k, since)).astype(np.float32)
    y = (k == sizes[repeats]).astype(np.int8)
    return X, y


def train(raw=RAW, out=MODEL_DIR, max_orders=80_000, seed=19, say=print):
    import lightgbm as lgb
    import onnx
    import onnxmltools
    import onnxruntime as rt
    from onnxmltools.convert.common.data_types import FloatTensorType
    from sklearn.metrics import brier_score_loss, roc_auc_score

    orders = order_lengths(Path(raw))
    rng = np.random.default_rng(seed)
    if len(orders) > max_orders:
        orders = orders.iloc[np.sort(rng.choice(len(orders), max_orders, replace=False))]
    split = (orders.user_id.to_numpy(dtype=np.int64) * 2654435761 % 100) // 10
    sets = {"train": orders[split < 8], "validation": orders[split == 8], "test": orders[split == 9]}
    say("Order examples: " + ", ".join(f"{k} {len(v):,}" for k, v in sets.items()))
    if min(map(len, sets.values())) < 100:
        raise ValueError("Too few orders for a user-separated train and test split")
    Xtr, ytr = examples(sets["train"])
    Xva, yva = examples(sets["validation"])
    Xte, yte = examples(sets["test"])
    model = lgb.LGBMClassifier(n_estimators=220, learning_rate=0.045, num_leaves=15,
                               min_child_samples=100, random_state=seed, verbose=-1)
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(25, verbose=False)])
    p = model.predict_proba(Xte)[:, 1]
    metrics = {"auc": round(float(roc_auc_score(yte, p)), 4),
               "brier": round(float(brier_score_loss(yte, p)), 4),
               "observed_stop_rate": round(float(yte.mean()), 4),
               "predicted_stop_rate": round(float(p.mean()), 4),
               "mean_order_lines": round(float(sets["test"].lines.mean()), 2)}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    graph = onnxmltools.convert_lightgbm(model, initial_types=[("x", FloatTensorType([None, len(FEATURES)]))],
                                         target_opset=15, zipmap=False)
    for output in graph.graph.output:
        dims = output.type.tensor_type.shape.dim
        if dims:
            dims[0].ClearField("dim_value")
            dims[0].dim_param = "N"
    onnx.save(graph, out / "trip_end.onnx")
    session = rt.InferenceSession(str(out / "trip_end.onnx"))
    onnx_p = np.asarray(session.run(None, {session.get_inputs()[0].name: Xte[:256]})[1])[:, 1]
    metrics["onnx_max_diff"] = float(np.max(np.abs(onnx_p - p[:256])))
    if metrics["onnx_max_diff"] > 1e-4:
        raise AssertionError("ONNX predictions differ from LightGBM")
    meta = {"version": f"trip-end-{date.today().isoformat()}", "features": FEATURES,
            "source": "Instacart Market Basket Analysis: prior orders and ordered product lines",
            "orders": {k: len(v) for k, v in sets.items()}, "test": metrics,
            "in_store_scale": 0.1, "in_store_scale_label": "ESTIMATED",
            "label": "last product line in an observed online order", "max_training_lines": 60,
            "limit": "No observed in-store checkout timing, physical route, or abandoned intended items. "
                     "Applied to simulated stores as a basket-completion proxy."}
    (out / "trip_end.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "MODEL_CARD.md").write_text(
        "# Basket completion student\n\n"
        "Trained on real Instacart prior orders. At each observed order prefix, the target is whether that "
        "product line was the final line. Training, validation, and test users are separate. "
        "At a simulated shelf the model estimates whether the current basket is complete; seeded dice turn "
        "that probability into a live checkout choice. One local ONNX inference per eligible shelf; no LLM call.\n\n"
        f"Test: {metrics}.\n\n"
        "**Limit:** Instacart records online order contents and cart order, not physical store paths, "
        "intentions, or early departures. Early checkout in RETINA is an inferred behavior, not a directly "
        "observed in-store label. The raw order hazard is scaled by 0.1 in RETINA (ESTIMATED) because "
        "the list student was already calibrated to basket size; an unscaled checkout hazard reduced "
        "the simulated mean from 9.92 to 7.68 lines in a one-day run. Store-specific basket size may "
        "need further calibration. Raw Instacart files remain local.\n",
        encoding="utf-8")
    say(f"Trip-end student -> {out}; test {metrics}")
    return meta
