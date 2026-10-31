"""Train and validate the optional local shelf SKIP policy.

Run: python -m retina.student.shelf_action_train

The input is existing LLM teacher labels. NPC IDs are held out as whole groups.
This does not establish accuracy against real shoppers; the model stays disabled
for live runs unless its held-out skip discrimination and calibration qualify.
"""

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

from retina.student.labels import LabelStore, TEACHER_DIR
from retina.student.shelf_action import FEATURES, MODEL_DIR, MODEL_FILE, feature_vector


def is_skip(row):
    choice = row.get("choice")
    return choice is None or int(choice) < 0


def split_rows(rows):
    y = np.asarray([is_skip(row) for row in rows], dtype=int)
    groups = np.asarray([int(row["npc"]) for row in rows])
    folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026).split(
        np.zeros(len(rows)), y, groups))
    test = folds[0][1]
    val = folds[1][1]
    train = np.asarray(sorted(set(range(len(rows))) - set(test) - set(val)), dtype=int)
    if set(groups[train]) & set(groups[val]) or set(groups[train]) & set(groups[test]) or set(groups[val]) & set(groups[test]):
        raise RuntimeError("NPC data leaked across shelf action splits")
    return train, val, test, y


def metrics(y, p):
    prevalence = float(np.mean(y))
    out = {"rows": len(y), "skips": int(np.sum(y)), "observed_skip_share": round(prevalence, 5),
           "predicted_skip_share": round(float(np.mean(p)), 5),
           "average_precision": round(float(average_precision_score(y, p)), 5),
           "baseline_average_precision": round(prevalence, 5),
           "brier": round(float(brier_score_loss(y, p)), 6),
           "baseline_brier": round(float(brier_score_loss(y, np.full(len(y), prevalence))), 6),
           "log_loss": round(float(log_loss(y, p, labels=[0, 1])), 5)}
    out["roc_auc"] = round(float(roc_auc_score(y, p)), 5) if 0 < np.sum(y) < len(y) else None
    return out


def calibrate_intercept(logits, target):
    """Fit a single intercept on validation NPCs, keeping the classifier fixed."""
    lo, hi = -20.0, 20.0
    for _ in range(70):
        mid = (lo + hi) / 2
        rate = np.mean(1.0 / (1.0 + np.exp(-np.clip(logits + mid, -30, 30))))
        if rate < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def train(rows):
    X = np.stack([feature_vector(r["npc_x"], r["route"], r["mission"], r["options"],
                                 r["engine_product"]) for r in rows])
    train_idx, val_idx, test_idx, y = split_rows(rows)
    scaler = StandardScaler().fit(X[train_idx])
    X_train = scaler.transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    X_test = scaler.transform(X[test_idx])
    candidates = []
    for c in (0.01, 0.1, 1.0):
        for weight in (None, "balanced"):
            model = LogisticRegression(C=c, class_weight=weight, max_iter=3000, random_state=7)
            model.fit(X_train, y[train_idx])
            base_logit = X_val @ model.coef_[0] + model.intercept_[0]
            offset = calibrate_intercept(base_logit, float(np.mean(y[val_idx])))
            p = 1 / (1 + np.exp(-np.clip(base_logit + offset, -30, 30)))
            score = metrics(y[val_idx], p)
            candidates.append((score["log_loss"], -score["average_precision"], c, weight, model, offset, score))
    _, _, c, weight, model, offset, val_score = min(candidates, key=lambda item: item[:2])
    test_logit = X_test @ model.coef_[0] + model.intercept_[0] + offset
    p_test = 1 / (1 + np.exp(-np.clip(test_logit, -30, 30)))
    test_score = metrics(y[test_idx], p_test)
    # Predeclared gate: good ranking on both held-out groups and no loss of
    # probability calibration versus a constant-rate predictor on test NPCs.
    qualified = bool(val_score["skips"] >= 8 and test_score["skips"] >= 8 and
                     val_score["average_precision"] >= 3 * val_score["baseline_average_precision"] and
                     test_score["average_precision"] >= 3 * test_score["baseline_average_precision"] and
                     test_score["brier"] <= test_score["baseline_brier"] and
                     0.5 * test_score["observed_skip_share"] <= test_score["predicted_skip_share"] <=
                     2.0 * test_score["observed_skip_share"])
    meta = {"version": 1, "name": "shelf-skip-student", "date": str(date.today()),
            "qualified": qualified, "features": FEATURES,
            "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
            "coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0] + offset),
            "training": {"source": "existing LLM teacher shelf labels; no observed shopper SKIPs",
                         "split": "StratifiedGroupKFold by NPC, seed 2026, fold 0 test, fold 1 validation",
                         "rows": len(rows), "skips": int(y.sum()),
                         "train_rows": len(train_idx), "train_skips": int(y[train_idx].sum()),
                         "validation": val_score, "test": test_score,
                         "selected_C": c, "selected_class_weight": weight,
                         "qualification_gate": "validation and test AP >= 3x prevalence; test Brier <= constant-rate baseline; test expected skip share within 0.5-2x observed"}}
    return meta


def model_card(meta):
    tr = meta["training"]
    va, te = tr["validation"], tr["test"]
    status = "QUALIFIED for optional live use" if meta["qualified"] else "EXPERIMENTAL; live use stays disabled"
    return f"""# Shelf action student — {meta['date']}

**Status: {status}.** This is a local binary SKIP-vs-BUY head. When it chooses
BUY, the existing product student chooses the SKU. The list student plans
quantity; live stock and remaining budget can reduce it. No live LLM call.

## Evidence

- Source: {tr['rows']:,} valid LLM teacher shelf labels, only **{tr['skips']} SKIPs**.
- Train: {tr['train_rows']:,} situations, {tr['train_skips']} SKIPs. Entire NPCs were held out for validation and test.
- Validation: {va['rows']} situations, {va['skips']} SKIPs; average precision {va['average_precision']:.3f} vs {va['baseline_average_precision']:.3f} prevalence baseline; Brier {va['brier']:.4f}.
- Test: {te['rows']} situations, {te['skips']} SKIPs; average precision {te['average_precision']:.3f} vs {te['baseline_average_precision']:.3f} prevalence baseline; Brier {te['brier']:.4f} vs {te['baseline_brier']:.4f} constant-rate baseline; predicted skip share {te['predicted_skip_share']:.1%} vs observed {te['observed_skip_share']:.1%}.
- Chosen logistic regularization C={tr['selected_C']}; class weighting {tr['selected_class_weight']}. Features include shopper traits, trip intent, stock, price, deals, and previous purchases; option order and teacher identity are excluded.

## Limits

This model imitates LLM labels, **not real shoppers**. SKIPs are sparse and
mostly from one teacher family; there are only {te['skips']} test SKIPs, so
precision has high uncertainty. A good teacher-label test is not proof of
retail realism. Store POS, shelf stockouts, and foot traffic are needed to
calibrate actions and evaluate the final baskets. Quantity remains the list
student's earlier plan, not a new quantity inferred at the shelf.

The `qualified` flag in `shelf_action.json` controls whether live runs may use
the policy. Re-run `python -m retina.student.shelf_action_train` to reproduce
the training and held-out evaluation from the local teacher label database.
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=TEACHER_DIR / "labels.sqlite")
    parser.add_argument("--out", type=Path, default=MODEL_DIR)
    parser.add_argument("--force", action="store_true", help="replace an earlier shelf-action artifact")
    args = parser.parse_args(argv)
    output = args.out / MODEL_FILE
    if output.exists() and not args.force:
        parser.error(f"{output} already exists; pass --force to replace it")
    labels = LabelStore(args.labels)
    try:
        rows = labels.rows()
    finally:
        labels.close()
    if not rows:
        parser.error(f"no valid labels in {args.labels}")
    meta = train(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    (args.out / "MODEL_CARD.md").write_text(model_card(meta), encoding="utf-8")
    print(json.dumps({"qualified": meta["qualified"], "validation": meta["training"]["validation"],
                      "test": meta["training"]["test"]}, indent=2))


if __name__ == "__main__":
    main()
