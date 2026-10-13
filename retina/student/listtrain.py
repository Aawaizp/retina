"""The list student, part 2: learn the teachers' shopping lists; export ONNX; model card.

One row per candidate product of a scenario (listplan.py): would this shopper put it on today's list
(yes/no), and how many. Features: the product (price, deal, how often bought, ran out / days left at home,
where it came from: usual / goes with / new idea / deal), the aisle (real Instacart re-buy rate) and the
shopper (price sensitivity, household, income, budget, usual trip size, days since the last trip).
NOT the product's position in the list.

Models: LightGBM classifier (buy?) + LightGBM regressor (how many, trained on bought rows).
Split by NPC (test NPCs never seen). Checks on the test NPCs, next to the teachers:
  list size (items per list), share of usual products (re-buys; real Instacart ~60%),
  share of deals, share of products that ran out, AUC and agreement (F1) with the teachers.
Runtime policy: each candidate is DRAWN with its probability (seeded dice) - a real shopper does not
write the same list every week - after a shift fitted on validation so the average list size matches
the teachers' (bias_shift in list_student.json).

  retina lists --train          -> models/list_student/ (list_student.onnx, list_qty.onnx, json, card)
"""

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from retina.student.listplan import ListLabels, load_pool

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "list_student"
AISLE_STATS = ROOT / "data" / "processed" / "instacart_aisle_stats.csv"
SOURCES = ["usual", "goes with", "new idea", "deal"]
FEATURES = ["price_log", "price_vs_list", "price_vs_budget", "deal", "off", "bought", "times_log", "times_rank",
            "ran_out", "days_left", "aisle_rebuy"] + [f"src_{s.replace(' ', '_')}" for s in SOURCES] + \
           ["sensitivity", "household", "kids", "income", "budget_log", "rebuy_rate", "usual_items", "since",
            "n_ran_out", "n_candidates"]
REAL_REPEAT = 0.60                     # Instacart reorder rate (PUBLIC_STATISTIC)


def split_of(npc):
    h = int(hashlib.md5(str(npc).encode()).hexdigest(), 16) % 100
    return "test" if h < 20 else ("val" if h < 36 else "train")


def scenario_matrix(t, rebuy):
    """Features (candidates x FEATURES, float32) of one scenario - the SAME function runs in the store."""
    c = t["candidates"]
    n = len(c)
    x = t["npc_x"]
    prices = np.array([max(o["price"], 0.01) for o in c])
    times = np.array([o["times"] for o in c], dtype=float)
    budget = t.get("budget") or 150.0
    rows = []
    for o, p, tm in zip(c, prices, times):
        rows.append([np.log(p), np.log(p) - np.log(prices).mean(), p / max(budget, 1.0), float(o["deal"]),
                     o.get("off", 0) / 100, float(o["times"] > 0), np.log1p(o["times"]),
                     tm / times.max() if times.max() > 0 else 0.0, float(o["ran_out"]),
                     min(o.get("days_left", 0), 30) / 30, rebuy.get(o["aisle"], 0.5)]
                    + [float(o["source"] == s) for s in SOURCES]
                    + [x["sensitivity"], x["household"], x["kids"], x["income"], np.log(max(budget, 1.0)), x["rebuy"],
                       t["usual_items"], min(t["since"] if t["since"] is not None else 7, 60) / 30,
                       sum(o["ran_out"] for o in c), n])
    return np.asarray(rows, dtype=np.float32)


def aisle_rebuy():
    if not AISLE_STATS.exists():
        return {}
    s = pd.read_csv(AISLE_STATS)
    return dict(zip(s["aisle"], s["rebuy_rate"]))


def examples(labels, pool, rebuy):
    by_id = {t["id"]: t for t in pool}
    sets = []
    for r in labels:
        t = by_id.get(r["id"])
        if t is None:
            continue
        chosen = {int(k): v for k, v in r["list"].items()}
        X = scenario_matrix(t, rebuy)
        y = np.array([float(o["product"] in chosen) for o in t["candidates"]])
        q = np.array([chosen.get(o["product"], 0) for o in t["candidates"]], dtype=float)
        sets.append({"id": t["id"], "teacher": r["teacher"], "npc": t["npc"], "split": split_of(t["npc"]),
                     "X": X, "y": y, "q": q, "cands": t["candidates"]})
    return sets


def stack(sets):
    return (np.vstack([s["X"] for s in sets]), np.concatenate([s["y"] for s in sets]),
            np.concatenate([s["q"] for s in sets]))


def list_stats(sets, picks):
    """Per-list behaviour: size, usual share, deal share, ran-out share (picks = 0/1 arrays per set)."""
    size, usual, deal, ran = [], [], [], []
    for s, p in zip(sets, picks):
        k = p > 0.5
        size.append(k.sum())
        if k.sum():
            usual.append(np.mean([o["times"] > 0 for o, b in zip(s["cands"], k) if b]))
            deal.append(np.mean([o["deal"] for o, b in zip(s["cands"], k) if b]))
            ran.append(np.mean([o["ran_out"] for o, b in zip(s["cands"], k) if b]))
    m = lambda v: round(float(np.mean(v)), 3) if v else None
    return {"items_per_list": m(size), "usual_share": m(usual), "deal_share": m(deal), "ran_out_share": m(ran)}


def fit_shift(probs, sets, target):
    """Logit shift so the EXPECTED list size (sum of drawn probabilities) matches the teachers' on validation."""
    logit = [np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))) for p in probs]
    lo, hi = -6.0, 6.0
    for _ in range(40):
        mid = (lo + hi) / 2
        size = np.mean([(1 / (1 + np.exp(-(l + mid)))).sum() for l in logit])
        lo, hi = (mid, hi) if size < target else (lo, mid)
    return round((lo + hi) / 2, 4)


def train(labels=None, pool=None, out=MODEL_DIR, say=print, min_lists=60, seed=7):
    import lightgbm as lgb
    from sklearn.metrics import f1_score, roc_auc_score
    labels = labels if labels is not None else ListLabels().rows()
    pool = pool if pool is not None else load_pool(say=say)
    rebuy = aisle_rebuy()
    sets = examples(labels, pool, rebuy)
    parts = {k: [s for s in sets if s["split"] == k] for k in ("train", "val", "test")}
    teachers = pd.Series([s["teacher"] for s in sets]).value_counts().to_dict()
    say(f"List labels: {len(sets):,} lists from {teachers} | " +
        " | ".join(f"{k} {len(v):,}" for k, v in parts.items()))
    if len(parts["train"]) < min_lists or len(parts["val"]) < 10 or len(parts["test"]) < 10:
        raise SystemExit(f"Too few teacher lists to train (need about {min_lists}+ training lists). "
                         "Run: retina lists --collect 500 --provider gemini (or a paste pack)")
    Xtr, ytr, qtr = stack(parts["train"])
    Xva, yva, _ = stack(parts["val"])
    buy = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=20,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
    buy.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=[lgb.early_stopping(40, verbose=False)])
    qty = lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=10,
                            random_state=seed, verbose=-1)
    qty.fit(Xtr[ytr > 0], qtr[ytr > 0])
    p_val = [buy.predict_proba(s["X"])[:, 1] for s in parts["val"]]
    teacher_size = float(np.mean([s["y"].sum() for s in parts["train"] + parts["val"]]))
    shift = fit_shift(p_val, parts["val"], teacher_size)

    def drawn(p, rng):
        logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))) + shift
        return (rng.random(len(p)) < 1 / (1 + np.exp(-logit))).astype(float)

    test = parts["test"]
    p_test = [buy.predict_proba(s["X"])[:, 1] for s in test]
    rng = np.random.default_rng(seed)
    student_picks = [drawn(p, rng) for p in p_test]
    _, yte, _ = stack(test)
    pte = np.concatenate(p_test)
    result = {"auc": round(float(roc_auc_score(yte, pte)), 3) if 0 < yte.mean() < 1 else None,
              "f1_with_teacher": round(float(f1_score(yte, np.concatenate(student_picks))), 3),
              "teacher": list_stats(test, [s["y"] for s in test]),
              "student": list_stats(test, student_picks),
              "qty_mae": round(float(np.mean(np.abs(np.clip(np.round(qty.predict(stack(test)[0][yte > 0])), 1, 6)
                                                    - stack(test)[2][yte > 0]))), 3) if (yte > 0).any() else None}
    say(f"  test NPCs: AUC {result['auc']}, F1 with the teachers {result['f1_with_teacher']}")
    for who in ("teacher", "student"):
        r = result[who]
        say(f"  {who:8} items/list {r['items_per_list']}, usual share {r['usual_share']}, deals {r['deal_share']}, "
            f"ran out {r['ran_out_share']}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    export(buy, out / "list_student.onnx", classifier=True)
    export(qty, out / "list_qty.onnx", classifier=False)
    meta = {"version": f"list-student-{date.today().isoformat()}", "features": FEATURES, "bias_shift": shift,
            "teacher_items_per_list": round(teacher_size, 2), "labels": len(sets), "teachers": teachers,
            "sets": {k: len(v) for k, v in parts.items()}, "test": result, "real_repeat_share": REAL_REPEAT,
            "policy": "draw each candidate with sigmoid(logit(p) + bias_shift); quantity = round(qty model), 1-6"}
    (out / "list_student.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    (out / "MODEL_CARD.md").write_text(card(meta), encoding="utf-8")
    say(f"List student -> {out}")
    return meta


def export(model, path, classifier):
    import onnx
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType
    n = len(FEATURES)
    conv = onnxmltools.convert_lightgbm(model, initial_types=[("x", FloatTensorType([None, n]))], target_opset=15,
                                        zipmap=False) if classifier else \
        onnxmltools.convert_lightgbm(model, initial_types=[("x", FloatTensorType([None, n]))], target_opset=15)
    onnx.save(conv, path)


def card(m):
    t, s = m["test"]["teacher"], m["test"]["student"]
    pct = lambda x: "-" if x is None else f"{x:.0%}"
    return "\n".join([
        f"# RETINA list student - {m['version']}", "",
        "Plans an NPC's shopping list: from ~30 real candidate products of the RETINA Supercenter (their usual "
        "products, what goes with them, new ideas, deals) it decides what goes on today's list and how many. "
        "Learned from LLM teachers' lists. Every output is SIMULATED behaviour.", "",
        f"- Lists: {m['labels']:,} from {', '.join(f'{k} {v}' for k, v in m['teachers'].items())}; split by NPC "
        f"(train {m['sets']['train']}, validation {m['sets']['val']}, test {m['sets']['test']}).", "",
        "| on test NPCs | items per list | usual products | deals | ran out at home |", "|---|---|---|---|---|",
        f"| teachers | {t['items_per_list']} | {pct(t['usual_share'])} | {pct(t['deal_share'])} | {pct(t['ran_out_share'])} |",
        f"| list student (drawn) | {s['items_per_list']} | {pct(s['usual_share'])} | {pct(s['deal_share'])} | "
        f"{pct(s['ran_out_share'])} |", "",
        f"AUC {m['test']['auc']}, F1 with the teachers {m['test']['f1_with_teacher']}, quantity error (MAE) "
        f"{m['test']['qty_mae']}. Real re-buy share for comparison: ~{m['real_repeat_share']:.0%} (Instacart).", "",
        f"Policy: {m['policy']}; bias_shift {m['bias_shift']} makes the average list size match the teachers' "
        f"({m['teacher_items_per_list']}).", ""])
