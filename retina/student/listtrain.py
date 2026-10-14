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
write the same list every week - after a shift (bias_shift in list_student.json) that sets the average
list size: ESTIMATED from real baskets (Instacart ~10.1 products / the share of planned items bought here),
not the teachers' longer lists. ESTIMATED habit correction: the teachers' lists are ~90%
usual products against a real ~60%, so one extra term lowers the logit of products bought before; it is
tuned (with its own shift, list size kept) on all pool scenarios of non-test NPCs, checked on the test NPCs.

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
REAL_LINES = 10.09                     # Instacart products per order (PUBLIC_STATISTIC, instacart_aisle_stats.csv)
PLAN_FILL = 0.871                      # bought lines per planned item (stock-outs, budget): 1-month student run, 14 Oct
LIST_SIZE = round(REAL_LINES / PLAN_FILL, 2)   # ESTIMATED list size that gives real-size baskets (~11.6)


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


BOUGHT = FEATURES.index("bought")


def policy(p, X, shift, habit=0.0):
    """The runtime probability of each candidate: logit(p) + shift - habit x (bought before)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-(np.log(p / (1 - p)) + shift - habit * X[:, BOUGHT])))


def fit_shift(probs, Xs, target, habit=0.0):
    """The shift that makes the EXPECTED list size (sum of the drawn probabilities) = target."""
    lo, hi = -8.0, 8.0
    for _ in range(40):
        mid = (lo + hi) / 2
        size = np.mean([policy(p, X, mid, habit).sum() for p, X in zip(probs, Xs)])
        lo, hi = (mid, hi) if size < target else (lo, mid)
    return round((lo + hi) / 2, 4)


def usual_share(probs, Xs, shift, habit):
    shares = []
    for p, X in zip(probs, Xs):
        q = policy(p, X, shift, habit)
        if q.sum() > 0:
            shares.append(float((q * X[:, BOUGHT]).sum() / q.sum()))
    return float(np.mean(shares))


def fit_habit(probs, Xs, size, target):
    """ESTIMATED habit correction: the habit term (with its own size shift) that brings the EXPECTED share of
    usual products on a list to `target` (the real re-buy rate) while the list size stays `size`.
    Uses only the student's own probabilities - no teacher answer."""
    if usual_share(probs, Xs, fit_shift(probs, Xs, size), 0.0) <= target:
        return 0.0, fit_shift(probs, Xs, size)
    lo, hi = 0.0, 12.0
    for _ in range(30):
        mid = (lo + hi) / 2
        share = usual_share(probs, Xs, fit_shift(probs, Xs, size, mid), mid)
        lo, hi = (mid, hi) if share > target else (lo, mid)
    habit = round((lo + hi) / 2, 4)
    return habit, fit_shift(probs, Xs, size, habit)


def train(labels=None, pool=None, out=MODEL_DIR, say=print, min_lists=60, seed=7, habit_target=REAL_REPEAT,
          size_target=LIST_SIZE):
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
    teacher_size = float(np.mean([s["y"].sum() for s in parts["train"] + parts["val"]]))
    p_val = [buy.predict_proba(s["X"])[:, 1] for s in parts["val"]]
    raw_shift = fit_shift(p_val, [s["X"] for s in parts["val"]], teacher_size)
    size = size_target or teacher_size               # ESTIMATED: real basket size, not the teachers' list size
    test_npcs = {s["npc"] for s in parts["test"]}    # tuned on ALL pool scenarios of non-test NPCs (days 14-91,
    X_cal = [scenario_matrix(t, rebuy) for t in pool   # labelled or not): no teacher answer is used
             if t["candidates"] and t["npc"] not in test_npcs and split_of(t["npc"]) != "test"]
    p_cal = [buy.predict_proba(X)[:, 1] for X in X_cal]
    habit, shift = fit_habit(p_cal, X_cal, size, habit_target) if habit_target is not None else         (0.0, fit_shift(p_cal, X_cal, size))

    def drawn(p, X, rng, sh, hb):
        return (rng.random(len(p)) < policy(p, X, sh, hb)).astype(float)

    test = parts["test"]
    p_test = [buy.predict_proba(s["X"])[:, 1] for s in test]
    raw_picks = [drawn(p, s["X"], np.random.default_rng([seed, i]), raw_shift, 0.0) for i, (p, s) in enumerate(zip(p_test, test))]
    student_picks = [drawn(p, s["X"], np.random.default_rng([seed, i]), shift, habit) for i, (p, s) in enumerate(zip(p_test, test))]
    _, yte, _ = stack(test)
    pte = np.concatenate(p_test)
    result = {"auc": round(float(roc_auc_score(yte, pte)), 3) if 0 < yte.mean() < 1 else None,
              "f1_with_teacher": round(float(f1_score(yte, np.concatenate(raw_picks))), 3),
              "teacher": list_stats(test, [s["y"] for s in test]),
              "student_without_correction": list_stats(test, raw_picks),
              "student": list_stats(test, student_picks),
              "qty_mae": round(float(np.mean(np.abs(np.clip(np.round(qty.predict(stack(test)[0][yte > 0])), 1, 6)
                                                    - stack(test)[2][yte > 0]))), 3) if (yte > 0).any() else None}
    say(f"  test NPCs: AUC {result['auc']}, F1 with the teachers {result['f1_with_teacher']}")
    for who in ("teacher", "student_without_correction", "student"):
        r = result[who]
        say(f"  {who:27} items/list {r['items_per_list']}, usual share {r['usual_share']}, deals {r['deal_share']}, "
            f"ran out {r['ran_out_share']}")
    if size_target:
        say(f"  list size (ESTIMATED): {size} items so baskets come to ~{REAL_LINES} products (Instacart)")
    if habit:
        say(f"  habit correction (ESTIMATED): {habit} -> usual share on test {result['student']['usual_share']} "
            f"(target {habit_target}, real Instacart re-buy rate)")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    export(buy, out / "list_student.onnx", classifier=True)
    export(qty, out / "list_qty.onnx", classifier=False)
    meta = {"version": f"list-student-{date.today().isoformat()}", "features": FEATURES, "bias_shift": shift,
            "habit_correction": {"coef": habit, "target": habit_target, "label": "ESTIMATED",
                                 "target_source": "Instacart reorder rate (PUBLIC_STATISTIC)"} if habit else None,
            "list_size_correction": {"items": size, "label": "ESTIMATED", "real_lines_per_basket": REAL_LINES,
                                     "plan_fill": PLAN_FILL, "target_source": "Instacart products per order "
                                     "(PUBLIC_STATISTIC) / bought-per-planned in a RETINA student run"}
            if size_target else None,
            "teacher_items_per_list": round(teacher_size, 2), "labels": len(sets), "teachers": teachers,
            "prompt_versions": pd.Series([r.get("prompt_version", 1) for r in labels]).value_counts().to_dict(),
            "sets": {k: len(v) for k, v in parts.items()}, "test": result, "real_repeat_share": REAL_REPEAT,
            "policy": "draw each candidate with sigmoid(logit(p) + bias_shift - habit x bought_before); "
                      "quantity = round(qty model), 1-6"}
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
    for o in conv.graph.output:                       # declare the real shapes ([N] not [1]): no run-time warnings
        dims = o.type.tensor_type.shape.dim
        if len(dims) == 1:
            del dims[:]
            dims.add().dim_param = "N"
    onnx.save(conv, path)


def card(m):
    t, s = m["test"]["teacher"], m["test"]["student"]
    r = m["test"].get("student_without_correction") or s
    h = m.get("habit_correction")
    pct = lambda x: "-" if x is None else f"{x:.0%}"
    row = lambda name, v: (f"| {name} | {v['items_per_list']} | {pct(v['usual_share'])} | {pct(v['deal_share'])} | "
                           f"{pct(v['ran_out_share'])} |")
    z = m.get("list_size_correction")
    size_lines = [f"List size correction (ESTIMATED): the teachers wrote {m['teacher_items_per_list']} items per list; "
                  f"real baskets have ~{z['real_lines_per_basket']} products (Instacart) and {z['plan_fill']:.0%} of "
                  f"planned items get bought here, so bias_shift is fitted to {z['items']} planned items "
                  f"(instead of the teachers' size).", ""] if z else []
    habit_lines = [f"Habit correction (ESTIMATED): the teachers put far more usual products on a list than real "
                   f"shoppers re-buy, so at run time each usual product's logit is lowered by {h['coef']} (tuned on "
                   f"all scenarios of non-test NPCs, list size kept) to bring the usual share to ~{h['target']:.0%} "
                   f"({h['target_source']}). Checked on the test NPCs above.", ""] if h else []
    return "\n".join([
        f"# RETINA list student - {m['version']}", "",
        "Plans an NPC's shopping list: from ~30 real candidate products of the RETINA Supercenter (their usual "
        "products, what goes with them, new ideas, deals) it decides what goes on today's list and how many. "
        "Learned from LLM teachers' lists. Every output is SIMULATED behaviour.", "",
        f"- Lists: {m['labels']:,} from {', '.join(f'{k} {v}' for k, v in m['teachers'].items())}; split by NPC "
        f"(train {m['sets']['train']}, validation {m['sets']['val']}, test {m['sets']['test']}).", "",
        "| on test NPCs | items per list | usual products | deals | ran out at home |", "|---|---|---|---|---|",
        row("teachers", t)] + ([row("list student, no correction", r)] if h else []) + [
        row("list student (drawn, used in the store)", s), "",
        f"AUC {m['test']['auc']}, F1 with the teachers {m['test']['f1_with_teacher']}, quantity error (MAE) "
        f"{m['test']['qty_mae']}. Real re-buy share for comparison: ~{m['real_repeat_share']:.0%} (Instacart).", "",
        f"Policy: {m['policy']}; bias_shift {m['bias_shift']} sets the average list size.", ""] + size_lines + habit_lines)
