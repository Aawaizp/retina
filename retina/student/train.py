"""Train the student on the teacher's labels, compare it, export it, write its model card.

Data   one teacher only (Groq openai/gpt-oss-120b): valid labels where the pick was a choosable option
       (SKIPs are counted, not learned). One row per choosable option (retina.student.features).
Split  by NPC (every 5th NPC id, by a hash) -> held-out NPCs the model never saw; a second slice of
       the training NPCs is the validation set (temperature, early stopping, threshold).
Models random (baseline) | engine (v1 pick weights, baseline) | conditional logit | LightGBM ranker.
Tests  on held-out NPCs: top-1 agreement with the teacher, log loss, calibration (ECE), own-price
       elasticity (same conditional-logit yardstick as the run report), repeat share and deal share
       of the picks (teacher vs student vs engine; real Instacart repeat ~60%), routing curve
       (confidence threshold -> share decided by the student and its agreement), teacher
       self-agreement (ceiling) and position bias of the teacher.
Export the best model on VALIDATION log loss -> models/student/student.onnx + student.json +
       MODEL_CARD.md; checks that ONNX gives the same scores as Python.
"""

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np

from retina.mind.report import conditional_logit, elasticity
from retina.student.collect import position_bias, self_agreement
from retina.student.features import FEATURES, matrix
from retina.student.labels import LabelStore, TEACHER

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "student"
REAL_REPEAT = 0.60                    # Instacart reorder rate (PUBLIC_STATISTIC)
THRESHOLDS = [round(t, 2) for t in np.arange(0.30, 0.96, 0.05)]


def split_of(npc):
    """'test' for ~20% of NPCs, 'val' for ~16%, else 'train' - fixed by a hash of the NPC id."""
    h = int(hashlib.md5(str(npc).encode()).hexdigest(), 16) % 100
    return "test" if h < 20 else ("val" if h < 36 else "train")


def examples(rows):
    """Labels -> decision sets {X, y (index of the pick), opts, w (engine weights), npc}. Skips SKIPs."""
    out, skips = [], 0
    for r in rows:
        X, opts = matrix(r["npc_x"], r["route"], r["mission"], r["options"], r["engine_product"])
        if r["choice"] is None or r["choice"] < 0:
            skips += 1
            continue
        pick = [i for i, o in enumerate(opts) if o["product"] == r["choice"]]
        if len(opts) < 2 or len(pick) != 1:
            continue
        w = np.array([max(o["w"], 1e-9) for o in opts])
        out.append({"id": r["id"], "weight": 1.0, "X": X, "y": pick[0], "opts": opts, "w": w / w.sum(), "npc": r["npc"],
                    "engine": int(np.argmax(w)), "split": split_of(r["npc"]), "variation": r.get("variation")})
    return out, skips


def softmax(scores, temperature=1.0):
    z = np.asarray(scores, dtype=float) / temperature
    z = np.exp(z - z.max())
    return z / z.sum()


# ================================================================== models
def weighted_logit(sets, weights, iters=60, ridge=1e-2):
    """Conditional logit (Newton) where each decision counts `weight` times (consensus weighting)."""
    k = sets[0][0].shape[1]
    b = np.zeros(k)
    for _ in range(iters):
        grad, hess = -ridge * b, -ridge * np.eye(k)
        for (X, y), w in zip(sets, weights):
            u = X @ b
            p = np.exp(u - u.max())
            p /= p.sum()
            xbar = p @ X
            grad += w * (y - p) @ X
            hess -= w * ((X * p[:, None]).T @ X - np.outer(xbar, xbar))
        step = np.linalg.solve(hess, grad)
        b -= step
        if np.abs(step).max() < 1e-6:
            break
    return b


def consensus_weights(sets, store, other, weight, disagree=1.0):
    """Give a decision `weight` when a second teacher picked the same option, `disagree` when it picked
    another one; 1.0 when the second teacher did not see it. Returns counts."""
    provider, _, model = other.partition("/")
    theirs = {r["id"]: r["choice"] for r in store.rows(0, provider, model or None)}
    counts = {"agree": 0, "disagree": 0, "not_seen": 0}
    for d in sets:
        if d["id"] not in theirs:
            counts["not_seen"] += 1
        elif theirs[d["id"]] == d["opts"][d["y"]]["product"]:
            d["weight"], counts["agree"] = weight, counts["agree"] + 1
        else:
            d["weight"], counts["disagree"] = disagree, counts["disagree"] + 1
    return counts


class Logit:
    name = "conditional logit"

    def fit(self, train, val):
        X = np.vstack([d["X"] for d in train])
        self.mean, self.std = X.mean(0), X.std(0) + 1e-6
        sets = [((d["X"] - self.mean) / self.std, np.eye(len(d["X"]))[d["y"]]) for d in train]
        self.b = weighted_logit(sets, [d["weight"] for d in train])
        return self

    def scores(self, X):
        return ((X - self.mean) / self.std) @ self.b


class Ranker:
    name = "LightGBM ranker"

    def fit(self, train, val):
        import lightgbm as lgb
        params = dict(objective="lambdarank", n_estimators=400, learning_rate=0.05, num_leaves=15,
                      min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                      reg_lambda=1.0, random_state=7, verbose=-1)
        X = np.vstack([d["X"] for d in train])
        y = np.concatenate([np.eye(len(d["X"]), dtype=int)[d["y"]] for d in train])
        Xv = np.vstack([d["X"] for d in val])
        yv = np.concatenate([np.eye(len(d["X"]), dtype=int)[d["y"]] for d in val])
        weights = np.concatenate([np.full(len(d["X"]), d["weight"]) for d in train])    # per row, same in a decision
        self.model = lgb.LGBMRanker(**params).fit(
            X, y, group=[len(d["X"]) for d in train], sample_weight=weights, eval_X=(Xv,), eval_y=(yv,), eval_group=[[len(d["X"]) for d in val]],
            eval_at=[1], callbacks=[lgb.early_stopping(40, verbose=False)])
        return self

    def scores(self, X):
        return self.model.predict(X, raw_score=True)


def fit_temperature(model, val):
    """The T that makes softmax(scores / T) best calibrated on validation (log loss)."""
    best = (np.inf, 1.0)
    for t in np.exp(np.linspace(np.log(0.2), np.log(5), 41)):
        loss = -np.mean([np.log(max(softmax(model.scores(d["X"]), t)[d["y"]], 1e-12)) for d in val])
        best = min(best, (loss, t))
    return float(best[1])


# ================================================================== metrics
def probs_of(kind, d, model=None, t=1.0):
    if kind == "random":
        return np.full(len(d["X"]), 1 / len(d["X"]))
    if kind == "engine":
        return d["w"]
    return softmax(model.scores(d["X"]), t)


def evaluate(sets, probs):
    """Agreement, log loss, calibration, picks' repeat/deal share, elasticity - for one chooser."""
    p = [probs(d) for d in sets]
    top = [int(np.argmax(x)) for x in p]
    conf = np.array([x.max() for x in p])
    hit = np.array([t == d["y"] for t, d in zip(top, sets)], dtype=float)
    bins = np.minimum((conf * 10).astype(int), 9)
    ece = sum(abs(hit[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
    el_sets = []
    for d, x in zip(sets, p):
        change = d["X"][:, FEATURES.index("price_change")]
        if np.std(change) < 0.01:
            continue
        cols = [FEATURES.index(c) for c in ("price_change", "price_level", "usual", "popularity_log")]
        el_sets.append((d["X"][:, cols].astype(float), x))
    el = None
    if len(el_sets) >= 30:
        b = conditional_logit(el_sets)
        el = round(elasticity(el_sets, b), 2)
    return {"agreement": round(float(hit.mean()), 3),
            "log_loss": round(float(-np.mean([np.log(max(x[d["y"]], 1e-12)) for x, d in zip(p, sets)])), 3),
            "ece": round(float(ece), 3),
            "repeat_share": round(float(np.mean([d["opts"][t]["times"] > 0 for t, d in zip(top, sets)])), 3),
            "deal_share": round(float(np.mean([d["opts"][t]["deal"] for t, d in zip(top, sets)])), 3),
            "elasticity": el, "elasticity_sets": len(el_sets)}


def teacher_side(sets):
    """The teacher's own picks on the same sets (repeat/deal share, elasticity)."""
    m = evaluate(sets, lambda d: np.eye(len(d["X"]))[d["y"]])
    return {k: m[k] for k in ("repeat_share", "deal_share", "elasticity", "elasticity_sets")}


def routing_curve(sets, model, t):
    out = []
    for th in THRESHOLDS:
        covered = [d for d in sets if softmax(model.scores(d["X"]), t).max() >= th]
        agree = np.mean([np.argmax(model.scores(d["X"])) == d["y"] for d in covered]) if covered else None
        out.append({"threshold": th, "coverage": round(len(covered) / len(sets), 3),
                    "agreement": None if agree is None else round(float(agree), 3)})
    return out


def pick_threshold(curve, target):
    """Lowest threshold whose agreement on validation reaches the target (else the highest one)."""
    for row in curve:
        if row["agreement"] is not None and row["agreement"] >= target and row["coverage"] > 0:
            return row
    return curve[-1]


# ================================================================== export
def export_onnx(model, path):
    """Logit: a tiny hand-made graph ((x - mean) / std) @ b. Ranker: onnxmltools (skl2onnx cannot)."""
    import onnx
    from onnx import TensorProto, helper
    n = len(FEATURES)
    if isinstance(model, Logit):
        graph = helper.make_graph(
            [helper.make_node("Sub", ["x", "mean"], ["c"]), helper.make_node("Div", ["c", "std"], ["z"]),
             helper.make_node("MatMul", ["z", "b"], ["scores"])], "student_logit",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, n])],
            [helper.make_tensor_value_info("scores", TensorProto.FLOAT, [None, 1])],
            [helper.make_tensor("mean", TensorProto.FLOAT, [n], model.mean.astype(np.float32)),
             helper.make_tensor("std", TensorProto.FLOAT, [n], model.std.astype(np.float32)),
             helper.make_tensor("b", TensorProto.FLOAT, [n, 1], model.b.astype(np.float32))])
        onx = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])
        onx.ir_version = 8
    else:
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType
        onx = onnxmltools.convert_lightgbm(model.model.booster_, initial_types=[("x", FloatTensorType([None, n]))],
                                           target_opset=15)
    onnx.save(onx, path)


def onnx_matches(path, model, sets):
    import onnxruntime as rt
    sess = rt.InferenceSession(str(path))
    X = np.vstack([d["X"] for d in sets]).astype(np.float32)
    got = np.asarray(sess.run(None, {sess.get_inputs()[0].name: X})[0]).ravel()
    return float(np.abs(got - np.asarray(model.scores(X)).ravel()).max())


# ================================================================== main
def train(store=None, provider=TEACHER[0], model=TEACHER[1], out=MODEL_DIR, say=print, min_sets=100,
          consensus=None, consensus_weight=2.0):
    """consensus = 'name/model' of a second teacher: training decisions where it picked the same option
    count consensus_weight times (validation and test are never weighted)."""
    store = store or LabelStore()
    rows = store.rows(0, provider, model)
    sets, skips = examples(rows)
    parts = {k: [d for d in sets if d["split"] == k] for k in ("train", "val", "test")}
    weighting = None
    if consensus:
        counts = consensus_weights(parts["train"], store, consensus, consensus_weight)
        weighting = {"with": consensus, "weight": consensus_weight, **counts}
        say(f"Consensus weighting with {consensus}: {counts['agree']} training decisions agree (x{consensus_weight:g}), "
            f"{counts['disagree']} disagree, {counts['not_seen']} not seen by it")
    say(f"Labels from {provider}/{model}: {len(rows):,} valid ({skips} SKIPs not learned) -> {len(sets):,} choice sets | "
        + " | ".join(f"{k} {len(v):,} sets / {len({d['npc'] for d in v})} NPCs" for k, v in parts.items()))
    if len(parts["train"]) < min_sets or len(parts["val"]) < 20 or len(parts["test"]) < 20:
        raise SystemExit(f"Too few labels to train (need about {min_sets}+ training sets). "
                         f"Run: retina collect --decisions 3000")
    results, fitted = {}, {}
    for kind in ("random", "engine"):
        results[kind] = evaluate(parts["test"], lambda d, k=kind: probs_of(k, d))
    for cls in (Logit, Ranker):
        m = cls().fit(parts["train"], parts["val"])
        t = fit_temperature(m, parts["val"])
        val_loss = evaluate(parts["val"], lambda d: probs_of("model", d, m, t))["log_loss"]
        fitted[m.name] = (m, t, val_loss)
        results[m.name] = evaluate(parts["test"], lambda d: probs_of("model", d, m, t)) | {"temperature": round(t, 3),
                                                                                           "val_log_loss": val_loss}
    best_name = min(fitted, key=lambda k: fitted[k][2])
    best, t, _ = fitted[best_name]
    ceiling, n_rep = self_agreement(store, provider, model)
    target = 0.95 * ceiling if ceiling else 0.80
    val_curve = routing_curve(parts["val"], best, t)
    chosen = pick_threshold(val_curve, target)
    test_curve = routing_curve(parts["test"], best, t)
    at = next(r for r in test_curve if r["threshold"] == chosen["threshold"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    export_onnx(best, out / "student.onnx")
    diff = onnx_matches(out / "student.onnx", best, parts["test"])
    meta = {"version": f"student-{date.today().isoformat()}", "model": best_name, "features": FEATURES,
            "temperature": t, "threshold": chosen["threshold"], "coverage_at_threshold": at["coverage"],
            "agreement_at_threshold": at["agreement"], "teacher": {"provider": provider, "model": model},
            "labels": len(rows), "skips": skips, "sets": {k: len(v) for k, v in parts.items()},
            "test": results, "teacher_on_test": teacher_side(parts["test"]),
            "self_agreement": {"rate": ceiling, "items": n_rep}, "position_bias": position_bias(rows),
            "routing_curve_test": test_curve, "onnx_max_diff": diff, "real_repeat_share": REAL_REPEAT,
            "output": "scores per choosable option; probabilities = softmax(scores / temperature) within a decision",
            "consensus_weighting": weighting}
    (out / "student.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    card = model_card(meta)
    (out / "MODEL_CARD.md").write_text(card, encoding="utf-8")
    say(card)
    return meta


def model_card(m):
    r, te = m["test"], m["teacher_on_test"]
    pct = lambda x: "-" if x is None else f"{x:.0%}"
    num = lambda x: "-" if x is None else f"{x:+.2f}"
    rows = ["| chooser | agreement with teacher | log loss | calibration error | repeat share | deal share | own-price elasticity |",
            "|---|---|---|---|---|---|---|",
            f"| teacher ({m['teacher']['model']}) | - | - | - | {pct(te['repeat_share'])} | {pct(te['deal_share'])} | {num(te['elasticity'])} |"]
    for k in ("random", "engine", "conditional logit", "LightGBM ranker"):
        x = r[k]
        rows.append(f"| {k}{' **(chosen)**' if k == m['model'] else ''} | {pct(x['agreement'])} | {x['log_loss']:.3f} | "
                    f"{x['ece']:.3f} | {pct(x['repeat_share'])} | {pct(x['deal_share'])} | {num(x['elasticity'])} |")
    sa = m["self_agreement"]
    bias = " ".join(f"{k} {v['picked']:.0%} (expected {v['expected']:.0%})" for k, v in m["position_bias"].items())
    curve = " | ".join(f"{c['threshold']:.2f}: {pct(c['coverage'])} decided, {pct(c['agreement'])} agree"
                       for c in m["routing_curve_test"][::2])
    return "\n".join([
        f"# RETINA student model - {m['version']}", "",
        "A small model that copies the LLM teacher's shelf choices for thinking NPCs, so most of them need no LLM call. "
        "It ranks the 2-5 real products offered for one shopping-list item. Every output is SIMULATED behaviour.", "",
        "## Data",
        f"- Teacher: **{m['teacher']['provider']} {m['teacher']['model']}** only (one mind; other providers' answers are not used).",
        f"- {m['labels']:,} labels ({m['skips']} SKIPs counted, not learned): situations from `retina collect` "
        "(store states varied on purpose: price up, deal, sold out, remembered price) and teacher answers from mix runs.",
        f"- Split by NPC: train {m['sets']['train']:,} / validation {m['sets']['val']:,} / test {m['sets']['test']:,} "
        "decision sets; test NPCs were never seen in training.",
        (f"- Consensus weighting: training decisions where {m['consensus_weighting']['with']} picked the same option "
         f"count x{m['consensus_weighting']['weight']:g} ({m['consensus_weighting']['agree']} agree, "
         f"{m['consensus_weighting']['disagree']} disagree, {m['consensus_weighting']['not_seen']} not seen); "
         "validation and test are not weighted." if m.get("consensus_weighting") else "- No consensus weighting."),
        f"- Features ({len(m['features'])}): NPC traits, why the item is on the list, trip type, price change, price level, "
        "deal, bought before, usual, engine pick, best seller, popularity, price vs last paid. NOT the option's position.", "",
        "## Results on held-out NPCs", *rows, "",
        f"Teacher self-agreement (the ceiling: the same situation asked twice): **{pct(sa['rate'])}** on {sa['items']} items.",
        f"Real repeat share for comparison: ~{m['real_repeat_share']:.0%} (Instacart reorder rate, PUBLIC_STATISTIC).",
        "Own-price elasticity: conditional logit on price change within the options (research on grocery choice: about -2.5).", "",
        "## Routing", f"Confidence threshold **{m['threshold']:.2f}** (the lowest one that reaches 95% of the teacher's "
        f"self-agreement on validation): the student decides **{pct(m['coverage_at_threshold'])}** of teacher-worthy choices, "
        f"agreeing {pct(m['agreement_at_threshold'])} with the teacher; the rest go to the teacher (and become new labels).",
        f"Curve (test): {curve}", "",
        "## Teacher position bias", f"Pick share by the letter shown, after shuffling: {bias}. "
        "The student cannot have this bias (position is not a feature).", "",
        "## Files and use",
        f"- `student.onnx` ({m['model']}): input `x` [options x {len(m['features'])}] float32 in the order of "
        "`student.json` features; output = one score per option; probabilities = softmax(scores / temperature) "
        f"within a decision (temperature {m['temperature']:.3f}). ONNX vs Python max difference: {m['onnx_max_diff']:.1e}.",
        "- Used by `retina --mind mix` (confident -> student, unsure -> teacher); `--student only` for no-LLM runs.", "",
        "## Limits",
        "- It copies ONE LLM's habits, including its biases (LLM shoppers are more deal-driven than real ones).",
        "- US products and prices; NPCs and situations are SIMULATED; agreement is with the teacher, not with real people.",
        "- Small data: a few thousand decisions. Retrain as labels grow (`retina train`).", ""])
