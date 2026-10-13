"""Run report for thinking runs: the thinking panel (LLM or student) next to the engine crowd.

  1. Shopping (receipts.csv): repeat share (product bought before), deal share, items and $ per
     trip - panel NPCs vs everyone else in the same store, same days.
  2. The mind (decisions log): how many choices went to autopilot / the LLM / fell back, how
     often the LLM agreed with the engine, cache hits, the reasons it gave.
  3. Price response on the SAME option sets: a conditional logit fitted to the LLM's picks and to
     the engine's own pick weights for those options. Features:
       own price change  log(today's price / that product's regular price)  -> the elasticity:
                         slope x (1 - chosen share); research on grocery choice: about -2.5
       price level       log(regular price), centred within the options  -> cheap vs dear brands
       usual             the NPC bought it before
       popularity        log(the product's base share of its aisle) - a control: without it the
                         huge spread in popularity hides the price effect
     Below MIN_DECISIONS option sets with a price change in them it says "too few".
Written to report.txt and report.json in the run folder. Every number is from SIMULATED data.
"""

import json

import numpy as np
import pandas as pd

MIN_DECISIONS = 30            # fewer LLM decisions with a price spread: no elasticity (small samples lie)
REAL = {"repeat_share": "~60%, Instacart reorder rate", "elasticity": "about -2.5"}


def conditional_logit(sets, iters=50, ridge=1e-3):
    """sets = [(X options x features, y weights summing to 1)] -> coefficients (Newton's method)."""
    k = sets[0][0].shape[1]
    b = np.zeros(k)
    for _ in range(iters):
        grad, hess = -ridge * b, -ridge * np.eye(k)
        for X, y in sets:
            u = X @ b
            p = np.exp(u - u.max())
            p /= p.sum()
            xbar = p @ X
            grad += (y - p) @ X
            hess -= (X * p[:, None]).T @ X - np.outer(xbar, xbar)
        step = np.linalg.solve(hess, grad)
        b -= step
        if np.abs(step).max() < 1e-6:
            break
    return b


def elasticity(sets, b):
    """Average own-price choice elasticity of what was chosen: slope x (1 - its probability)."""
    out, total = 0.0, 0.0
    for X, y in sets:
        u = X @ b
        p = np.exp(u - u.max())
        p /= p.sum()
        out += float(y @ (b[0] * (1 - p)))
        total += float(y.sum())
    return out / total


def option_sets(decisions):
    """Thinking decisions (LLM or student) -> (features, chosen label, engine weights) per decision with a real price spread."""
    llm_sets, engine_sets = [], []
    for d in decisions:
        opts = json.loads(d["options"]) if isinstance(d["options"], str) else d["options"]
        opts = [o for o in opts if o.get("in_stock", True)]     # a sold-out option could not be picked:
        if d["mind"] not in ("llm", "student") or d["product"] is None or d["product"] < 0 or len(opts) < 2:
            continue
        price = np.maximum([o["price"] for o in opts], 0.01)
        regular = np.maximum([o["regular"] for o in opts], 0.01)
        change = np.log(price / regular)                         # 0 = normal price, below 0 = on deal
        if change.std() < 0.01:
            continue                                             # no price change among these options
        level = np.log(regular)
        pop = np.log(np.maximum([o.get("pop", 1.0) for o in opts], 1e-9))
        X = np.column_stack([change, level - level.mean(), [o["times"] > 0 for o in opts],
                             pop - pop.mean()]).astype(float)
        chosen = np.array([o["product"] == d["product"] for o in opts], dtype=float)
        w = np.array([o["w"] for o in opts], dtype=float)
        if chosen.sum() != 1 or w.sum() <= 0:
            continue
        llm_sets.append((X, chosen))
        engine_sets.append((X, w / w.sum()))
    return llm_sets, engine_sets


def shopping_table(receipts, panel, mode):
    r = receipts.assign(group=np.where(receipts["shopper_id"].isin(panel), "thinking panel", "engine crowd"))
    rows = {}
    for name, g in r.groupby("group"):
        trips = g["trip_id"].nunique()
        rows[name] = {"trips": trips, "lines": len(g),
                      "repeat_share": round(float(g["is_repeat"].mean()), 3),
                      "deal_share": round(float(g["on_promo"].mean()), 3),
                      "items_per_trip": round(float(g["quantity"].sum() / max(trips, 1)), 1),
                      "spend_per_trip": round(float(g["line_total"].sum() / max(trips, 1)), 2)}
    return rows


def build_report(mind, run_dir):
    path = run_dir / "receipts.csv"
    if not path.exists():
        return "No days simulated yet in this run - nothing to report."
    receipts = pd.read_csv(path, usecols=["trip_id", "shopper_id", "quantity", "line_total", "on_promo", "is_repeat"])
    decisions = mind.memory.decisions()
    shop = shopping_table(receipts, mind.panel if mind.mode == "mix" else set(receipts["shopper_id"]), mind.mode)
    d = pd.DataFrame(decisions)
    out = {"mode": mind.mode, "panel": len(mind.panel), "shopping": shop, "real": REAL}
    lines = [f"RETINA thinking run report ({mind.mode}; SIMULATED data)", ""]
    lines.append(f"{'':<16}{'trips':>7}{'lines':>8}{'repeat':>8}{'deals':>7}{'items/trip':>11}{'$/trip':>8}")
    for name, s in shop.items():
        lines.append(f"{name:<16}{s['trips']:>7,}{s['lines']:>8,}{s['repeat_share']:>8.0%}{s['deal_share']:>7.0%}"
                     f"{s['items_per_trip']:>11.1f}{s['spend_per_trip']:>8.2f}")
    lines.append(f"  repeat = product bought before (real {REAL['repeat_share']}); deals = lines on promotion")
    if mind.mode == "student":
        lines.append("  (--mind student: every NPC decides with the student, so there is no engine crowd here - "
                     "compare with a fast run)")
    if mind.mode == "llm":
        lines.append("  (--mind llm: every NPC thinks, so there is no engine crowd here - compare with a fast run)")
    if len(d):
        kinds = d["mind"].value_counts()
        llm_d = d[d["mind"] == "llm"]
        asked = d[d["mind"].isin(["llm", "llm_fallback"])]
        mind_stats = {"choices": len(d), "autopilot": int(kinds.get("autopilot", 0)), "llm": int(kinds.get("llm", 0)),
                      "student": int(kinds.get("student", 0)),
                      "fallback": int(kinds.get("llm_fallback", 0)),
                      "llm_agrees_with_engine": round(float((llm_d["product"] == llm_d["engine_product"]).mean()), 3)
                      if len(llm_d) else None,
                      "llm_skips": int((llm_d["product"] < 0).sum()) if len(llm_d) else 0,
                      "from_cache": int(asked["cached"].sum()) if len(asked) else 0,
                      "triggers": d["trigger"].value_counts().to_dict(),
                      "reasons": llm_d["why_type"].value_counts().to_dict() if len(llm_d) else {},
                      "providers": asked["provider"].fillna("none").value_counts().to_dict() if len(asked) else {}}
        out["mind"] = mind_stats
        lines += ["", f"Mind: {len(d):,} choices by thinking NPCs - {mind_stats['autopilot']:,} autopilot, "
                      + (f"{mind_stats['student']:,} student, " if mind_stats["student"] else "")
                      + f"{mind_stats['llm']:,} LLM ({mind_stats['from_cache']:,} from cache), "
                      f"{mind_stats['fallback']:,} fell back to the engine"]
        if len(llm_d):
            lines.append(f"  LLM picked the engine's product {mind_stats['llm_agrees_with_engine']:.0%} of the time; "
                         f"skipped {mind_stats['llm_skips']} items")
            lines.append("  why sent to the LLM: " + ", ".join(f"{k} {v}" for k, v in mind_stats["triggers"].items()
                                                               if k != "routine"))
            lines.append("  LLM reasons: " + ", ".join(f"{k} {v}" for k, v in mind_stats["reasons"].items()))
            lines.append("  answered by: " + ", ".join(f"{k} {v}" for k, v in mind_stats["providers"].items()))
        llm_sets, engine_sets = option_sets(decisions)
        if len(llm_sets) >= MIN_DECISIONS:
            b_llm, b_eng = conditional_logit(llm_sets), conditional_logit(engine_sets)
            e_llm, e_eng = elasticity(llm_sets, b_llm), elasticity(engine_sets, b_eng)
            out["price"] = {"decisions": len(llm_sets), "llm_elasticity": round(e_llm, 2),
                            "engine_elasticity": round(e_eng, 2),
                            "llm_price_level": round(float(b_llm[1]), 2), "engine_price_level": round(float(b_eng[1]), 2),
                            "llm_usual_effect": round(float(b_llm[2]), 2), "engine_usual_effect": round(float(b_eng[2]), 2)}
            p = out["price"]
            lines += ["", f"Price response on the same {p['decisions']} option sets with a price change "
                          f"(conditional logit):",
                      f"  own-price elasticity  mind {p['llm_elasticity']:+.2f} | engine {p['engine_elasticity']:+.2f}"
                      f"   (research {REAL['elasticity']})",
                      f"  dear-brand slope      mind {p['llm_price_level']:+.2f} | engine {p['engine_price_level']:+.2f}"
                      f"   (below 0 = prefers cheaper products)",
                      f"  pull of the usual     mind {p['llm_usual_effect']:+.2f} | engine {p['engine_usual_effect']:+.2f}"
                      f"   (the engine's shelf weights do not know personal habits)"]
        else:
            out["price"] = {"decisions": len(llm_sets), "note": "too few"}
            lines += ["", f"Price response: too few thinking (LLM or student) decisions with a price change among the options "
                          f"({len(llm_sets)}; needs {MIN_DECISIONS}) - run more days."]
    text = "\n".join(lines)
    (run_dir / "report.txt").write_text(text + "\n", encoding="utf-8")
    (run_dir / "report.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return text
