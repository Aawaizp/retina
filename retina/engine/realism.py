"""RETINA realism score: how close are simulated baskets to REAL ones the simulator never saw?

Real baskets: Instacart's "train" orders (131,209 baskets, each shopper's last order). RETINA
learned habits and product popularity only from the "prior" orders, so these are held out.
Both sides are compared at the AISLE level (soft drinks, milk, fresh fruits ...): RETINA's
products and Instacart's products have different names, but both use Instacart's 134 aisles.
Real items in aisles our store does not sell (babies, pets ...) are left out.

Tests
  1. Distributions: basket size, department and aisle shares, repeat share, hour of day.
  2. Bought together: do the same aisle pairs go together (lift correlation)?
  3. Can a computer tell them apart? A classifier sees one basket at a time (which aisles,
     how many items) and guesses real or simulated. AUC 0.5 = can't tell, 1.0 = always can.
  4. Train on simulated, test on real: models that predict "does this basket contain milk?"
     from the rest of the basket. Trained on RETINA data, how well do they work on real
     baskets, compared with the same models trained on real data?

The held-out baskets are split in two by order number: the TUNING half may be used to adjust
settings; the FINAL half is only for the reported score (python -m retina.engine.realism --final).

Run (after retina.engine.retina_mart):   python -m retina.engine.realism [--final]
Saved: data/simulated/<store>/realism_report.json (or realism_report_final.json)
"""

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, pearsonr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split

from .config import *

INSTACART_DIR = ROOT / "data" / "raw" / "instacart"
SAMPLE = 20_000              # baskets from each side
TARGET_AISLES = 15           # "does the basket contain ...?" tasks for test 4
MIN_PAIR_BASKETS = 200       # aisle pairs seen together in at least this many real baskets
SEED_SPLIT = 7
WARMUP_DAYS = 120            # simulated trips before this day are left out: NPCs start the year with
                             # no shopping history, while real Instacart shoppers have months of it


# ================================================================ baskets
def simulated_baskets():
    """One row per simulated trip line: trip, aisle, department, hour, repeat."""
    log = pd.read_parquet(OUTPUT_DIR / "sales_log.parquet",
                          columns=["trip_id", "day", "product_id", "hour", "is_repeat"])
    log = log[log["day"] >= WARMUP_DAYS]
    cat = pd.read_parquet(OUTPUT_DIR / "store_catalog.parquet", columns=["product_id", "aisle", "department"])
    lines = log.merge(cat, on="product_id", how="left")
    return lines.rename(columns={"trip_id": "basket"})[["basket", "aisle", "department", "hour", "is_repeat"]]


def real_baskets(store_aisles, final=False):
    """Held-out Instacart baskets, limited to aisles our store sells. Even order numbers = tuning
    half, odd = final half."""
    items = pd.read_csv(INSTACART_DIR / "order_products__train.csv",
                        usecols=["order_id", "product_id", "reordered"])
    products = pd.read_csv(INSTACART_DIR / "products.csv", usecols=["product_id", "aisle_id", "department_id"])
    products = (products.merge(pd.read_csv(INSTACART_DIR / "aisles.csv"), on="aisle_id")
                .merge(pd.read_csv(INSTACART_DIR / "departments.csv"), on="department_id"))
    orders = pd.read_csv(INSTACART_DIR / "orders.csv", usecols=["order_id", "eval_set", "order_hour_of_day"])
    orders = orders[(orders["eval_set"] == "train") & (orders["order_id"] % 2 == (1 if final else 0))]
    lines = items.merge(products[["product_id", "aisle", "department"]], on="product_id").merge(
        orders[["order_id", "order_hour_of_day"]], on="order_id")
    n_before = lines["order_id"].nunique()
    lines = lines[lines["aisle"].isin(store_aisles)]
    lines = lines.rename(columns={"order_id": "basket", "order_hour_of_day": "hour", "reordered": "is_repeat"})
    return lines[["basket", "aisle", "department", "hour", "is_repeat"]], n_before


def sample_baskets(lines, n, seed):
    ids = lines["basket"].drop_duplicates()
    keep = ids.sample(min(n, len(ids)), random_state=seed)
    return lines[lines["basket"].isin(keep)]


def aisle_matrix(lines, aisles):
    """Baskets x aisles: number of lines from each aisle."""
    m = pd.crosstab(lines["basket"], lines["aisle"]).reindex(columns=aisles, fill_value=0)
    return m


# ================================================================ tests
def tvd(p, q):
    """Total variation distance: share of the mix that would have to move (0 = same, 1 = no overlap)."""
    p, q = p.align(q, fill_value=0)
    return 0.5 * (p / p.sum() - q / q.sum()).abs().sum()


def distributions(sim, real):
    out = {}
    s_size, r_size = sim.groupby("basket").size(), real.groupby("basket").size()
    out["basket_size"] = {"simulated_mean": round(float(s_size.mean()), 2), "real_mean": round(float(r_size.mean()), 2),
                          "simulated_median": float(s_size.median()), "real_median": float(r_size.median()),
                          "ks_distance": round(float(ks_2samp(s_size, r_size).statistic), 3)}
    out["department_mix_tvd"] = round(float(tvd(sim["department"].value_counts(), real["department"].value_counts())), 3)
    out["aisle_mix_tvd"] = round(float(tvd(sim["aisle"].value_counts(), real["aisle"].value_counts())), 3)
    out["repeat_share"] = {"simulated": round(float(sim["is_repeat"].mean()), 3), "real": round(float(real["is_repeat"].mean()), 3)}
    sh = sim.drop_duplicates("basket")["hour"].value_counts()
    rh = real.drop_duplicates("basket")["hour"].value_counts()
    out["hour_mix_tvd"] = round(float(tvd(sh, rh)), 3)
    # How often baskets contain each department (penetration)
    sp = sim.drop_duplicates(["basket", "department"])["department"].value_counts() / sim["basket"].nunique()
    rp = real.drop_duplicates(["basket", "department"])["department"].value_counts() / real["basket"].nunique()
    sp, rp = sp.align(rp, fill_value=0)
    out["department_penetration_corr"] = round(float(pearsonr(sp, rp)[0]), 3)
    gap = (sp - rp).sort_values()
    out["department_penetration_gaps"] = {k: round(float(v), 3) for k, v in pd.concat([gap.head(3), gap.tail(3)]).items()}
    return out


def pair_lifts(m):
    """Lift of every aisle pair: P(both) / (P(a) P(b))."""
    x = (m.to_numpy() > 0).astype(np.float64)
    n = len(x)
    p = x.mean(axis=0)
    both = (x.T @ x) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = both / np.outer(p, p)
    return lift, both * n


def bought_together(m_sim, m_real):
    ls, _ = pair_lifts(m_sim)
    lr, nr = pair_lifts(m_real)
    iu = np.triu_indices(m_real.shape[1], 1)
    ok = (nr[iu] >= MIN_PAIR_BASKETS) & np.isfinite(ls[iu]) & (ls[iu] > 0) & np.isfinite(lr[iu]) & (lr[iu] > 0)
    a, b = np.log(ls[iu][ok]), np.log(lr[iu][ok])
    names = np.array(m_real.columns)
    pairs = list(zip(names[iu[0]][ok], names[iu[1]][ok], np.exp(a), np.exp(b)))
    strongest = sorted(pairs, key=lambda t: -t[3])[:5]
    return {"pairs_compared": int(ok.sum()), "log_lift_corr": round(float(pearsonr(a, b)[0]), 3),
            "strongest_real_pairs": [{"aisles": f"{x} + {y}", "simulated_lift": round(float(s), 2), "real_lift": round(float(r), 2)}
                                     for x, y, s, r in strongest]}


def features(m):
    x = m.to_numpy().astype(np.float32)
    size = x.sum(axis=1, keepdims=True)
    return np.hstack([x / np.maximum(size, 1), np.log1p(size)])   # aisle mix + basket size


def tell_apart(m_sim, m_real):
    """Classifier two-sample test: can a model tell a simulated basket from a real one?"""
    X = np.vstack([features(m_sim), features(m_real)])
    y = np.r_[np.ones(len(m_sim)), np.zeros(len(m_real))]
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=SEED_SPLIT)
    auc = cross_val_score(model, X, y, cv=3, scoring="roc_auc").mean()
    # What gives simulated baskets away: aisles whose penetration differs most
    ps = (m_sim > 0).mean(); pr = (m_real > 0).mean()
    gap = (ps - pr).sort_values()
    return {"auc": round(float(auc), 3),
            "too_rare_in_simulation": {k: round(float(v), 3) for k, v in gap.head(5).items()},
            "too_common_in_simulation": {k: round(float(v), 3) for k, v in gap.tail(5)[::-1].items()}}


def sim_to_real(m_sim, m_real):
    """Train on simulated, test on real (TSTR) vs train on real, test on real (TRTR)."""
    targets = (m_real > 0).mean().nlargest(TARGET_AISLES).index
    real_train, real_test = train_test_split(m_real, test_size=0.5, random_state=SEED_SPLIT)
    rows = []
    for t in targets:
        def xy(m):
            others = m.drop(columns=[t])
            x = (others.to_numpy() > 0).astype(np.float32)
            x = np.hstack([x, np.log1p(others.to_numpy().sum(axis=1, keepdims=True))])
            return x, (m[t].to_numpy() > 0).astype(int)
        xs, ys = xy(m_sim)
        xr, yr = xy(real_train)
        xt, yt = xy(real_test)
        if ys.min() == ys.max():
            continue
        tstr = roc_auc_score(yt, LogisticRegression(max_iter=1000).fit(xs, ys).predict_proba(xt)[:, 1])
        trtr = roc_auc_score(yt, LogisticRegression(max_iter=1000).fit(xr, yr).predict_proba(xt)[:, 1])
        rows.append((t, tstr, trtr))
    tstr = np.mean([r[1] for r in rows]); trtr = np.mean([r[2] for r in rows])
    return {"tasks": len(rows), "train_simulated_test_real_auc": round(float(tstr), 3),
            "train_real_test_real_auc": round(float(trtr), 3), "ratio": round(float(tstr / trtr), 3),
            "per_task": [{"contains": t, "tstr": round(float(a), 3), "trtr": round(float(b), 3)} for t, a, b in rows]}


def grade(value, bands):
    for limit, label in bands:
        if value <= limit:
            return label
    return bands[-1][1]


# ================================================================ main
def main(final=None):
    final = ("--final" in sys.argv) if final is None else final
    half = "FINAL half (never used for tuning)" if final else "TUNING half"
    print(f"RETINA realism score - simulated baskets vs real Instacart baskets the simulator never saw [{half}]")
    sim_all = simulated_baskets()
    store_aisles = set(sim_all["aisle"].dropna())
    real_all, n_real = real_baskets(store_aisles, final)
    print(f"  real: {real_all['basket'].nunique():,} held-out baskets (of {n_real:,}; items in aisles "
          f"we don't sell left out) | simulated: {sim_all['basket'].nunique():,} trips after day {WARMUP_DAYS}")
    sim = sample_baskets(sim_all, SAMPLE, 1)
    real = sample_baskets(real_all, SAMPLE, 2)
    aisles = sorted(store_aisles & set(real_all["aisle"]))
    m_sim, m_real = aisle_matrix(sim, aisles), aisle_matrix(real, aisles)
    print(f"  comparing {len(m_sim):,} vs {len(m_real):,} baskets over {len(aisles)} aisles\n")

    d = distributions(sim, real)
    b = d["basket_size"]
    print("1. Distributions")
    print(f"   basket size (lines): simulated mean {b['simulated_mean']} / median {b['simulated_median']:.0f} | "
          f"real mean {b['real_mean']} / median {b['real_median']:.0f} | KS distance {b['ks_distance']}")
    print(f"   department mix difference {d['department_mix_tvd']:.1%} | aisle mix difference {d['aisle_mix_tvd']:.1%} "
          f"| hour mix difference {d['hour_mix_tvd']:.1%}   (share of the mix that differs)")
    print(f"   repeat items: simulated {d['repeat_share']['simulated']:.0%} | real {d['repeat_share']['real']:.0%}")
    print(f"   baskets containing each department: correlation {d['department_penetration_corr']} | "
          f"biggest gaps: " + ", ".join(f"{k} {v:+.0%}" for k, v in d["department_penetration_gaps"].items()))

    t = bought_together(m_sim, m_real)
    print(f"\n2. Bought together: {t['pairs_compared']} aisle pairs | lift correlation {t['log_lift_corr']}")
    for p in t["strongest_real_pairs"][:3]:
        print(f"   {p['aisles']}: real lift {p['real_lift']} | simulated {p['simulated_lift']}")

    c = tell_apart(m_sim, m_real)
    print(f"\n3. Can a computer tell them apart?  AUC {c['auc']}  (0.5 = no, 1.0 = always)")
    print("   too rare in simulation:   " + ", ".join(f"{k} {v:+.0%}" for k, v in c["too_rare_in_simulation"].items()))
    print("   too common in simulation: " + ", ".join(f"{k} {v:+.0%}" for k, v in c["too_common_in_simulation"].items()))

    s = sim_to_real(m_sim, m_real)
    print(f"\n4. Train on simulated, test on real ({s['tasks']} tasks: 'does the basket contain <aisle>?')")
    print(f"   trained on RETINA: AUC {s['train_simulated_test_real_auc']} | trained on real: "
          f"AUC {s['train_real_test_real_auc']} | RETINA reaches {s['ratio']:.0%} of real-data quality")

    verdict = {
        "tell_apart": grade(c["auc"], [(0.6, "excellent"), (0.7, "good"), (0.8, "fair"), (1.0, "weak")]),
        "sim_to_real": grade(-s["ratio"], [(-0.95, "excellent"), (-0.9, "good"), (-0.8, "fair"), (0.0, "weak")]),
        "bought_together": grade(-t["log_lift_corr"], [(-0.8, "excellent"), (-0.6, "good"), (-0.4, "fair"), (1.0, "weak")]),
    }
    print(f"\nREALISM SCORE: tell-apart {verdict['tell_apart']} | sim-to-real {verdict['sim_to_real']} | "
          f"bought-together {verdict['bought_together']}   (bands are RETINA's own rule of thumb)")
    print("Note: the simulator was calibrated on Instacart prior orders; these real baskets were held out.")

    report = {"what": "RETINA simulated baskets vs held-out real Instacart baskets (aisle level)", "half": half,
              "store_profile": PROFILE["id"], "seed": SEED, "sample_per_side": SAMPLE,
              "aisles": len(aisles), "distributions": d, "bought_together": t, "tell_apart": c,
              "sim_to_real": s, "verdict": verdict,
              "data_labels": {"real": "PUBLIC_STATISTIC (Instacart Market Basket, non-commercial)",
                              "simulated": "SIMULATED"}}
    path = OUTPUT_DIR / ("realism_report_final.json" if final else "realism_report.json")
    path.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"Saved {path}")
    return report


if __name__ == "__main__":
    main()
