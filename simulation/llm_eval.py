"""RETINA LLM shopper tests: are LLM-guided NPCs believable enough to make data?

Same focus group as focus.py (same NPCs, same day, same lists and options), then three tests:
  1. PRICE TEST   ask again with the product each LLM shopper chose made 20% dearer. How many
                  switch away? -> the LLM's price response ("elasticity"), next to the engine's
                  (research-based, about -2.5 x the NPC's sensitivity) on exactly the same options.
  2. REAL POPULARITY  when several options were matched to real Instacart products, does the
                  pick follow what real shoppers buy most? LLM vs engine vs picking at random.
  3. HABITS & DEALS   share of picks that are the NPC's usual product / a deal; agreement with
                  the engine, compared with what random picking would give.
Each provider is tested separately (no mixing), so models can be compared fairly.
All answers are cached: a rerun costs no LLM calls.

Run:  python simulation\\llm_eval.py                         Groq and Gemini, 10 NPCs, day 30
      python simulation\\llm_eval.py --providers groq,gemini,ollama --n 5
Output: data/focus/<store>/day<D>/eval_report.txt, eval_summary.json, eval_<provider>.csv
"""

import argparse
import json
import math
import time

import numpy as np
import pandas as pd

from config import ROOT, PRICE_ELASTICITY, REFERENCE_SHARE, LOSS_AVERSION
import focus as F
import personas as P
from engine import Simulation

PRICE_UP = 1.20
MIN_ITEMS = 8                 # fewer items than this in a group: "too few to judge" instead of a number


def prepare(n, day, say=print):
    """Personas, the store on that day, and every NPC's list with its options (no LLM yet)."""
    sim0 = Simulation(verbose=False)
    ids = P.varied_sample(sim0, n)
    people = P.build(sim0, ids, offline=True, say=lambda *a, **k: None)   # saved personas, no new calls
    del sim0
    sim, day, prices_g, regular, promo_g = F.store_on(day, say)
    npcs = []
    for s in ids:
        g = int(sim.shoppers.sensitivity_group[s])
        decisions, info, hour = F.engine_trip(sim, day, s, prices_g, promo_g)
        items = []
        for k, d in enumerate(decisions):
            aisle, opts = F.options_for(sim, s, d, k, day, prices_g[g], promo_g[g])
            if opts:
                for o in opts:
                    p = o["product"]
                    o["real_popularity"] = (float(sim.store.products["popularity"].iat[p])
                                            if pd.notna(sim.store.products["instacart_match"].iat[p]) else None)
                    o["engine_weight"] = engine_weight(sim, s, aisle, p)
                items.append({**d, "aisle": aisle, "options": opts})
        if items:
            npcs.append({"s": s, "persona": people[s], "items": items, "group": g,
                         "trip": {"mission": info["mission"], "budget": float(min(info["budget"], 1e6))}})
    return sim, day, npcs


def engine_weight(sim, s, aisle, p):
    """The engine's pick weight for product p in its aisle today (popularity x price effect)."""
    a = sim.store.aisles[aisle]
    cum = sim.tables[int(sim.shoppers.sensitivity_group[s])]["cum"][aisle]
    share = np.diff(np.r_[0.0, cum])
    return float(share[int(np.flatnonzero(a.product_idx == p)[0])])


def engine_price_factor(sim, g):
    """How much the engine's pick weight falls when a price rises 20% (memory part feels it as a loss)."""
    sens = sim.group_sensitivity[g]
    e_mem = PRICE_ELASTICITY * REFERENCE_SHARE
    e_perm = PRICE_ELASTICITY - e_mem
    return PRICE_UP ** (-(e_perm + e_mem * LOSS_AVERSION) * sens)


DEAD = set()                  # providers that stopped answering during this run


def ask_all(npcs, provider, model, cache, raise_letter=None, say=print):
    """One LLM call per NPC. raise_letter[(s, item)] = letter of the option to make 20% dearer."""
    answers, lost = {}, 0
    for k, npc in enumerate(npcs, 1):
        if lost >= 2 or provider in DEAD:                     # provider not reachable: stop asking
            DEAD.add(provider)
            answers[npc["s"]] = {}
            continue
        say(f"    NPC {k}/{len(npcs)} {npc['persona']['facts']['name']} ({len(npc['items'])} items)", end="\r")
        items = npc["items"]
        if raise_letter is not None:
            items = []
            for i, it in enumerate(npc["items"], 1):
                up = raise_letter.get((npc["s"], i))
                opts = [dict(o, price=round(o["price"] * PRICE_UP, 2)) if o["letter"] == up else o
                        for o in it["options"]]
                items.append({**it, "options": opts})
        try:
            text, source, cached = F.ask(F.prompt_for(npc["persona"], npc["trip"], items), provider, model, cache)
            answers[npc["s"]] = F.parse_choices(text, items)
            if not cached and provider != "ollama":
                time.sleep(1.0)
        except Exception as error:
            lost = lost + 1 if any(k in str(error) for k in ("Could not reach", "No key", "No LLM")) else 0
            say(f"    {npc['persona']['facts']['name']}: {str(error).splitlines()[0][:100]}")
            answers[npc["s"]] = {}
    return answers


def evaluate(sim, npcs, provider, model, cache, say=print):
    t = time.time()
    say(f"\n[{provider}] base choices...")
    base = ask_all(npcs, provider, model, cache, say=say)
    chosen = {(n["s"], i): c["choice"] for n in npcs for i, c in base[n["s"]].items() if c["choice"] != "SKIP"}
    if provider in DEAD:
        say(f"[{provider}] stopped: not answering (for Ollama on a laptop CPU these long prompts are too slow)")
        return pd.DataFrame(), {"provider": provider, "answered": 0}
    say(f"[{provider}] price test (the chosen product 20% dearer)...")
    test = ask_all(npcs, provider, model, cache, raise_letter=chosen, say=say)
    rows = []
    for n in npcs:
        factor = engine_price_factor(sim, n["group"])
        for i, it in enumerate(n["items"], 1):
            opts = {o["letter"]: o for o in it["options"]}
            c, c2 = base[n["s"]].get(i), test[n["s"]].get(i)
            pick = opts.get(c["choice"]) if c and c["choice"] != "SKIP" else None
            real = [o for o in it["options"] if o["real_popularity"] is not None]
            best_real = max(real, key=lambda o: o["real_popularity"])["product"] if len(real) >= 2 else None
            w = np.array([o["engine_weight"] for o in it["options"]])
            engine_p = w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / len(w))
            letters = [o["letter"] for o in it["options"]]
            row = {"shopper_id": n["s"], "name": n["persona"]["facts"]["name"], "item": i, "aisle": it["aisle"],
                   "price_sensitivity": n["persona"]["facts"]["price_sensitivity"], "options": len(opts),
                   "answered": c is not None, "llm_pick": pick["name"] if pick else ("SKIP" if c else ""),
                   "llm_price": pick["price"] if pick else None,
                   "llm_usual": bool(pick and pick["usual_times"]), "llm_deal": bool(pick and pick["deal"]),
                   "deal_offered": any(o["deal"] for o in it["options"]),
                   "usual_offered": any(o["usual_times"] for o in it["options"]),
                   "engine_bought": it["bought"], "agree": bool(pick and pick["product"] == it["bought"]),
                   "llm_reason": c["reason"] if c else ""}
            # what the engine's NPC actually bought on its trip (habits, needs, prices, deals)
            bought = next((o for o in it["options"] if o["product"] == it["bought"]), None)
            row["engine_usual"] = bool(bought["usual_times"]) if bought else np.nan
            row["engine_deal"] = bool(bought["deal"]) if bought else np.nan
            if best_real is not None:
                row["real_best_offered"] = True
                row["llm_picked_real_best"] = bool(pick and pick["product"] == best_real)
                row["engine_real_best"] = (bought["product"] == best_real) if bought else np.nan
                row["random_p_real_best"] = 1 / len(opts)
            if pick:
                j = letters.index(pick["letter"])
                p_old = engine_p[j]
                w2 = w.copy()
                w2[j] *= factor
                p_new = w2[j] / w2.sum() if w2.sum() > 0 else p_old
                row["engine_stay_ratio"] = float(p_new / p_old) if p_old > 0 else np.nan
                row["llm_stayed"] = bool(c2 and c2["choice"] == pick["letter"]) if c2 else None
                row["llm_test_pick"] = (opts[c2["choice"]]["name"] if c2 and c2["choice"] in opts else
                                        ("SKIP" if c2 else ""))
            rows.append(row)
    df = pd.DataFrame(rows)
    return df, metrics(df, provider, time.time() - t)


def elasticity(stay_share):
    """Share still choosing the product after +20% -> price elasticity (log ratio / log 1.2)."""
    return math.log(max(stay_share, 0.01)) / math.log(PRICE_UP)


def metrics(df, provider, seconds):
    a = df[df["answered"]]
    m = {"provider": provider, "items": int(len(df)), "answered": int(len(a)), "seconds": round(seconds)}
    pt = a[a["llm_stayed"].notna()] if "llm_stayed" in a else a.iloc[0:0]
    if len(pt):
        stay = pt["llm_stayed"].astype(float).mean()
        m["llm_stay_after_+20%"] = round(stay, 3)
        m["llm_elasticity"] = round(elasticity(stay), 2)
        m["engine_stay_after_+20%"] = round(pt["engine_stay_ratio"].mean(), 3)
        m["engine_elasticity"] = round(elasticity(pt["engine_stay_ratio"].mean()), 2)
        by = pt.groupby("price_sensitivity")["llm_stayed"].apply(
            lambda x: elasticity(x.astype(float).mean()) if len(x) >= MIN_ITEMS else None)
        m["llm_elasticity_by_sensitivity"] = {k: (round(v, 2) if v is not None and v == v else None)
                                              for k, v in by.items()}
    rb = a[a["real_best_offered"] == True] if "real_best_offered" in a else a.iloc[0:0]
    if len(rb):
        m["picks_real_best_seller"] = {"llm": round(rb["llm_picked_real_best"].mean(), 3),
                                       "engine": round(rb["engine_real_best"].astype(float).mean(), 3),
                                       "random": round(rb["random_p_real_best"].mean(), 3), "items": int(len(rb))}
    u, d = a[a["usual_offered"]], a[a["deal_offered"]]
    m["picks_usual_product"] = {"llm": round(u["llm_usual"].mean(), 3) if len(u) else None,
                                "engine": round(u["engine_usual"].astype(float).mean(), 3) if len(u) else None}
    m["picks_deal"] = {"llm": round(d["llm_deal"].mean(), 3) if len(d) else None,
                       "engine": round(d["engine_deal"].astype(float).mean(), 3) if len(d) else None}
    m["agrees_with_engine"] = round(a["agree"].mean(), 3) if len(a) else None
    m["random_agreement"] = round((1 / a["options"]).mean(), 3) if len(a) else None
    m["skips"] = round((a["llm_pick"] == "SKIP").mean(), 3) if len(a) else None
    return m


def report(results, day, npcs, say=print):
    weak = [m for m in results if m["answered"] < 0.5 * m["items"]]
    results = [m for m in results if m not in weak] or results
    def pct(x):
        return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.0f}%"
    lines = [f"RETINA LLM shopper tests - day {day}, {len(npcs)} NPCs, {sum(len(n['items']) for n in npcs)} list items (SIMULATED)",
             "", f"{'test':<46}" + "".join(f"{m['provider']:>12}" for m in results) + f"{'engine':>12}{'random':>10}"]

    def row(name, llm_key, engine_val=None, random_val=None, fmt=pct):
        vals = "".join(f"{fmt(llm_key(m)):>12}" for m in results)
        lines.append(f"{name:<46}{vals}{(fmt(engine_val) if engine_val is not None else ''):>12}"
                     f"{(fmt(random_val) if random_val is not None else ''):>10}")
    first = results[0]
    num = lambda x: "-" if x is None or x != x else f"{x:+.2f}"          # x != x: not a number
    row("1. still buys it after +20% price", lambda m: m.get("llm_stay_after_+20%"),
        first.get("engine_stay_after_+20%"))
    row("   price elasticity (research: about -2.5)", lambda m: m.get("llm_elasticity"),
        first.get("engine_elasticity"), fmt=num)
    for k in first.get("llm_elasticity_by_sensitivity", {}):
        row(f"   elasticity, {k}", lambda m, k=k: m.get("llm_elasticity_by_sensitivity", {}).get(k), fmt=num)
    rb = first.get("picks_real_best_seller", {})
    row("2. picks the real best seller (Instacart)", lambda m: m.get("picks_real_best_seller", {}).get("llm"),
        rb.get("engine"), rb.get("random"))
    row("3. picks their usual product (when offered)", lambda m: m["picks_usual_product"]["llm"],
        first["picks_usual_product"]["engine"])
    row("   picks a deal (when offered)", lambda m: m["picks_deal"]["llm"], first["picks_deal"]["engine"])
    row("   same product as the engine's trip", lambda m: m["agrees_with_engine"], None, first["random_agreement"])
    row("   skips the item", lambda m: m["skips"])
    lines.append(f"{'   answered / items':<46}" + "".join(f"{str(m['answered']) + '/' + str(m['items']):>12}" for m in results))
    lines.append(f"{'   time (seconds, 0 = all saved)':<46}" + "".join(f"{m['seconds']:>12}" for m in results))
    for m in weak:
        lines.append(f"Left out: {m['provider']} answered only {m['answered']} of {m['items']} items - too few to judge.")
    lines += ["'-' in a sensitivity row = fewer than 8 items in that group (too few to judge).",
              "", "How to read it: 'engine' = RETINA's statistical NPC (test 1: its choice rule on exactly the same",
              "options; tests 2-3: what it actually bought on its trip), 'random' = picking an option blindly. Elasticity: % change in choices for",
              "a 1% price rise (-2.5 means a 20% dearer product loses about 37% of its buyers). Small samples: these",
              "are early signals, not proof."]
    say("\n" + "\n".join(lines))
    return lines


def main():
    ap = argparse.ArgumentParser(description="Test LLM-guided shoppers against the engine and real data.")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--day", type=int, default=30)
    ap.add_argument("--providers", default="groq,gemini", help="comma list: groq,gemini,ollama")
    args = ap.parse_args()
    sim, day, npcs = prepare(args.n, args.day)
    cache = F.load_cache()
    out_dir = F.FOCUS_DIR / f"day{day:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for provider in [p.strip() for p in args.providers.split(",") if p.strip()]:
        try:
            df, m = evaluate(sim, npcs, provider, None, cache)
        except Exception as error:
            print(f"[{provider}] skipped: {str(error)[:120]}")
            continue
        if m["answered"] == 0:
            print(f"[{provider}] gave no usable answers - left out")
            continue
        df.assign(data_label="SIMULATED").to_csv(out_dir / f"eval_{provider}.csv", index=False)
        results.append(m)
    if not results:
        print("No provider answered. Check keys ($env:GROQ_API_KEY / $env:GEMINI_API_KEY) or start Ollama.")
        return
    lines = report(results, day, npcs)
    (out_dir / "eval_report.txt").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "eval_summary.json").write_text(json.dumps({"day": day, "results": results,
                                                           "data_label": "SIMULATED"}, indent=1), encoding="utf-8")
    print(f"\nSaved to {out_dir.relative_to(ROOT)} (eval_report.txt, eval_summary.json, eval_<provider>.csv)")


if __name__ == "__main__":
    main()
