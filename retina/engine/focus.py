"""RETINA focus group: persona NPCs shop one trip in the real store, and an LLM makes their choices.

How it works:
  1. The store runs normally up to the chosen day (same seed = same store, prices, stock, NPCs).
  2. Each focus NPC (a varied sample with personas) does a trip with the NORMAL engine: their
     shopping list (needs, habits, missions), the walk, today's prices and deals, stock. This is
     the statistical NPC's answer.
  3. For every item on the list the engine shows the LLM 3-5 REAL products from that aisle, with
     today's price, deal, stock and "your usual". ONE LLM call per NPC: as that persona, pick a
     product (or skip) for each item and say why in a few words.
  4. Both answers are saved side by side: did the LLM agree, take more deals, pay less, stick to
     habits, stay within budget? Today the LLM choices are a SHADOW: they are recorded, not
     added to the store's sales (the statistical engine still makes the dataset).
The LLM can only pick from the options given (no made-up products or prices); bad answers fall
back to the engine's choice and are counted. Answers are cached: rerunning costs no LLM calls.

Run:  python -m retina.engine.focus                  10 NPCs on day 30 (LLM chain: Groq -> Gemini -> Ollama)
      python -m retina.engine.focus --n 20 --day 45
      python -m retina.engine.focus --provider ollama   one provider only
      python -m retina.engine.focus --offline           no LLM: shows the trips and options only
Output: data/focus/<store>/day<D>/decisions.csv, npcs.csv, summary.json, report.txt  (SIMULATED)
"""

import argparse
import hashlib
import json
import re
import time

import numpy as np
import pandas as pd

from . import config, csvsafe
from .config import ROOT, PROFILE, SENSITIVITY_GROUPS, STREAM_TRIPS, stream
from . import shopping
from . import traffic
from . import personas as P
from .engine import Simulation, shelf_prices, apply_store_changes
from .world import remove_spoiled, receive_deliveries, sample_from

PROMPT_VERSION = 1
STREAM_FOCUS = 9                       # dice for shuffling the order options are shown in
MAX_OPTIONS = 5
FOCUS_DIR = ROOT / "data" / "focus" / config.RUN_ID
CACHE_FILE = FOCUS_DIR / "llm_cache.json"
LETTERS = "ABCDE"


# ============================================================ the store on day D
def store_on(day, say=print):
    """Run the store up to `day` and do that morning's work (spoiled stock out, deliveries in)."""
    if day % 7 == 0:
        day += 1                                   # weekly price/table updates happen on day % 7 == 0
    say(f"Running the store to day {day} (same seed = same store)...")
    sim = Simulation(verbose=False)
    while sim.day < day:
        sim.run_day()
    prices_g, regular, promo_g, shelf = shelf_prices(day, sim.store, sim.book, sim.iv, SENSITIVITY_GROUPS)
    apply_store_changes(day, sim.iv, sim.inv)
    remove_spoiled(day, sim.inv)
    receive_deliveries(day, sim.inv)
    return sim, day, prices_g, regular, promo_g


def engine_trip(sim, day, s, prices_g, promo_g):
    """The statistical NPC's trip (normal engine), with every step recorded."""
    sh = sim.shoppers
    g = int(sh.sensitivity_group[s])
    rng = stream(STREAM_TRIPS, day, s)
    hour = int(sample_from(traffic.hour_table() if traffic.enabled() else sim.params["orders_by_hour"], 1, rng)[0])
    shopping.RECORDER.update(day=day, events=[])
    try:
        basket, missed, eff, walk, info = shopping.fill_basket(s, day, sim.store, sh, sim.tables, sim.inv,
                                                               promo_g[g], rng, False, prices_g[g])
        events = list(shopping.RECORDER["events"])
    finally:
        shopping.RECORDER.update(day=None, events=[])
    decisions, cur = [], None
    for e in events:
        if e["event"] == "VIEW_PRODUCT":
            cur = {"wanted": e["product"], "route": e["route"], "bought": -1, "qty": 0, "skip": "", "swap": ""}
            decisions.append(cur)
        elif cur is not None and "stop" in e:
            if e["event"] == "PURCHASE" and e.get("route") != "impulse":
                cur["bought"], cur["qty"] = e["product"], e["qty"]
            elif e["event"] == "SKIP":
                cur["skip"] = e["reason"]
            elif e["event"] == "SUBSTITUTE":
                cur["swap"] = e["reason"]
    return decisions, info, hour


# ============================================================ options for one item
def options_for(sim, s, d, k, day, prices, promo):
    """3-5 real products from the item's aisle: the engine's pick, their usual ones, a deal, best sellers."""
    store, sh = sim.store, sim.shoppers
    aisle = store.aisle_of[d["wanted"]]
    cum = sim.tables[int(sh.sensitivity_group[s])]["cum"][aisle]
    if cum is None:
        return aisle, []
    a = store.aisles[aisle]
    share = np.diff(np.r_[0.0, cum])
    ranked = a.product_idx[np.argsort(-share, kind="stable")]
    counts = sh.counts[s]
    usual = sorted(sh.by_aisle[s].get(aisle, []), key=lambda p: -counts.get(p, 0))[:2]
    deal = [p for p in ranked[:30] if promo[p]][:1]
    picks = []
    for p in [d["wanted"], d["bought"]] + usual + deal + list(ranked[:8]):
        if p >= 0 and p not in picks:
            picks.append(int(p))
    picks = picks[:MAX_OPTIONS]
    order = stream(STREAM_FOCUS, day, s, k).permutation(len(picks))   # no "A is always right" bias
    out = []
    for letter, i in zip(LETTERS, order):
        p = picks[i]
        out.append({"letter": letter, "product": p, "name": P.clean(store.products["product_name"].iat[p]),
                    "price": round(float(prices[p]), 2),
                    "regular": round(float(store.products["regular_price"].iat[p]), 2),
                    "deal": bool(promo[p]), "usual_times": int(counts.get(p, 0)),
                    "in_stock": bool(sim.inv.stock[p] > 0) or p == d["bought"]})
    return aisle, out


def option_line(o):
    deal = f", DEAL ({(1 - o['price'] / o['regular']) * 100:.0f}% off)" if o["deal"] and o["regular"] > o["price"] else ""
    usual = f", you bought it {o['usual_times']}x before" if o["usual_times"] else ""
    stock = "" if o["in_stock"] else ", SOLD OUT"
    return f"   {o['letter']}) {o['name']} - ${o['price']:.2f}{deal}{usual}{stock}"


# ============================================================ the LLM
SYSTEM = ("You role-play ONE shopper in a SIMULATED supermarket used to make synthetic retail data. "
          "Stay in character (the persona and facts given). For each item on the shopping list pick exactly "
          "one of the offered products by its letter, or SKIP if none fits (too expensive, not needed today). "
          "Think like a real person: habits, price, deals, budget, household. Never pick a SOLD OUT product. "
          "Items from the same aisle are different things on the list: pick different products for them "
          "(want more of one product? raise its qty instead). "
          "Reply with JSON only.")


def prompt_for(persona, trip, items):
    f = persona["facts"]
    lines = [f"You are {f['name']}, {f['age']}. {persona['bio']}",
             f"Shopping style: {persona['shopping_style']}. You care about: {', '.join(persona['cares_about'])}.",
             f"Facts: {P.household_phrase(f)}; {f['income']}; about ${f['weekly_food_budget']} a week for food; "
             f"{f['price_sensitivity']}.",
             f"Today: a {trip['mission'].replace('_', '-')} trip. Money for this trip: about ${trip['budget']:.0f}.",
             "", "Shopping list (each item: choose one letter or SKIP):"]
    for i, it in enumerate(items, 1):
        lines.append(f"{i}. {it['aisle']}:")
        lines += [option_line(o) for o in it["options"]]
    lines += ["", 'Reply as a JSON list: [{"item": 1, "choice": "A", "qty": 1, "reason": "<max 12 words>"}, ...] '
              "- one object per item, qty 1-4."]
    return "\n".join(lines)


def parse_choices(text, items):
    """Valid choices per item number; anything broken is left out (the engine's pick is used)."""
    from .llm import json_from
    out = {}
    for c in json_from(text):
        if not isinstance(c, dict):
            continue
        try:
            i = int(c.get("item"))
        except (TypeError, ValueError):
            continue
        if not 1 <= i <= len(items):
            continue
        choice = str(c.get("choice", "")).strip().upper()[:4]
        opts = {o["letter"]: o for o in items[i - 1]["options"]}
        if choice != "SKIP" and (choice not in opts or not opts[choice]["in_stock"]):
            continue                                # not an option, or sold out
        try:
            qty = min(max(int(c.get("qty", 1)), 1), 4)
        except (TypeError, ValueError):
            qty = 1
        out[i] = {"choice": choice, "qty": qty, "reason": P.tidy(str(c.get("reason", "")))[:120]}
    return out


def load_cache():
    return json.loads(CACHE_FILE.read_text(encoding="utf-8")) if CACHE_FILE.exists() else {}


def ask(prompt, provider, model, cache):
    from . import llm
    key = hashlib.sha1(f"{PROMPT_VERSION}|{provider}|{model}|{prompt}".encode()).hexdigest()[:20]
    if key in cache:
        return cache[key]["text"], cache[key]["source"], True
    text = llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                    provider=provider, model=model, temperature=0.4, reasoning="low")
    source = f"{llm.LAST_USED['provider']}/{llm.LAST_USED['model']}"
    cache[key] = {"text": text, "source": source}
    FOCUS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
    return text, source, False


# ============================================================ run
def run(n=10, day=30, provider=None, model=None, offline=False, say=print):
    from . import llm
    provider = provider or llm.DEFAULT_PROVIDER
    sim0 = Simulation(verbose=False)                         # personas are built from day-0 facts
    ids = P.varied_sample(sim0, n)
    people = P.build(sim0, ids, provider, model, offline=offline, say=say)
    del sim0
    sim, day, prices_g, regular, promo_g = store_on(day, say)
    date = sim.book.dates[day]
    say(f"\nFocus group: {len(ids)} NPCs shop on {date:%a %d %b %Y} (day {day}) "
        f"- LLM: {'none (offline)' if offline else provider}")
    cache = load_cache()
    rows, npcs = [], []
    for s in ids:
        g = int(sim.shoppers.sensitivity_group[s])
        prices, promo = prices_g[g], promo_g[g]
        decisions, info, hour = engine_trip(sim, day, s, prices_g, promo_g)
        items = []
        for k, d in enumerate(decisions):
            aisle, opts = options_for(sim, s, d, k, day, prices, promo)
            if opts:
                items.append({**d, "aisle": aisle, "options": opts})
        persona = people[s]
        trip = {"mission": info["mission"], "budget": float(min(info["budget"], 1e6))}
        name = persona["facts"]["name"]
        if not items:
            continue
        choices, source, cached, problem = {}, "none", False, ""
        if not offline:
            try:
                text, source, cached = ask(prompt_for(persona, trip, items), provider, model, cache)
                choices = parse_choices(text, items)
            except Exception as error:
                problem = str(error)[:120]
                say(f"  {name}: LLM problem ({problem}) - engine choices kept")
        spent_rule = spent_llm = 0.0
        for i, it in enumerate(items, 1):
            opts = {o["product"]: o for o in it["options"]}
            by_letter = {o["letter"]: o for o in it["options"]}
            rule = opts.get(it["bought"]) if it["bought"] >= 0 else None
            c = choices.get(i)
            llm_opt = by_letter.get(c["choice"]) if c and c["choice"] != "SKIP" else None
            rule_cost = rule["price"] * it["qty"] if rule else 0.0
            llm_cost = llm_opt["price"] * c["qty"] if llm_opt else 0.0
            spent_rule += rule_cost
            spent_llm += llm_cost
            deal_offered = any(o["deal"] for o in it["options"])
            rows.append({
                "date": date.date(), "shopper_id": s, "name": name, "item": i, "aisle": it["aisle"],
                "list_reason": it["route"], "options": len(it["options"]), "deal_offered": deal_offered,
                "engine_product": rule["name"] if rule else "", "engine_price": rule["price"] if rule else None,
                "engine_qty": it["qty"], "engine_deal": bool(rule and rule["deal"]),
                "engine_usual": bool(rule and rule["usual_times"]),
                "engine_skip": "" if rule else (it["skip"] or "not bought"),
                "llm_choice": (c["choice"] if c else "invalid") if not offline else "",
                "llm_product": llm_opt["name"] if llm_opt else ("SKIP" if c and c["choice"] == "SKIP" else ""),
                "llm_price": llm_opt["price"] if llm_opt else None, "llm_qty": c["qty"] if llm_opt else 0,
                "llm_deal": bool(llm_opt and llm_opt["deal"]), "llm_usual": bool(llm_opt and llm_opt["usual_times"]),
                "llm_reason": c["reason"] if c else problem,
                "agree": bool(rule and llm_opt and rule["product"] == llm_opt["product"]) or (not rule and c is not None and c["choice"] == "SKIP"),
                "llm_source": source, "data_label": "SIMULATED"})
        f = persona["facts"]
        npcs.append({"shopper_id": s, "name": name, "age": f["age"], "household": P.household_phrase(f),
                     "income": f["income"], "price_sensitivity": f["price_sensitivity"], "mission": trip["mission"],
                     "trip_budget": round(trip["budget"], 2), "items": len(items),
                     "engine_spend": round(spent_rule, 2), "llm_spend": round(spent_llm, 2),
                     "llm_answered": len(choices), "llm_source": source, "cached": cached})
        say(f"  {name:<12} {len(items):>2} items | engine ${spent_rule:6.2f} | LLM ${spent_llm:6.2f} "
            f"| budget ${trip['budget']:.0f} | {len(choices)}/{len(items)} LLM answers"
            + (" (saved)" if cached else f" [{source}]" if not offline else ""))
        if not offline and not cached and provider != "ollama":
            time.sleep(1.0)
    return save_and_report(pd.DataFrame(rows), pd.DataFrame(npcs), day, offline, say)


def share(x):
    return f"{x.mean() * 100:.0f}%" if len(x) else "-"


def save_and_report(rows, npcs, day, offline, say=print):
    out_dir = FOCUS_DIR / f"day{day:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    csvsafe.to_csv(rows, out_dir / "decisions.csv", index=False)
    csvsafe.to_csv(npcs, out_dir / "npcs.csv", index=False)
    lines = [f"RETINA focus group - day {day} - {len(npcs)} NPCs, {len(rows)} list items (SIMULATED)"]
    if offline or rows.empty:
        lines.append("Offline: trips and options saved; no LLM choices to compare.")
    else:
        answered = rows[rows["llm_choice"] != "invalid"]
        bought = answered[answered["engine_product"] != ""]
        both = answered[(answered["engine_product"] != "") & (answered["llm_product"] != "SKIP")]
        cheaper = (both["llm_price"] < both["engine_price"]).mean() if len(both) else np.nan
        dearer = (both["llm_price"] > both["engine_price"]).mean() if len(both) else np.nan
        deal = answered[answered["deal_offered"]]
        lines += [
            f"LLM answered {len(answered)} of {len(rows)} items ({share(rows['llm_choice'] != 'invalid')}); "
            f"the rest kept the engine's choice",
            f"Same product as the engine: {share(bought['agree'])} of items the engine bought",
            f"LLM skipped: {share(answered['llm_product'] == 'SKIP')} | engine did not buy: {share(answered['engine_product'] == '')}",
            f"Picked a deal when one was offered: LLM {share(deal['llm_deal'])} vs engine {share(deal['engine_deal'])}",
            f"Picked a product bought before: LLM {share(answered['llm_usual'])} vs engine {share(answered['engine_usual'])}",
            f"When both bought: LLM chose cheaper {cheaper * 100:.0f}% | dearer {dearer * 100:.0f}% | same price "
            f"{(1 - cheaper - dearer) * 100:.0f}%",
            f"Trip spend: LLM ${npcs['llm_spend'].sum():,.2f} vs engine ${npcs['engine_spend'].sum():,.2f} | "
            f"LLM over the trip budget: {(npcs['llm_spend'] > npcs['trip_budget']).sum()} of {len(npcs)} NPCs"]
        by = rows.merge(npcs[["shopper_id", "price_sensitivity"]], on="shopper_id")
        by = by[(by["engine_product"] != "") & by["llm_price"].notna()]
        if len(by):
            t = by.groupby("price_sensitivity").apply(lambda d: (d["llm_price"] < d["engine_price"]).mean(),
                                                      include_groups=False)
            lines.append("LLM chose cheaper than the engine, by price sensitivity: " +
                         " | ".join(f"{k} {v * 100:.0f}%" for k, v in t.items()))
        ex = answered[(answered["llm_reason"] != "") & (answered["llm_product"] != "")].head(6)
        lines.append("Examples:")
        for r in ex.itertuples():
            lines.append(f"  {r.name}: {r.aisle} -> LLM {r.llm_product[:34]}"
                         + (f" ${r.llm_price:.2f}" if r.llm_price == r.llm_price and r.llm_price is not None else "")
                         + f" | engine {(r.engine_product or r.engine_skip)[:30]} | \"{r.llm_reason}\"")
    summary = {"day": day, "npcs": int(len(npcs)), "items": int(len(rows)), "report": lines,
               "data_label": "SIMULATED - focus group (LLM choices are a shadow, not in the sales data)"}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    (out_dir / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    say("\n" + "\n".join(lines))
    say(f"\nSaved to {out_dir.relative_to(ROOT)} (decisions.csv, npcs.csv, summary.json, report.txt)")
    return rows, npcs


def main():
    ap = argparse.ArgumentParser(description="RETINA focus group: persona NPCs, LLM choices.")
    ap.add_argument("--n", type=int, default=10, help="how many NPCs (default 10)")
    ap.add_argument("--day", type=int, default=30, help="simulation day of the trip (default 30)")
    ap.add_argument("--provider", help="auto (Groq -> Gemini -> Ollama, default) | groq | gemini | ollama")
    ap.add_argument("--model", help="model name for that provider")
    ap.add_argument("--offline", action="store_true", help="no LLM: trips and options only")
    args = ap.parse_args()
    run(args.n, args.day, args.provider, args.model, args.offline)


if __name__ == "__main__":
    main()
