"""Collect mode: varied shelf situations, labelled by ONE teacher (Groq openai/gpt-oss-120b).

Natural mix runs give only a few hundred LLM decisions per 10 days, most of them "new on the list".
Collect mode builds a POOL of situations straight from the store and varies them on purpose:
  natural    as the store has it
  price_up   the engine's pick costs 10-30% more than the NPC paid last time
  deal       another option is 10-30% off
  sold_out   the engine's pick (often their usual) is sold out
  paid_less  they remember paying 5-20% less for it
The pool: the supermarket on days 10, 24 and 38, a varied panel of 150 NPCs, each NPC's real
shopping list and real options (shuffled order). Saved once in data/teacher/pool.jsonl.gz.

Labelling packs whole NPC trips into requests of ~6,000 tokens (~25 items). Strict rules:
  - Groq only, no fallback: when its free day is used up, collect stops (or --wait for tomorrow)
  - it stops at 80% of the free daily limit (requests and tokens), leaving room for other use
  - every label keeps the provider and model that answered; resumes where it stopped
`--repeat 100` asks 100 labelled items again: the teacher's self-agreement = the student's ceiling.
"""

import gzip
import json
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from retina.engine import llm
from retina.engine.config import PROFILE, SENSITIVITY_GROUPS, stream
from retina.mind.mind import build_options, WHY_HERE
from retina.mind.teacher import (LLMTeacher, FakeTeacher, npc_block, request_tokens, MAX_REQUEST_TOKENS,
                                 PROMPT_VERSION)
from retina.student.features import npc_numbers
from retina.student.labels import LabelStore, TEACHER, TEACHER_DIR

POOL_DAYS = (10, 24, 38)
POOL_PANEL = 150
BUDGET_SHARE = 0.80                     # stop at 80% of the free daily limit (owner's rule)
STREAM_COLLECT, STREAM_COLLECT_ORDER, STREAM_COLLECT_VARY = 12, 13, 14
VARIATIONS = [("natural", 0.30), ("price_up", 0.20), ("deal", 0.20), ("sold_out", 0.15), ("paid_less", 0.15)]


# ================================================================== the pool
def vary(options, wanted, rng):
    """Change one thing on purpose. Returns the variation's name (options are edited in place)."""
    u = rng.random()
    kind = next(name for name, cum in zip([v[0] for v in VARIATIONS], np.cumsum([v[1] for v in VARIATIONS]))
                if u < cum)
    own = next(o for o in options if o["product"] == wanted)
    others = [o for o in options if o["product"] != wanted and o["in_stock"]]
    if kind == "price_up":
        own["last_price"] = own["price"]
        own["price"] = round(own["price"] * rng.uniform(1.10, 1.30), 2)
        own["deal"], own["off"] = False, 0.0
    elif kind == "deal" and others:
        o = others[int(rng.integers(len(others)))]
        o["price"] = round(o["regular"] * rng.uniform(0.70, 0.90), 2)
        o["deal"], o["off"] = True, round((1 - o["price"] / o["regular"]) * 100, 1)
    elif kind == "sold_out" and others:
        own["in_stock"] = False
    elif kind == "paid_less":
        own["last_price"] = round(own["price"] * rng.uniform(0.80, 0.95), 2)
    else:
        kind = "natural"
    return kind


def build_pool(path=None, days=POOL_DAYS, panel=POOL_PANEL, say=print):
    """Make the situation pool (deterministic: same seed -> same pool). Returns the trips."""
    from retina.engine.engine import Simulation, shelf_prices
    from retina.engine.personas import npc_facts, template_persona, varied_sample
    from retina.engine.shopping import make_list
    path = Path(path or TEACHER_DIR / "pool.jsonl.gz")
    say(f"Building the situation pool: {PROFILE['name']}, days {', '.join(map(str, days))}, {panel} NPCs...")
    sim = Simulation(verbose=False)
    trips = []
    for day in days:
        while sim.day < day:
            sim.run_day()
        prices_g, regular, promo_g, _ = shelf_prices(day, sim.store, sim.book, sim.iv, SENSITIVITY_GROUPS)
        sh = sim.shoppers
        for s in varied_sample(sim, panel):
            g = int(sh.sensitivity_group[s])
            trip = make_list(s, day, sim.store, sh, sim.tables, promo_g[g], stream(STREAM_COLLECT, day, s), prices_g[g])
            facts = npc_facts(sim, s)
            items = []
            for k, (wanted, route, _, _, _) in enumerate(trip["plan"]):
                aisle, options = build_options(sim, s, k, day, int(wanted), sim.tables[g], prices_g[g], promo_g[g],
                                               regular, order_key=(STREAM_COLLECT_ORDER, day, s, k))
                if len(options) < 2:
                    continue
                kind = vary(options, int(wanted), stream(STREAM_COLLECT_VARY, day, s, k))
                if not any(o["in_stock"] for o in options):
                    continue
                items.append({"id": f"c-{PROFILE['id']}-{sim.seed}-{day}-{s}-{k}", "k": k, "aisle": aisle,
                              "route": route, "why_here": WHY_HERE.get(route, route), "variation": kind,
                              "engine_product": int(wanted), "options": options})
            if items:
                trips.append({"day": day, "npc": int(s), "card": {"facts": facts, "persona": template_persona(facts)},
                              "npc_x": npc_numbers(sh, s), "mission": trip["mission"],
                              "budget": float(min(trip["budget"], 1e6)), "items": items})
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for t in trips:
            f.write(json.dumps(t) + "\n")
    say(f"  {sum(len(t['items']) for t in trips):,} situations in {len(trips):,} trips -> {path}")
    return trips


def load_pool(path=None, say=print):
    path = Path(path or TEACHER_DIR / "pool.jsonl.gz")
    if not path.exists():
        return build_pool(path, say=say)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ================================================================ labelling
def requests_for(trips):
    """Pack trips (with only their items still to label) into requests under MAX_REQUEST_TOKENS."""
    groups, current = [], []
    for t in trips:
        trial = current + [t]
        blocks = [npc_block(n, x["card"], [], x, x["items"]) for n, x in enumerate(trial, 1)]
        if current and request_tokens(blocks) > MAX_REQUEST_TOKENS:
            groups.append(current)
            current = [t]
        else:
            current = trial
    return groups + ([current] if current else [])


def room_today(provider, tokens, share=BUDGET_SHARE):
    """Is there room for one more request inside `share` of today's free limit? (+ a message)"""
    lim = llm.limits(provider)
    used_req, used_tok = llm.PACER.used_today(provider)
    if lim["rpd"] is not None and used_req + 1 > share * lim["rpd"]:
        return False, f"{used_req:,} of {lim['rpd']:,} requests used today ({share:.0%} stop)"
    if lim["tpd"] is not None and used_tok + tokens > share * lim["tpd"]:
        return False, f"{used_tok:,} of {lim['tpd']:,} tokens used today ({share:.0%} stop)"
    return True, f"{used_tok:,} tokens used today" + (f" of {lim['tpd']:,}" if lim["tpd"] else "")


def until_tomorrow():
    now = datetime.now(timezone.utc)
    return ((now + timedelta(days=1)).replace(hour=0, minute=1, second=0, microsecond=0) - now).total_seconds()


def label_rows(group, answers, provider, model, repeat):
    rows = []
    for n, t in enumerate(group, 1):
        for i, it in enumerate(t["items"], 1):
            ans = answers.get((n, i))
            rows.append({"id": it["id"], "repeat": repeat, "source": "collect", "provider": provider, "model": model,
                         "prompt_version": PROMPT_VERSION, "npc": t["npc"], "day": t["day"], "valid": ans is not None,
                         "store": PROFILE["id"], "aisle": it["aisle"], "route": it["route"], "mission": t["mission"],
                         "variation": it["variation"], "npc_x": t["npc_x"], "options": it["options"],
                         "engine_product": it["engine_product"],
                         "choice": None if ans is None else ans["product"],
                         "letter": None if ans is None else ans["choice"],
                         "why_type": None if ans is None else ans["why_type"], "why": None if ans is None else ans["why"]})
    return rows


def collect(target=3000, repeat=0, provider="groq", model=None, wait=False, store=None, pool=None, say=print,
            share=BUDGET_SHARE, teacher=None):
    """Label pool situations until `target` valid labels (repeat=0), or ask `target` labelled items again
    (repeat=1). Returns the number of valid labels for that repeat. teacher: a stand-in (tests)."""
    model = model or (TEACHER[1] if provider == TEACHER[0] else None)
    store = store or LabelStore()
    trips = pool if pool is not None else load_pool(say=say)
    teacher = teacher or (FakeTeacher(say) if provider == "fake" else LLMTeacher(provider, model, say))
    if repeat:
        firsts = [r["id"] for r in store.rows(0, provider, model if provider != "fake" else None)][:target]
        wanted, goal = set(firsts) - store.ids(repeat), len(firsts)
        if len(firsts) < target:
            say(f"Only {len(firsts)} labels to repeat (asked for {target}).")
    else:
        wanted, goal = None, target
    done = lambda: len([r for r in store.rows(repeat) if provider == "fake" or r["provider"] == provider])
    have = done()
    asked = store.ids(repeat)
    todo = []
    for t in trips:
        items = [it for it in t["items"] if it["id"] not in asked and (wanted is None or it["id"] in wanted)]
        if items:
            todo.append({**t, "items": items})
    say(f"Collect ({'repeat' if repeat else 'labels'}): {have:,} of {goal:,} done; teacher {provider}"
        + (f" {model}" if model else "") + (" (fixed rules, NOT an LLM)" if provider == "fake" else ""))
    queue = deque(requests_for(todo))
    while queue:
        group = queue.popleft()
        if have >= goal:
            break
        blocks = [npc_block(n, t["card"], [], t, t["items"]) for n, t in enumerate(group, 1)]
        if provider != "fake":
            ok, note = room_today(provider, request_tokens(blocks), share)
            while not ok:
                if not wait:
                    say(f"Stopped: {note}. {have:,}/{goal:,} labels saved - run again tomorrow to continue.")
                    return have
                pause = until_tomorrow()
                say(f"Waiting {pause / 3600:.1f} h for {provider}'s next free day ({note}) - Ctrl+C stops safely.")
                time.sleep(pause)
                ok, note = room_today(provider, request_tokens(blocks), share)
        batch = [{"block": b, "items": t["items"]} for b, t in zip(blocks, group)]
        answers, got_provider, got_model, problem = teacher.ask(batch)
        if problem and "413" in problem:                      # request too large for the free tier:
            if hasattr(teacher, "failures"):                   # not a dead provider - split and retry
                teacher.failures = 0
            if len(group) > 1:
                half = len(group) // 2
                queue.extendleft([group[half:], group[:half]])
                say(f"  (request too large for {provider} - splitting it in two)")
            else:
                say(f"  (one trip is too large for {provider} on its own - skipped)")
            continue
        if problem:
            if "daily limit" in problem or "429" in problem:
                say(f"Stopped: {provider} says its limit is reached ({problem[:80]}). Run again tomorrow.")
                return have
            if teacher.offline:
                say(f"Stopped: {provider} did not answer twice in a row ({problem[:80]}).")
                return have
            continue                                           # one failed request: try the next one
        if provider != "fake" and (got_provider, got_model) != (provider, model):
            say(f"Stopped: answered by {got_provider}/{got_model}, not {provider}/{model} - labels not saved.")
            return have
        rows = label_rows(group, answers, got_provider, got_model, repeat)
        store.add(rows)
        have += sum(r["valid"] for r in rows)
        note = room_today(provider, 0, share)[1] if provider != "fake" else "no LLM"
        say(f"  {have:,}/{goal:,} labels | {sum(not r['valid'] for r in rows)} invalid in this request | {note}")
    return have


def self_agreement(store, provider, model):
    """Share of repeated situations where the teacher picked the same option again (None if none)."""
    first = {r["id"]: r["choice"] for r in store.rows(0, provider, model)}
    again = [(first[r["id"]], r["choice"]) for r in store.rows(1, provider, model) if r["id"] in first]
    return (sum(a == b for a, b in again) / len(again), len(again)) if again else (None, 0)


def position_bias(rows):
    """Pick share by the letter shown (A, B, ...) vs what an unbiased chooser would give
    (each choosable option equally likely to sit at each position after shuffling)."""
    picked, expected = {}, {}
    for r in rows:
        letters = [o["letter"] for o in r["options"] if o["in_stock"]]
        if not r.get("letter") or r["letter"] == "SKIP" or not letters:
            continue
        picked[r["letter"]] = picked.get(r["letter"], 0) + 1
        for x in letters:
            expected[x] = expected.get(x, 0) + 1 / len(letters)
    total = sum(picked.values())
    return {x: {"picked": picked.get(x, 0) / total, "expected": expected[x] / total}
            for x in sorted(expected)} if total else {}


def status(store, say=print):
    say("Teacher labels (data/teacher/labels.sqlite):")
    for repeat, provider, model, source, valid, n in store.counts():
        say(f"  {'again' if repeat else 'label'} | {provider}/{model} | {source} | {'valid' if valid else 'invalid'}: {n:,}")
    rate, n = self_agreement(store, *TEACHER)
    if rate is not None:
        say(f"  teacher self-agreement on {n} repeated items: {rate:.0%} (the student's ceiling)")
    for p in ("groq",):
        req, tok = llm.PACER.used_today(p)
        lim = llm.limits(p)
        say(f"  {p} today: {req:,} requests, {tok:,} tokens (free day: {lim['rpd']:,} / {lim['tpd']:,}; "
            f"collect stops at {BUDGET_SHARE:.0%})")
