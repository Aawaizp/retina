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
from retina.student.labels import LabelStore, TEACHER, TEACHER_DIR, teacher_of

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
def blocks_of(group, version):
    return [npc_block(n, t["card"], [], t, t["items"], version) for n, t in enumerate(group, 1)]


def requests_for(trips, version=PROMPT_VERSION):
    """Pack trips (with only their items still to label) into requests under MAX_REQUEST_TOKENS."""
    groups, current = [], []
    for t in trips:
        trial = current + [t]
        if current and request_tokens(blocks_of(trial, version), version) > MAX_REQUEST_TOKENS:
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


def label_rows(group, answers, provider, model, repeat, version=PROMPT_VERSION):
    rows = []
    for n, t in enumerate(group, 1):
        for i, it in enumerate(t["items"], 1):
            ans = answers.get((n, i))
            rows.append({"id": it["id"], "repeat": repeat, "source": "collect", "provider": provider, "model": model,
                         "prompt_version": version, "npc": t["npc"], "day": t["day"], "valid": ans is not None,
                         "store": PROFILE["id"], "aisle": it["aisle"], "route": it["route"], "mission": t["mission"],
                         "variation": it["variation"], "npc_x": t["npc_x"], "options": it["options"],
                         "engine_product": it["engine_product"],
                         "choice": None if ans is None else ans["product"],
                         "letter": None if ans is None else ans["choice"],
                         "why_type": None if ans is None else ans["why_type"], "why": None if ans is None else ans["why"]})
    return rows


def collect(target=3000, repeat=0, provider="groq", model=None, wait=False, store=None, pool=None, say=print,
            share=BUDGET_SHARE, teacher=None, only_ids=None):
    """Label pool situations until `target` valid labels (repeat=0), or ask `target` labelled items again
    (repeat=1). Returns the number of valid labels for that repeat. teacher: a stand-in (tests).
    only_ids: label just these situations (e.g. the ones another teacher labelled - a comparison)."""
    model = model or ("rules-v1" if provider == "fake" else llm.PROVIDERS[provider]["model"])
    store = store or LabelStore()
    trips = pool if pool is not None else load_pool(say=say)
    teacher = teacher or (FakeTeacher(say) if provider == "fake" else LLMTeacher(provider, model, say))
    me = teacher_of(provider, model)
    version_of = {}                       # a repeat / comparison asks in the prompt version of the ORIGINAL label
    if repeat:
        originals = store.rows(0, provider, model)[:target]
        firsts = [r["id"] for r in originals]
        version_of = {r["id"]: r.get("prompt_version", 2) for r in originals}
        wanted, goal = set(firsts) - store.ids(repeat, me), len(firsts)
        if len(firsts) < target:
            say(f"Only {len(firsts)} labels to repeat (asked for {target}).")
    elif only_ids is not None:
        version_of = {r["id"]: r.get("prompt_version", 2) for r in store.rows(0, *TEACHER)}
        wanted, goal = set(only_ids), len(set(only_ids))
    else:
        wanted, goal = None, target
    mine = lambda: [r for r in store.rows(repeat, provider, model) if wanted is None or r["id"] in wanted]
    have = len(mine())
    asked = store.ids(repeat, me)
    todo = {}                                                  # prompt version -> trips (items to ask)
    for t in trips:
        for v in sorted({version_of.get(it["id"], PROMPT_VERSION) for it in t["items"]}):
            items = [it for it in t["items"] if it["id"] not in asked and (wanted is None or it["id"] in wanted)
                     and version_of.get(it["id"], PROMPT_VERSION) == v]
            if items:
                todo.setdefault(v, []).append({**t, "items": items})
    say(f"Collect ({'repeat' if repeat else 'labels'}): {have:,} of {goal:,} done; teacher {provider}"
        + (f" {model}" if model else "") + (" (fixed rules, NOT an LLM)" if provider == "fake" else "")
        + f"; prompt version {', '.join(map(str, sorted(todo))) or PROMPT_VERSION}")
    queue = deque((v, g) for v in sorted(todo) for g in requests_for(todo[v], v))
    while queue:
        version, group = queue.popleft()
        if have >= goal:
            break
        blocks = blocks_of(group, version)
        if provider != "fake":
            ok, note = room_today(provider, request_tokens(blocks, version), share)
            while not ok:
                if not wait:
                    say(f"Stopped: {note}. {have:,}/{goal:,} labels saved - run again tomorrow to continue.")
                    return have
                pause = until_tomorrow()
                say(f"Waiting {pause / 3600:.1f} h for {provider}'s next free day ({note}) - Ctrl+C stops safely.")
                time.sleep(pause)
                ok, note = room_today(provider, request_tokens(blocks, version), share)
        batch = [{"block": b, "items": t["items"], "version": version} for b, t in zip(blocks, group)]
        answers, got_provider, got_model, problem = teacher.ask(batch)
        if problem and "413" in problem:                      # request too large for the free tier:
            if hasattr(teacher, "failures"):                   # not a dead provider - split and retry
                teacher.failures = 0
            if len(group) > 1:
                half = len(group) // 2
                queue.extendleft([(version, group[half:]), (version, group[:half])])
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
        rows = label_rows(group, answers, got_provider, got_model, repeat, version)
        if not any(r["valid"] for r in rows):                 # the whole reply was unreadable: a failed
            say(f"  (reply could not be read - these {len(rows)} items are asked again next run)")   # request,
            continue                                          # not {len(rows)} wrong answers
        store.add(rows)
        have += sum(r["valid"] for r in rows)
        note = room_today(provider, 0, share)[1] if provider != "fake" else "no LLM"
        say(f"  {have:,}/{goal:,} labels | {sum(not r['valid'] for r in rows)} invalid in this request | {note}")
    return have


def compare(other="gemini", n=200, store=None, pool=None, say=print, share=BUDGET_SHARE, teacher=None):
    """Ask `other` the first n situations THE teacher (Groq) already labelled - same prompt, same
    shuffled option order - then report how much the two agree. Uses only `other`'s quota."""
    store = store or LabelStore()
    ids = [r["id"] for r in store.rows(0, *TEACHER)][:n]
    if len(ids) < n:
        say(f"Only {len(ids)} {TEACHER[0]} labels to compare with (asked for {n}).")
    collect(len(ids), provider=other, store=store, pool=pool, say=say, share=share, teacher=teacher, only_ids=ids)
    return compare_report(store, other, say=say)


def pick_shares(rows):
    """Repeat share (picked something bought before) and deal share of a teacher's picks; skips."""
    picks = [next((o for o in r["options"] if o["product"] == r["choice"]), None) for r in rows]
    chosen = [o for o in picks if o is not None]
    offered = [any(o["times"] > 0 and o["in_stock"] for o in r["options"]) for r in rows]
    return {"picks": len(chosen), "skips": sum(r["choice"] is not None and r["choice"] < 0 for r in rows),
            "repeat_possible": round(float(np.mean(offered)), 3) if rows else None,
            "repeat_share": round(float(np.mean([o["times"] > 0 for o in chosen])), 3) if chosen else None,
            "deal_share": round(float(np.mean([o["deal"] for o in chosen])), 3) if chosen else None}


def compare_report(store, other="gemini", other_model=None, close_gap=0.05, say=print):
    """Groq vs `other` on the same situations, next to Groq's self-agreement (the ceiling) and chance.
    Decision rule (owner's): agreement close to self-agreement (within close_gap) -> both may teach;
    else Groq only trains and the other's labels stay a comparison set."""
    other_model = other_model or (llm.PROVIDERS[other]["model"] if other in llm.PROVIDERS else None)
    base = {r["id"]: r for r in store.rows(0, *TEACHER)}
    them = {r["id"]: r for r in store.rows(0, other, other_model) if r["id"] in base}
    both = [i for i in them if base[i]["choice"] is not None and them[i]["choice"] is not None]
    if not both:
        say(f"No situations labelled by both {TEACHER[0]} and {other} yet.")
        return None
    agree = float(np.mean([base[i]["choice"] == them[i]["choice"] for i in both]))
    chance = float(np.mean([1 / max(sum(o["in_stock"] for o in base[i]["options"]), 1) for i in both]))
    self_rate, self_n = self_agreement(store, *TEACHER)
    by_variation = {}
    for i in both:
        by_variation.setdefault(base[i].get("variation", "natural"), []).append(i)
    kinds = [k for k in ("natural", "deal", "price_up", "paid_less", "sold_out") if k in by_variation]
    out = {"situations": len(both), "agreement": round(agree, 3), "chance": round(chance, 3),
           "groq_self_agreement": None if self_rate is None else round(self_rate, 3), "groq_self_items": self_n,
           "by_variation": {k: round(float(np.mean([base[i]["choice"] == them[i]["choice"] for i in by_variation[k]])), 3)
                            for k in kinds},
           TEACHER[0]: pick_shares([base[i] for i in both]), other: pick_shares([them[i] for i in both]),
           "shares_by_variation": {name: {k: pick_shares([rows[i] for i in by_variation[k]]) for k in kinds}
                                   for name, rows in ((TEACHER[0], base), (other, them))},
           "real_repeat_share": 0.60, "real_compares_with": "natural only (other types are changed on purpose)"}
    if self_rate is None:
        out["decision"] = "pending: Groq's self-agreement is not measured yet (retina collect --repeat 100)"
    elif agree >= self_rate - close_gap:
        out["decision"] = f"close to Groq's self-agreement -> {other} may teach too (labels keep their teacher)"
    else:
        out["decision"] = f"not close -> Groq only trains the student; {other} labels kept as a comparison set"
    pct = lambda x: "-" if x is None else f"{x:.0%}"
    g, o = out[TEACHER[0]], out[other]
    say(f"\n{TEACHER[0]} vs {other} on the same {len(both)} situations (same prompt, same option order):")
    say(f"  agreement {pct(agree)} | chance {pct(chance)} | Groq self-agreement "
        + (f"{pct(self_rate)} on {self_n} items" if self_rate is not None else "not measured yet"))
    say("  agreement by situation type: " + ", ".join(f"{k} {pct(v)}" for k, v in out["by_variation"].items()))
    say(f"  {'':<8}{'repeat share':>14}{'deal share':>12}{'skips':>7}   (all types together)")
    for name, s in ((TEACHER[0], g), (other, o)):
        say(f"  {name:<8}{pct(s['repeat_share']):>14}{pct(s['deal_share']):>12}{s['skips']:>7}")
    say(f"\n  By situation type - repeat share / deal share (n = situations); real ~60% repeat applies to "
        f"'natural' ONLY:")
    say(f"  {'':<11}" + "".join(f"{k:>17}" for k in kinds))
    for name in (TEACHER[0], other):
        cells = [out["shares_by_variation"][name][k] for k in kinds]
        say(f"  {name:<11}" + "".join(f"{pct(c['repeat_share']) + ' / ' + pct(c['deal_share']):>17}" for c in cells))
    say(f"  {'n':<11}" + "".join(f"{len(by_variation[k]):>17}" for k in kinds))
    say(f"  {'possible':<11}" + "".join(f"{pct(out['shares_by_variation'][TEACHER[0]][k]['repeat_possible']):>17}"
                                        for k in kinds) + "   (a bought-before option was in stock)")
    say(f"  {'real':<11}" + "".join(f"{'~60% / -' if k == 'natural' else '-':>17}" for k in kinds))
    say(f"  Decision: {out['decision']}")
    (store.path.parent / f"compare_{other}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


# ===================================================== export / import (any teacher)
def trip_key(t):
    return f"{t['day']}-{t['npc']}"


def choose_situations(which, n, store, pool):
    """Trips holding n situations: 'labelled' = the first n THE teacher (Groq) labelled (for comparing
    teachers), 'new' = the first n of the pool Groq has not labelled yet."""
    groq = store.ids(0, teacher_of(*TEACHER))
    if which == "labelled":
        wanted = [r["id"] for r in store.rows(0, *TEACHER)][:n]
    else:
        wanted = [it["id"] for t in pool for it in t["items"] if it["id"] not in groq][:n]
    keep = set(wanted)
    trips = []
    for t in pool:
        items = [it for it in t["items"] if it["id"] in keep]
        if items:
            trips.append({**t, "items": items})
    return trips, len(wanted)


def export_situations(path, n=200, which="new", store=None, pool=None, say=print):
    """One JSON line per trip: the exact prompt THE teacher gets (system + situation + reply format),
    options in the same shuffled order, and every item's situation id. No answers in the file."""
    from retina.mind.teacher import batch_prompt, system_for
    store = store or LabelStore()
    pool = pool if pool is not None else load_pool(say=say)
    trips, count = choose_situations(which, n, store, pool)
    base = {r["id"]: r.get("prompt_version", 2) for r in store.rows(0, *TEACHER)}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for t in trips:
            version = base.get(t["items"][0]["id"], PROMPT_VERSION) if which == "labelled" else PROMPT_VERSION
            items = [{"id": it["id"], "item": i, "aisle": it["aisle"],
                      "options": [{k: o[k] for k in ("letter", "name", "price", "regular", "deal", "in_stock", "times")}
                                  for o in it["options"]]}
                     for i, it in enumerate(t["items"], 1)]
            f.write(json.dumps({"trip": trip_key(t), "prompt_version": version, "system": system_for(version),
                                "prompt": batch_prompt([npc_block(1, t["card"], [], t, t["items"], version)], version),
                                "items": items, "data_label": "SIMULATED"}, ensure_ascii=False) + "\n")
    say(f"Exported {count} situations in {len(trips)} trips ({which}) -> {path}\n"
        f"  Import answers with: retina collect --import <answers.jsonl>  - one line per answer:\n"
        '  {"id": "<situation id>", "teacher": "<name>/<model>", "choice": "B", "reason_type": "price", "why": "..."}\n'
        '  or a whole reply per trip: the exported line itself plus "teacher": "<name>/<model>" and '
        '"reply": "<the JSON list the teacher wrote>"')
    return count


def import_labels(path, store=None, pool=None, say=print):
    """Read answers from another teacher and check them exactly like Groq's (offered letter, not sold
    out, or SKIP). Valid ones become labels with that teacher (source 'import'). An import may NOT
    claim to be THE training teacher: only labels RETINA collected itself can train the student."""
    from retina.mind.teacher import REASON_TYPES, parse_answers
    store = store or LabelStore()
    pool = pool if pool is not None else load_pool(say=say)
    by_id = {it["id"]: (t, it) for t in pool for it in t["items"]}
    by_trip = {trip_key(t): t for t in pool}
    base = {r["id"]: r.get("prompt_version", 2) for r in store.rows(0, *TEACHER)}
    rows, problems, seen = [], [], set()
    record_version = {}

    def keep(t, it, teacher, ans, line):
        provider, _, model = teacher.partition("/")
        if not provider or not model:
            problems.append((line, f"teacher '{teacher}' must look like name/model"))
            return
        if (provider, model) == TEACHER:
            problems.append((line, f"{teacher} is the training teacher - its labels come only from retina collect"))
            return
        if (it["id"], teacher) in seen:
            problems.append((line, f"{it['id']}: answered twice by {teacher} (first kept)"))
            return
        seen.add((it["id"], teacher))
        version = record_version.get(line) or base.get(it["id"], PROMPT_VERSION)   # the prompt it answered
        rows.extend(label_rows([{**t, "items": [it]}], {(1, 1): ans}, provider, model, 0, version))
        rows[-1]["source"] = "import"

    with open(path, encoding="utf-8-sig") as f:
        for line, text in enumerate(f, 1):
            if not text.strip():
                continue
            try:
                rec = json.loads(text)
            except json.JSONDecodeError as error:
                problems.append((line, f"not JSON ({error.msg})"))
                continue
            teacher = str(rec.get("teacher", "")).strip()
            if not teacher:
                problems.append((line, "no teacher field"))
                continue
            if isinstance(rec.get("prompt_version"), int):
                record_version[line] = rec["prompt_version"]
            if "reply" in rec:                                    # a whole raw reply for one trip
                t = by_trip.get(str(rec.get("trip")))
                if t is None:
                    problems.append((line, f"unknown trip '{rec.get('trip')}'"))
                    continue
                items = [it for it in t["items"] if it["id"] in {x.get("id") for x in rec.get("items", [])}] \
                    if rec.get("items") else t["items"]
                answers = parse_answers(str(rec["reply"]), [{"items": items}])
                for i, it in enumerate(items, 1):
                    if (1, i) in answers:
                        keep(t, it, teacher, answers[(1, i)], line)
                    else:
                        problems.append((line, f"{it['id']}: no valid answer in the reply"))
                continue
            found = by_id.get(str(rec.get("id")))
            if found is None:
                problems.append((line, f"unknown situation id '{rec.get('id')}'"))
                continue
            t, it = found
            one = json.dumps([{"shopper": 1, "item": 1, "choice": rec.get("choice"),
                               "reason_type": rec.get("reason_type", "other"), "why": rec.get("why", "")}])
            answers = parse_answers(one, [{"items": [it]}])
            if (1, 1) not in answers:
                letters = [o["letter"] for o in it["options"] if o["in_stock"]]
                problems.append((line, f"{it['id']}: choice '{rec.get('choice')}' is not one of "
                                       f"{', '.join(letters)} (in stock) or SKIP"))
                continue
            keep(t, it, teacher, answers[(1, 1)], line)
    if rows:
        store.add(rows)
    teachers = sorted({r["teacher"] for r in rows})
    say(f"Imported {len(rows)} labels from {path}" + (f" ({', '.join(teachers)})" if teachers else "")
        + f"; {len(problems)} problem(s)")
    for line, message in problems[:30]:
        say(f"  line {line}: {message}")
    if len(problems) > 30:
        say(f"  ... and {len(problems) - 30} more")
    return len(rows), problems


PASTE_DIR = TEACHER_DIR / "paste"
PASTE_HOWTO = """How to use a chat AI (ChatGPT, ...) as a teacher - by copy and paste
1. Open a NEW chat. Do not explain anything, do not mention real shoppers or numbers.
2. Paste the whole of message_01.txt and send it.
3. Copy the AI's whole answer (the JSON list) into reply_01.txt and save it.
4. Same for message_02.txt -> reply_02.txt, and so on (a new chat every few messages is fine).
5. Then run:  retina collect --paste-read {folder} --teacher chatgpt/<model you used>
   Missing or unreadable replies are listed; fix them and run the same command again
   (labels already imported are kept, not counted twice).
"""


def paste_pack(folder=PASTE_DIR, n=300, which="new", per_message=4, same_as=None, store=None, pool=None, say=print):
    """Ready-to-paste messages for a chat AI without a free API: message_KK.txt (instructions +
    up to `per_message` shoppers, the same prompt THE teacher gets) and an empty reply_KK.txt for
    each, plus pack.json (which situation is shopper S item I). same_as = 'name/model': the
    situations that teacher answered (to compare teachers on the same ones)."""
    from retina.mind.teacher import batch_prompt, system_for
    store = store or LabelStore()
    pool = pool if pool is not None else load_pool(say=say)
    if same_as:
        p, _, m = same_as.partition("/")
        keep = {r["id"] for r in store.rows(0, p, m or None)}
        trips = [{**t, "items": [it for it in t["items"] if it["id"] in keep]} for t in pool]
        trips = [t for t in trips if t["items"]]
        if not trips:
            raise SystemExit(f"No labels from {same_as} to copy the situations from.")
    else:
        trips, _ = choose_situations(which, n, store, pool)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for old in list(folder.glob("message_*.txt")) + list(folder.glob("reply_*.txt")):
        old.unlink()                                          # a new pack replaces the old one
    pack = []
    for k in range(0, len(trips), per_message):
        group, number = trips[k:k + per_message], k // per_message + 1
        blocks = [npc_block(s, t["card"], [], t, t["items"], PROMPT_VERSION) for s, t in enumerate(group, 1)]
        text = system_for(PROMPT_VERSION) + "\n\n" + batch_prompt(blocks, PROMPT_VERSION)
        (folder / f"message_{number:02d}.txt").write_text(text, encoding="utf-8")
        (folder / f"reply_{number:02d}.txt").write_text("", encoding="utf-8")
        pack.append({"message": number, "trips": [{"trip": trip_key(t), "ids": [it["id"] for it in t["items"]]}
                                                  for t in group]})
    (folder / "pack.json").write_text(json.dumps({"prompt_version": PROMPT_VERSION, "messages": pack}), encoding="utf-8")
    (folder / "HOW_TO.txt").write_text(PASTE_HOWTO.format(folder=shown_path(folder)), encoding="utf-8")
    items = sum(len(t["items"]) for t in trips)
    say(f"Wrote {len(pack)} messages ({items} situations, {len(trips)} shoppers) -> {shown_path(folder)}\n"
        f"  Read HOW_TO.txt there: paste each message_KK.txt into the chat, save the answer in reply_KK.txt,\n"
        f"  then: retina collect --paste-read {shown_path(folder)} --teacher chatgpt/<model>")
    return len(pack), items


def paste_read(folder=PASTE_DIR, teacher=None, store=None, pool=None, say=print):
    """Read reply_KK.txt files of a paste pack, turn them into one answer per situation and import them
    with import_labels (the same checks as every other teacher). Already imported ones are skipped."""
    from retina.mind.teacher import parse_answers
    if not teacher or "/" not in teacher:
        raise SystemExit("Say which AI answered: --teacher chatgpt/<model>, e.g. chatgpt/gpt-5")
    store = store or LabelStore()
    pool = pool if pool is not None else load_pool(say=say)
    folder = Path(folder)
    meta = json.loads((folder / "pack.json").read_text(encoding="utf-8"))
    by_id = {it["id"]: it for t in pool for it in t["items"]}
    provider, _, model = teacher.partition("/")
    done = {r["id"] for r in store.rows(0, provider, model, valid_only=False)}
    rows, missing, unread = [], [], []
    for msg in meta["messages"]:
        reply = folder / f"reply_{msg['message']:02d}.txt"
        text = reply.read_text(encoding="utf-8-sig").strip() if reply.exists() else ""
        if not text:
            missing.append(msg["message"])
            continue
        batch = [{"items": [by_id[i] for i in t["ids"]]} for t in msg["trips"]]
        answers = parse_answers(text, batch)
        for s, t in enumerate(msg["trips"], 1):
            for i, sid in enumerate(t["ids"], 1):
                if sid in done:
                    continue
                a = answers.get((s, i))
                if a is None:
                    unread.append(f"message {msg['message']:02d}: shopper {s} item {i}")
                    continue
                rows.append({"id": sid, "teacher": teacher, "choice": a["choice"], "reason_type": a["why_type"],
                             "why": a["why"], "prompt_version": meta["prompt_version"]})
    answers_file = folder / f"answers_{provider}.jsonl"
    answers_file.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    n, problems = import_labels(answers_file, store=store, pool=pool, say=say) if rows else (0, [])
    if missing:
        say(f"  No reply yet in: {', '.join(f'reply_{k:02d}.txt' for k in missing)}")
    if unread:
        say(f"  {len(unread)} answers missing or not valid (wrong letter, sold out, or not in the JSON), e.g. "
            + "; ".join(unread[:5]))
    total = sum(len(t["ids"]) for m in meta["messages"] for t in m["trips"])
    say(f"{teacher}: {len(done) + n} of {total} situations labelled. "
        f"Compare: retina collect --compare-with {teacher}")
    return n, missing, unread


def shown_path(path):
    try:
        return str(Path(path).resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def token_report(pool, store=None, say=print, real_per_label=None):
    """Estimated tokens per label: old prompt (v2) vs the compact one (v3), on the same trips, packed into
    requests the same way (prompt + reply text only - real use adds the model's hidden reasoning)."""
    out = {}
    for version in (2, PROMPT_VERSION):
        groups = requests_for(pool, version)
        tokens = sum(request_tokens(blocks_of(g, version), version) for g in groups)
        items = sum(len(t["items"]) for g in groups for t in g)
        out[version] = {"requests": len(groups), "items": items, "per_label": tokens / items}
    old, new = out[2]["per_label"], out[PROMPT_VERSION]["per_label"]
    say(f"Estimated tokens per label on {out[2]['items']:,} situations (prompt + reply, packed in requests):")
    say(f"  old prompt v2: {old:6.0f} tokens/label in {out[2]['requests']:,} requests")
    say(f"  new prompt v{PROMPT_VERSION}: {new:6.0f} tokens/label in {out[PROMPT_VERSION]['requests']:,} requests "
        f"({1 - new / old:.0%} fewer)")
    if real_per_label:
        say(f"  real (Groq, v2, measured 12 Oct): ~{real_per_label:.0f} tokens/label = ~{real_per_label / old:.1f}x the "
            f"estimate - the rest is the model's hidden reasoning, which a shorter prompt does not shrink,\n"
            f"  so the real saving will be SMALLER than {1 - new / old:.0%}; the next real collect measures it.")
    return out


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
    for p in ("groq", "gemini"):
        req, tok = llm.PACER.used_today(p)
        lim = llm.limits(p)
        cap = lambda x: "no limit" if x is None else f"{x:,}"
        say(f"  {p} today: {req:,} requests, {tok:,} tokens (free day: {cap(lim['rpd'])} requests / "
            f"{cap(lim['tpd'])} tokens; collect stops at {BUDGET_SHARE:.0%})")
