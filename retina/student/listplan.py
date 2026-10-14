"""The list student, part 1: scenarios and teacher shopping lists (Phase B).

The NPC brain plans the shopping list. A teacher (an LLM) gets one NPC's SCENARIO - who they are, when
they last shopped, their budget and ~30 real products they could buy today, each with its price, deal,
how often they bought it and whether it has run out at home - and writes the list: which products and
how many. The list student (part 2, listtrain.py) learns from these lists.

Candidates (real store products, never invented), from the NPC's own state:
  usual       products they bought before (their habits), with "ran out at home" / "days left"
  goes with   real Instacart "bought together" partners of their usual products
  new idea    products a shopper of this store would pick (real department and aisle shares x the
              store's pick tables, as the engine does)
  deal        products on promotion this week in aisles they use
Shuffled (no position bias); the teacher sees only numbers 1..n for them.

  retina lists --pool                       build the scenario pool (RETINA Supercenter)
  retina lists --collect 500 --provider gemini      ask a teacher (paced, stops at 80% of the free day)
  retina lists --status
  retina lists --paste-pack 200 / --paste-read FOLDER --teacher chatgpt/<model>   chat AIs, no API
"""

import gzip
import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from retina.engine import llm
from retina.engine.config import PROFILE, SENSITIVITY_GROUPS, stream
from retina.student.features import npc_numbers
from retina.student.labels import TEACHER_DIR

LIST_DIR = TEACHER_DIR / "lists"
POOL_FILE = LIST_DIR / "pool.jsonl.gz"
CALIB_FILE = LIST_DIR / "calibration_student_run.jsonl.gz"   # scenarios from a STUDENT-run store (no labels)
LABELS_FILE = LIST_DIR / "labels.sqlite"
PASTE_DIR = LIST_DIR / "paste"
POOL_DAYS = tuple(range(14, 92, 7))        # 12 days: NPCs have real habits by then
POOL_NPCS = 180                           # per day -> ~2,100 scenarios
MAX_CANDIDATES = 30
PER_SOURCE = {"usual": 14, "goes with": 4, "new idea": 10, "deal": 4}
PER_REQUEST = 4                           # NPCs per teacher request
BUDGET_SHARE = 0.80                       # stop at 80% of the free daily limit (owner's rule)
PROMPT_VERSION = 2                         # 2 (14 Oct): neutral - no hint about what "real shoppers" do
STREAM_LIST, STREAM_LIST_ORDER = 31, 32

SYSTEMS = {
    # v1 steered the teachers ("mostly buy what has run out and their usual items"): all three teachers
    # then put ~90% usual products on their lists (real shoppers ~60%). Kept only to read old labels.
    1: ("You plan grocery shopping lists for shoppers in a SIMULATED store (synthetic data). Stay in each "
        "shopper's character. Use ONLY the numbered products given. Real shoppers mostly buy what has run out "
        "and their usual items, add a few new things or deals, and keep to their budget. Reply with JSON only."),
    2: ("You plan grocery shopping lists for shoppers in a SIMULATED store (synthetic data). Stay in each "
        "shopper's character and decide as that person would today. Use ONLY the numbered products given. "
        "Reply with JSON only."),
}
SYSTEM = SYSTEMS[PROMPT_VERSION]


# ================================================================ scenarios
def clean(text, n=48):
    return re.sub(r"\s+", " ", str(text)).strip()[:n].rstrip(" ,.")


def candidates(sim, s, day, prices, regular, promo, rng):
    """~30 real products this NPC could put on today's list (see the module notes)."""
    sh, store, inv = sim.shoppers, sim.store, sim.inv
    p = store.products
    counts, home, pantry = sh.counts[s] or {}, sh.home[s] or {}, sh.pantry[s] or {}
    group = sim.tables[int(sh.sensitivity_group[s])]
    out = {}

    def add(product, source):
        product = int(product)
        if product in out or not inv.sellable[product] or sum(1 for c in out.values() if c["source"] == source) \
                >= PER_SOURCE[source]:
            return
        times = int(counts.get(product, 0))
        left = min(home.get(product, -1), pantry.get(product, -1))
        price, reg = float(prices[product]), float(regular[product])
        out[product] = {"product": product, "name": clean(p["product_name"].iat[product]),
                        "brand": str(p["brand"].iat[product] or ""), "aisle": str(store.aisle_of[product]),
                        "department": str(p["department"].iat[product]), "price": round(price, 2),
                        "regular": round(reg, 2), "deal": bool(promo[product]),
                        "off": round((1 - price / reg) * 100) if promo[product] and reg > 0 else 0,
                        "times": times, "ran_out": bool(times and left <= day),
                        "days_left": int(max(0, left - day)) if times else 0, "source": source}

    habits = sorted(counts, key=lambda x: (-counts[x], x))
    for x in habits:
        add(x, "usual")
    for x in habits[:8]:
        pair = store.partners.get(x)
        if pair is not None:
            add(pair[0][int(np.argmax(np.diff(np.r_[0, pair[1]])))], "goes with")
    for _ in range(40):                                        # new ideas: department -> aisle -> product
        if sum(1 for c in out.values() if c["source"] == "new idea") >= PER_SOURCE["new idea"]:
            break
        d = store.departments[store.dept_names[int(np.searchsorted(store.dept_cum, rng.random() * store.dept_cum[-1]))]]
        aisle = d.aisle_names[min(int(np.searchsorted(d.aisle_cum, rng.random() * d.aisle_cum[-1])), len(d.aisle_names) - 1)]
        if aisle in store.aisles and group["cum"][aisle] is not None:
            a, cum = store.aisles[aisle], group["cum"][aisle]
            add(a.product_idx[min(int(np.searchsorted(cum, rng.random() * cum[-1])), len(a.product_idx) - 1)], "new idea")
    used = {c["aisle"] for c in out.values()}
    deals = [i for a in used if a in store.aisles for i in store.aisles[a].product_idx if promo[i]]
    for i in rng.permutation(deals)[:PER_SOURCE["deal"] * 3] if deals else []:
        add(i, "deal")
    cands = list(out.values())[:MAX_CANDIDATES]
    return [cands[i] for i in rng.permutation(len(cands))]


def build_pool(path=POOL_FILE, days=POOL_DAYS, npcs=POOL_NPCS, say=print):
    """Scenarios from a running RETINA store (deterministic: same seed -> same pool)."""
    from retina.engine.engine import Simulation
    say(f"Building the list-scenario pool: {PROFILE['name']}, {len(days)} days x {npcs} NPCs...")
    return save_scenarios(scenarios(Simulation(verbose=False), days, npcs, "l"), path, say)


def build_calibration(path=CALIB_FILE, days=POOL_DAYS, npcs=POOL_NPCS, say=print, list_dir=None):
    """Scenarios of the NPCs who really shop each day, in a store where the STUDENT runs every NPC, so the
    shoppers' histories are the ones the list student meets at run time. Only for tuning the ESTIMATED
    list size and habit corrections - never labelled, never shown to a teacher."""
    import tempfile
    from retina.engine.engine import Simulation
    from retina.mind.mind import Mind
    say(f"Building calibration scenarios from a student run: {PROFILE['name']}, days up to {max(days)}...")
    with tempfile.TemporaryDirectory() as tmp:
        sim = Simulation(verbose=False)
        sim.mind = Mind(sim, "student", Path(tmp), 50, "fake", None, list_dir=list_dir)
        if sim.mind.lister is None:
            raise SystemExit("No list student yet: retina lists --train first")
        found = scenarios(sim, days, npcs, "c", after_day=sim.mind.end_day, who=todays_shoppers)
        sim.mind.close()
    return save_scenarios(found, path, say)


def todays_shoppers(sim, n):
    """n of the NPCs who really shop today (chosen exactly as the store chooses them), spread evenly.
    They have been away longer than an average NPC, so more has run out at home - a fixed focus group
    (varied_sample) gives shorter lists than the store's real shoppers."""
    from retina.engine import traffic
    due = sim.shoppers.next_visit_day
    today = traffic.choose_visitors(sim.day, due, traffic.npcs_today(sim.day)) if traffic.enabled()         else np.flatnonzero(due == sim.day)
    return [int(x) for x in today[np.unique(np.linspace(0, len(today) - 1, min(n, len(today))).round().astype(int))]]


def scenarios(sim, days, npcs, prefix, after_day=None, who=None):
    from retina.engine.engine import shelf_prices
    from retina.engine.personas import npc_facts, template_persona, varied_sample
    pool = []
    for day in days:
        while sim.day < day:
            d = sim.day
            sim.run_day()
            if after_day:
                after_day(d)
        prices_g, regular, promo_g, _ = shelf_prices(day, sim.store, sim.book, sim.iv, SENSITIVITY_GROUPS)
        sh = sim.shoppers
        for s in (who or varied_sample)(sim, npcs):
            g = int(sh.sensitivity_group[s])
            cands = candidates(sim, s, day, prices_g[g], regular, promo_g[g], stream(STREAM_LIST, day, s))
            if len(cands) < 8:
                continue
            last = sh.last_trip_day[s]
            facts = npc_facts(sim, s)
            pool.append({"id": f"{prefix}-{PROFILE['id']}-{sim.seed}-{day}-{s}", "day": day, "npc": int(s),
                         "card": {"facts": facts, "persona": template_persona(facts)},
                         "npc_x": npc_numbers(sh, s), "since": int(day - last) if last >= 0 else None,
                         "usual_items": round(float(sh.basket_mean[s]), 1),
                         "budget": round(float(sh.weekly_budget[s]) * max((day - last) if last >= 0 else 7, 7) / 7, 2)
                         if sh.weekly_budget is not None else None,
                         "candidates": cands})
    return pool


def save_scenarios(pool, path, say=print):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for t in pool:
            f.write(json.dumps(t) + "\n")
    say(f"  {len(pool):,} scenarios ({sum(len(t['candidates']) for t in pool):,} candidate products) -> {path}")
    return pool


def load_pool(path=POOL_FILE, say=print):
    path = Path(path)
    if not path.exists():
        return build_pool(path, say=say)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# ================================================================ the prompt
def line(n, c):
    deal = f" (deal -{c['off']}%)" if c["deal"] and c["off"] else (" (deal)" if c["deal"] else "")
    have = f"bought {c['times']}x, " + ("none left at home" if c["ran_out"] else f"{c['days_left']} days left at home") \
        if c["times"] else "never bought"
    return f"{n}) {c['name']} - ${c['price']:.2f}{deal} | {have}"


def scenario_block(k, t):
    f, persona = t["card"]["facts"], t["card"]["persona"]
    since = f"last trip {t['since']} days ago" if t["since"] is not None else "first trip"
    budget = f"food budget for this trip about ${t['budget']:.0f}" if t["budget"] else "no fixed budget"
    head = (f"S{k} {persona['bio']} Style: {persona['shopping_style']}. Usually about {t['usual_items']:.0f} items a "
            f"trip; {since}; {budget}.")
    return "\n".join([head] + [line(n, c) for n, c in enumerate(t["candidates"], 1)])


def request_text(group):
    blocks = "\n\n".join(scenario_block(k, t) for k, t in enumerate(group, 1))
    return (f"{blocks}\n\nFor EACH shopper, write today's shopping list from their numbered products: what they buy "
            f"and how many. Reply ONE JSON list, one object per product bought: "
            f'[{{"s": <shopper number 1-{len(group)}>, "n": <product number>, "q": <how many, 1-6>}}]. '
            f"Products not bought: leave out.")


def parse_reply(text, group):
    """{shopper k: {product: qty}} for valid answers; a shopper with no line at all is missing."""
    try:
        rows = llm.json_from(text)
    except ValueError:
        return {}
    out = {}
    first = {str(t["card"]["facts"]["name"]).split()[0].lower(): k for k, t in enumerate(group, 1)}
    for r in rows if isinstance(rows, list) else []:
        if not isinstance(r, dict):
            continue
        try:                                                    # "S1", "s1", 1 or the first name "Alice"
            who = str(r.get("s", r.get("shopper", ""))).strip()
            k = first[who.split()[0].lower()] if who and who.split()[0].lower() in first else int(re.sub(r"\D", "", who))
            n = int(re.sub(r"\D", "", str(r.get("n", r.get("product", "")))))
            q = int(round(float(r.get("q", r.get("qty", 1)))))
        except (TypeError, ValueError):
            continue
        if 1 <= k <= len(group) and 1 <= n <= len(group[k - 1]["candidates"]):
            out.setdefault(k, {})[group[k - 1]["candidates"][n - 1]["product"]] = int(np.clip(q, 1, 6))
    return out


# ================================================================ labels
class ListLabels:
    def __init__(self, path=LABELS_FILE):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS lists (id TEXT, teacher TEXT, repeat INTEGER, valid INTEGER, "
                        "data TEXT, at TEXT, PRIMARY KEY (id, teacher, repeat))")

    def add(self, rows):
        self.db.executemany("INSERT OR REPLACE INTO lists VALUES (?, ?, ?, ?, ?, ?)",
                            [(r["id"], r["teacher"], r.get("repeat", 0), int(r["valid"]), json.dumps(r),
                              datetime.now(timezone.utc).isoformat(timespec="seconds")) for r in rows])
        self.db.commit()

    def rows(self, teacher=None, repeat=0, valid_only=True):
        sql, args = "SELECT data FROM lists WHERE repeat = ?", [repeat]
        if teacher:
            sql, args = sql + " AND teacher = ?", args + [teacher]
        if valid_only:
            sql += " AND valid = 1"
        return [json.loads(r[0]) for r in self.db.execute(sql, args)]

    def done(self, teacher, repeat=0):
        return {r[0] for r in self.db.execute("SELECT id FROM lists WHERE teacher = ? AND repeat = ? AND valid = 1",
                                              (teacher, repeat))}

    def counts(self):
        return self.db.execute("SELECT teacher, repeat, valid, COUNT(*) FROM lists GROUP BY teacher, repeat, valid").fetchall()


def label_row(t, chosen, teacher, version=PROMPT_VERSION):
    return {"id": t["id"], "teacher": teacher, "valid": chosen is not None, "prompt_version": version,
            "list": {str(k): v for k, v in (chosen or {}).items()}}


# ================================================================ teachers
class FakeListTeacher:
    """No LLM: fixed rules (tests, dry runs). Buys what ran out and usual items with 2+ days... and one deal."""
    name = "fake/rules-v1"

    def plan(self, group):
        out = {}
        for k, t in enumerate(group, 1):
            pick = {c["product"]: 1 for c in t["candidates"] if c["ran_out"]}
            for c in t["candidates"]:
                if c["deal"] and len(pick) < t["usual_items"]:
                    pick[c["product"]] = 2
                    break
            out[k] = pick
        return out


def room_today(provider, tokens, share=BUDGET_SHARE):
    lim = llm.limits(provider)
    used_req, used_tok = llm.PACER.used_today(provider)
    if lim["rpd"] is not None and used_req + 1 > share * lim["rpd"]:
        return False, f"{used_req:,} of {lim['rpd']:,} requests used today ({share:.0%} stop)"
    if lim["tpd"] is not None and used_tok + tokens > share * lim["tpd"]:
        return False, f"{used_tok:,} of {lim['tpd']:,} tokens used today ({share:.0%} stop)"
    return True, ""


def collect(target=500, provider="gemini", model=None, labels=None, pool=None, say=print, teacher=None,
            share=BUDGET_SHARE):
    """Ask a teacher for lists until `target` valid lists from it (paced; stops at 80% of the free day)."""
    labels = labels or ListLabels()
    pool = pool if pool is not None else load_pool(say=say)
    if teacher is None and provider == "fake":
        teacher = FakeListTeacher()
    model = model or (None if provider == "fake" else llm.PROVIDERS[provider]["model"])
    name = teacher.name if teacher is not None else f"{provider}/{model}"
    done = labels.done(name)
    todo = [t for t in pool if t["id"] not in done]
    say(f"List teacher {name}: {len(done):,} lists so far, target {target:,}, {len(todo):,} scenarios left")
    got, unreadable = len(done), 0
    for start in range(0, len(todo), PER_REQUEST):
        if got >= target:
            break
        group = todo[start:start + PER_REQUEST]
        if teacher is not None:
            answers = teacher.plan(group)
        else:
            text = request_text(group)
            ok, why = room_today(provider, llm.estimate_tokens([{"content": SYSTEM}, {"content": text}], 900), share)
            if not ok:
                say(f"  stopped: {why} - run again tomorrow")
                break
            try:
                reply = llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
                                 provider=provider, model=model, temperature=0.4, reasoning="low")
            except Exception as error:
                say(f"  stopped: {str(error).splitlines()[0][:140]}")
                break
            answers = parse_reply(reply, group)
            if not answers:                                     # nothing readable: keep it, never waste quota
                unreadable += 1
                with open(LIST_DIR / "unreadable_replies.txt", "a", encoding="utf-8") as f:
                    f.write(f"--- {name} {datetime.now(timezone.utc).isoformat(timespec='seconds')} "
                            f"ids={','.join(t['id'] for t in group)}\n{reply}\n")
                say(f"  could not read a reply ({unreadable} in a row) - saved in unreadable_replies.txt")
                if unreadable >= 3:
                    say("  stopped: 3 unreadable replies in a row - the reply format needs a fix (no more quota used)")
                    break
                continue
            unreadable = 0
        rows = [label_row(t, answers.get(k), name) for k, t in enumerate(group, 1)]
        rows = [r for r in rows if r["valid"]]                  # a missing shopper is asked again later
        labels.add(rows)
        got += len(rows)
        if teacher is None and (start // PER_REQUEST) % 10 == 0:
            say(f"  {got:,} lists ({llm.PACER.used_today(provider)[1]:,} tokens today)")
    say(f"List teacher {name}: {got:,} valid lists")
    return got


# ================================================================ paste packs (chat AIs without a free API)
def paste_pack(n=200, folder=PASTE_DIR, labels=None, pool=None, say=print):
    labels = labels or ListLabels()
    pool = pool if pool is not None else load_pool(say=say)
    taken = {r[0] for r in labels.db.execute("SELECT DISTINCT id FROM lists WHERE valid = 1")}
    todo = [t for t in reversed(pool) if t["id"] not in taken][:n]       # from the END: API collects start first
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for old in list(folder.glob("message_*.txt")) + list(folder.glob("reply_*.txt")):
        old.unlink()
    pack = []
    for m, start in enumerate(range(0, len(todo), PER_REQUEST), 1):
        group = todo[start:start + PER_REQUEST]
        (folder / f"message_{m:02d}.txt").write_text(SYSTEM + "\n\n" + request_text(group), encoding="utf-8")
        (folder / f"reply_{m:02d}.txt").write_text("", encoding="utf-8")
        pack.append({"message": m, "ids": [t["id"] for t in group]})
    (folder / "pack.json").write_text(json.dumps({"prompt_version": PROMPT_VERSION, "messages": pack}), encoding="utf-8")
    say(f"Wrote {len(pack)} messages ({len(todo)} scenarios) -> {folder}")
    return len(pack)


def paste_read(teacher, folder=PASTE_DIR, labels=None, pool=None, say=print):
    if not teacher or "/" not in teacher:
        raise SystemExit("Say which AI answered: --teacher chatgpt/<model>")
    labels = labels or ListLabels()
    by_id = {t["id"]: t for t in (pool if pool is not None else load_pool(say=say))}
    folder = Path(folder)
    meta = json.loads((folder / "pack.json").read_text(encoding="utf-8"))
    rows, missing, unread = [], [], 0
    for msg in meta["messages"]:
        reply = folder / f"reply_{msg['message']:02d}.txt"
        text = reply.read_text(encoding="utf-8-sig").strip() if reply.exists() else ""
        if not text:
            missing.append(msg["message"])
            continue
        group = [by_id[i] for i in msg["ids"]]
        answers = parse_reply(text, group)
        for k, t in enumerate(group, 1):
            if k in answers:
                rows.append(label_row(t, answers[k], teacher, meta.get("prompt_version", 1)))
            else:
                unread += 1
    labels.add(rows)
    say(f"{teacher}: {len(rows)} lists read" + (f", {unread} shoppers without a readable list" if unread else "")
        + (f"; no reply yet in {len(missing)} message(s)" if missing else ""))
    return len(rows), missing


def status(labels=None, say=print):
    labels = labels or ListLabels()
    say("List labels (data/teacher/lists/labels.sqlite):")
    for teacher, repeat, valid, n in labels.counts():
        say(f"  {teacher} | {'again' if repeat else 'list'} | {'valid' if valid else 'invalid'}: {n:,}")
    for p in ("groq", "gemini"):
        req, tok = llm.PACER.used_today(p)
        say(f"  {p} today: {req:,} requests, {tok:,} tokens")
