"""RETINA personas: a short, human profile for each NPC, written by an LLM from the NPC's FACTS.

The facts come from the simulation, never from the LLM: household size, children, income fifth,
weekly food budget, how often and how much they shop, how price-sensitive they are and the
products they buy again and again. Python picks a (made-up) first name and an age that fits
the household. The LLM only turns those facts into words - a short bio, a shopping style, what
they care about and a typical line they would say. Every answer is checked against the facts;
anything that contradicts them (children that do not exist, living alone in a family of four)
is thrown away and a plain template is used instead.

Personas are SAVED (data/personas/<store>/personas.json) with a fingerprint of the facts, the
prompt version and the model - so the next run reuses them: same NPCs, same words, no new LLM
calls. Today personas only DESCRIBE the NPCs; the focus group (next step) will let them talk.

Run:  python -m retina.engine.personas --n 20            20 varied NPCs (LLM: Groq, or $env:RETINA_LLM)
      python -m retina.engine.personas --n 20 --offline  templates only, no LLM, no key needed
      python -m retina.engine.personas --show 7          print NPC 7's persona
      python -m retina.engine.personas --all             every NPC (slow on a free tier)
      python -m retina.engine.personas --n 20 --provider ollama --model qwen2.5:7b   free, on your own PC
Keys come from environment variables only (GROQ_API_KEY / GEMINI_API_KEY) - never in code.
Names are invented and do not describe real people. Text is SIMULATED.
"""

import argparse
import hashlib
import json
import re
import time

import numpy as np
import pandas as pd

from .config import ROOT, PROFILE, stream

PROMPT_VERSION = 2                      # change it when the prompt changes: personas are rewritten
BATCH = 8                               # NPCs per LLM request (fewer requests on free tiers)
STREAM_PERSONA = 8                      # random stream for names and ages
PERSONA_DIR = ROOT / "data" / "personas" / PROFILE["id"]
PERSONA_FILE = PERSONA_DIR / "personas.json"
INCOME = ["lower income", "lower-middle income", "middle income", "upper-middle income", "higher income"]
SENSITIVITY = ["not very price-sensitive", "somewhat price-sensitive", "very price-sensitive"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
FIRST_NAMES = """Maria James Aisha Daniel Sofia Michael Priya David Elena Robert Grace Carlos Hannah Kevin
Fatima Brian Chloe Marcus Leah Anthony Mei Jason Olivia Andre Nadia Tyler Rosa Ethan Amara Luis Julia Omar
Emma Samuel Ana Derek Zoe Victor Lily Nathan Imani Diego Rachel Kwame Sara Patrick Yuki Jordan Ines Caleb
Naomi Hector Ruth Raj Claire Dante Vera Malik Erin Tomas Hope Felix Lena Rashid Megan Ivan Joy Mateo Alice
Darius Kim Theo Nora Ali Paige Hugo Rita Simon Tara Kofi Laura Owen Dina Ravi Beth Jamal Cora Leo""".split()
LAST_INITIALS = "ABCDEFGHJKLMNPRSTVW"


# ============================================================= facts
def npc_facts(sim, s):
    """Everything the persona is built on - all from the simulation."""
    sh, p = sim.shoppers, sim.store.products
    rng = stream(STREAM_PERSONA, s)
    size, kids = int(sh.household_size[s]), int(sh.kids[s])
    if kids:                                                    # ESTIMATED ages that fit the home
        age = int(rng.integers(27, 49))
    elif size == 1:
        age = int(rng.integers(21, 82))
    elif size == 2:
        age = int(rng.integers(23, 80))
    else:
        age = int(rng.integers(35, 70))                         # adult children, relatives, roommates
    counts, fav = sh.counts[s], sh.favorites[s]
    last_seen = {x: i for i, x in enumerate(fav)}             # later = bought more recently
    habits = sorted((x for x in counts if counts[x] >= 2), key=lambda x: (-counts[x], -last_seen[x]))[:5]
    recent = list(dict.fromkeys(reversed(fav)))[:5]
    depts = pd.Series([p["department"].iat[x] for x in sh.favorites[s]]).value_counts(normalize=True)
    return {
        "shopper_id": int(s),
        "name": f"{FIRST_NAMES[int(rng.integers(len(FIRST_NAMES)))]} {LAST_INITIALS[int(rng.integers(len(LAST_INITIALS)))]}.",
        "age": age,
        "household_size": size,
        "kids": kids,
        "income": INCOME[int(sh.income_quintile[s])],
        "weekly_food_budget": round(float(sh.weekly_budget[s])),
        "items_per_trip": round(float(sh.basket_mean[s]), 1),
        "days_between_trips": round(float(sh.gap_mean[s]), 1),
        "repeat_buying": f"{sh.reorder_prob[s]:.0%}",
        "price_sensitivity": SENSITIVITY[int(sh.sensitivity_group[s])],
        "favorite_day": WEEKDAYS[int(sh.favorite_weekday[s])],
        "habit_products": [f"{clean(p['product_name'].iat[x])} ({p['department'].iat[x]})" for x in habits],
        "recent_products": [f"{clean(p['product_name'].iat[x])} ({p['department'].iat[x]})" for x in recent],
        "top_departments": [f"{d} {v:.0%}" for d, v in depts.head(3).items()],
    }


def clean(name):
    return str(name)[:40].strip().rstrip(".,")


def fingerprint(facts, model):
    raw = json.dumps({"facts": facts, "prompt": PROMPT_VERSION, "model": model}, sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def household_phrase(f):
    if f["household_size"] == 1:
        return "lives alone"
    kids = f"{f['kids']} child{'ren' if f['kids'] > 1 else ''}" if f["kids"] else ""
    adults = f["household_size"] - f["kids"]
    if kids:
        return f"shops for a home of {f['household_size']} ({adults} adult{'s' if adults > 1 else ''}, {kids})"
    return f"shops for a home of {f['household_size']} adults"


# ========================================================== template
def template_persona(f):
    """No LLM: a plain, fact-only persona."""
    style = {"very price-sensitive": "Deal hunter - checks prices, stocks up on offers",
             "somewhat price-sensitive": "Steady regular - mostly the usual items, notices big deals",
             "not very price-sensitive": "Convenience first - buys what they like, rarely compares prices"}
    trip = f"shops about every {f['days_between_trips']:.0f} days"
    cares = ["price" if "very" in f["price_sensitivity"] and "not" not in f["price_sensitivity"] else "favourite brands",
             f["top_departments"][0].rsplit(" ", 1)[0] if f["top_departments"] else "groceries",
             "kids' food" if f["kids"] else ("quick meals" if f["household_size"] == 1 else "family meals")]
    return {"bio": f"{f['name']}, {f['age']}, {household_phrase(f)}. {f['income'].capitalize()} home with about "
                   f"${f['weekly_food_budget']} a week for food; {trip}, around {f['items_per_trip']:.0f} items a trip.",
            "shopping_style": style[f["price_sensitivity"]],
            "cares_about": cares,
            "quote": (f"I always grab {f['habit_products'][0].rsplit(' (', 1)[0]}." if f["habit_products"] else
                      "Let's see what's on offer today." if "very" in f["price_sensitivity"] else
                      "Just a few things today.")}


# =============================================================== LLM
SYSTEM = ("You write short, realistic shopper personas for a SIMULATED supermarket used to generate synthetic "
          "retail data. Use ONLY the facts given - do not add children, partners, jobs, health conditions or "
          "money details that contradict them, and do not mention real people or brands that are not in the "
          "facts. Product names can be short or odd: the department in brackets says what the product is - do not "
          "guess beyond it. Describe how often they shop using days_between_trips (about 7 = weekly, about 14 = "
          "every two weeks, about 30 = monthly). Keep it neutral and respectful; no stereotypes. Reply with JSON only.")


def llm_prompt(batch):
    return ("For each NPC below, return a JSON list with one object per NPC, in the same order:\n"
            '{"shopper_id": <same id>, "bio": "<2 sentences, max 280 characters, uses the name and age>", '
            '"shopping_style": "<max 12 words>", "cares_about": ["<3 short items>"], '
            '"quote": "<one short sentence they might say in the store>"}\n\n'
            "NPC facts:\n" + json.dumps(batch, indent=1))


def parse_reply(text):
    """The JSON list inside the reply (models sometimes add ``` fences or a sentence)."""
    from .llm import json_from
    return json_from(text)


def check(f, p):
    """Reject anything that is missing or contradicts the facts. Returns a reason or None."""
    for key in ("bio", "shopping_style", "cares_about", "quote"):
        if not p.get(key):
            return f"missing {key}"
    bio = (p["bio"] + " " + p["quote"]).lower()
    if len(p["bio"]) > 400:
        return "bio too long"
    if f["kids"] == 0 and re.search(r"\b(kids?|child|children|son|daughter|baby|toddler)\b", bio):
        return "mentions children the NPC does not have"
    if f["household_size"] > 1 and re.search(r"\b(lives alone|living alone|on (his|her|their) own)\b", bio):
        return "says lives alone"
    if f["household_size"] == 1 and re.search(r"\b(wife|husband|partner|family of|kids?|children)\b", bio):
        return "adds people to a one-person home"
    if f["name"].split()[0].lower() not in bio:
        return "does not use the NPC's name"
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}
    num = r"(\d+|one|two|three|four|five|six|seven|eight)"
    for m in re.finditer(rf"\bwith {num} others?\b", bio):          # "lives with two others"
        n = words.get(m.group(1), None) or int(m.group(1))
        if n != f["household_size"] - 1:
            return f"says lives with {n} others but the home has {f['household_size']} people"
    for m in re.finditer(rf"\b(?:family|home|household) of {num}\b|\b{num}-person\b", bio):
        g = m.group(1) or m.group(2)
        n = words.get(g, None) or int(g)
        if n != f["household_size"]:
            return f"says a home of {n} but it has {f['household_size']} people"
    days = f["days_between_trips"]
    bio = bio + " " + str(p["shopping_style"]).lower()           # how often: style text counts too
    if re.search(r"\b(bi-?weekly|fortnightly|every (two|2) weeks)\b", bio) and not 10 <= days <= 18:
        return "says every two weeks but shops at another pace"
    if re.search(r"\b(monthly|each month|every month|once a month)\b", bio) and days < 22:
        return "says monthly but shops more often"
    if re.search(r"\b(weekly|each week|every week|once a week)\b", bio) and days > 11:
        return "says weekly but shops less often"
    if re.search(r"\b(daily|every day)\b", bio) and days > 3:
        return "says daily"
    if not isinstance(p["cares_about"], list):
        return "cares_about is not a list"
    return None


def tidy(text):
    """Small clean-ups: a missing space after a full stop, odd dashes."""
    text = re.sub(r"\.(?=[A-Z])", ". ", str(text))
    text = re.sub(r",(?=[A-Za-z])", ", ", text)                # "familiar,fits" -> "familiar, fits"
    return text.replace("\u2011", "-").replace("\u2013", "-").replace("\u2014", "-").strip()


def ask_llm(batch, provider, model):
    from .llm import chat
    text = chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": llm_prompt(batch)}],
                provider=provider, model=model, temperature=0.7, reasoning="low")
    out = {}
    for p in parse_reply(text):
        if isinstance(p, dict):
            for key in ("bio", "shopping_style", "quote"):
                if key in p:
                    p[key] = tidy(p[key])
            if isinstance(p.get("cares_about"), list):
                p["cares_about"] = [tidy(c) for c in p["cares_about"]]
            out[int(p.get("shopper_id", -1))] = p
    return out


# ============================================================ build
def load_saved():
    if PERSONA_FILE.exists():
        return json.loads(PERSONA_FILE.read_text(encoding="utf-8"))
    return {}


def save(saved):
    PERSONA_DIR.mkdir(parents=True, exist_ok=True)
    PERSONA_FILE.write_text(json.dumps(saved, indent=1, ensure_ascii=False), encoding="utf-8")
    rows = []
    for v in saved.values():
        facts = {k: ("; ".join(x) if isinstance(x, list) else x) for k, x in v["facts"].items()}
        rows.append({**facts, **{k: v[k] for k in ("bio", "shopping_style", "quote", "source")},
                     "cares_about": "; ".join(v["cares_about"]), "data_label": "SIMULATED"})
    pd.DataFrame(rows).sort_values("shopper_id").to_csv(PERSONA_DIR / "personas.csv", index=False)


def varied_sample(sim, n):
    """n NPCs spread across household sizes, incomes and price sensitivity (for a focus group)."""
    sh = sim.shoppers
    order = np.lexsort((sh.shopper_id, sh.sensitivity_group, sh.income_quintile, sh.household_size))
    base = order[np.linspace(0, len(order) - 1, max(n, 20)).round().astype(int)]
    if n < 20:                                                # a smaller group is part of the 20,
        base = base[np.linspace(0, 19, n).round().astype(int)]   # so their personas are reused
    return [int(x) for x in base]


def build(sim, ids, provider=None, model=None, offline=False, say=print):
    """Personas for these NPCs: reuse saved ones, write new ones with the LLM (or a template)."""
    from . import llm
    provider = provider or llm.DEFAULT_PROVIDER
    if offline:
        model = "template"
    elif provider == "auto":
        model = model or "auto"                               # whoever in the chain answers
    else:
        model = model or llm.PROVIDERS[provider]["model"]
    saved = load_saved()
    facts = {s: npc_facts(sim, s) for s in ids}

    def still_good(s):                                        # saved, same facts/prompt/model, and still
        old = saved.get(str(s), {})                           # passes today's fact check
        src = old.get("source", "")
        made_by = "template" if src == "template" else src.split("/", 1)[-1]
        if not old or old.get("key") != fingerprint(facts[s], made_by):
            return False
        if src == "template":
            return model == "template"                        # online runs upgrade templates
        return (model in ("auto", "template", made_by)) and check(facts[s], old) is None   # offline keeps LLM ones
    todo = [s for s in ids if not still_good(s)]
    say(f"Personas: {len(ids)} NPCs | {len(ids) - len(todo)} already saved | {len(todo)} to write "
        f"({'template' if offline else provider + ' / ' + model})")
    stats = {"llm": 0, "template": 0, "rejected": 0}
    lost = 0                                                  # batches in a row with no connection
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        answers, why = {}, None
        if not offline:
            try:
                answers = ask_llm([facts[s] for s in batch], provider, model)
            except Exception as error:                        # no key, no internet, bad JSON ...
                why = str(error)[:160]
                lost = lost + 1 if "Could not reach" in why else 0
                if "No key" in why or "No LLM provider" in why or lost >= 2:
                    offline = True                            # no point trying the other batches
                    say(f"  {why}\n  -> templates for the rest (not saved as final: run again later to upgrade them)")
                else:
                    say(f"  LLM problem ({why}) - templates for this batch, trying the next one")
            else:
                lost = 0
        for s in batch:
            f, p = facts[s], answers.get(s)
            reason = check(f, p) if p else (why or "no answer")
            if p and reason is None:
                used = f"{llm.LAST_USED['provider']}/{llm.LAST_USED['model']}" if provider == "auto" else f"{provider}/{model}"
                persona, source = p, f"llm:{used}"
                stats["llm"] += 1
            else:
                if p:
                    stats["rejected"] += 1
                    say(f"  NPC {s}: LLM persona rejected ({reason}) - template used")
                persona, source = template_persona(f), "template"
                stats["template"] += 1
            saved[str(s)] = {"facts": f, "bio": persona["bio"], "shopping_style": persona["shopping_style"],
                             "cares_about": [str(c) for c in persona["cares_about"]][:4],
                             "quote": persona["quote"], "source": source, "prompt_version": PROMPT_VERSION,
                             "data_label": "SIMULATED - words written from simulated NPC facts; invented name",
                             "key": (fingerprint(f, source.split("/", 1)[-1]) if source != "template" else
                                     fingerprint(f, "template") if model == "template" else None)}
        save(saved)                                           # save as we go (free tiers can stop)
        if not offline and i + BATCH < len(todo):
            time.sleep(1.0)
    if todo:
        say(f"  written: {stats['llm']} by the LLM, {stats['template']} templates "
            f"({stats['rejected']} LLM answers rejected by the fact check)")
    return {s: saved[str(s)] for s in ids}


def show(p):
    f = p["facts"]
    print(f"\nNPC {f['shopper_id']} - {f['name']}, {f['age']} | {household_phrase(f)} | {f['income']} | "
          f"${f['weekly_food_budget']}/week | {f['price_sensitivity']}")
    habits = ", ".join(f["habit_products"][:3]) or "none yet (recent: " + ", ".join(f["recent_products"][:2]) + ")"
    print(f"  facts: ~{f['items_per_trip']:.0f} items every ~{f['days_between_trips']:.0f} days, "
          f"{f['repeat_buying']} repeats, likes {f['favorite_day']}s | habits: {habits}")
    print(f"  bio:    {p['bio']}")
    print(f"  style:  {p['shopping_style']} | cares about: {', '.join(p['cares_about'])}")
    print(f"  says:   \"{p['quote']}\"   [{p['source']}]")


def main():
    ap = argparse.ArgumentParser(description="Write personas for RETINA NPCs.")
    ap.add_argument("--n", type=int, default=20, help="how many varied NPCs (default 20)")
    ap.add_argument("--all", action="store_true", help="every NPC in the pool")
    ap.add_argument("--show", type=int, help="print one NPC's persona (writes it if needed)")
    ap.add_argument("--offline", action="store_true", help="templates only, no LLM")
    ap.add_argument("--provider", help="auto (Groq -> Gemini -> Ollama, default) | groq | gemini | hf | ollama")
    ap.add_argument("--model", help="model name for that provider, e.g. qwen2.5:7b for ollama")
    args = ap.parse_args()
    from .engine import Simulation
    print("Building the store and its NPCs (same seed = same NPCs)...")
    sim = Simulation(verbose=False)
    if args.show is not None:
        ids = [args.show]
    elif args.all:
        ids = [int(s) for s in sim.shoppers.shopper_id]
    else:
        ids = varied_sample(sim, args.n)
    personas = build(sim, ids, args.provider, args.model, offline=args.offline)
    for s in ids[::max(1, len(ids) // 6)][:6]:              # a spread of households
        show(personas[s])
    if len(ids) > 6:
        print(f"\n... and {len(ids) - 6} more in {(PERSONA_DIR / 'personas.csv').relative_to(ROOT)}")
    print(f"\nSaved to {PERSONA_FILE.relative_to(ROOT)} - the next run reuses them (no new LLM calls).")


if __name__ == "__main__":
    main()
