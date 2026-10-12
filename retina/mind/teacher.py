"""The LLM teacher: one request decides the shelf choices of up to 5 NPCs (a batch).

Each NPC in the batch gets a short block: who they are (card), what they remember, today's trip,
and for every item the router sent here, 2-5 REAL products from that aisle with today's price,
deal, stock and their own history. The LLM answers with one letter (or SKIP), a reason type and a
few words of why. Python then checks every answer (the option exists and is not sold out);
anything missing or broken falls back to the engine's own choice and is counted.

Answers are cached per NPC + day + situation (retina.mind.cache), so a rerun is identical.
FakeTeacher answers with simple fixed rules - for tests and dry runs, never presented as an LLM.
"""

from retina.engine import llm
from retina.engine.personas import household_phrase

PROMPT_VERSION = 3            # part of the cache key: change it whenever the prompt changes
# v2 = the first prompt (long option lines); v3 (13 Oct 2026) = compact: short names, symbols, short reply keys.
# Labels keep the version they were made with; a repeat / comparison of old labels uses THEIR version.
MAX_REQUEST_TOKENS = 4_500    # ESTIMATED tokens per request. Real use runs ~1.25x the estimate (gpt-oss
                              # reasoning; first real collect, 12 Oct 2026: 38 items ~7,500 tokens) and Groq's
                              # free tier refuses a request above 8,000 tokens a minute (error 413)
REASON_TYPES = ("habit", "price", "deal", "quality", "need", "try_new", "out_of_stock", "budget", "other")
LETTERS = "ABCDE"

SYSTEM = ("You role-play SEVERAL different shoppers in a SIMULATED supermarket used to make synthetic retail "
          "data. Treat each shopper separately and stay in character: their facts, habits, memory, budget. "
          "For each listed item pick exactly ONE offered product by its letter, or SKIP if none fits today. "
          "Real shoppers mostly stick to products they know, switch for a clearly better price or deal, "
          "and avoid SOLD OUT products (never pick one). Two items from the same aisle are different things on "
          "the list: pick different products for them. Reply with JSON only.")


SYSTEM_V3 = ("Role-play each shopper below in a SIMULATED store (synthetic data); stay in character. Per item pick "
             "ONE option letter, or SKIP if none fits today. Real shoppers mostly keep to products they know and "
             "switch for a clearly better price or deal. Never pick an OUT option. Two items from the same aisle "
             "are different things: pick different products. JSON only.")
LEGEND_V3 = "Legend: -N% = deal, xN = you bought it N times, paid$ = you paid this last time, wasOUT = sold out last time, OUT = sold out now."
WHY_SHORT = {"ran out at home": "ran out", "you buy this regularly": "regular", "new on your list today": "new",
             "goes with another item on your list": "goes with list"}
REPLY_TOKENS = {2: 35, 3: 16}          # ESTIMATED reply tokens per item


def system_for(version=PROMPT_VERSION):
    return SYSTEM if version <= 2 else SYSTEM_V3


def option_line(o, version=PROMPT_VERSION):
    if version <= 2:
        deal = f", DEAL {o['off']:.0f}% off" if o["deal"] and o["off"] >= 1 else (", deal" if o["deal"] else "")
        seen = f", you bought it {o['times']}x" if o["times"] else ", never bought"
        paid = f" (you paid ${o['last_price']:.2f} last time)" if o.get("last_price") else ""
        gone = ", was sold out last time" if o.get("sold_out") else ""
        stock = ", SOLD OUT" if not o["in_stock"] else ""
        return f"   {o['letter']}) {o['name']} - ${o['price']:.2f}{paid}{deal}{seen}{gone}{stock}"
    marks = [f"-{o['off']:.0f}%" if o["deal"] and o["off"] >= 1 else ("deal" if o["deal"] else ""),
             f"x{o['times']}" if o["times"] else "",
             f"paid${o['last_price']:.2f}" if o.get("last_price") else "",
             "wasOUT" if o.get("sold_out") else "", "OUT" if not o["in_stock"] else ""]
    name = o["name"] if len(o["name"]) <= 32 else o["name"][:33].rsplit(" ", 1)[0]     # whole words only
    return (f" {o['letter']}) {name.rstrip(' ,-')} ${o['price']:.2f} " + " ".join(m for m in marks if m)).rstrip()


def npc_block(n, card, memory_lines, trip, items, version=PROMPT_VERSION):
    """The situation of one NPC, as text. Also the heart of the cache key."""
    f, p = card["facts"], card["persona"]
    money = trip["budget"] < 1e6
    if version <= 2:
        lines = [f"Shopper {n}: {f['name']}, {f['age']}, {household_phrase(f)}; {f['income']}; "
                 f"about ${f['weekly_food_budget']}/week for food; {f['price_sensitivity']}.",
                 f"  Style: {p['shopping_style']}."]
        if memory_lines:
            lines.append("  Remembers: " + "; ".join(memory_lines) + ".")
        spend = f"about ${trip['budget']:.0f} to spend" if money else "no fixed budget"
        lines.append(f"  Today: a {trip['mission'].replace('_', '-')} trip, {spend}.")
        for i, it in enumerate(items, 1):
            lines.append(f"  Item {i} - {it['aisle']} ({it['why_here']}):")
            lines += [option_line(o, version) for o in it["options"]]
        return "\n".join(lines)
    lines = [f"S{n} {f['name']} {f['age']}, {household_phrase(f)}, {f['income']}, ${f['weekly_food_budget']}/wk food, "
             f"{f['price_sensitivity']}; {p['shopping_style'].split(' - ')[0]}. "
             f"{trip['mission'].replace('_', '-')} trip" + (f", ${trip['budget']:.0f}." if money else ".")]
    if memory_lines:
        lines.append(" Remembers: " + "; ".join(memory_lines) + ".")
    for i, it in enumerate(items, 1):
        lines.append(f"I{i} {it['aisle']} ({WHY_SHORT.get(it['why_here'], it['why_here'])}):")
        lines += [option_line(o, version) for o in it["options"]]
    return "\n".join(lines)


def batch_prompt(blocks, version=PROMPT_VERSION):
    if version <= 2:
        return ("\n\n".join(blocks) + "\n\nReply as ONE JSON list, one object per item of every shopper:\n"
                '[{"shopper": 1, "item": 1, "choice": "A", "reason_type": "habit", "why": "<max 12 words>"}, ...]\n'
                f"reason_type is one of: {', '.join(REASON_TYPES)}.")
    return (LEGEND_V3 + "\n\n" + "\n\n".join(blocks) + "\n\nReply: ONE JSON list, one object per item "
            '(s = shopper, i = item, c = letter or SKIP, r = reason, w = max 6 words):\n'
            '[{"s":1,"i":1,"c":"A","r":"habit","w":"..."}]\n'
            f"r is one of: {', '.join(REASON_TYPES)}.")


def request_tokens(blocks, version=PROMPT_VERSION):
    """Tokens one request would use: the prompt plus the reply (per item, by prompt version)."""
    items = sum(b.count("\n  Item ") + b.count("\nI") for b in blocks)
    return llm.estimate_tokens([{"content": system_for(version)}, {"content": batch_prompt(blocks, version)}],
                               reply_tokens=REPLY_TOKENS.get(version, 35) * items)


def parse_answers(text, batch):
    """{(shopper n, item i): answer} for valid answers only. batch[n-1]["items"][i-1]["options"]."""
    out = {}
    try:
        rows = llm.json_from(text)
    except ValueError:
        return out
    for r in rows if isinstance(rows, list) else []:
        if not isinstance(r, dict):
            continue
        get = lambda long, short, default=None: r.get(long, r.get(short, default))   # v2 or v3 (short) keys
        try:
            n, i = int(get("shopper", "s")), int(get("item", "i"))
        except (TypeError, ValueError):
            continue
        if not (1 <= n <= len(batch) and 1 <= i <= len(batch[n - 1]["items"])):
            continue
        choice = str(get("choice", "c", "")).strip().upper()[:4]
        options = {o["letter"]: o for o in batch[n - 1]["items"][i - 1]["options"]}
        if choice != "SKIP" and (choice not in options or not options[choice]["in_stock"]):
            continue                                          # not offered, or sold out
        kind = str(get("reason_type", "r", "other")).strip().lower()
        out[(n, i)] = {"choice": choice,
                       "product": -1 if choice == "SKIP" else options[choice]["product"],
                       "why_type": kind if kind in REASON_TYPES else "other",
                       "why": " ".join(str(get("why", "w", "")).split())[:120]}
    return out


class LLMTeacher:
    """Asks the real LLM chain (retina.engine.llm: paced, retried, provider fallback)."""
    name = "llm"

    def __init__(self, provider="auto", model=None, say=print):
        self.provider, self.model, self.say = provider, model, say
        self.failures = 0                                     # whole-chain failures in a row
        self.calls = 0

    @property
    def offline(self):
        return self.failures >= 2                             # stop asking; the engine decides

    def ask(self, batch):
        """batch = [{"block": text, "items": [...], "version"?: prompt version}]
        -> ({(n, i): answer}, provider, model, problem). Blocks must be written in that version."""
        version = batch[0].get("version", PROMPT_VERSION)
        prompt = batch_prompt([b["block"] for b in batch], version)
        try:
            text = llm.chat([{"role": "system", "content": system_for(version)}, {"role": "user", "content": prompt}],
                            provider=self.provider, model=self.model, temperature=0.4, reasoning="low")
        except Exception as error:                            # no provider, limits, network...
            self.failures += 1
            problem = str(error).splitlines()[0][:160]
            self.say(f"  (LLM teacher: {problem}" + (" - the engine decides from now on)" if self.offline else ")"))
            return {}, None, None, problem
        self.failures = 0
        self.calls += 1
        return parse_answers(text, batch), llm.LAST_USED["provider"], llm.LAST_USED["model"], None


class FakeTeacher:
    """No LLM: fixed rules, same answer every time. For tests and dry runs (labelled provider 'fake')."""
    name = "fake"
    offline = False

    def __init__(self, say=print):
        self.calls = 0

    def ask(self, batch):
        self.calls += 1
        out = {}
        for n, b in enumerate(batch, 1):
            for i, it in enumerate(b["items"], 1):
                ok = [o for o in it["options"] if o["in_stock"]]
                if not ok:
                    out[(n, i)] = {"choice": "SKIP", "product": -1, "why_type": "out_of_stock", "why": "nothing left"}
                    continue
                best = max(ok, key=lambda o: (o["times"] > 0, o["deal"], -o["price"], o["letter"]))
                kind = "habit" if best["times"] else ("deal" if best["deal"] else "price")
                out[(n, i)] = {"choice": best["letter"], "product": best["product"], "why_type": kind,
                               "why": f"rule: {kind}"}
        return out, "fake", "rules-v1", None
