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

PROMPT_VERSION = 2            # part of the cache key: change it whenever the prompt changes
MAX_REQUEST_TOKENS = 6_000    # one request stays under Groq's free 8,000 tokens a minute (~200 per item)
REASON_TYPES = ("habit", "price", "deal", "quality", "need", "try_new", "out_of_stock", "budget", "other")
LETTERS = "ABCDE"

SYSTEM = ("You role-play SEVERAL different shoppers in a SIMULATED supermarket used to make synthetic retail "
          "data. Treat each shopper separately and stay in character: their facts, habits, memory, budget. "
          "For each listed item pick exactly ONE offered product by its letter, or SKIP if none fits today. "
          "Real shoppers mostly stick to products they know, switch for a clearly better price or deal, "
          "and avoid SOLD OUT products (never pick one). Two items from the same aisle are different things on "
          "the list: pick different products for them. Reply with JSON only.")


def option_line(o):
    deal = f", DEAL {o['off']:.0f}% off" if o["deal"] and o["off"] >= 1 else (", deal" if o["deal"] else "")
    seen = f", you bought it {o['times']}x" if o["times"] else ", never bought"
    paid = f" (you paid ${o['last_price']:.2f} last time)" if o.get("last_price") else ""
    gone = ", was sold out last time" if o.get("sold_out") else ""
    stock = ", SOLD OUT" if not o["in_stock"] else ""
    return f"   {o['letter']}) {o['name']} - ${o['price']:.2f}{paid}{deal}{seen}{gone}{stock}"


def npc_block(n, card, memory_lines, trip, items):
    """The situation of one NPC, as text. Also the heart of the cache key."""
    f, p = card["facts"], card["persona"]
    lines = [f"Shopper {n}: {f['name']}, {f['age']}, {household_phrase(f)}; {f['income']}; "
             f"about ${f['weekly_food_budget']}/week for food; {f['price_sensitivity']}.",
             f"  Style: {p['shopping_style']}."]
    if memory_lines:
        lines.append("  Remembers: " + "; ".join(memory_lines) + ".")
    money = f"about ${trip['budget']:.0f} to spend" if trip["budget"] < 1e6 else "no fixed budget"
    lines.append(f"  Today: a {trip['mission'].replace('_', '-')} trip, {money}.")
    for i, it in enumerate(items, 1):
        lines.append(f"  Item {i} - {it['aisle']} ({it['why_here']}):")
        lines += [option_line(o) for o in it["options"]]
    return "\n".join(lines)


def batch_prompt(blocks):
    return ("\n\n".join(blocks) + "\n\nReply as ONE JSON list, one object per item of every shopper:\n"
            '[{"shopper": 1, "item": 1, "choice": "A", "reason_type": "habit", "why": "<max 12 words>"}, ...]\n'
            f"reason_type is one of: {', '.join(REASON_TYPES)}.")


def request_tokens(blocks):
    """Tokens one request would use: the prompt plus the reply (~35 tokens per item)."""
    items = sum(b.count("\n  Item ") for b in blocks)
    return llm.estimate_tokens([{"content": SYSTEM}, {"content": batch_prompt(blocks)}], reply_tokens=35 * items)


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
        try:
            n, i = int(r.get("shopper")), int(r.get("item"))
        except (TypeError, ValueError):
            continue
        if not (1 <= n <= len(batch) and 1 <= i <= len(batch[n - 1]["items"])):
            continue
        choice = str(r.get("choice", "")).strip().upper()[:4]
        options = {o["letter"]: o for o in batch[n - 1]["items"][i - 1]["options"]}
        if choice != "SKIP" and (choice not in options or not options[choice]["in_stock"]):
            continue                                          # not offered, or sold out
        kind = str(r.get("reason_type", "other")).strip().lower()
        out[(n, i)] = {"choice": choice,
                       "product": -1 if choice == "SKIP" else options[choice]["product"],
                       "why_type": kind if kind in REASON_TYPES else "other",
                       "why": " ".join(str(r.get("why", "")).split())[:120]}
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
        """batch = [{"block": text, "items": [...]}] -> ({(n, i): answer}, provider, model, problem)."""
        prompt = batch_prompt([b["block"] for b in batch])
        try:
            text = llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
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
