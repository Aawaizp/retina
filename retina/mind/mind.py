"""The mind of thinking NPCs: decides product + reason at the shelf, one batch of NPCs at a time.

Who thinks:  mix = a fixed PANEL of NPCs (default 50, varied by household, income, price
             sensitivity); everyone else shops with the engine. llm = every NPC (small runs only).
For each item on a thinking NPC's list (the list itself comes from the engine) the ROUTER picks
the cheapest mind that can decide:
  autopilot  routine buy: their usual product, in stock, price as remembered, no deal in sight
             -> the engine's own rules choose (free, instant)
  teacher    something to think about: a new item, the usual is sold out or dearer than
             remembered, a deal next to it, or a random 5% check of routine buys -> the LLM
             (mode llm: every item goes to the teacher)
Python checks every LLM answer (offered, not sold out); at the shelf the budget and the stock
are checked again by the engine. Every decision is logged in the NPC memory (retina.memory).
"""

import numpy as np

from retina.engine.config import ROOT, PROFILE, stream
from retina.engine.personas import npc_facts, template_persona, varied_sample, clean
from retina.memory.store import MemoryStore
from retina.mind.cache import AnswerCache, fingerprint
from retina.mind.teacher import (LLMTeacher, FakeTeacher, npc_block, request_tokens, PROMPT_VERSION, LETTERS,
                                 MAX_REQUEST_TOKENS)

STREAM_MIND_ORDER = 10        # dice: the order options are shown in (no "A is always right" bias)
STREAM_MIND_CHECK = 11        # dice: which routine buys get a sampled LLM check
CHECK_RATE = 0.05             # ESTIMATED: share of routine buys the teacher double-checks
PRICE_UP = 1.05               # dearer than remembered by more than 5% = worth a thought (ESTIMATED)
BATCH = 5                     # NPCs per LLM request
PANEL = 50                    # thinking NPCs in a mix run (decided 12 Oct 2026)
MAX_OPTIONS = 5
CACHE_FILE = ROOT / "data" / "cache" / "mind_cache.sqlite"
WHY_HERE = {"need": "ran out at home", "repeat": "you buy this regularly",
            "new": "new on your list today", "partner": "goes with another item on your list"}


class Mind:
    def __init__(self, sim, mode, run_dir, panel_size=PANEL, provider="auto", model=None,
                 cache_file=CACHE_FILE, say=print):
        if mode not in ("mix", "llm"):
            raise ValueError(f"unknown mind '{mode}'")
        self.sim, self.say = sim, say
        self.memory = MemoryStore(run_dir / "memory.sqlite")
        settings = self.memory.get("settings")
        if settings is None:                                   # a new run: these settings stay with it
            settings = {"mode": mode, "panel_size": int(panel_size), "store": PROFILE["id"], "seed": sim.seed}
            self.memory.set("settings", settings)
        elif settings["mode"] != mode or settings["panel_size"] != panel_size:
            say(f"Note: this run was started with --mind {settings['mode']} --panel {settings['panel_size']};"
                f" those stay (use --reset to change).")
        self.mode, self.panel_size = settings["mode"], settings["panel_size"]
        self.memory.rollback_to(sim.day)                       # forget a half-saved day, if any
        self.cache = AnswerCache(cache_file)
        self.teacher = FakeTeacher(say) if provider == "fake" else LLMTeacher(provider, model, say)
        self.panel = set(self.memory.panel())
        if self.mode == "mix" and not self.panel:              # new run: choose and remember the panel
            for s in varied_sample(sim, self.panel_size):
                self.card(s, sim.day)
            self.panel = set(self.memory.panel())
        self.memory.db.commit()
        self.today = {}
        self.batch = BATCH

    # ---------------------------------------------------------------- who
    def thinks(self, s):
        return self.mode == "llm" or s in self.panel

    def card(self, s, day):
        """The NPC's card: written once (facts on the day they join), read every trip after."""
        card = self.memory.card(s)
        if card is None:
            facts = npc_facts(self.sim, s)
            card = {"facts": facts, "persona": template_persona(facts)}
            self.memory.add_card(s, card, day)
        return card

    # ------------------------------------------------------------ options
    def options(self, s, k, day, wanted, table, prices, promo, regular):
        """2-5 real products from the item's aisle: the engine's pick, their usual ones, a deal,
        the best sellers - with today's price, deal, stock and what this NPC remembers."""
        store, sh, inv = self.sim.store, self.sim.shoppers, self.sim.inv
        aisle = store.aisle_of[wanted]
        cum = table["cum"][aisle]
        if cum is None:
            return aisle, []
        a = store.aisles[aisle]
        share = np.diff(np.r_[0.0, cum])
        ranked = a.product_idx[np.argsort(-share, kind="stable")]
        weight = dict(zip(a.product_idx.tolist(), share.tolist()))         # today, with price effects
        base = dict(zip(a.product_idx.tolist(), (a.pick_prob / a.pick_prob.sum()).tolist()))   # popularity only
        counts = sh.counts[s]
        usual = sorted(sh.by_aisle[s].get(aisle, []), key=lambda p: -counts.get(p, 0))[:2]
        deal = [p for p in ranked[:30] if promo[p]][:1]
        picks = []
        for p in [wanted] + usual + deal + list(ranked[:8]):
            if int(p) not in picks:
                picks.append(int(p))
        picks = picks[:MAX_OPTIONS]
        top = set(ranked[:8].tolist())
        recall = self.memory.recall(s, picks, day)
        order = stream(STREAM_MIND_ORDER, day, s, k).permutation(len(picks))
        names = store.products["product_name"]
        out = []
        for letter, i in zip(LETTERS, order):
            p = picks[i]
            price, reg = round(float(prices[p]), 2), round(float(regular[p]), 2)
            mem = recall.get(p, {})
            out.append({"letter": letter, "product": p, "name": clean(names.iat[p]), "price": price,
                        "regular": reg, "deal": bool(promo[p]),
                        "off": round((1 - price / reg) * 100, 1) if reg > 0 else 0.0,
                        "times": int(counts.get(p, 0)), "in_stock": bool(inv.stock[p] > 0),
                        "usual": p in usual, "top": p in top, "w": round(weight.get(p, 0.0), 6),
                        "pop": round(base.get(p, 0.0), 6),
                        "last_price": mem.get("last_price"), "sold_out": int(mem.get("sold_out", 0))})
        return aisle, out

    # ------------------------------------------------------------- router
    def route(self, s, k, day, wanted, route, options, prices):
        """None = autopilot (routine buy), else the reason the teacher is asked."""
        if self.mode == "llm":
            return "llm_mode"
        if route in ("new", "partner"):
            return "new_choice"
        own = next(o for o in options if o["product"] == wanted)
        if not own["in_stock"]:
            return "usual_sold_out"
        remembered = self.sim.memory[wanted] if self.sim.memory is not None else prices[wanted]
        if prices[wanted] > remembered * PRICE_UP:
            return "price_up"
        if any(o["deal"] and o["in_stock"] and o["product"] != wanted and (o["top"] or o["usual"]) for o in options):
            return "deal_nearby"
        if stream(STREAM_MIND_CHECK, day, s, k).random() < CHECK_RATE:
            return "check"
        return None

    def remembers(self, s, day):
        out = []
        for d, items, spend, top in self.memory.last_trips(s, day):
            out.append(f"{day - d} days ago {items} items for ${spend:.0f}" + (f" ({top})" if top else ""))
        return out

    # ------------------------------------------------------------ decide
    def decide(self, day, trips, ctx):
        """trips = {npc: trip from make_list} -> {npc: (trip, {list position: decision})}.
        Routine items go to autopilot; the rest of the whole batch goes to the teacher in ONE request."""
        sh = self.sim.shoppers
        out, rows, asking = {}, [], []
        for s, trip in trips.items():
            g = int(sh.sensitivity_group[s])
            prices, promo, table = ctx["prices_g"][g], ctx["promo_g"][g], ctx["tables"][g]
            decisions, items = {}, []
            for k, (wanted, route, _, _, _) in enumerate(trip["plan"]):
                aisle, options = self.options(s, k, day, int(wanted), table, prices, promo, ctx["regular"])
                trigger = self.route(s, k, day, int(wanted), route, options, prices) if options else None
                row = {"day": day, "hour": ctx["hour"][s], "npc": int(s), "item": k, "aisle": aisle,
                       "route": route, "trigger": trigger or "routine", "engine_product": int(wanted),
                       "options": [{x: o[x] for x in ("letter", "product", "price", "regular", "deal", "times",
                                                       "in_stock", "usual", "w", "pop")} for o in options],
                       "provider": None, "model": None, "cached": 0, "valid": 1, "problem": None}
                if trigger is None:
                    decisions[k] = {"mind": "autopilot", "product": int(wanted), "why_type": "habit", "why": "usual buy"}
                else:
                    items.append({"k": k, "aisle": aisle, "why_here": WHY_HERE.get(route, route),
                                  "options": options, "row": row})
                rows.append((row, s, k))
            out[s] = (trip, decisions)
            if items:
                card = self.card(s, day)
                memory_lines = self.remembers(s, day)
                situation = npc_block(0, card, memory_lines, trip, items)
                key = fingerprint(PROFILE["id"], self.sim.seed, PROMPT_VERSION, int(s), day, situation)
                asking.append({"s": s, "items": items, "key": key, "card": card, "memory": memory_lines,
                               "trip": trip, "cached": self.cache.get(key)})
        self._teach(asking, out)
        for row, s, k in rows:
            d = out[s][1][k]
            row.update(mind=d["mind"], product=d["product"], why_type=d["why_type"], why=d["why"])
            if d.get("source"):
                row.update(provider=d["source"][0], model=d["source"][1], cached=int(d["source"][2]))
            if d.get("problem"):
                row.update(valid=0, problem=d["problem"])
        self.memory.add_decisions([r for r, _, _ in rows])
        self.today.setdefault("rows", []).extend(r for r, _, _ in rows)
        return out

    def _requests(self, fresh):
        """Pack the NPCs that need an answer into requests that stay under MAX_REQUEST_TOKENS."""
        groups, current = [], []
        for a in fresh:
            trial = current + [a]
            blocks = [npc_block(n, x["card"], x["memory"], x["trip"], x["items"]) for n, x in enumerate(trial, 1)]
            if current and request_tokens(blocks) > MAX_REQUEST_TOKENS:
                groups.append(current)
                current = [a]
            else:
                current = trial
        return groups + ([current] if current else [])

    def _teach(self, asking, out):
        """Cached NPCs: reuse their answers. The rest: as few requests as fit (none if the teacher is offline)."""
        answers, sources, problems = {}, {}, {}
        for group in self._requests([a for a in asking if a["cached"] is None]):
            if self.teacher.offline:
                for a in group:
                    problems[a["s"]] = "teacher offline"
                continue
            batch = [{"block": npc_block(n, a["card"], a["memory"], a["trip"], a["items"]), "items": a["items"]}
                     for n, a in enumerate(group, 1)]
            got, provider, model, problem = self.teacher.ask(batch)
            for n, a in enumerate(group, 1):
                answers[a["s"]] = {i: got.get((n, i)) for i in range(1, len(a["items"]) + 1)}
                sources[a["s"]], problems[a["s"]] = (provider, model, False), problem
                if problem is None:                             # failures are not cached: try again next time
                    self.cache.put(a["key"], {"answers": {str(i): v for i, v in answers[a["s"]].items()},
                                              "provider": provider, "model": model})
        for a in asking:
            if a["cached"] is not None:
                got = {int(i): v for i, v in a["cached"]["answers"].items()}
                src, problem = (a["cached"]["provider"], a["cached"]["model"], True), None
            else:
                got, src, problem = answers.get(a["s"], {}), sources.get(a["s"], (None, None, False)), problems.get(a["s"])
            for i, it in enumerate(a["items"], 1):
                ans = got.get(i)
                if ans is None:                                # missing / invalid / no answer: engine decides
                    out[a["s"]][1][it["k"]] = {"mind": "llm_fallback", "product": it["row"]["engine_product"],
                                               "why_type": "", "why": "", "source": src,
                                               "problem": problem or "no valid answer for this item"}
                else:
                    out[a["s"]][1][it["k"]] = {"mind": "llm", "product": int(ans["product"]),
                                               "why_type": ans["why_type"], "why": ans["why"], "source": src}

    # ------------------------------------------------------------ remember
    def after_trip(self, day, s, hour, basket, info, missed, prices, regular, promo):
        """Write the trip into the NPC's memory: what they bought (and why), what was sold out."""
        store = self.sim.store
        names = store.products["product_name"]
        minds = info["minds"]
        top = ", ".join(clean(names.iat[p]) for p, _ in sorted(basket.items(), key=lambda x: -x[1])[:3])
        self.memory.add_trip(s, day, hour, info["mission"], sum(basket.values()), info["spent"], info["budget"], top)
        rows = []
        for p, qty in basket.items():
            d = minds.get(p) or {"mind": "engine", "why_type": "", "why": ""}
            rows.append((int(s), day, int(p), clean(names.iat[p]), store.aisle_of[p], int(qty), float(prices[p]),
                         float(regular[p]), int(bool(promo[p])), d["mind"], d["why_type"], d["why"]))
        self.memory.add_purchases(rows)
        self.memory.add_stockouts([(int(s), day, int(w), clean(names.iat[w]), r) for _, w, r, _ in missed])

    def end_day(self, day):
        """Make the day permanent in memory. Returns the day's decision rows (for decisions.csv)."""
        self.memory.commit(day)
        rows, self.today = self.today.get("rows", []), {}
        return rows

    def close(self):
        self.memory.close()
        self.cache.close()
