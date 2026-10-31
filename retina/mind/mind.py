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

import os
from pathlib import Path

import numpy as np

from retina.engine.config import ROOT, PROFILE, RUN_ID, stream
from retina.engine.personas import npc_facts, template_persona, varied_sample, clean
from retina.memory.store import MemoryStore
from retina.mind.cache import AnswerCache, fingerprint
from retina.mind.teacher import (LLMTeacher, FakeTeacher, npc_block, request_tokens, PROMPT_VERSION, LETTERS,
                                 MAX_REQUEST_TOKENS)

STREAM_MIND_ORDER = 10        # dice: the order options are shown in (no "A is always right" bias)
STREAM_MIND_CHECK = 11        # dice: which routine buys get a sampled LLM check
STREAM_MIND_STUDENT = 12      # dice: the student's draw from its probabilities (models with pick "draw")
STREAM_MIND_LIST = 33         # dice: which candidates the list student puts on the list
STREAM_MIND_LIST_ITEM = 34    # dice: each planned item's own fixed block (as make_list gives)
CHECK_RATE = 0.05             # ESTIMATED: share of routine buys the teacher double-checks
PRICE_UP = 1.05               # dearer than remembered by more than 5% = worth a thought (ESTIMATED)
BATCH = 5                     # NPCs per LLM request
PANEL = 50                    # thinking NPCs in a mix run (decided 12 Oct 2026)
MAX_OPTIONS = 5
CACHE_FILE = (Path(os.environ["RETINA_CACHE_DIR"]) if os.environ.get("RETINA_CACHE_DIR")      # (release builds,
              else ROOT / "data" / "cache") / "mind_cache.sqlite"                               # tests)
WHY_HERE = {"need": "ran out at home", "repeat": "you buy this regularly",
            "new": "new on your list today", "partner": "goes with another item on your list"}


def build_options(sim, s, k, day, wanted, table, prices, promo, regular, recall=None, order_key=None):
    """2-5 real products from the item's aisle: the engine's pick, their usual ones, a deal, the best
    sellers - with today's price, deal, stock and what this NPC remembers (recall(picks) -> {product:
    memory}). Shared by the live mind, collect mode (retina.student) and the student's features.
    The order is shuffled (no "A is always right" bias); order_key overrides the dice key."""
    store, sh, inv = sim.store, sim.shoppers, sim.inv
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
    memory = recall(picks) if recall else {}
    order = stream(*(order_key or (STREAM_MIND_ORDER, day, s, k))).permutation(len(picks))
    names = store.products["product_name"]
    out = []
    for letter, i in zip(LETTERS, order):
        p = picks[i]
        price, reg = round(float(prices[p]), 2), round(float(regular[p]), 2)
        mem = memory.get(p, {})
        out.append({"letter": letter, "product": p, "name": clean(names.iat[p]), "price": price,
                    "regular": reg, "deal": bool(promo[p]),
                    "off": round((1 - price / reg) * 100, 1) if reg > 0 else 0.0,
                    "times": int(counts.get(p, 0)), "in_stock": bool(inv.stock[p] > 0),
                    "usual": p in usual, "top": p in top, "w": round(weight.get(p, 0.0), 6),
                    "pop": round(base.get(p, 0.0), 6),
                    "last_price": mem.get("last_price"), "sold_out": int(mem.get("sold_out", 0))})
    return aisle, out


WHY_WORDS = {"habit": "my usual one", "deal": "took the deal", "try_new": "tried something new"}


def role_of(card):
    """The NPC's role from its card: 'Deal hunter', 'Steady regular' or 'Convenience first'."""
    return card["persona"]["shopping_style"].split(" - ")[0] if card else None


class Mind:
    def __init__(self, sim, mode, run_dir, panel_size=PANEL, provider="auto", model=None,
                 cache_file=CACHE_FILE, say=print, student="auto", student_dir=None, labels_file=None,
                 list_dir=None, shelf_action_dir=None):
        if mode not in ("mix", "llm", "student"):
            raise ValueError(f"unknown mind '{mode}'")
        if student not in ("auto", "off", "only"):
            raise ValueError("student must be auto, off or only")
        self.sim, self.say = sim, say
        if mode == "student":                                  # every NPC, the student only, never an LLM
            student = "only"
        self.student, self.student_only = None, student == "only"
        if mode in ("mix", "student") and student != "off":
            from retina.student.model import Student, MODEL_DIR
            folder = student_dir or MODEL_DIR
            if Student.available(folder):
                self.student = Student(folder)
            elif student == "only":
                raise SystemExit("--student only / --mind student need a trained student: retina collect, "
                                 "then retina train.")
        self.lister = None                                     # the list student plans lists (student runs)
        self.ender = None                                      # local basket-completion model at each shelf
        self.shelf_action = None                              # qualified local buy/skip shelf policy
        self.trip_size = None
        if mode == "student":
            from retina.student.listmodel import ListStudent, MODEL_DIR as LIST_DIR
            from retina.student.tripend import TripEndStudent
            from retina.student.shelf_action import ShelfActionPolicy, MODEL_DIR as ACTION_DIR
            if ListStudent.available(list_dir or LIST_DIR):
                self.lister = ListStudent(list_dir or LIST_DIR)
            else:
                say("List student unavailable: NPC lists use the rule brain; shelf choices still use the student.")
            if os.environ.get("RETINA_TRIP_SIZE_CALIBRATION") == "1" and PROFILE["id"] in {"retina_supercenter", "supercenter"}:
                from retina.student.trip_size import TripSizePrior
                self.trip_size = TripSizePrior()
            if TripEndStudent.available():
                self.ender = TripEndStudent()
            action_folder = shelf_action_dir or ACTION_DIR
            experimental = os.environ.get("RETINA_SHELF_ACTION_EXPERIMENTAL") == "1"
            if ShelfActionPolicy.available(action_folder) or (
                    experimental and ShelfActionPolicy.experimental_available(action_folder)):
                self.shelf_action = ShelfActionPolicy(action_folder, product_model=self.student)
                if experimental and not ShelfActionPolicy.available(action_folder):
                    say("Experimental shelf action model enabled; its skip behavior is not yet qualified.")
        self.labels_file, self.labels = labels_file, None
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
        self.roles = {}                                        # npc -> role from its card (student runs)

    # ---------------------------------------------------------------- who
    def thinks(self, s):
        return self.mode in ("llm", "student") or s in self.panel

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
        recall = lambda picks: self.memory.recall(s, picks, day)
        return build_options(self.sim, s, k, day, wanted, table, prices, promo, regular, recall)

    # ------------------------------------------------------------- router
    def route(self, s, k, day, wanted, route, options, prices):
        """None = autopilot (routine buy), else the reason the teacher is asked."""
        if self.mode == "llm":
            return "llm_mode"
        if self.mode == "student":                             # the student decides EVERY shelf choice
            return "student_mode"
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
        if self.lister is not None:                            # the NPC brain plans the list itself
            for s, trip in trips.items():
                self._plan_list(s, day, ctx, trip)
        for s, trip in trips.items():
            if self.mode == "student":                         # every NPC gets its role card (kept in memory)
                self.roles.setdefault(s, role_of(self.card(s, day)))
            g = int(sh.sensitivity_group[s])
            prices, promo, table = ctx["prices_g"][g], ctx["promo_g"][g], ctx["tables"][g]
            decisions, items = {}, []
            for k, (wanted, route, *_) in enumerate(trip["plan"]):
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
                    pick = self._student_pick(s, route, trip["mission"], options, int(wanted),
                                              stream(STREAM_MIND_STUDENT, day, s, k))
                    if pick is not None:
                        decisions[k] = pick
                    else:
                        items.append({"k": k, "aisle": aisle, "why_here": WHY_HERE.get(route, route),
                                      "options": options, "row": row})
                rows.append((row, s, k))
            out[s] = (trip, decisions)
            if items:
                card = self.card(s, day)
                memory_lines = self.remembers(s, day)
                situation = npc_block(0, card, memory_lines, trip, items)
                key = fingerprint(RUN_ID, self.sim.seed, PROMPT_VERSION, int(s), day, situation)
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

    # ------------------------------------------------- real time (realtime.py)
    def start_trip(self, day, s, trip, ctx):
        """The NPC walks in: the list student plans today's list (student runs) and the role card is read."""
        if self.lister is not None:
            self._plan_list(s, day, ctx, trip)
        self.roles.setdefault(s, role_of(self.card(s, day)))
        return trip

    def at_shelf(self, day, s, trip, ctx):
        """{list position: decision} that decides WHEN the shopper stands at the shelf - with the stock,
        prices and options of that moment (student runs; the student never needs an LLM)."""
        mind = self

        class Shelf(dict):
            def __bool__(self):                                # empty until the first shelf - still a brain
                return True

            def get(self, k, default=None):
                if k not in self:
                    self[k] = mind.decide_item(day, s, k, trip, ctx)
                return dict.get(self, k, default)

            def act(self, k, spent, budget):
                return mind.decide_shelf_action(day, s, k, trip, ctx, spent, budget)
        return Shelf()

    def decide_item(self, day, s, k, trip, ctx):
        g = int(self.sim.shoppers.sensitivity_group[s])
        prices, promo, table = ctx["prices_g"][g], ctx["promo_g"][g], ctx["tables"][g]
        wanted, route = trip["plan"][k][:2]
        aisle, options = self.options(s, k, day, int(wanted), table, prices, promo, ctx["regular"])
        trigger = self.route(s, k, day, int(wanted), route, options, prices) if options else None
        row = {"day": day, "hour": ctx["hour"][s], "npc": int(s), "item": k, "aisle": aisle,
               "route": route, "trigger": trigger or "routine", "engine_product": int(wanted),
               "options": [{x: o[x] for x in ("letter", "product", "price", "regular", "deal", "times",
                                               "in_stock", "usual", "w", "pop")} for o in options],
               "provider": None, "model": None, "cached": 0, "valid": 1, "problem": None}
        if s in ctx.get("minute", {}):
            row["minute"] = ctx["minute"][s]
        d = None
        if trigger is not None:
            d = self._student_pick(s, "new" if route == "impulse" else route, trip["mission"], options,
                                   int(wanted), stream(STREAM_MIND_STUDENT, day, s, k))
        if d is None:
            d = {"mind": "autopilot" if trigger is None else "engine", "product": int(wanted),
                 "why_type": "habit" if trigger is None else "", "why": "usual buy" if trigger is None else ""}
        row.update(mind=d["mind"], product=d["product"], why_type=d["why_type"], why=d["why"])
        self.memory.add_decisions([row])
        self.today.setdefault("rows", []).append(row)
        return d

    def decide_shelf_action(self, day, s, k, trip, ctx, spent, budget):
        """Let the qualified local policy act on the shelf as it exists this minute.

        Its proposal is logged separately from the store's final stock and budget
        checks. This path is used only when a qualified action model is installed.
        """
        if self.shelf_action is None:
            raise RuntimeError("no shelf action policy is installed")
        from retina.student.features import npc_numbers

        g = int(self.sim.shoppers.sensitivity_group[s])
        prices, promo, table = ctx["prices_g"][g], ctx["promo_g"][g], ctx["tables"][g]
        entry = trip["plan"][k]
        wanted, route = int(entry[0]), entry[1]
        desired_qty = int(entry[5]) if len(entry) > 5 else 1
        aisle, options = self.options(s, k, day, wanted, table, prices, promo, ctx["regular"])
        remaining_budget = max(0.0, float(budget - spent)) if np.isfinite(budget) else None
        decision = self.shelf_action.choose(
            npc_numbers(self.sim.shoppers, s), "new" if route == "impulse" else route,
            trip["mission"], options, wanted, remaining_budget, desired_qty,
            stream(STREAM_MIND_STUDENT, day, s, k),
        )
        if decision["action"] not in ("buy", "skip"):
            raise ValueError("the NPC shelf policy returned an unknown action")
        product = int(decision["product"])
        if decision["action"] == "buy" and not any(
                int(o["product"]) == product and o["in_stock"] for o in options):
            raise ValueError("the NPC shelf policy chose a product unavailable at this shelf")
        if decision["action"] == "skip" and product != -1:
            raise ValueError("a skip action must not select a product")
        if decision["action"] == "buy" and int(decision["qty"]) < 1:
            raise ValueError("a buy action must request at least one unit")
        row = {"day": day, "hour": ctx["hour"][s], "npc": int(s), "item": k, "aisle": aisle,
               "route": route, "trigger": "shelf_action", "engine_product": wanted,
               "options": [{x: o[x] for x in ("letter", "product", "price", "regular", "deal", "times",
                                               "in_stock", "usual", "w", "pop")} for o in options],
               "provider": "local", "model": self.shelf_action.meta.get("version", "shelf_action"),
               "cached": 0, "valid": 1, "problem": None,
               "mind": "student", "product": product, "why_type": decision["action"],
               "why": decision.get("reason", "local NPC shelf action"),
               "qty": int(decision["qty"]),
               "skip_probability": float(decision.get("skip_probability", 0.0))}
        if s in ctx.get("minute", {}):
            row["minute"] = ctx["minute"][s]
        self.memory.add_decisions([row])
        self.today.setdefault("rows", []).append(row)
        return decision

    def _plan_list(self, s, day, ctx, trip):
        """Replace the habit-rule list by the list student's: products and quantities from the NPC's
        scenario (same candidates as the teachers saw). Routes keep the engine's meaning."""
        from retina.student.features import npc_numbers
        from retina.student.listplan import STREAM_LIST, candidates
        sh = self.sim.shoppers
        g = int(sh.sensitivity_group[s])
        cands = candidates(self.sim, s, day, ctx["prices_g"][g], ctx["regular"], ctx["promo_g"][g],
                           stream(STREAM_LIST, day, s))
        last = sh.last_trip_day[s]
        budget = trip["budget"] if np.isfinite(trip["budget"]) else None
        scenario = {"candidates": cands, "npc_x": npc_numbers(sh, s), "usual_items": float(sh.basket_mean[s]),
                    "since": int(day - last) if last >= 0 else None, "budget": budget}
        prior = getattr(self, "trip_size", None)
        target = prior.sample(stream(STREAM_MIND_LIST, day, s, 1), len(cands)) if prior and cands else None
        requirements = trip.get("requirements")
        if requirements is None:
            from retina.engine.requirements import NPCRequirements
            from retina.engine.shopping import trip_mission
            planned = max(1, int(round(sh.basket_mean[s])))
            requirements = NPCRequirements.for_trip(
                sh, s, day, planned, trip_mission(planned, sh.basket_mean[s]), trip["budget"])
            trip["requirements"] = requirements
        if target is not None:
            from dataclasses import replace
            from retina.engine.shopping import trip_mission
            mission = trip_mission(target, sh.basket_mean[s])
            trip["mission"] = mission
            essential_count = min(len(requirements.due_products), max(1, min(3, (target + 4) // 5)), target)
            trip["requirements"] = requirements = replace(
                requirements, mission=mission, target_items=target,
                essential_products=requirements.due_products[:essential_count])
        chosen = list(self.lister.plan(scenario, stream(STREAM_MIND_LIST, day, s), target_lines=target,
                                       required_products=requirements.essential_products if target is not None else ()))
        picked = {int(c["product"]) for c, _ in chosen}
        available = {int(c["product"]): c for c in cands}
        for product in reversed(requirements.essential_products if target is None else ()):
            if product not in picked and product in available:
                chosen.insert(0, (available[product], 1))
                picked.add(product)
        plan, intent_candidates = [], {}
        for k, (c, qty) in enumerate(chosen):
            route = ("need" if c["ran_out"] else "repeat") if c["source"] == "usual" else                 "partner" if c["source"] == "goes with" else "new"
            plan.append((c["product"], route, False, route == "partner",
                         stream(STREAM_MIND_LIST_ITEM, day, s, k).random(11), qty))
            intent_candidates[k] = c
        trip.update(plan=plan, planned=len(plan), list_by="list student",
                    intent_candidates=intent_candidates, browse_candidates=cands)

    def next_aisle(self, s, trip, aisles, pos, spent, prices, promo):
        """Reconsider remaining aisles in the store using the local list model and current state."""
        floor = self.sim.store.floor
        candidates = trip.get("intent_candidates") or {}
        if self.lister is None or not candidates:
            return min(aisles, key=lambda a: floor._dist(pos, a))
        from retina.student.features import npc_numbers
        from retina.student.listtrain import scenario_matrix, policy

        available = set(aisles)
        indices = [k for k, c in candidates.items()
                   if int(floor.product_aisle[int(c["product"])]) in available]
        if not indices:
            return min(aisles, key=lambda a: floor._dist(pos, a))
        cands = []
        for k in indices:
            c = dict(candidates[k])
            product = int(c["product"])
            c["price"] = float(prices[product])
            c["deal"] = bool(promo[product])
            c["off"] = max(0.0, (1 - c["price"] / max(float(c["regular"]), 0.01)) * 100)
            cands.append(c)
        last = self.sim.shoppers.last_trip_day[s]
        budget = trip["budget"] - spent if np.isfinite(trip["budget"]) else None
        scenario = {"candidates": cands, "npc_x": npc_numbers(self.sim.shoppers, s),
                    "usual_items": float(self.sim.shoppers.basket_mean[s]),
                    "since": int(self.sim.day - last) if last >= 0 else None, "budget": budget}
        X = scenario_matrix(scenario, self.lister.rebuy)
        chance = policy(self.lister.probabilities(X), X, self.lister.shift, self.lister.habit)
        scores = {}
        for c, p in zip(cands, chance):
            aisle = int(floor.product_aisle[int(c["product"])])
            in_stock = self.sim.inv.stock[int(c["product"])] > 0
            scores[aisle] = scores.get(aisle, 0.0) + float(p) * (1.0 if in_stock else 0.35)
        return max(aisles, key=lambda a: (scores.get(a, 0.0) / (1 + floor._dist(pos, a) / 30), -a))

    def checkout_chance(self, s, day, lines):
        """Local inference after a shelf. No LLM request or look-ahead to unvisited stock."""
        if self.ender is None:
            return 0.0
        shoppers = self.sim.shoppers
        last = int(shoppers.last_trip_day[s])
        since = day - last if last >= 0 else shoppers.gap_mean[s]
        return float(self.ender.meta["in_store_scale"]) * self.ender.probability(
            lines, shoppers.basket_mean[s], since)

    def browse_offer(self, s, day, trip, visible, spent, prices, promo, in_basket):
        """Rescore previously declined products when their aisle comes into view."""
        from retina.engine.config import IMPULSE_RATE
        if self.lister is None or IMPULSE_RATE <= 0 or not visible:
            return None
        from retina.student.features import npc_numbers
        from retina.student.listtrain import scenario_matrix, policy

        store, shoppers = self.sim.store, self.sim.shoppers
        floor = store.floor
        planned = {int(entry[0]) for entry in trip["plan"]}
        candidates = trip.get("browse_candidates") or []
        visible = set(visible)
        eligible = {int(c["product"]) for c in candidates
                    if int(c["product"]) not in planned and int(c["product"]) not in in_basket
                    and int(floor.product_aisle[int(c["product"])]) in visible
                    and self.sim.inv.stock[int(c["product"])] > 0
                    and shoppers.pantry[s].get(int(c["product"]), -1) <= day}
        if not eligible:
            return None
        current = []
        for old in candidates:
            c = dict(old)
            product = int(c["product"])
            c["price"] = float(prices[product])
            c["deal"] = bool(promo[product])
            c["off"] = max(0.0, (1 - c["price"] / max(float(c["regular"]), 0.01)) * 100)
            current.append(c)
        last = int(shoppers.last_trip_day[s])
        budget = trip["budget"] - spent if np.isfinite(trip["budget"]) else None
        scenario = {"candidates": current, "npc_x": npc_numbers(shoppers, s),
                    "usual_items": float(shoppers.basket_mean[s]),
                    "since": day - last if last >= 0 else None, "budget": budget}
        X = scenario_matrix(scenario, self.lister.rebuy)
        chance = policy(self.lister.probabilities(X), X, self.lister.shift, self.lister.habit)
        choices = [(c, float(p)) for c, p in zip(current, chance)
                   if int(c["product"]) in eligible and (budget is None or c["price"] <= budget)]
        if not choices:
            return None
        chosen, p = max(choices, key=lambda item: (item[1] * (1.5 if item[0]["deal"] else 1.0),
                                                   -int(item[0]["product"])))
        # ESTIMATED: interest is learned from the list student; seeing a shelf turns that
        # interest into a new stop at a rate controlled by the store's impulse_rate.
        browse_probability = min(0.4, IMPULSE_RATE * 0.4 * p * (1.5 if chosen["deal"] else 1.0))
        return {"aisle": int(floor.product_aisle[int(chosen["product"])]),
                "product": int(chosen["product"]), "probability": browse_probability,
                "deal": bool(chosen["deal"])}

    def _student_pick(self, s, route, mission, options, wanted, rng=None):
        """The student's decision when it is sure enough (or always with --student only), else None."""
        if self.student is None:
            return None
        from retina.student.features import npc_numbers
        from retina.student.model import why_type
        option, p, _ = self.student.choose(npc_numbers(self.sim.shoppers, s), route, mission, options, wanted, rng)
        if option is None and self.student_only:          # nothing choosable (all sold out): the engine's
            return {"mind": "engine", "product": wanted,   # empty-shelf rules decide - never the LLM
                    "why_type": "", "why": "nothing in stock to choose"}
        if option is None or (p < self.student.threshold and not self.student_only):
            return None
        kind = why_type(option, wanted)
        role = self.roles.get(s)
        why = f"student, {p:.0%} sure" if role is None else f"{role}: {WHY_WORDS.get(kind, kind)} (student, {p:.0%} sure)"
        return {"mind": "student", "product": int(option["product"]), "why_type": kind, "why": why}

    def _save_labels(self, a, day, answers, provider, model):
        """Active learning: a teacher answer from THE teacher becomes a training label (source 'run')."""
        from retina.student.features import npc_numbers
        from retina.student.labels import LabelStore, TEACHER
        if (provider, model) != TEACHER:
            return
        if self.labels is None:
            self.labels = LabelStore(self.labels_file)
        s, rows = a["s"], []
        for i, it in enumerate(a["items"], 1):
            ans = answers.get(i)
            rows.append({"id": f"r-{RUN_ID}-{self.sim.seed}-{day}-{s}-{it['k']}", "repeat": 0, "source": "run",
                         "provider": provider, "model": model, "prompt_version": PROMPT_VERSION, "npc": int(s),
                         "day": day, "valid": ans is not None, "store": PROFILE["id"], "aisle": it["aisle"],
                         "route": it["row"]["route"], "mission": a["trip"]["mission"], "variation": "natural",
                         "npc_x": npc_numbers(self.sim.shoppers, s), "options": it["options"],
                         "engine_product": it["row"]["engine_product"],
                         "choice": None if ans is None else ans["product"], "letter": None if ans is None else ans["choice"],
                         "why_type": None if ans is None else ans["why_type"], "why": None if ans is None else ans["why"]})
        self.labels.add(rows)

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
                    self._save_labels(a, a["items"][0]["row"]["day"], answers[a["s"]], provider, model)
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
        if self.labels is not None:
            self.labels.close()
