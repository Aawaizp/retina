"""RETINA Mart shopping: how NPCs fill their baskets and how one day in the store runs."""

import json
from bisect import bisect_right

import numpy as np
import pandas as pd

from .config import *
from .world import *
from .floor import STREAM_WALK
from . import traffic

# ============================================================ trip needs
# Real Instacart trips (loaders/trip_needs.py): the longer since the last trip, the bigger the
# basket, and one shopper's baskets swing from trip to trip (more than plain luck would give).
TRIP = json.loads(TRIP_STATS_FILE.read_text(encoding="utf-8")) if NEEDS_ON else None
GAP_BINS = [(3, "1-3"), (7, "4-7"), (14, "8-14"), (29, "15-29"), (10 ** 9, "30+")]
TOP_UP, STOCK_UP = 0.5, 1.5          # trip size vs the NPC's own average (same rule as the loader)


def gap_factor(days):
    for top, key in GAP_BINS:
        if days <= top:
            return TRIP["basket_vs_days_by_gap"][key]


def trip_mission(n_items, usual):
    ratio = n_items / usual
    return "top_up" if ratio <= TOP_UP else ("stock_up" if ratio >= STOCK_UP else "routine")

# ======================================================== event recorder
# The 2D replay (replay.py) shows what shoppers do. It switches this on for ONE day. Recording
# only writes down what happens - it never rolls dice or changes a decision.
# Events: SHOPPER_ENTER, MOVE_TO_DEPARTMENT, VIEW_PRODUCT, PURCHASE, SUBSTITUTE, SKIP,
#         STOCKOUT, CHECKOUT, LEAVE
RECORDER = {"day": None, "events": []}


def record(day, s, event, product=-1, **info):
    if RECORDER["day"] == day:
        RECORDER["events"].append({"shopper": int(s), "event": event, "product": int(product), **info})

# ============================================================= shopping

def build_partners(store):
    """Real 'bought together' pairs (Instacart) mapped onto products the store carries.
    For each product: its partners, the EXTRA chance of adding each one beyond chance
    (P(partner | product) - P(partner), stored as a running total) and the real lift."""
    pairs = pd.read_csv(PAIRS_FILE)
    p = store.products
    carried = np.zeros(len(p), dtype=bool)
    for d in store.departments.values():
        carried[d.product_idx] = True
    on_shelf = p[carried & p["instacart_match"].notna()]
    # One shelf product per Instacart name: the most popular one the store carries
    best = on_shelf.sort_values("popularity", ascending=False).drop_duplicates("instacart_match")
    best_idx = pd.Series(best.index.to_numpy(), index=best["instacart_match"].to_numpy())
    pairs = pairs.assign(partner_idx=pairs["partner"].map(best_idx),
                         extra=pairs["p_partner_given_product"] - pairs["p_partner"])
    pairs = pairs.dropna(subset=["partner_idx"])
    by_name = {name: (g["partner_idx"].to_numpy(int), g["extra"].to_numpy(), g["lift"].to_numpy())
               for name, g in pairs.groupby("product")}
    partners = {}
    for i, name in zip(on_shelf.index, on_shelf["instacart_match"]):
        if name in by_name:
            idx, extra, lift = by_name[name]
            keep = idx != i
            if keep.any():
                partners[i] = (idx[keep], np.cumsum(extra[keep]), lift[keep])
    print(f"\nPairs loaded: {len(pairs):,} real Instacart pairs -> "
          f"{len(partners):,} shelf products have partners")
    return partners

def build_aisle_partners(store):
    """Shopping missions: for each aisle, which aisles real shoppers buy WITH it (Instacart prior
    orders). Partner weight = the aisle's normal share x lift (how many times more often the two
    are bought together than by chance), so missions add real pairings (pasta -> pasta sauce)
    without changing the overall department mix. Includes a second item from the same aisle."""
    if AISLE_AFFINITY <= 0 or not AISLE_PAIRS_FILE.exists():
        return {}
    stats = pd.read_csv(AISLE_STATS_FILE).set_index("aisle")
    total = stats["orders_total"].iat[0]
    pairs = pd.read_csv(AISLE_PAIRS_FILE)
    pairs = pd.concat([pairs, pairs.rename(columns={"aisle_a": "aisle_b", "aisle_b": "aisle_a"})])
    together = pairs.set_index(["aisle_a", "aisle_b"])["together_orders"]
    # each aisle's normal share of NEW picks: department share x share inside the department
    base = {}
    if NOT_SOLD_HERE:                     # custom store: missions can lead to aisles it does not sell
        want = store.want
        for name, share in zip(want["dept_names"], np.diff(np.r_[0, want["dept_cum"]])):
            names, cum = want["aisles"][name]
            for a, s in zip(names, np.diff(np.r_[0, cum])):
                base[a] = share * s
    for name, share in zip(store.dept_names, np.diff(np.r_[0, store.dept_cum])) if not NOT_SOLD_HERE else ():
        d = store.departments[name]
        for a, s in zip(d.aisle_names, np.diff(np.r_[0, d.aisle_cum])):
            base[a] = share * s
    names = np.array([a for a in base if a in stats.index])
    orders = stats.loc[names, "orders"].to_numpy(float)
    lines = stats.loc[names, "lines"].to_numpy(float)
    b = np.array([base[a] for a in names])
    partners = {}
    for i, a in enumerate(names):
        both = np.array([together.get((a, x), np.nan) for x in names])
        lift = both * total / (orders[i] * orders)
        lift = np.where(np.isnan(lift), 1.0, lift)                # rarely seen together: no evidence
        lift[i] = max(lines[i] - orders[i], 0) / orders[i] / (orders[i] / total)   # 2nd item, same aisle
        partners[a] = (names, np.cumsum(b * lift ** MISSION_SHARPNESS))
    print(f"Shopping missions: {len(partners)} aisles with real partner aisles "
          f"({AISLE_AFFINITY:.0%} of new picks follow an aisle already in the basket)")
    return partners


def pick_product(dept, cum_prob, u):
    """Product from a department for a random number u (0-1), using a running total of pick
    probabilities. searchsorted (binary search) is much faster than rng.choice for big shelves.
    Passing the same u to two tables shows what the shopper would have picked otherwise."""
    i = np.searchsorted(cum_prob, u * cum_prob[-1])
    return dept.product_idx[min(i, len(dept.product_idx) - 1)]


def true_share(flags):
    """Share of True in an array; NaN (no warning) when the array is empty."""
    return flags.mean() if len(flags) else np.nan


def poisson_from_u(mean, u):
    """Poisson random number from ONE uniform u (inverse method) - keeps the dice in fixed slots."""
    k, p = 0, np.exp(-mean)
    total = p
    while u > total and k < 50:
        k += 1
        p *= mean / k
        total += p
    return k


def fill_basket(s, day, store, shoppers, tables, inv, promo, rng, with_points=False, prices=None):
    """One shopper's trip = make_list (in their head, at home) + walk_shelves (in the store).
    Same dice, same order as one function: the list never looks at the shelves."""
    trip = make_list(s, day, store, shoppers, tables, promo, rng, prices)
    for event, product, info in trip["log"]:
        record(day, s, event, product, **info)
    return walk_shelves(s, day, store, shoppers, tables, inv, promo, rng, trip, with_points, prices)


def make_list(s, day, store, shoppers, tables, promo, rng, prices=None):
    """Step 1 of a trip (see walk_shelves for step 2).
    1) The shopping list (in their head): each item is a repeat (favorites), a partner of the
       previous item (real 'bought together' pairs) or a new pick: department (real share) ->
       aisle (fixed share) -> product (popularity x price). Items still at home are skipped.
    2) The walk: they go to the nearest shelf on their list first, and SEE every shelf they pass.
    3) At each shelf: price check (a pricier usual item may be swapped for a similar one; if the
       whole aisle feels pricier some walk away), empty shelf reactions, promotions (brand switch,
       stocking up). On the way, deals on shelves they pass can catch their eye (unplanned buys).
    Needs (when on): trip size follows days since the last trip; usual items are re-bought when
    they have RUN OUT at home (each product lasts a real number of trips); extras stop once the
    trip's share of the household's food budget is spent.
    Every list item uses its own fixed block of random numbers, so an experiment that changes one
    item's outcome does not shift the dice of the shopper's other items.
    The list depends only on the shopper's own state (home, pantry, habits) and this week's pick
    tables - never on the shelves - so lists can be made ahead (the mind batches NPCs this way).
    Events go into trip["log"] (the caller records them when the shopper really comes in).
    Returns the trip: the list (plan), effects so far, budget, mission and planned size."""
    last_trip = shoppers.last_trip_day[s]
    since = day - last_trip if last_trip >= 0 else shoppers.gap_mean[s]
    if NEEDS_ON:
        mean = shoppers.basket_mean[s] * gap_factor(since)
        v = TRIP["basket_extra_variance"]
        mean *= rng.gamma(1 / v, v)                            # trips swing like real ones
        n_items = max(1, rng.poisson(mean))
    else:
        n_items = max(1, rng.poisson(shoppers.basket_mean[s]))
    trip_type = trip_mission(n_items, shoppers.basket_mean[s])
    favorites, pantry, home = shoppers.favorites[s], shoppers.pantry[s], shoppers.home[s]
    group = tables[shoppers.sensitivity_group[s]]
    effects, log, not_sold = [], [], []
    budget = np.inf
    if BUDGET_ON and shoppers.weekly_budget is not None and prices is not None:
        budget = shoppers.weekly_budget[s] * max(since, 7) / 7   # a week's food money (more if longer)

    # 1) the shopping list
    plan, last, plan_aisles = [], -1, []                       # last = previous item on the list
    on_list = set()
    for u in rng.random((n_items, 11)):                        # 11 fixed dice per item
        switched = together = False
        pair = store.partners.get(last)
        wanted = None
        if favorites and u[1] < shoppers.reorder_prob[s] * REPEAT_BOOST:
            # Re-buy something bought before: staples (milk, eggs, water) far more than one-offs
            # (spices) - weighted by how often real shoppers re-buy that aisle. With needs on, only
            # something that has run out at home (they look through a few usual items)
            cum = shoppers.favorite_cum[s]
            if NEEDS_ON:
                # an aisle they usually buy from; inside it, a habit product that has RUN OUT at
                # home if there is one, otherwise their usual one anyway (buying ahead)
                usual = habit_aisle(store, shoppers, s, u[2])
                first_pick = None
                for k in range(NEED_TRIES if len(usual) > 1 else 1):
                    p = habit_product(shoppers, s, usual, (u[3] + k * 0.6180339887) % 1.0)
                    if p in on_list:                               # already on today's list
                        continue
                    if first_pick is None:
                        first_pick = p
                    if home.get(p, -1) <= day and pantry.get(p, -1) <= day:
                        wanted, route = p, "need"                  # ran out at home
                        break
                if wanted is None and first_pick is not None and pantry.get(first_pick, -1) <= day:
                    wanted, route = first_pick, "repeat"           # a usual item, buying ahead
                if wanted is None:                                 # their usual ones are listed: try
                    name = store.aisle_of[usual[0]]                # something else in the same aisle
                    if group["cum"][name] is not None:
                        wanted, route = pick_product(store.aisles[name], group["cum"][name], u[4]), "new"
            else:
                wanted, route = favorites[min(bisect_right(cum, u[2] * cum[-1]), len(cum) - 1)], "repeat"
        if wanted is not None:
            pass
        elif pair is not None and u[0] < pair[1][-1]:          # goes with the last item (real pairs)
            wanted, route = pair[0][np.searchsorted(pair[1], u[0], side="right")], "partner"
            together = True
        else:
            # Shopping mission: often the next aisle goes with one already on the list (pasta ->
            # pasta sauce, real Instacart pairs); otherwise department (real share) -> aisle
            on_mission = bool(plan_aisles) and u[2] < AISLE_AFFINITY
            if on_mission:                                     # rescale the die to 0-1 again
                ud = u[2] / AISLE_AFFINITY
            else:
                ud = (u[2] - AISLE_AFFINITY) / (1 - AISLE_AFFINITY) if plan_aisles else u[2]
            mission = store.aisle_partners.get(plan_aisles[int(u[3] * len(plan_aisles))]) if on_mission else None
            if mission is not None:
                aisle_name = mission[0][pick_index(mission[1], ud)]
            elif NOT_SOLD_HERE:                                # custom store: wants follow ALL aisles
                want_dept = store.want["dept_names"][pick_index(store.want["dept_cum"], ud)]
                want_aisles, want_cum = store.want["aisles"][want_dept]
                aisle_name = want_aisles[pick_index(want_cum, u[3])]
            else:
                d = store.departments[store.dept_names[pick_index(store.dept_cum, ud)]]
                aisle_name = d.aisle_names[pick_index(d.aisle_cum, u[3])]
            if aisle_name not in store.aisles:                 # custom store does not sell this aisle:
                not_sold.append(aisle_name)                    # lost demand, never a substitute
                log.append(("SKIP", -1, {"reason": "not_sold_here", "aisle": aisle_name}))
                continue
            aisle = store.aisles[aisle_name]
            if group["cum"][aisle.name] is None:              # whole aisle taken off the shelves
                effects.append((s, aisle.product_idx[0], "walked_away", 1))
                log.append(("SKIP", -1, {"reason": "aisle_removed"}))
                continue
            wanted, route = pick_product(aisle, group["cum"][aisle.name], u[4]), "new"
            # Brand switch = without promotions (base table) the same shopper picks something else
            switched = promo[wanted] and wanted != pick_product(aisle, aisle.cum_prob, u[4])
        if pantry.get(wanted, -1) > day:                       # still have some at home
            effects.append((s, wanted, "had_at_home", -1))
            log.append(("SKIP", wanted, {"reason": "had_at_home"}))
            continue
        plan.append((wanted, route, switched, together, u))
        on_list.add(wanted)
        plan_aisles.append(store.aisle_of[wanted])
        last = wanted
    return {"plan": plan, "effects": effects, "log": log, "budget": budget,
            "mission": trip_type, "planned": n_items, "not_sold": not_sold}


def walk_shelves(s, day, store, shoppers, tables, inv, promo, rng, trip, with_points=False, prices=None,
                 decisions=None):
    """Step 2 of a trip, in one go (the classic engine): every step of walk_steps, straight through."""
    steps = walk_steps(s, day, store, shoppers, tables, inv, promo, rng, trip, with_points, prices, decisions)
    try:
        while True:
            next(steps)
    except StopIteration as done:
        return done.value


def walk_steps(s, day, store, shoppers, tables, inv, promo, rng, trip, with_points=False, prices=None,
               decisions=None):
    """Step 2 of a trip: walk the store with the list from make_list, shelf by shelf.
    A generator: it stops ("route", walk) once the route is known, ("leg", k) before the deals that catch
    the eye on the way to shelf k, and ("stop", k) when the shopper reaches shelf k - so the real-time
    engine (realtime.py) can run every shopper on one clock; the classic engine runs it straight through.
    Returns (basket, misses, effects, walk, info) at the end.
    decisions = {list position: the mind's decision} for thinking NPCs (retina.mind), else None.
    A decision with mind "llm" or "student" replaces the engine's shelf choice (product, or skip =
    product -1); every other decision ("autopilot", "llm_fallback", "engine") leaves the engine's own
    rules to choose.
    The student's pick still gets the crowd's walk-away check ("too expensive today"): unlike an LLM it
    cannot SKIP, so without it student shoppers would buy more items than anyone else.
    Python still checks what the mind picked: budget, and stock when the NPC reaches the shelf."""
    floor = store.floor
    plan, budget, trip_type, n_items = trip["plan"], trip["budget"], trip["mission"], trip["planned"]
    pantry, home = shoppers.pantry[s], shoppers.home[s]
    sens = shoppers.price_sensitivity[s]
    group = tables[shoppers.sensitivity_group[s]]
    basket, misses, routes, minds = {}, [], {}, {}
    effects = list(trip["effects"])
    spent = 0.0

    # 2) the walk
    walk = floor.plan_route([floor.product_aisle[p[0]] for p in plan], with_points)
    n_stops = len(walk.stops)
    items_at = [[] for _ in range(n_stops)]
    for i in walk.order:
        items_at[walk.stop_of_item[i]].append(i)
    # unplanned buys: deals that catch the eye on shelves they walk past
    impulse = {}
    seen = np.array([a for a in walk.passed if group["cum"][floor.aisle_names[a]] is not None], dtype=int)
    if len(seen):
        leg_of = {}
        for k, passed in enumerate(walk.legs):
            for a in passed:
                leg_of.setdefault(a, k)
        for k, a in enumerate(walk.stops):
            leg_of.setdefault(int(a), k)
        seen_cum = np.cumsum(floor.weight[seen])
        for _ in range(rng.poisson(IMPULSE_RATE * sens)):
            for _ in range(10):                                # browse until a deal catches the eye
                a = int(seen[pick_index(seen_cum, rng.random())])
                aisle = store.aisles[floor.aisle_names[a]]
                product = pick_product(aisle, group["cum"][aisle.name], rng.random())
                if promo[product]:
                    break
            if promo[product]:
                impulse.setdefault(leg_of[a], []).append(product)
    yield ("route", walk)

    # 3) shelf by shelf
    picks = 0
    for k in range(n_stops + 1):
        if impulse.get(k):
            yield ("leg", k)
        for product in impulse.get(k, []):                    # on the way to the next shelf
            if inv.stock[product] > 0 and pantry.get(product, -1) <= day:
                if prices is not None and spent + prices[product] > budget:
                    effects.append((s, product, "over_budget", 1))
                    record(day, s, "SKIP", product, reason="over_budget", leg=k, aisle=int(floor.product_aisle[product]))
                    continue
                take_unit(product, inv)
                basket[product] = basket.get(product, 0) + 1
                routes.setdefault(product, "impulse")
                spent += prices[product] if prices is not None else 0.0
                picks += 1
                effects.append((s, product, "impulse", 1))
                record(day, s, "PURCHASE", product, qty=1, promo=True, route="impulse", leg=k,
                       aisle=int(floor.product_aisle[product]))
        if k == n_stops:
            break
        stop = int(walk.stops[k])
        yield ("stop", k)
        record(day, s, "MOVE_TO_DEPARTMENT", department=floor.aisle_dept[stop],
               aisle=floor.aisle_names[stop], stop=k)
        for i in items_at[k]:
            wanted, route, switched, together, u = plan[i][:5]
            want_qty = plan[i][5] if len(plan[i]) > 5 else None    # the list student says how many
            picks += 1
            aisle_name = store.aisle_of[wanted]
            record(day, s, "VIEW_PRODUCT", wanted, route=route, stop=k)
            decision = decisions.get(i) if decisions else None
            if decision is not None and decision["mind"] in ("llm", "student"):   # the mind already chose here
                if decision["product"] < 0:
                    record(day, s, "SKIP", wanted, reason="mind_skip", stop=k)
                    continue
                if decision["product"] != wanted:
                    record(day, s, "SUBSTITUTE", decision["product"], instead_of=int(wanted),
                           reason="mind_choice", stop=k)
                    wanted, switched = decision["product"], False
                # the student never learned to SKIP (an LLM can): it gets the crowd's walk-away check,
                # with the same dice - "the aisle feels too expensive today"
                if decision["mind"] == "student" and u[7] > min(1.0, group["inclusive"][aisle_name] ** CATEGORY_RESPONSE):
                    effects.append((s, wanted, "walked_away", 1))
                    record(day, s, "SKIP", wanted, reason="too_expensive", stop=k)
                    continue
            else:
                boost = group["boost"][wanted]
                if route != "new" and u[5] < 1 - boost:            # usual item got pricier (or delisted)
                    if group["cum"][aisle_name] is None:              # nothing left in the aisle
                        effects.append((s, wanted, "walked_away", 1))
                        record(day, s, "SKIP", wanted, reason="aisle_removed", stop=k)
                        continue
                    choice = pick_product(store.aisles[aisle_name], group["cum"][aisle_name], u[6])
                    if choice != wanted:
                        why = "delisted_switch" if boost == 0 else "price_switch"
                        effects.append((s, choice, why, 1))
                        record(day, s, "SUBSTITUTE", choice, instead_of=int(wanted), reason=why, stop=k)
                        wanted = choice
                if u[7] > min(1.0, group["inclusive"][aisle_name] ** CATEGORY_RESPONSE):
                    effects.append((s, wanted, "walked_away", 1))      # aisle feels too expensive today
                    record(day, s, "SKIP", wanted, reason="too_expensive", stop=k)
                    continue
            if route != "need" and prices is not None and spent + prices[wanted] > budget:
                effects.append((s, wanted, "over_budget", 1))      # money for this trip is spent
                record(day, s, "SKIP", wanted, reason="over_budget", stop=k)
                continue
            product, reaction = take_from_shelf(wanted, store, inv, u[8], u[9])
            if reaction is not None:
                misses.append((s, wanted, reaction, product))
                record(day, s, "STOCKOUT", wanted, reaction=reaction, stop=k)
                if product >= 0:
                    record(day, s, "SUBSTITUTE", product, instead_of=int(wanted), reason="out_of_stock", stop=k)
                else:
                    record(day, s, "SKIP", wanted, reason=reaction, stop=k)
            if product < 0:
                continue
            basket[product] = basket.get(product, 0) + 1
            routes.setdefault(product, route)
            if decision is not None:
                minds.setdefault(product, decision)
            if prices is not None:
                spent += prices[product]
            if together and product == wanted:
                effects.append((s, product, "bought_together", 1))
            qty = 1
            if want_qty and want_qty > 1:                          # planned quantity: what stock and budget allow
                extra = min(want_qty - 1, int(inv.stock[product]))
                if budget < np.inf and prices is not None:
                    extra = min(extra, max(0, int((budget - spent) // max(prices[product], 0.01))))
                if extra > 0:
                    if prices is not None:
                        spent += extra * prices[product]
                    for _ in range(extra):
                        take_unit(product, inv)
                    basket[product] += extra
                    qty += extra
            elif promo[product]:
                if switched and product == wanted:
                    effects.append((s, product, "brand_switch", 1))
                extra = min(poisson_from_u(STOCKPILE_RATE * sens, u[10]), inv.stock[product])
                if extra > 0 and budget < np.inf:                 # only what the budget allows
                    extra = min(extra, max(0, int((budget - spent) // max(prices[product], 0.01))))
                if extra > 0:                                      # stock up while it is cheap
                    if prices is not None:
                        spent += extra * prices[product]
                    for _ in range(extra):
                        take_unit(product, inv)
                    basket[product] += extra
                    pantry[product] = day + int(extra * shoppers.gap_mean[s])
                    effects.append((s, product, "stock_up", extra))
                    qty += extra
            record(day, s, "PURCHASE", product, qty=qty, promo=bool(promo[product]), route=route, stop=k)
    walk.picks = picks
    if NEEDS_ON:                                               # what they bought now lasts a while at home
        gap = shoppers.gap_mean[s]
        for (product, qty), j in zip(basket.items(), rng.random(len(basket))):
            lasts = HOME_FACTOR * store.product_cycle[product] * gap * qty * (0.5 + j)
            home[product] = max(home.get(product, -1), day + int(lasts))
    info = {"mission": trip_type, "routes": routes, "budget": budget, "spent": spent, "planned": n_items,
            "minds": minds, "not_sold": trip.get("not_sold", []), "list_by": trip.get("list_by")}
    return basket, misses, effects, walk, info


def trip_times(floor, walk, arrive):
    """Minute the shopper reaches each stop and the checkout: walking + time to pick items."""
    s = floor.settings
    per_metre = 1 / s["walk_speed_m_per_s"] / 60
    pause = np.bincount(walk.stop_of_item, minlength=len(walk.stops)) * s["seconds_per_item"] / 60
    reach = arrive + walk.cum_m * per_metre + np.r_[0, np.cumsum(pause)]
    impulse_time = max(walk.picks - len(walk.stop_of_item), 0) * s["seconds_per_item"] / 60
    reach[-1] += impulse_time                                  # unplanned picks take time too
    return reach, np.r_[pause, 0]


def simulate_day(day, store, shoppers, params, prices_g, regular, promo_g, tables, inv, mind=None):
    """Today's shoppers come in at their own time (earliest first), walk the store, shop, queue
    at a checkout and leave. With a traffic schedule (traffic.py) the number who come follows
    real store footfall, scaled to NPCs; without one, everyone whose trip is planned today comes.
    Each shopper sees their own group's prices (a targeted deal is seen by one group only).
    Evening: the store orders what is running low.
    mind (retina.mind.Mind, optional): thinking NPCs get their shelf choices from it. When the first
    one of a batch walks in, the lists of the next few thinking NPCs are made ahead (lists never look
    at the shelves, and each NPC keeps their own dice) and decided together in one LLM request.
    Without a mind nothing changes - the same data as before, byte for byte.
    Returns receipt lines, out-of-stock moments, today's stock numbers, behaviour effects, trips."""
    floor = store.floor
    recording = RECORDER["day"] == day
    if traffic.enabled():
        visitors = traffic.choose_visitors(day, shoppers.next_visit_day, traffic.npcs_today(day))
        hours_table = traffic.hour_table()
        floor.expected_trips_per_day = traffic.expected_npcs(day)   # lanes open to match today
    else:
        visitors = np.flatnonzero(shoppers.next_visit_day == day)
        hours_table = params["orders_by_hour"]
    dice, arrive = {}, {}
    for s in visitors:                                         # arrival: hour + minute
        dice[s] = stream(STREAM_TRIPS, day, s)                 # this shopper's own dice for today
        hour = int(sample_from(hours_table, 1, dice[s])[0])
        arrive[s] = hour * 60 + stream(STREAM_WALK, day, s).random() * 60
    order = sorted(visitors, key=lambda v: arrive[v])          # early shoppers find fuller shelves

    rows, misses, effects, trips, chunks, not_sold_trips, list_by = [], [], [], [], {}, [], []
    thinkers = [s for s in order if mind.thinks(s)] if mind is not None else []
    ready, ctx = {}, {"prices_g": prices_g, "regular": regular, "promo_g": promo_g, "tables": tables,
                      "hour": {s: int(arrive[s] // 60) for s in order}}
    for s in order:
        rng = dice[s]
        prices, promo = prices_g[shoppers.sensitivity_group[s]], promo_g[shoppers.sensitivity_group[s]]
        first = len(RECORDER["events"])
        record(day, s, "SHOPPER_ENTER", hour=int(arrive[s] // 60), group=int(shoppers.sensitivity_group[s]))
        thinking = mind is not None and mind.thinks(s)
        if thinking:
            if s not in ready:                                  # lists of the next batch, made ahead
                at = thinkers.index(s)
                batch = [t for t in thinkers[at:] if t not in ready][:mind.batch]
                lists = {t: make_list(t, day, store, shoppers, tables, promo_g[shoppers.sensitivity_group[t]],
                                      dice[t], prices_g[shoppers.sensitivity_group[t]]) for t in batch}
                ready.update(mind.decide(day, lists, ctx))
            trip, decisions = ready.pop(s)
            for event, product, extra in trip["log"]:
                record(day, s, event, product, **extra)
            basket, missed, eff, walk, info = walk_shelves(s, day, store, shoppers, tables, inv, promo, rng, trip,
                                                           recording, prices, decisions)
        else:
            basket, missed, eff, walk, info = fill_basket(s, day, store, shoppers, tables, inv, promo, rng,
                                                          recording, prices)
        shoppers.last_trip_day[s] = day
        reach, pause = trip_times(floor, walk, arrive[s])
        misses.extend((day, int(arrive[s] // 60), *m) for m in missed)
        effects.extend((day, *e) for e in eff)
        trip_id = f"D{day:03d}-S{s:04d}"
        for product, qty in basket.items():
            is_repeat = product in shoppers.counts[s]
            row = (day, int(arrive[s] // 60), trip_id, s, product, qty, prices[product],
                   regular[product], promo[product], is_repeat, info["mission"],
                   info["routes"].get(product, "new"))
            if mind is not None:                               # who decided, and why (thinking runs)
                d = info["minds"].get(product)
                row += (d["mind"], d["why_type"], d["why"]) if d else ("engine", "", "")
            rows.append(row)
        if thinking:
            mind.after_trip(day, s, int(arrive[s] // 60), basket, info, missed, prices, regular, promo)
        floor.traffic[walk.passed] += 1                        # who walked past each aisle
        bought = np.unique(floor.product_aisle[list(basket)]) if basket else []
        floor.buyers[bought] += 1                              # ... and who bought there
        items = int(sum(basket.values()))
        trips.append([day, trip_id, s, arrive[s], reach[-1], walk.cum_m[-1], len(walk.stops),
                      len(walk.passed), items, len(basket), info["mission"], info["planned"],
                      round(min(info["budget"], 1e6), 2), round(info["spent"], 2)])
        not_sold_trips.append(info["not_sold"])
        list_by.append(info.get("list_by"))
        if recording:
            chunk = RECORDER["events"][first:]
            del RECORDER["events"][first:]
            chunk[0]["mission"] = info["mission"]
            for e in chunk:
                if e["event"] == "PURCHASE":
                    e["price"] = round(float(prices[e["product"]]), 2)
            chunks[s] = (chunk, walk, reach, pause, basket)
        add_favorites(store, shoppers, s, basket)
        gap = max(1, rng.poisson(shoppers.gap_mean[s]))          # plan the next trip
        shoppers.next_visit_day[s] = plan_visit(day + gap, day + 1,
                                                shoppers.favorite_weekday[s], rng)

    # Checkout: first come, first served at the lanes the store has open that hour
    trips = pd.DataFrame(trips, columns=["day", "trip_id", "shopper_id", "arrive_min", "ready_min",
                                         "walk_m", "shelves_visited", "aisles_seen", "items", "lines",
                                         "mission", "planned_items", "trip_budget", "spent"])
    lane, wait, leave = floor.checkout_queue(trips["ready_min"].to_numpy(), trips["items"].to_numpy())
    trips["lane"], trips["wait_min"], trips["leave_min"] = lane, wait, leave
    trips["minutes_in_store"] = trips["leave_min"] - trips["arrive_min"]
    if NOT_SOLD_HERE:                                          # custom store: wants it does not sell
        trips["not_sold_here"] = [len(x) for x in not_sold_trips]
        trips["not_sold_aisles"] = ["; ".join(x) for x in not_sold_trips]
    if any(list_by):                                           # student runs: who planned each list
        trips["list_planned_by"] = [b or "store rule" for b in list_by]
    if recording:
        add_trip_events(day, floor, trips, chunks)

    # Stock check at closing time (before tonight's orders arrive)
    empty = inv.stock == 0
    promo = np.logical_or.reduce(promo_g)                     # on promotion for anyone
    eligible = inv.sellable & store.products["promo_eligible"].to_numpy()
    stock_today = {"day": day,
                   "empty_share": true_share(empty[inv.sellable]),
                   "empty_share_promo": true_share(empty[eligible & promo]),       # empty (NaN) on a day with
                   "empty_share_no_promo": true_share(empty[eligible & ~promo])}   # no deals - small ranges
    place_orders(day, inv, stream(STREAM_DELIVERIES, day))
    columns = ["day", "hour", "trip_id", "shopper_id", "product_idx", "quantity",
               "unit_price", "regular_price", "on_promo", "is_repeat", "mission", "reason"]
    if mind is not None:
        columns += ["mind", "why_type", "why"]
    miss_columns = ["day", "hour", "shopper_id", "product_idx", "reaction", "substitute_idx"]
    return (pd.DataFrame(rows, columns=columns), pd.DataFrame(misses, columns=miss_columns),
            stock_today, effects, trips)


def add_trip_events(day, floor, trips, chunks):
    """Replay day only: give every event its minute and place, add the route, checkout and exit."""
    per_metre = 1 / floor.settings["walk_speed_m_per_s"] / 60
    for row in trips.itertuples():
        chunk, walk, reach, pause, basket = chunks[row.shopper_id]
        t0 = row.arrive_min
        # route with times: walk between corners, pause at each shelf
        lane_x, mid_x = float(floor.lanes_x[row.lane]), float(floor.lanes_x.mean())
        path, t, k = [], t0, 0
        prev = walk.points[0]
        path.append([round(t, 2), round(prev[0], 1), round(prev[1], 1)])
        for p in walk.points[1:]:
            if k == len(walk.stops):                           # last leg: head for their lane
                p = (lane_x if abs(p[0] - mid_x) < 1e-6 else p[0],
                     floor.lane_y - 1.5 if p[2] == -2 else p[1], p[2])
            t += (abs(p[0] - prev[0]) + abs(p[1] - prev[1])) * per_metre
            if p[2] != -9 and k < len(walk.stops):            # a shelf on the list
                t = reach[k]
                path.append([round(t, 2), round(p[0], 1), round(p[1], 1)])
                t += pause[k]
                k += 1
            elif p[2] != -9:                                   # joins the queue at their lane
                t = max(t, reach[-1])
            path.append([round(t, 2), round(p[0], 1), round(p[1], 1)])
            prev = p
        path.append([round(max(t, row.ready_min + row.wait_min), 2), round(lane_x, 1), round(floor.lane_y, 1)])
        path.append([round(row.leave_min, 2), round(lane_x, 1), round(floor.lane_y + 1.5, 1)])
        path.append([round(row.leave_min + 1.0, 2), round(floor.exit[0], 1), round(floor.exit[1], 1)])
        for e in chunk:
            if e["event"] == "SHOPPER_ENTER":
                e.update(t=round(t0, 2), path=path, seen=walk.passed.tolist(), minutes=round(float(row.minutes_in_store), 1))
            elif "stop" in e:
                a = int(walk.stops[e["stop"]])
                e.update(t=round(reach[e["stop"]], 2), x=round(floor.ax[a], 1), y=round(floor.ay[a], 1))
            elif "leg" in e:
                a = e["aisle"]
                before = reach[e["leg"] - 1] + pause[e["leg"] - 1] if e["leg"] > 0 else t0
                e.update(t=round((before + reach[e["leg"]]) / 2, 2), x=round(floor.ax[a], 1), y=round(floor.ay[a], 1))
            else:
                e.update(t=round(t0, 2))
        chunk.append({"shopper": int(row.shopper_id), "event": "CHECKOUT", "product": -1,
                      "items": int(row.items), "total": round(float(sum(
                          e["qty"] * e["price"] for e in chunk if e["event"] == "PURCHASE")), 2),
                      "lane": int(row.lane), "wait_min": round(float(row.wait_min), 2),
                      "t": round(float(row.ready_min + row.wait_min), 2)})
        chunk.append({"shopper": int(row.shopper_id), "event": "LEAVE", "product": -1,
                      "t": round(float(row.leave_min + 1.0), 2)})
        RECORDER["events"].extend(chunk)
