"""RETINA real-time engine: every shopper on ONE clock, like NPCs in a game.

The classic engine (shopping.simulate_day) finishes one shopper's whole trip - every shelf, every unit
taken - before the next shopper starts. Here the store runs as a game loop on a shared clock (an
event-driven simulation: the clock jumps to the next thing that happens):

  walks in      -> the list is made (in student runs the LIST STUDENT plans it, from the NPC's own state)
  reaches shelf -> takes what is on the shelf AT THAT MINUTE (in student runs the SHELF STUDENT decides
                   now, seeing today's prices and the stock left by the shoppers before)
  reaches the checkout -> joins the queue at the open lanes, in the order shoppers really arrive there
  pays          -> the receipt lines (sales rows) are written at that minute
  leaves

So shoppers compete for the last units on a shelf, empty shelves meet the next shopper, and the
sales rows appear in the order they are paid. The shopping rules themselves (lists, prices, empty-
shelf reactions, deals, budgets) are the classic engine's own code (shopping.walk_steps), unchanged.

on_event(kind, minute, data) is called as things happen ("enter", "shelf", "pay", "leave") - the live
stream for the 3D store page.
"""

import heapq
import itertools

import numpy as np
import pandas as pd

from .config import *
from .world import *
from . import traffic
from .floor import STREAM_WALK
from .shopping import RECORDER, record, make_list, walk_steps, trip_times, add_trip_events, true_share


def simulate_day_realtime(day, store, shoppers, params, prices_g, regular, promo_g, tables, inv, mind=None,
                          on_event=None):
    """Same inputs and outputs as shopping.simulate_day; every shopper runs on one clock."""
    if mind is not None and mind.mode != "student":
        raise ValueError("the real-time engine runs student or engine-only stores (an LLM decides in batches)")
    floor = store.floor
    settings = floor.settings
    per_metre = 1 / settings["walk_speed_m_per_s"] / 60
    per_item = settings["seconds_per_item"] / 60
    recording = RECORDER["day"] == day
    emit = on_event or (lambda kind, minute, data: None)
    if traffic.enabled():                                      # who comes today and when: as the classic engine
        visitors = traffic.choose_visitors(day, shoppers.next_visit_day, traffic.npcs_today(day))
        hours_table = traffic.hour_table()
        floor.expected_trips_per_day = traffic.expected_npcs(day)
    else:
        visitors = np.flatnonzero(shoppers.next_visit_day == day)
        hours_table = params["orders_by_hour"]
    dice, arrive = {}, {}
    for s in visitors:
        dice[s] = stream(STREAM_TRIPS, day, s)
        hour = int(sample_from(hours_table, 1, dice[s])[0])
        arrive[s] = hour * 60 + stream(STREAM_WALK, day, s).random() * 60
    ctx = {"prices_g": prices_g, "regular": regular, "promo_g": promo_g, "tables": tables,
           "hour": {s: int(arrive[s] // 60) for s in visitors}}

    clock, order = [], itertools.count()                       # (minute, tie-break, what, shopper)
    for s in visitors:
        heapq.heappush(clock, (arrive[s], next(order), "enter", s))
    agents, free = {}, np.zeros(settings["checkout_lanes"])    # minute each lane is free again
    rows, misses, effects, trips, chunks, not_sold, list_by = [], [], [], [], {}, [], []

    def advance(a):
        """Let the shopper act until their next moment in the store (or the end of the walk)."""
        n = len(RECORDER["events"])
        try:
            step = next(a["walk"])
        except StopIteration as done:
            step, a["result"] = None, done.value
        if recording:                                          # keep each shopper's own story together
            a["chunk"].extend(RECORDER["events"][n:])
            del RECORDER["events"][n:]
        return step

    def schedule(s, a, step):
        if step is None:                                       # walked every shelf: to the checkout
            walk = a["result"][3]
            impulse = max(walk.picks - len(walk.stop_of_item), 0) * per_item
            heapq.heappush(clock, (a["reach"][-1] + impulse, next(order), "checkout", s))
            return
        kind, k = step
        if kind == "route":
            walk = k
            pause = np.bincount(walk.stop_of_item, minlength=len(walk.stops)) * per_item
            a["reach"] = arrive[s] + walk.cum_m * per_metre + np.r_[0, np.cumsum(pause)]
            a["pause"] = pause
            schedule(s, a, advance(a))
        elif kind == "stop":
            heapq.heappush(clock, (a["reach"][k], next(order), "shelf", s))
        else:                                                  # deals on the way: halfway along the leg
            before = a["reach"][k - 1] + a["pause"][k - 1] if k > 0 else arrive[s]
            heapq.heappush(clock, ((before + a["reach"][k]) / 2, next(order), "shelf", s))

    while clock:
        t, _, what, s = heapq.heappop(clock)
        if what == "enter":
            rng, g = dice[s], shoppers.sensitivity_group[s]
            prices, promo = prices_g[g], promo_g[g]
            a = agents[s] = {"chunk": [], "prices": prices, "promo": promo}
            n = len(RECORDER["events"])
            record(day, s, "SHOPPER_ENTER", hour=int(arrive[s] // 60), group=int(g))
            trip = make_list(s, day, store, shoppers, tables, promo, rng, prices)
            decisions = None
            if mind is not None:                               # the list student plans the list now
                mind.start_trip(day, s, trip, ctx)
                decisions = mind.at_shelf(day, s, trip, ctx)   # ... the shelf student decides at each shelf
            for event, product, info in trip["log"]:
                record(day, s, event, product, **info)
            if recording:
                a["chunk"].extend(RECORDER["events"][n:])
                del RECORDER["events"][n:]
            a["walk"] = walk_steps(s, day, store, shoppers, tables, inv, promo, rng, trip, recording, prices, decisions)
            emit("enter", t, {"shopper": int(s), "mission": trip["mission"], "planned": trip["planned"]})
            schedule(s, a, advance(a))
        elif what == "shelf":
            a = agents[s]
            schedule(s, a, advance(a))
            emit("shelf", t, {"shopper": int(s)})
        elif what == "checkout":                               # first come, first served at the open lanes
            a = agents[s]
            basket = a["result"][0]
            open_lanes = floor.lanes_open(int(t // 60) % 24)
            j = int(np.argmin(free[:open_lanes]))
            start = max(t, free[j])
            served = start + (settings["checkout_seconds_base"] + settings["checkout_seconds_per_item"]
                              * int(sum(basket.values()))) / 60
            free[j] = served
            a.update(ready=t, lane=j, wait=start - t, leave=served)
            heapq.heappush(clock, (served, next(order), "pay", s))
        else:                                                  # "pay": the receipt lines are written now
            a = agents.pop(s)
            basket, missed, eff, walk, info = a["result"]
            prices, promo, rng = a["prices"], a["promo"], dice[s]
            shoppers.last_trip_day[s] = day
            misses.extend((day, int(arrive[s] // 60), *m) for m in missed)
            effects.extend((day, *e) for e in eff)
            trip_id = f"D{day:03d}-S{s:04d}"
            paid = []
            for product, qty in basket.items():
                row = (day, int(arrive[s] // 60), trip_id, s, product, qty, prices[product], regular[product],
                       promo[product], product in shoppers.counts[s], info["mission"],
                       info["routes"].get(product, "new"))
                if mind is not None:
                    d = info["minds"].get(product)
                    row += (d["mind"], d["why_type"], d["why"]) if d else ("engine", "", "")
                rows.append(row)
                paid.append({"product": int(product), "qty": int(qty), "price": round(float(prices[product]), 2),
                             "promo": bool(promo[product])})
            if mind is not None:
                mind.after_trip(day, s, int(arrive[s] // 60), basket, info, missed, prices, regular, promo)
            floor.traffic[walk.passed] += 1
            floor.buyers[np.unique(floor.product_aisle[list(basket)]) if basket else []] += 1
            items = int(sum(basket.values()))
            trips.append([day, trip_id, s, arrive[s], a["ready"], walk.cum_m[-1], len(walk.stops), len(walk.passed),
                          items, len(basket), info["mission"], info["planned"], round(min(info["budget"], 1e6), 2),
                          round(info["spent"], 2), a["lane"], a["wait"], a["leave"]])
            not_sold.append(info["not_sold"])
            list_by.append(info.get("list_by"))
            if recording:
                chunk = a["chunk"]
                chunk[0]["mission"] = info["mission"]
                for e in chunk:
                    if e["event"] == "PURCHASE":
                        e["price"] = round(float(prices[e["product"]]), 2)
                reach, pause = trip_times(floor, walk, arrive[s])
                chunks[s] = (chunk, walk, reach, pause, basket)
            add_favorites(store, shoppers, s, basket)
            gap = max(1, rng.poisson(shoppers.gap_mean[s]))    # plan the next trip
            shoppers.next_visit_day[s] = plan_visit(day + gap, day + 1, shoppers.favorite_weekday[s], rng)
            emit("pay", t, {"shopper": int(s), "trip_id": trip_id, "lane": int(a["lane"]), "lines": paid,
                            "total": round(float(sum(x["qty"] * x["price"] for x in paid)), 2)})

    trips = pd.DataFrame(trips, columns=["day", "trip_id", "shopper_id", "arrive_min", "ready_min", "walk_m",
                                         "shelves_visited", "aisles_seen", "items", "lines", "mission",
                                         "planned_items", "trip_budget", "spent", "lane", "wait_min", "leave_min"])
    trips["minutes_in_store"] = trips["leave_min"] - trips["arrive_min"]
    if NOT_SOLD_HERE:
        trips["not_sold_here"] = [len(x) for x in not_sold]
        trips["not_sold_aisles"] = ["; ".join(x) for x in not_sold]
    if any(list_by):
        trips["list_planned_by"] = [b or "store rule" for b in list_by]
    if recording:
        add_trip_events(day, floor, trips, chunks)

    empty = inv.stock == 0                                     # closing time: stock check, tonight's orders
    promo_any = np.logical_or.reduce(promo_g)
    eligible = inv.sellable & store.products["promo_eligible"].to_numpy()
    stock_today = {"day": day, "empty_share": true_share(empty[inv.sellable]),
                   "empty_share_promo": true_share(empty[eligible & promo_any]),
                   "empty_share_no_promo": true_share(empty[eligible & ~promo_any])}
    place_orders(day, inv, stream(STREAM_DELIVERIES, day))
    columns = ["day", "hour", "trip_id", "shopper_id", "product_idx", "quantity", "unit_price", "regular_price",
               "on_promo", "is_repeat", "mission", "reason"]
    if mind is not None:
        columns += ["mind", "why_type", "why"]
    miss_columns = ["day", "hour", "shopper_id", "product_idx", "reaction", "substitute_idx"]
    return (pd.DataFrame(rows, columns=columns), pd.DataFrame(misses, columns=miss_columns), stock_today, effects,
            trips)
