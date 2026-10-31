"""RETINA real-time engine: every shopper on ONE clock, like NPCs in a game.

The classic engine (shopping.simulate_day) finishes one shopper's whole trip - every shelf, every unit
taken - before the next shopper starts. Here the store runs as a game loop on a shared clock (an
event-driven simulation: the clock jumps to the next thing that happens):

  walks in      -> the list student sets goals from that NPC's state
  chooses route -> the list student rescores remaining goals against current stock, prices and budget;
                   walking distance also matters. This happens again after every shelf.
  sees another shelf -> the list student may add an unplanned browse stop on the next leg
  reaches shelf -> the shelf student chooses a product AT THAT MINUTE, after other shoppers have acted
  completes basket -> the trip-end student may choose checkout before visiting every planned aisle
  reaches the checkout -> joins the queue at the open lanes, in the order shoppers really arrive there
  pays          -> the receipt lines (sales rows) are written at that minute
  leaves

So shoppers compete for the last units on a shelf, empty shelves meet the next shopper, and the
sales rows appear in the order they are paid. The engine enforces stock and budget and runs the
checkout; both rule and student NPCs use the same live action loop.

on_event(kind, minute, data) is called as things happen - the live stream for the 3D store page:
  "day"      {day, date}                                        the store opens
  "enter"    {shopper, mission, group, path: [[minute, x, y]] at the entrance}
  "move"     {shopper, path: [...], to, brain}                  next destination chosen in the store
  "shelf"    {shopper, decisions: [{mind, product, name, why}], events: [...]} choices and actions now
  "checkout" {shopper, lane, start, served}                     queue: starts paying at `start`
  "pay"      {shopper, trip_id, customer, lines: [{product, name, qty, price, promo}], total}
  "leave"    {shopper}
If on_event has a pace(day, minute) method, the clock waits for it before each moment (watching live).
"""

import heapq
import itertools

import numpy as np
import pandas as pd

from .config import *
from . import config
from .world import *
from . import traffic
from .floor import STREAM_WALK
from .shopping import RECORDER, record, make_list, make_trip_context, trip_times, add_trip_events, true_share
from .student_agent import NPCAgent


def simulate_day_realtime(day, store, shoppers, params, prices_g, regular, promo_g, tables, inv, mind=None,
                          on_event=None):
    """Same inputs and outputs as shopping.simulate_day; every shopper runs on one clock."""
    if mind is not None and mind.mode != "student":
        raise ValueError("the real-time engine runs local student or engine-only stores; LLM batch modes use the classic engine")
    floor = store.floor
    settings = floor.settings
    per_metre = 1 / settings["walk_speed_m_per_s"] / 60
    per_item = settings["seconds_per_item"] / 60
    streaming = on_event is not None and not getattr(on_event, "payments_only", False)
    if streaming:                                              # a new store day starts with an empty log
        RECORDER.update(day=day, events=[])
    elif RECORDER["day"] == day and RECORDER["events"]:
        # A completed streamed run can leave its replay in this process-wide log.
        # A new non-streamed run on the same day must not inherit those shoppers.
        # Explicit focus/replay recording starts with an empty event list.
        RECORDER.update(day=None, events=[])
    recording = RECORDER["day"] == day
    emit = on_event or (lambda kind, minute, data: None)
    pace = getattr(on_event, "pace", None)
    names = store.products["product_name"]
    name = lambda p: str(names.iat[p]) if p >= 0 else ""
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
    start_minute = int(getattr(config, "LIVE_START_MINUTE", 0))
    visitors = np.asarray([s for s in visitors if arrive[s] >= start_minute], dtype=int)
    floor.expected_trips_per_day = len(visitors)
    ctx = {"prices_g": prices_g, "regular": regular, "promo_g": promo_g, "tables": tables,
           "hour": {s: int(arrive[s] // 60) for s in visitors}, "minute": {}}

    clock, order = [], itertools.count()                       # (minute, tie-break, what, shopper)
    for s in visitors:
        heapq.heappush(clock, (arrive[s], next(order), "enter", s))
    agents, free = {}, np.zeros(settings["checkout_lanes"])    # minute each lane is free again
    rows, misses, effects, trips, chunks, not_sold, list_by = [], [], [], [], {}, [], []

    def npc_move(s, a, minute):
        n = len(RECORDER["events"])
        move = a["agent"].next_move(minute, per_metre)
        if recording:
            a["chunk"].extend(RECORDER["events"][n:])
            del RECORDER["events"][n:]
        if move["kind"] == "checkout":
            a["result"] = a["agent"].finish()
        if streaming:
            emit("move", minute, {"shopper": int(s), "path": move["path"],
                                   "to": move["aisle"] if move["aisle"] is not None else "checkout",
                                   "to_name": floor.aisle_names[move["aisle"]] if move["aisle"] is not None else "checkout",
                                   "brain": move["brain"], "decision": move["decision"],
                                   "lines": len(a["agent"].basket),
                                   "browse": dict(move["browse"], name=name(move["browse"]["product"]))
                                   if move["browse"] else None})
        heapq.heappush(clock, (move["minute"], next(order), move["kind"], s))

    if pace is not None:
        pace(day, start_minute)
    emit("day", start_minute, {"day": int(day),
                               "date": str(pd.Timestamp(START_DATE) + pd.Timedelta(days=int(day)))[:10],
                               "wall_started_at": getattr(on_event, "wall_started_at", None),
                               "clock_rate": getattr(on_event, "rate", None)})
    last_minute = start_minute
    while clock:
        t, _, what, s = heapq.heappop(clock)
        last_minute = t
        if pace is not None:
            pace(day, t)
        if what == "enter":
            rng, g = dice[s], shoppers.sensitivity_group[s]
            prices, promo = prices_g[g], promo_g[g]
            a = agents[s] = {"chunk": [], "prices": prices, "promo": promo, "group": g}
            n = len(RECORDER["events"])
            record(day, s, "SHOPPER_ENTER", hour=int(arrive[s] // 60), group=int(g))
            # A trained list brain starts with requirements only. It chooses the
            # products itself at this entrance; the rule list is not generated.
            trained_list = mind is not None and getattr(mind, "lister", None) is not None
            trip = (make_trip_context(s, day, shoppers, rng, prices) if trained_list else
                    make_list(s, day, store, shoppers, tables, promo, rng, prices))
            decisions = None
            if mind is not None:                               # the list student plans the list now
                mind.start_trip(day, s, trip, ctx)
                decisions = mind.at_shelf(day, s, trip, ctx)   # ... the shelf student decides at each shelf
            for event, product, info in trip["log"]:
                record(day, s, event, product, **info)
            if recording:
                a["chunk"].extend(RECORDER["events"][n:])
                del RECORDER["events"][n:]
            a["mission"] = trip["mission"]
            a["requirements"] = trip["requirements"]
            a["list_brain"] = "student" if trained_list else "rule"
            a["agent"] = NPCAgent(day, s, store, shoppers, tables, inv, promo, prices, rng,
                                   trip, mind, decisions)
            if streaming:
                emit("enter", arrive[s], {"shopper": int(s), "mission": a["mission"], "group": int(g),
                                          "list_brain": a["list_brain"],
                                          "requirements": a["requirements"].public(store),
                                          "path": [[round(float(arrive[s]), 2),
                                                    round(float(floor.entrance[0]), 1),
                                                    round(float(floor.entrance[1]), 1)]]})
            npc_move(s, a, arrive[s])
        elif what == "shelf":
            a = agents[s]
            n = len(a["chunk"])
            ctx["hour"][s] = int(t // 60)
            ctx["minute"][s] = round(float(t), 2)
            items = a["agent"].shop(a["agent"].stops[-1])
            if recording:
                a["chunk"].extend(RECORDER["events"])
                RECORDER["events"].clear()
            heapq.heappush(clock, (t + items * per_item, next(order), "choose", s))
            if streaming:                                      # what they just did at this shelf
                choices = [dict(d, name=name(d["product"] if d["product"] >= 0 else d["wanted"]))
                           for d in a["agent"].last_decisions]
                did = []
                for e in a["chunk"][n:]:
                    if e["event"] in ("MOVE_TO_DEPARTMENT", "VIEW_PRODUCT"):
                        continue
                    e = dict(e, name=name(e["product"]), instead_of_name=name(e.get("instead_of", -1)))
                    if e["event"] == "PURCHASE":
                        e["price"] = round(float(a["prices"][e["product"]]), 2)
                    did.append(e)
                aisle = a["agent"].stops[-1]
                emit("shelf", t, {"shopper": int(s), "aisle": int(aisle),
                                  "aisle_name": floor.aisle_names[aisle],
                                  "decisions": choices, "events": did})
        elif what == "choose":
            npc_move(s, agents[s], t)
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
            emit("checkout", t, {"shopper": int(s), "lane": j, "start": round(start, 2), "served": round(served, 2)})
        elif what == "pay":                                    # the receipt lines are written now
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
                paid.append({"product": int(product), "product_id": str(store.products["product_id"].iat[product]),
                             "name": name(product), "qty": int(qty),
                             "price": round(float(prices[product]), 2), "promo": bool(promo[product]),
                             "mind": d["mind"] if mind is not None and d else "rule"})
            if mind is not None:
                mind.after_trip(day, s, int(arrive[s] // 60), basket, info, missed, prices, regular, promo)
            floor.traffic[walk.passed] += 1
            floor.buyers[np.unique(floor.product_aisle[list(basket)]) if basket else []] += 1
            items = int(sum(basket.values()))
            trips.append([day, trip_id, s, arrive[s], a["ready"], walk.cum_m[-1], len(walk.stops), len(walk.passed),
                          items, len(basket), info["mission"], info["planned"], round(min(info["budget"], 1e6), 2),
                          round(info["spent"], 2), a["lane"], a["wait"], a["leave"],
                          len(info["requirements"].essential_products), info["requirements"].bought_count(basket)])
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
            emit("pay", t, {"shopper": int(s), "trip_id": trip_id, "customer": f"C{int(s):05d}",
                            "lane": int(a["lane"]), "lines": paid, "list_brain": a["list_brain"],
                            "essential_items": len(info["requirements"].essential_products),
                            "essential_items_bought": info["requirements"].bought_count(basket),
                            "total": round(float(sum(x["qty"] * x["price"] for x in paid)), 2)})
            heapq.heappush(clock, (t + 1.0, next(order), "leave", s))
        elif what == "leave":
            emit("leave", t, {"shopper": int(s)})

    close_minute = int(TRAFFIC["open_hours"][1]) * 60 if traffic.enabled() else last_minute
    end_minute = max(last_minute, close_minute)
    if pace is not None:
        pace(day, end_minute)
    emit("day_end", end_minute, {"day": int(day)})

    trips = pd.DataFrame(trips, columns=["day", "trip_id", "shopper_id", "arrive_min", "ready_min", "walk_m",
                                         "shelves_visited", "aisles_seen", "items", "lines", "mission",
                                         "planned_items", "trip_budget", "spent", "lane", "wait_min", "leave_min",
                                         "essential_items", "essential_items_bought"])
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
