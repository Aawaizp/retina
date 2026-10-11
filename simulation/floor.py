"""RETINA floor plan: where every aisle group sits in the store, how shoppers walk, checkout queues.

The store is a grid of long shelf racks with walkways (corridors) between them, a back aisle and
a front aisle - like most supermarkets. Fresh departments sit on the walls (produce next to the
entrance, meat/deli/bakery at the back, dairy on the far wall), packaged goods on the racks.
A shopper walks from shelf to shelf (nearest next shelf first) and SEES every shelf they pass:
unplanned buys come from those shelves, so the layout changes what people buy.
Everything here is in metres and minutes. Settings: the store profile's "floor" section.
"""

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import *

FLOOR_DEFAULTS = {                     # all ESTIMATED - chosen by us, see the store profile labels
    "corridors": 16,                   # walkways between shelf racks
    "corridor_spacing_m": 4.0,         # walkway + rack
    "floor_m_per_product": 0.06,       # shelf length along the floor per product (about 5 shelf levels)
    "walk_speed_m_per_s": 0.7,         # strolling with a basket or cart
    "seconds_per_item": 20,            # find, compare and pick one item
    "checkout_lanes": 8,               # lanes the store can open
    "checkout_seconds_base": 60,       # paying and bagging
    "checkout_seconds_per_item": 4,    # scanning
    "lane_target_busy": 0.7,           # managers open lanes so cashiers are ~70% busy
    "walls": {"left": ["produce"], "back": ["meat seafood", "deli", "bakery"], "right": ["dairy eggs"]},
    "rack_order": ["international", "pantry", "canned goods", "dry goods pasta", "breakfast", "snacks",
                   "beverages", "alcohol", "household", "personal care", "frozen"],
}
STREAM_WALK = 6       # random stream for arrival minutes (config.py uses keys 1-5)
RACK_W = 1.6          # shelf rack depth (both sides)
WALL_W = 0.8          # wall shelf depth
CROSS = 3.0           # back / front cross aisle width
SEE_M = 2.0           # shoppers notice shelves up to 2 m ahead / around a corner


def floor_settings():
    s = dict(FLOOR_DEFAULTS)
    s.update(PROFILE.get("floor", {}))
    return s


@dataclass
class Walk:
    """One shopper's route: the order of their stops and everything they walked past."""
    stops: np.ndarray          # aisle number of each stop, in walking order
    order: np.ndarray          # shopping-list items sorted into walking order
    stop_of_item: np.ndarray   # stop number of each shopping-list item
    cum_m: np.ndarray          # metres walked when reaching each stop (and the checkout, last)
    legs: list                 # aisles passed on the way to each stop (and to the checkout, last)
    passed: np.ndarray         # every aisle seen on this trip
    points: list = None        # corner points of the route (only for the replay)


@dataclass
class Floor:
    settings: dict
    width: float
    depth: float
    corridor_x: np.ndarray
    y_back: float
    y_front: float
    y_top: float
    y_bottom: float
    entrance: tuple
    exit: tuple
    lanes_x: np.ndarray
    lane_y: float
    aisle_names: list
    aisle_index: dict
    aisle_dept: list
    weight: np.ndarray         # share of all items bought in each aisle (for unplanned buys)
    ax: np.ndarray             # where a shopper stands to shop each aisle
    ay: np.ndarray
    ac: np.ndarray             # its corridor (-1 = back aisle)
    segments: list             # shelf pieces: (aisle, x0, y0, x1, y1, corridor or -1)
    corridor_segs: list        # per corridor: (aisles, y0, y1) of the shelves facing it
    back_segs: tuple           # (aisles, x0, x1) of the back-wall shelves
    product_aisle: np.ndarray  # aisle number of each product (-1 = not carried)
    traffic: np.ndarray = None     # trips that walked past each aisle
    buyers: np.ndarray = None      # trips that bought something there
    expected_trips_per_day: float = 0.0
    hour_share: np.ndarray = None

    # ------------------------------------------------------------ walking
    def distance(self, x1, y1, c1, x2, y2, c2):
        """Walking distance: along the same corridor, or round the back or the front aisle."""
        around = np.minimum(np.abs(y1 - self.y_back) + np.abs(y2 - self.y_back),
                            np.abs(y1 - self.y_front) + np.abs(y2 - self.y_front)) + np.abs(x1 - x2)
        return np.where((c1 == c2) & (c1 >= 0), np.abs(y1 - y2), around)

    def _dist(self, p, a):
        """Walking distance from point p to the shopping spot of aisle a (plain Python: fast)."""
        x2, y2, c2 = self._spots[a]
        x1, y1, c1 = p
        if c1 == c2 and c1 >= 0:
            return abs(y1 - y2)
        yb, yf = self.y_back, self.y_front
        return min(abs(y1 - yb) + abs(y2 - yb), abs(y1 - yf) + abs(y2 - yf)) + abs(x1 - x2)

    def _corridor_pass(self, c, ya, yb):
        if c < 0:
            return []
        lo, hi = min(ya, yb) - SEE_M, max(ya, yb) + SEE_M
        return [a for a, y0, y1 in self._corridor_lists[c] if y1 >= lo and y0 <= hi]

    def leg(self, p, q, with_points=False):
        """Walk from point p to point q (x, y, corridor). Returns metres, aisles passed, corners."""
        (x1, y1, c1), (x2, y2, c2) = p, q
        if c1 == c2 and c1 >= 0:
            return abs(y1 - y2), self._corridor_pass(c1, y1, y2), ([q] if with_points else None)
        back = abs(y1 - self.y_back) + abs(y2 - self.y_back)
        front = abs(y1 - self.y_front) + abs(y2 - self.y_front)
        turn = self.y_back if back <= front else self.y_front
        passed = self._corridor_pass(c1, y1, turn) + self._corridor_pass(c2, turn, y2)
        if turn == self.y_back:                              # the back aisle runs along the back wall
            lo, hi = min(x1, x2) - SEE_M, max(x1, x2) + SEE_M
            passed += [a for a, x0, xe in self._back_list if xe >= lo and x0 <= hi]
        pts = [(x1, turn, -9), (x2, turn, -9), q] if with_points else None
        return min(back, front) + abs(x1 - x2), passed, pts

    def plan_route(self, item_aisles, with_points=False):
        """Order the stops (always walk to the nearest shelf still on the list), entrance -> checkout."""
        item_aisles = [int(a) for a in item_aisles]
        todo = list(dict.fromkeys(item_aisles))              # unique aisles, first-seen order
        pos, stops = self.entrance, []
        while todo:
            best = min(range(len(todo)), key=lambda j: self._dist(pos, todo[j]))
            nxt = todo.pop(best)
            stops.append(nxt)
            pos = self._spots[nxt]
        rank = {a: k for k, a in enumerate(stops)}
        stop_of_item = np.array([rank[a] for a in item_aisles], dtype=int)
        order = np.argsort(stop_of_item, kind="stable")
        # walk it: entrance -> stop 1 -> ... -> checkout lane area
        pos, metres, cum, legs, points = self.entrance, 0.0, [], [], ([self.entrance] if with_points else None)
        seen = set(stops)
        for target in [self._spots[a] for a in stops] + [self._checkout]:
            m, passed, pts = self.leg(pos, target, with_points)
            metres += m
            cum.append(metres)
            legs.append(passed)
            seen.update(passed)
            if with_points:
                points.extend(pts)
            pos = target
        return Walk(np.array(stops, dtype=int), order, stop_of_item, np.array(cum), legs,
                    np.array(sorted(seen), dtype=int), points)

    def prepare(self):
        """Plain-Python copies of the shelf positions (fast for short lists)."""
        self._spots = [(float(x), float(y), int(c)) for x, y, c in zip(self.ax, self.ay, self.ac)]
        self._corridor_lists = [list(zip(a.tolist(), y0.tolist(), y1.tolist())) for a, y0, y1 in self.corridor_segs]
        self._back_list = list(zip(*(v.tolist() for v in self.back_segs)))
        self._checkout = (float(self.lanes_x.mean()), float(self.lane_y), -2)

    # ------------------------------------------------------------ checkout
    def lanes_open(self, hour):
        s = self.settings
        service = s["checkout_seconds_base"] + s["checkout_seconds_per_item"] * 10
        need = self.expected_trips_per_day * self.hour_share[hour] * service / 3600 / s["lane_target_busy"]
        return int(np.clip(np.ceil(need), 1, s["checkout_lanes"]))

    def checkout_queue(self, ready, items):
        """First come, first served at the open lanes. ready = minute each shopper reaches the
        checkout. Returns lane, minutes waited and minute they leave, per shopper."""
        s = self.settings
        n = len(ready)
        lane, wait, leave = np.zeros(n, dtype=int), np.zeros(n), np.zeros(n)
        free = np.zeros(s["checkout_lanes"])                 # minute each lane is free again
        for i in np.argsort(ready, kind="stable"):
            k = self.lanes_open(int(ready[i] // 60) % 24)
            j = int(np.argmin(free[:k]))
            start = max(ready[i], free[j])
            service = (s["checkout_seconds_base"] + s["checkout_seconds_per_item"] * items[i]) / 60
            lane[i], wait[i], leave[i] = j, start - ready[i], start + service
            free[j] = start + service
        return lane, wait, leave


# ================================================================ building the floor
def build_floor(store, shoppers, params):
    """Place every aisle group on the walls or racks; shelf length follows the number of
    products (packaged goods) or the share of sales (fresh walls)."""
    s = floor_settings()
    K, sp, mpp = s["corridors"], s["corridor_spacing_m"], s["floor_m_per_product"]
    depts = store.departments
    on_walls = {w: [d for d in s["walls"].get(w, []) if d in depts] for w in ("left", "back", "right")}
    wall_depts = {d for ds in on_walls.values() for d in ds}
    racks = [d for d in s["rack_order"] if d in depts and d not in wall_depts]
    racks += [d for d in sorted(depts, key=lambda n: -depts[n].share) if d not in wall_depts and d not in racks]

    # rack length L: all rack products must fit on the 2K - 2 corridor sides
    rack_products = sum(len(depts[d].product_idx) for d in racks)
    L = float(np.clip(rack_products * mpp / (2 * K - 2), 12, 80))
    width = K * sp
    y_back = WALL_W + CROSS / 2
    y_top = WALL_W + CROSS
    y_bottom = y_top + L
    y_front = y_bottom + CROSS / 2
    lane_y = y_bottom + CROSS + 2.5
    depth = lane_y + 6.0
    corridor_x = sp * (np.arange(K) + 0.5)

    # every aisle group: its share of the store's items, its number of products
    names, dept_of, weight, n_products = [], [], [], []
    for d in store.dept_names:
        dep = depts[d]
        shares = np.diff(np.r_[0, dep.aisle_cum])
        for a, sh in zip(dep.aisle_names, shares):
            names.append(a)
            dept_of.append(d)
            weight.append(dep.share * sh)
            n_products.append(len(store.aisles[a].product_idx))
    index = {a: i for i, a in enumerate(names)}
    weight, n_products = np.array(weight), np.array(n_products, dtype=float)

    # faces: pieces of shelf along which aisles are laid out, in walking order
    def rect_for(face, s0, s1):
        if face["kind"] == "v":
            a, b = face["a"], face["b"]
            ya, yb = a + (b - a) * s0, a + (b - a) * s1
            return face["x0"], min(ya, yb), face["x1"], max(ya, yb)
        return face["a"] + (face["b"] - face["a"]) * s0, 0.0, face["a"] + (face["b"] - face["a"]) * s1, WALL_W

    def shelf_rect(c, side):
        """x-range of the shelf on one side of corridor c."""
        edge = corridor_x[c] + side * sp / 2
        if c == 0 and side < 0:
            return 0.0, WALL_W
        if c == K - 1 and side > 0:
            return width - WALL_W, width
        return (edge, edge + RACK_W / 2) if side < 0 else (edge - RACK_W / 2, edge)

    def vface(c, side, front_to_back):
        x0, x1 = shelf_rect(c, side)
        return {"kind": "v", "c": c, "x0": x0, "x1": x1,
                "a": y_bottom if front_to_back else y_top, "b": y_top if front_to_back else y_bottom}

    walls = {"left": [vface(0, -1, True)],
             "back": [{"kind": "h", "c": -1, "a": 0.0, "b": width}],
             "right": [vface(K - 1, +1, False)]}
    rack_faces = []
    for c in range(K):
        sides = ([+1] if c == 0 else [-1, +1]) if c < K - 1 else [-1]
        for k, side in enumerate(sides):
            rack_faces.append(vface(c, side, (c + k) % 2 == 0))

    segments, access = [], {}
    best_len = {}

    def lay_out(faces, aisles, lengths):
        """Lay aisles end to end along the faces; lengths are scaled to fill them exactly."""
        face_len = np.array([abs(f["b"] - f["a"]) for f in faces])
        lengths = np.asarray(lengths, dtype=float)
        lengths = lengths / lengths.sum() * face_len.sum()
        starts = np.r_[0, np.cumsum(face_len)]
        pos = 0.0
        for a, ln in zip(aisles, lengths):
            lo, hi = pos, pos + ln
            for f, (fs, fe) in enumerate(zip(starts[:-1], starts[1:])):
                p0, p1 = max(lo, fs), min(hi, fe)
                if p1 - p0 <= 1e-9:
                    continue
                face = faces[f]
                x0, y0, x1, y1 = rect_for(face, (p0 - fs) / face_len[f], (p1 - fs) / face_len[f])
                segments.append((a, x0, y0, x1, y1, face["c"]))
                if p1 - p0 > best_len.get(a, 0):            # shop it at its longest piece
                    best_len[a] = p1 - p0
                    if face["kind"] == "v":
                        access[a] = (corridor_x[face["c"]], (y0 + y1) / 2, face["c"])
                    else:
                        access[a] = ((x0 + x1) / 2, y_back, -1)
            pos = hi

    for wall in ("left", "back", "right"):
        aisles = [index[a] for d in on_walls[wall] for a in depts[d].aisle_names]
        if aisles:
            lay_out(walls[wall], aisles, weight[aisles])        # fresh walls: space follows sales
    aisles = [index[a] for d in racks for a in depts[d].aisle_names]
    lay_out(rack_faces, aisles, np.maximum(n_products[aisles], 3))   # racks: space follows products

    n = len(names)
    ax, ay, ac = np.zeros(n), np.zeros(n), np.full(n, -1)
    for a, (x, y, c) in access.items():
        ax[a], ay[a], ac[a] = x, y, c
    corridor_segs = []
    for c in range(K):
        own = [sg for sg in segments if sg[5] == c]
        corridor_segs.append((np.array([sg[0] for sg in own], dtype=int),
                              np.array([sg[2] for sg in own]), np.array([sg[4] for sg in own])))
    back = [sg for sg in segments if sg[5] == -1]
    back_segs = (np.array([sg[0] for sg in back], dtype=int),
                 np.array([sg[1] for sg in back]), np.array([sg[3] for sg in back]))
    product_aisle = np.full(len(store.products), -1)
    for a, i in index.items():
        product_aisle[store.aisles[a].product_idx] = i
    hours = np.zeros(24)
    for h, v in params["orders_by_hour"].items():
        hours[int(h)] = v
    floor = Floor(settings=s, width=width, depth=depth, corridor_x=corridor_x, y_back=y_back,
                  y_front=y_front, y_top=y_top, y_bottom=y_bottom,
                  entrance=(2.0, depth - 2.0, -2), exit=(width - 2.0, depth - 2.0, -2),
                  lanes_x=np.linspace(0.35 * width, 0.8 * width, s["checkout_lanes"]), lane_y=lane_y,
                  aisle_names=names, aisle_index=index, aisle_dept=dept_of, weight=weight,
                  ax=ax, ay=ay, ac=ac, segments=segments, corridor_segs=corridor_segs,
                  back_segs=back_segs, product_aisle=product_aisle,
                  traffic=np.zeros(n, dtype=np.int64), buyers=np.zeros(n, dtype=np.int64),
                  expected_trips_per_day=float((1 / shoppers.gap_mean).sum()),
                  hour_share=hours / hours.sum())
    floor.prepare()
    return floor


def summarize_floor(floor):
    print(f"\nFloor (v5): {floor.width:.0f} m x {floor.depth:.0f} m "
          f"({floor.width * floor.depth:,.0f} m2), {len(floor.corridor_x)} walkways, "
          f"{len(floor.aisle_names)} aisle groups on {len(floor.segments)} shelf pieces, "
          f"{floor.settings['checkout_lanes']} checkout lanes")


# ================================================================ reports
def aisle_traffic(floor):
    """Per aisle: how many trips walked past it and how many bought something there."""
    total = max(int(floor.traffic.max()), 1)
    t = pd.DataFrame({"aisle": floor.aisle_names, "department": floor.aisle_dept,
                      "trips_passed": floor.traffic, "trips_bought": floor.buyers,
                      "x_m": floor.ax.round(1), "y_m": floor.ay.round(1)})
    t["conversion"] = (t["trips_bought"] / t["trips_passed"].clip(lower=1)).round(3)
    return t


def summarize_walking(trips, floor):
    """What walking adds: time in the store, distance, what people saw, checkout queues."""
    n = len(trips)
    print("\nWalking (v5): shoppers walk the floor plan (SIMULATED; speed, pick time and lane "
          "staffing are ESTIMATED)")
    m = trips["minutes_in_store"]
    print(f"  minutes in the store: average {m.mean():.1f} | median {m.median():.1f} | "
          f"90% of trips under {m.quantile(.9):.0f}")
    print(f"  walked per trip: {trips['walk_m'].mean():.0f} m | shelves visited "
          f"{trips['shelves_visited'].mean():.1f} | aisles seen {trips['aisles_seen'].mean():.0f} "
          f"of {len(floor.aisle_names)}")
    w = trips["wait_min"]
    by_hour = trips.groupby((trips["ready_min"] // 60).astype(int) % 24)["wait_min"].mean()
    print(f"  checkout wait: average {w.mean():.1f} min | 95% of trips under {w.quantile(.95):.1f} min | "
          f"longest hour {by_hour.idxmax():02d}:00 (average {by_hour.max():.1f} min)")
    t = aisle_traffic(floor)
    t["seen_by"] = t["trips_passed"] / n
    busy = t.nlargest(3, "trips_passed")
    print("  most walked-past aisles: " + ", ".join(f"{r.aisle} ({r.seen_by:.0%} of trips)"
                                                    for r in busy.itertuples()))
    quiet = t.nsmallest(3, "trips_passed")
    print("  least seen aisles:       " + ", ".join(f"{r.aisle} ({r.seen_by:.0%})" for r in quiet.itertuples()))
    conv = t[t["trips_passed"] >= 100].nlargest(3, "conversion")
    print("  best 'walk past -> buy': " + ", ".join(f"{r.aisle} ({r.conversion:.0%})" for r in conv.itertuples()))


def save_walking(trips, floor):
    """Trips, aisle traffic and the floor plan itself (the replay draws it)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trips = trips.assign(data_label="SIMULATED")
    trips.round(2).to_parquet(OUTPUT_DIR / "store_trips.parquet", index=False)
    aisle_traffic(floor).assign(data_label="SIMULATED").to_parquet(
        OUTPUT_DIR / "store_aisle_traffic.parquet", index=False)
    plan = floor_plan(floor)
    with open(OUTPUT_DIR / "store_floor.json", "w", encoding="utf-8") as f:
        json.dump(plan, f)
    print(f"  floor plan, trips and aisle traffic saved: store_floor.json, store_trips.parquet, "
          f"store_aisle_traffic.parquet")


def floor_plan(floor):
    """The floor as plain numbers (metres) for drawing: shelves, walkways, door, lanes."""
    return {"width": floor.width, "depth": floor.depth, "y_back": floor.y_back, "y_front": floor.y_front,
            "y_top": floor.y_top, "y_bottom": floor.y_bottom,
            "corridor_x": [round(x, 2) for x in floor.corridor_x],
            "entrance": [floor.entrance[0], floor.entrance[1]], "exit": [floor.exit[0], floor.exit[1]],
            "lanes_x": [round(x, 2) for x in floor.lanes_x], "lane_y": floor.lane_y,
            "aisles": [{"name": a, "department": d, "x": round(float(floor.ax[i]), 1),
                        "y": round(float(floor.ay[i]), 1)}
                       for i, (a, d) in enumerate(zip(floor.aisle_names, floor.aisle_dept))],
            "shelves": [[int(a), round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]
                        for a, x0, y0, x1, y1, c in floor.segments]}
