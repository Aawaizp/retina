"""NPC memory: one SQLite file per run (data/live/<store>_<mind>/memory.sqlite).

A card per NPC id (persona + facts, written once when the NPC joins the panel) and a LOG of what
happened to them: trips, purchases, empty shelves, and every decision the mind made. What an NPC
"remembers" (last price paid, how often, sold out last time, recent trips) is read from that log,
so a returning NPC is the same person with the same history.

Only the log is stored - no running totals - so a run that stopped half-way through a day can be
rolled back exactly to the last saved day (rollback_to). The engine's own state (pantry, habits,
price memory) stays in state.pkl; this file holds what the MIND needs. Every row is SIMULATED.
"""

import json
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS cards (npc INTEGER PRIMARY KEY, name TEXT, card TEXT, since_day INTEGER);
CREATE TABLE IF NOT EXISTS trips (npc INTEGER, day INTEGER, hour INTEGER, mission TEXT, items INTEGER,
                                  spend REAL, budget REAL, top TEXT);
CREATE TABLE IF NOT EXISTS purchases (npc INTEGER, day INTEGER, product INTEGER, name TEXT, aisle TEXT,
                                      qty INTEGER, price REAL, regular REAL, promo INTEGER, mind TEXT,
                                      why_type TEXT, why TEXT);
CREATE TABLE IF NOT EXISTS stockouts (npc INTEGER, day INTEGER, product INTEGER, name TEXT, reaction TEXT);
CREATE TABLE IF NOT EXISTS decisions (day INTEGER, hour INTEGER, npc INTEGER, item INTEGER, aisle TEXT,
                                      route TEXT, mind TEXT, trigger TEXT, engine_product INTEGER,
                                      product INTEGER, why_type TEXT, why TEXT, options TEXT, provider TEXT,
                                      model TEXT, cached INTEGER, valid INTEGER, problem TEXT);
CREATE INDEX IF NOT EXISTS purchases_npc ON purchases (npc, product);
CREATE INDEX IF NOT EXISTS trips_npc ON trips (npc, day);
CREATE INDEX IF NOT EXISTS stockouts_npc ON stockouts (npc, product);
"""
DAY_TABLES = ("trips", "purchases", "stockouts", "decisions")
DECISION_COLUMNS = ["day", "hour", "npc", "item", "aisle", "route", "mind", "trigger", "engine_product",
                    "product", "why_type", "why", "options", "provider", "model", "cached", "valid", "problem"]


class MemoryStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.executescript(SCHEMA)

    # ------------------------------------------------------------- run info
    def get(self, key, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, json.dumps(value)))

    # ---------------------------------------------------------------- cards
    def add_card(self, npc, card, day):
        self.db.execute("INSERT OR REPLACE INTO cards VALUES (?, ?, ?, ?)",
                        (int(npc), card["facts"]["name"], json.dumps(card), int(day)))

    def card(self, npc):
        row = self.db.execute("SELECT card FROM cards WHERE npc = ?", (int(npc),)).fetchone()
        return json.loads(row[0]) if row else None

    def panel(self):
        return [r[0] for r in self.db.execute("SELECT npc FROM cards ORDER BY npc")]

    # ------------------------------------------------------------------ log
    def add_trip(self, npc, day, hour, mission, items, spend, budget, top):
        self.db.execute("INSERT INTO trips VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (int(npc), int(day), int(hour), mission, int(items), float(spend),
                         float(min(budget, 1e6)), top))

    def add_purchases(self, rows):
        self.db.executemany("INSERT INTO purchases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)

    def add_stockouts(self, rows):
        self.db.executemany("INSERT INTO stockouts VALUES (?, ?, ?, ?, ?)", rows)

    def add_decisions(self, rows):
        """rows = dicts with DECISION_COLUMNS (options as a list -> stored as JSON)."""
        self.db.executemany(
            f"INSERT INTO decisions VALUES ({', '.join('?' * len(DECISION_COLUMNS))})",
            [tuple(json.dumps(r[c]) if c == "options" else r.get(c) for c in DECISION_COLUMNS) for r in rows])

    # --------------------------------------------------------------- recall
    def last_trips(self, npc, before_day, n=2):
        return self.db.execute("SELECT day, items, spend, top FROM trips WHERE npc = ? AND day < ? "
                               "ORDER BY day DESC LIMIT ?", (int(npc), int(before_day), n)).fetchall()

    def recall(self, npc, products, before_day):
        """What this NPC remembers about these products: {product: {times, last_price, last_day,
        last_why, sold_out}} - only products with a memory are returned."""
        products = [int(p) for p in products]
        if not products:
            return {}
        marks = ", ".join("?" * len(products))
        out = {}
        for p, times, price, day, why in self.db.execute(
                f"SELECT product, COUNT(*), price, MAX(day), why FROM purchases WHERE npc = ? AND day < ? "
                f"AND product IN ({marks}) GROUP BY product", (int(npc), int(before_day), *products)):
            out[p] = {"times": times, "last_price": price, "last_day": day, "last_why": why, "sold_out": 0}
        for p, n in self.db.execute(
                f"SELECT product, COUNT(*) FROM stockouts WHERE npc = ? AND day < ? AND product IN ({marks}) "
                f"GROUP BY product", (int(npc), int(before_day), *products)):
            out.setdefault(p, {"times": 0, "last_price": None, "last_day": None, "last_why": None})["sold_out"] = n
        return out

    def decisions(self, since_day=0):
        cur = self.db.execute("SELECT * FROM decisions WHERE day >= ? ORDER BY day, hour, npc, item", (int(since_day),))
        return [dict(zip(DECISION_COLUMNS, r)) for r in cur]

    def count(self, table):
        return self.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    # ----------------------------------------------------------- save / undo
    def commit(self, day_done):
        """Make everything up to and including `day_done` permanent."""
        self.set("days_done", int(day_done) + 1)
        self.db.commit()

    def rollback_to(self, days_done):
        """Forget anything logged on day >= days_done (a run stopped mid-day, or state.pkl is older)."""
        for table in DAY_TABLES:
            self.db.execute(f"DELETE FROM {table} WHERE day >= ?", (int(days_done),))
        self.set("days_done", int(days_done))
        self.db.commit()

    def close(self):
        self.db.close()
