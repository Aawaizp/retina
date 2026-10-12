"""Answer cache for the LLM teacher: one answer per NPC + day + situation.

The key is a fingerprint of the store, seed, prompt version, NPC id, day and the exact situation
the NPC faced (their card, memory and the options shown). Same situation -> same answer, so:
  - rerunning a mix run gives identical data and costs no LLM calls,
  - 10 days in one go == 5 + 5 days (stop and continue),
  - which provider answered does not matter for the key (the answer records who it was).
Lives in data/cache/mind_cache.sqlite - outside the run folder, so it survives `--reset`.
"""

import hashlib
import json
import sqlite3
from pathlib import Path


def fingerprint(*parts):
    raw = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class AnswerCache:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        self.db.execute("CREATE TABLE IF NOT EXISTS answers (key TEXT PRIMARY KEY, value TEXT)")
        self.hits = self.misses = 0

    def get(self, key):
        row = self.db.execute("SELECT value FROM answers WHERE key = ?", (key,)).fetchone()
        if row:
            self.hits += 1
            return json.loads(row[0])
        self.misses += 1
        return None

    def put(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO answers VALUES (?, ?)", (key, json.dumps(value)))
        self.db.commit()                       # an answer paid for is never lost

    def close(self):
        self.db.close()
