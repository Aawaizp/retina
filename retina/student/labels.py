"""Teacher labels: every LLM shelf decision the student learns from, in data/teacher/labels.sqlite.

One row per (situation item, repeat): repeat 0 = the label, repeat 1 = the same situation asked
again (teacher self-agreement). Every row records the provider and model that answered - training
uses only one teacher (Groq openai/gpt-oss-120b, decided 12 Oct 2026), so labels never mix minds.
source "collect" = made by retina collect; "run" = a teacher answer from a mix run (active learning).
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEACHER_DIR = ROOT / "data" / "teacher"
TEACHER = ("groq", "openai/gpt-oss-120b")          # the ONE teacher whose labels train the student


SCHEMA = """CREATE TABLE IF NOT EXISTS labels (
    id TEXT, repeat INTEGER, teacher TEXT, source TEXT, provider TEXT, model TEXT, prompt_version INTEGER,
    npc INTEGER, day INTEGER, valid INTEGER, created TEXT, data TEXT, PRIMARY KEY (id, repeat, teacher))"""


def teacher_of(provider, model):
    return f"{provider}/{model}"


class LabelStore:
    """Labels keyed by (situation, repeat, teacher): two teachers can label the same situation."""

    def __init__(self, path=None):
        self.path = Path(path or TEACHER_DIR / "labels.sqlite")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path)
        columns = [r[1] for r in self.db.execute("PRAGMA table_info(labels)")]
        if columns and "teacher" not in columns:            # file from before the teacher column: move over
            self.db.execute("ALTER TABLE labels RENAME TO labels_old")
            self.db.execute(SCHEMA)
            self.db.execute("INSERT INTO labels SELECT id, repeat, provider || '/' || model, source, provider, model, "
                            "prompt_version, npc, day, valid, created, data FROM labels_old")
            self.db.execute("DROP TABLE labels_old")
            self.db.commit()
        self.db.execute(SCHEMA)

    def add(self, rows):
        """rows = dicts with id, repeat, source, provider, model, prompt_version, npc, day, valid + the rest."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        for r in rows:
            r["teacher"] = teacher_of(r["provider"], r["model"])
        self.db.executemany("INSERT OR REPLACE INTO labels VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            [(r["id"], r["repeat"], r["teacher"], r["source"], r["provider"], r["model"],
                              r["prompt_version"], r["npc"], r["day"], int(r["valid"]), now, json.dumps(r))
                             for r in rows])
        self.db.commit()

    def ids(self, repeat=0, teacher=None):
        if teacher is None:
            return {r[0] for r in self.db.execute("SELECT id FROM labels WHERE repeat = ?", (repeat,))}
        return {r[0] for r in self.db.execute("SELECT id FROM labels WHERE repeat = ? AND teacher = ?",
                                              (repeat, teacher))}

    def rows(self, repeat=0, provider=None, model=None, valid_only=True):
        sql, args = "SELECT data FROM labels WHERE repeat = ?", [repeat]
        if provider:
            sql, args = sql + " AND provider = ?", args + [provider]
        if model:
            sql, args = sql + " AND model = ?", args + [model]
        if valid_only:
            sql += " AND valid = 1"
        return [json.loads(r[0]) for r in self.db.execute(sql + " ORDER BY id", args)]

    def counts(self):
        return self.db.execute("SELECT repeat, provider, model, source, valid, COUNT(*) FROM labels "
                               "GROUP BY repeat, teacher, source, valid").fetchall()

    def close(self):
        self.db.close()
