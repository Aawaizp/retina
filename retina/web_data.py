"""retina web-data: the recorded store days the public website plays (web/public/data).

  retina web-data                  RETINA Supercenter, the student runs every NPC, 3 days -> the last 2 kept
  retina web-data --no-student     engine only (a quick try)

The website is ONE page: the 3D RETINA Supercenter with the NPCs walking and the sales data they make.
On the owner's PC (retina app) it watches the run being generated. On the public website there is no
simulator, so it plays days recorded here: every NPC route, choice, price and receipt comes from the
simulation (SIMULATED). The run uses RETINA_NO_PROFILES=1, exactly like a fresh clone of the repository,
so the shoppers never come from the private per-shopper file.
"""

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from retina.release import ROOT, STUDENT, git_commit, run

WEB = ROOT / "web" / "public"
STORE = "retina_supercenter"
LIST_STUDENT = ROOT / "models" / "list_student" / "list_student.onnx"


def web_data(days=3, keep=2, out=WEB, student=True, say=print):
    out = Path(out)
    data = out / "data"
    student = student and STUDENT.exists() and LIST_STUDENT.exists()
    mind = "student" if student else "fast"
    with tempfile.TemporaryDirectory(prefix="retina_web_") as tmp:
        tmp = Path(tmp)
        env = dict(os.environ, RETINA_EXPORT_DIR=str(tmp / "export"), RETINA_LIVE_ROOT=str(tmp / "live"),
                   RETINA_CACHE_DIR=str(tmp / "cache"), RETINA_NO_PROFILES="1", RETINA_NO_BROWSER="1",
                   PYTHONIOENCODING="utf-8")
        say(f"Simulating RETINA Supercenter ({days} days, mind: {mind}, NPC walks recorded)...")
        run(["--store", STORE, "--mind", mind, "--days", str(days), "--reset", "--yes", "--view"], env, say)
        recorded = sorted((tmp / "live" / (STORE if mind == "fast" else f"{STORE}_{mind}") / "view").glob("day_*.json"))
        if data.exists():                                      # the older dashboard files go
            shutil.rmtree(data)
        if (out / "model").exists():
            shutil.rmtree(out / "model")
        data.mkdir(parents=True)
        days_kept = []
        for f in recorded[-keep:]:
            shutil.copy2(f, data / f.name)
            days_kept.append(f.name)
    manifest = {"built": datetime.now(timezone.utc).isoformat(timespec="seconds"), "commit": git_commit(),
                "store": "RETINA Supercenter", "mind": mind, "days": days_kept, "data_label": "SIMULATED",
                "note": "recorded store days (NPC routes, choices, receipts); built without the private "
                        "Instacart per-shopper profiles (like a fresh clone)"}
    (data / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    size = sum(f.stat().st_size for f in data.iterdir())
    say(f"Website data ready: {data} ({len(days_kept)} days, {size / 1e6:.1f} MB)")
    return manifest
