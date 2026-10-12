"""retina release: build the sample datasets a version tag publishes (GitHub Actions runs the same command).

  retina release --version v2.0.0-alpha.1            4 store formats x 1 year (+ a student dataset)
  retina release --version test --days 3 --stores convenience     a quick try

For each store format (engine mind): `retina export` (Parquet + CSV, data card, dictionary, scorecard,
checks), zipped. When a trained student model exists (models/student/), one more dataset: the
supermarket with a panel of 50 thinking NPCs decided by the student - no LLM call at all.
Everything is built in a temporary folder and only the finished files land in dist/<version>/:
  retina_<store>_<version>.zip ...     the datasets
  manifest.json                        files, sizes, sha256, rows per table, scorecard, how it was built
  RELEASE_NOTES.md                     the text for the GitHub release page
Honest build notes: a build without the private Instacart per-shopper profiles (always the case on
GitHub Actions) or without the realism test says so in the manifest, the notes and each data card.
Every row in every dataset is SIMULATED.
"""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STORES = ["generic_supermarket", "supercenter", "neighborhood_market", "convenience"]
PROFILES = ROOT / "data" / "processed" / "instacart_shopper_profiles.csv"
STUDENT = (Path(os.environ["RETINA_STUDENT_DIR"]) if os.environ.get("RETINA_STUDENT_DIR")
           else ROOT / "models" / "student") / "student.onnx"
LIVE_FILES = ["receipts.csv", "trips.csv", "stockouts.csv", "hourly.csv", "decisions.csv", "report.txt", "report.json"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def run(args, env, say):
    """Run a retina command in a fresh process (the engine reads its store when imported)."""
    out = subprocess.run([sys.executable, "-m", "retina", *args], cwd=ROOT, env=env, capture_output=True,
                         text=True, encoding="utf-8", errors="replace")
    if out.returncode != 0:
        say(out.stdout[-1500:] + out.stderr[-2500:])
        raise SystemExit(f"Release stopped: `retina {' '.join(args)}` failed.")
    return out.stdout


def zip_folder(folder, target):
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(Path(folder).rglob("*")):
            if p.is_file():
                z.write(p, Path(folder.name) / p.relative_to(folder))


def parquet_rows(folder):
    import pyarrow.parquet as pq
    return {str(p.relative_to(folder).with_suffix("")).replace("\\", "/"): pq.ParquetFile(p).metadata.num_rows
            for p in sorted(Path(folder).rglob("*.parquet"))}


def scorecard_of(folder):
    import pandas as pd
    path = Path(folder) / "scorecard.csv"
    if not path.exists():
        return None
    sc = pd.read_csv(path)
    return {"passed": int((sc["result"] == "PASS").sum()), "judged": int((sc["result"] != "N/A").sum()),
            "checks": len(sc)}


def build_note(realism, profiles):
    notes = []
    if not profiles:
        notes.append("built WITHOUT the private Instacart per-shopper profiles: NPC habits are drawn from the "
                     "published distributions instead (the same path a fresh clone of the repository takes)")
    if not realism:
        notes.append("the realism test against held-out real baskets was NOT run (it needs the raw Instacart "
                     "files, which are never shared)")
    return notes


def add_card_note(folder, notes):
    card = Path(folder) / "DATASET_CARD.md"
    if notes and card.exists():
        with open(card, "a", encoding="utf-8") as f:
            f.write("\n## Release build\n" + "".join(f"- {n}\n" for n in notes))


def student_card(folder, version, days, notes):
    (Path(folder) / "DATASET_CARD.md").write_text("\n".join([
        f"# RETINA thinking-NPC sample - {version}", "",
        "**SIMULATED** supermarket data where a panel of 50 NPCs decide at the shelf with the RETINA student "
        "model (a small model trained to copy an LLM teacher's choices); everyone else shops with the "
        "statistical engine. No LLM was called to make this data.", "",
        f"- {days} days, generic supermarket format; files: receipts, trips, stockouts, hourly, decisions "
        "(every thinking-NPC choice with its options and who decided: autopilot / student / engine), report.",
        "- `mind` column on receipts: engine, autopilot, student. `why_type` / `why`: the reason category.",
        "- The student's own quality and limits: see its model card in the repository (models/student/MODEL_CARD.md).",
        *[f"- {n}" for n in notes], ""]), encoding="utf-8")


def release(version, days=None, stores=None, out=None, realism=False, student=True, say=print):
    stores = stores or STORES
    out = Path(out or ROOT / "dist" / version)
    out.mkdir(parents=True, exist_ok=True)
    profiles = PROFILES.exists()
    notes = build_note(realism, profiles)
    datasets = []
    with tempfile.TemporaryDirectory(prefix="retina_release_") as tmp:
        tmp = Path(tmp)
        env = dict(os.environ, RETINA_EXPORT_DIR=str(tmp / "export"), RETINA_LIVE_ROOT=str(tmp / "live"),
                   RETINA_CACHE_DIR=str(tmp / "cache"), PYTHONIOENCODING="utf-8")
        for store in stores:
            name = f"retina_{store}_{version}"
            say(f"Building {name} ({days or 365} days)...")
            args = ["export", "--store", store, "--csv", "--name", name] + (["--days", str(days)] if days else [])
            run(args + ([] if realism else ["--no-realism"]), env, say)
            folder = tmp / "export" / name
            add_card_note(folder, notes)
            target = out / f"{name}.zip"
            zip_folder(folder, target)
            datasets.append({"name": name, "kind": "engine", "store": store, "file": target.name,
                             "bytes": target.stat().st_size, "sha256": sha256(target),
                             "tables": parquet_rows(folder), "scorecard": scorecard_of(folder)})
        if student and STUDENT.exists() and "generic_supermarket" in stores:
            name = f"retina_student_supermarket_{version}"
            say(f"Building {name} (student model, no LLM)...")
            run(["--store", "supermarket", "--mind", "mix", "--student", "only", "--days", str(days or 365),
                 "--reset", "--yes"], env, say)
            run_dir = tmp / "live" / "generic_supermarket_mix"
            folder = tmp / name
            folder.mkdir()
            for f in LIVE_FILES:
                if (run_dir / f).exists():
                    (folder / f).write_bytes((run_dir / f).read_bytes())
            student_card(folder, version, days or 365, notes)
            target = out / f"{name}.zip"
            zip_folder(folder, target)
            datasets.append({"name": name, "kind": "student", "store": "generic_supermarket", "file": target.name,
                             "bytes": target.stat().st_size, "sha256": sha256(target),
                             "files": sorted(p.name for p in folder.iterdir()), "scorecard": None})
        elif student and not STUDENT.exists():
            say("No trained student model yet (models/student/) - the thinking-NPC dataset is left out.")
    manifest = {"version": version, "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "commit": git_commit(), "days": days or 365, "data_label": "SIMULATED",
                "built_with": {"instacart_shopper_profiles": profiles, "realism_test": realism},
                "notes": notes, "datasets": datasets,
                "licences": "Code MIT. Product names/brands/sizes: Open Food Facts; prices: Open Prices (ODbL 1.0). "
                            "Instacart 2017: aggregate statistics only, non-commercial use."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (out / "RELEASE_NOTES.md").write_text(release_notes(manifest), encoding="utf-8")
    say(f"Release {version}: {len(datasets)} dataset(s) -> {out}")
    return manifest


def release_notes(m):
    mb = lambda b: f"{b / 1e6:.1f} MB"
    rows = []
    for d in m["datasets"]:
        lines = d.get("tables", {}).get("transactions")
        sc = d["scorecard"]
        score = f"{sc['passed']}/{sc['judged']}" if sc else "-"
        lines = f"{lines:,}" if lines else "-"
        rows.append(f"| `{d['file']}` | {d['kind']} | {d['store']} | {lines} | {score} | {mb(d['bytes'])} |")
    return "\n".join([
        f"# RETINA {m['version']} - sample datasets", "",
        "Synthetic retail data from RETINA's virtual stores. **Every row is SIMULATED** - no real customer is in it. "
        "Store formats are generic (not any real retailer).", "",
        f"{m['days']} days per store, built from commit `{m['commit']}`. Each zip has Parquet + CSV tables, a "
        "dataset card (read it first), a data dictionary and the scorecard.", "",
        "| file | kind | store | receipt lines | scorecard | size |", "|---|---|---|---|---|---|", *rows, "",
        *([f"**Build notes:** " + "; ".join(m["notes"]) + "."] if m["notes"] else []), "",
        "Checksums (sha256) and row counts: `manifest.json`. Licences: " + m["licences"], ""])
