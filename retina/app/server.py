"""The local RETINA app: a small API over the `retina` command, for your own PC or Docker.

  retina app                 http://localhost:8000  (the website + its Generate page; API docs at /docs)

It is NOT a public server: it binds to 127.0.0.1 by default, takes no API keys (the LLM keys stay in
your environment variables), and every run it starts is the normal `retina` command in a background
process - one run at a time. Endpoints:
  GET  /health                      is it up
  GET  /stores                      the store formats
  POST /custom-stores               create a US store from details and an optional product CSV
  GET  /runs                        runs in data/live (days done, files)
  POST /runs                        start: {"store": "supermarket", "days": 3 or "period": "1w"|"1m"|"3m"|"6m"|"1y",
                                     "mind": "fast"|"student"|"mix", "student": "auto"|"off"|"only"}
  GET  /runs/current                the run in progress (and its last output lines)
  POST /runs/current/watched        {"day": n}: the 3D page has played day n (the next day may be made)
  POST /runs/current/nowait         stop waiting for the 3D page (full speed)
  GET  /runs/{name}/layout          the store's floor plan (live runs)
  GET  /runs/{name}/stream?offset=  live runs: what the NPCs did since offset (bytes), as it happens
  GET  /runs/{name}/report          the thinking-run report (report.txt)
  GET  /runs/{name}/view            days whose NPC walk can be watched (last 3); /view/{day} = that day
  GET  /runs/{name}/files/{file}    download receipts.csv, trips.csv, ... of a run
  GET  /runs/{name}/business/{file} download the business tables (sales_transactions.csv, ...)
  GET  /runs/{name}/validation      integrity and observed-data comparison summary
  POST /runs/{name}/validation      compare aggregate store CSV with a completed run
  GET  /runs/{name}/zip             the business tables + DATA_DICTIONARY + DATASET_CARD in one ZIP,
                                    the simulation logs in extra/
  GET  /datasets                    exported datasets (data/export) and release builds (dist/)
  GET  /                           the website (web/dist, built with `npm run build` in web/), when built
Every file it serves is SIMULATED data made on this PC.
"""

import io
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import zipfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from retina import __version__
from retina.app.custom_stores import StoreSetup, create_store, custom_ids, profile_path

ROOT = Path(__file__).resolve().parents[2]
STORES = {"retina": "retina_supercenter", "supermarket": "generic_supermarket", "supercenter": "supercenter",
          "neighborhood": "neighborhood_market", "convenience": "convenience"}
PERIODS = {"1w": 7, "2w": 14, "1m": 30, "3m": 91, "6m": 182, "1y": 365}
WEB_DIST = ROOT / "web" / "dist"
BUSINESS_FILES = ["sales_transactions.csv", "baskets.csv", "daily_product_sales.csv", "inventory_daily.csv",
                  "daily_store_kpis.csv", "products.csv", "customers.csv", "promotions.csv", "npc_profiles.csv",
                  "DATA_DICTIONARY.md", "DATASET_CARD.md", "VALIDATION.md", "VALIDATION.json"]
RUN_FILES = {"receipts.csv", "trips.csv", "stockouts.csv", "hourly.csv", "decisions.csv", "not_sold_here.csv",
             "checkout_events.csv", "sales.csv", "visits.csv", "npc_actions.csv",
             "report.txt", "report.json"}


def live_root():
    return Path(os.environ["RETINA_LIVE_ROOT"]) if os.environ.get("RETINA_LIVE_ROOT") else ROOT / "data" / "live"


class RunRequest(BaseModel):
    store: str = Field(default="retina", min_length=1, max_length=80)
    days: int = Field(1, ge=1, le=365)
    period: Optional[Literal["1w", "2w", "1m", "3m", "6m", "1y"]] = None     # instead of days
    mind: Literal["fast", "student", "mix"] = "student"
    student: Literal["auto", "off", "only"] = "auto"
    reset: bool = False
    view: bool = True                     # record the NPCs' walk each day for the 3D store page
    wait_for_view: bool = False           # the 3D page asks for it: next day only after it played this one
    start_time: Optional[str] = Field(default=None, pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")
    npc_limit: Optional[int] = Field(default=None, ge=1, le=100)
    scale: Optional[float] = Field(default=None, gt=0, le=1)
    clock_rate: Literal[1, 5, 20] = 5
    seed: Optional[int] = Field(default=None, ge=0, le=2**32 - 1)


def store_profile(name):
    """Only built-in aliases or created local stores may be launched."""
    store_id = STORES.get(name, name)
    path = ROOT / "stores" / f"{store_id}.json" if store_id in STORES.values() else profile_path(store_id)
    if path is None or not path.is_file():
        raise HTTPException(422, f"Unknown store '{name}'. Choose a store from GET /stores.")
    return store_id, json.loads(path.read_text(encoding="utf-8"))


class Runner:
    """One background `retina` process at a time; keeps its last output lines."""

    def __init__(self):
        self.process, self.request, self.lines = None, None, deque(maxlen=40)
        self.lock = threading.Lock()

    def busy(self):
        return self.process is not None and self.process.poll() is None

    def start(self, req: RunRequest):
        with self.lock:
            if self.busy():
                raise HTTPException(409, "A run is already going - wait for it (GET /runs/current).")
            store_id, profile = store_profile(req.store)
            if req.start_time:
                hour, minute = map(int, req.start_time.split(":"))
                opening, closing = profile.get("traffic", {}).get("open_hours", [0, 24])
                if not (opening * 60 <= hour * 60 + minute < closing * 60):
                    raise HTTPException(422, f"Start time must be within store hours: {opening:02d}:00–{closing:02d}:00.")
            realtime = req.mind in ("fast", "student")
            if req.seed is not None and not req.reset:
                raise HTTPException(422, "A seed applies when opening a fresh store (reset=true).")
            seed = req.seed if req.seed is not None else secrets.randbits(32) if req.reset else None
            how_much = ["--period", req.period] if req.period else ["--days", str(req.days)]
            args = [sys.executable, "-u", "-m", "retina", "--store", store_id, *how_much, "--mind", req.mind,
                    "--student", req.student, "--yes"]
            if req.reset:
                args.append("--reset")
            if req.start_time:
                args.extend(("--start-time", req.start_time))
            if req.npc_limit:
                args.extend(("--npc-limit", str(req.npc_limit)))
            if req.scale:
                args.extend(("--scale", str(req.scale)))
            args.extend(("--clock-rate", str(req.clock_rate)))
            if seed is not None:
                args.extend(("--seed", str(seed)))
            if req.view and not (realtime and req.wait_for_view):
                args.append("--view")
            if realtime:
                args.extend(("--realtime", "--live" if req.view and req.wait_for_view else "--payments"))
            elif req.view and req.wait_for_view:
                args.append("--sync-view")
            self.lines.clear()
            if req.reset:                    # a new run: the page must never play days left from an older one
                name = store_id + ("" if req.mind == "fast" else f"_{req.mind}")
                root = live_root().resolve()
                run_folder = (root / name).resolve()
                view_folder = (run_folder / "view").resolve()
                if run_folder.parent != root or view_folder.parent != run_folder:
                    raise HTTPException(422, "Run folder is outside the configured live-data directory.")
                shutil.rmtree(view_folder, ignore_errors=True)
            self.request = req.model_dump()
            self.store_id = store_id
            self.store_title = profile["name"]
            self.realtime = realtime
            self.seed = seed
            self.started = datetime.now(timezone.utc).isoformat(timespec="microseconds")
            self.process = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                            encoding="utf-8", errors="replace",
                                            env=dict(os.environ, PYTHONIOENCODING="utf-8", RETINA_NO_BROWSER="1"))
            threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.process.stdout:
            if line.strip():
                self.lines.append(line.rstrip())

    def state(self):
        if self.process is None:
            return {"running": False, "request": None, "exit_code": None, "output": []}
        req = self.request
        store_id = getattr(self, "store_id", STORES.get(req["store"], req["store"]))
        name = store_id + ("" if req["mind"] == "fast" else f"_{req['mind']}")
        folder = live_root() / name
        return {"running": self.busy(), "request": req, "exit_code": self.process.poll(),
                "started": getattr(self, "started", None),
                "store_title": getattr(self, "store_title", req["store"]),
                "seed": getattr(self, "seed", None),
                "realtime": getattr(self, "realtime", req["mind"] in ("fast", "student")),
                "run": name, "days_wanted": PERIODS.get(req["period"], req["days"]),
                "days_done": days_done(folder) if folder.exists() else 0, "output": list(self.lines)}


app = FastAPI(title="RETINA local app", version=__version__,
              description="Run RETINA's simulated stores on this PC. Every row it makes is SIMULATED.")
runner = Runner()


def days_done(folder):
    hourly = Path(folder) / "hourly.csv"
    if not hourly.exists():
        return 0
    with open(hourly, encoding="utf-8") as f:
        next(f, None)
        return len({line.split(",", 1)[0] for line in f})


def run_dir(name):
    folder = (live_root() / name).resolve()
    if folder.parent != live_root().resolve() or not folder.is_dir():      # no paths outside data/live
        raise HTTPException(404, f"No run '{name}'.")
    return folder


@app.get("/health")
def health():
    from retina.student.listmodel import ListStudent
    from retina.student.model import Student
    from retina.student.tripend import TripEndStudent

    return {"status": "ok", "version": __version__,
            "student_models": {"list": ListStudent.available(), "shelf": Student.available(),
                               "checkout": TripEndStudent.available()}}


@app.get("/stores")
def stores():
    out = []
    for short in list(STORES) + custom_ids():
        store_id, p = store_profile(short)
        out.append({"name": short, "id": store_id, "title": p["name"],
                    "shoppers_per_week": p.get("traffic", {}).get("real_shoppers_per_week"),
                    "open_hours": p.get("traffic", {}).get("open_hours", [0, 24]),
                    "start_date": p.get("simulation", {}).get("start_date"),
                    "custom": short not in STORES, "realtime": p.get("engine") == "realtime"})
    return out


@app.post("/custom-stores", status_code=201)
def add_custom_store(setup: StoreSetup):
    try:
        return create_store(setup)
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.get("/runs")
def runs():
    root = live_root()
    out = []
    for folder in sorted(p for p in root.glob("*") if p.is_dir()) if root.exists() else []:
        files = sorted(f for f in folder.iterdir() if f.name in RUN_FILES)
        biz = [folder / "business" / f for f in BUSINESS_FILES if (folder / "business" / f).exists()]
        out.append({"name": folder.name, "days_done": days_done(folder), "files": [f.name for f in files],
                    "sizes": {f.name: f.stat().st_size for f in files},
                    "business": [f.name for f in biz], "business_sizes": {f.name: f.stat().st_size for f in biz},
                    "updated": datetime.fromtimestamp(max((f.stat().st_mtime for f in files), default=0)).isoformat(
                        timespec="seconds") if files else None})
    return out


@app.delete("/runs/{name}")
def remove_run(name: str):
    """Remove a saved generated run; store profiles and source catalogues are untouched."""
    folder = run_dir(name)
    state = runner.state()
    if state.get("running") and state.get("run") == name:
        raise HTTPException(409, "This store is running. Wait for it to finish before removing its data.")
    shutil.rmtree(folder)
    return {"removed": name}


@app.post("/runs", status_code=202)
def start_run(req: RunRequest):
    runner.start(req)
    return {"started": True, "request": req.model_dump()}


@app.get("/runs/current")
def current():
    return runner.state()


class ClockControl(BaseModel):
    rate: Literal[1, 5, 20]
    paused: bool


@app.post("/runs/current/control")
def control(value: ClockControl):
    if not runner.busy():
        raise HTTPException(409, "No live run is active.")
    folder = current_view()
    path = folder / "control.json"
    tmp = folder / "control.tmp"
    payload = value.model_dump_json()
    tmp.write_text(payload, encoding="utf-8")
    try:
        tmp.replace(path)
    except OSError:
        # The live process ignores an incomplete JSON read and checks again on its next tick.
        path.write_text(payload, encoding="utf-8")
    return value.model_dump()


@app.get("/runs/{name}/clock")
def run_clock(name: str):
    path = run_dir(name) / "view" / "clock.json"
    if not path.exists():
        return {"day": None, "minute": None, "rate": None, "paused": False}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # The live writer can be between truncate and flush; the next poll gets a complete clock.
        return {"day": None, "minute": None, "rate": None, "paused": False}


class Watched(BaseModel):
    day: int = Field(ge=0)


def current_view():
    state = runner.state()
    if not state.get("run"):
        raise HTTPException(404, "No run on this PC yet.")
    folder = live_root() / state["run"] / "view"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


@app.post("/runs/current/watched")
def watched(w: Watched):
    """The 3D page has played day w.day: the simulator may make the next day."""
    (current_view() / "watched.txt").write_text(str(w.day), encoding="utf-8")
    return {"watched": w.day}


@app.post("/runs/current/nowait")
def nowait():
    """Stop waiting for the 3D page: make the remaining days at full speed."""
    (current_view() / "nowait").touch()
    return {"waiting": False}


@app.get("/runs/{name}/report", response_class=PlainTextResponse)
def report(name: str):
    path = run_dir(name) / "report.txt"
    if not path.exists():
        raise HTTPException(404, "No report - only thinking runs (mind mix) have one.")
    return path.read_text(encoding="utf-8")


STREAM_CHUNK = 2_000_000                  # bytes per read of the live stream


@app.get("/runs/{name}/layout")
def layout(name: str):
    path = run_dir(name) / "view" / "layout.json"
    if not path.exists():                    # the store is still opening: ask again in a moment
        return {"layout": None}
    return FileResponse(path, media_type="application/json")


@app.get("/runs/{name}/stream")
def stream(name: str, offset: int = 0):
    """Live runs: the events written since `offset` (bytes) - NPCs entering, at shelves, paying - and the
    offset to ask from next time. Only whole lines are returned."""
    path = run_dir(name) / "view" / "stream.ndjson"
    if not path.exists():
        return {"offset": 0, "events": []}
    with open(path, "rb") as f:
        f.seek(max(0, offset))
        data = f.read(STREAM_CHUNK)
    end = data.rfind(b"\n") + 1
    events = [json.loads(line) for line in data[:end].splitlines() if line.strip()]
    return {"offset": offset + end, "events": events}


@app.get("/runs/{name}/view")
def view_days(name: str):
    """The simulated days whose walk can be watched (the run keeps the last 3)."""
    folder = run_dir(name) / "view"
    return {"days": sorted(int(f.stem[4:]) for f in folder.glob("day_*.json")) if folder.is_dir() else []}


@app.get("/runs/{name}/view/{day}")
def view_day(name: str, day: int):
    path = run_dir(name) / "view" / f"day_{day:03d}.json"
    if not path.exists():
        raise HTTPException(404, f"Day {day} is not (or no longer) kept for watching.")
    return FileResponse(path, media_type="application/json")


@app.get("/runs/{name}/files/{file}")
def download(name: str, file: str):
    if file not in RUN_FILES:
        raise HTTPException(404, f"Only these files can be downloaded: {', '.join(sorted(RUN_FILES))}")
    path = run_dir(name) / file
    if not path.exists():
        raise HTTPException(404, f"{file} not in this run.")
    return FileResponse(path, filename=f"{name}_{file}")


README = """RETINA - generated sales data (SIMULATED: not real sales)

Run: {name}, {days} day(s), made on this PC with the RETINA local app.
Start with DATASET_CARD.md, DATA_DICTIONARY.md, and VALIDATION.md. The business tables:
sales_transactions.csv   one row per product line sold (price, discount, cost, profit)
baskets.csv              one row per receipt
daily_product_sales.csv  product x day: units, sales, profit, promotion, lost sales (forecasting)
inventory_daily.csv      product x day with a stock movement: opening, delivered, wasted, sold, closing
daily_store_kpis.csv     one row per day: customers, sales, margin, basket, stock-outs, waste
products.csv / customers.csv / promotions.csv   product master, customers, promotion calendar

extra/ holds the simulation logs:
sales.csv      clean live CSV: one paid product per row
visits.csv     clean live CSV: one completed NPC visit per row
npc_actions.csv   live aisle moves and shelf decisions, before payment
receipts.csv   one row per product line bought (date, hour, trip, shopper, product, price, deal, mind, why)
checkout_events.csv   live runs: one product line appended when a shopper pays
trips.csv      one row per shopping trip (arrive / leave time, items, spend, mission)
stockouts.csv  empty shelves met by a shopper (what they did instead)
hourly.csv     trips per hour of each day
decisions.csv  thinking runs: every shelf choice with its options and who decided (autopilot / student)
report.txt     thinking runs: what the thinking shoppers did, next to public statistics
Every sale is SIMULATED. DATASET_CARD.md identifies product and price sources for this run.
VALIDATION.md explains integrity checks and whether observed store data was available for comparison.
"""


@app.get("/runs/{name}/business/{file}")
def download_business(name: str, file: str):
    if file not in BUSINESS_FILES:
        raise HTTPException(404, f"Only these files: {', '.join(BUSINESS_FILES)}")
    path = run_dir(name) / "business" / file
    if not path.exists():
        raise HTTPException(404, f"{file} not in this run (business tables are made at the end of a run).")
    return FileResponse(path, filename=f"{name}_{file}")


class ValidationUpload(BaseModel):
    kind: Literal["daily_product_sales", "daily_store_kpis"]
    csv_text: str = Field(min_length=1, max_length=10_000_000)


@app.get("/runs/{name}/validation")
def run_validation(name: str):
    path = run_dir(name) / "business" / "VALIDATION.json"
    if not path.is_file():
        raise HTTPException(404, "This run has no validation report yet.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/runs/{name}/validation")
def compare_with_store_data(name: str, upload: ValidationUpload):
    """Compare permitted aggregate store facts with an already generated synthetic run."""
    folder = run_dir(name)
    state = runner.state()
    if state.get("running") and state.get("run") == name:
        raise HTTPException(409, "Wait until this run finishes before comparing its data.")
    business_dir = folder / "business"
    if not (business_dir / "sales_transactions.csv").is_file():
        raise HTTPException(404, "Generate this store dataset before comparing it with real data.")
    if len(upload.csv_text.encode("utf-8")) > 10_000_000:
        raise HTTPException(422, "The reference CSV must be 10 MB or smaller.")

    from retina.engine.validation import generate_report, validate_reference_file

    reference_dir = folder / "reference"         # real aggregates stay out of the synthetic-data ZIP
    reference_dir.mkdir(exist_ok=True)
    target = reference_dir / f"{upload.kind}.csv"
    descriptor, temporary = tempfile.mkstemp(prefix="reference-", suffix=".csv", dir=reference_dir)
    os.close(descriptor)
    candidate = Path(temporary)
    previous = target.read_bytes() if target.exists() else None
    report_files = [business_dir / "VALIDATION.json", business_dir / "VALIDATION.md"]
    previous_reports = {path: path.read_bytes() if path.exists() else None for path in report_files}
    try:
        candidate.write_text(upload.csv_text, encoding="utf-8")
        validate_reference_file(candidate, upload.kind)
        candidate.replace(target)
        return generate_report(business_dir, reference_dir)
    except Exception as exc:
        if previous is None:
            target.unlink(missing_ok=True)
        else:
            target.write_bytes(previous)
        for path, contents in previous_reports.items():
            if contents is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(contents)
        if isinstance(exc, ValueError):
            raise HTTPException(422, str(exc)) from exc
        raise
    finally:
        candidate.unlink(missing_ok=True)


@app.get("/runs/{name}/zip")
def download_zip(name: str):
    folder = run_dir(name)
    files = sorted(f for f in folder.iterdir() if f.name in RUN_FILES)
    biz = [folder / "business" / f for f in BUSINESS_FILES if (folder / "business" / f).exists()]
    if not files and not biz:
        raise HTTPException(404, "No files in this run yet.")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        z.writestr(f"{name}/README.txt", README.format(name=name, days=days_done(folder)))
        for f in biz:
            z.write(f, f"{name}/{f.name}")
        for f in files:
            z.write(f, f"{name}/extra/{f.name}" if biz else f"{name}/{f.name}")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="application/zip",
                             headers={"Content-Disposition": f'attachment; filename="retina_{name}.zip"'})


@app.get("/datasets")
def datasets():
    export = ROOT / "data" / "export"
    dist = ROOT / "dist"
    return {"exports": sorted(p.name for p in export.iterdir() if p.is_dir()) if export.exists() else [],
            "releases": sorted(json.loads(m.read_text(encoding="utf-8"))["version"]
                               for m in dist.glob("*/manifest.json")) if dist.exists() else []}


# The website (built with `npm run build` in web/) - mounted LAST, so every API route above wins.
if (WEB_DIST / "index.html").exists():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="website")
