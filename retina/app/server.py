"""The local RETINA app: a small API over the `retina` command, for your own PC or Docker.

  retina app                 http://localhost:8000  (the website + its Generate page; API docs at /docs)

It is NOT a public server: it binds to 127.0.0.1 by default, takes no API keys (the LLM keys stay in
your environment variables), and every run it starts is the normal `retina` command in a background
process - one run at a time. Endpoints:
  GET  /health                      is it up
  GET  /stores                      the store formats
  GET  /runs                        runs in data/live (days done, files)
  POST /runs                        start: {"store": "supermarket", "days": 3 or "period": "1w"|"1m"|"3m"|"6m"|"1y",
                                     "mind": "fast"|"student"|"mix", "student": "auto"|"off"|"only"}
  GET  /runs/current                the run in progress (and its last output lines)
  GET  /runs/{name}/report          the thinking-run report (report.txt)
  GET  /runs/{name}/view            days whose NPC walk can be watched (last 3); /view/{day} = that day
  GET  /runs/{name}/files/{file}    download receipts.csv, trips.csv, ... of a run
  GET  /runs/{name}/business/{file} download the business tables (sales_transactions.csv, ...)
  GET  /runs/{name}/zip             the business tables + DATA_DICTIONARY + DATASET_CARD in one ZIP,
                                    the simulation logs in extra/
  GET  /datasets                    exported datasets (data/export) and release builds (dist/)
  GET  /                           the website (web/dist, built with `npm run build` in web/), when built
Every file it serves is SIMULATED data made on this PC.
"""

import io
import json
import os
import subprocess
import sys
import threading
import zipfile
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from retina import __version__

ROOT = Path(__file__).resolve().parents[2]
STORES = {"retina": "retina_supercenter", "supermarket": "generic_supermarket", "supercenter": "supercenter",
          "neighborhood": "neighborhood_market", "convenience": "convenience"}
PERIODS = {"1w": 7, "2w": 14, "1m": 30, "3m": 91, "6m": 182, "1y": 365}
WEB_DIST = ROOT / "web" / "dist"
BUSINESS_FILES = ["sales_transactions.csv", "baskets.csv", "daily_product_sales.csv", "inventory_daily.csv",
                  "daily_store_kpis.csv", "products.csv", "customers.csv", "promotions.csv",
                  "DATA_DICTIONARY.md", "DATASET_CARD.md"]
RUN_FILES = {"receipts.csv", "trips.csv", "stockouts.csv", "hourly.csv", "decisions.csv", "not_sold_here.csv",
             "report.txt", "report.json"}


def live_root():
    return Path(os.environ["RETINA_LIVE_ROOT"]) if os.environ.get("RETINA_LIVE_ROOT") else ROOT / "data" / "live"


class RunRequest(BaseModel):
    store: Literal["retina", "supermarket", "supercenter", "neighborhood", "convenience"] = "retina"
    days: int = Field(1, ge=1, le=365)
    period: Optional[Literal["1w", "2w", "1m", "3m", "6m", "1y"]] = None     # instead of days
    mind: Literal["fast", "student", "mix"] = "fast"
    student: Literal["auto", "off", "only"] = "auto"
    reset: bool = False
    view: bool = True                     # record the NPCs' walk each day for the 3D store page


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
            how_much = ["--period", req.period] if req.period else ["--days", str(req.days)]
            args = [sys.executable, "-u", "-m", "retina", "--store", req.store, *how_much, "--mind", req.mind,
                    "--student", req.student, "--yes"] + (["--reset"] if req.reset else []) +                 (["--view"] if req.view else [])
            self.lines.clear()
            self.request = req.model_dump()
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
        name = STORES[req["store"]] + ("" if req["mind"] == "fast" else f"_{req['mind']}")
        folder = live_root() / name
        return {"running": self.busy(), "request": req, "exit_code": self.process.poll(),
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
    return {"status": "ok", "version": __version__}


@app.get("/stores")
def stores():
    out = []
    for short, store_id in STORES.items():
        p = json.loads((ROOT / "stores" / f"{store_id}.json").read_text(encoding="utf-8"))
        out.append({"name": short, "id": store_id, "title": p["name"],
                    "shoppers_per_week": p.get("traffic", {}).get("real_shoppers_per_week")})
    return out


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


@app.post("/runs", status_code=202)
def start_run(req: RunRequest):
    runner.start(req)
    return {"started": True, "request": req.model_dump()}


@app.get("/runs/current")
def current():
    return runner.state()


@app.get("/runs/{name}/report", response_class=PlainTextResponse)
def report(name: str):
    path = run_dir(name) / "report.txt"
    if not path.exists():
        raise HTTPException(404, "No report - only thinking runs (mind mix) have one.")
    return path.read_text(encoding="utf-8")


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
Start with DATASET_CARD.md and DATA_DICTIONARY.md. The business tables:
sales_transactions.csv   one row per product line sold (price, discount, cost, profit)
baskets.csv              one row per receipt
daily_product_sales.csv  product x day: units, sales, profit, promotion, lost sales (forecasting)
inventory_daily.csv      product x day with a stock movement: opening, delivered, wasted, sold, closing
daily_store_kpis.csv     one row per day: customers, sales, margin, basket, stock-outs, waste
products.csv / customers.csv / promotions.csv   product master, customers, promotion calendar

extra/ holds the simulation logs:
receipts.csv   one row per product line bought (date, hour, trip, shopper, product, price, deal, mind, why)
trips.csv      one row per shopping trip (arrive / leave time, items, spend, mission)
stockouts.csv  empty shelves met by a shopper (what they did instead)
hourly.csv     trips per hour of each day
decisions.csv  thinking runs: every shelf choice with its options and who decided (autopilot / student)
report.txt     thinking runs: what the thinking shoppers did, next to public statistics
Every value is SIMULATED. Product names / brands / sizes: Open Food Facts; prices: Open Prices (ODbL 1.0).
"""


@app.get("/runs/{name}/business/{file}")
def download_business(name: str, file: str):
    if file not in BUSINESS_FILES:
        raise HTTPException(404, f"Only these files: {', '.join(BUSINESS_FILES)}")
    path = run_dir(name) / "business" / file
    if not path.exists():
        raise HTTPException(404, f"{file} not in this run (business tables are made at the end of a run).")
    return FileResponse(path, filename=f"{name}_{file}")


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
