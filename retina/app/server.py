"""The local RETINA app: a small API over the `retina` command, for your own PC or Docker.

  retina app                 http://localhost:8000  (API docs at /docs)

It is NOT a public server: it binds to 127.0.0.1 by default, takes no API keys (the LLM keys stay in
your environment variables), and every run it starts is the normal `retina` command in a background
process - one run at a time. Endpoints:
  GET  /health                      is it up
  GET  /stores                      the store formats
  GET  /runs                        runs in data/live (days done, files)
  POST /runs                        start: {"store": "supermarket", "days": 3, "mind": "fast"|"mix", "student": "auto"|"off"|"only"}
  GET  /runs/current                the run in progress (and its last output lines)
  GET  /runs/{name}/report          the thinking-run report (report.txt)
  GET  /runs/{name}/files/{file}    download receipts.csv, trips.csv, ... of a run
  GET  /datasets                    exported datasets (data/export) and release builds (dist/)
Every file it serves is SIMULATED data made on this PC.
"""

import json
import os
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from retina import __version__

ROOT = Path(__file__).resolve().parents[2]
STORES = {"supermarket": "generic_supermarket", "supercenter": "supercenter",
          "neighborhood": "neighborhood_market", "convenience": "convenience"}
RUN_FILES = {"receipts.csv", "trips.csv", "stockouts.csv", "hourly.csv", "decisions.csv", "not_sold_here.csv",
             "report.txt", "report.json"}


def live_root():
    return Path(os.environ["RETINA_LIVE_ROOT"]) if os.environ.get("RETINA_LIVE_ROOT") else ROOT / "data" / "live"


class RunRequest(BaseModel):
    store: Literal["supermarket", "supercenter", "neighborhood", "convenience"] = "supermarket"
    days: int = Field(1, ge=1, le=365)
    mind: Literal["fast", "mix"] = "fast"
    student: Literal["auto", "off", "only"] = "auto"
    reset: bool = False


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
            args = [sys.executable, "-m", "retina", "--store", req.store, "--days", str(req.days), "--mind", req.mind,
                    "--student", req.student, "--yes"] + (["--reset"] if req.reset else [])
            self.lines.clear()
            self.request = req.model_dump()
            self.process = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                            encoding="utf-8", errors="replace", env=dict(os.environ, PYTHONIOENCODING="utf-8"))
            threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.process.stdout:
            if line.strip():
                self.lines.append(line.rstrip())

    def state(self):
        if self.process is None:
            return {"running": False, "request": None, "exit_code": None, "output": []}
        return {"running": self.busy(), "request": self.request, "exit_code": self.process.poll(),
                "output": list(self.lines)}


app = FastAPI(title="RETINA local app", version=__version__,
              description="Run RETINA's simulated stores on this PC. Every row it makes is SIMULATED.")
runner = Runner()


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
        files = sorted(f.name for f in folder.iterdir() if f.name in RUN_FILES)
        hourly = folder / "hourly.csv"
        days = len({line.split(",", 1)[0] for line in hourly.read_text(encoding="utf-8").splitlines()[1:]}) \
            if hourly.exists() else 0
        out.append({"name": folder.name, "days_done": days, "files": files})
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


@app.get("/runs/{name}/files/{file}")
def download(name: str, file: str):
    if file not in RUN_FILES:
        raise HTTPException(404, f"Only these files can be downloaded: {', '.join(sorted(RUN_FILES))}")
    path = run_dir(name) / file
    if not path.exists():
        raise HTTPException(404, f"{file} not in this run.")
    return FileResponse(path, filename=f"{name}_{file}")


@app.get("/datasets")
def datasets():
    export = ROOT / "data" / "export"
    dist = ROOT / "dist"
    return {"exports": sorted(p.name for p in export.iterdir() if p.is_dir()) if export.exists() else [],
            "releases": sorted(json.loads(m.read_text(encoding="utf-8"))["version"]
                               for m in dist.glob("*/manifest.json")) if dist.exists() else []}
