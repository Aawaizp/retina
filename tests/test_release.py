"""Release pipeline and local app (v2 Phase 5) - temp folders only, no LLM, no internet."""

import hashlib
import io
import json
import time
import zipfile

import pytest
from fastapi.testclient import TestClient

from retina import release as R


@pytest.fixture(scope="module")
def student_dir(tmp_path_factory):
    """A small student model trained on fixed-rule labels (stands in for the real one)."""
    from retina.student import collect as C
    from retina.student.labels import LabelStore
    from retina.student.train import train
    folder = tmp_path_factory.mktemp("student")
    pool = C.build_pool(folder / "pool.jsonl.gz", days=(3,), panel=50, say=lambda *a: None)
    store = LabelStore(folder / "labels.sqlite")
    C.collect(350, provider="fake", store=store, pool=pool, say=lambda *a: None)
    train(store, "fake", "rules-v1", out=folder / "model", say=lambda *a: None, min_sets=50)
    return folder / "model"


def test_release_builds_zips_manifest_and_notes(tmp_path, monkeypatch, student_dir):
    monkeypatch.setenv("RETINA_STUDENT_DIR", str(student_dir))
    monkeypatch.setattr(R, "STUDENT", student_dir / "student.onnx")
    m = R.release("test", days=2, stores=["convenience", "generic_supermarket"], out=tmp_path, say=lambda *a: None)
    names = [d["name"] for d in m["datasets"]]
    assert names == ["retina_convenience_test", "retina_generic_supermarket_test", "retina_student_supermarket_test"]
    for d in m["datasets"]:
        path = tmp_path / d["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == d["sha256"] and path.stat().st_size == d["bytes"]
        with zipfile.ZipFile(path) as z:
            card = z.read(f"{d['name']}/DATASET_CARD.md").decode("utf-8")
        assert "SIMULATED" in card
        if not m["built_with"]["realism_test"]:
            assert "realism test" in card                                  # honest build note
    engine = m["datasets"][0]
    assert engine["tables"]["transactions"] > 0 and engine["scorecard"]["judged"] > 0
    student = m["datasets"][2]
    assert {"receipts.csv", "decisions.csv", "report.txt"} <= set(student["files"])
    with zipfile.ZipFile(tmp_path / student["file"]) as z:
        decisions = z.read("retina_student_supermarket_test/decisions.csv").decode("utf-8")
    assert "student" in decisions and ",llm," not in decisions             # no LLM was called
    notes = (tmp_path / "RELEASE_NOTES.md").read_text(encoding="utf-8")
    assert "retina_convenience_test.zip" in notes and "SIMULATED" in notes
    assert json.loads((tmp_path / "manifest.json").read_text())["version"] == "test"


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RETINA_LIVE_ROOT", str(tmp_path / "live"))
    monkeypatch.setenv("RETINA_CACHE_DIR", str(tmp_path / "cache"))
    from retina.app import server
    server.runner = server.Runner()
    return TestClient(server.app), tmp_path / "live"


def test_app_reads_runs_and_blocks_other_paths(client):
    c, live = client
    run = live / "convenience"
    run.mkdir(parents=True)
    (run / "hourly.csv").write_bytes(b"date,hour\n2025-09-01,7\n2025-09-02,7\n")      # bytes: same size on every OS
    (run / "receipts.csv").write_bytes(b"a,b\n1,2\n")
    (run / "state.pkl").write_bytes(b"secret")
    assert c.get("/health").json()["status"] == "ok"
    assert {s["name"] for s in c.get("/stores").json()} == {"retina", "supermarket", "supercenter", "neighborhood", "convenience"}
    listed = c.get("/runs").json()
    assert [(r["name"], r["days_done"], r["files"]) for r in listed] == [("convenience", 2, ["hourly.csv", "receipts.csv"])]
    assert listed[0]["sizes"] == {"hourly.csv": 36, "receipts.csv": 8}
    with zipfile.ZipFile(io.BytesIO(c.get("/runs/convenience/zip").content)) as z:      # download all
        assert sorted(z.namelist()) == ["convenience/README.txt", "convenience/hourly.csv", "convenience/receipts.csv"]
        assert "SIMULATED" in z.read("convenience/README.txt").decode("utf-8")
    assert c.get("/runs/..%2F..%2Fpyproject.toml/zip").status_code == 404
    assert c.get("/runs/convenience/files/receipts.csv").text == "a,b\n1,2\n"
    assert c.get("/runs/convenience/files/state.pkl").status_code == 404          # not a data file
    assert c.get("/runs/..%2F..%2Fpyproject.toml/files/receipts.csv").status_code == 404
    assert c.get("/runs/nope/report").status_code == 404
    assert c.post("/runs", json={"store": "walmart"}).status_code == 422
    assert c.post("/runs", json={"days": 0}).status_code == 422


def test_app_starts_one_run_at_a_time(client):
    c, live = client
    assert c.post("/runs", json={"store": "convenience", "days": 1}).status_code == 202
    assert c.post("/runs", json={"store": "convenience", "days": 1}).status_code in (409, 202)
    for _ in range(240):
        state = c.get("/runs/current").json()
        if not state["running"]:
            break
        time.sleep(0.5)
    assert state["exit_code"] == 0, state["output"][-5:]
    assert (live / "convenience" / "receipts.csv").exists()
    assert c.get("/runs").json()[0]["days_done"] >= 1


def test_app_generates_student_data_for_a_period(client, monkeypatch, student_dir):
    """The website's Generate button: --mind student for a chosen period, progress, then the files."""
    c, live = client
    monkeypatch.setenv("RETINA_STUDENT_DIR", str(student_dir))
    assert c.post("/runs", json={"store": "convenience", "period": "1w", "mind": "student", "reset": True}).status_code == 202
    for _ in range(480):
        state = c.get("/runs/current").json()
        if not state["running"]:
            break
        time.sleep(0.5)
    assert state["exit_code"] == 0, state["output"][-5:]
    assert state["run"] == "convenience_student" and state["days_wanted"] == 7 and state["days_done"] == 7
    run = next(r for r in c.get("/runs").json() if r["name"] == "convenience_student")
    assert {"receipts.csv", "trips.csv", "decisions.csv", "report.txt"} <= set(run["files"])
    receipts = c.get("/runs/convenience_student/files/receipts.csv").text
    assert ",student," in receipts and ",llm," not in receipts


def test_app_accepts_every_store_it_lists():
    """Every store GET /stores shows can be started with POST /runs (the request check knows them all)."""
    from retina.app.server import STORES, RunRequest
    for name in STORES:
        assert RunRequest(store=name).store == name
    assert RunRequest().store == "retina"
