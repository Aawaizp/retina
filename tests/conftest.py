"""Shared set-up for RETINA's tests: make the retina package importable and run the store ONCE
(a short run) so every test can look at the same results. Run all tests:  python -m pytest"""

import pickle
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # works with or without `pip install -e .`

DAYS = 10          # short runs keep the tests fast (about half a minute in total)


@pytest.fixture(scope="session")
def pipeline():
    """A 10-day run through the normal pipeline (as retina_mart.py, without saving files)."""
    from retina.engine.retina_mart import run_pipeline
    return run_pipeline(days=DAYS, save=False)


@pytest.fixture(scope="session")
def tables(pipeline):
    """The export tables built from that run."""
    from retina.engine.export import build_tables
    return build_tables(pipeline)


def run_days(days, chunks=None):
    """Sales of a run of `days` days, optionally stopped and restarted (via pickle) after each chunk."""
    from retina.engine.engine import Simulation
    sim = Simulation(days=days, verbose=False)
    for stop in (chunks or []):
        while sim.day < stop:
            sim.run_day()
        sim = pickle.loads(pickle.dumps(sim))          # like live.py saving and loading the store
    while sim.day < days:
        sim.run_day()
    return sim.results().sales
