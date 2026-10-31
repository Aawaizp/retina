"""Observed super-store trip-size prior for NPC requirements, separate from SKU choice."""

import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILE = ROOT / "models" / "trip_size" / "foodaps_superstore.json"


class TripSizePrior:
    def __init__(self, path=None):
        path = Path(path or os.environ.get("RETINA_TRIP_SIZE_FILE", DEFAULT_FILE))
        self.meta = json.loads(path.read_text(encoding="utf-8"))
        if self.meta.get("version") != 1 or not self.meta.get("qualified"):
            raise ValueError("trip-size prior is not qualified")
        self.counts = np.asarray(self.meta["counts"], dtype=int)
        self.probabilities = np.asarray(self.meta["probabilities"], dtype=float)
        if len(self.counts) != len(self.probabilities) or not np.isclose(self.probabilities.sum(), 1):
            raise ValueError("invalid trip-size distribution")

    def sample(self, rng, maximum):
        return min(int(rng.choice(self.counts, p=self.probabilities)), int(maximum))
