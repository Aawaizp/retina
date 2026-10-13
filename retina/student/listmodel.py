"""The list student at run time (ONNX Runtime): an NPC's scenario -> today's shopping list.

Each candidate is DRAWN with probability sigmoid(logit(p) + bias_shift) (seeded dice: the same day and
NPC give the same list); its quantity comes from the quantity model (rounded, 1-6)."""

import json
import os
from pathlib import Path

import numpy as np

from retina.student.listtrain import FEATURES, aisle_rebuy, scenario_matrix

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.environ["RETINA_LIST_DIR"]) if os.environ.get("RETINA_LIST_DIR") else ROOT / "models" / "list_student"


class ListStudent:
    def __init__(self, folder=MODEL_DIR):
        import onnxruntime as rt
        folder = Path(folder)
        self.meta = json.loads((folder / "list_student.json").read_text(encoding="utf-8"))
        if self.meta["features"] != FEATURES:
            raise ValueError("list_student.json features do not match this RETINA version - retrain: retina lists --train")
        self.buy = rt.InferenceSession(str(folder / "list_student.onnx"))
        self.qty = rt.InferenceSession(str(folder / "list_qty.onnx"))
        self.shift = float(self.meta["bias_shift"])
        self.rebuy = aisle_rebuy()

    @staticmethod
    def available(folder=MODEL_DIR):
        folder = Path(folder)
        return all((folder / f).exists() for f in ("list_student.onnx", "list_qty.onnx", "list_student.json"))

    def probabilities(self, X):
        outs = self.buy.run(None, {self.buy.get_inputs()[0].name: X})
        prob = np.asarray(outs[1])                                 # [n, 2] class probabilities (no zipmap)
        return prob[:, 1] if prob.ndim == 2 else prob

    def plan(self, scenario, rng):
        """[(candidate, qty)] for today's list."""
        cands = scenario["candidates"]
        if not cands:
            return []
        X = scenario_matrix(scenario, self.rebuy)
        p = np.clip(self.probabilities(X), 1e-6, 1 - 1e-6)
        p = 1 / (1 + np.exp(-(np.log(p / (1 - p)) + self.shift)))
        pick = rng.random(len(p)) < p
        if not pick.any():
            return []
        q = np.asarray(self.qty.run(None, {self.qty.get_inputs()[0].name: X[pick]})[0]).ravel()
        q = np.clip(np.round(q), 1, 6).astype(int)
        return [(c, int(n)) for c, n in zip([c for c, b in zip(cands, pick) if b], q)]
