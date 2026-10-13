"""The trained student at run time: models/student/student.onnx + student.json, run with ONNX Runtime
(the same file the website will use in the browser). probabilities = softmax(scores / temperature)."""

import json
import os
from pathlib import Path

import numpy as np

from retina.student.features import FEATURES, matrix

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = Path(os.environ["RETINA_STUDENT_DIR"]) if os.environ.get("RETINA_STUDENT_DIR") else ROOT / "models" / "student"


class Student:
    def __init__(self, folder=MODEL_DIR):
        import onnxruntime as rt
        folder = Path(folder)
        self.meta = json.loads((folder / "student.json").read_text(encoding="utf-8"))
        if self.meta["features"] != FEATURES:
            raise ValueError("student.json features do not match this RETINA version - retrain: retina train")
        self.session = rt.InferenceSession(str(folder / "student.onnx"))
        self.input = self.session.get_inputs()[0].name
        self.temperature = float(self.meta["temperature"])
        self.threshold = float(self.meta["threshold"])
        self.pick = self.meta.get("pick", "top")

    @staticmethod
    def available(folder=MODEL_DIR):
        return (Path(folder) / "student.onnx").exists() and (Path(folder) / "student.json").exists()

    def choose(self, npc, route, mission, options, engine_product, rng=None):
        """(chosen option, confidence, all probabilities) or (None, 0, []) when nothing is choosable.
        Models trained with pick "draw" DRAW the option from the probabilities (with rng), like a real
        shopper who does not always buy the same thing; otherwise the most likely option. Confidence =
        the top probability either way (it decides student vs teacher in mix runs)."""
        X, opts = matrix(npc, route, mission, options, engine_product)
        if not opts:
            return None, 0.0, []
        scores = np.asarray(self.session.run(None, {self.input: X})[0], dtype=float).ravel() / self.temperature
        p = np.exp(scores - scores.max())
        p /= p.sum()
        i = int(np.argmax(p))
        if self.pick == "draw" and rng is not None:
            i = int(rng.choice(len(p), p=p))
        return opts[i], float(p.max()), p


def why_type(option, engine_product):
    """A reason category for a student pick (the student ranks; it does not write sentences)."""
    if option["times"] > 0:
        return "habit"
    if option["deal"]:
        return "deal"
    return "try_new"
