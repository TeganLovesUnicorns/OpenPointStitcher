"""The trained judge for "Run from my alignment" from a rough start: one list of clues per candidate, shared by
training and the run, and the model's probability that a candidate is right."""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

FEATURES = ["surface_fit", "fitness", "rmse", "normal_agree", "n_matched", "tilt_deg", "plane_agreement",
            "colour_agreement", "n_planes_considered", "corner_agreement", "n_corners_considered", "edge_agreement",
            "n_edges_considered", "opening_agreement", "n_openings_considered", "landmark_agreement",
            "n_landmarks_considered", "vote_score", "detector_score",
            "move_m", "move_deg", "is_proposer", "proposer_agree", "fit_gain"]
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed_judge.json")
_MODEL: Optional[dict] = None


def clues(values: dict) -> np.ndarray:
    """FEATURES in order from a dict; a missing, empty or non-finite clue reads as 0, as the answer key writes it."""
    out = []
    for f in FEATURES:
        v = values.get(f, 0.0)
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = 0.0
        out.append(v if np.isfinite(v) else 0.0)
    return np.array(out)


def predict(model: dict, x: np.ndarray) -> np.ndarray:
    """Probability each row is right: boosted stumps or a logistic fit, whichever `model` holds."""
    x = np.atleast_2d(x)
    if "stumps" in model:
        f = sum(a * np.where(x[:, j] > t, s, -s) for j, t, s, a in model["stumps"])
        return 1 / (1 + np.exp(-2 * f))
    z = (x - np.asarray(model["mean"])) / np.asarray(model["std"])
    return 1 / (1 + np.exp(-(z @ np.asarray(model["w"]) + model["b"])))


def load_model(path: str = MODEL_PATH) -> dict:
    global _MODEL
    if _MODEL is None:
        with open(path) as fh:
            model = json.load(fh)
        if list(model["features"]) != FEATURES:
            raise ValueError("seed judge was trained on a different list of clues than this code computes")
        _MODEL = model
    return _MODEL
