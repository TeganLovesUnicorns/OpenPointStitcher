"""Colour as a tolerant second opinion: hue families and local colour composition."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def hue_family(hue_deg: float, saturation: float, cfg) -> str:
    """Nearest named hue family, or "neutral" below `cfg.colour_saturation_floor` (C2).
    Nearest-centre over `cfg.colour_family_edges`, so no special case at the 0/360 wrap."""
    if np.isnan(hue_deg) or saturation < cfg.colour_saturation_floor:
        return "neutral"
    h = hue_deg % 360.0
    best_name, best_d = "", 361.0
    for name, centre in cfg.colour_family_edges:
        d = abs(h - centre) % 360.0
        d = min(d, 360.0 - d)
        if d < best_d:
            best_name, best_d = name, d
    return best_name


def composition(
    families: Iterable[str], weights: Iterable[float],
) -> dict[str, float]:
    """Colour signature (C4): weighted proportion of each family; {} for no input."""
    totals: dict[str, float] = {}
    total_w = 0.0
    for fam, w in zip(families, weights):
        w = float(w)
        if w <= 0:
            continue
        totals[fam] = totals.get(fam, 0.0) + w
        total_w += w
    if total_w <= 0:
        return {}
    return {fam: w / total_w for fam, w in totals.items()}


def composition_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """0 to 1 histogram intersection of two compositions. Two empty ones agree (1.0)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    return float(sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in keys))


def hue_agreement(hue_a: float, sat_a: float, hue_b: float, sat_b: float, cfg) -> float:
    """0-1 agreement of two hues: neutral+neutral 1, neutral+coloured 0, else a circular
    tolerance (`colour_hue_tolerance_deg`), never a family-boundary cliff."""
    fam_a = hue_family(hue_a, sat_a, cfg)
    fam_b = hue_family(hue_b, sat_b, cfg)
    if fam_a == "neutral" and fam_b == "neutral":
        return 1.0
    if fam_a == "neutral" or fam_b == "neutral":
        return 0.0
    d = abs(hue_a - hue_b) % 360.0
    d = min(d, 360.0 - d)
    return float(max(0.0, 1.0 - d / cfg.colour_hue_tolerance_deg))
