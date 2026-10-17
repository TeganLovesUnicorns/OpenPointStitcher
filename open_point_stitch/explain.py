"""Plain words for what a run did: what happened to each pair of scans (the links table; the technical reason stays
alongside for reports), and why each scan moved, which links pulled it and what the move did to its other links."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .lib.poses import yaw_gap_deg, yaw_of

def pct(frac: float) -> str:
    """0.027 -> '2.7%', 0.31 -> '31%'."""
    v = frac * 100
    return f"{v:.1f}%" if v < 10 else f"{v:.0f}%"


def cm(metres: float) -> str:
    v = metres * 100
    return f"{v:.1f} cm" if v < 10 else f"{v:.0f} cm"


def too_few_on_surface(src: str, tgt: str, fitness: float, needed: float) -> str:
    return f"only {pct(fitness)} of {src}'s points landed on a surface of {tgt} ({pct(needed)} needed)"


def too_few_of_possible(src: str, tgt: str, ratio: float, needed: float) -> str:
    return (f"of the points of {src} that could have landed on {tgt} there, only {pct(ratio)} did "
            f"({pct(needed)} needed)")


def too_loose(rmse: float, allowed: float) -> str:
    return f"the matched points sat {cm(rmse)} apart on average (at most {cm(allowed)} allowed)"


def too_few_matched(n: int, needed: int) -> str:
    return f"only {n:,} points matched up ({needed:,} needed)"


def too_little_overlap(frac: float, needed: float, within_m: float) -> str:
    return f"only {pct(frac)} of the two scans' points lie within {cm(within_m)} of each other ({pct(needed)} needed)"


def matched(src: str, tgt: str, fitness: float, rmse: float) -> str:
    return (f"Matched: {pct(fitness)} of {src}'s points landed on a surface of {tgt}, "
            f"{cm(rmse)} apart on average.")


def kept_exactly(check: str) -> str:
    return f"Kept exactly where you put it. The search's best other position was not trusted: {check}."


def kept_and_fine_tuned(moved_m: float, turned_deg: float, check: str) -> str:
    return (f"Kept near where you put it, fine-tuned by {cm(moved_m)} and {turned_deg:.1f} degrees. The search's "
            f"best other position was not trusted: {check}." +
            (" A fine-tune this big is worth a look." if moved_m >= 0.10 or turned_deg >= 1.0 else ""))


def rejected(check: str) -> str:
    return f"Not linked: {check}."


def kept_no_better(best: float, own: float, needed_frac: float) -> str:
    return (f"Kept exactly where you put it: nothing the search found fitted the walls, corners and edges better "
            f"than your placement by the {needed_frac:.0%} it takes to move (best {best:.3f}, yours {own:.3f}).")


KEPT_NO_EVIDENCE = ("Kept exactly where you put it: no wall, corner or edge could be matched between these two "
                    "scans, so there is nothing to move it on.")
KEPT_NOTHING_ELIGIBLE = "Kept exactly where you put it: the search found no other position worth trying."
NO_START = "Not linked: no starting position could be found for this pair."
EXCLUDED_PAIR = "Not tried: this pair is switched off."


def all_tilted(n: int, max_deg: float) -> str:
    return (f"Not linked: all {n} positions the search found would tip one scan more than {max_deg:g} degrees off "
            f"level, which a levelled scan cannot be. Place this pair by hand.")


def all_disagreed(n: int) -> str:
    return (f"Not linked: in all {n} positions the search found, a matched wall or corner landed in the wrong "
            f"place relative to the others.")


def tips_off_level(deg: float) -> str:
    return f"Not linked: the best position would tip the scan {deg:.1f} degrees off level."


NOTHING_SURVIVED = "Not linked: no position the search found passed the basic checks."


# -------------------------------------------------------------------------------------------------------------------
# Why each scan moved in a run, which links pulled it, and what the move did to its other links. Fits are measured
# on the thinned copies registration already uses, at the poses before and after the run.
# -------------------------------------------------------------------------------------------------------------------

NEAR_M = 0.05        # a point fits when another scan's point is this close
GOOD_FIT = 0.20      # a link is "good" when this share of the smaller scan fits
HURT_DROP = 0.05     # a good link losing more than this share of fit counts as hurt
HELPED_GAIN = 0.10   # a link gaining this much share counts as helped (and is named in the account)
MAX_HELPED_NAMED = 4
MOVED_M = 0.01       # a scan moved when its points shift this far on average ...
MOVED_DEG = 0.2      # ... or it turned this much
KEPT_METHODS = ("kept_placement",)


@dataclass
class Link:
    a: str
    b: str
    method: str              # the pair's outcome: point_to_plane / kept_placement / refined_seed / ...
    accepted: bool
    reason: str
    fit_before: float        # share of the smaller scan within NEAR_M of the other, at the starting poses
    fit_after: float         # the same at the poses the run ends with

    @property
    def pulled(self) -> bool:
        """The link asked for something other than the placement it started from."""
        return self.accepted and self.method not in KEPT_METHODS

    @property
    def hurt(self) -> bool:
        return self.fit_before >= GOOD_FIT and self.fit_before - self.fit_after > HURT_DROP


@dataclass
class Move:
    scan: str
    moved_m: float
    turned_deg: float
    height_m: float
    pulled_by: list = field(default_factory=list)     # names of linked scans whose links asked for a change
    held_by: list = field(default_factory=list)       # names whose links handed the placement back
    toward: Optional[str] = None                      # the pulling link whose fit gained most
    toward_is_stronger: Optional[bool] = None         # that scan's fit with its OTHER neighbours beats this scan's
    hurt: list = field(default_factory=list)          # [(other scan, fit before, fit after)] good links made worse
    helped: list = field(default_factory=list)        # [(other scan, fit before, fit after)] links made better
    guarded: bool = False                             # kept at its old position because of `hurt`
    text: str = ""


def fit_share(a: np.ndarray, b: np.ndarray, ma: np.ndarray, mb: np.ndarray, trees: dict, ka: str, kb: str) -> float:
    """Share of the smaller of two raw-frame clouds within NEAR_M of the other, both placed by 4x4 world matrices."""
    if len(a) > len(b):
        a, b, ma, mb, ka, kb = b, a, mb, ma, kb, ka
    rel = np.linalg.inv(mb) @ ma                      # a's raw frame into b's raw frame
    if kb not in trees:
        trees[kb] = cKDTree(b)
    d, _ = trees[kb].query(a @ rel[:3, :3].T + rel[:3, 3], distance_upper_bound=NEAR_M)
    return float(np.mean(np.isfinite(d))) if len(a) else 0.0


def shift_of(points: np.ndarray, m0: np.ndarray, m1: np.ndarray) -> tuple[float, float, float]:
    """(mean point shift m, yaw turn deg, height change m) of one scan between two world matrices."""
    p0 = points @ m0[:3, :3].T + m0[:3, 3]
    p1 = points @ m1[:3, :3].T + m1[:3, 3]
    return float(np.linalg.norm(p1 - p0, axis=1).mean()), yaw_gap_deg(yaw_of(m1), yaw_of(m0)), float(m1[2, 3] - m0[2, 3])


def measure_links(points: dict, start: dict, final: dict, outcomes: list[dict]) -> list[Link]:
    """One Link per attempted pair. `outcomes`: dicts with source_name, target_name, method, accepted, reason."""
    trees: dict = {}                                  # a tree on a scan's raw points serves every pose
    links = []
    for o in outcomes:
        a, b = o["source_name"], o["target_name"]
        if a not in points or b not in points:
            continue
        before = fit_share(points[a], points[b], start[a], start[b], trees, a, b)
        after = fit_share(points[a], points[b], final[a], final[b], trees, a, b)
        links.append(Link(a, b, o.get("method", ""), bool(o.get("accepted")), o.get("reason", ""), before, after))
    return links


def _strength(scan: str, links: list[Link], exclude: str) -> float:
    """A scan's best starting fit with any neighbour other than `exclude`; 0 when it has none."""
    fits = [l.fit_before for l in links if scan in (l.a, l.b) and exclude not in (l.a, l.b)]
    return max(fits) if fits else 0.0


def explain(points: dict, start: dict, final: dict, links: list[Link], locked: set) -> list[Move]:
    """A Move for every unlocked scan that moved, with its reasons and its effect on every link it is part of."""
    moves = []
    for name in sorted(points):
        if name in locked or name not in start or name not in final:
            continue
        m, deg, dz = shift_of(points[name], start[name], final[name])
        if m < MOVED_M and deg < MOVED_DEG:
            continue
        mv = Move(name, m, deg, dz)
        mine = [l for l in links if name in (l.a, l.b)]
        other = {id(l): (l.b if l.a == name else l.a) for l in mine}
        mv.pulled_by = [other[id(l)] for l in mine if l.pulled]
        mv.held_by = [other[id(l)] for l in mine if l.accepted and not l.pulled]
        pulling = [l for l in mine if l.pulled]
        if pulling:
            best = max(pulling, key=lambda l: l.fit_after - l.fit_before)
            mv.toward = other[id(best)]
            mv.toward_is_stronger = _strength(mv.toward, links, name) >= _strength(name, links, mv.toward)
        mv.hurt = [(other[id(l)], l.fit_before, l.fit_after) for l in mine if l.hurt]
        mv.helped = sorted([(other[id(l)], l.fit_before, l.fit_after) for l in mine if l.fit_after - l.fit_before >= HELPED_GAIN],
                           key=lambda h: h[1] - h[2])
        mv.text = describe(mv)
        moves.append(mv)
    return moves


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def describe(mv: Move) -> str:
    """The plain-language account the report and the page show for one moved scan."""
    parts = [f"{mv.scan} moved {mv.moved_m * 100:.0f} cm"
             + (f" and turned {mv.turned_deg:.1f} degrees" if mv.turned_deg >= MOVED_DEG else "")
             + (f", {'up' if mv.height_m > 0 else 'down'} {abs(mv.height_m) * 100:.0f} cm" if abs(mv.height_m) >= MOVED_M else "")
             + "."]
    if mv.pulled_by:
        parts.append(f"Pulled by its match with {', '.join(mv.pulled_by)}.")
    else:
        parts.append("None of its own links asked for a change; it moved because scans it is linked to moved.")
    if mv.held_by:
        parts.append(f"Held near your placement by {', '.join(mv.held_by)}.")
    if mv.toward is not None:
        parts.append(f"It went toward {mv.toward}, which fits its other neighbours "
                     + ("at least as well as this scan does." if mv.toward_is_stronger else
                        "less well than this scan does: a stronger scan was pulled toward a weaker one, check it."))
    for o, b, a in mv.helped[:MAX_HELPED_NAMED]:
        parts.append(f"Fit with {o.strip()} went from {_pct(b)} to {_pct(a)}.")
    if len(mv.helped) > MAX_HELPED_NAMED:
        parts.append(f"{len(mv.helped) - MAX_HELPED_NAMED} more matches also improved.")
    for o, b, a in mv.hurt:
        parts.append(f"WORSE: fit with {o} fell from {_pct(b)} to {_pct(a)}.")
    if mv.guarded:
        parts.append("Kept at its old position because the move made a good match worse; the move is offered "
                     "as a suggestion instead.")
    parts.append("Every link was solved together in one pass, not one scan at a time.")
    return " ".join(parts)


def guard(points: dict, start: dict, final: dict, outcomes: list[dict], locked: set,
          max_rounds: int = 10) -> tuple[dict, list[Move], list[Link], dict]:
    """Keeps every scan whose move made a good link worse at its starting pose, repeating until no moved scan
    hurts a good link. Returns (poses to apply, moves, links, {scan: the solved pose it was offered})."""
    poses = dict(final)
    offered: dict = {}
    for _ in range(max_rounds):
        links = measure_links(points, start, poses, outcomes)
        moves = explain(points, start, poses, links, locked)
        bad = [m.scan for m in moves if m.hurt]
        if not bad:
            break
        for name in bad:
            offered.setdefault(name, final[name])
            poses[name] = start[name]
    links = measure_links(points, start, poses, outcomes)
    moves = explain(points, start, poses, links, locked)
    for name in offered:                              # report what the kept-back move would have done
        orig = explain(points, start, {**poses, name: final[name]},
                       measure_links(points, start, {**poses, name: final[name]}, outcomes), locked)
        mv = next((m for m in orig if m.scan == name), None)
        if mv is not None:
            mv.guarded = True
            mv.text = describe(mv)
            moves.append(mv)
    return poses, moves, links, offered


def summary(moves: list[Move], links: list[Link]) -> dict:
    """Counts for the report: the number to bring down is `helped_but_hurt`."""
    return {
        "moved": sum(1 for m in moves if not m.guarded),
        "helped_but_hurt": sum(1 for m in moves if m.helped and m.hurt and not m.guarded),
        "good_links_hurt": sum(1 for l in links if l.hurt),
        "good_links": sum(1 for l in links if l.fit_before >= GOOD_FIT),
        "guarded": sum(1 for m in moves if m.guarded),
    }
