"""explain_moves: link fits before and after, move accounts, the collateral count and the guard."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch import explain_moves as em  # noqa: E402


def wall(x0: float, n: int = 20000, seed: int = 0) -> np.ndarray:
    """A 4 m by 2.5 m wall facing x, standing at x = x0, as a scan's raw points."""
    rng = np.random.default_rng(seed)
    return np.column_stack([np.full(n, x0), rng.uniform(0, 4, n), rng.uniform(0, 2.5, n)])


def at(dx: float = 0.0, dy: float = 0.0) -> np.ndarray:
    m = np.eye(4)
    m[0, 3], m[1, 3] = dx, dy
    return m


def outcome(a: str, b: str, method: str = "point_to_plane") -> dict:
    return {"source_name": a, "target_name": b, "method": method, "accepted": True, "reason": "ok"}


def three_scans():
    """A and B see the same wall, C sees it too; only C's link to A asks for a change."""
    pts = {"A": wall(0.0, seed=1), "B": wall(0.0, seed=2), "C": wall(0.0, seed=3)}
    start = {"A": at(), "B": at(), "C": at(0.3)}
    return pts, start


def test_fit_share_counts_points_within_reach():
    pts, _ = three_scans()
    trees = {}
    assert em.fit_share(pts["A"], pts["B"], at(), at(), trees, "A", "B") > 0.9
    assert em.fit_share(pts["A"], pts["B"], at(), at(0.2), trees, "A", "B") == 0.0
    print("    a shared wall fits when placed together and not when 20 cm apart")


def test_move_names_who_pulled_and_what_got_better():
    pts, start = three_scans()
    final = {**start, "C": at()}
    out = [outcome("A", "B", "kept_placement"), outcome("A", "C"), outcome("B", "C", "kept_placement")]
    links = em.measure_links(pts, start, final, out)
    moves = em.explain(pts, start, final, links, locked=set())
    assert [m.scan for m in moves] == ["C"]
    c = moves[0]
    assert c.pulled_by == ["A"] and c.held_by == ["B"] and c.toward == "A"
    assert {o for o, _, _ in c.helped} == {"A", "B"} and not c.hurt
    assert "Pulled by its match with A" in c.text and "WORSE" not in c.text
    print("    the moved scan names the link that pulled it, the one that held it, and the fits it improved")


def test_collateral_is_counted_and_the_guard_keeps_the_scan_back():
    pts, start = three_scans()
    start = {**start, "C": at()}                     # all three start lined up: every link is good
    final = {**start, "B": at(0.3)}                  # B is moved off the wall it shares with A and C
    out = [outcome("A", "B"), outcome("A", "C", "kept_placement"), outcome("B", "C", "kept_placement")]
    links = em.measure_links(pts, start, final, out)
    moves = em.explain(pts, start, final, links, locked=set())
    s = em.summary(moves, links)
    assert s["good_links_hurt"] == 2 and s["moved"] == 1
    assert "WORSE: fit with A" in moves[0].text
    poses, gmoves, glinks, offered = em.guard(pts, start, final, out, locked=set())
    assert list(offered) == ["B"] and np.allclose(poses["B"], start["B"])
    assert em.summary(gmoves, glinks)["good_links_hurt"] == 0
    assert any(m.guarded and "suggestion" in m.text for m in gmoves)
    print("    a move that breaks two good links is counted, and the guard keeps that scan where it was")


def test_locked_scans_are_never_explained_as_moved():
    pts, start = three_scans()
    final = {**start, "C": at()}
    links = em.measure_links(pts, start, final, [outcome("A", "C")])
    assert em.explain(pts, start, final, links, locked={"C"}) == []
    print("    a locked scan is left out of the move accounts")


if __name__ == "__main__":
    from support import run_tests
    raise SystemExit(run_tests(globals()))
