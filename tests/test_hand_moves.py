"""The hand-nudge log: what gets written, the size cap, the reader, and that tidying saved versions never
touches it."""
from __future__ import annotations

import datetime
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "measurements"))

from open_point_stitch import backup_cleaner, hand_moves  # noqa: E402
try:
    from hand_moves_summary import summarise  # noqa: E402  (a developer tool, not in the public repository)
except ImportError:
    summarise = None


def pose(x=0.0, yaw=0.0):
    return {"x": x, "y": 1.0, "z": 0.0, "yaw": yaw, "pitch": 0.0, "roll": 0.0}


def move(scan="kitchen", x=0.5, kind="nudge", source="hand"):
    return {"scan": scan, "kind": kind, "how": "drag", "view": "plan", "before": pose(0.0), "after": pose(x, 2.0),
            "before_source": {"kind": source, "run": "align_auto at 2026-10-02T20:02:38Z" if source != "hand" else ""}}


def test_valid_moves_are_appended_one_line_each_and_bad_ones_dropped():
    d = tempfile.mkdtemp()
    try:
        still = {**move(), "after": pose(0.0)}
        n = hand_moves.append(d, [move(), still, {"scan": "x"}, "nonsense", move(source="automatic run")])
        assert n == 2, n                                        # unchanged pose, missing poses and junk are dropped
        lines = [json.loads(l) for l in open(os.path.join(d, hand_moves.FILE_NAME))]
        assert [l["before_source"]["kind"] for l in lines] == ["hand", "automatic run"]
        assert set(lines[0]) == {"at", "scan", "kind", "how", "view", "before", "after", "before_source"}
    finally:
        shutil.rmtree(d)
    print("    two good moves written as two lines; an unchanged pose and malformed input are not written")


def test_a_full_log_rolls_over_instead_of_growing():
    d = tempfile.mkdtemp()
    saved = hand_moves.MAX_BYTES
    try:
        line = len(json.dumps(hand_moves.clean(move()))) + 1
        hand_moves.MAX_BYTES = 2 * line                         # two lines fill it
        for _ in range(3):
            hand_moves.append(d, [move()])
        assert os.path.exists(os.path.join(d, hand_moves.OLD_NAME))
        assert len(hand_moves.read(d)) == 3                     # nothing lost at the first roll-over
        for _ in range(5):
            hand_moves.append(d, [move()])
        assert os.path.getsize(os.path.join(d, hand_moves.FILE_NAME)) <= 2 * line   # never grows past the cap
    finally:
        hand_moves.MAX_BYTES = saved
        shutil.rmtree(d)
    print("    past the cap the file is renamed .old and a new one starts")


def test_reader_counts_moves_corrections_and_undos():
    if summarise is None:
        print("    SKIP: tools/measurements/hand_moves_summary.py is not in this checkout")
        return
    lines = [hand_moves.clean(m) for m in (move(), move(source="automatic run"), move(kind="undo"))]
    s = summarise(lines)
    assert s["kitchen"]["moves"] == 2 and s["kitchen"]["corrections"] == 1 and s["kitchen"]["undone"] == 1
    assert abs(s["kitchen"]["mean_m"] - 0.5) < 1e-6 and abs(s["kitchen"]["mean_deg"] - 2.0) < 1e-6
    print("    the reader counts 2 moves, 1 of them fixing an automatic run, and 1 undo kept apart")


def test_tidying_saved_versions_never_touches_the_log():
    d = tempfile.mkdtemp()
    try:
        hand_moves.append(d, [move()])
        open(os.path.join(d, hand_moves.OLD_NAME), "w").write("{}\n")
        now = datetime.datetime(2026, 10, 2, 22, 0, tzinfo=datetime.timezone.utc)
        old = "placement.backup-20260101T000000.000Z.json"
        open(os.path.join(d, old), "w").write("{}")
        listing = [{"id": hand_moves.FILE_NAME, "at": "2026-01-01T00:00:00.000Z", "size": 10},
                   {"id": hand_moves.OLD_NAME, "at": "2026-01-01T00:00:01.000Z", "size": 10},
                   {"id": old, "at": "2026-01-01T00:00:02.000Z", "size": 2},
                   {"id": "placement.backup-20260101T000003.000Z.json", "at": "2026-01-01T00:00:03.000Z", "size": 2},
                   {"id": "placement.backup-20261002T215959.000Z.json", "at": "2026-10-02T21:59:59.000Z", "size": 2}]
        for b in listing[3:]:
            open(os.path.join(d, b["id"]), "w").write("{}")
        planned = backup_cleaner.plan(d, listing, now, 1.0, 1.0)["remove_ids"]
        assert hand_moves.FILE_NAME in planned        # the retention rules alone WOULD pick it; the guard is what saves it
        result = backup_cleaner.tidy(d, listing, lambda i: os.path.join(d, i), now, 1.0, 1.0)
        assert os.path.exists(os.path.join(d, hand_moves.FILE_NAME)) and os.path.exists(os.path.join(d, hand_moves.OLD_NAME))
        assert hand_moves.FILE_NAME not in result["removed_ids"] and hand_moves.OLD_NAME not in result["removed_ids"]
        assert old in result["removed_ids"], result["removed_ids"]   # the tidy still does its own job
    finally:
        shutil.rmtree(d)
    print("    the tidy-up of saved versions leaves hand-moves.jsonl and its .old file alone, even if handed them")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print(f"PASS {name}")
            except Exception as exc:
                failures += 1; print(f"FAIL {name}: {exc}")
    raise SystemExit(1 if failures else 0)
