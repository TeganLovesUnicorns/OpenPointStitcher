"""The hand-move log: one line per finished hand move, appended to a plain text file in the project folder.
Local only."""
from __future__ import annotations

import datetime
import json
import os
from typing import Optional

FILE_NAME = "hand-moves.jsonl"
OLD_NAME = "hand-moves.old.jsonl"      # the previous file, kept once when the current one is full
MAX_BYTES = 2_000_000                  # about 5,000 moves; then it rolls over and the oldest file is replaced
POSE_KEYS = ("x", "y", "z", "yaw", "pitch", "roll")
KINDS = ("nudge", "undo", "redo", "revert")
HOWS = ("drag", "arrow keys", "typed number", "nudge button", "reset to as-imported", "revert to placement",
        "undo", "redo", "other")
SOURCES = ("hand", "automatic run", "as imported", "unknown")


def _pose(p) -> Optional[dict]:
    if not isinstance(p, dict) or not all(isinstance(p.get(k), (int, float)) for k in POSE_KEYS):
        return None
    return {k: round(float(p[k]), 5) for k in POSE_KEYS}


def clean(move: dict) -> Optional[dict]:
    """One line as it will be written, or None when it is not a valid move. Unknown fields are dropped."""
    if not isinstance(move, dict) or not isinstance(move.get("scan"), str) or not move["scan"]:
        return None
    before, after = _pose(move.get("before")), _pose(move.get("after"))
    if before is None or after is None or before == after:
        return None
    src = move.get("before_source") if isinstance(move.get("before_source"), dict) else {}
    return {
        "at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scan": move["scan"][:200],
        "kind": move.get("kind") if move.get("kind") in KINDS else "nudge",
        "how": move.get("how") if move.get("how") in HOWS else "other",
        "view": str(move.get("view", ""))[:40],
        "before": before, "after": after,
        "before_source": {"kind": src.get("kind") if src.get("kind") in SOURCES else "unknown",
                          "run": str(src.get("run", ""))[:80]},
    }


def append(pdir: str, moves: list) -> int:
    """Append every valid move as one line; roll the file over first when it is full. Returns lines written."""
    lines = [m for m in (clean(x) for x in (moves if isinstance(moves, list) else [])) if m is not None]
    if not lines:
        return 0
    path = os.path.join(pdir, FILE_NAME)
    if os.path.exists(path) and os.path.getsize(path) >= MAX_BYTES:
        os.replace(path, os.path.join(pdir, OLD_NAME))
    with open(path, "a", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line) + "\n")
    return len(lines)


def read(pdir: str) -> list[dict]:
    """Every line, oldest first (the rolled-over file, then the current one); unreadable lines are skipped."""
    out = []
    for name in (OLD_NAME, FILE_NAME):
        path = os.path.join(pdir, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                try:
                    out.append(json.loads(raw))
                except ValueError:
                    continue
    return out
