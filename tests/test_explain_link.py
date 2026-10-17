"""explain.py: the links table's plain-language sentences."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from open_point_stitch import explain as w  # noqa: E402


def test_numbers_read_as_people_say_them():
    assert w.pct(0.027) == "2.7%" and w.pct(0.31) == "31%"
    assert w.cm(0.015) == "1.5 cm" and w.cm(0.33) == "33 cm"


def test_a_fine_tune_names_the_failed_check_and_flags_a_big_one():
    check = w.too_few_on_surface("GR from entry", "HL Aisle 2", 0.027, 0.05)
    assert check == "only 2.7% of GR from entry's points landed on a surface of HL Aisle 2 (5.0% needed)", check
    big = w.kept_and_fine_tuned(0.33, 1.7, check)
    assert big.startswith("Kept near where you put it, fine-tuned by 33 cm and 1.7 degrees.") and check in big, big
    assert "worth a look" in big
    assert "worth a look" not in w.kept_and_fine_tuned(0.02, 0.2, check)


def test_no_sentence_uses_the_word_fitness():
    texts = [w.too_few_on_surface("a", "b", 0.01, 0.05), w.too_few_of_possible("a", "b", 0.3, 0.5),
             w.too_loose(0.04, 0.03), w.too_few_matched(80, 100), w.too_little_overlap(0.1, 0.2, 0.15),
             w.matched("a", "b", 0.3, 0.01), w.kept_no_better(0.1, 0.1, 0.05), w.all_tilted(4, 5),
             w.all_disagreed(3), w.tips_off_level(6.2), w.KEPT_NO_EVIDENCE, w.NO_START]
    assert not [t for t in texts if "fitness" in t.lower() or "rmse" in t.lower()], texts


if __name__ == "__main__":
    from support import run_tests
    raise SystemExit(run_tests(globals()))
