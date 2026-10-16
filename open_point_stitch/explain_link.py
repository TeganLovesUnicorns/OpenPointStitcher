"""Plain-language sentences saying what happened to one pair of scans, for the links table. Each function words
one outcome or one failed check; the technical reason stays alongside for reports. Shared by align_pair.py and
align_pair_cuda.py."""
from __future__ import annotations


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
