"""Shared ranking helpers for the repository's main evaluation protocol."""

from __future__ import annotations

from collections.abc import Collection, Mapping


TIE_MODES = ("average", "optimistic")


def average_tie_rank(greater: int, equal: int) -> float:
    """Return the average 1-based rank among entities tied with the target."""
    if greater < 0 or equal < 1:
        raise ValueError(f"Invalid rank counts: greater={greater}, equal={equal}")
    return float(greater + (equal + 1) / 2.0)


def optimistic_tie_rank(greater: int) -> float:
    """Return the best-case 1-based rank: the target is placed first among ties."""
    if greater < 0:
        raise ValueError(f"Invalid rank count: greater={greater}")
    return float(greater + 1)


def rank_from_counts(greater: int, equal: int, tie_mode: str) -> float:
    """Dispatch to the requested tie-handling rule given (greater, equal) counts."""
    if tie_mode == "average":
        return average_tie_rank(greater, equal)
    if tie_mode == "optimistic":
        return optimistic_tie_rank(greater)
    raise ValueError(f"Unsupported tie_mode={tie_mode!r}; expected one of {TIE_MODES}")


def sparse_filtered_rank(
    scores: Mapping[str, float],
    target: str,
    entity_universe: Collection[str],
    true_answers: Collection[str],
    default_score: float = 0.0,
    tie_mode: str = "average",
) -> float:
    """Filtered full-entity rank for a sparse candidate score mapping.

    Entities absent from ``scores`` receive ``default_score`` and are ranked
    against the target exactly like any other competing entity -- they are
    never simply skipped. ``default_score`` must be a genuine floor/neutral
    value for the method's score semantics (e.g. 0 for a non-negative
    evidence sum where "not returned" means "no evidence"), not an
    arbitrary placeholder chosen for convenience. Other known correct
    answers are excluded (filtered) before ranking. ``tie_mode`` selects
    "average" (Main Protocol) or "optimistic" (SOTA Protocol) tie handling;
    both share the same filtering and full-entity-universe logic, so at this
    function's level -- a single query's rank computation -- they differ
    only in how ties are broken. The Main and SOTA protocols overall also
    differ in query direction (Main is bidirectional, SOTA is tail-only);
    that axis is decided by the caller, not by this function.
    """
    if tie_mode not in TIE_MODES:
        raise ValueError(f"Unsupported tie_mode={tie_mode!r}; expected one of {TIE_MODES}")
    universe = entity_universe if isinstance(entity_universe, set) else set(entity_universe)
    if target not in universe:
        raise ValueError(f"Target entity {target!r} is missing from the ranking universe")

    filtered = {answer for answer in true_answers if answer != target and answer in universe}
    eligible_count = len(universe) - len(filtered)
    explicit = {
        entity: float(score)
        for entity, score in scores.items()
        if entity in universe and entity not in filtered
    }
    target_score = explicit.get(target, float(default_score))
    implicit_count = eligible_count - len(explicit)
    if implicit_count < 0:
        raise ValueError("Sparse score mapping contains more entities than the ranking universe")

    greater = sum(score > target_score for score in explicit.values())
    equal = sum(score == target_score for score in explicit.values())
    if default_score > target_score:
        greater += implicit_count
    elif default_score == target_score:
        equal += implicit_count
    return rank_from_counts(greater, equal, tie_mode)
