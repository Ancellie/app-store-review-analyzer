from __future__ import annotations

from collections import Counter
from typing import Any, Sequence


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

VALID_RATINGS: frozenset[int] = frozenset(range(1, 6))  # 1 – 5 inclusive


def compute_metrics(reviews: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return aggregate metrics for a processed review dataset.

    Parameters
    ----------
    reviews:
        A sequence of review dicts.  Each dict must contain a ``rating``
        key whose value is an integer in [1, 5].  Entries with missing or
        invalid ratings are counted separately and excluded from numeric
        aggregations — they never cause a crash.

    Returns
    -------
    dict with the following keys:

    ``total``
        Total number of entries passed in (including invalid ones).

    ``valid_count``
        Number of entries whose rating was a valid integer in [1, 5].

    ``invalid_count``
        Number of entries skipped due to missing / out-of-range rating.

    ``average_rating``
        Mean rating rounded to two decimal places, or ``None`` when there
        are no valid reviews.

    ``rating_counts``
        ``{1: n, 2: n, 3: n, 4: n, 5: n}`` — always contains all five
        keys so callers can iterate without extra guard clauses.

    ``rating_distribution``
        ``{1: pct, …, 5: pct}`` — percentage share of each rating
        relative to *valid* reviews, rounded to two decimal places.
        All values are ``0.0`` when there are no valid reviews.
    """
    if not reviews:
        return _empty_metrics(total=0)

    total = len(reviews)
    rating_counts: Counter[int] = Counter()
    invalid_count = 0

    for entry in reviews:
        rating = _extract_rating(entry)
        if rating is None:
            invalid_count += 1
        else:
            rating_counts[rating] += 1

    valid_count = total - invalid_count

    # --- average rating ---------------------------------------------------
    if valid_count == 0:
        average_rating = None
    else:
        weighted_sum = sum(r * n for r, n in rating_counts.items())
        average_rating = round(weighted_sum / valid_count, 2)

    # --- ensure all 5 buckets are always present --------------------------
    counts = {r: rating_counts.get(r, 0) for r in range(1, 6)}

    # --- percentage distribution ------------------------------------------
    if valid_count == 0:
        distribution = {r: 0.0 for r in range(1, 6)}
    else:
        distribution = {
            r: round(counts[r] / valid_count * 100, 2) for r in range(1, 6)
        }

    return {
        "total": total,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "average_rating": average_rating,
        "rating_counts": counts,
        "rating_distribution": distribution,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_rating(entry: dict[str, Any]) -> int | None:
    """Return the integer rating from *entry*, or ``None`` if invalid."""
    raw = entry.get("rating")
    if raw is None:
        return None
    try:
        rating = int(raw)
    except (TypeError, ValueError):
        return None
    return rating if rating in VALID_RATINGS else None


def _empty_metrics(total: int) -> dict[str, Any]:
    """Return a zeroed-out metrics dict for edge-cases (empty input)."""
    return {
        "total": total,
        "valid_count": 0,
        "invalid_count": 0,
        "average_rating": None,
        "rating_counts": {r: 0 for r in range(1, 6)},
        "rating_distribution": {r: 0.0 for r in range(1, 6)},
    }