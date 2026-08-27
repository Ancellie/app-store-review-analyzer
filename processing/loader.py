from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from processing.cleaner import clean_review

logger = logging.getLogger(__name__)

_REQUIRED_FIELDS: tuple[str, ...] = ("rating", "title", "review")
_VALID_RATINGS: frozenset[int] = frozenset(range(1, 6))

_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "review.json"


def load_reviews(path: str | Path = _DEFAULT_PATH) -> list[dict[str, Any]]:
    """Load and validate reviews from *path*.

    Args:
        path: Path to a JSON file that contains either a JSON array of review
              objects or a single review object.

    Returns:
        A list of processed review dicts.  Records that fail validation are
        silently skipped after logging a warning.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Review file not found: {path}")

    with path.open(encoding="utf-8") as fh:
        try:
            raw = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Cannot parse {path} as JSON: {exc}") from exc

    if isinstance(raw, dict):
        if "reviews" in raw:
            raw = raw["reviews"]
        else:
            raw = [raw]

    if not isinstance(raw, list):
        raise ValueError(
            f"Expected a JSON array (or object) at the top level of {path}, "
            f"got {type(raw).__name__}."
        )

    results: list[dict[str, Any]] = []
    for idx, entry in enumerate(raw):
        record = _validate_and_build(entry, idx)
        if record is not None:
            results.append(record)

    logger.info(
        "Loaded %d valid review(s) out of %d total entries from '%s'.",
        len(results),
        len(raw),
        path,
    )
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_and_build(entry: Any, idx: int) -> dict[str, Any] | None:
    """Validate a single raw entry and return a processed record, or None."""
    if not isinstance(entry, dict):
        logger.warning("Entry #%d is not a JSON object — skipped.", idx)
        return None

    # Check required fields exist and are non-empty strings / valid ints.
    for field in _REQUIRED_FIELDS:
        if field not in entry or entry[field] is None:
            logger.warning(
                "Entry #%d is missing required field '%s' — skipped.", idx, field
            )
            return None

    # --- rating ---
    try:
        rating = int(entry["rating"])
    except (TypeError, ValueError):
        logger.warning(
            "Entry #%d has non-integer rating %r — skipped.", idx, entry["rating"]
        )
        return None

    if rating not in _VALID_RATINGS:
        logger.warning(
            "Entry #%d has out-of-range rating %d (expected 1–5) — skipped.",
            idx,
            rating,
        )
        return None

    # --- title ---
    title = _coerce_str(entry.get("title"), idx, "title")
    if title is None:
        return None

    # --- review ---
    review = _coerce_str(entry.get("review"), idx, "review")
    if review is None:
        return None

    return {
        "rating": rating,
        "title": title,
        "review": review,
        "clean_review": clean_review(review),
    }


def _coerce_str(value: Any, idx: int, field: str) -> str | None:
    """Return *value* stripped as a string, or None if blank/invalid."""
    if not isinstance(value, str):
        logger.warning(
            "Entry #%d: field '%s' is not a string (%r) — skipped.", idx, field, value
        )
        return None
    stripped = value.strip()
    if not stripped:
        logger.warning(
            "Entry #%d: field '%s' is empty after stripping — skipped.", idx, field
        )
        return None
    return stripped