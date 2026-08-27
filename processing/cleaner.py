from __future__ import annotations

import re
import unicodedata


# ---------------------------------------------------------------------------
# Compiled patterns — defined once at import time for performance
# ---------------------------------------------------------------------------

# Matches any HTML tag (simple heuristic; sufficient for app-store reviews).
_RE_HTML = re.compile(r"<[^>]+>", re.UNICODE)

# Matches http/https/ftp URLs.
_RE_URL = re.compile(
    r"https?://\S+|ftp://\S+",
    re.UNICODE | re.IGNORECASE,
)

# Collapses runs of horizontal whitespace (spaces, tabs) into a single space.
_RE_SPACES = re.compile(r"[^\S\n]+", re.UNICODE)

# Collapses runs of newlines (including carriage returns) into a single newline.
_RE_NEWLINES = re.compile(r"[\r\n]+", re.UNICODE)


def clean_review(review: str) -> str:
    """Return a language-agnostically cleaned version of *review*.

    Steps (in order):
        1. Unicode NFC normalisation — resolves composed vs. decomposed
           characters so that Cyrillic/Latin look-alikes are stable.
        2. Strip HTML tags.
        3. Replace URLs with a single space (URLs carry no sentiment signal
           and would confuse tokenisers).
        4. Normalise whitespace — collapse spaces and newlines independently
           so paragraph structure is preserved.
        5. Strip leading / trailing whitespace.

    Args:
        review: Raw review review (may be empty or already clean).

    Returns:
        Cleaned review string.  Never raises; returns an empty string for
        non-string or empty input.
    """
    if not isinstance(review, str):
        return ""

    # 1. Unicode normalisation (NFC)
    review = unicodedata.normalize("NFC", review)

    # 2. Remove HTML tags
    review = _RE_HTML.sub(" ", review)

    # 3. Remove URLs
    review = _RE_URL.sub(" ", review)

    # 4. Normalise whitespace
    review = _RE_SPACES.sub(" ", review)
    review = _RE_NEWLINES.sub("\n", review)

    # 5. Strip edges
    review = review.strip()

    return review