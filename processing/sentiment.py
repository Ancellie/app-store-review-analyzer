from __future__ import annotations

import logging
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

SentimentLabel = Literal["positive", "negative", "neutral"]

# Standard VADER thresholds (per the library's own documentation).
_POSITIVE_THRESHOLD = 0.05
_NEGATIVE_THRESHOLD = -0.05

# One analyzer instance per process — it's stateless after construction,
# so re-creating it per call would just waste time re-parsing the lexicon.
_analyzer = SentimentIntensityAnalyzer()


class SentimentResult(BaseModel):
    """Normalized sentiment output, shared shape across all NLP approaches.

    ``method`` identifies which underlying approach produced this result
    (e.g. "vader-en" here; "transformer-xlmr" or "llm-groq" later), so
    results from different approaches can be compared side by side.
    """

    label: SentimentLabel
    compound: float
    pos: float
    neu: float
    neg: float
    method: str = Field(default="vader-en")


def analyze_sentiment(text: str) -> SentimentResult:
    """Classify *text* as positive, negative, or neutral using VADER.

    VADER's lexicon is English-only.
    Args:
        text: Review text to classify. Must be a string.

    Returns:
        A SentimentResult with a normalized label and the raw VADER scores.

    Raises:
        TypeError: If *text* is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"analyze_sentiment expects str, got {type(text).__name__}")

    stripped = text.strip()
    if not stripped:
        # No signal to analyze. We treat this as neutral by convention
        # rather than calling VADER on an empty string (which would also
        # yield compound=0.0, but this makes the decision explicit).
        logger.debug("Empty/whitespace-only text passed to analyze_sentiment; returning neutral.")
        return SentimentResult(label="neutral", compound=0.0, pos=0.0, neu=1.0, neg=0.0)

    scores = _analyzer.polarity_scores(stripped)
    compound = scores["compound"]

    if compound >= _POSITIVE_THRESHOLD:
        label: SentimentLabel = "positive"
    elif compound <= _NEGATIVE_THRESHOLD:
        label = "negative"
    else:
        label = "neutral"

    return SentimentResult(
        label=label,
        compound=compound,
        pos=scores["pos"],
        neu=scores["neu"],
        neg=scores["neg"],
    )


def attach_sentiment(
    records: Sequence[dict[str, Any]],
    text_field: str = "clean_review",
) -> list[dict[str, Any]]:
    """Return copies of *records* enriched with a ``"sentiment"`` key.

    Args:
        records: Processed review records, as produced by
            ``processing.loader.load_reviews``.
        text_field: Key to read the text to analyze from.

    Returns:
        New list of dicts, each with all original keys plus ``"sentiment"``
        (the ``SentimentResult`` as a plain dict via ``.model_dump()``).
    """
    enriched: list[dict[str, Any]] = []
    for record in records:
        raw_text = record.get(text_field, "")
        text = raw_text if isinstance(raw_text, str) else ""
        result = analyze_sentiment(text)
        enriched.append({**record, "sentiment": result.model_dump()})
    return enriched