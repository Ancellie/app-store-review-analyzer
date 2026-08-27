from __future__ import annotations

import logging
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field
from transformers import Pipeline, pipeline

logger = logging.getLogger(__name__)

SentimentLabel = Literal["positive", "negative", "neutral"]

_MODEL_NAME = "tabularisai/multilingual-sentiment-analysis"
_METHOD_TAG = "transformer-mdistilbert-tabularisai"

# The model's native 5-class labels, mapped to our normalized 3-class scheme.
# Kept explicit (not inferred from string patterns) so a label the model
# didn't document doesn't silently fall into the wrong bucket.
_RAW_LABEL_TO_BUCKET: dict[str, SentimentLabel] = {
    "Very Negative": "negative",
    "Negative": "negative",
    "Neutral": "neutral",
    "Positive": "positive",
    "Very Positive": "positive",
}

# Lazily created — loading the model at import time would slow down every
# test/module that imports `processing`, even ones that never touch
# sentiment analysis. Populated on first call, reused after that.
_pipeline: Pipeline | None = None


class TransformerSentimentResult(BaseModel):
    """Normalized sentiment output from the Transformer analyzer.

    Shape is intentionally close to ``sentiment.SentimentResult`` so the
    two approaches can be compared side by side, but ``score``/``raw_label``/
    ``raw_scores`` replace VADER's pos/neu/neg compound breakdown since the
    two models don't produce comparable internal scores.
    """

    label: SentimentLabel
    score: float = Field(ge=0.0, le=1.0, description="Confidence of the normalized label")
    raw_label: str = Field(description="Highest-scoring native model label, e.g. 'Very Positive'")
    raw_scores: dict[str, float] = Field(description="Full probability distribution over native labels")
    method: str = Field(default=_METHOD_TAG)


def _get_pipeline() -> Pipeline:
    """Return the cached HF pipeline, creating it on first call."""
    global _pipeline
    if _pipeline is None:
        logger.info("Loading Transformer sentiment model '%s' (first use)...", _MODEL_NAME)
        _pipeline = pipeline(
            "text-classification",
            model=_MODEL_NAME,
            tokenizer=_MODEL_NAME,
            top_k=None,  # return scores for all 5 classes, not just the top one
        )
    return _pipeline


def _collapse_scores(raw_scores: list[dict[str, Any]]) -> tuple[SentimentLabel, float, str, dict[str, float]]:
    """Collapse the model's native 5-class distribution into 3 buckets.

    Returns (normalized_label, bucket_confidence, top_raw_label, raw_scores_dict).
    """
    scores_by_label = {item["label"]: float(item["score"]) for item in raw_scores}

    bucket_totals: dict[SentimentLabel, float] = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    for raw_label, score in scores_by_label.items():
        bucket = _RAW_LABEL_TO_BUCKET.get(raw_label)
        if bucket is None:
            raise ValueError(
                f"Unrecognized label '{raw_label}' from {_MODEL_NAME}; "
                f"update _RAW_LABEL_TO_BUCKET before trusting results."
            )
        bucket_totals[bucket] += score

    normalized_label = max(bucket_totals, key=bucket_totals.get)
    top_raw_label = max(scores_by_label, key=scores_by_label.get)

    return normalized_label, round(bucket_totals[normalized_label], 4), top_raw_label, scores_by_label


def analyze_sentiment(text: str) -> TransformerSentimentResult:
    """Classify *text* as positive, negative, or neutral using the
    multilingual Transformer model.

    Unlike VADER, this model was explicitly fine-tuned to support Ukrainian,
    Russian, and English, so results on non-English review text should be
    meaningfully more reliable than the VADER baseline.

    Args:
        text: Review text to classify. Must be a string.

    Returns:
        A TransformerSentimentResult with the normalized label, confidence,
        and full native-label distribution for interpretability.

    Raises:
        TypeError: If *text* is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"analyze_sentiment expects str, got {type(text).__name__}")

    stripped = text.strip()
    if not stripped:
        logger.debug("Empty/whitespace-only text passed to analyze_sentiment; returning neutral.")
        return TransformerSentimentResult(
            label="neutral",
            score=0.0,
            raw_label="Neutral",
            raw_scores={"Neutral": 0.0},
        )

    raw_scores = _get_pipeline()(stripped)[0]  # top_k=None -> list of {label, score} per input
    label, score, raw_label, scores_dict = _collapse_scores(raw_scores)

    return TransformerSentimentResult(
        label=label,
        score=score,
        raw_label=raw_label,
        raw_scores=scores_dict,
    )


def analyze_sentiment_batch(texts: Sequence[str], batch_size: int = 16) -> list[TransformerSentimentResult]:
    """Classify multiple texts in one pass.

    Empty/whitespace-only entries are handled without a model call (same
    behavior as :func:`analyze_sentiment`) and are re-inserted at their
    original position, so output order always matches input order.

    Args:
        texts: Review texts to classify.
        batch_size: Batch size passed to the HF pipeline. 16 is a
            reasonable default for CPU inference on short review text;
            increase only if profiling shows headroom.

    Returns:
        List of TransformerSentimentResult, same length and order as *texts*.
    """
    results: list[TransformerSentimentResult | None] = [None] * len(texts)
    non_empty_indices: list[int] = []
    non_empty_texts: list[str] = []

    for idx, text in enumerate(texts):
        if not isinstance(text, str):
            raise TypeError(f"analyze_sentiment_batch expects str items, got {type(text).__name__} at index {idx}")
        stripped = text.strip()
        if not stripped:
            results[idx] = TransformerSentimentResult(
                label="neutral", score=0.0, raw_label="Neutral", raw_scores={"Neutral": 0.0},
            )
        else:
            non_empty_indices.append(idx)
            non_empty_texts.append(stripped)

    if non_empty_texts:
        batch_raw_scores = _get_pipeline()(non_empty_texts, batch_size=batch_size)
        for idx, raw_scores in zip(non_empty_indices, batch_raw_scores):
            label, score, raw_label, scores_dict = _collapse_scores(raw_scores)
            results[idx] = TransformerSentimentResult(
                label=label, score=score, raw_label=raw_label, raw_scores=scores_dict,
            )

    return results  # type: ignore[return-value]  # every slot is filled by this point


def attach_sentiment_transformer(
    records: Sequence[dict[str, Any]],
    text_field: str = "clean_review",
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    """Return copies of *records* enriched with a ``"sentiment_transformer"`` key.

    Mirrors ``sentiment.attach_sentiment``'s contract (new dicts, original
    keys preserved) but uses batch inference since it's operating on the
    whole dataset at once rather than one review at a time.

    Args:
        records: Processed review records, as produced by
            ``processing.loader.load_reviews``.
        text_field: Key to read the text to analyze from.
        batch_size: Forwarded to :func:`analyze_sentiment_batch`.

    Returns:
        New list of dicts, each with all original keys plus
        ``"sentiment_transformer"`` (the TransformerSentimentResult as a
        plain dict via ``.model_dump()``).
    """
    texts = [
        record.get(text_field, "") if isinstance(record.get(text_field, ""), str) else ""
        for record in records
    ]
    results = analyze_sentiment_batch(texts, batch_size=batch_size)

    return [
        {**record, "sentiment_transformer": result.model_dump()}
        for record, result in zip(records, results)
    ]