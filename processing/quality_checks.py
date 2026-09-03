from __future__ import annotations

import logging
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field

from processing.contrastive import document_frequency
from processing.keywords import NegativeTerm, NegativeTermsReport, extract_negative_texts, extract_positive_texts

logger = logging.getLogger(__name__)

FIVE_STAR_NEGATIVE_WARN_RATIO = 0.15
ONE_STAR_POSITIVE_WARN_RATIO = 0.15

_DEFAULT_TOP_N = 5


class RatingSentimentSummary(BaseModel):
    """How often the sentiment label disagrees with an unambiguous rating."""

    five_star_negative_count: int
    five_star_negative_ratio: float = Field(description="Relative to all 5-star reviews")
    one_star_positive_count: int
    one_star_positive_ratio: float = Field(description="Relative to all 1-star reviews")


class SuspiciousTerm(BaseModel):
    """A top-ranked negative term that is at least as common in positive
    reviews as in negative ones -- i.e. it doesn't look genuinely
    negative-specific despite ranking highly."""

    term: str
    kind: Literal["keyword", "phrase"]
    negative_prevalence: float = Field(description="Share of negative reviews containing this term")
    positive_prevalence: float = Field(description="Share of positive reviews containing this term")


class NLPQualityReport(BaseModel):
    """Lightweight, diagnostic-only sanity report. Never modifies or
    corrects the underlying data -- it only surfaces numbers a human
    should look at."""

    negative_review_count: int
    positive_review_count: int
    rating_sentiment_inconsistency: RatingSentimentSummary
    top_negative_terms: dict[str, list[str]] = Field(
        description="Method name -> top negative-specific keywords (post contrastive ranking)"
    )
    top_negative_phrases: dict[str, list[str]] = Field(
        description="Method name -> top negative-specific phrases (post contrastive ranking)"
    )
    suspicious_terms: dict[str, list[SuspiciousTerm]] = Field(
        default_factory=dict,
        description="Method name -> flagged top terms/phrases that aren't clearly negative-specific",
    )
    warnings: list[str] = Field(default_factory=list)


def _rating_sentiment_summary(
    records: Sequence[dict[str, Any]],
    sentiment_field: str,
    label_field: str = "label",
) -> RatingSentimentSummary:
    five_star_total = 0
    five_star_negative = 0
    one_star_total = 0
    one_star_positive = 0

    for record in records:
        rating = record.get("rating")
        sentiment = record.get(sentiment_field)
        label = sentiment.get(label_field) if isinstance(sentiment, dict) else None

        if rating == 5:
            five_star_total += 1
            if label == "negative":
                five_star_negative += 1
        elif rating == 1:
            one_star_total += 1
            if label == "positive":
                one_star_positive += 1

    return RatingSentimentSummary(
        five_star_negative_count=five_star_negative,
        five_star_negative_ratio=round(five_star_negative / five_star_total, 4) if five_star_total else 0.0,
        one_star_positive_count=one_star_positive,
        one_star_positive_ratio=round(one_star_positive / one_star_total, 4) if one_star_total else 0.0,
    )


def _flag_suspicious_terms(
    terms: Sequence[NegativeTerm],
    negative_texts: Sequence[str],
    positive_texts: Sequence[str],
    kind: Literal["keyword", "phrase"],
) -> list[SuspiciousTerm]:
    neg_total = len(negative_texts)
    pos_total = len(positive_texts)

    flagged: list[SuspiciousTerm] = []
    for term in terms:
        neg_df = document_frequency(term.term, negative_texts)
        pos_df = document_frequency(term.term, positive_texts)
        neg_prevalence = neg_df / neg_total if neg_total else 0.0
        pos_prevalence = pos_df / pos_total if pos_total else 0.0

        if pos_prevalence >= neg_prevalence:
            flagged.append(
                SuspiciousTerm(
                    term=term.term,
                    kind=kind,
                    negative_prevalence=round(neg_prevalence, 4),
                    positive_prevalence=round(pos_prevalence, 4),
                )
            )
    return flagged


def run_quality_checks(
    records: Sequence[dict[str, Any]],
    keyword_reports: dict[str, NegativeTermsReport],
    sentiment_field: str = "sentiment_transformer",
    text_field: str = "clean_review",
    top_n: int = _DEFAULT_TOP_N,
) -> NLPQualityReport:
    """Run all lightweight quality checks and return a single report."""
    negative_texts = extract_negative_texts(records, sentiment_field=sentiment_field, text_field=text_field)
    positive_texts = extract_positive_texts(records, sentiment_field=sentiment_field, text_field=text_field)

    rating_summary = _rating_sentiment_summary(records, sentiment_field)

    top_negative_terms = {name: [t.term for t in report.keywords[:top_n]] for name, report in keyword_reports.items()}
    top_negative_phrases = {name: [p.term for p in report.phrases[:top_n]] for name, report in keyword_reports.items()}

    suspicious_terms: dict[str, list[SuspiciousTerm]] = {}
    for name, report in keyword_reports.items():
        flagged = _flag_suspicious_terms(report.keywords[:top_n], negative_texts, positive_texts, "keyword")
        flagged += _flag_suspicious_terms(report.phrases[:top_n], negative_texts, positive_texts, "phrase")
        if flagged:
            suspicious_terms[name] = flagged

    warnings: list[str] = []

    if rating_summary.five_star_negative_count and rating_summary.five_star_negative_ratio >= FIVE_STAR_NEGATIVE_WARN_RATIO:
        warnings.append(
            f"{rating_summary.five_star_negative_count} five-star review(s) "
            f"({rating_summary.five_star_negative_ratio:.1%}) were classified as negative sentiment. "
            "This is a sentiment-model issue, not a keyword-ranking issue -- check the "
            "confusion matrix/evaluation for this method before trusting its negative subset."
        )

    if rating_summary.one_star_positive_count and rating_summary.one_star_positive_ratio >= ONE_STAR_POSITIVE_WARN_RATIO:
        warnings.append(
            f"{rating_summary.one_star_positive_count} one-star review(s) "
            f"({rating_summary.one_star_positive_ratio:.1%}) were classified as positive sentiment."
        )

    for name, flagged in suspicious_terms.items():
        warnings.append(
            f"{len(flagged)} of the top {top_n} negative term(s)/phrase(s) from '{name}' are at least "
            "as common in positive reviews as in negative ones -- they may not be genuinely "
            "negative-specific."
        )

    if not negative_texts:
        warnings.append("No negative reviews were found; keyword and quality checks have nothing to evaluate.")

    return NLPQualityReport(
        negative_review_count=len(negative_texts),
        positive_review_count=len(positive_texts),
        rating_sentiment_inconsistency=rating_summary,
        top_negative_terms=top_negative_terms,
        top_negative_phrases=top_negative_phrases,
        suspicious_terms=suspicious_terms,
        warnings=warnings,
    )