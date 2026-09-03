from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from processing.keywords import NegativeTerm, NegativeTermsReport

# Add-alpha (Laplace-style) smoothing for the log-odds ratio. Small
# relative to typical review counts (tens-to-hundreds per corpus) so it
# barely perturbs well-attested terms, but large enough to keep the
# ratio finite for terms that occur zero times in one corpus.
DEFAULT_SMOOTHING_ALPHA = 0.5

# How much wider a candidate pool to pull from a method-specific
# extractor before re-ranking (see `expand_pool_size`).
CANDIDATE_POOL_MULTIPLIER = 4
MAX_CANDIDATE_POOL_SIZE = 60

_METHOD_SUFFIX = "-contrastive"

# Words shorter than this don't get a trailing wildcard when matched
# against review text (see `_word_pattern`).
_MIN_LENGTH_FOR_SUFFIX_WILDCARD = 4


def expand_pool_size(requested: int) -> int:
    """How many raw candidates to request from a method-specific"""
    return min(requested * CANDIDATE_POOL_MULTIPLIER, MAX_CANDIDATE_POOL_SIZE)


def _word_pattern(word: str) -> str:
    r"""Regex fragment matching *word* (and its common inflections)."""
    escaped = re.escape(word)
    if len(word) < _MIN_LENGTH_FOR_SUFFIX_WILDCARD:
        return rf"\b{escaped}\b"
    return rf"\b{escaped}\w*"


def _term_present(term: str, text: str) -> bool:
    """Return True if every word of *term* appears (per `_word_pattern`) in *text*."""
    words = term.lower().split()
    if not words:
        return False
    return all(re.search(_word_pattern(word), text, re.IGNORECASE) for word in words)


def document_frequency(term: str, texts: Sequence[str]) -> int:
    """Count how many of *texts* contain *term* (see `_term_present`)."""
    return sum(1 for text in texts if isinstance(text, str) and _term_present(term, text))


def smoothed_log_odds_ratio(
    neg_df: int,
    neg_total: int,
    pos_df: int,
    pos_total: int,
    alpha: float = DEFAULT_SMOOTHING_ALPHA,
) -> float:
    """Log-odds ratio of *term* appearing in a negative vs. a positive review."""
    neg_odds = (neg_df + alpha) / (neg_total - neg_df + alpha)
    pos_odds = (pos_df + alpha) / (pos_total - pos_df + alpha)
    return math.log(neg_odds) - math.log(pos_odds)


def apply_contrastive_ranking(
    candidates: Sequence["NegativeTerm"],
    negative_texts: Sequence[str],
    positive_texts: Sequence[str],
    alpha: float = DEFAULT_SMOOTHING_ALPHA,
) -> list["NegativeTerm"]:
    """Re-score and re-sort *candidates* by negative specificity."""
    neg_total = len(negative_texts)
    pos_total = len(positive_texts)

    rescored: list["NegativeTerm"] = []
    for candidate in candidates:
        neg_df = document_frequency(candidate.term, negative_texts)
        pos_df = document_frequency(candidate.term, positive_texts)
        score = smoothed_log_odds_ratio(neg_df, neg_total, pos_df, pos_total, alpha)
        rescored.append(
            candidate.model_copy(update={"score": round(score, 4), "document_frequency": neg_df})
        )

    rescored.sort(key=lambda term: term.score, reverse=True)
    return rescored


def rerank_report_contrastively(
    report: "NegativeTermsReport",
    negative_texts: Sequence[str],
    positive_texts: Sequence[str],
    max_keywords: int | None = None,
    max_phrases: int | None = None,
    alpha: float = DEFAULT_SMOOTHING_ALPHA,
) -> "NegativeTermsReport":
    reranked_keywords = apply_contrastive_ranking(report.keywords, negative_texts, positive_texts, alpha)
    reranked_phrases = apply_contrastive_ranking(report.phrases, negative_texts, positive_texts, alpha)

    if max_keywords is not None:
        reranked_keywords = reranked_keywords[:max_keywords]
    if max_phrases is not None:
        reranked_phrases = reranked_phrases[:max_phrases]

    return report.model_copy(
        update={
            "keywords": reranked_keywords,
            "phrases": reranked_phrases,
            "method": f"{report.method}{_METHOD_SUFFIX}",
        }
    )