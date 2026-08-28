from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class NegativeTerm(BaseModel):
    """A single ranked term (keyword or phrase) from the negative-review corpus."""

    term: str
    score: float
    document_frequency: int


class NegativeTermsReport(BaseModel):
    """Structured result of negative keyword/phrase analysis.

    ``score`` reflects TF-IDF importance within the negative-review corpus,
    not sentiment polarity of the term itself — negativity is established
    upstream by filtering to reviews already classified as negative.
    """

    negative_review_count: int
    keywords: list[NegativeTerm] = Field(default_factory=list)
    phrases: list[NegativeTerm] = Field(default_factory=list)
    method: str = "tfidf-ngram"


def extract_negative_texts(
    records: Sequence[dict[str, Any]],
    sentiment_field: str = "sentiment_transformer",
    text_field: str = "clean_review",
) -> list[str]:
    """Return the cleaned text of every record classified as negative.

    This is the only function that knows the shape of the enriched review
    records (``record[sentiment_field]["label"]``). Records missing the
    sentiment field, or not labeled "negative", are skipped.

    Args:
        records: Processed review records, optionally enriched with a
            sentiment result (e.g. via ``attach_sentiment_transformer``).
        sentiment_field: Key holding the sentiment result dict. Defaults to
            the multilingual Transformer output, since VADER is English-only
            and using it to decide "is this negative" would misclassify
            Ukrainian/Russian reviews.
        text_field: Key holding the text to analyze.

    Returns:
        List of clean_review strings for negative-labeled records, in
        their original order. May contain empty strings if a record's
        text field is blank.
    """
    texts: list[str] = []
    for record in records:
        sentiment_data = record.get(sentiment_field)
        if not isinstance(sentiment_data, dict):
            continue
        if sentiment_data.get("label") != "negative":
            continue
        text = record.get(text_field, "")
        texts.append(text if isinstance(text, str) else "")
    return texts


def analyze_negative_terms(
    texts: Sequence[str],
    max_keywords: int = 15,
    max_phrases: int = 15,
) -> NegativeTermsReport:
    """Rank unigram keywords and multi-word phrases from negative review text.

    Uses a single TF-IDF vectorizer over unigrams through trigrams; terms
    are split into "keywords" (single word) and "phrases" (2-3 words) based
    on the same underlying corpus statistics, so scores are comparable
    across both buckets.

    ``min_df``/``max_df`` are adapted to the corpus size: fixed values that
    work on hundreds of documents can silently erase the entire vocabulary
    on a handful of negative reviews (e.g. ``max_df`` as a proportion is
    trivially exceeded by every term when there's only one document).

    Args:
        texts: Text of reviews already identified as negative. Order does
            not matter to this function.
        max_keywords: Max number of ranked unigram keywords to return.
        max_phrases: Max number of ranked 2-3 word phrases to return.

    Returns:
        A NegativeTermsReport. Empty ``keywords``/``phrases`` (with a
        non-zero ``negative_review_count``) is a valid, expected result
        when there is too little text to extract a meaningful vocabulary.
    """
    negative_review_count = len(texts)
    non_empty = [t for t in texts if isinstance(t, str) and t.strip()]

    if not non_empty:
        logger.info("No non-empty negative review text available for keyword analysis.")
        return NegativeTermsReport(negative_review_count=negative_review_count)

    n_docs = len(non_empty)
    min_df = 1 if n_docs < 5 else 2
    max_df = 1.0 if n_docs < 10 else 0.9

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        min_df=min_df,
        max_df=max_df,
        max_features=500,
        lowercase=True,
        sublinear_tf=True,
        stop_words=None,
    )

    try:
        matrix = vectorizer.fit_transform(non_empty)
    except ValueError as exc:
        # e.g. every remaining "non-empty" text tokenizes to nothing
        # (pure punctuation/whitespace) — sklearn raises rather than
        # returning an empty vocabulary silently.
        logger.warning("TF-IDF vectorization produced an empty vocabulary: %s", exc)
        return NegativeTermsReport(negative_review_count=negative_review_count)

    vocabulary = vectorizer.get_feature_names_out()
    doc_frequency = np.asarray((matrix > 0).sum(axis=0)).ravel()
    corpus_score = np.asarray(matrix.sum(axis=0)).ravel()

    keyword_candidates: list[NegativeTerm] = []
    phrase_candidates: list[NegativeTerm] = []

    for term, score, df in zip(vocabulary, corpus_score, doc_frequency):
        entry = NegativeTerm(term=term, score=round(float(score), 4), document_frequency=int(df))
        if " " in term:
            phrase_candidates.append(entry)
        else:
            keyword_candidates.append(entry)

    keyword_candidates.sort(key=lambda e: e.score, reverse=True)
    phrase_candidates.sort(key=lambda e: e.score, reverse=True)

    return NegativeTermsReport(
        negative_review_count=negative_review_count,
        keywords=keyword_candidates[:max_keywords],
        phrases=phrase_candidates[:max_phrases],
    )


def analyze_negative_keywords_and_phrases(
    records: Sequence[dict[str, Any]],
    sentiment_field: str = "sentiment_transformer",
    text_field: str = "clean_review",
    max_keywords: int = 15,
    max_phrases: int = 15,
) -> NegativeTermsReport:
    """Convenience wrapper: filter records to negative reviews, then analyze.

    See ``extract_negative_texts`` and ``analyze_negative_terms`` for the
    individual steps and their parameters.
    """
    texts = extract_negative_texts(records, sentiment_field=sentiment_field, text_field=text_field)
    return analyze_negative_terms(texts, max_keywords=max_keywords, max_phrases=max_phrases)