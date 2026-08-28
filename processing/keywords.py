from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


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


# ---------------------------------------------------------------------------
# Stopword configuration
# ---------------------------------------------------------------------------
#
# sklearn's built-in "english" stopword list removes grammatical scaffolding
# (the, to, my, you, ...), which is what turns noise like "you guys" / "to
# post" / "my fb" into meaningless n-grams in the first place: TF-IDF has no
# notion of "content word" vs "function word" on its own, so without this
# filter it happily ranks function-word phrases as highly as product terms.
#
# Two adjustments on top of the default list:
#
# 1. Negation words are put BACK IN. sklearn's default list treats "not",
#    "no", "never", "cannot", etc. as stopwords, which would turn a useful
#    complaint phrase like "cannot log in" into just "log in" — losing the
#    exact signal that makes it a complaint. Negation is domain-agnostic
#    (it matters for any app's reviews), so we always keep it.
# 2. A small, generic filler list is added for conversational words that
#    are common in review text but aren't in sklearn's classic list (it
#    predates casual internet writing). These are generic across any app
#    review corpus — not specific to any one product.
_NEGATION_WORDS: frozenset[str] = frozenset(
    {"not", "no", "never", "cannot", "none", "nothing", "neither", "nor", "without"}
)

_GENERIC_FILLER_WORDS: frozenset[str] = frozenset(
    {
        "guys", "gonna", "wanna", "yeah", "okay", "hey",
        "really", "literally", "actually", "please",
        "thanks", "thank", "lol", "omg",
    }
)

_DOMAIN_STOP_WORDS: frozenset[str] = frozenset(
    {"app", "application", "music", "song", "spotify"}
)

_STOP_WORDS: list[str] = sorted(
    (ENGLISH_STOP_WORDS - _NEGATION_WORDS) | _GENERIC_FILLER_WORDS | _DOMAIN_STOP_WORDS
)


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
    max_keywords: int = 30,
    max_phrases: int = 30,
) -> NegativeTermsReport:
    """Rank unigram keywords and multi-word phrases from negative review text.

    Uses a single TF-IDF vectorizer over unigrams through trigrams; terms
    are split into "keywords" (single word) and "phrases" (2-3 words) based
    on the same underlying corpus statistics, so scores are comparable
    across both buckets.

    Stopwords (English function words, plus a small generic conversational-
    filler list, but with negation words deliberately preserved — see
    module-level comment) are stripped from the token stream before n-grams
    are formed. This matters because TF-IDF score alone conflates three
    different things:

    - TF-IDF importance: how statistically distinctive a term is in this
      corpus relative to others. A pure grammatical phrase can score just
      as high as a product term — TF-IDF has no concept of "content word".
    - Raw frequency: how often a term appears. Frequent isn't the same as
      informative ("app" may be frequent and empty of signal; "data loss"
      may be rare and critical).
    - Meaningful product issue: requires the term to carry actual product
      vocabulary. This only emerges once stopword filtering removes the
      grammatical scaffolding that inflates statistically-valid-but-useless
      n-grams like "you guys" or "to post".

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
        stop_words=_STOP_WORDS,
    )

    try:
        matrix = vectorizer.fit_transform(non_empty)
    except ValueError as exc:
        # e.g. every remaining "non-empty" text tokenizes to nothing
        # (pure punctuation/whitespace, or entirely stopwords) — sklearn
        # raises rather than returning an empty vocabulary silently.
        logger.warning("TF-IDF vectorization produced an empty vocabulary: %s", exc)
        return NegativeTermsReport(negative_review_count=negative_review_count)

    vocabulary = vectorizer.get_feature_names_out()
    doc_frequency = np.asarray((matrix > 0).sum(axis=0)).ravel()
    corpus_score = np.asarray(matrix.sum(axis=0)).ravel()

    keyword_candidates: list[NegativeTerm] = []
    phrase_candidates: list[NegativeTerm] = []

    for term, score, df in zip(vocabulary, corpus_score, doc_frequency):
        # Guard against purely numeric n-grams (e.g. "5", "10 15") slipping
        # through — they're statistically valid but carry no product signal.
        if term.replace(" ", "").isdigit():
            continue

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