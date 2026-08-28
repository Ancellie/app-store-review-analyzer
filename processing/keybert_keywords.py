from __future__ import annotations

import logging
from typing import Any, Sequence

from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

from processing.keywords import NegativeTerm, NegativeTermsReport, extract_negative_texts, _STOP_WORDS

logger = logging.getLogger(__name__)

# Small multilingual sentence-embedding model. Chosen over an English-only
# model because reviews may be English/Ukrainian/Russian, and over the
# larger paraphrase-multilingual-mpnet-base-v2 for CPU/memory friendliness
# at 1,000+ review scale. Russian is in its officially fine-tuned language
# set; Ukrainian is not, but shares enough script/subword overlap to
# degrade gracefully rather than fail outright — documented limitation,
# same spirit as the spaCy English-only baseline.
_EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_METHOD_TAG = "keybert"

_kw_model: KeyBERT | None = None


def _get_model() -> KeyBERT:
    """Return the shared KeyBERT model, loading it on first use.

    The underlying sentence-transformer weights are downloaded from the
    Hugging Face hub automatically on first use and cached locally after
    that — no API key required, fully local/deterministic inference.
    """
    global _kw_model
    if _kw_model is None:
        logger.info("Loading KeyBERT embedding model '%s' (first use)...", _EMBEDDING_MODEL_NAME)
        sentence_model = SentenceTransformer(_EMBEDDING_MODEL_NAME)
        _kw_model = KeyBERT(model=sentence_model)
    return _kw_model


def analyze_negative_terms_keybert(
    texts: Sequence[str],
    max_keywords: int = 15,
    max_phrases: int = 15,
    top_n_per_review: int = 5,
    diversity: float = 0.5,
) -> NegativeTermsReport:
    """Rank semantically representative keywords/phrases from negative reviews.

    Extraction runs on each negative review individually, in a *single*
    batched call (``extract_keywords(docs=[...])``), rather than the whole
    corpus concatenated into one document. This preserves the ability to
    count how many distinct reviews raise a given phrase (document
    frequency), which is what "common" means for this requirement — a
    single-document-corpus run would instead surface whatever dominates
    the concatenated text, which can just be one long review.

    ``score`` is the mean cosine similarity of the term to the review(s) it
    was drawn from; ranking prioritizes ``document_frequency`` (how many
    separate reviews raise it) over raw similarity, so a phrase repeated
    across many reviews outranks one review's unusually strong phrase.

    Args:
        texts: Text of reviews already identified as negative.
        max_keywords: Max number of ranked single-word keywords to return.
        max_phrases: Max number of ranked multi-word phrases to return.
        top_n_per_review: How many candidate phrases KeyBERT keeps per
            review before aggregation.
        diversity: MMR diversity parameter (0-1); keeps a single review
            from contributing several near-duplicate phrase variants.

    Returns:
        A NegativeTermsReport with ``method="keybert"``.
    """
    negative_review_count = len(texts)
    non_empty = [t for t in texts if isinstance(t, str) and t.strip()]

    if not non_empty:
        logger.info("No non-empty negative review text available for KeyBERT extraction.")
        return NegativeTermsReport(negative_review_count=negative_review_count, method=_METHOD_TAG)

    model = _get_model()

    try:
        per_doc_results = model.extract_keywords(
            docs=non_empty,
            keyphrase_ngram_range=(1, 3),
            stop_words=_STOP_WORDS,
            use_mmr=True,
            diversity=diversity,
            top_n=top_n_per_review,
        )
    except ValueError as exc:
        # e.g. every review tokenizes to nothing after stopword removal.
        logger.warning("KeyBERT extraction produced no candidates: %s", exc)
        return NegativeTermsReport(negative_review_count=negative_review_count, method=_METHOD_TAG)

    # extract_keywords(docs=[...]) returns list[list[(term, score)]] — but
    # defensively normalize in case a single-document input collapses to a
    # flat list[(term, score)] depending on KeyBERT version behavior.
    if per_doc_results and isinstance(per_doc_results[0], tuple):
        per_doc_results = [per_doc_results]

    keyword_stats: dict[str, list[float]] = {}
    phrase_stats: dict[str, list[float]] = {}

    for doc_keywords in per_doc_results:
        for term, score in doc_keywords:
            term_norm = term.lower().strip()
            if not term_norm:
                continue
            bucket = phrase_stats if " " in term_norm else keyword_stats
            bucket.setdefault(term_norm, []).append(float(score))

    def _to_terms(stats: dict[str, list[float]], limit: int) -> list[NegativeTerm]:
        aggregated = [
            (term, sum(scores) / len(scores), len(scores)) for term, scores in stats.items()
        ]
        aggregated.sort(key=lambda item: (item[2], item[1]), reverse=True)
        return [
            NegativeTerm(term=term, score=round(mean_score, 4), document_frequency=df)
            for term, mean_score, df in aggregated[:limit]
        ]

    return NegativeTermsReport(
        negative_review_count=negative_review_count,
        keywords=_to_terms(keyword_stats, max_keywords),
        phrases=_to_terms(phrase_stats, max_phrases),
        method=_METHOD_TAG,
    )


def analyze_negative_keywords_and_phrases_keybert(
    records: Sequence[dict[str, Any]],
    sentiment_field: str = "sentiment_transformer",
    text_field: str = "clean_review",
    max_keywords: int = 15,
    max_phrases: int = 15,
) -> NegativeTermsReport:
    """Convenience wrapper: filter records to negative reviews, then analyze."""
    texts = extract_negative_texts(records, sentiment_field=sentiment_field, text_field=text_field)
    return analyze_negative_terms_keybert(texts, max_keywords=max_keywords, max_phrases=max_phrases)