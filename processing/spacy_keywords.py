from __future__ import annotations

import logging
from typing import Any, Sequence

import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token

from processing.contrastive import expand_pool_size, rerank_report_contrastively
from processing.keywords import (
    NegativeTerm,
    NegativeTermsReport,
    extract_negative_texts,
    extract_positive_texts,
    _DOMAIN_STOP_WORDS,
)

logger = logging.getLogger(__name__)

_SPACY_MODEL_NAME = "en_core_web_sm"
_METHOD_TAG = "spacy-pos"

# Two-token POS sequences that tend to yield meaningful product-complaint
# phrases ("slow loading", "app crashes", "login problem", "constant ads").
# Both tokens must also pass `_is_meaningful_token` before a pattern counts.
_TWO_TOKEN_PATTERNS: frozenset[tuple[str, str]] = frozenset(
    {
        ("ADJ", "NOUN"),
        ("NOUN", "NOUN"),
        ("VERB", "NOUN"),
        ("NOUN", "VERB"),
    }
)

# Lazily loaded, module-level, reused across all calls in this process —
# mirrors the pattern already used for the sentiment transformer pipeline.
_nlp: Language | None = None


def _get_nlp() -> Language:
    """Return the shared spaCy pipeline, loading it on first use.

    NOTE (language limitation): this loads the English model only. spaCy
    has no single small model covering English + Ukrainian + Russian POS
    tagging, and wiring three separate per-language pipelines was out of
    scope for this step. Non-English (Ukrainian/Russian) review text will
    still be tagged without error, but the POS tags are not reliable for
    those languages — this is a documented baseline limitation, exactly
    like VADER's English-only sentiment scoring.

    `ner` and `parser` are disabled: neither named entities nor dependency
    parses are used here, only POS tags and lemmas, so skipping them
    reduces per-review inference cost.
    """
    global _nlp
    if _nlp is None:
        logger.info("Loading spaCy model '%s' (first use)...", _SPACY_MODEL_NAME)
        try:
            _nlp = spacy.load(_SPACY_MODEL_NAME, disable=["ner", "parser"])
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{_SPACY_MODEL_NAME}' is not installed. "
                f"Run: python -m spacy download {_SPACY_MODEL_NAME}"
            ) from exc
    return _nlp


def _is_meaningful_token(token: Token) -> bool:
    """Filter out grammatical scaffolding, punctuation, and numerals."""

    if token.lemma_.lower() in _DOMAIN_STOP_WORDS:
        return False

    if token.is_stop or token.is_punct or token.is_space or token.like_num:
        return False
    if not token.is_alpha:
        return False
    if len(token.text) < 2:
        return False
    return True


def _extract_doc_terms(doc: Doc) -> set[tuple[str, str]]:
    """Return the set of (kind, normalized_term) pairs found in *doc*.

    A set, not a list: within a single review we only care whether a term
    appears at all, since document_frequency is computed by counting how
    many distinct reviews mention it — repeated mentions in one review
    should not inflate that count.

    Single meaningful NOUN/ADJ tokens become "keyword" candidates (single
    words); two-token POS-pattern matches become "phrase" candidates.
    Lemmatization normalizes inflected forms ("crashes"/"crashed" ->
    "crash") so the same underlying complaint isn't split into several
    near-duplicate terms.
    """
    terms: set[tuple[str, str]] = set()
    tokens = list(doc)

    for tok in tokens:
        if tok.pos_ in ("NOUN", "ADJ") and _is_meaningful_token(tok):
            terms.add(("keyword", tok.lemma_.lower()))

    for i in range(len(tokens) - 1):
        first, second = tokens[i], tokens[i + 1]
        if not (_is_meaningful_token(first) and _is_meaningful_token(second)):
            continue
        if (first.pos_, second.pos_) in _TWO_TOKEN_PATTERNS:
            phrase = f"{first.lemma_.lower()} {second.lemma_.lower()}"
            terms.add(("phrase", phrase))

    return terms


def analyze_negative_terms_spacy(
    texts: Sequence[str],
    max_keywords: int = 15,
    max_phrases: int = 15,
    batch_size: int = 64,
) -> NegativeTermsReport:
    """Rank POS-pattern keywords/phrases from negative review text.

    Unlike TF-IDF's statistical weighting, ``score`` here is the fraction
    of negative reviews that mention the (lemmatized) term at least once —
    an interpretable "how common is this complaint" figure, since POS
    extraction has no natural TF-IDF-style weighting of its own.

    Note: this ranks by prevalence *within the negative corpus only*. See
    ``processing.contrastive.rerank_report_contrastively`` for the second
    pass that compares against positive reviews, applied by the
    convenience wrapper below.

    Args:
        texts: Text of reviews already identified as negative.
        max_keywords: Max number of ranked single-word keywords to return.
        max_phrases: Max number of ranked two-word phrases to return.
        batch_size: Forwarded to ``nlp.pipe`` for batched processing.

    Returns:
        A NegativeTermsReport with ``method="spacy-pos"``. Empty
        ``keywords``/``phrases`` is a valid result when there's too little
        text to find a meaningful pattern.
    """
    negative_review_count = len(texts)
    non_empty = [t for t in texts if isinstance(t, str) and t.strip()]

    if not non_empty:
        logger.info("No non-empty negative review text available for spaCy extraction.")
        return NegativeTermsReport(negative_review_count=negative_review_count, method=_METHOD_TAG)

    nlp = _get_nlp()

    keyword_doc_freq: dict[str, int] = {}
    phrase_doc_freq: dict[str, int] = {}

    for doc in nlp.pipe(non_empty, batch_size=batch_size):
        for kind, term in _extract_doc_terms(doc):
            bucket = keyword_doc_freq if kind == "keyword" else phrase_doc_freq
            bucket[term] = bucket.get(term, 0) + 1

    n_docs = len(non_empty)

    def _to_terms(freqs: dict[str, int], limit: int) -> list[NegativeTerm]:
        ranked = sorted(freqs.items(), key=lambda kv: kv[1], reverse=True)[:limit]
        return [
            NegativeTerm(term=term, score=round(df / n_docs, 4), document_frequency=df)
            for term, df in ranked
        ]

    return NegativeTermsReport(
        negative_review_count=negative_review_count,
        keywords=_to_terms(keyword_doc_freq, max_keywords),
        phrases=_to_terms(phrase_doc_freq, max_phrases),
        method=_METHOD_TAG,
    )


def analyze_negative_keywords_and_phrases_spacy(
    records: Sequence[dict[str, Any]],
    sentiment_field: str = "sentiment_transformer",
    text_field: str = "clean_review",
    max_keywords: int = 15,
    max_phrases: int = 15,
) -> NegativeTermsReport:
    """Filter records to negative reviews, extract POS-pattern candidates
    over a wider pool than requested, then rank them by negative
    specificity relative to positive reviews.

    Mirrors ``keywords.analyze_negative_keywords_and_phrases``'s contract.
    """
    negative_texts = extract_negative_texts(records, sentiment_field=sentiment_field, text_field=text_field)
    positive_texts = extract_positive_texts(records, sentiment_field=sentiment_field, text_field=text_field)

    raw_report = analyze_negative_terms_spacy(
        negative_texts,
        max_keywords=expand_pool_size(max_keywords),
        max_phrases=expand_pool_size(max_phrases),
    )

    return rerank_report_contrastively(
        raw_report,
        negative_texts=negative_texts,
        positive_texts=positive_texts,
        max_keywords=max_keywords,
        max_phrases=max_phrases,
    )