from __future__ import annotations

import json
import logging
import os
from typing import Any, Sequence
from dotenv import load_dotenv

load_dotenv()

import ollama
from groq import Groq
from groq import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
DEFAULT_OLLAMA_MODEL = "llama3.1:latest"

_SUPPORTED_PROVIDERS = ("groq", "ollama")

_MAX_NEGATIVE_REVIEWS = 50
_MAX_EVIDENCE_CHARS = 300
_MAX_RETRIES = 3
_REQUEST_TIMEOUT_SECONDS = 30.0


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class LLMInsightsError(Exception):
    """Base exception for the LLM insights layer."""


class LLMConfigError(LLMInsightsError):
    """Raised when required configuration (e.g. GROQ_API_KEY, LLM_PROVIDER) is missing/invalid."""


class LLMRequestError(LLMInsightsError):
    """Raised when the LLM provider call itself fails (network/timeout/rate limit/API error)."""


class LLMResponseError(LLMInsightsError):
    """Raised when the LLM response cannot be parsed into a valid InsightReport."""


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class Insight(BaseModel):
    """A single grounded, actionable finding derived from negative reviews."""

    problem_area: str = Field(description="Short name of the recurring product problem")
    evidence: list[str] = Field(
        min_length=1,
        max_length=5,
        description="Near-verbatim excerpts copied from the supplied reviews that support this insight",
    )
    impact: str = Field(description="Effect on users, based only on the supplied evidence")
    recommendation: str = Field(description="Concrete, actionable product/engineering recommendation")


class InsightReport(BaseModel):
    """Top-level structured output of the LLM insight-generation layer."""

    summary: str = Field(description="2-4 sentence overview of the main themes in the negative reviews")
    insights: list[Insight] = Field(default_factory=list)
    model: str
    method: str = Field(default="llm-groq")
    reviews_analyzed: int = Field(description="Number of negative reviews actually sent to the LLM")


_INSIGHT_REPORT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "insights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "problem_area": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                    "impact": {"type": "string"},
                    "recommendation": {"type": "string"},
                },
                "required": ["problem_area", "evidence", "impact", "recommendation"],
            },
        },
    },
    "required": ["summary", "insights"],
}


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are a product review analyst helping an app's engineering \
and product team understand recurring problems reported by users.

You will be given a list of negative App Store reviews (each with a rating, title, \
review text, and a sentiment confidence score) plus some aggregate counts.

Your job:
1. Identify recurring problem patterns across the reviews — group semantically \
   related complaints into a small number of distinct problem areas, even if the \
   reviews use different wording for the same underlying issue.
2. For each problem area, cite 1-5 short, near-verbatim excerpts from the SUPPLIED \
   reviews as evidence. Never cite text that is not present in the input.
3. Describe the impact on users using only what the evidence supports.
4. Propose a concrete, actionable recommendation for the product/engineering team.
5. Prioritize problem areas that appear in multiple reviews over one-off complaints, \
   unless a single review describes a severe issue (e.g. crashes, data loss, billing).

Strict rules:
- Do not invent facts, numbers, or user quotes that are not present in the supplied reviews.
- Do not report statistics (counts, percentages, averages) — those are computed separately.
- If the supplied reviews do not support a clear pattern, say so in the summary \
  and return fewer insights rather than inventing one.
- Respond with ONLY the JSON object described by the schema. No prose outside the JSON.
"""


def _truncate(text: str, limit: int = _MAX_EVIDENCE_CHARS) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _select_negative_reviews(
    records: Sequence[dict[str, Any]],
    sentiment_field: str,
    label_field: str,
    confidence_field: str,
    text_field: str,
) -> list[dict[str, Any]]:
    negatives = []
    for record in records:
        sentiment = record.get(sentiment_field)
        if not isinstance(sentiment, dict):
            continue
        if sentiment.get(label_field) != "negative":
            continue
        negatives.append(record)

    # Lowest rating first: if we have to truncate, keep the most severe complaints.
    negatives.sort(key=lambda r: r.get("rating", 15))

    truncated = negatives[:_MAX_NEGATIVE_REVIEWS]
    if len(negatives) > _MAX_NEGATIVE_REVIEWS:
        logger.info(
            "Found %d negative reviews; capping LLM input at %d (lowest-rated first).",
            len(negatives),
            _MAX_NEGATIVE_REVIEWS,
        )
    return truncated


def _build_user_payload(
    negatives: Sequence[dict[str, Any]],
    total_reviews: int,
    sentiment_field: str,
    label_field: str,
    confidence_field: str,
    text_field: str,
) -> dict[str, Any]:
    rating_breakdown: dict[int, int] = {}
    for record in negatives:
        rating = record.get("rating")
        if isinstance(rating, int):
            rating_breakdown[rating] = rating_breakdown.get(rating, 0) + 1

    items = []
    for record in negatives:
        sentiment = record.get(sentiment_field, {})
        items.append(
            {
                "rating": record.get("rating"),
                "title": _truncate(record.get("title", ""), 120),
                "review": _truncate(record.get(text_field, "")),
                "sentiment_confidence": sentiment.get(confidence_field),
            }
        )

    return {
        "aggregate_counts": {
            "total_reviews": total_reviews,
            "negative_reviews_total": len(negatives),
            "negative_reviews_sent_to_llm": len(items),
            "negative_rating_breakdown": rating_breakdown,
        },
        "negative_reviews": items,
    }


# ---------------------------------------------------------------------------
# Provider selection / model resolution
# ---------------------------------------------------------------------------

def _resolve_provider() -> str:
    """Read and validate LLM_PROVIDER from the environment.

    Defaults to "groq" for backward compatibility with existing deployments
    that don't set LLM_PROVIDER at all.
    """
    provider = os.environ.get("LLM_PROVIDER", "groq").strip().lower()
    if provider not in _SUPPORTED_PROVIDERS:
        raise LLMConfigError(
            f"Unsupported LLM_PROVIDER '{provider}'. Supported providers: "
            f"{', '.join(_SUPPORTED_PROVIDERS)}."
        )
    return provider


def _resolve_model(model: str | None, provider: str) -> str:
    """Resolve which model id to use for *provider*.

    Precedence: explicit `model` argument > provider-specific env var >
    provider-specific hardcoded default.
    """
    if model:
        return model

    if provider == "groq":
        return os.environ.get("GROQ_MODEL") or DEFAULT_GROQ_MODEL
    if provider == "ollama":
        return os.environ.get("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL

    # Unreachable in practice — _resolve_provider already validates this —
    # but kept explicit rather than falling through silently.
    raise LLMConfigError(f"Unsupported LLM_PROVIDER '{provider}'.")


# ---------------------------------------------------------------------------
# Groq client + request
# ---------------------------------------------------------------------------

def _build_groq_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise LLMConfigError(
            "GROQ_API_KEY is not set. Set it in your environment or .env file."
        )
    return Groq(api_key=api_key, timeout=_REQUEST_TIMEOUT_SECONDS)


def _call_groq(client: Groq, model: str, payload: dict[str, Any]) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "insight_report",
                    "strict": False,  # gpt-oss-120b has had regressions with strict mode;
                    "schema": _INSIGHT_REPORT_JSON_SCHEMA,  # we validate manually below instead.
                },
            },
            temperature=0.2,
        )
    except RateLimitError as exc:
        raise LLMRequestError("Groq API rate limit exceeded.") from exc
    except APITimeoutError as exc:
        raise LLMRequestError("Groq API request timed out.") from exc
    except APIConnectionError as exc:
        raise LLMRequestError("Could not connect to the Groq API.") from exc
    except APIStatusError as exc:
        raise LLMRequestError(f"Groq API returned an error (status {exc.status_code}).") from exc

    content = response.choices[0].message.content
    if not content:
        raise LLMResponseError("Groq API returned an empty response.")
    return content


# ---------------------------------------------------------------------------
# Ollama request
# ---------------------------------------------------------------------------

def _call_ollama(model: str, payload: dict[str, Any]) -> str:
    """Call a local/self-hosted Ollama server.

    The `ollama` package reads the server address from the OLLAMA_HOST
    environment variable itself (e.g. "http://ollama:11434" in Docker
    Compose), so no explicit client/host wiring is needed here.
    """
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            format=_INSIGHT_REPORT_JSON_SCHEMA,
            options={"temperature": 0.2},
        )
    except ConnectionError as exc:
        host = os.environ.get("OLLAMA_HOST", "default host")
        raise LLMRequestError(f"Could not connect to Ollama at {host}: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - ollama surfaces its own exception types
        raise LLMRequestError(f"Ollama API returned an error: {exc}") from exc

    content = response["message"]["content"] if isinstance(response, dict) else response.message.content
    if not content:
        raise LLMResponseError("Ollama API returned an empty response.")
    return content


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _call_llm(provider: str, groq_client: Groq | None, model: str, payload: dict[str, Any]) -> str:
    """Single dispatch point so `generate_insight_report` stays provider-agnostic."""
    if provider == "groq":
        assert groq_client is not None  # built by the caller whenever provider == "groq"
        return _call_groq(groq_client, model, payload)
    if provider == "ollama":
        return _call_ollama(model, payload)

    raise LLMConfigError(f"Unsupported LLM_PROVIDER '{provider}'.")


def _parse_and_validate(raw_content: str, model: str, method: str, reviews_analyzed: int) -> InsightReport:
    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as exc:
        raise LLMResponseError(f"LLM response was not valid JSON: {exc}") from exc

    data["model"] = model
    data["method"] = method
    data["reviews_analyzed"] = reviews_analyzed

    try:
        return InsightReport.model_validate(data)
    except ValidationError as exc:
        raise LLMResponseError(f"LLM response did not match the expected schema: {exc}") from exc


def _empty_report(model: str, method: str, reason: str) -> InsightReport:
    return InsightReport(
        summary=reason,
        insights=[],
        model=model,
        method=method,
        reviews_analyzed=0,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_insight_report(
    records: Sequence[dict[str, Any]],
    *,
    model: str | None = None,
    sentiment_field: str = "sentiment_transformer",
    label_field: str = "label",
    confidence_field: str = "score",
    text_field: str = "clean_review",
) -> InsightReport:
    """Generate an LLM-based InsightReport from processed, sentiment-tagged reviews.

    The LLM backend is selected via the LLM_PROVIDER environment variable
    ("groq" or "ollama", defaulting to "groq"). The rest of the pipeline
    (review selection, prompt, schema, validation, retries) is identical
    regardless of provider.

    Args:
        records: Review records already enriched with a sentiment field, as
            produced by ``transformer_sentiment.attach_sentiment_transformer``
            (default) or ``sentiment.attach_sentiment`` (pass
            ``sentiment_field="sentiment"``, ``confidence_field="compound"``).
        model: Explicit model id, overriding provider defaults/env vars.
        sentiment_field / label_field / confidence_field / text_field: Let
            callers point this at either sentiment layer without code changes.

    Returns:
        A validated InsightReport. Returns an empty-but-valid report (no LLM
        call made) if there are no records or no negative reviews.

    Raises:
        LLMConfigError: LLM_PROVIDER is invalid, or required provider
            configuration (e.g. GROQ_API_KEY) is missing.
        LLMRequestError: The provider API call failed after retries.
        LLMResponseError: The LLM never returned a schema-valid response.
    """
    provider = _resolve_provider()
    resolved_model = _resolve_model(model, provider)
    method = f"llm-{provider}"

    if not records:
        logger.info("generate_insight_report called with no records; skipping LLM call.")
        return _empty_report(resolved_model, method, "No reviews were supplied.")

    negatives = _select_negative_reviews(
        records, sentiment_field, label_field, confidence_field, text_field
    )
    if not negatives:
        logger.info("No negative reviews found; skipping LLM call.")
        return _empty_report(resolved_model, method, "No negative reviews were found in this dataset.")

    payload = _build_user_payload(
        negatives, len(records), sentiment_field, label_field, confidence_field, text_field
    )

    groq_client = _build_groq_client() if provider == "groq" else None

    last_error: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        raw_content = _call_llm(provider, groq_client, resolved_model, payload)
        try:
            return _parse_and_validate(raw_content, resolved_model, method, len(negatives))
        except LLMResponseError as exc:
            last_error = exc
            logger.warning("LLM response invalid on attempt %d/%d: %s", attempt, _MAX_RETRIES, exc)

    raise LLMResponseError(
        f"LLM did not return a schema-valid response after {_MAX_RETRIES} attempts."
    ) from last_error