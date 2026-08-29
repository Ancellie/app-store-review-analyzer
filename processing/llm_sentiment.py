from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal, Sequence

from dotenv import load_dotenv
from groq import Groq
from groq import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
import ollama
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger(__name__)

SentimentLabel = Literal["positive", "negative", "neutral"]

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
_GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower()

_groq_client: Groq | None = None


class SentimentAnalysisError(Exception):
    """Base class for recoverable, per-review LLM sentiment failures."""


class ProviderError(SentimentAnalysisError):
    """The provider/API call itself failed: network error, timeout, rate
    limit, or the API rejected the request (e.g. invalid/oversized text —
    this is how "invalid individual review data" surfaces in practice,
    since the provider is the one that validates the request)."""


class MalformedResponseError(SentimentAnalysisError):
    """The provider responded, but the content wasn't a usable
    {"label": ..., "score": ...} JSON payload."""


class LLMSentimentResult(BaseModel):
    label: SentimentLabel | None
    score: float = Field(ge=0.0, le=1.0, description="Confidence of the label; 0.0 when label is None")
    method: str = Field(description="Which LLM provider (and outcome) produced this result")
    error: str | None = Field(default=None, description="Failure reason; set only when label is None")


def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY", "").strip()
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is missing.")
        logger.info("Initializing Groq client...")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


def _analyze_with_groq(text: str, system_prompt: str) -> LLMSentimentResult:
    client = _get_groq_client()
    try:
        completion = client.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
    except RateLimitError as exc:
        raise ProviderError(f"Groq rate limit exceeded: {exc}") from exc
    except APITimeoutError as exc:
        raise ProviderError(f"Groq request timed out: {exc}") from exc
    except APIConnectionError as exc:
        raise ProviderError(f"Could not connect to Groq: {exc}") from exc
    except APIStatusError as exc:
        raise ProviderError(f"Groq API returned an error (status {exc.status_code}): {exc}") from exc

    response_text = completion.choices[0].message.content
    if not response_text:
        raise MalformedResponseError("Groq returned an empty response body.")
    return _parse_json_response(response_text, method="llm-groq")


def _analyze_with_ollama(text: str, system_prompt: str) -> LLMSentimentResult:
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            format="json",
            options={"temperature": 0.0}
        )
    except ollama.ResponseError as exc:
        raise ProviderError(f"Ollama API returned an error: {exc}") from exc
    except (ConnectionError, TimeoutError, OSError) as exc:
        raise ProviderError(f"Could not reach Ollama: {exc}") from exc

    try:
        response_text = response["message"]["content"]
    except (KeyError, TypeError) as exc:
        raise MalformedResponseError(f"Ollama response missing expected fields: {exc}") from exc

    if not response_text:
        raise MalformedResponseError("Ollama returned an empty response body.")
    return _parse_json_response(response_text, method="llm-ollama")


_VALID_LABELS = {"positive", "negative", "neutral"}


def _parse_json_response(response_text: str, method: str) -> LLMSentimentResult:
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise MalformedResponseError(f"Response was not valid JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise MalformedResponseError(f"Response JSON was not an object: {type(parsed).__name__}")

    raw_label = parsed.get("label")
    if not isinstance(raw_label, str) or raw_label.lower() not in _VALID_LABELS:
        raise MalformedResponseError(f"Response missing a valid 'label' field: {parsed!r}")

    try:
        score = float(parsed.get("score", 1.0))
    except (TypeError, ValueError) as exc:
        raise MalformedResponseError(f"Response 'score' was not numeric: {parsed.get('score')!r}") from exc

    return LLMSentimentResult(label=raw_label.lower(), score=score, method=method)


def analyze_sentiment(text: str) -> LLMSentimentResult:
    if not isinstance(text, str):
        raise TypeError(f"analyze_sentiment expects str, got {type(text).__name__}")

    stripped = text.strip()
    if not stripped:
        return LLMSentimentResult(label="neutral", score=0.0, method=f"llm-{_PROVIDER}")

    system_prompt = (
        "You are a strict sentiment analysis API. "
        "Analyze the user's review and determine if the sentiment is 'positive', 'negative', or 'neutral'. "
        "You must respond ONLY with a valid JSON object containing exactly two keys: "
        "'label' (the string 'positive', 'negative', or 'neutral') and "
        "'score' (a float between 0.0 and 1.0 indicating confidence). "
        "Do not include markdown blocks, explanations, or any other text."
    )

    try:
        if _PROVIDER == "ollama":
            return _analyze_with_ollama(stripped, system_prompt)
        return _analyze_with_groq(stripped, system_prompt)
    except SentimentAnalysisError as exc:
        logger.warning("LLM sentiment failed for text '%s...': %s", stripped[:20], exc)
        return LLMSentimentResult(label=None, score=0.0, method=f"llm-{_PROVIDER}-error", error=str(exc))


def attach_sentiment_llm(
        records: Sequence[dict[str, Any]],
        text_field: str = "clean_review",
        max_workers: int = 4,
) -> list[dict[str, Any]]:
    results_map: dict[int, LLMSentimentResult] = {}
    logger.info(f"Starting LLM sentiment analysis using provider: {_PROVIDER.upper()}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                analyze_sentiment,
                record.get(text_field, "") if isinstance(record.get(text_field, ""), str) else ""
            ): idx
            for idx, record in enumerate(records)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results_map[idx] = future.result()

            if _PROVIDER == "groq":
                time.sleep(0.3)

    failed_count = sum(1 for result in results_map.values() if result.label is None)
    if failed_count:
        logger.warning(
            "LLM sentiment analysis completed with %d/%d review(s) failed "
            "(see warnings above for individual reasons); those reviews carry "
            "sentiment_llm.label=None and are excluded from LLM sentiment stats.",
            failed_count,
            len(records),
        )
    else:
        logger.info("LLM sentiment analysis completed: %d/%d review(s) succeeded.", len(records), len(records))

    return [
        {**record, "sentiment_llm": results_map[i].model_dump()}
        for i, record in enumerate(records)
    ]