from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal, Sequence

from dotenv import load_dotenv
from groq import Groq
import ollama
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger(__name__)

SentimentLabel = Literal["positive", "negative", "neutral"]

_GROQ_MODEL = "openai/gpt-oss-20b"
_OLLAMA_MODEL = "llama3.1"

_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()

_groq_client: Groq | None = None


class LLMSentimentResult(BaseModel):
    label: SentimentLabel
    score: float = Field(ge=0.0, le=1.0, description="Confidence of the label")
    method: str = Field(description="Which LLM provider was used")


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
        response_text = completion.choices[0].message.content
        return _parse_json_response(response_text, method="llm-groq")
    except Exception as e:
        logger.warning(f"Groq API failed for text '{text[:20]}...': {e}")
        return LLMSentimentResult(label="neutral", score=0.0, method="llm-groq-error")


def _analyze_with_ollama(text: str, system_prompt: str) -> LLMSentimentResult:
    try:
        response = ollama.chat(
            model=_OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            format="json",
            options={"temperature": 0.0}
        )
        response_text = response['message']['content']
        return _parse_json_response(response_text, method="llm-ollama")
    except Exception as e:
        logger.warning(f"Ollama failed for text '{text[:20]}...': {e}")
        return LLMSentimentResult(label="neutral", score=0.0, method="llm-ollama-error")


def _parse_json_response(response_text: str, method: str) -> LLMSentimentResult:
    try:
        parsed = json.loads(response_text)
        label = parsed.get("label", "neutral").lower()
        if label not in ("positive", "negative", "neutral"):
            label = "neutral"
        score = float(parsed.get("score", 1.0))
        return LLMSentimentResult(label=label, score=score, method=method)
    except json.JSONDecodeError:
        return LLMSentimentResult(label="neutral", score=0.0, method=f"{method}-parse-error")


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

    if _PROVIDER == "ollama":
        return _analyze_with_ollama(stripped, system_prompt)
    else:
        return _analyze_with_groq(stripped, system_prompt)


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

    return [
        {**record, "sentiment_llm": results_map[i].model_dump()}
        for i, record in enumerate(records)
    ]