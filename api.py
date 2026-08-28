from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Імпорти з ваших модулів
from collector.fetchlayer_client import FetchLayerReviewClient
from processing.loader import load_reviews
from processing.metrics import compute_metrics
from processing.keywords import analyze_negative_keywords_and_phrases
from processing.spacy_keywords import analyze_negative_keywords_and_phrases_spacy
from processing.keybert_keywords import analyze_negative_keywords_and_phrases_keybert

# Імпорти всіх 4 методів видобування ключових слів
from processing.sentiment import attach_sentiment
from processing.transformer_sentiment import attach_sentiment_transformer
from processing.llm_sentiment import attach_sentiment_llm
from processing.llm_insights import generate_insight_report

# Налаштування шляхів
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_REVIEWS_PATH = PROJECT_ROOT / "review.json"
RESULTS_DIR = PROJECT_ROOT / "results"

# Ініціалізація FastAPI
app = FastAPI(
    title="App Store Review Analysis API",
    description="API for collecting and analyzing App Store reviews.",
    version="1.0.0"
)


# --- Допоміжні функції ---

def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def _label_distribution(records: list[dict[str, Any]], sentiment_field: str) -> dict[str, int]:
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for record in records:
        sentiment = record.get(sentiment_field)
        if isinstance(sentiment, dict) and sentiment.get("label") in counts:
            counts[sentiment["label"]] += 1
    return counts


def _build_sentiment_export(records: list[dict[str, Any]], sentiment_field: str) -> list[dict[str, Any]]:
    return [
        {
            "index": idx,
            "rating": record.get("rating"),
            "title": record.get("title"),
            "review": record.get("review"),
            "clean_review": record.get("clean_review"),
            sentiment_field: record.get(sentiment_field),
        }
        for idx, record in enumerate(records)
    ]


def run_pipeline(app_id: str, country: str, limit: int) -> None:
    """Фонова задача для повного циклу збору та обробки."""
    try:
        logging.info(f"Starting pipeline for app_id={app_id}")

        # 1. Collection
        client = FetchLayerReviewClient()
        reviews = client.get_reviews(app_id=app_id, limit=limit, country=country)
        if not reviews:
            logging.error("Collector returned zero reviews.")
            return

        raw_records = [r.model_dump(mode="json") for r in reviews]
        save_json(RAW_REVIEWS_PATH, raw_records)

        # 2. Loading & Cleaning
        records = load_reviews(RAW_REVIEWS_PATH)
        if not records:
            logging.error("No valid reviews remained after validation/cleaning.")
            return

        # 3. Processing
        metrics = compute_metrics(records)
        records = attach_sentiment(records)
        records = attach_sentiment_transformer(records)
        records = attach_sentiment_llm(records)

        # Запуск усіх трьох методів видобування
        keyword_reports = {
            "tfidf": analyze_negative_keywords_and_phrases(records),
            "spacy": analyze_negative_keywords_and_phrases_spacy(records),
            "keybert": analyze_negative_keywords_and_phrases_keybert(records),
        }

        insight_report = generate_insight_report(records)

        # 4. Saving Results
        RESULTS_DIR.mkdir(exist_ok=True)
        save_json(RESULTS_DIR / "metrics.json", metrics)
        save_json(RESULTS_DIR / "sentiment_vader.json", _build_sentiment_export(records, "sentiment"))
        save_json(RESULTS_DIR / "sentiment_transformer.json", _build_sentiment_export(records, "sentiment_transformer"))
        save_json(RESULTS_DIR / "sentiment_llm.json", _build_sentiment_export(records, "sentiment_llm"))

        # Збереження звітів ключових слів в окремі файли
        for name, report in keyword_reports.items():
            save_json(RESULTS_DIR / f"negative_keywords_{name}.json", report.model_dump())

        save_json(RESULTS_DIR / "insights.json", insight_report.model_dump())

        # Формування загального звіту
        analysis = {
            "metrics": metrics,
            "sentiment_distribution": {
                "vader": _label_distribution(records, "sentiment"),
                "transformer": _label_distribution(records, "sentiment_transformer"),
                "llm": _label_distribution(records, "sentiment_llm"),
            },
            "top_negative_keywords": {
                name: [k.model_dump() for k in report.keywords[:5]]
                for name, report in keyword_reports.items()
            },
            "top_negative_phrases": {
                name: [p.model_dump() for p in report.phrases[:5]]
                for name, report in keyword_reports.items()
            },
            "insights_summary": {
                "summary": insight_report.summary,
                "problem_areas": [i.problem_area for i in insight_report.insights],
                "reviews_analyzed": insight_report.reviews_analyzed,
                "model": insight_report.model,
            },
        }
        save_json(RESULTS_DIR / "analysis.json", analysis)
        logging.info("Pipeline completed successfully.")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")


# --- API Models ---

class CollectRequest(BaseModel):
    country: str = "us"
    limit: int = 100


# --- Endpoints ---

@app.post("/api/reviews/{app_id}/collect", summary="Collect reviews and trigger analysis")
def collect_reviews(app_id: str, req: CollectRequest, background_tasks: BackgroundTasks):
    """
    Triggers the review collection and analysis pipeline in the background.
    """
    background_tasks.add_task(run_pipeline, app_id, req.country, req.limit)

    return {
        "status": "processing",
        "message": f"Collection and analysis started for app_id {app_id} in background.",
        "details": f"Country: {req.country}, Limit: {req.limit}"
    }


@app.get("/api/analysis", summary="Get metrics and insights")
def get_analysis():
    """
    Returns the calculated metrics and aggregated insights from the most recent run.
    """
    analysis_file = RESULTS_DIR / "analysis.json"

    if not analysis_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Analysis not found. Please run the collection endpoint first and wait for it to finish."
        )

    with open(analysis_file, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/keywords/{method}", summary="Get negative keywords by extraction method")
def get_keywords_by_method(method: str):
    """
    Returns the full keyword extraction report for a specific method.
    Valid methods: 'tfidf', 'spacy', 'keybert'.
    """
    if method not in ["tfidf", "spacy", "keybert"]:
        raise HTTPException(status_code=400, detail="Invalid method. Choose from: tfidf, spacy, keybert.")

    keywords_file = RESULTS_DIR / f"negative_keywords_{method}.json"

    if not keywords_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Keyword results for '{method}' not found. Please run the collection endpoint first."
        )

    with open(keywords_file, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/reviews/download", summary="Download raw review data")
def download_raw_reviews():
    """
    Provides the raw review.json file for download.
    """
    if not RAW_REVIEWS_PATH.exists():
        raise HTTPException(status_code=404, detail="Raw reviews not found. Have you collected them yet?")

    return FileResponse(
        path=RAW_REVIEWS_PATH,
        filename="review.json",
        media_type="application/json"
    )