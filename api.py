from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi.templating import Jinja2Templates
from fastapi import Request
templates = Jinja2Templates(directory="templates")

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel, Field
from fastapi import BackgroundTasks, Path as FastAPIPath

from processing.visualization import (
    render_rating_distribution,
    render_sentiment_distribution,
    render_sentiment_by_rating,
    render_top_negative_terms,
)

from collector.fetchlayer_client import FetchLayerReviewClient
from processing.loader import load_reviews
from processing.metrics import compute_metrics
from processing.keywords import analyze_negative_keywords_and_phrases
from processing.spacy_keywords import analyze_negative_keywords_and_phrases_spacy
from processing.keybert_keywords import analyze_negative_keywords_and_phrases_keybert

from processing.sentiment import attach_sentiment
from processing.transformer_sentiment import attach_sentiment_transformer
from processing.llm_sentiment import attach_sentiment_llm
from processing.llm_insights import generate_insight_report
from processing.results import save_json, save_pipeline_results

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_REVIEWS_PATH = PROJECT_ROOT / "review.json"
RESULTS_DIR = PROJECT_ROOT / "results"

app = FastAPI(
    title="App Store Review Analysis API",
    description="API for collecting and analyzing App Store reviews.",
    version="1.0.0"
)



def _process_and_save_reviews(file_path: Path, results_dir: Path) -> None:

    records = load_reviews(file_path)
    if not records:
        logging.error("No valid reviews remained after validation/cleaning.")
        return

    metrics = compute_metrics(records)
    records = attach_sentiment(records)
    records = attach_sentiment_transformer(records)
    records = attach_sentiment_llm(records)

    keyword_reports = {
        "tfidf": analyze_negative_keywords_and_phrases(records),
        "spacy": analyze_negative_keywords_and_phrases_spacy(records),
        "keybert": analyze_negative_keywords_and_phrases_keybert(records),
    }

    insight_report = generate_insight_report(records)

    save_pipeline_results(results_dir, metrics, records, keyword_reports, insight_report)


def run_pipeline(app_id: str, country: str, limit: int) -> None:
    try:
        logging.info(f"Starting pipeline for app_id={app_id}")

        client = FetchLayerReviewClient()
        reviews = client.get_reviews(app_id=app_id, limit=limit, country=country)
        if not reviews:
            logging.error("Collector returned zero reviews.")
            return

        raw_records = [r.model_dump(mode="json") for r in reviews]
        save_json(RAW_REVIEWS_PATH, raw_records)

        _process_and_save_reviews(RAW_REVIEWS_PATH, RESULTS_DIR)

        logging.info("Pipeline completed successfully.")

    except Exception as e:
        logging.exception(f"Pipeline failed for app_id={app_id}")


def run_pipeline_from_file() -> None:
    try:
        logging.info(f"Starting analysis from {RAW_REVIEWS_PATH}")

        if not RAW_REVIEWS_PATH.exists():
            logging.error(f"File {RAW_REVIEWS_PATH} not found.")
            return

        _process_and_save_reviews(RAW_REVIEWS_PATH, RESULTS_DIR)

        logging.info("Pipeline from existing reviews completed successfully.")

    except Exception:
        logging.exception("Pipeline from file failed.")


# --- API Models ---

class CollectRequest(BaseModel):
    country: str = Field(default="us", min_length=2, max_length=2)

    limit: int = Field(default=100, gt=0, description="Limit must be a positive integer")


# --- Endpoints ---

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/reviews/{app_id}/collect", summary="Collect reviews and trigger analysis")
def collect_reviews(
        app_id: str = FastAPIPath(..., pattern=r"^\d+$"),
        req: CollectRequest = None,
        background_tasks: BackgroundTasks = None
):
    """
    Triggers the review collection and analysis pipeline in the background.
    """
    background_tasks.add_task(run_pipeline, app_id, req.country, req.limit)

    return {
        "status": "processing",
        "message": f"Collection and analysis started for app_id {app_id} in background.",
        "details": f"Country: {req.country}, Limit: {req.limit}"
    }

@app.post(
    "/api/reviews/{app_id}/fetch",
    summary="Fetch and save reviews without processing",
)
def fetch_reviews_only(
    app_id: str = FastAPIPath(..., pattern=r"^\d+$"),
    req: CollectRequest = None,
):
    """
    Fetch reviews from FetchLayer and save them to review.json
    without running any analysis.
    """
    try:
        client = FetchLayerReviewClient()

        reviews = client.get_reviews(
            app_id=app_id,
            limit=req.limit,
            country=req.country,
        )

        review_records = [
            review.model_dump(mode="json")
            for review in reviews
        ]

        if not review_records:
            return {
                "status": "success",
                "app_id": app_id,
                "country": req.country,
                "count": 0,
                "reviews": [],
            }

        save_json(RAW_REVIEWS_PATH, review_records)

        return {
            "status": "success",
            "app_id": app_id,
            "country": req.country,
            "count": len(review_records),
            "reviews": review_records,
            "saved_to": str(RAW_REVIEWS_PATH),
        }

    except Exception as e:
        logging.exception(
            f"Failed to fetch reviews for app_id={app_id}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch reviews: {str(e)}",
        )

@app.post("/api/reviews/analyze")
def analyze_existing_reviews(background_tasks: BackgroundTasks):
    """
    Analyze already collected reviews from review.json.
    """
    if not RAW_REVIEWS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="review.json not found. Collect reviews first."
        )

    background_tasks.add_task(run_pipeline_from_file)

    return {
        "status": "processing",
        "message": "Analysis of existing reviews started in background."
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

@app.get(
    "/api/visualizations/rating-distribution",
    summary="Rating distribution chart",
)
def rating_distribution_chart():
    metrics_file = RESULTS_DIR / "metrics.json"

    if not metrics_file.exists():
        raise HTTPException(status_code=404, detail="Metrics not found.")

    with open(metrics_file, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    image = render_rating_distribution(metrics)

    return Response(
        content=image,
        media_type="image/png",
    )


@app.get(
    "/api/visualizations/sentiment-distribution",
    summary="Sentiment distribution chart",
)
def sentiment_distribution_chart():
    analysis_file = RESULTS_DIR / "analysis.json"

    if not analysis_file.exists():
        raise HTTPException(status_code=404, detail="Analysis not found.")

    with open(analysis_file, "r", encoding="utf-8") as f:
        analysis = json.load(f)

    sentiment_distribution = analysis.get("sentiment_distribution", {})

    image = render_sentiment_distribution(sentiment_distribution)

    return Response(
        content=image,
        media_type="image/png",
    )


@app.get(
    "/api/visualizations/sentiment-by-rating",
    summary="Sentiment by rating chart",
)
def sentiment_by_rating_chart():
    sentiment_file = RESULTS_DIR / "sentiment_transformer.json"

    if not sentiment_file.exists():
        raise HTTPException(
            status_code=404,
            detail="Transformer sentiment results not found.",
        )

    with open(sentiment_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    image = render_sentiment_by_rating(
        records,
        sentiment_field="sentiment_transformer",
    )

    return Response(
        content=image,
        media_type="image/png",
    )


@app.get(
    "/api/visualizations/top-negative-terms",
    summary="Top negative keywords and phrases chart",
)
def top_negative_terms_chart(
    method: str = "tfidf",
    kind: str = "keywords",
):
    if method not in {"tfidf", "keybert", "spacy-pos"}:
        raise HTTPException(
            status_code=400,
            detail="method must be one of: tfidf, keybert, spacy-pos.",
        )

    if kind not in {"keywords", "phrases"}:
        raise HTTPException(
            status_code=400,
            detail="kind must be either 'keywords' or 'phrases'.",
        )

    files = {
        "tfidf": RESULTS_DIR / "negative_keywords_tfidf.json",
        "keybert": RESULTS_DIR / "negative_keywords_keybert.json",
        "spacy-pos": RESULTS_DIR / "negative_keywords_spacy.json",
    }

    keywords_file = files[method]

    if not keywords_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Negative keyword results for {method} not found.",
        )

    with open(keywords_file, "r", encoding="utf-8") as f:
        report = json.load(f)

    terms = report.get(kind, [])

    titles = {
        "tfidf": "Top Negative Terms — TF-IDF",
        "keybert": "Top Negative Terms — KeyBERT",
        "spacy-pos": "Top Negative Terms — spaCy POS",
    }

    image = render_top_negative_terms(
        terms=terms,
        title=titles[method],
    )

    return Response(
        content=image,
        media_type="image/png",
    )


@app.get(
    "/dashboard",
    response_class=HTMLResponse,
    summary="Review analysis dashboard",
)
def dashboard(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
    )