"""Orchestration entry point for the App Store review analysis pipeline.

This module contains no analysis logic of its own. It sequences the
existing collector / processing components and persists their outputs
under results/. All algorithms live in their respective modules
(collector, processing.*) — this file only wires them together.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Callable, NoReturn, TypeVar

from collector import ReviewCollectionError
from collector.fetchlayer_client import FetchLayerReviewClient
from processing.loader import load_reviews
from processing.metrics import compute_metrics
from processing.sentiment import attach_sentiment
from processing.transformer_sentiment import attach_sentiment_transformer
from processing.llm_sentiment import attach_sentiment_llm
from processing.keywords import analyze_negative_keywords_and_phrases, NegativeTermsReport
from processing.spacy_keywords import analyze_negative_keywords_and_phrases_spacy
from processing.keybert_keywords import analyze_negative_keywords_and_phrases_keybert
from processing.quality_checks import run_quality_checks, NLPQualityReport
from processing.llm_insights import generate_insight_report, InsightReport, LLMInsightsError
from processing.results import save_json, save_pipeline_results

from processing.evaluation import run_evaluation, save_evaluation_report, EvaluationReport

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_REVIEWS_PATH = PROJECT_ROOT / "review.json"
RESULTS_DIR = PROJECT_ROOT / "results"

TOTAL_STEPS = 10

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _progress(step: int, message: str) -> None:
    print(f"[{step}/{TOTAL_STEPS}] {message}")


def _fail(message: str) -> NoReturn:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def _run_stage(description: str, func: Callable[[], T]) -> T:
    try:
        return func()
    except Exception as exc:  # noqa: BLE001 - CLI boundary: any failure becomes a clean exit(1)
        _fail(f"{description} failed: {exc}")


def _quiet_third_party_logging() -> None:
    """Keep the terminal concise: only our own progress lines should print."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full App Store review analysis pipeline end to end."
    )
    parser.add_argument(
        "app_id",
        nargs="?",
        default=os.environ.get("APP_ID"),
        help="Apple App Store numeric app id. Falls back to the APP_ID environment variable.",
    )
    parser.add_argument(
        "--skip-collection",
        action="store_true",
        help="Skip FetchLayer collection and process the existing review.json file.",
    )
    parser.add_argument("--country", default="us", help="Storefront country code (default: us)")
    parser.add_argument("--limit", type=int, default=100, help="Max reviews to collect (default: 100)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def collect_and_save_raw_reviews(app_id: str, country: str, limit: int) -> None:
    try:
        client = FetchLayerReviewClient()
    except ValueError as exc:
        _fail(f"Collector configuration error: {exc}")

    try:
        reviews = client.get_reviews(app_id=app_id, limit=limit, country=country)
    except ReviewCollectionError as exc:
        _fail(f"Review collection failed: {exc}")

    if not reviews:
        _fail("Collector returned zero reviews.")

    raw_records = [r.model_dump(mode="json") for r in reviews]
    save_json(RAW_REVIEWS_PATH, raw_records)


def load_and_clean_reviews() -> list[dict[str, Any]]:
    try:
        records = load_reviews(RAW_REVIEWS_PATH)
    except (FileNotFoundError, ValueError) as exc:
        _fail(f"Failed to load/validate reviews: {exc}")

    if not records:
        _fail("No valid reviews remained after validation/cleaning.")

    return records


def run_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    return compute_metrics(records)


def run_vader(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _run_stage("VADER sentiment analysis", lambda: attach_sentiment(records))


def run_transformer(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _run_stage("Transformer sentiment analysis", lambda: attach_sentiment_transformer(records))


def run_llm(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _run_stage("LLM sentiment analysis", lambda: attach_sentiment_llm(records))


def run_keywords(records: list[dict[str, Any]]) -> dict[str, NegativeTermsReport]:
    return _run_stage(
        "Negative keyword extraction",
        lambda: {
            "tfidf": analyze_negative_keywords_and_phrases(records),
            "spacy": analyze_negative_keywords_and_phrases_spacy(records),
            "keybert": analyze_negative_keywords_and_phrases_keybert(records),
        },
    )


def run_quality_check_stage(
    records: list[dict[str, Any]],
    keyword_reports: dict[str, NegativeTermsReport],
) -> NLPQualityReport:
    """Lightweight sanity checks on sentiment/rating agreement and on
    whether the top negative terms are genuinely negative-specific.

    Diagnostic only -- it never modifies `records` or the keyword
    reports, so a failure here should not abort an otherwise successful
    pipeline run (mirrors how model evaluation is handled below).
    """
    return run_quality_checks(records, keyword_reports)


def run_insights(records: list[dict[str, Any]]) -> InsightReport:
    try:
        return generate_insight_report(records)
    except LLMInsightsError as exc:
        _fail(f"LLM insight generation failed: {exc}")
    except Exception as exc:  # noqa: BLE001
        _fail(f"Unexpected error during LLM insight generation: {exc}")


def run_model_evaluation() -> EvaluationReport:
    """Run the evaluation against the hand-labelled dataset."""
    return _run_stage(
        "Model evaluation against labelled dataset",
        lambda: run_evaluation()
    )


def save_results(
        metrics: dict[str, Any],
        records: list[dict[str, Any]],
        keyword_reports: dict[str, NegativeTermsReport],
        insight_report: InsightReport,
        evaluation_report: EvaluationReport,
        quality_report: NLPQualityReport | None = None,
) -> None:
    # Зберігаємо результати основного пайплайну
    save_pipeline_results(RESULTS_DIR, metrics, records, keyword_reports, insight_report)

    # Зберігаємо звіт про оцінку (evaluation)
    if evaluation_report:
        save_evaluation_report(evaluation_report, RESULTS_DIR)

    # Зберігаємо звіт про якість NLP (rating/sentiment + keyword leakage)
    if quality_report:
        RESULTS_DIR.mkdir(exist_ok=True)
        save_json(RESULTS_DIR / "quality_report.json", quality_report.model_dump())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    _quiet_third_party_logging()
    args = parse_args()

    if args.skip_collection:
        print("Skipping review collection. Using existing review.json...")
        _progress(1, "Loading and preprocessing...")
        records = load_and_clean_reviews()
    else:
        if not args.app_id:
            _fail("No app_id provided. Pass it as an argument or set the APP_ID environment variable.")

        _progress(1, "Collecting reviews...")
        collect_and_save_raw_reviews(args.app_id, args.country, args.limit)

        _progress(2, "Loading and preprocessing...")
        records = load_and_clean_reviews()

    _progress(3, "Calculating metrics...")
    metrics = run_metrics(records)

    _progress(4, "Running VADER sentiment...")
    records = run_vader(records)

    _progress(5, "Running Transformer sentiment...")
    records = run_transformer(records)

    _progress(6, "Running LLM sentiment...")
    records = run_llm(records)

    _progress(7, "Extracting negative keywords...")
    keyword_reports = run_keywords(records)

    _progress(8, "Running NLP quality checks...")
    try:
        quality_report = run_quality_check_stage(records, keyword_reports)
        if quality_report.warnings:
            print("    -> Quality check warnings:")
            for warning in quality_report.warnings:
                print(f"       - {warning}")
    except Exception as exc:
        print(f"    -> Warning: Quality checks skipped ({exc})")
        quality_report = None

    _progress(9, "Generating LLM insights...")
    insight_report = run_insights(records)

    _progress(10, "Evaluating sentiment models on reference dataset...")
    try:
        evaluation_report = run_model_evaluation()
    except Exception as exc:
        print(f"    -> Warning: Evaluation skipped ({exc})")
        evaluation_report = None

    save_results(metrics, records, keyword_reports, insight_report, evaluation_report, quality_report)

    print("\nPipeline completed successfully.")

    if evaluation_report:
        print(f"\nEvaluation Summary ({evaluation_report.dataset_size} reviews):")
        for name, ev in evaluation_report.models.items():
            print(
                f"  {name:12s} accuracy={ev.accuracy:.3f}  macro_f1={ev.f1_macro:.3f}  "
                f"failed={ev.failed_count}"
            )

    print(f"\nResults saved to: {RESULTS_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())