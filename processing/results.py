from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from processing.keywords import NegativeTermsReport
from processing.llm_insights import InsightReport


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding="utf-8")


def label_distribution(records: list[dict[str, Any]], sentiment_field: str) -> dict[str, int]:
    """Count how many *records* fall into each sentiment label for one method."""
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for record in records:
        sentiment = record.get(sentiment_field)
        if isinstance(sentiment, dict) and sentiment.get("label") in counts:
            counts[sentiment["label"]] += 1
    return counts


def count_sentiment_errors(records: list[dict[str, Any]], sentiment_field: str) -> int:
    """Count records whose sentiment result recorded a failure rather than a label.

    ``label_distribution`` only counts real labels ("positive"/"neutral"/
    "negative"), so a failed call (``label=None``, see
    ``processing.llm_sentiment``) is silently excluded from it rather than
    miscounted as neutral — which is correct, but on its own gives no
    visibility into *whether* any calls failed. This is that visibility.
    """
    count = 0
    for record in records:
        sentiment = record.get(sentiment_field)
        if isinstance(sentiment, dict) and sentiment.get("label") is None:
            count += 1
    return count


def build_sentiment_export(records: list[dict[str, Any]], sentiment_field: str) -> list[dict[str, Any]]:
    """Slim per-review export for one sentiment method (rating/title/text/result)."""
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


def build_analysis_summary(
    metrics: dict[str, Any],
    records: list[dict[str, Any]],
    keyword_reports: dict[str, NegativeTermsReport],
    insight_report: InsightReport,
) -> dict[str, Any]:

    return {
        "metrics": metrics,
        "sentiment_distribution": {
            "vader": label_distribution(records, "sentiment"),
            "transformer": label_distribution(records, "sentiment_transformer"),
            "llm": label_distribution(records, "sentiment_llm"),
        },
        "llm_sentiment_errors": count_sentiment_errors(records, "sentiment_llm"),
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


def save_pipeline_results(
    results_dir: Path,
    metrics: dict[str, Any],
    records: list[dict[str, Any]],
    keyword_reports: dict[str, NegativeTermsReport],
    insight_report: InsightReport,
) -> None:

    results_dir.mkdir(exist_ok=True)

    save_json(results_dir / "metrics.json", metrics)
    save_json(results_dir / "sentiment_vader.json", build_sentiment_export(records, "sentiment"))
    save_json(results_dir / "sentiment_transformer.json", build_sentiment_export(records, "sentiment_transformer"))
    save_json(results_dir / "sentiment_llm.json", build_sentiment_export(records, "sentiment_llm"))

    for name, report in keyword_reports.items():
        save_json(results_dir / f"negative_keywords_{name}.json", report.model_dump())

    save_json(results_dir / "insights.json", insight_report.model_dump())

    analysis = build_analysis_summary(metrics, records, keyword_reports, insight_report)
    save_json(results_dir / "analysis.json", analysis)