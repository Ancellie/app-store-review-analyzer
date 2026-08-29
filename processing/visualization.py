from __future__ import annotations

import io
import logging
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")  # non-interactive backend; required for server-side rendering

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

_FIGSIZE = (7, 4.5)
_DPI = 110


def _figure_to_png_bytes(fig: "plt.Figure") -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=_DPI, bbox_inches="tight")
    plt.close(fig)  # release memory immediately; matplotlib figures are not garbage-collected
    return buf.getvalue()


def render_rating_distribution(metrics: dict[str, Any]) -> bytes:
    """Bar chart of review counts per star rating (1-5).

    Reads the ``rating_counts`` produced by ``processing.metrics.compute_metrics``
    -- no recomputation, just visual rendering of an existing result.
    """
    counts = metrics.get("rating_counts", {})
    ratings = [1, 2, 3, 4, 5]
    values = [counts.get(str(r), counts.get(r, 0)) for r in ratings]

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ax.bar([str(r) for r in ratings], values, color="#4C72B0")
    ax.set_xlabel("Rating (stars)")
    ax.set_ylabel("Number of reviews")
    ax.set_title("Rating distribution")
    for i, v in enumerate(values):
        ax.text(i, v, str(v), ha="center", va="bottom")

    return _figure_to_png_bytes(fig)


def render_sentiment_distribution(sentiment_distribution: dict[str, dict[str, int]]) -> bytes:
    """Grouped bar chart comparing sentiment label counts across methods.

    Reads the ``sentiment_distribution`` block already assembled in
    ``analysis.json`` (vader / transformer / llm), one grouped bar chart
    per method rather than three separate charts.
    """
    methods = list(sentiment_distribution.keys())
    labels = ["positive", "neutral", "negative"]
    colors = {"positive": "#55A868", "neutral": "#8C8C8C", "negative": "#C44E52"}

    x = range(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    for i, label in enumerate(labels):
        values = [sentiment_distribution[m].get(label, 0) for m in methods]
        offsets = [xi + (i - 1) * width for xi in x]
        ax.bar(offsets, values, width=width, label=label, color=colors[label])

    ax.set_xticks(list(x))
    ax.set_xticklabels(methods)
    ax.set_ylabel("Number of reviews")
    ax.set_title("Sentiment distribution by method")
    ax.legend()

    return _figure_to_png_bytes(fig)


def render_sentiment_by_rating(records: Sequence[dict[str, Any]], sentiment_field: str) -> bytes:
    """Stacked bar chart: sentiment label breakdown within each star rating.

    ``records`` is the already-persisted per-review export (e.g. the
    contents of ``sentiment_transformer.json``), which already carries both
    ``rating`` and the sentiment field -- so this aggregates existing data
    rather than re-running any model.
    """
    labels = ["positive", "neutral", "negative"]
    colors = {"positive": "#55A868", "neutral": "#8C8C8C", "negative": "#C44E52"}

    counts: dict[int, dict[str, int]] = {r: {l: 0 for l in labels} for r in range(1, 6)}
    for record in records:
        rating = record.get("rating")
        sentiment = record.get(sentiment_field)
        if rating not in counts or not isinstance(sentiment, dict):
            continue
        label = sentiment.get("label")
        if label in labels:
            counts[rating][label] += 1

    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ratings = list(range(1, 6))
    bottom = [0] * len(ratings)
    for label in labels:
        values = [counts[r][label] for r in ratings]
        ax.bar([str(r) for r in ratings], values, bottom=bottom, label=label, color=colors[label])
        bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_xlabel("Rating (stars)")
    ax.set_ylabel("Number of reviews")
    ax.set_title("Sentiment by rating")
    ax.legend()

    return _figure_to_png_bytes(fig)


def render_top_negative_terms(terms: Sequence[dict[str, Any]], title: str, top_n: int = 15) -> bytes:
    """Horizontal bar chart of the top-N ranked negative keywords/phrases.

    ``terms`` is a list of ``NegativeTerm``-shaped dicts (``term``, ``score``)
    as already produced by the keyword extraction layer -- reads the ranking,
    does not recompute it.
    """
    top = list(terms)[:top_n]
    top.reverse()  # so the highest-ranked term ends up at the top of the chart

    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(top))))
    if not top:
        ax.text(0.5, 0.5, "No terms available", ha="center", va="center")
        ax.axis("off")
    else:
        labels = [t["term"] for t in top]
        scores = [t["score"] for t in top]
        ax.barh(labels, scores, color="#C44E52")
        ax.set_xlabel("Score")

    ax.set_title(title)
    return _figure_to_png_bytes(fig)
