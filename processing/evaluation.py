from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, Field, ValidationError
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from processing.cleaner import clean_review
from processing.results import save_json
from processing.sentiment import SentimentLabel, analyze_sentiment as _vader_analyze_sentiment
from processing.transformer_sentiment import analyze_sentiment as _transformer_analyze_sentiment
from processing.llm_sentiment import analyze_sentiment as _llm_analyze_sentiment

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "evaluation" / "labelled_reviews.json"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"

CANONICAL_LABELS: tuple[SentimentLabel, ...] = ("negative", "neutral", "positive")

_METHOD_NAMES: tuple[str, ...] = ("vader", "transformer", "llm")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class LabelledReview(BaseModel):

    review_id: str
    title: str
    review: str
    rating: int = Field(ge=1, le=5)
    label: SentimentLabel


class ReviewPrediction(BaseModel):
    """Per-review audit record: what each method predicted vs. the reference."""

    review_id: str
    reference_label: SentimentLabel
    predictions: dict[str, SentimentLabel | None] = Field(
        description="Method name -> predicted label, or None if that method failed on this review."
    )
    errors: dict[str, str] = Field(
        default_factory=dict,
        description="Method name -> error message, present only for methods that failed on this review.",
    )


class PerClassMetrics(BaseModel):
    """Precision/recall/F1 for a single class, plus how many reference examples had it."""

    precision: float
    recall: float
    f1: float
    support: int = Field(description="Number of reference (human-labelled) examples with this label.")


class MethodEvaluation(BaseModel):
    """Classification metrics for one sentiment method against the reference labels."""

    method: str
    evaluated_count: int = Field(description="Reviews this method produced a usable label for.")
    failed_count: int = Field(description="Reviews this method failed to produce a label for.")
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    per_class: dict[str, PerClassMetrics] = Field(
        description="Label -> precision/recall/F1/support for that class alone (unaveraged)."
    )
    confusion_matrix: list[list[int]] = Field(
        description="Rows = reference label, columns = predicted label, both ordered per `labels`."
    )


class EvaluationReport(BaseModel):
    """Top-level result of a full evaluation run."""

    dataset_size: int
    labels: list[str] = Field(default_factory=lambda: list(CANONICAL_LABELS))
    models: dict[str, MethodEvaluation]
    predictions: list[ReviewPrediction]


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_labelled_dataset(path: str | Path = DEFAULT_DATASET_PATH) -> list[LabelledReview]:
    """Load and validate the hand-labelled reference dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Labelled evaluation dataset not found: {path}")

    with path.open(encoding="utf-8") as fh:
        try:
            raw = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Cannot parse {path} as JSON: {exc}") from exc

    if not isinstance(raw, list):
        raise ValueError(
            f"Expected a JSON array at the top level of {path}, got {type(raw).__name__}."
        )
    if not raw:
        raise ValueError(f"Labelled evaluation dataset at {path} is empty.")

    dataset: list[LabelledReview] = []
    for idx, entry in enumerate(raw):
        try:
            dataset.append(LabelledReview.model_validate(entry))
        except ValidationError as exc:
            raise ValueError(f"Entry #{idx} in {path} failed validation: {exc}") from exc

    return dataset


# ---------------------------------------------------------------------------
# Per-method prediction (thin wrappers around the existing analyze_sentiment
# functions — each returns (label_or_none, error_or_none) so a failure in
# any one method never stops the evaluation of the other two, or of the
# remaining reviews).
# ---------------------------------------------------------------------------

def _run_vader(text: str) -> tuple[SentimentLabel | None, str | None]:
    try:
        return _vader_analyze_sentiment(text).label, None
    except Exception as exc:  # noqa: BLE001 - defensive: VADER shouldn't fail on str input
        logger.warning("VADER prediction failed: %s", exc)
        return None, str(exc)


def _run_transformer(text: str) -> tuple[SentimentLabel | None, str | None]:
    try:
        return _transformer_analyze_sentiment(text).label, None
    except Exception as exc:  # noqa: BLE001 - e.g. an unmapped raw label (see transformer_sentiment.py)
        logger.warning("Transformer prediction failed: %s", exc)
        return None, str(exc)


def _run_llm(text: str) -> tuple[SentimentLabel | None, str | None]:
    try:
        result = _llm_analyze_sentiment(text)
        return result.label, result.error
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM prediction failed unexpectedly: %s", exc)
        return None, str(exc)


_RUNNERS: dict[str, Any] = {
    "vader": _run_vader,
    "transformer": _run_transformer,
    "llm": _run_llm,
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_method_evaluation(
    method: str,
    y_true: Sequence[SentimentLabel],
    y_pred: Sequence[SentimentLabel | None],
    labels: Sequence[str] = CANONICAL_LABELS,
) -> MethodEvaluation:
    """Compute accuracy/precision/recall/macro-F1/confusion matrix for one method."""
    total = len(y_true)
    paired = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    failed_count = total - len(paired)
    label_list = list(labels)

    if not paired:
        size = len(label_list)
        return MethodEvaluation(
            method=method,
            evaluated_count=0,
            failed_count=failed_count,
            accuracy=0.0,
            precision_macro=0.0,
            recall_macro=0.0,
            f1_macro=0.0,
            per_class={
                label: PerClassMetrics(precision=0.0, recall=0.0, f1=0.0, support=0)
                for label in label_list
            },
            confusion_matrix=[[0] * size for _ in range(size)],
        )

    filtered_true = [t for t, _ in paired]
    filtered_pred = [p for _, p in paired]

    accuracy = accuracy_score(filtered_true, filtered_pred)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        filtered_true,
        filtered_pred,
        labels=label_list,
        average="macro",
        zero_division=0,
    )
    precision_per, recall_per, f1_per, support_per = precision_recall_fscore_support(
        filtered_true,
        filtered_pred,
        labels=label_list,
        average=None,
        zero_division=0,
    )
    cm = confusion_matrix(filtered_true, filtered_pred, labels=label_list)

    per_class = {
        label: PerClassMetrics(
            precision=round(float(p), 4),
            recall=round(float(r), 4),
            f1=round(float(f), 4),
            support=int(s),
        )
        for label, p, r, f, s in zip(label_list, precision_per, recall_per, f1_per, support_per)
    }

    return MethodEvaluation(
        method=method,
        evaluated_count=len(paired),
        failed_count=failed_count,
        accuracy=round(float(accuracy), 4),
        precision_macro=round(float(precision_macro), 4),
        recall_macro=round(float(recall_macro), 4),
        f1_macro=round(float(f1_macro), 4),
        per_class=per_class,
        confusion_matrix=cm.tolist(),
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_evaluation(dataset_path: str | Path = DEFAULT_DATASET_PATH) -> EvaluationReport:
    """Run all three sentiment methods against the labelled dataset and score them."""
    dataset = load_labelled_dataset(dataset_path)

    y_true: list[SentimentLabel] = []
    predictions_by_method: dict[str, list[SentimentLabel | None]] = {name: [] for name in _METHOD_NAMES}
    review_predictions: list[ReviewPrediction] = []

    for item in dataset:
        text = clean_review(item.review)
        y_true.append(item.label)

        labels_for_review: dict[str, SentimentLabel | None] = {}
        errors_for_review: dict[str, str] = {}

        for name, runner in _RUNNERS.items():
            label, error = runner(text)
            predictions_by_method[name].append(label)
            labels_for_review[name] = label
            if error:
                errors_for_review[name] = error

        review_predictions.append(
            ReviewPrediction(
                review_id=item.review_id,
                reference_label=item.label,
                predictions=labels_for_review,
                errors=errors_for_review,
            )
        )

    models = {
        name: _compute_method_evaluation(name, y_true, predictions_by_method[name])
        for name in _METHOD_NAMES
    }

    return EvaluationReport(
        dataset_size=len(dataset),
        labels=list(CANONICAL_LABELS),
        models=models,
        predictions=review_predictions,
    )


def save_evaluation_report(report: EvaluationReport, results_dir: str | Path = DEFAULT_RESULTS_DIR) -> None:
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)
    save_json(results_dir / "evaluation.json", report.model_dump())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    report = run_evaluation()
    save_evaluation_report(report)

    print(f"Evaluated {report.dataset_size} labelled review(s).\n")
    for name in _METHOD_NAMES:
        ev = report.models[name]
        print(
            f"  {name:12s} accuracy={ev.accuracy:.3f}  macro_f1={ev.f1_macro:.3f}  "
            f"precision={ev.precision_macro:.3f}  recall={ev.recall_macro:.3f}  "
            f"failed={ev.failed_count}/{report.dataset_size}"
        )

    if report.dataset_size < 30:
        print(
            f"\nNote: {report.dataset_size} labelled examples is a starting point, not a "
            f"statistically reliable sample — treat these numbers as directional until "
            f"the dataset is expanded."
        )

    print(f"\nResults saved to: {DEFAULT_RESULTS_DIR / 'evaluation.json'}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())