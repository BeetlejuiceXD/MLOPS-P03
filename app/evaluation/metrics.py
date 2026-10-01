"""D03-05 — Motor de métricas de evaluación (stub Red)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from presentation.contracts import MANIFEST_CLASSES, EvaluationReady

LABELS: tuple[str, ...] = MANIFEST_CLASSES
ACCEPTANCE_ACCURACY = Fraction(85, 100)


class MetricsInputError(ValueError):
    """Entradas que no se pueden evaluar."""


@dataclass(frozen=True)
class ClassMetrics:
    class_name: str
    precision: float
    recall: float
    f1: float
    support: int
    predicted: int


@dataclass(frozen=True)
class MetricsReport:
    labels: tuple[str, ...]
    rows: tuple[tuple[int, ...], ...]
    n_test: int
    correct: int
    accuracy: float
    macro_f1: float
    per_class: tuple[ClassMetrics, ...]
    majority_baseline_accuracy: float

    def meets_acceptance(self) -> bool:
        raise NotImplementedError

    def to_ready_response(
        self, *, candidate_run_id: str, closed_at: str, manifest_hash: str, evaluated_at: str
    ) -> EvaluationReady:
        raise NotImplementedError


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str]) -> tuple[tuple[int, ...], ...]:
    raise NotImplementedError


def compute_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> MetricsReport:
    raise NotImplementedError


def meets_acceptance(correct: int, total: int) -> bool:
    raise NotImplementedError
