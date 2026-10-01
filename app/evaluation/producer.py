"""D04-05 — Productor de la evaluación del frozen test (stub Red)."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from presentation.contracts import EvaluationPredictions, EvaluationReady

NAMESPACES = ("official", "synthetic")


class EvaluationRefusedError(Exception):
    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class SamplePrediction:
    crop_id: int
    true_class: str
    predicted_class: str
    probabilities: Mapping[str, float]


@dataclass(frozen=True)
class ClosedSelection:
    candidate_run_id: str
    closed_at: datetime
    manifest_hash: str


@dataclass(frozen=True)
class TestPartition:
    __test__ = False

    manifest_hash: str
    test_split_hash: str
    crop_ids: tuple[int, ...]

    @classmethod
    def from_manifest(cls, manifest: Any) -> TestPartition:
        raise NotImplementedError


@dataclass(frozen=True)
class EvaluationRecord:
    namespace: str
    evaluation: EvaluationReady
    predictions: EvaluationPredictions


def confusion_rows(predictions: EvaluationPredictions) -> list[list[int]]:
    raise NotImplementedError


def build_evaluation(
    samples: Iterable[SamplePrediction],
    *,
    namespace: str,
    model_run_id: str,
    selection: ClosedSelection,
    partition: TestPartition,
    evaluated_at: datetime,
) -> EvaluationRecord:
    raise NotImplementedError


def produce_evaluation(
    store: Any,
    samples: Iterable[SamplePrediction],
    *,
    namespace: str,
    model_run_id: str,
    partition: TestPartition,
    clock: Callable[[], datetime],
) -> EvaluationRecord:
    raise NotImplementedError
