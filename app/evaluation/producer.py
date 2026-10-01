"""D04-05 — Productor de la evaluación del frozen test (custodia: Ale).

Une las piezas ya entregadas:

- el motor de métricas de D03-05 (`evaluation.metrics`) calcula matriz y métricas;
- la guarda de D04-04 (`p3_model_selection` en estado `closed`) se comprueba ANTES
  de tocar una sola predicción;
- la exportación por muestra (`EvaluationPredictions`) lleva, por crop, la clase
  real, la predicha y las probabilidades, con candidato, manifest y `test_split_hash`.

El resultado se guarda en `p3_evaluation` por namespace: `official` es la evaluación
del frozen test (D06-01, una sola vez, nunca se sobrescribe) y `synthetic` los
recorridos de prueba con predicciones conocidas, que la API nunca sirve como oficiales.

Este módulo no carga modelos, imágenes ni el manifest: recibe predicciones ya
calculadas y la partición test que el llamador tomó del manifest congelado.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from pydantic import ValidationError

from evaluation.metrics import LABELS, MetricsInputError, compute_metrics
from presentation.contracts import (
    EVALUATION_NAMESPACES,
    EvaluationPredictions,
    EvaluationReady,
    frozen_test_split_hash,
)

NAMESPACES = EVALUATION_NAMESPACES


class EvaluationRefusedError(Exception):
    """La evaluación no se produce. `reason` es estable (para tests y logs);
    `detail` explica el caso concreto."""

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
    """MODEL SELECTION CLOSED tal como quedó persistido (D04-04 / D05-02)."""

    candidate_run_id: str
    closed_at: datetime  # con zona (UTC)
    manifest_hash: str


@dataclass(frozen=True)
class TestPartition:
    """Partición test del manifest congelado: solo IDs y hashes, ninguna imagen."""

    __test__ = False  # no es una clase de tests de pytest

    manifest_hash: str
    test_split_hash: str
    crop_ids: tuple[int, ...]

    @classmethod
    def from_manifest(cls, manifest: Any) -> TestPartition:
        """Desde un `FrozenManifest` (D03-01)."""
        return cls(
            manifest_hash=manifest.manifest_hash,
            test_split_hash=manifest.test_split_hash,
            crop_ids=tuple(sorted(manifest.assignments.test)),
        )


@dataclass(frozen=True)
class EvaluationRecord:
    namespace: str
    evaluation: EvaluationReady
    predictions: EvaluationPredictions


class SelectionStore(Protocol):
    def closed_selection(self) -> ClosedSelection: ...

    def save(self, record: EvaluationRecord) -> None: ...


def _to_millis(moment: datetime) -> datetime:
    """MariaDB `timestamp(3)` y el contrato trabajan al milisegundo."""
    moment = moment.astimezone(UTC)
    return moment.replace(microsecond=moment.microsecond // 1000 * 1000)


def iso_utc(moment: datetime) -> str:
    """`2026-10-01T13:00:00.456Z`: el mismo formato que `Date.toISOString()`."""
    moment = _to_millis(moment)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.") + f"{moment.microsecond // 1000:03d}Z"


def confusion_rows(predictions: EvaluationPredictions) -> list[list[int]]:
    """Matriz reconstruida desde la exportación: filas reales, columnas predichas."""
    index = {name: i for i, name in enumerate(predictions.classes)}
    rows = [[0] * len(index) for _ in index]
    for sample in predictions.predictions:
        rows[index[sample.true_class]][index[sample.predicted_class]] += 1
    return rows


def build_evaluation(
    samples: Iterable[SamplePrediction],
    *,
    namespace: str,
    model_run_id: str,
    selection: ClosedSelection,
    partition: TestPartition,
    evaluated_at: datetime,
) -> EvaluationRecord:
    """Evaluación + exportación de `samples`, o `EvaluationRefusedError` con el motivo.

    No consulta la base: la guarda de selección cerrada la aplica `produce_evaluation`.
    """
    if namespace not in NAMESPACES:
        raise EvaluationRefusedError("unknown_namespace", f"{namespace!r} no es {NAMESPACES}")
    if model_run_id != selection.candidate_run_id:
        raise EvaluationRefusedError(
            "not_selected_candidate",
            f"las predicciones son del run {model_run_id}; el candidato cerrado es "
            f"{selection.candidate_run_id}",
        )
    if partition.manifest_hash != selection.manifest_hash:
        raise EvaluationRefusedError(
            "manifest_mismatch",
            f"partición del manifest {partition.manifest_hash}; la selección se cerró "
            f"con {selection.manifest_hash}",
        )
    if frozen_test_split_hash(list(partition.crop_ids)) != partition.test_split_hash:
        raise EvaluationRefusedError(
            "test_split_hash_mismatch", "los crop_id de la partición no dan su test_split_hash"
        )

    samples = list(samples)
    ids = [sample.crop_id for sample in samples]
    if len(set(ids)) != len(ids):
        raise EvaluationRefusedError("duplicate_crop_id", "hay crops evaluados más de una vez")
    expected = set(partition.crop_ids)
    missing, extra = expected - set(ids), set(ids) - expected
    if missing or extra:
        raise EvaluationRefusedError(
            "crop_ids_not_test_split",
            f"faltan {len(missing)}, sobran {len(extra)} respecto a la partición test",
        )

    evaluated_at, closed_at = _to_millis(evaluated_at), _to_millis(selection.closed_at)
    if evaluated_at <= closed_at:
        raise EvaluationRefusedError(
            "evaluated_before_close",
            f"evaluated_at {iso_utc(evaluated_at)} no es posterior al cierre {iso_utc(closed_at)}",
        )

    ordered = sorted(samples, key=lambda sample: sample.crop_id)
    try:
        predictions = EvaluationPredictions.model_validate(
            {
                "namespace": namespace,
                "candidate_run_id": selection.candidate_run_id,
                "manifest_hash": partition.manifest_hash,
                "test_split_hash": partition.test_split_hash,
                "evaluated_at": iso_utc(evaluated_at),
                "n_test": len(ordered),
                "classes": list(LABELS),
                "predictions": [
                    {
                        "crop_id": sample.crop_id,
                        "true_class": sample.true_class,
                        "predicted_class": sample.predicted_class,
                        "probabilities": dict(sample.probabilities),
                    }
                    for sample in ordered
                ],
            }
        )
        report = compute_metrics(
            [sample.true_class for sample in ordered],
            [sample.predicted_class for sample in ordered],
        )
    except (ValidationError, MetricsInputError) as error:
        raise EvaluationRefusedError("invalid_prediction", str(error)) from error

    evaluation = report.to_ready_response(
        candidate_run_id=selection.candidate_run_id,
        closed_at=iso_utc(closed_at),
        manifest_hash=selection.manifest_hash,
        evaluated_at=iso_utc(evaluated_at),
    )
    return EvaluationRecord(namespace=namespace, evaluation=evaluation, predictions=predictions)


def produce_evaluation(
    store: SelectionStore,
    samples: Iterable[SamplePrediction],
    *,
    namespace: str,
    model_run_id: str,
    partition: TestPartition,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> EvaluationRecord:
    """Guarda primero (selección cerrada), luego calcula y persiste."""
    selection = store.closed_selection()
    record = build_evaluation(
        samples,
        namespace=namespace,
        model_run_id=model_run_id,
        selection=selection,
        partition=partition,
        evaluated_at=clock(),
    )
    store.save(record)
    return record
