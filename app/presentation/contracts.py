"""Frozen JSON v1.0 contracts; validation only, without pipeline execution or I/O."""

import hashlib
import json
from datetime import datetime
from math import isclose
from re import fullmatch
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    TypeAdapter,
    ValidationInfo,
    field_validator,
    model_validator,
)

from analyzers.base import AnalyzerResult

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
Count = Annotated[int, Field(ge=0)]
Ratio = Annotated[float, Field(ge=0, le=1)]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class QualityCheck(AnalyzerResult):
    """Analyzer output plus the policy action supplied by the upstream gate."""

    model_config = ContractModel.model_config

    check_name: Identifier
    metric_value: float
    details: dict[str, JsonValue] = Field(default_factory=dict)
    action: Literal["warn", "fail"]


class QualityReport(ContractModel):
    schema_version: Literal["1.0"]
    dataset_version: Identifier
    status: Literal["passed", "warning", "failed"]
    checks: list[QualityCheck] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_check_names(self) -> Self:
        names = [check.check_name for check in self.checks]
        if len(names) != len(set(names)):
            raise ValueError("check_name must be unique within a quality report")
        return self


class SplitSummary(ContractModel):
    image_count: Count
    ratio: Ratio


class DatasetSplits(ContractModel):
    train: SplitSummary
    validation: SplitSummary
    test: SplitSummary


class SplitsReport(ContractModel):
    schema_version: Literal["1.0"]
    dataset_version: Identifier
    total_images: int = Field(gt=0)
    splits: DatasetSplits
    class_distribution: dict[str, dict[str, Count]] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    leakage: dict[str, JsonValue] = Field(default_factory=dict, exclude_if=lambda value: not value)

    @model_validator(mode="after")
    def consistent_split_summary(self) -> Self:
        splits = (self.splits.train, self.splits.validation, self.splits.test)
        if sum(split.image_count for split in splits) != self.total_images:
            raise ValueError("split counts must sum to total_images")
        for split in splits:
            if not isclose(
                split.ratio, split.image_count / self.total_images, rel_tol=0, abs_tol=1e-6
            ):
                raise ValueError("each ratio must match image_count / total_images within 1e-6")
        return self


class DatasetRelease(ContractModel):
    dataset_version: Identifier
    quality_file: str
    splits_file: str

    @field_validator("quality_file", "splits_file")
    @classmethod
    def relative_report_reference(cls, value: str, info: ValidationInfo) -> str:
        expected = "quality.json" if info.field_name == "quality_file" else "splits.json"
        parts = value.split("/")
        if (
            parts[-1] != expected
            or any(part in ("", ".", "..") for part in parts)
            or any(fullmatch(r"[A-Za-z0-9._-]+", part) is None for part in parts)
        ):
            raise ValueError(f"reference must be a relative POSIX path ending in {expected}")
        return value


class VersionsReport(ContractModel):
    schema_version: Literal["1.0"]
    releases: list[DatasetRelease]

    @model_validator(mode="after")
    def unique_dataset_versions(self) -> Self:
        versions = [release.dataset_version for release in self.releases]
        if len(versions) != len(set(versions)):
            raise ValueError("dataset_version must be unique within the release catalog")
        return self


# --- Manifest P3 70/20/10 (D02-04) -------------------------------------------------
#
# Espejo exacto de `manifestSummarySchema` en `backend/src/logic/p3.contracts.ts`
# (D01-05, protocolo #33): mismos nombres de campo, mismos literales y las mismas
# reglas, sin `schema_version` (esa API no la versiona así). `test_manifest_contract
# _matches_typescript_fixtures.py` valida este modelo contra los fixtures reales de
# `contracts/p3/fixtures/manifest_summary/`, compartidos con los tests de TS.

DatasetVersion = Annotated[str, StringConstraints(pattern=r"^v\d+\.\d+\.\d+$")]
Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
ManifestClassName = Literal["cat", "dog"]
MANIFEST_CLASSES = ("cat", "dog")
MANIFEST_SEED = 42
MANIFEST_TARGET_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}
MANIFEST_SPLIT_TOLERANCE = 0.05


class ManifestSplitCounts(ContractModel):
    crops: Count
    originals: Count
    crops_per_class: dict[ManifestClassName, Count]

    @model_validator(mode="after")
    def crops_per_class_declares_exactly_the_frozen_classes(self) -> Self:
        """#33 congela `cat`/`dog`: un valor 0 explícito es válido, pero una clave
        ausente NO equivale a 0 (revisión de Heri en PR #54). En el espejo de
        TypeScript, `z.record(classSchema, ...)` con una clave de tipo enum exige
        las dos claves, no solo que las que aparezcan sean válidas — `dict[K, V]`
        de Python no hace eso por sí solo, así que se fuerza aquí."""
        if set(self.crops_per_class) != set(MANIFEST_CLASSES):
            raise ValueError(
                f"crops_per_class debe declarar exactamente {sorted(MANIFEST_CLASSES)}"
            )
        self.crops_per_class = dict(sorted(self.crops_per_class.items()))
        return self

    @model_validator(mode="after")
    def crops_per_class_sums_to_crops(self) -> Self:
        if sum(self.crops_per_class.values()) != self.crops:
            raise ValueError("crops_per_class debe sumar crops")
        return self


class ManifestTargetRatios(ContractModel):
    train: Literal[0.7]
    val: Literal[0.2]
    test: Literal[0.1]


class ManifestSplits(ContractModel):
    train: ManifestSplitCounts
    val: ManifestSplitCounts
    test: ManifestSplitCounts


class ManifestSummary(ContractModel):
    """`GET /api/manifest` (D01-05): la implementación es de Hannah, el contenido es
    de Ale. `frozen=False` es lo único que este ticket (D02-04) puede producir; la
    congelación oficial (`frozen=True`) es de D03-01."""

    manifest_version: Annotated[str, Field(min_length=1)]
    manifest_hash: Sha256Hex
    dataset_version: DatasetVersion
    dvc_release_hash: Sha256Hex
    seed: Literal[42]
    target_ratios: ManifestTargetRatios
    frozen: bool
    classes: list[ManifestClassName]
    splits: ManifestSplits

    @model_validator(mode="after")
    def classes_are_exactly_the_frozen_set(self) -> Self:
        if set(self.classes) != set(MANIFEST_CLASSES) or len(self.classes) != len(MANIFEST_CLASSES):
            raise ValueError(f"classes debe ser exactamente {MANIFEST_CLASSES}")
        return self

    @model_validator(mode="after")
    def each_split_within_tolerance_of_its_target(self) -> Self:
        splits = {"train": self.splits.train, "val": self.splits.val, "test": self.splits.test}
        total = sum(split.crops for split in splits.values())
        targets = {
            "train": self.target_ratios.train,
            "val": self.target_ratios.val,
            "test": self.target_ratios.test,
        }
        for name, split in splits.items():
            if (
                total == 0
                or abs(split.crops / total - targets[name]) > MANIFEST_SPLIT_TOLERANCE + 1e-12
            ):
                raise ValueError(
                    f"{name} fuera de ±5 pp de {targets[name] * 100:.0f}% (medido en crops)"
                )
        return self

    @model_validator(mode="after")
    def val_and_test_contain_every_class(self) -> Self:
        for name, split in (("val", self.splits.val), ("test", self.splits.test)):
            for class_name in MANIFEST_CLASSES:
                if split.crops_per_class.get(class_name, 0) == 0:
                    raise ValueError(f"La clase {class_name} debe estar presente en {name}")
        return self


# ---------------------------------------------------------------------------
# D03-03 — Artefacto del manifest P3 CONGELADO (lo produce D03-01, lo consume Training).
# ---------------------------------------------------------------------------


class FrozenAssignments(ContractModel):
    """crop_id por partición. El trainer solo recibe `train`/`val`; `test` se incluye
    para poder verificar integridad (hash, cobertura, fuga), nunca para entrenar."""

    train: Annotated[list[Count], Field(min_length=1)]
    val: Annotated[list[Count], Field(min_length=1)]
    test: Annotated[list[Count], Field(min_length=1)]


class FrozenManifest(ContractModel):
    """Manifest oficial congelado de D03-01 (versionado con DVC).

    No basta `frozen: true`: Training recalcula `manifest_hash` desde el contenido,
    `test_split_hash` desde la partición test, compara la identidad DVC contra el
    release resuelto y vuelve a comprobar cobertura/fuga contra los crops reales
    (`trainer_worker.sources.verify_training_sources`)."""

    manifest_version: Annotated[str, Field(min_length=1)]
    manifest_hash: Sha256Hex
    dataset_version: DatasetVersion
    dvc_release_hash: Sha256Hex
    images_md5: Annotated[str, Field(min_length=1)]
    annotations_md5: Annotated[str, Field(min_length=1)]
    seed: Literal[42]
    target_ratios: ManifestTargetRatios
    frozen: Literal[True]
    test_split_hash: Sha256Hex
    assignments: FrozenAssignments


def frozen_test_split_hash(test_crop_ids: list[int] | tuple[int, ...]) -> str:
    """Hash de la partición test: SHA-256 del JSON canónico de sus crop_id ordenados."""
    canonical = json.dumps(sorted(test_crop_ids), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# D03-05 — `GET /api/evaluation` (custodia del frozen test: Ale).
#
# Espejo de `evaluationResponseSchema` en `backend/src/logic/p3.contracts.ts`
# (D01-05): mismos campos, literales, tolerancias y las mismas ramas del
# `superRefine`, ni más ni menos estrictas. `test_evaluation_contract.py` lo valida
# contra los fixtures compartidos de `contracts/p3/fixtures/evaluation_response/`.
# ---------------------------------------------------------------------------

MlflowRunId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
# z.iso.datetime({ offset: true }): fecha y hora ISO 8601 con `Z` u offset explícito.
IsoTimestamp = Annotated[
    str,
    StringConstraints(
        pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:\d{2})$"
    ),
]
EVALUATION_REPORTED_METRIC_TOLERANCE = 1e-4
EVALUATION_EXACT_TOLERANCE = 1e-9


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _close(a: float, b: float, tolerance: float) -> bool:
    return abs(a - b) <= tolerance


class EvaluationSelection(ContractModel):
    candidate_run_id: MlflowRunId
    metric: Literal["val_accuracy"]
    closed_at: IsoTimestamp


class EvaluationConfusionMatrix(ContractModel):
    """Filas = clase real, columnas = clase predicha (rúbrica 4.2)."""

    labels: list[ManifestClassName]
    rows: list[list[Count]]


class EvaluationClassMetrics(ContractModel):
    class_name: ManifestClassName
    precision: Ratio
    recall: Ratio
    f1: Ratio
    support: Count


class EvaluationMetrics(ContractModel):
    accuracy: Ratio
    macro_f1: Ratio
    per_class: list[EvaluationClassMetrics]


class EvaluationBlocked(ContractModel):
    """Antes de MODEL SELECTION CLOSED no se revela nada del test."""

    state: Literal["blocked"]
    reason: Literal["model_selection_open"]
    detail: str


# D05-05: `synthetic` = recorridos de prueba; nunca se presenta como evaluación oficial.
# `local_test` es del registro de modelos (D04-06), no un namespace de Evaluation.
EvaluationNamespace = Literal["official", "synthetic"]


class EvaluationPending(ContractModel):
    """D05-05: selección cerrada sin evaluación guardada (resultado ausente, no un fallo).
    Solo la identidad de la selección cerrada; nada del test."""

    state: Literal["pending"]
    namespace: EvaluationNamespace
    reason: Literal["evaluation_missing"]
    selection: EvaluationSelection
    manifest_hash: Sha256Hex
    detail: str


def _exactly_frozen_classes(classes: list[str]) -> bool:
    return len(classes) == len(MANIFEST_CLASSES) and set(classes) == set(MANIFEST_CLASSES)


class EvaluationReady(ContractModel):
    state: Literal["ready"]
    namespace: EvaluationNamespace
    selection: EvaluationSelection
    manifest_hash: Sha256Hex
    evaluated_at: IsoTimestamp
    n_test: Annotated[int, Field(gt=0)]
    classes: list[ManifestClassName]
    confusion_matrix: EvaluationConfusionMatrix
    metrics: EvaluationMetrics
    majority_baseline_accuracy: Ratio

    @model_validator(mode="after")
    def classes_are_exactly_the_frozen_set(self) -> Self:
        for name, classes in (
            ("classes", self.classes),
            ("confusion_matrix.labels", self.confusion_matrix.labels),
        ):
            if not _exactly_frozen_classes(classes):
                raise ValueError(f"{name} debe declarar exactamente {MANIFEST_CLASSES}")
        return self

    @model_validator(mode="after")
    def evaluated_only_after_model_selection_closed(self) -> Self:
        if _parse_timestamp(self.evaluated_at) <= _parse_timestamp(self.selection.closed_at):
            raise ValueError("El test se evalúa solo después de MODEL SELECTION CLOSED")
        return self

    @model_validator(mode="after")
    def metrics_are_coherent_with_the_matrix(self) -> Self:
        labels = self.confusion_matrix.labels
        rows = self.confusion_matrix.rows
        if len(rows) != len(labels) or any(len(row) != len(labels) for row in rows):
            raise ValueError("La matriz debe ser cuadrada con una fila/columna por clase")
        if sum(map(sum, rows)) != self.n_test:
            raise ValueError("La matriz debe sumar n_test")
        trace = sum(rows[i][i] for i in range(len(labels)))
        if not _close(self.metrics.accuracy, trace / self.n_test, EVALUATION_EXACT_TOLERANCE):
            raise ValueError("accuracy debe ser traza / n_test, sin redondear")

        supports = [sum(row) for row in rows]
        reported_f1 = []
        for i, label in enumerate(labels):
            stats = next((s for s in self.metrics.per_class if s.class_name == label), None)
            if stats is None:
                raise ValueError(f"Faltan métricas de la clase {label}")
            tp = rows[i][i]
            predicted = sum(row[i] for row in rows)
            precision = 0 if predicted == 0 else tp / predicted
            recall = 0 if supports[i] == 0 else tp / supports[i]
            f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
            reported_f1.append(stats.f1)
            if stats.support != supports[i]:
                raise ValueError(f"support de {label} ≠ suma de su fila")
            if not all(
                _close(got, want, EVALUATION_REPORTED_METRIC_TOLERANCE)
                for got, want in (
                    (stats.precision, precision),
                    (stats.recall, recall),
                    (stats.f1, f1),
                )
            ):
                raise ValueError(f"precision/recall/F1 de {label} no coinciden con la matriz")

        macro = sum(reported_f1) / max(len(reported_f1), 1)
        if not _close(self.metrics.macro_f1, macro, EVALUATION_REPORTED_METRIC_TOLERANCE):
            raise ValueError("macro_f1 debe ser el promedio de los F1 por clase")
        majority = max(supports) / self.n_test
        if not _close(self.majority_baseline_accuracy, majority, EVALUATION_EXACT_TOLERANCE):
            raise ValueError("Baseline = soporte de la clase mayoritaria / n_test")
        return self


EvaluationResponse = Annotated[
    EvaluationBlocked | EvaluationPending | EvaluationReady, Field(discriminator="state")
]
EVALUATION_RESPONSE: TypeAdapter[EvaluationBlocked | EvaluationPending | EvaluationReady] = (
    TypeAdapter(EvaluationResponse)
)


# ---------------------------------------------------------------------------
# D04-05 — `GET /api/evaluation/predictions`: exportación por muestra.
#
# Espejo de `evaluationPredictionsSchema` (`backend/src/logic/p3.contracts.ts`).
# `synthetic` marca los recorridos de prueba: nunca se sirven como evaluación oficial.
# ---------------------------------------------------------------------------
EVALUATION_NAMESPACES = ("official", "synthetic")
PROBABILITY_SUM_TOLERANCE = 1e-3


class EvaluationSample(ContractModel):
    crop_id: Annotated[int, Field(gt=0)]
    true_class: ManifestClassName
    predicted_class: ManifestClassName
    probabilities: dict[ManifestClassName, Ratio]


class EvaluationPredictions(ContractModel):
    namespace: Literal["official", "synthetic"]
    candidate_run_id: MlflowRunId
    manifest_hash: Sha256Hex
    test_split_hash: Sha256Hex
    evaluated_at: IsoTimestamp
    n_test: Annotated[int, Field(gt=0)]
    classes: list[ManifestClassName]
    predictions: list[EvaluationSample]

    @field_validator("evaluated_at")
    @classmethod
    def evaluated_at_is_a_real_instant(cls, value: str) -> str:
        """El patrón solo revisa el formato; `z.iso.datetime` también rechaza fechas que
        no existen (p. ej. 30 de febrero), y el espejo debe rechazar lo mismo."""
        try:
            _parse_timestamp(value)
        except ValueError as error:
            raise ValueError(f"evaluated_at no es una fecha/hora real: {value}") from error
        return value

    @model_validator(mode="after")
    def classes_are_exactly_the_frozen_set(self) -> Self:
        if not _exactly_frozen_classes(self.classes):
            raise ValueError(f"classes debe declarar exactamente {MANIFEST_CLASSES}")
        return self

    @model_validator(mode="after")
    def samples_are_coherent(self) -> Self:
        if len(self.predictions) != self.n_test:
            raise ValueError("Debe haber exactamente n_test predicciones")
        for previous, sample in zip(self.predictions, self.predictions[1:], strict=False):
            if sample.crop_id <= previous.crop_id:
                raise ValueError("crop_id en orden estrictamente creciente (sin repetidos)")
        for sample in self.predictions:
            probabilities = sample.probabilities
            if len(probabilities) != len(self.classes) or set(probabilities) != set(self.classes):
                raise ValueError(
                    f"crop {sample.crop_id}: una probabilidad por clase declarada y ninguna otra"
                )
            if not _close(sum(probabilities.values()), 1, PROBABILITY_SUM_TOLERANCE):
                raise ValueError(f"crop {sample.crop_id}: las probabilidades deben sumar ~1")
            if probabilities[sample.predicted_class] != max(probabilities.values()):
                raise ValueError(
                    f"crop {sample.crop_id}: predicted_class debe ser el argmax de probabilities"
                )
        return self
