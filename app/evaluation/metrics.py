"""D03-05 — Motor de métricas de evaluación para el clasificador cat/dog de P3.

Función pura: recibe etiquetas reales y predichas YA calculadas y devuelve la matriz
de confusión, accuracy, macro-F1, precision/recall/F1/support por clase y el
baseline de clase mayoritaria. No lee archivos ni conoce particiones: quién le pasa
las etiquetas decide sobre qué datos se evalúa. Abrir el test oficial es de D06-01,
después de MODEL SELECTION CLOSED (D05-02); este ticket solo se prueba con
predicciones controladas.

Reglas (contrato `evaluation_response` de D01-05, protocolo #33):

- Filas = clase REAL, columnas = clase PREDICHA, siempre en el orden congelado
  (cat, dog), aunque una clase no aparezca en las entradas.
- accuracy = traza / n, sin redondear. support = suma de la fila de la clase.
- Una clase sin predicciones tiene precision 0; una sin soporte, recall 0; sin
  división por cero. Ambas clases cuentan siempre en el macro-F1.
- Baseline = soporte de la clase mayoritaria / n.
- Criterio de aceptación: accuracy >= 0.85, comparado con conteos enteros.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction

from presentation.contracts import MANIFEST_CLASSES, EvaluationReady

LABELS: tuple[str, ...] = MANIFEST_CLASSES
# Fracción exacta: 0.85 como float es 0.84999999999999997779…, y redondear la
# accuracy antes de comparar dejaría pasar, p. ej., 0.8499 (→ "0.85").
ACCEPTANCE_ACCURACY = Fraction(85, 100)


class MetricsInputError(ValueError):
    """Entradas que no se pueden evaluar: longitudes distintas, vacías, clases fuera
    de cat/dog o etiquetas que no son nombres de clase."""


@dataclass(frozen=True)
class ClassMetrics:
    class_name: str
    precision: float
    recall: float
    f1: float
    support: int  # ejemplos reales de la clase (suma de su fila)
    predicted: int  # veces que se predijo la clase (suma de su columna)


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
        """Usa los conteos (`correct`/`n_test`), no `accuracy` ya convertida a float."""
        return meets_acceptance(self.correct, self.n_test)

    def to_ready_response(
        self, *, candidate_run_id: str, closed_at: str, manifest_hash: str, evaluated_at: str
    ) -> EvaluationReady:
        """Salida contractual (`GET /api/evaluation`, estado `ready`) validada con las
        mismas reglas que backend y frontend. La validación rechaza `evaluated_at`
        anterior o igual al cierre de la selección."""
        return EvaluationReady.model_validate(
            {
                "state": "ready",
                "selection": {
                    "candidate_run_id": candidate_run_id,
                    "metric": "val_accuracy",
                    "closed_at": closed_at,
                },
                "manifest_hash": manifest_hash,
                "evaluated_at": evaluated_at,
                "n_test": self.n_test,
                "classes": list(self.labels),
                "confusion_matrix": {
                    "labels": list(self.labels),
                    "rows": [list(row) for row in self.rows],
                },
                "metrics": {
                    "accuracy": self.accuracy,
                    "macro_f1": self.macro_f1,
                    "per_class": [
                        {
                            "class_name": c.class_name,
                            "precision": c.precision,
                            "recall": c.recall,
                            "f1": c.f1,
                            "support": c.support,
                        }
                        for c in self.per_class
                    ],
                },
                "majority_baseline_accuracy": self.majority_baseline_accuracy,
            }
        )


def _validated(y_true: Sequence[str], y_pred: Sequence[str]) -> None:
    for name, values in (("y_true", y_true), ("y_pred", y_pred)):
        if isinstance(values, str | bytes) or not isinstance(values, Sequence):
            raise MetricsInputError(f"{name} debe ser una secuencia de nombres de clase")
        unknown = {repr(v) for v in values if not isinstance(v, str) or v not in LABELS}
        if unknown:
            raise MetricsInputError(
                f"{name} contiene etiquetas fuera de {LABELS}: {sorted(unknown)}"
            )
    if len(y_true) != len(y_pred):
        raise MetricsInputError(
            f"y_true ({len(y_true)}) y y_pred ({len(y_pred)}) deben tener la misma longitud"
        )
    if not y_true:
        raise MetricsInputError("No hay predicciones que evaluar")


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str]) -> tuple[tuple[int, ...], ...]:
    """`rows[i][j]` = ejemplos de clase real `LABELS[i]` predichos como `LABELS[j]`."""
    _validated(y_true, y_pred)
    index = {label: i for i, label in enumerate(LABELS)}
    counts = [[0] * len(LABELS) for _ in LABELS]
    for real, predicted in zip(y_true, y_pred, strict=True):
        counts[index[real]][index[predicted]] += 1
    return tuple(tuple(row) for row in counts)


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def compute_metrics(y_true: Sequence[str], y_pred: Sequence[str]) -> MetricsReport:
    rows = confusion_matrix(y_true, y_pred)
    n = sum(map(sum, rows))
    correct = sum(rows[i][i] for i in range(len(LABELS)))

    per_class = []
    for i, label in enumerate(LABELS):
        tp = rows[i][i]
        support = sum(rows[i])
        predicted = sum(row[i] for row in rows)
        # F1 = 2·TP / (2·TP + FP + FN) = 2·TP / (predicted + support): equivale a
        # 2PR/(P+R) y vale 0 sin dividir entre cero cuando TP = 0.
        per_class.append(
            ClassMetrics(
                class_name=label,
                precision=_ratio(tp, predicted),
                recall=_ratio(tp, support),
                f1=_ratio(2 * tp, predicted + support),
                support=support,
                predicted=predicted,
            )
        )

    return MetricsReport(
        labels=LABELS,
        rows=rows,
        n_test=n,
        correct=correct,
        accuracy=correct / n,
        macro_f1=sum(c.f1 for c in per_class) / len(per_class),
        per_class=tuple(per_class),
        majority_baseline_accuracy=max(c.support for c in per_class) / n,
    )


def meets_acceptance(correct: int, total: int) -> bool:
    """`correct / total >= 0.85` exacto, en aritmética entera (sin floats ni redondeo)."""
    if total <= 0 or not 0 <= correct <= total:
        raise MetricsInputError(f"Conteos imposibles: correct={correct}, total={total}")
    return correct * ACCEPTANCE_ACCURACY.denominator >= ACCEPTANCE_ACCURACY.numerator * total
