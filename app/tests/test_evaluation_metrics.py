"""D03-05 — motor de métricas con predicciones CONOCIDAS (nunca el frozen test).

Cada caso declara predicciones controladas cat/dog y los valores esperados
calculados a mano como fracciones exactas (en el docstring o en `Fraction`), no
con el propio motor. Un segundo cálculo independiente con scikit-learn
(`labels=` explícito) cubre los mismos casos.

Orientación (rúbrica 4.2, contrato D01-05): filas = clase REAL, columnas = clase
PREDICHA, en el orden congelado (cat, dog).
"""

import json
from fractions import Fraction
from pathlib import Path

import pytest
from pydantic import ValidationError
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

from evaluation.metrics import (
    ACCEPTANCE_ACCURACY,
    LABELS,
    MetricsInputError,
    compute_metrics,
    confusion_matrix,
    meets_acceptance,
)
from presentation.contracts import EVALUATION_RESPONSE, EvaluationReady

REPO_ROOT = Path(__file__).resolve().parents[2]
VALID_READY = (
    REPO_ROOT / "contracts" / "p3" / "fixtures" / "evaluation_response" / "valid-ready.json"
)

CLOSE = 1e-12


def _from_rows(rows):
    """Predicciones controladas a partir de una matriz filas=reales/columnas=predichas."""
    y_true, y_pred = [], []
    for real, row in zip(LABELS, rows, strict=True):
        for predicted, count in zip(LABELS, row, strict=True):
            y_true += [real] * count
            y_pred += [predicted] * count
    return y_true, y_pred


def _expect_input_error(fn, *args):
    """Exige `MetricsInputError` por aserción: un `KeyError`/`ZeroDivisionError`
    accidental también "rechaza" la entrada, pero no es un rechazo explícito."""
    with pytest.raises(Exception) as caught:
        fn(*args)
    assert isinstance(caught.value, MetricsInputError), repr(caught.value)


def _by_class(report):
    return {c.class_name: c for c in report.per_class}


# --- Matriz asimétrica: detecta transposición ----------------------------------------
#
#              pred cat  pred dog
#   real cat       6         1      support cat = 7
#   real dog       4         2      support dog = 6        n = 13
#
# cat: tp=6, fp=4, fn=1 → precision 6/10, recall 6/7, F1 = 2·6/(2·6+4+1) = 12/17
# dog: tp=2, fp=1, fn=4 → precision 2/3,  recall 2/6, F1 = 2·2/(2·2+1+4) = 4/9
# accuracy = 8/13, macro-F1 = (12/17 + 4/9)/2 = 88/153, baseline = 7/13
# Transpuesta daría support cat = 10: cualquier inversión filas/columnas se nota.

ASYMMETRIC_ROWS = ((6, 1), (4, 2))


def test_rows_are_real_class_and_columns_are_predicted_class():
    # Un solo ejemplo real cat predicho como dog: debe caer en [cat][dog] = [0][1].
    assert confusion_matrix(["cat"], ["dog"]) == ((0, 1), (0, 0))
    assert confusion_matrix(["dog"], ["cat"]) == ((0, 0), (1, 0))


def test_asymmetric_matrix_matches_hand_calculation():
    report = compute_metrics(*_from_rows(ASYMMETRIC_ROWS))

    assert report.labels == ("cat", "dog")
    assert report.rows == ASYMMETRIC_ROWS
    assert report.n_test == 13
    assert report.correct == 8
    assert report.accuracy == 8 / 13
    assert report.majority_baseline_accuracy == 7 / 13
    assert report.macro_f1 == pytest.approx(float(Fraction(88, 153)), abs=CLOSE)

    cat, dog = _by_class(report)["cat"], _by_class(report)["dog"]
    assert (cat.support, cat.predicted) == (7, 10)
    assert (dog.support, dog.predicted) == (6, 3)
    assert cat.precision == pytest.approx(6 / 10, abs=CLOSE)
    assert cat.recall == pytest.approx(6 / 7, abs=CLOSE)
    assert cat.f1 == pytest.approx(float(Fraction(12, 17)), abs=CLOSE)
    assert dog.precision == pytest.approx(2 / 3, abs=CLOSE)
    assert dog.recall == pytest.approx(2 / 6, abs=CLOSE)
    assert dog.f1 == pytest.approx(float(Fraction(4, 9)), abs=CLOSE)


def test_input_order_does_not_change_label_order():
    """Aunque el primer ejemplo sea dog, la salida sigue el orden congelado cat, dog."""
    y_true, y_pred = _from_rows(ASYMMETRIC_ROWS)
    report = compute_metrics(y_true[::-1], y_pred[::-1])

    assert report.labels == ("cat", "dog")
    assert report.rows == ASYMMETRIC_ROWS
    assert [c.class_name for c in report.per_class] == ["cat", "dog"]


# --- Clases sin predicción / sin soporte ------------------------------------------------


def test_class_never_predicted_has_zero_precision_without_error():
    """real 3 cat + 2 dog, el modelo SIEMPRE dice cat.
    cat: precision 3/5, recall 1, F1 = 6/(6+2+0) = 3/4. dog: 0/0/0 (nunca predicha).
    macro-F1 = 3/8 — ambas clases cuentan, aunque dog no tenga predicciones."""
    report = compute_metrics(["cat"] * 3 + ["dog"] * 2, ["cat"] * 5)

    assert report.rows == ((3, 0), (2, 0))
    cat, dog = _by_class(report)["cat"], _by_class(report)["dog"]
    assert (dog.predicted, dog.support) == (0, 2)
    assert (dog.precision, dog.recall, dog.f1) == (0.0, 0.0, 0.0)
    assert cat.f1 == pytest.approx(3 / 4, abs=CLOSE)
    assert report.macro_f1 == pytest.approx(3 / 8, abs=CLOSE)
    assert report.accuracy == 3 / 5
    assert report.majority_baseline_accuracy == 3 / 5


def test_class_without_support_still_appears_with_zeros():
    """real 4 cat, predicho [cat, cat, dog, cat]. dog: support 0, predicha 1 vez →
    precision 0/1, recall 0 (sin soporte), F1 0. cat: precision 1, recall 3/4,
    F1 = 6/(6+0+1) = 6/7. macro-F1 = 3/7. baseline = 4/4."""
    report = compute_metrics(["cat"] * 4, ["cat", "cat", "dog", "cat"])

    assert report.rows == ((3, 1), (0, 0))
    assert [c.class_name for c in report.per_class] == ["cat", "dog"]
    cat, dog = _by_class(report)["cat"], _by_class(report)["dog"]
    assert (dog.support, dog.predicted) == (0, 1)
    assert (dog.precision, dog.recall, dog.f1) == (0.0, 0.0, 0.0)
    assert cat.f1 == pytest.approx(6 / 7, abs=CLOSE)
    assert report.macro_f1 == pytest.approx(3 / 7, abs=CLOSE)
    assert report.majority_baseline_accuracy == 1.0


# --- Totales y soporte --------------------------------------------------------------


@pytest.mark.parametrize(
    "rows", [ASYMMETRIC_ROWS, ((31, 3), (4, 28)), ((0, 5), (7, 0)), ((10, 0), (0, 1))]
)
def test_totals_support_and_predicted_are_coherent(rows):
    report = compute_metrics(*_from_rows(rows))

    assert sum(map(sum, report.rows)) == report.n_test
    for i, stats in enumerate(report.per_class):
        assert stats.support == sum(report.rows[i])
        assert stats.predicted == sum(row[i] for row in report.rows)
    assert sum(c.support for c in report.per_class) == report.n_test
    assert report.correct == sum(report.rows[i][i] for i in range(len(LABELS)))
    assert report.accuracy == report.correct / report.n_test
    assert report.majority_baseline_accuracy == max(c.support for c in report.per_class) / (
        report.n_test
    )


@pytest.mark.parametrize(
    "rows", [ASYMMETRIC_ROWS, ((31, 3), (4, 28)), ((0, 5), (7, 0)), ((3, 0), (2, 0))]
)
def test_independent_recalculation_with_scikit_learn(rows):
    """Segundo cálculo, independiente del motor, con `labels=` explícito: sin él,
    scikit-learn ignora una clase ausente y el macro-F1 se calcula sobre menos clases."""
    y_true, y_pred = _from_rows(rows)
    report = compute_metrics(y_true, y_pred)

    expected_rows = sk_confusion_matrix(y_true, y_pred, labels=list(LABELS)).tolist()
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(LABELS), zero_division=0
    )
    assert [list(r) for r in report.rows] == expected_rows
    for i, stats in enumerate(report.per_class):
        assert stats.precision == pytest.approx(precision[i], abs=CLOSE)
        assert stats.recall == pytest.approx(recall[i], abs=CLOSE)
        assert stats.f1 == pytest.approx(f1[i], abs=CLOSE)
        assert stats.support == support[i]
    assert report.macro_f1 == pytest.approx(f1.mean(), abs=CLOSE)


# --- Umbral 0.85 sin redondeo -------------------------------------------------------


def test_acceptance_threshold_is_exactly_0_85():
    assert Fraction(85, 100) == ACCEPTANCE_ACCURACY


@pytest.mark.parametrize(
    ("correct", "total", "expected"),
    [
        (849, 1000, False),  # inmediatamente debajo
        (850, 1000, True),  # exactamente en
        (851, 1000, True),  # inmediatamente encima
        (8499, 10000, False),  # 0.8499 → "0.85" a 2 decimales: NO cuenta
        (84999999, 100000000, False),  # a 1e-8 del umbral
        (17, 20, True),  # 0.85 exacto con otro denominador
        (11, 13, False),  # 0.8461… → "0.85" redondeado a 2 decimales
        (6, 7, True),  # 0.857…
    ],
)
def test_acceptance_compares_without_rounding(correct, total, expected):
    assert meets_acceptance(correct, total) is expected


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (((17, 0), (3, 0)), True),  # 17/20 = 0.85 exacto
        (((16, 1), (2, 1)), True),  # 17/20 con errores en ambas clases
        (((16, 1), (3, 0)), False),  # 16/20 = 0.80
        (((11, 0), (2, 0)), False),  # 11/13 = 0.846…
    ],
)
def test_report_acceptance_uses_counts_not_the_rounded_accuracy(rows, expected):
    assert compute_metrics(*_from_rows(rows)).meets_acceptance() is expected


@pytest.mark.parametrize(("correct", "total"), [(1, 0), (-1, 10), (11, 10), (0, -5)])
def test_acceptance_rejects_impossible_counts(correct, total):
    _expect_input_error(meets_acceptance, correct, total)


# --- Entradas incompatibles -----------------------------------------------------------


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        pytest.param(["cat", "dog"], ["cat"], id="longitudes-distintas"),
        pytest.param([], [], id="vacio"),
        pytest.param(["cat", "horse"], ["cat", "dog"], id="clase-real-desconocida"),
        pytest.param(["cat", "dog"], ["cat", "bird"], id="clase-predicha-desconocida"),
        pytest.param([0, 1], [0, 1], id="indices-en-vez-de-nombres"),
        pytest.param(["Cat", "dog"], ["cat", "dog"], id="mayusculas"),
        pytest.param(["cat", None], ["cat", "dog"], id="none"),
        pytest.param("catdog", "catdog", id="string-no-secuencia"),
    ],
)
def test_incompatible_inputs_are_rejected(y_true, y_pred):
    _expect_input_error(compute_metrics, y_true, y_pred)
    _expect_input_error(confusion_matrix, y_true, y_pred)


# --- Salida contractual consumible por la pantalla Evaluation / D06-01 -------------------

SELECTION = {
    "namespace": "official",
    "candidate_run_id": "a" * 32,
    "closed_at": "2026-09-28T20:00:00Z",
    "manifest_hash": "d" * 64,
    "evaluated_at": "2026-09-29T15:00:00Z",
}


def test_shared_ready_fixture_is_reproduced_from_its_predictions():
    """Las predicciones que implica el fixture válido compartido reproducen su matriz,
    support, accuracy exacta, precision/recall y baseline; el F1 del fixture viene a 4
    decimales (dentro de la tolerancia 1e-4 del contrato)."""
    fixture = json.loads(VALID_READY.read_text(encoding="utf-8"))["payload"]
    rows = tuple(tuple(r) for r in fixture["confusion_matrix"]["rows"])

    try:
        response = compute_metrics(*_from_rows(rows)).to_ready_response(**SELECTION)
    except ValidationError as error:
        pytest.fail(f"El contrato rechazó la salida del motor: {error}")
    produced = response.model_dump()

    assert produced["confusion_matrix"] == fixture["confusion_matrix"]
    assert produced["n_test"] == fixture["n_test"]
    assert produced["metrics"]["accuracy"] == fixture["metrics"]["accuracy"]
    assert produced["majority_baseline_accuracy"] == fixture["majority_baseline_accuracy"]
    for got, want in zip(
        produced["metrics"]["per_class"], fixture["metrics"]["per_class"], strict=True
    ):
        assert got["class_name"] == want["class_name"]
        assert got["support"] == want["support"]
        assert got["precision"] == pytest.approx(want["precision"], abs=CLOSE)
        assert got["recall"] == pytest.approx(want["recall"], abs=CLOSE)
        assert got["f1"] == pytest.approx(want["f1"], abs=1e-4)
    assert produced["metrics"]["macro_f1"] == pytest.approx(
        fixture["metrics"]["macro_f1"], abs=1e-4
    )


@pytest.mark.parametrize(
    "rows", [ASYMMETRIC_ROWS, ((3, 0), (2, 0)), ((3, 1), (0, 0)), ((0, 5), (7, 0))]
)
def test_ready_response_validates_against_the_shared_contract(rows):
    try:
        response = compute_metrics(*_from_rows(rows)).to_ready_response(**SELECTION)
    except ValidationError as error:
        pytest.fail(f"El contrato rechazó la salida del motor: {error}")

    assert isinstance(response, EvaluationReady)
    assert response.state == "ready"
    assert response.selection.metric == "val_accuracy"
    assert response.classes == ["cat", "dog"]
    # Ida y vuelta por JSON, como lo recibirá el backend.
    EVALUATION_RESPONSE.validate_json(response.model_dump_json())


def test_metrics_are_reported_without_rounding():
    """El contrato tolera 1e-4, pero el motor no redondea: 4/9 sale completo."""
    response = compute_metrics(*_from_rows(ASYMMETRIC_ROWS)).to_ready_response(**SELECTION)
    dog = response.metrics.per_class[1]

    assert dog.f1 == pytest.approx(float(Fraction(4, 9)), abs=CLOSE)
    assert dog.f1 != round(dog.f1, 4)


def test_ready_response_refuses_evaluation_before_selection_closed():
    report = compute_metrics(*_from_rows(ASYMMETRIC_ROWS))
    with pytest.raises(ValueError, match="MODEL SELECTION CLOSED"):
        report.to_ready_response(**{**SELECTION, "evaluated_at": "2026-09-28T19:59:59Z"})


# --- Aislamiento: el motor nunca abre el frozen test -----------------------------------


def test_metrics_engine_has_no_io_or_manifest_access():
    """El motor recibe etiquetas ya predichas; no lee archivos, manifest ni DVC. Abrir
    el test oficial es de D06-01, después de MODEL SELECTION CLOSED (#33)."""
    source = (REPO_ROOT / "app" / "evaluation" / "metrics.py").read_text(encoding="utf-8")
    for banned in ("open(", "read_text", "Path(", "manifest", "dvc", "torch", "mlflow"):
        assert banned not in source.lower().replace("manifest_classes", "").replace(
            "manifest_hash", ""
        ), banned
