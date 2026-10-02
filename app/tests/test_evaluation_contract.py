"""D03-05 — `presentation.contracts.EVALUATION_RESPONSE` es el espejo en Python de
`evaluationResponseSchema` (`backend/src/logic/p3.contracts.ts`, D01-05). Igual que
`test_manifest_contract.py`: se valida contra los MISMOS fixtures que usan
`backend/tests/p3-contracts.test.ts` y `frontend/tests/p3-contracts.test.ts`
(`contracts/p3/fixtures/evaluation_response/*.json`), no contra fixtures propios.

Así, lo que produce el motor de métricas (`evaluation.metrics`) se comprueba con las
mismas reglas que aplicarán el backend y la pantalla Evaluation."""

import copy
import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from presentation.contracts import EVALUATION_RESPONSE

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = REPO_ROOT / "contracts" / "p3" / "fixtures" / "evaluation_response"

fixture_paths = sorted(FIXTURES_DIR.glob("*.json"))


def _payload(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))["payload"]


def _assert_valid(payload: dict) -> None:
    """Un payload válido rechazado es un veredicto del test, no un crash."""
    try:
        EVALUATION_RESPONSE.validate_python(payload)
    except ValidationError as error:
        pytest.fail(f"Payload válido rechazado: {error}")


def _assert_rejected_for(payload: dict, reason: str) -> None:
    """Exige el rechazo POR LA REGLA indicada: si otra regla lo rechaza primero (p. ej.
    un caso que además rompe el macro-F1), el test no demostraría que esta regla
    existe."""
    with pytest.raises(ValidationError, match=re.escape(reason)):
        EVALUATION_RESPONSE.validate_python(payload)


# Motivo esperado de cada fixture inválido compartido (mensaje de la regla).
FIXTURE_REASONS = {
    "invalid-accuracy-mismatch": "accuracy debe ser traza / n_test",
    "invalid-evaluated-before-closed": "MODEL SELECTION CLOSED",
    "invalid-matrix-sum-mismatch": "La matriz debe sumar n_test",
    "invalid-selected-by-test": "'val_accuracy'",
    "invalid-support-mismatch": "support de cat",
    # D05-05: namespace visible y estado `pending` (cerrada, sin resultado).
    "invalid-ready-without-namespace": "namespace",
    "invalid-ready-local-test-namespace": "'synthetic'",
    "invalid-pending-with-results": "confusion_matrix",
    "invalid-pending-selected-by-test": "'val_accuracy'",
}


@pytest.mark.skipif(
    not fixture_paths, reason="contracts/p3/fixtures/evaluation_response no está disponible"
)
@pytest.mark.parametrize("fixture_path", fixture_paths, ids=lambda p: p.stem)
def test_evaluation_response_matches_the_shared_typescript_fixture(fixture_path):
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["contract"] == "evaluation_response"

    if fixture["valid"]:
        _assert_valid(fixture["payload"])
    else:
        # Un fixture nuevo sin motivo registrado solo exige el rechazo.
        _assert_rejected_for(fixture["payload"], FIXTURE_REASONS.get(fixture_path.stem, ""))


def test_at_least_the_known_fixture_files_are_present():
    """Si alguien mueve los fixtures, esto falla en vez de recolectar 0 tests."""
    assert {p.name for p in fixture_paths} >= {
        "valid-ready.json",
        "valid-blocked.json",
        "invalid-accuracy-mismatch.json",
        "invalid-evaluated-before-closed.json",
        "invalid-matrix-sum-mismatch.json",
        "invalid-selected-by-test.json",
        "invalid-support-mismatch.json",
        "valid-ready-synthetic.json",
        "valid-pending.json",
        "invalid-ready-without-namespace.json",
        "invalid-ready-local-test-namespace.json",
        "invalid-pending-with-results.json",
        "invalid-pending-selected-by-test.json",
    }


# Reglas de `evaluationReadySchema` que los fixtures compartidos no ejercitan. No son
# reglas nuevas: cada caso replica una rama del `superRefine` de TypeScript.


def _mutated(mutate) -> dict:
    payload = copy.deepcopy(_payload("valid-ready.json"))
    mutate(payload)
    return payload


def _set_cat(field, value):
    def mutate(p):
        p["metrics"]["per_class"][0][field] = value

    return mutate


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        pytest.param(
            _set_cat("precision", 0.9117647058823529),
            "precision/recall/F1 de cat",
            id="precision-de-otra-clase",
        ),
        pytest.param(
            _set_cat("recall", 0.8857142857142857),
            "precision/recall/F1 de cat",
            id="recall-intercambiado",
        ),
        pytest.param(_set_cat("f1", 0.8988), "precision/recall/F1 de cat", id="f1-fuera-de-1e-4"),
        pytest.param(
            lambda p: p["metrics"].update(macro_f1=0.8939),
            "macro_f1 debe ser el promedio",
            id="macro-f1-no-promedio",
        ),
        pytest.param(
            lambda p: p.update(majority_baseline_accuracy=0.5),
            "Baseline = soporte de la clase mayoritaria",
            id="baseline-no-mayoritaria",
        ),
        pytest.param(
            lambda p: p["metrics"].update(per_class=p["metrics"]["per_class"][:1]),
            "Faltan métricas de la clase dog",
            id="falta-una-clase",
        ),
        pytest.param(
            lambda p: p["confusion_matrix"].update(rows=[[31, 3, 0], [4, 28, 0]]),
            "La matriz debe ser cuadrada",
            id="matriz-no-cuadrada",
        ),
        pytest.param(
            lambda p: p.update(classes=["cat"]),
            "classes debe declarar exactamente",
            id="clases-incompletas",
        ),
        pytest.param(
            lambda p: p.update(classes=["cat", "dog", "dog"]),
            "classes debe declarar exactamente",
            id="clase-repetida",
        ),
        pytest.param(
            lambda p: p["confusion_matrix"].update(labels=["dog", "dog"]),
            "confusion_matrix.labels debe declarar exactamente",
            id="labels-repetidas",
        ),
        pytest.param(
            lambda p: p.update(evaluated_at="2026-09-29 15:00"),
            "String should match pattern",
            id="fecha-sin-zona",
        ),
        pytest.param(
            lambda p: p.update(evaluated_at=p["selection"]["closed_at"]),
            "MODEL SELECTION CLOSED",
            id="evaluado-al-cierre",
        ),
        pytest.param(
            lambda p: p.update(extra_metric=1.0),
            "Extra inputs are not permitted",
            id="campo-extra",
        ),
    ],
)
def test_ready_payload_that_breaks_a_typescript_rule_is_rejected(mutate, reason):
    _assert_rejected_for(_mutated(mutate), reason)


def test_n_test_that_does_not_match_the_matrix_is_rejected():
    """Solo la regla "la matriz suma n_test" atrapa esto: accuracy, support, baseline y
    métricas por clase son coherentes con n_test=67. El fixture compartido
    `invalid-matrix-sum-mismatch` además rompe el support de dog, así que por sí solo
    no demuestra que esa regla exista."""

    def mutate(p):
        p["n_test"] = 67
        p["metrics"]["accuracy"] = 59 / 67
        p["majority_baseline_accuracy"] = 34 / 67

    _assert_valid(_mutated(lambda p: None))  # control: la base es válida
    _assert_rejected_for(_mutated(mutate), "La matriz debe sumar n_test")


def test_reported_metrics_within_1e_4_are_accepted_like_in_typescript():
    """El fixture válido reporta F1 a 4 decimales: la tolerancia 1e-4 es del contrato,
    no un redondeo que haga el motor (el motor reporta sin redondear)."""
    _assert_valid(_mutated(_set_cat("f1", 0.89855)))
