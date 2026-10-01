"""D03-05 — `presentation.contracts.EVALUATION_RESPONSE` es el espejo en Python de
`evaluationResponseSchema` (`backend/src/logic/p3.contracts.ts`, D01-05). Igual que
`test_manifest_contract.py`: se valida contra los MISMOS fixtures que usan
`backend/tests/p3-contracts.test.ts` y `frontend/tests/p3-contracts.test.ts`
(`contracts/p3/fixtures/evaluation_response/*.json`), no contra fixtures propios.

Así, lo que produce el motor de métricas (`evaluation.metrics`) se comprueba con las
mismas reglas que aplicarán el backend y la pantalla Evaluation."""

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from presentation.contracts import EVALUATION_RESPONSE

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = REPO_ROOT / "contracts" / "p3" / "fixtures" / "evaluation_response"

fixture_paths = sorted(FIXTURES_DIR.glob("*.json"))


def _payload(name: str) -> dict:
    return json.loads((FIXTURES_DIR / name).read_text(encoding="utf-8"))["payload"]


@pytest.mark.skipif(
    not fixture_paths, reason="contracts/p3/fixtures/evaluation_response no está disponible"
)
@pytest.mark.parametrize("fixture_path", fixture_paths, ids=lambda p: p.stem)
def test_evaluation_response_matches_the_shared_typescript_fixture(fixture_path):
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["contract"] == "evaluation_response"

    if fixture["valid"]:
        EVALUATION_RESPONSE.validate_python(fixture["payload"])
    else:
        with pytest.raises(ValidationError):
            EVALUATION_RESPONSE.validate_python(fixture["payload"])


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
    "mutate",
    [
        pytest.param(_set_cat("precision", 0.9117647058823529), id="precision-de-otra-clase"),
        pytest.param(_set_cat("recall", 0.8857142857142857), id="recall-intercambiado"),
        pytest.param(_set_cat("f1", 0.8988), id="f1-fuera-de-1e-4"),
        pytest.param(lambda p: p["metrics"].update(macro_f1=0.8939), id="macro-f1-no-promedio"),
        pytest.param(
            lambda p: p.update(majority_baseline_accuracy=0.5), id="baseline-no-mayoritaria"
        ),
        pytest.param(
            lambda p: p["metrics"].update(per_class=p["metrics"]["per_class"][:1]),
            id="falta-una-clase",
        ),
        pytest.param(
            lambda p: p["confusion_matrix"].update(rows=[[31, 3, 0], [4, 28, 0]]),
            id="matriz-no-cuadrada",
        ),
        pytest.param(lambda p: p.update(classes=["cat"]), id="clases-incompletas"),
        pytest.param(lambda p: p.update(classes=["cat", "dog", "dog"]), id="clase-repetida"),
        pytest.param(lambda p: p.update(evaluated_at="2026-09-29 15:00"), id="fecha-sin-zona"),
        pytest.param(
            lambda p: p.update(evaluated_at=p["selection"]["closed_at"]), id="evaluado-al-cierre"
        ),
        pytest.param(lambda p: p.update(extra_metric=1.0), id="campo-extra"),
    ],
)
def test_ready_payload_that_breaks_a_typescript_rule_is_rejected(mutate):
    with pytest.raises(ValidationError):
        EVALUATION_RESPONSE.validate_python(_mutated(mutate))


def test_reported_metrics_within_1e_4_are_accepted_like_in_typescript():
    """El fixture válido reporta F1 a 4 decimales: la tolerancia 1e-4 es del contrato,
    no un redondeo que haga el motor (el motor reporta sin redondear)."""
    payload = _mutated(_set_cat("f1", 0.89855))
    EVALUATION_RESPONSE.validate_python(payload)
