"""D04-05 — `presentation.contracts.EvaluationPredictions` es el espejo en Python de
`evaluationPredictionsSchema` (`backend/src/logic/p3.contracts.ts`). Se valida contra
los MISMOS fixtures que usan los tests de contratos del backend y del frontend
(`contracts/p3/fixtures/evaluation_predictions/*.json`)."""

import json
import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from presentation.contracts import EvaluationPredictions

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = REPO_ROOT / "contracts" / "p3" / "fixtures" / "evaluation_predictions"
fixture_paths = sorted(FIXTURES_DIR.glob("*.json"))

# Motivo esperado de cada fixture inválido (mensaje de la regla que lo rechaza).
FIXTURE_REASONS = {
    "invalid-crop-id-not-increasing": "orden estrictamente creciente",
    "invalid-missing-class-probability": "una probabilidad por clase declarada",
    "invalid-n-test-mismatch": "exactamente n_test predicciones",
    "invalid-predicted-not-argmax": "argmax",
    "invalid-probabilities-sum": "deben sumar ~1",
    "invalid-unknown-namespace": "'official' or 'synthetic'",
}


def test_every_invalid_fixture_has_its_expected_reason():
    invalid = {p.stem for p in fixture_paths if p.stem.startswith("invalid-")}
    assert invalid == set(FIXTURE_REASONS)


@pytest.mark.parametrize("fixture_path", fixture_paths, ids=lambda p: p.stem)
def test_evaluation_predictions_matches_the_shared_typescript_fixture(fixture_path):
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["contract"] == "evaluation_predictions"
    if fixture["valid"]:
        try:
            EvaluationPredictions.model_validate(fixture["payload"])
        except ValidationError as error:
            pytest.fail(f"Payload válido rechazado: {error}")
    else:
        with pytest.raises(ValidationError, match=re.escape(FIXTURE_REASONS[fixture_path.stem])):
            EvaluationPredictions.model_validate(fixture["payload"])


def test_probability_of_an_undeclared_class_is_rejected():
    payload = json.loads((FIXTURES_DIR / "valid-synthetic.json").read_text("utf-8"))["payload"]
    payload["predictions"][0]["probabilities"] = {"cat": 0.9, "dog": 0.1, "bird": 0.0}
    with pytest.raises(ValidationError):
        EvaluationPredictions.model_validate(payload)
