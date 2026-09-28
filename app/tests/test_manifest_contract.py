"""D02-04 — `presentation.contracts.ManifestSummary` es el espejo en Python de
`manifestSummarySchema` (`backend/src/logic/p3.contracts.ts`, D01-05). Estos tests
corren el modelo de Python contra los MISMOS fixtures reales que ya usan
`backend/tests/p3-contracts.test.ts` y `frontend/tests/p3-contracts.test.ts`
(`contracts/p3/fixtures/manifest_summary/*.json`) — no fixtures propios: si el
esquema de Python diverge del de TypeScript, esta prueba lo detecta igual que las
de ellos, sin duplicar las reglas a mano dos veces."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from presentation.contracts import ManifestSummary

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = REPO_ROOT / "contracts" / "p3" / "fixtures" / "manifest_summary"

fixture_paths = sorted(FIXTURES_DIR.glob("*.json"))


@pytest.mark.skipif(
    not fixture_paths, reason="contracts/p3/fixtures/manifest_summary no está disponible"
)
@pytest.mark.parametrize("fixture_path", fixture_paths, ids=lambda p: p.stem)
def test_manifest_summary_matches_the_shared_typescript_fixture(fixture_path):
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["contract"] == "manifest_summary"

    if fixture["valid"]:
        # No debe lanzar; si el modelo de Python es más estricto que el de
        # TypeScript en algún campo, esto lo revienta aquí, no en producción.
        ManifestSummary.model_validate(fixture["payload"])
    else:
        with pytest.raises(ValidationError):
            ManifestSummary.model_validate(fixture["payload"])


@pytest.mark.skipif(
    not fixture_paths, reason="contracts/p3/fixtures/manifest_summary no está disponible"
)
@pytest.mark.parametrize("classes", [["cat", "dog", "horse"], ["cat"], ["dog", "dog"]])
def test_classes_other_than_exactly_cat_and_dog_are_rejected(classes):
    """No hay un fixture compartido para esto (los de TS no varían `classes`):
    cobertura propia del espejo de Python, no una regla nueva — #33 congela
    exactamente `cat`/`dog`."""
    base = json.loads((FIXTURES_DIR / "valid-frozen.json").read_text(encoding="utf-8"))["payload"]
    payload = {**base, "classes": classes}

    with pytest.raises(ValidationError):
        ManifestSummary.model_validate(payload)


def test_at_least_the_known_fixture_files_are_present():
    """Si alguien renombra/mueve los fixtures compartidos, esta prueba falla en vez
    de quedarse en 0 tests recolectados (ver el `skipif` de arriba)."""
    names = {p.name for p in fixture_paths}
    assert {
        "valid-frozen.json",
        "valid-not-frozen.json",
        "invalid-wrong-seed.json",
        "invalid-ratio-out-of-tolerance.json",
        "invalid-class-missing-in-test.json",
        "invalid-per-class-mismatch.json",
    } <= names
