"""D06-02 — `build_official_package`: cruce acta/D06-01 antes de empaquetar,
tarjeta oficial con métricas reales o pendiente explícito, y roundtrip con
el loader de D05-01 sin tocarlo.

Pesos de FIXTURE (reutiliza `checkpoint` de `tests.test_model_package`,
mismo patrón que `test_smoke.py`): esto prueba el mecanismo, no acredita
el paquete final del candidato real — eso se evidencia aparte cuando
D06-01 entregue resultado real del mismo run/checkpoint del acta.

No repite las negativas internas del loader (pesos/hash/preprocessing/
clase/provenance ya cubiertas en `test_model_package.py`) — solo lo nuevo
de esta capa: el cruce acta↔métricas, el estado pendiente, y la
integración con el artefacto final.
"""

from __future__ import annotations

import json

import pytest

from model_package import load_package
from model_package.format import PackageError
from model_package.official import (
    LOW_ACCURACY_THRESHOLD,
    OfficialRecord,
    OfficialTestMetrics,
    build_official_package,
)
from model_package.version import ModelVersion
from tests.test_model_package import RUN_ID, _sha256, checkpoint  # noqa: F401

EXPERIMENT = "p3-cnn-classifier"


def _record(checkpoint_dir) -> OfficialRecord:
    return OfficialRecord(run_id=RUN_ID, checkpoint_sha256=_sha256(checkpoint_dir / "model.pt"))


def _metrics(run_id: str = RUN_ID, accuracy: float = 0.92) -> OfficialTestMetrics:
    return OfficialTestMetrics(
        run_id=run_id, accuracy=accuracy, macro_f1=0.91, loss=0.2, test_split_hash="f" * 64
    )


def _card(out) -> dict:
    return json.loads((out / "smoke_card.json").read_text(encoding="utf-8"))


# --- Positivo: roundtrip con el loader de D05-01, sin tocarlo -------------------


def test_official_package_roundtrips_with_the_d05_01_loader(checkpoint, tmp_path):  # noqa: F811
    out = tmp_path / "official"

    manifest = build_official_package(
        checkpoint,
        out,
        experiment=EXPERIMENT,
        record=_record(checkpoint),
        semver=ModelVersion(1, 0, 0),
        best_epoch=3,
        metrics=_metrics(),
    )

    assert manifest.kind == "official"
    loaded = load_package(out)
    assert loaded.identity()["mlflow_run_id"] == RUN_ID


def test_official_card_matches_record_and_metrics(checkpoint, tmp_path):  # noqa: F811
    out = tmp_path / "official"

    build_official_package(
        checkpoint,
        out,
        experiment=EXPERIMENT,
        record=_record(checkpoint),
        semver=ModelVersion(2, 1, 0),
        best_epoch=5,
        metrics=_metrics(accuracy=0.93),
    )

    card = _card(out)
    assert card["kind"] == "official"
    assert card["model_version"] == "2.1.0"
    assert card["mlflow_run_id"] == RUN_ID
    assert card["official_test_metrics"]["accuracy"] == 0.93
    assert card["pending_official_results"] is False


# --- Negativos: el cruce acta <-> D06-01, antes de tocar el checkpoint ----------


def test_metrics_from_another_run_block_packaging_before_touching_checkpoint(
    checkpoint,  # noqa: F811
    tmp_path,
):
    out = tmp_path / "official"
    metrics_from_another_run = _metrics(run_id="f" * 32)

    with pytest.raises(PackageError, match="otro checkpoint"):
        build_official_package(
            checkpoint,
            out,
            experiment=EXPERIMENT,
            record=_record(checkpoint),
            semver=ModelVersion(1, 0, 0),
            metrics=metrics_from_another_run,
        )
    assert not out.exists()


def test_record_with_wrong_checkpoint_sha_is_rejected(checkpoint, tmp_path):  # noqa: F811
    out = tmp_path / "official"
    wrong_record = OfficialRecord(run_id=RUN_ID, checkpoint_sha256="0" * 64)

    with pytest.raises(PackageError):
        build_official_package(
            checkpoint,
            out,
            experiment=EXPERIMENT,
            record=wrong_record,
            semver=ModelVersion(1, 0, 0),
            metrics=_metrics(),
        )
    assert not out.exists()


def test_wrong_experiment_is_rejected(checkpoint, tmp_path):  # noqa: F811
    out = tmp_path / "official"

    with pytest.raises(PackageError):
        build_official_package(
            checkpoint,
            out,
            experiment="otro-experimento",
            record=_record(checkpoint),
            semver=ModelVersion(1, 0, 0),
            metrics=_metrics(),
        )


# --- Sin resultado de D06-01: pendiente explícito, nunca inventado -------------


def test_without_metrics_the_card_is_explicitly_pending(checkpoint, tmp_path):  # noqa: F811
    out = tmp_path / "official"

    build_official_package(
        checkpoint,
        out,
        experiment=EXPERIMENT,
        record=_record(checkpoint),
        semver=ModelVersion(0, 1, 0),
        metrics=None,
    )

    card = _card(out)
    assert card["pending_official_results"] is True
    assert card["official_test_metrics"] is None
    assert "PLANTILLA" in card["statement"]


# --- Umbral de accuracy declarado en la tarjeta --------------------------------


def test_low_accuracy_is_disclosed_in_the_card(checkpoint, tmp_path):  # noqa: F811
    out = tmp_path / "official"

    build_official_package(
        checkpoint,
        out,
        experiment=EXPERIMENT,
        record=_record(checkpoint),
        semver=ModelVersion(1, 0, 0),
        metrics=_metrics(accuracy=0.80),
    )

    assert _card(out)["low_accuracy_disclosed"] is True


def test_accuracy_at_threshold_is_not_flagged_as_low(checkpoint, tmp_path):  # noqa: F811
    out = tmp_path / "official"

    build_official_package(
        checkpoint,
        out,
        experiment=EXPERIMENT,
        record=_record(checkpoint),
        semver=ModelVersion(1, 0, 0),
        metrics=_metrics(accuracy=LOW_ACCURACY_THRESHOLD),
    )

    assert _card(out)["low_accuracy_disclosed"] is False


# --- ModelVersion ---------------------------------------------------------------


def test_model_version_str_and_parse_roundtrip():
    version = ModelVersion(1, 2, 3)

    assert str(version) == "1.2.3"
    assert ModelVersion.parse("1.2.3") == version


def test_model_version_rejects_invalid_string():
    with pytest.raises(ValueError, match="semver"):
        ModelVersion.parse("not-a-version")


def test_model_version_rejects_negative_components():
    with pytest.raises(ValueError):
        ModelVersion(major=-1, minor=0, patch=0)
