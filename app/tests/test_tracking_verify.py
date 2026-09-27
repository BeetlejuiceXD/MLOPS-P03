"""D02-01 — `tracking.verify`: persistencia de MLflow sin falsos positivos.

Estos tests corren sin servidor: usan el store de archivos local de MLflow para la
lógica de escritura/comprobación, y un puerto cerrado para "servidor no disponible".
La prueba con los servicios reales (MariaDB + MinIO + reinicio) la hace el job
`MLflow persistente` de `.github/workflows/ci.yml`.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from tracking import verify
from tracking.settings import P3_EXPERIMENT


def _closed_port_uri() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"  # el socket ya se cerró: nadie escucha ahí


@pytest.fixture
def file_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    uri = (tmp_path / "mlruns").as_uri()
    # Solo para tests: el proyecto usa el servidor con MariaDB, no este store.
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    return uri


def _write(tmp_path: Path) -> tuple[Path, dict]:
    evidence_path = tmp_path / "evidence.json"
    assert verify.main(["write", "--evidence", str(evidence_path)]) == verify.VERIFIED
    return evidence_path, json.loads(evidence_path.read_text(encoding="utf-8"))


def _client(uri: str):
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=uri)


def test_write_then_check_round_trip(file_store, tmp_path, capsys):
    evidence_path, evidence = _write(tmp_path)

    assert evidence["experiment_name"] == P3_EXPERIMENT
    assert len(evidence["run_id"]) == 32
    assert len(evidence["artifact_sha256"]) == 64
    assert evidence["metric_history"] == list(verify.METRIC_STEPS)

    assert verify.main(["check", "--evidence", str(evidence_path)]) == verify.VERIFIED
    assert evidence["run_id"] in capsys.readouterr().out


def test_verification_run_is_tagged_out_of_the_campaign(file_store, tmp_path):
    _, evidence = _write(tmp_path)
    run = _client(file_store).get_run(evidence["run_id"])
    assert run.data.tags[verify.RUN_KIND_TAG] == verify.RUN_KIND
    assert run.info.status == "FINISHED"


def test_reuses_the_existing_experiment(file_store, tmp_path):
    _, first = _write(tmp_path)
    _, second = _write(tmp_path)
    assert first["experiment_id"] == second["experiment_id"]
    assert first["run_id"] != second["run_id"]


def test_changed_artifact_is_detected(file_store, tmp_path, capsys):
    evidence_path, evidence = _write(tmp_path)
    run = _client(file_store).get_run(evidence["run_id"])
    stored = Path(run.info.artifact_uri.removeprefix("file://")) / evidence["artifact_path"]
    stored.write_text("contenido alterado\n", encoding="utf-8")

    assert verify.main(["check", "--evidence", str(evidence_path)]) == verify.MISMATCH
    assert "SHA-256" in capsys.readouterr().err


def test_changed_metric_history_is_detected(file_store, tmp_path, capsys):
    evidence_path, evidence = _write(tmp_path)
    _client(file_store).log_metric(evidence["run_id"], verify.METRIC, 0.99, step=3)

    assert verify.main(["check", "--evidence", str(evidence_path)]) == verify.MISMATCH
    assert "Historial" in capsys.readouterr().err


def test_missing_run_is_detected(file_store, tmp_path, capsys):
    evidence_path, evidence = _write(tmp_path)
    evidence["run_id"] = "0" * 32
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    assert verify.main(["check", "--evidence", str(evidence_path)]) == verify.MISMATCH
    assert "ya no existe" in capsys.readouterr().err


def test_unavailable_server_never_reports_success(tmp_path, monkeypatch, capsys):
    """Servicio caído: sale con 2, no escribe evidencia y no imprime éxito."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", _closed_port_uri())
    evidence_path = tmp_path / "evidence.json"

    assert verify.main(["write", "--evidence", str(evidence_path)]) == verify.UNAVAILABLE
    assert not evidence_path.exists()

    evidence_path.write_text(
        json.dumps(
            {
                "tracking_uri": "x",
                "experiment_name": P3_EXPERIMENT,
                "experiment_id": "1",
                "run_id": "a" * 32,
                "metric": verify.METRIC,
                "metric_history": list(verify.METRIC_STEPS),
                "artifact_path": "persistence-check/evidence.txt",
                "artifact_sha256": "b" * 64,
                "written_at": "2026-09-27T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    assert verify.main(["check", "--evidence", str(evidence_path)]) == verify.UNAVAILABLE
    captured = capsys.readouterr()
    assert "no disponible" in captured.err
    assert "verificada" not in captured.out


def test_invalid_backend_is_unavailable_not_a_crash(tmp_path, monkeypatch, capsys):
    """Un backend que MLflow rechaza al conectar sale con 2, sin traceback."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    monkeypatch.delenv("MLFLOW_ALLOW_FILE_STORE", raising=False)

    assert verify.main(["write", "--evidence", str(tmp_path / "e.json")]) == verify.UNAVAILABLE
    assert "no disponible" in capsys.readouterr().err


def test_missing_tracking_uri_is_a_configuration_error(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert verify.main(["write", "--evidence", str(tmp_path / "e.json")]) == verify.UNAVAILABLE
    assert "MLFLOW_TRACKING_URI" in capsys.readouterr().err


def test_unreadable_evidence_is_not_success(file_store, tmp_path, capsys):
    broken = tmp_path / "broken.json"
    broken.write_text("{no es json", encoding="utf-8")
    assert verify.main(["check", "--evidence", str(broken)]) == verify.UNAVAILABLE
    assert "ilegible" in capsys.readouterr().err
