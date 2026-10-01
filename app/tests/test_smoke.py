"""D03-04 — Verificador del smoke real Training → MLflow → checkpoint.

`tracking.smoke.verify_smoke` contrasta un job terminado (como lo devuelve la API)
con su run en MLflow y su checkpoint descargado del servidor: estado, experimento,
config, procedencia, curvas/resumen, ausencia de métricas de test, hash y carga
del modelo con su config, class map y preprocessing.

Aquí el job y el run los produce el worker REAL sobre el release/manifest sintético
de `test_training_sources.py` (mismo arnés que `test_trainer_worker_training.py`);
cada test negativo altera una sola cosa. El smoke real contra Compose y v0.1.1 se
evidencia en el PR, no con estos fixtures.
"""

from __future__ import annotations

import io
import json

import pytest
import torch
from sqlalchemy import select

from tests.test_trainer_worker_training import (  # noqa: F401  (fixtures)
    CONFIG,
    _client,
    _insert,
    _worker,
    engine,
    frozen,
    job_store,
    tracking,
)
from tracking.smoke import verify_smoke
from trainer_worker.store import training_jobs
from training.config import TrainingConfig
from training.model import build_model


def _job(db, job_id) -> dict:
    """El job como lo serializa `GET /api/training/jobs/:id` (contrato training_job)."""
    with db.connect() as conn:
        row = conn.execute(select(training_jobs).where(training_jobs.c.id == job_id)).one()
    return {
        "id": row.id,
        "task": row.task,
        "status": row.status,
        "dataset_version": row.dataset_version,
        "manifest_hash": row.manifest_hash,
        "config": json.loads(row.config),
        "progress": {"epoch": row.progress_epoch, "total_epochs": row.total_epochs},
        "mlflow_run_id": row.mlflow_run_id,
        "error": row.error,
    }


@pytest.fixture
def trained(engine, job_store, tracking, frozen):  # noqa: F811
    doc, verify = frozen
    job_id = _insert(engine, doc)
    assert _worker(job_store, tracking, verify).run_once() == "succeeded"
    return _job(engine, job_id), tracking


def _replace_checkpoint(client, run_id, state_dict, tmp_path, *, retag: bool):
    """Sube otro model.pt encima del del run (artefacto incorrecto en el servidor)."""
    folder = tmp_path / "checkpoint"
    folder.mkdir(exist_ok=True)
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    (folder / "model.pt").write_bytes(buffer.getvalue())
    client.log_artifact(run_id, str(folder / "model.pt"), "checkpoint")
    if retag:
        import hashlib

        client.set_tag(run_id, "checkpoint_sha256", hashlib.sha256(buffer.getvalue()).hexdigest())


def test_real_job_run_and_checkpoint_verify_clean(trained, tmp_path):
    job, uri = trained

    report = verify_smoke(job, tracking_uri=uri, workdir=tmp_path)

    assert report.problems == []
    assert report.run_id == job["mlflow_run_id"]
    assert report.run_status == "FINISHED"
    assert report.epochs_logged == job["progress"]["epoch"]
    run_tags = _client(uri).get_run(job["mlflow_run_id"]).data.tags
    assert report.checkpoint_sha256 == run_tags["checkpoint_sha256"]
    assert report.manifest_hash == job["manifest_hash"]
    # La carga real: el modelo de la config acepta el state_dict y clasifica una imagen
    # preprocesada con el transform de evaluación en exactamente las clases del class map.
    assert report.loaded_classes == ["cat", "dog"]
    assert report.probabilities_sum == pytest.approx(1.0, abs=1e-5)


def test_job_that_did_not_succeed_is_not_a_smoke(trained, tmp_path):
    job, uri = trained

    report = verify_smoke({**job, "status": "failed"}, tracking_uri=uri, workdir=tmp_path)

    assert any("succeeded" in p for p in report.problems)


def test_run_config_must_be_the_job_config(trained, tmp_path):
    job, uri = trained
    other = {**job, "config": {**job["config"], "learning_rate": 0.005}}

    report = verify_smoke(other, tracking_uri=uri, workdir=tmp_path)

    assert any("learning_rate" in p for p in report.problems)


def test_run_provenance_must_be_the_job_manifest(trained, tmp_path):
    job, uri = trained

    report = verify_smoke({**job, "manifest_hash": "b" * 64}, tracking_uri=uri, workdir=tmp_path)

    assert any("manifest_hash" in p for p in report.problems)


def test_a_test_metric_in_the_run_fails_the_smoke(trained, tmp_path):
    job, uri = trained
    _client(uri).log_metric(job["mlflow_run_id"], "test_accuracy", 0.9)

    report = verify_smoke(job, tracking_uri=uri, workdir=tmp_path)

    assert any("test" in p and "métrica" in p for p in report.problems)


def test_checkpoint_replaced_on_the_server_fails_the_hash(trained, tmp_path):
    job, uri = trained
    model = build_model(TrainingConfig(**CONFIG))
    for tensor in model.state_dict().values():
        if tensor.is_floating_point():
            tensor.add_(1.0)
    _replace_checkpoint(
        _client(uri), job["mlflow_run_id"], model.state_dict(), tmp_path, retag=False
    )

    report = verify_smoke(job, tracking_uri=uri, workdir=tmp_path / "w")

    assert any("sha256" in p for p in report.problems)


def test_checkpoint_that_does_not_fit_the_config_model_fails_loading(trained, tmp_path):
    """Hash coherente con el tag, pero el state_dict no es del modelo de la config."""
    job, uri = trained
    bogus = {"fc.weight": torch.zeros(3, 7)}
    _replace_checkpoint(_client(uri), job["mlflow_run_id"], bogus, tmp_path, retag=True)

    report = verify_smoke(job, tracking_uri=uri, workdir=tmp_path / "w")

    assert any("cargar" in p for p in report.problems)


def test_class_map_artifact_must_match_the_frozen_classes(trained, tmp_path):
    job, uri = trained
    folder = tmp_path / "checkpoint"
    folder.mkdir()
    (folder / "class_map.json").write_text(json.dumps({"dog": 0, "cat": 1}), encoding="utf-8")
    _client(uri).log_artifact(job["mlflow_run_id"], str(folder / "class_map.json"), "checkpoint")

    report = verify_smoke(job, tracking_uri=uri, workdir=tmp_path / "w")

    assert any("class_map" in p for p in report.problems)


def test_run_outside_the_p3_experiment_is_rejected(trained, tmp_path):
    job, uri = trained
    client = _client(uri)
    other = client.create_run(client.create_experiment("otro-experimento")).info.run_id

    report = verify_smoke({**job, "mlflow_run_id": other}, tracking_uri=uri, workdir=tmp_path)

    assert any("p3-cnn-classifier" in p for p in report.problems)
