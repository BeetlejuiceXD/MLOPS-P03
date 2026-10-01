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
from pathlib import Path

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
from tracking.smoke import EPOCH_METRICS, verify_smoke
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


@pytest.fixture
def official(frozen):  # noqa: F811
    """Lo que publican `GET /api/releases` y `GET /api/manifest` para el release y el
    manifest congelado sintéticos (mismos campos del contrato que lee el smoke)."""
    doc, _verify = frozen
    return {
        "releases": {
            "approved": [
                {
                    "dataset_version": doc["dataset_version"],
                    "images_md5": doc["images_md5"],
                    "annotations_md5": doc["annotations_md5"],
                }
            ],
            "rejected": [],
        },
        "manifest": {
            "manifest_version": doc["manifest_version"],
            "manifest_hash": doc["manifest_hash"],
            "dataset_version": doc["dataset_version"],
            "dvc_release_hash": doc["dvc_release_hash"],
            "classes": ["cat", "dog"],
            "frozen": True,
        },
    }


def _clone(uri, job, tmp_path, *, history=None, summary=None, drop=()):
    """Copia el run real del job a un run nuevo del mismo experimento (tags, params,
    curvas, resumen y checkpoint) cambiando SOLO lo indicado: así cada negativo
    altera una pieza y el resto sigue siendo el de un entrenamiento real."""
    client = _client(uri)
    source = client.get_run(job["mlflow_run_id"])
    tags = {k: v for k, v in source.data.tags.items() if not k.startswith("mlflow.")}
    run_id = client.create_run(source.info.experiment_id, tags=tags).info.run_id
    for name, value in source.data.params.items():
        client.log_param(run_id, name, value)
    for name in EPOCH_METRICS:
        points = [(p.step, p.value) for p in client.get_metric_history(source.info.run_id, name)]
        points = sorted(points)
        if history is not None:
            points = history(name, points)
        for step, value in points:
            client.log_metric(run_id, name, value, step=step)
    final = {k: v for k, v in source.data.metrics.items() if k not in EPOCH_METRICS}
    final.update(summary or {})
    for name, value in final.items():
        if name not in drop:
            client.log_metric(run_id, name, value)
    local = client.download_artifacts(source.info.run_id, "checkpoint", str(tmp_path / "src"))
    client.log_artifacts(run_id, local, "checkpoint")
    client.set_terminated(run_id, "FINISHED")
    return {**job, "mlflow_run_id": run_id}


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


def test_real_job_run_and_checkpoint_verify_clean(trained, official, tmp_path):
    job, uri = trained

    report = verify_smoke(job, tracking_uri=uri, workdir=tmp_path, sources=official)

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


def test_job_that_did_not_succeed_is_not_a_smoke(trained, official, tmp_path):
    job, uri = trained

    report = verify_smoke(
        {**job, "status": "failed"}, tracking_uri=uri, workdir=tmp_path, sources=official
    )

    assert any("succeeded" in p for p in report.problems)


def test_run_config_must_be_the_job_config(trained, official, tmp_path):
    job, uri = trained
    other = {**job, "config": {**job["config"], "learning_rate": 0.005}}

    report = verify_smoke(other, tracking_uri=uri, workdir=tmp_path, sources=official)

    assert any("learning_rate" in p for p in report.problems)


def test_run_provenance_must_be_the_job_manifest(trained, official, tmp_path):
    job, uri = trained

    report = verify_smoke(
        {**job, "manifest_hash": "b" * 64}, tracking_uri=uri, workdir=tmp_path, sources=official
    )

    assert any("manifest_hash" in p for p in report.problems)


def test_a_test_metric_in_the_run_fails_the_smoke(trained, official, tmp_path):
    job, uri = trained
    _client(uri).log_metric(job["mlflow_run_id"], "test_accuracy", 0.9)

    report = verify_smoke(job, tracking_uri=uri, workdir=tmp_path, sources=official)

    assert any("test" in p and "métrica" in p for p in report.problems)


def test_checkpoint_replaced_on_the_server_fails_the_hash(trained, official, tmp_path):
    job, uri = trained
    model = build_model(TrainingConfig(**CONFIG))
    for tensor in model.state_dict().values():
        if tensor.is_floating_point():
            tensor.add_(1.0)
    _replace_checkpoint(
        _client(uri), job["mlflow_run_id"], model.state_dict(), tmp_path, retag=False
    )

    report = verify_smoke(job, tracking_uri=uri, workdir=tmp_path / "w", sources=official)

    assert any("sha256" in p for p in report.problems)


def test_checkpoint_that_does_not_fit_the_config_model_fails_loading(trained, official, tmp_path):
    """Hash coherente con el tag, pero el state_dict no es del modelo de la config."""
    job, uri = trained
    bogus = {"fc.weight": torch.zeros(3, 7)}
    _replace_checkpoint(_client(uri), job["mlflow_run_id"], bogus, tmp_path, retag=True)

    report = verify_smoke(job, tracking_uri=uri, workdir=tmp_path / "w", sources=official)

    assert any("cargar" in p for p in report.problems)


def test_class_map_artifact_must_match_the_frozen_classes(trained, official, tmp_path):
    job, uri = trained
    folder = tmp_path / "checkpoint"
    folder.mkdir()
    (folder / "class_map.json").write_text(json.dumps({"dog": 0, "cat": 1}), encoding="utf-8")
    _client(uri).log_artifact(job["mlflow_run_id"], str(folder / "class_map.json"), "checkpoint")

    report = verify_smoke(job, tracking_uri=uri, workdir=tmp_path / "w", sources=official)

    assert any("class_map" in p for p in report.problems)


def test_run_outside_the_p3_experiment_is_rejected(trained, official, tmp_path):
    job, uri = trained
    client = _client(uri)
    other = client.create_run(client.create_experiment("otro-experimento")).info.run_id

    report = verify_smoke(
        {**job, "mlflow_run_id": other}, tracking_uri=uri, workdir=tmp_path, sources=official
    )

    assert any("p3-cnn-classifier" in p for p in report.problems)


def test_numbers_as_the_api_serializes_them_still_match_the_run(trained, official, tmp_path):
    """La API (JSON vía JavaScript) devuelve `dropout: 0` y MLflow guarda "0.0": es el
    mismo valor. Visto en el smoke real con v0.1.1; no debe reportarse como distinto."""
    job, uri = trained
    as_api = {**job, "config": {**job["config"], "dropout": 0, "learning_rate": 0.001}}

    report = verify_smoke(as_api, tracking_uri=uri, workdir=tmp_path, sources=official)

    assert report.problems == []


def test_checkpoint_config_must_be_the_job_config(trained, official, tmp_path):
    """El model.pt es el correcto, pero el training_config.json guardado junto al
    checkpoint no es el del job: no se puede cargar/reutilizar con certeza."""
    job, uri = trained
    folder = tmp_path / "checkpoint"
    folder.mkdir()
    other = {**job["config"], "seed": job["config"]["seed"] + 1}
    (folder / "training_config.json").write_text(json.dumps(other), encoding="utf-8")
    _client(uri).log_artifact(
        job["mlflow_run_id"], str(folder / "training_config.json"), "checkpoint"
    )

    report = verify_smoke(job, tracking_uri=uri, workdir=tmp_path / "w", sources=official)

    assert any("training_config.json" in p for p in report.problems)


# --- B1: historial por época y resumen best_* ------------------------------------------


def test_faithful_clone_of_the_real_run_still_verifies(trained, official, tmp_path):
    """Control del arnés: el clon sin cambios pasa, así cada negativo prueba una pieza."""
    job, uri = trained
    clone = _clone(uri, job, tmp_path)

    report = verify_smoke(clone, tracking_uri=uri, workdir=tmp_path / "w", sources=official)

    assert report.problems == []


def test_duplicated_epoch_step_is_rejected(trained, official, tmp_path):
    """Mismo número de puntos que épocas, pero la época 2 aparece dos veces (sin la 3)."""
    job, uri = trained

    def duplicate(name, points):
        if name != "val_accuracy":
            return points
        return [(2 if step == 3 else step, value) for step, value in points]

    clone = _clone(uri, job, tmp_path, history=duplicate)
    report = verify_smoke(clone, tracking_uri=uri, workdir=tmp_path / "w", sources=official)

    assert any("val_accuracy" in p and "duplicad" in p for p in report.problems)


def test_incomplete_epoch_steps_are_rejected(trained, official, tmp_path):
    """Mismo número de puntos y sin duplicados, pero falta la última época (hay un hueco)."""
    job, uri = trained
    last = job["progress"]["epoch"]

    def gap(name, points):
        if name != "val_loss":
            return points
        return [(last + 3 if step == last else step, value) for step, value in points]

    clone = _clone(uri, job, tmp_path, history=gap)
    report = verify_smoke(clone, tracking_uri=uri, workdir=tmp_path / "w", sources=official)

    assert any("val_loss" in p and "1.." in p for p in report.problems)


@pytest.mark.parametrize("best_epoch", [99.0, 0.0, 2.5])
def test_best_epoch_must_be_an_existing_integer_epoch(trained, official, tmp_path, best_epoch):
    job, uri = trained
    clone = _clone(uri, job, tmp_path, summary={"best_epoch": best_epoch})

    report = verify_smoke(clone, tracking_uri=uri, workdir=tmp_path / "w", sources=official)

    # El motivo debe ser el propio best_epoch, no un efecto secundario (p. ej. que la
    # curva no tenga esa época): así el chequeo de best_epoch se prueba por sí solo.
    assert any(f"best_epoch={best_epoch} no es una época registrada" in p for p in report.problems)


@pytest.mark.parametrize(
    ("summary", "curve"),
    [
        ("best_val_accuracy", "val_accuracy"),
        ("best_val_macro_f1", "val_macro_f1"),
        ("best_val_loss", "val_loss"),
    ],
)
def test_each_best_metric_must_be_the_curve_at_best_epoch(
    trained, official, tmp_path, summary, curve
):
    job, uri = trained
    real = _client(uri).get_run(job["mlflow_run_id"]).data.metrics[summary]
    clone = _clone(uri, job, tmp_path, summary={summary: real + 0.01})

    report = verify_smoke(clone, tracking_uri=uri, workdir=tmp_path / "w", sources=official)

    assert any(summary in p and curve in p for p in report.problems)


@pytest.mark.parametrize("missing", ["best_epoch", "best_val_macro_f1", "best_val_loss"])
def test_missing_best_summary_metric_is_rejected(trained, official, tmp_path, missing):
    job, uri = trained
    clone = _clone(uri, job, tmp_path, drop=(missing,))

    report = verify_smoke(clone, tracking_uri=uri, workdir=tmp_path / "w", sources=official)

    assert any(missing in p for p in report.problems)


# --- B2: procedencia contra las fuentes oficiales, no solo presencia de tags --------


@pytest.mark.parametrize(
    ("tag", "fake"),
    [
        ("classes", "cat,dog,bird"),
        ("dvc_release_hash", "0" * 64),
        ("dvc_images_md5", "f" * 32 + ".dir"),
        ("dvc_annotations_md5", "e" * 32 + ".dir"),
        ("manifest_version", "p3-v9.9.9-s42"),
        ("dvc_release", "v9.9.9"),
        ("seed", "8"),
        ("job_id", "999"),
    ],
)
def test_one_false_provenance_tag_is_rejected(trained, official, tmp_path, tag, fake):
    """Cambia una sola pieza de procedencia en el run; el resto es el run real."""
    job, uri = trained
    _client(uri).set_tag(job["mlflow_run_id"], tag, fake)

    report = verify_smoke(job, tracking_uri=uri, workdir=tmp_path, sources=official)

    assert any(f"tag {tag}=" in p for p in report.problems), report.problems


def test_dvc_release_hash_must_be_recomputable_from_the_official_md5s(trained, official, tmp_path):
    """El tag y el manifest coinciden entre sí, pero no son sha256(images:annotations)."""
    job, uri = trained
    fake = "a" * 64
    _client(uri).set_tag(job["mlflow_run_id"], "dvc_release_hash", fake)
    forged = {**official, "manifest": {**official["manifest"], "dvc_release_hash": fake}}

    report = verify_smoke(job, tracking_uri=uri, workdir=tmp_path, sources=forged)

    assert any("dvc_release_hash" in p and "recalcul" in p for p in report.problems)


def test_run_identity_must_match_the_official_release_not_only_itself(trained, official, tmp_path):
    """El run es coherente consigo mismo, pero el release oficial publicado es otro."""
    job, uri = trained
    other = dict(official["releases"]["approved"][0], images_md5="d" * 32 + ".dir")
    changed = {**official, "releases": {"approved": [other], "rejected": []}}

    report = verify_smoke(job, tracking_uri=uri, workdir=tmp_path, sources=changed)

    assert any("dvc_images_md5" in p for p in report.problems)


def test_job_must_use_an_officially_approved_release(trained, official, tmp_path):
    job, uri = trained
    none = {**official, "releases": {"approved": [], "rejected": []}}

    report = verify_smoke(job, tracking_uri=uri, workdir=tmp_path, sources=none)

    assert any("aprobado" in p for p in report.problems)


def test_checkpoint_sources_json_must_match_the_run_provenance(trained, official, tmp_path):
    job, uri = trained
    folder = tmp_path / "checkpoint"
    folder.mkdir()
    client = _client(uri)
    stored = client.download_artifacts(
        job["mlflow_run_id"], "checkpoint/sources.json", str(tmp_path)
    )
    data = json.loads(Path(stored).read_text(encoding="utf-8"))
    data["manifest_hash"] = "c" * 64
    (folder / "sources.json").write_text(json.dumps(data), encoding="utf-8")
    client.log_artifact(job["mlflow_run_id"], str(folder / "sources.json"), "checkpoint")

    report = verify_smoke(job, tracking_uri=uri, workdir=tmp_path / "w", sources=official)

    assert any("sources.json" in p and "manifest_hash" in p for p in report.problems)
