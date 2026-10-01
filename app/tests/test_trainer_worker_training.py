"""D03-03 — el trainer-worker ejecuta `task=training` con el trainer real (D02-03)
sobre las fuentes verificadas (release aprobado + manifest congelado) y deja job,
run de MLflow y checkpoint trazables.

Mismo arnés que `test_trainer_worker.py` (SQLite con las columnas de la migración y
MLflow de archivos local) y el release/manifest congelado sintético de
`test_training_sources.py`. Configuración pequeña (sin pesos preentrenados, 128 px)
para que el entrenamiento real sea rápido; acredita el componente, no la campaña.
"""

from __future__ import annotations

import functools
import hashlib
import json
from datetime import UTC, datetime

import pytest
import torch
from sqlalchemy import create_engine, select

from tests.test_training_sources import VERSION, _freeze, _release
from trainer.engine import train as real_train
from trainer_worker import runner, store
from trainer_worker.sources import verify_training_sources
from trainer_worker.store import JobStore, training_job_logs, training_jobs
from training.config import TrainingConfig
from training.model import build_model

CONFIG = TrainingConfig(
    seed=7, pretrained=False, image_size=128, batch_size=8, max_epochs=10, patience=3
).model_dump()
EPOCH_FIELDS = (
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "val_macro_f1",
    "learning_rate",
)
CONTRACT_TAGS = (
    "git_commit",
    "dvc_release",
    "dvc_images_md5",
    "dvc_annotations_md5",
    "dvc_release_hash",
    "manifest_version",
    "manifest_hash",
    "classes",
    "seed",
    "job_id",
)


@pytest.fixture
def engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'jobs.db'}")
    store.metadata.create_all(engine)
    return engine


@pytest.fixture
def job_store(engine) -> JobStore:
    return JobStore(engine)


@pytest.fixture
def tracking(tmp_path, monkeypatch) -> str:
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    return (tmp_path / "mlruns").as_uri()


@pytest.fixture
def frozen(tmp_path):
    """Release sintético + manifest congelado + verificador apuntando a ellos."""
    source, policy = _release(tmp_path)
    path, doc = _freeze(tmp_path, source, policy)
    verify = functools.partial(
        verify_training_sources,
        manifest_path=path,
        repo_root=tmp_path,
        reports_dir=tmp_path / "reports",
        sources={VERSION: source},
        policy=policy,
    )
    return doc, verify


def _now():
    return datetime.now(UTC).replace(tzinfo=None)


def _insert(engine, doc, **overrides) -> int:
    row = {
        "task": "training",
        "status": "queued",
        "dataset_version": doc["dataset_version"],
        "manifest_hash": doc["manifest_hash"],
        "config": json.dumps(CONFIG),
        "controlled_fail_at_epoch": None,
        "cancel_requested": False,
        "created_at": _now(),
        "updated_at": _now(),
    } | overrides
    with engine.begin() as conn:
        return conn.execute(training_jobs.insert().values(**row)).inserted_primary_key[0]


def _row(engine, job_id):
    with engine.connect() as conn:
        return conn.execute(select(training_jobs).where(training_jobs.c.id == job_id)).one()


def _logs(engine, job_id) -> list[str]:
    with engine.connect() as conn:
        rows = conn.execute(
            select(training_job_logs.c.message)
            .where(training_job_logs.c.job_id == job_id)
            .order_by(training_job_logs.c.id)
        ).all()
    return [message for (message,) in rows]


def _worker(job_store, tracking, verify, **kwargs) -> runner.Worker:
    return runner.Worker(
        store=job_store,
        tracker=runner.MlflowTracker(tracking),
        worker_id="worker-test",
        epoch_seconds=0,
        verify_sources=verify,
        **kwargs,
    )


def _client(tracking):
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=tracking)


# --- camino real elegible ---------------------------------------------------------------


def test_training_job_trains_for_real_and_leaves_job_run_and_checkpoint_traceable(
    engine, job_store, tracking, frozen, tmp_path
):
    doc, verify = frozen
    job_id = _insert(engine, doc)

    assert _worker(job_store, tracking, verify).run_once() == "succeeded"

    row = _row(engine, job_id)
    assert row.status == "succeeded" and row.error is None
    assert row.total_epochs == CONFIG["max_epochs"]
    assert 1 <= row.progress_epoch <= CONFIG["max_epochs"]
    assert len(row.mlflow_run_id) == 32
    assert row.mlflow_close_status is None

    client = _client(tracking)
    run = client.get_run(row.mlflow_run_id)
    assert run.info.status == "FINISHED"
    tags = run.data.tags
    for tag in CONTRACT_TAGS:
        assert tags.get(tag), f"falta el tag {tag}"
    assert tags["p3.run_kind"] == "training"
    assert tags["dvc_release"] == VERSION
    assert tags["dvc_images_md5"] == doc["images_md5"]
    assert tags["dvc_annotations_md5"] == doc["annotations_md5"]
    expected_release_hash = hashlib.sha256(
        f"{doc['images_md5']}:{doc['annotations_md5']}".encode()
    ).hexdigest()
    assert tags["dvc_release_hash"] == expected_release_hash == doc["dvc_release_hash"]
    assert tags["manifest_version"] == doc["manifest_version"]
    assert tags["manifest_hash"] == doc["manifest_hash"]
    assert tags["classes"] == "cat,dog"
    assert tags["seed"] == "7"
    assert tags["job_id"] == str(job_id)

    params = run.data.params
    for name, value in CONFIG.items():
        assert params[name] == str(value)
    assert params["classes"] == "cat,dog"

    # Una métrica por época real, y el resumen coherente con ese historial.
    history = {f: client.get_metric_history(run.info.run_id, f) for f in EPOCH_FIELDS}
    epochs = {len(points) for points in history.values()}
    assert epochs == {row.progress_epoch}
    metrics = run.data.metrics
    best_epoch = int(metrics["best_epoch"])
    at_best = {p.step: p.value for p in history["val_accuracy"]}[best_epoch]
    assert metrics["best_val_accuracy"] == pytest.approx(at_best)
    for name in ("best_val_macro_f1", "best_val_loss", "duration_seconds", "peak_memory_mb"):
        assert name in metrics
    assert not any(name.startswith("test") for name in metrics)

    # Checkpoint recuperable del servidor, con el hash registrado y cargable en el modelo.
    local = client.download_artifacts(run.info.run_id, "checkpoint/model.pt", str(tmp_path))
    with open(local, "rb") as handle:
        sha = hashlib.sha256(handle.read()).hexdigest()
    assert tags["checkpoint_sha256"] == sha
    model = build_model(TrainingConfig(**CONFIG))
    model.load_state_dict(torch.load(local, weights_only=True))

    logs = _logs(engine, job_id)
    assert any(line.startswith(f"Época {row.progress_epoch}/") for line in logs)
    assert any("Entrenamiento terminado" in line for line in logs)


def test_only_train_and_val_reach_the_trainer(engine, job_store, tracking, frozen):
    doc, verify = frozen
    _insert(engine, doc)
    seen = {}

    def spy_train(config, dataset, **kwargs):
        seen["train"] = [s.sample_id for s in dataset.train]
        seen["val"] = [s.sample_id for s in dataset.val]
        seen["test"] = list(dataset.test)
        return real_train(config, dataset, **kwargs)

    assert _worker(job_store, tracking, verify, train_fn=spy_train).run_once() == "succeeded"
    assert seen["train"] == doc["assignments"]["train"]
    assert seen["val"] == doc["assignments"]["val"]
    assert seen["test"] == []


# --- rechazos antes de entrenar ---------------------------------------------------------


def test_non_eligible_sources_fail_before_training_and_without_a_run(
    engine, job_store, tracking, frozen
):
    doc, verify = frozen
    job_id = _insert(engine, doc, manifest_hash="a" * 64)
    called = []

    worker = _worker(job_store, tracking, verify, train_fn=lambda *a, **k: called.append(1))
    assert worker.run_once() == "failed"

    row = _row(engine, job_id)
    assert "request_mismatch" in row.error
    assert row.mlflow_run_id is None
    assert called == []


def test_training_without_configured_sources_is_rejected(engine, job_store, tracking, frozen):
    doc, _verify = frozen
    job_id = _insert(engine, doc)

    assert _worker(job_store, tracking, None).run_once() == "failed"

    row = _row(engine, job_id)
    assert "fuentes" in row.error.lower()
    assert row.mlflow_run_id is None


# --- garantías de D02-05 durante el entrenamiento real ----------------------------------


def _cancel(engine, job_id):
    with engine.begin() as conn:
        conn.execute(
            training_jobs.update().where(training_jobs.c.id == job_id).values(cancel_requested=True)
        )


def test_cancellation_during_real_training(engine, job_store, tracking, frozen):
    doc, verify = frozen
    job_id = _insert(engine, doc)
    worker = _worker(job_store, tracking, verify)
    worker.after_epoch = lambda epoch: epoch == 2 and _cancel(engine, job_id)

    assert worker.run_once() == "cancelled"

    row = _row(engine, job_id)
    assert (row.status, row.progress_epoch) == ("cancelled", 2)
    assert _client(tracking).get_run(row.mlflow_run_id).info.status == "KILLED"


def test_sigterm_during_real_training_is_interrupted_not_succeeded(
    engine, job_store, tracking, frozen
):
    doc, verify = frozen
    job_id = _insert(engine, doc)
    worker = _worker(job_store, tracking, verify)
    worker.after_epoch = lambda epoch: epoch == 1 and worker.request_stop()

    assert worker.run_once() == "failed"

    row = _row(engine, job_id)
    assert "SIGTERM" in row.error and row.progress_epoch == 1
    assert _client(tracking).get_run(row.mlflow_run_id).info.status == "KILLED"


def test_error_inside_the_trainer_fails_the_job_and_the_run(engine, job_store, tracking, frozen):
    doc, verify = frozen
    job_id = _insert(engine, doc)

    def broken_train(config, dataset, **kwargs):
        raise RuntimeError("fallo del trainer")

    assert _worker(job_store, tracking, verify, train_fn=broken_train).run_once() == "failed"

    row = _row(engine, job_id)
    assert "fallo del trainer" in row.error
    assert _client(tracking).get_run(row.mlflow_run_id).info.status == "FAILED"


def test_checkpoint_that_does_not_match_after_download_is_a_failure(
    engine, job_store, tracking, frozen, monkeypatch
):
    doc, verify = frozen
    job_id = _insert(engine, doc)
    monkeypatch.setattr(runner.MlflowTracker, "_served_sha256", lambda *a, **k: "0" * 64)

    assert _worker(job_store, tracking, verify).run_once() == "failed"

    row = _row(engine, job_id)
    assert "checkpoint" in row.error.lower()
    assert _client(tracking).get_run(row.mlflow_run_id).info.status == "FAILED"


def test_train_reports_every_real_epoch_through_on_epoch_end():
    """Gancho opcional de D03-03 en trainer.engine.train: una llamada por época real,
    en orden y con las mismas métricas del historial (no cambia el entrenamiento)."""
    from trainer.dataset import build_fixture_dataset

    config = TrainingConfig(**CONFIG)
    seen = []
    result = real_train(config, build_fixture_dataset(seed=123), on_epoch_end=seen.append)

    assert [m.epoch for m in seen] == [m.epoch for m in result.history]
    assert seen == list(result.history)
