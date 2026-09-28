"""D02-05 — trainer-worker: jobs persistentes ejecutados fuera del request HTTP.

Base de datos SQLite temporal con las mismas columnas que la migración de MariaDB
(`backend/src/data/db/migrations/0003_training_jobs.sql`, comprobado abajo) y MLflow con
store de archivos local. La prueba con MariaDB + MLflow reales y reinicio es el job de
CI "Jobs persistentes". La tarea controlada no entrena: recorre la máquina de estados.
"""

from __future__ import annotations

import json
import re
import socket
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from trainer_worker import runner, store
from trainer_worker.store import JobStore, training_job_logs, training_jobs

REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATION = REPO_ROOT / "backend/src/data/db/migrations/0003_training_jobs.sql"
CONFIG = json.loads(
    (REPO_ROOT / "contracts/p3/fixtures/training_config/valid-defaults.json").read_text()
)["payload"]


def _naive_utc(delta: timedelta = timedelta()) -> datetime:
    return (datetime.now(UTC) + delta).replace(tzinfo=None)


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


def _insert(engine, **overrides) -> int:
    row = {
        "task": "controlled",
        "status": "queued",
        "dataset_version": "v0.1.1",
        "manifest_hash": "d" * 64,
        "config": json.dumps(dict(CONFIG, max_epochs=10)),
        "controlled_fail_at_epoch": None,
        "cancel_requested": False,
        "created_at": _naive_utc(),
        "updated_at": _naive_utc(),
    }
    row.update(overrides)
    with engine.begin() as conn:
        return conn.execute(training_jobs.insert().values(**row)).inserted_primary_key[0]


def _row(engine, job_id: int):
    with engine.connect() as conn:
        return conn.execute(select(training_jobs).where(training_jobs.c.id == job_id)).one()


def _logs(engine, job_id: int) -> list[str]:
    with engine.connect() as conn:
        rows = conn.execute(
            select(training_job_logs.c.message)
            .where(training_job_logs.c.job_id == job_id)
            .order_by(training_job_logs.c.id)
        ).all()
    return [message for (message,) in rows]


def _worker(job_store, tracking_uri, **kwargs) -> runner.Worker:
    return runner.Worker(
        store=job_store,
        tracker=runner.MlflowTracker(tracking_uri),
        worker_id="worker-test",
        epoch_seconds=0,
        **kwargs,
    )


def _mlflow_run(tracking_uri: str, run_id: str):
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=tracking_uri).get_run(run_id)


def test_worker_tables_match_the_backend_migration():
    sql = MIGRATION.read_text(encoding="utf-8")
    for table in (training_jobs, training_job_logs):
        block = re.search(rf"CREATE TABLE `{table.name}` \((.*?)\n\);", sql, re.S)
        assert block, f"{table.name} no está en la migración"
        migration_columns = set(re.findall(r"^\t`([a-z_]+)`", block.group(1), re.M))
        assert migration_columns == {column.name for column in table.columns}


def test_claim_is_atomic_and_only_takes_queued_jobs(engine, job_store):
    first = _insert(engine)
    _insert(engine, status="cancelled")

    claimed = job_store.claim_next("worker-a")
    assert claimed is not None and claimed.id == first
    assert job_store.claim_next("worker-b") is None  # el cancelado y el tomado no se repiten

    row = _row(engine, first)
    assert (row.status, row.worker_id, row.progress_epoch, row.total_epochs) == (
        "running",
        "worker-a",
        0,
        10,
    )
    assert row.started_at is not None and row.heartbeat_at is not None


def test_controlled_task_succeeds_with_progress_logs_and_mlflow_run(engine, job_store, tracking):
    job_id = _insert(engine)

    assert _worker(job_store, tracking).run_once() == "succeeded"

    row = _row(engine, job_id)
    assert row.status == "succeeded"
    assert (row.progress_epoch, row.total_epochs) == (10, 10)
    assert row.error is None and row.finished_at is not None
    assert len(row.mlflow_run_id) == 32
    logs = _logs(engine, job_id)
    assert any("Época 10/10" in line for line in logs)
    assert any("tarea controlada" in line.lower() for line in logs)

    run = _mlflow_run(tracking, row.mlflow_run_id)
    assert run.info.status == "FINISHED"
    assert run.data.tags["p3.run_kind"] == "controlled_task"
    assert run.data.tags["p3.job_id"] == str(job_id)
    assert run.data.params["learning_rate"] == str(CONFIG["learning_rate"])


def test_run_id_is_saved_as_soon_as_the_run_exists(engine, job_store, tracking):
    job_id = _insert(engine, controlled_fail_at_epoch=1)
    _worker(job_store, tracking).run_once()
    assert _row(engine, job_id).mlflow_run_id is not None  # aunque falle en la época 1


def test_controlled_failure_is_failed_not_success(engine, job_store, tracking):
    job_id = _insert(engine, controlled_fail_at_epoch=3)

    assert _worker(job_store, tracking).run_once() == "failed"

    row = _row(engine, job_id)
    assert row.status == "failed"
    assert "época 3" in row.error
    assert row.progress_epoch == 2
    assert _mlflow_run(tracking, row.mlflow_run_id).info.status == "FAILED"
    assert any(line.startswith("Fallo") for line in _logs(engine, job_id))


def test_cancellation_requested_while_running(engine, job_store, tracking):
    job_id = _insert(engine)

    def cancel_after_epoch_2(epoch: int) -> None:
        if epoch == 2:
            with engine.begin() as conn:
                conn.execute(
                    training_jobs.update()
                    .where(training_jobs.c.id == job_id)
                    .values(cancel_requested=True)
                )

    worker = _worker(job_store, tracking, after_epoch=cancel_after_epoch_2)
    assert worker.run_once() == "cancelled"

    row = _row(engine, job_id)
    assert row.status == "cancelled"
    assert row.progress_epoch == 2
    assert _mlflow_run(tracking, row.mlflow_run_id).info.status == "KILLED"


def test_invalid_config_fails_in_the_worker_without_creating_a_run(engine, job_store, tracking):
    job_id = _insert(engine, config=json.dumps(dict(CONFIG, learning_rate=0.05)))

    assert _worker(job_store, tracking).run_once() == "failed"

    row = _row(engine, job_id)
    assert "TrainingConfig inválido" in row.error
    assert "learning_rate" in row.error
    assert row.mlflow_run_id is None


def test_real_training_is_not_run_by_this_worker(engine, job_store, tracking):
    job_id = _insert(engine, task="training")

    assert _worker(job_store, tracking).run_once() == "failed"

    row = _row(engine, job_id)
    assert "D03-03" in row.error
    assert row.mlflow_run_id is None


def test_unavailable_mlflow_fails_the_job_instead_of_faking_success(engine, job_store, monkeypatch):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    job_id = _insert(engine)

    assert _worker(job_store, f"http://127.0.0.1:{port}").run_once() == "failed"

    row = _row(engine, job_id)
    assert "MLflow" in row.error
    assert row.mlflow_run_id is None


def test_stop_signal_marks_the_job_interrupted(engine, job_store, tracking):
    job_id = _insert(engine)
    worker = _worker(job_store, tracking)
    worker_after = worker.after_epoch

    def stop_after_epoch_1(epoch: int) -> None:
        worker_after(epoch)
        if epoch == 1:
            worker.request_stop()

    worker.after_epoch = stop_after_epoch_1
    assert worker.run_once() == "failed"

    row = _row(engine, job_id)
    assert "Interrumpido" in row.error
    assert row.progress_epoch == 1


def test_recovery_marks_stale_running_jobs_failed_without_rerunning(engine, job_store, tracking):
    stale = _insert(
        engine,
        status="running",
        worker_id="worker-muerto",
        started_at=_naive_utc(timedelta(minutes=-10)),
        heartbeat_at=_naive_utc(timedelta(minutes=-5)),
        progress_epoch=4,
        total_epochs=10,
        mlflow_run_id="a" * 32,
    )
    fresh = _insert(
        engine,
        status="running",
        worker_id="worker-vivo",
        started_at=_naive_utc(),
        heartbeat_at=_naive_utc(),
        progress_epoch=1,
        total_epochs=10,
    )

    recovered = job_store.recover_interrupted("worker-test", stale_after=timedelta(seconds=60))

    assert recovered == [stale]
    row = _row(engine, stale)
    assert row.status == "failed"
    assert "no se reintenta" in row.error
    # Se conserva lo que ya había: progreso y vínculo al run.
    assert (row.progress_epoch, row.mlflow_run_id) == (4, "a" * 32)
    assert _row(engine, fresh).status == "running"
    # No vuelve a la cola: ningún worker lo ejecuta dos veces.
    assert job_store.claim_next("worker-test") is None
    assert any("Interrumpido" in line for line in _logs(engine, stale))


def test_finish_never_overwrites_a_terminal_job(engine, job_store):
    job_id = _insert(engine, status="cancelled", finished_at=_naive_utc())
    assert job_store.finish(job_id, "worker-test", "succeeded") is False
    assert _row(engine, job_id).status == "cancelled"


def test_idle_worker_does_nothing(job_store, tracking):
    assert _worker(job_store, tracking).run_once() is None
