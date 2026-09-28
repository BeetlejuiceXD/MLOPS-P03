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

from trainer_worker import main as worker_main
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


def test_claim_skips_a_job_cancelled_between_read_and_update(engine, job_store, monkeypatch):
    """Carrera real: la API cancela (u otro worker toma) el job justo después del SELECT."""
    raced = _insert(engine)
    second = _insert(engine)
    original_config = store._config
    interfered: list[int] = []

    def cancel_during_claim(value):
        if not interfered:
            with engine.begin() as conn:
                conn.execute(
                    training_jobs.update()
                    .where(training_jobs.c.id == raced)
                    .values(status="cancelled", finished_at=_naive_utc())
                )
            interfered.append(raced)
        return original_config(value)

    monkeypatch.setattr(store, "_config", cancel_during_claim)

    claimed = job_store.claim_next("worker-a")

    assert interfered == [raced]
    assert claimed is not None and claimed.id == second
    # El cancelado no se "revive" como running: el UPDATE exige status queued.
    assert (_row(engine, raced).status, _row(engine, raced).worker_id) == ("cancelled", None)


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


def _stale_running(engine, run_id: str | None) -> int:
    return _insert(
        engine,
        status="running",
        worker_id="worker-muerto",
        started_at=_naive_utc(timedelta(minutes=-10)),
        heartbeat_at=_naive_utc(timedelta(minutes=-5)),
        progress_epoch=2,
        total_epochs=10,
        mlflow_run_id=run_id,
    )


def test_recovery_also_closes_the_orphan_mlflow_run_as_failed(engine, job_store, tracking):
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking)
    experiment_id = client.create_experiment("p3-recovery")
    orphan_run = client.create_run(experiment_id).info.run_id
    job_id = _stale_running(engine, orphan_run)
    worker = _worker(job_store, tracking)

    recovered = job_store.recover_interrupted(worker.worker_id, timedelta(seconds=60))
    worker.close_interrupted_runs(recovered)

    # Sin esto, MLflow mostraría para siempre un run RUNNING de un job ya fallido.
    assert _mlflow_run(tracking, orphan_run).info.status == "FAILED"
    assert _row(engine, job_id).mlflow_run_id == orphan_run
    assert any("marcado FAILED" in line for line in _logs(engine, job_id))


def test_closing_orphan_runs_tolerates_jobs_without_run_and_mlflow_errors(engine, job_store):
    class BrokenTracker:
        def end(self, run_id, status):
            raise ConnectionError("mlflow caído")

    without_run = _stale_running(engine, None)
    with_run = _stale_running(engine, "b" * 32)
    worker = runner.Worker(store=job_store, tracker=BrokenTracker(), worker_id="worker-test")

    recovered = job_store.recover_interrupted(worker.worker_id, timedelta(seconds=60))
    worker.close_interrupted_runs(recovered)  # no lanza: el job ya quedó failed

    assert _row(engine, without_run).status == "failed"
    assert _row(engine, with_run).status == "failed"
    assert any("No se pudo cerrar el run" in line for line in _logs(engine, with_run))


def test_finish_never_overwrites_a_terminal_job(engine, job_store):
    job_id = _insert(engine, status="cancelled", finished_at=_naive_utc())
    assert job_store.finish(job_id, "worker-test", "succeeded") is False
    assert _row(engine, job_id).status == "cancelled"


def test_idle_worker_does_nothing(job_store, tracking):
    assert _worker(job_store, tracking).run_once() is None


# --- Proceso principal (python -m trainer_worker.main) ---------------------------------


class _FakeWorker:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0
        self.stopped = False
        self.worker_id = "worker-test"
        self.closed: list[list[int]] = []

    def run_once(self):
        self.calls += 1
        result = self.results.pop(0) if self.results else None
        if isinstance(result, Exception):
            raise result
        return result

    def request_stop(self):
        self.stopped = True

    def close_interrupted_runs(self, job_ids):
        self.closed.append(list(job_ids))

    @property
    def stopping(self):
        return self.stopped


class _FakeStore:
    def __init__(self, recovered=()):
        self.recoveries = 0
        self.recovered = list(recovered)

    def recover_interrupted(self, worker_id, stale_after):
        self.recoveries += 1
        recovered, self.recovered = self.recovered, []
        return recovered


def test_serve_polls_recovers_and_sleeps_only_when_idle():
    sleeps: list[float] = []
    fake = _FakeWorker(["succeeded", None, "failed", None])
    store_ = _FakeStore()

    worker_main.serve(
        fake,
        store_,
        poll_seconds=2.0,
        stale_after=timedelta(seconds=60),
        sleep=sleeps.append,
        max_iterations=4,
    )

    assert fake.calls == 4
    assert store_.recoveries == 4
    assert sleeps == [2.0, 2.0]  # sin espera tras ejecutar un job


def test_serve_closes_the_runs_of_recovered_jobs():
    fake = _FakeWorker([None, None])
    worker_main.serve(
        fake,
        _FakeStore(recovered=[7, 9]),
        poll_seconds=1.0,
        stale_after=timedelta(seconds=60),
        sleep=lambda _s: None,
        max_iterations=2,
    )
    assert fake.closed == [[7, 9]]


def test_serve_survives_database_errors_until_the_tables_exist():
    from sqlalchemy.exc import OperationalError

    sleeps: list[float] = []
    fake = _FakeWorker([OperationalError("SELECT", {}, Exception("no such table")), "succeeded"])

    worker_main.serve(
        fake,
        _FakeStore(),
        poll_seconds=1.0,
        stale_after=timedelta(seconds=60),
        sleep=sleeps.append,
        max_iterations=2,
    )

    assert fake.calls == 2
    assert sleeps == [1.0]


def test_serve_stops_when_asked():
    fake = _FakeWorker(["succeeded"] * 10)
    fake.request_stop()
    worker_main.serve(
        fake,
        _FakeStore(),
        poll_seconds=1.0,
        stale_after=timedelta(seconds=60),
        sleep=lambda _s: None,
    )
    assert fake.calls == 0


def test_settings_need_database_and_tracking(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(Exception, match=r"database_url|DATABASE_URL"):
        worker_main.WorkerSettings()

    monkeypatch.setenv("DATABASE_URL", "mysql+pymysql://u:p@mariadb:3306/image_repo")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    settings = worker_main.WorkerSettings()
    assert settings.poll_seconds > 0
    assert settings.stale_after_seconds >= 30
    assert "u:p@" not in repr(settings)  # la URL con contraseña no se imprime
