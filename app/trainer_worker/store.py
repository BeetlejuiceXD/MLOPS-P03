"""D02-05 — Acceso del worker a `training_jobs` y `training_job_logs`.

Las tablas las crea la migración del backend (Drizzle, 0003_training_jobs.sql); aquí se
declaran las mismas columnas para SQLAlchemy Core (un test compara ambas). Toda
transición se hace con un UPDATE condicionado al estado actual, así dos procesos no
pueden tomar ni cerrar el mismo job dos veces.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    and_,
    select,
)
from sqlalchemy.engine import Engine

metadata = MetaData()

# BigInteger().with_variant(Integer, "sqlite"): autoincremento real también en SQLite (tests).
_ID = BigInteger().with_variant(Integer, "sqlite")

training_jobs = Table(
    "training_jobs",
    metadata,
    Column("id", _ID, primary_key=True, autoincrement=True),
    Column("task", String(16), nullable=False),
    Column("status", String(16), nullable=False),
    Column("dataset_version", String(32), nullable=False),
    Column("manifest_hash", String(64), nullable=False),
    Column("config", Text, nullable=False),
    Column("controlled_fail_at_epoch", Integer),
    Column("progress_epoch", Integer),
    Column("total_epochs", Integer),
    Column("mlflow_run_id", String(32)),
    Column("error", Text),
    Column("cancel_requested", Boolean, nullable=False, default=False),
    Column("worker_id", String(128)),
    Column("heartbeat_at", DateTime),
    Column("created_at", DateTime, nullable=False),
    Column("started_at", DateTime),
    Column("finished_at", DateTime),
    Column("updated_at", DateTime, nullable=False),
)

training_job_logs = Table(
    "training_job_logs",
    metadata,
    Column("id", _ID, primary_key=True, autoincrement=True),
    Column("job_id", _ID, nullable=False),
    Column("ts", DateTime, nullable=False),
    Column("level", String(8), nullable=False),
    Column("message", Text, nullable=False),
)

TERMINAL = ("succeeded", "failed", "cancelled")
MAX_ERROR_CHARS = 2000


def utcnow() -> datetime:
    """UTC sin zona: MariaDB `timestamp` y los contenedores trabajan en UTC."""
    return datetime.now(UTC).replace(tzinfo=None)


@dataclass(frozen=True)
class ClaimedJob:
    id: int
    task: str
    dataset_version: str
    manifest_hash: str
    config: dict[str, Any]
    controlled_fail_at_epoch: int | None


def _config(value: Any) -> dict[str, Any]:
    # MariaDB guarda JSON como LONGTEXT; algunos drivers ya lo entregan decodificado.
    return json.loads(value) if isinstance(value, str | bytes) else dict(value)


class JobStore:
    def __init__(self, engine: Engine):
        self.engine = engine

    def claim_next(self, worker_id: str) -> ClaimedJob | None:
        """Toma el job en cola más antiguo. `None` si no hay ninguno."""
        while True:
            with self.engine.begin() as conn:
                row = conn.execute(
                    select(training_jobs)
                    .where(training_jobs.c.status == "queued")
                    .order_by(training_jobs.c.id)
                    .limit(1)
                ).first()
                if row is None:
                    return None
                config = _config(row.config)
                now = utcnow()
                claimed = conn.execute(
                    training_jobs.update()
                    .where(and_(training_jobs.c.id == row.id, training_jobs.c.status == "queued"))
                    .values(
                        status="running",
                        worker_id=worker_id,
                        started_at=now,
                        heartbeat_at=now,
                        progress_epoch=0,
                        total_epochs=config.get("max_epochs"),
                        updated_at=now,
                    )
                ).rowcount
            if claimed == 1:
                return ClaimedJob(
                    id=row.id,
                    task=row.task,
                    dataset_version=row.dataset_version,
                    manifest_hash=row.manifest_hash,
                    config=config,
                    controlled_fail_at_epoch=row.controlled_fail_at_epoch,
                )
            # Otro proceso lo tomó o lo cancelaron entre la lectura y el UPDATE: siguiente.

    def _update_running(self, job_id: int, worker_id: str, **values: Any) -> bool:
        now = utcnow()
        with self.engine.begin() as conn:
            return (
                conn.execute(
                    training_jobs.update()
                    .where(
                        and_(
                            training_jobs.c.id == job_id,
                            training_jobs.c.status == "running",
                            training_jobs.c.worker_id == worker_id,
                        )
                    )
                    .values(heartbeat_at=now, updated_at=now, **values)
                ).rowcount
                == 1
            )

    def heartbeat(self, job_id: int, worker_id: str) -> bool:
        return self._update_running(job_id, worker_id)

    def set_progress(self, job_id: int, worker_id: str, epoch: int) -> bool:
        return self._update_running(job_id, worker_id, progress_epoch=epoch)

    def set_run_id(self, job_id: int, worker_id: str, run_id: str) -> bool:
        return self._update_running(job_id, worker_id, mlflow_run_id=run_id)

    def cancel_requested(self, job_id: int) -> bool:
        with self.engine.connect() as conn:
            value = conn.execute(
                select(training_jobs.c.cancel_requested).where(training_jobs.c.id == job_id)
            ).scalar()
        return bool(value)

    def log(self, job_id: int, level: str, message: str) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                training_job_logs.insert().values(
                    job_id=job_id, ts=utcnow(), level=level, message=message
                )
            )

    def finish(self, job_id: int, worker_id: str, status: str, error: str | None = None) -> bool:
        """Cierra un job propio en ejecución. Nunca pisa un job ya terminado."""
        if status not in TERMINAL:
            raise ValueError(f"Estado final inválido: {status}")
        if (status == "failed") != (error is not None):
            raise ValueError("Solo 'failed' lleva mensaje de error, y siempre lo lleva")
        return self._update_running(
            job_id,
            worker_id,
            status=status,
            error=None if error is None else error[:MAX_ERROR_CHARS],
            finished_at=utcnow(),
        )

    def recover_interrupted(self, worker_id: str, stale_after: timedelta) -> list[int]:
        """Marca como fallidos los jobs `running` de otro worker sin latido reciente.

        No los vuelve a encolar: reejecutar en silencio duplicaría el entrenamiento.
        Conserva progreso y run de MLflow; el error explica qué pasó.
        """
        limit = utcnow() - stale_after
        with self.engine.connect() as conn:
            candidates = conn.execute(
                select(training_jobs.c.id, training_jobs.c.worker_id, training_jobs.c.heartbeat_at)
                .where(training_jobs.c.status == "running")
                .order_by(training_jobs.c.id)
            ).all()
        recovered: list[int] = []
        for job_id, owner, heartbeat in candidates:
            if owner == worker_id or (heartbeat is not None and heartbeat > limit):
                continue
            message = (
                f"Interrumpido: el worker {owner} dejó de responder (último latido {heartbeat}). "
                "El job no se reintenta automáticamente para no duplicar el entrenamiento; "
                "vuelve a lanzarlo si corresponde."
            )
            now = utcnow()
            with self.engine.begin() as conn:
                updated = conn.execute(
                    training_jobs.update()
                    .where(
                        and_(
                            training_jobs.c.id == job_id,
                            training_jobs.c.status == "running",
                            training_jobs.c.worker_id == owner,
                        )
                    )
                    .values(status="failed", error=message, finished_at=now, updated_at=now)
                ).rowcount
            if updated == 1:
                self.log(job_id, "error", message)
                recovered.append(job_id)
        return recovered
