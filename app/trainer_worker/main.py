"""D02-05 — Proceso `trainer-worker` (docker-compose): `python -m trainer_worker.main`.

Hace polling de `training_jobs` en MariaDB (sin otra tecnología de colas). En cada
vuelta primero marca como fallidos los jobs de otro worker que dejaron de latir (un
reinicio no los reejecuta en silencio) y luego toma el siguiente job en cola.
SIGTERM/SIGINT: el job en curso termina como `failed` con motivo "Interrumpido".
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import time
import uuid
from collections.abc import Callable
from datetime import timedelta

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

from trainer_worker.runner import MlflowTracker, Worker
from trainer_worker.store import JobStore

logger = logging.getLogger("trainer_worker")


class WorkerSettings(BaseSettings):
    """Único lugar de trainer_worker que lee variables de entorno."""

    model_config = SettingsConfigDict(extra="ignore")

    database_url: SecretStr
    mlflow_tracking_uri: str = Field(min_length=1)
    poll_seconds: float = Field(default=2.0, gt=0, alias="TRAINER_POLL_SECONDS")
    epoch_seconds: float = Field(default=0.5, ge=0, alias="TRAINER_CONTROLLED_EPOCH_SECONDS")
    stale_after_seconds: float = Field(default=60.0, ge=30, alias="TRAINER_STALE_AFTER_SECONDS")


def serve(
    worker,
    store,
    *,
    poll_seconds: float,
    stale_after: timedelta,
    sleep: Callable[[float], None] = time.sleep,
    max_iterations: int | None = None,
) -> None:
    iteration = 0
    while not worker.stopping and (max_iterations is None or iteration < max_iterations):
        iteration += 1
        try:
            recovered = store.recover_interrupted(worker.worker_id, stale_after)
            if recovered:
                logger.warning("Jobs interrumpidos marcados como fallidos: %s", recovered)
            # Cierres de runs pendientes (MLflow caído antes): se reintentan cada vuelta.
            worker.retry_pending_run_closes()
            result = worker.run_once()
        except SQLAlchemyError as error:
            # P. ej. el backend aún no aplica la migración de training_jobs.
            logger.warning("Base de datos no disponible todavía: %s", type(error).__name__)
            result = None
        if result is None:
            sleep(poll_seconds)
        else:
            logger.info("Job terminado con estado %s", result)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = WorkerSettings()
    engine = create_engine(settings.database_url.get_secret_value(), pool_pre_ping=True)
    store = JobStore(engine)
    worker = Worker(
        store=store,
        tracker=MlflowTracker(settings.mlflow_tracking_uri),
        # Único por arranque: tras un reinicio, los jobs del proceso anterior son "de otro".
        worker_id=f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}",
        epoch_seconds=settings.epoch_seconds,
    )

    def stop(signum, _frame) -> None:
        logger.info("Señal %s: se detiene al terminar el paso actual", signum)
        worker.request_stop()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    logger.info("trainer-worker %s escuchando training_jobs", worker.worker_id)
    serve(
        worker,
        store,
        poll_seconds=settings.poll_seconds,
        stale_after=timedelta(seconds=settings.stale_after_seconds),
    )
    logger.info("trainer-worker detenido")


if __name__ == "__main__":
    main()
