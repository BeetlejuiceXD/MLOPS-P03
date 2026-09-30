"""D02-05 — Proceso `trainer-worker` (docker-compose): `python -m trainer_worker.main`.

Hace polling de `training_jobs` en MariaDB (sin otra tecnología de colas). En cada
vuelta primero marca como fallidos los jobs de otro worker que dejaron de latir (un
reinicio no los reejecuta en silencio) y luego toma el siguiente job en cola.
SIGTERM/SIGINT: el job en curso termina como `failed` con motivo "Interrumpido".

D03-03: además publica el snapshot de fuentes (releases + manifest congelado
verificado) para el backend, al arrancar y cada `TRAINER_SOURCES_REFRESH_SECONDS`, y
ejecuta `task=training` verificando esas fuentes contra los archivos reales.
"""

from __future__ import annotations

import functools
import logging
import os
import signal
import socket
import time
import uuid
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError

from policies.models import load_quality_policy
from presentation.release_resolver import load_release_sources
from trainer_worker.runner import MlflowTracker, Worker
from trainer_worker.sources import compute_sources_snapshot, verify_training_sources
from trainer_worker.store import JobStore

# Fuera de Docker el repo es el padre de app/; en la imagen, /app (ver docker-compose).
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger("trainer_worker")


class WorkerSettings(BaseSettings):
    """Único lugar de trainer_worker que lee variables de entorno."""

    model_config = SettingsConfigDict(extra="ignore")

    database_url: SecretStr
    mlflow_tracking_uri: str = Field(min_length=1)
    poll_seconds: float = Field(default=2.0, gt=0, alias="TRAINER_POLL_SECONDS")
    epoch_seconds: float = Field(default=0.5, ge=0, alias="TRAINER_CONTROLLED_EPOCH_SECONDS")
    stale_after_seconds: float = Field(default=60.0, ge=30, alias="TRAINER_STALE_AFTER_SECONDS")
    # D03-03: dónde están los datos DVC (data/raw), los reportes de calidad y el
    # manifest P3 congelado de D03-01 (con su .dvc al lado).
    repo_root: Path = Field(default=DEFAULT_REPO_ROOT, alias="TRAINER_REPO_ROOT")
    manifest_path: Path | None = Field(default=None, alias="P3_MANIFEST_PATH")
    sources_refresh_seconds: float = Field(
        default=300.0, gt=0, alias="TRAINER_SOURCES_REFRESH_SECONDS"
    )

    def resolved_manifest_path(self) -> Path:
        return self.manifest_path or self.repo_root / "data" / "p3" / "manifest.json"

    def sources_kwargs(self) -> dict:
        return {
            "manifest_path": self.resolved_manifest_path(),
            "repo_root": self.repo_root,
            "reports_dir": self.repo_root / "reports",
            "sources": load_release_sources(),
            "policy": load_quality_policy(),
        }


def serve(
    worker,
    store,
    *,
    poll_seconds: float,
    stale_after: timedelta,
    sleep: Callable[[float], None] = time.sleep,
    max_iterations: int | None = None,
    sources_refresh_seconds: float = 300.0,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    iteration = 0
    last_refresh: float | None = None
    while not worker.stopping and (max_iterations is None or iteration < max_iterations):
        iteration += 1
        now = clock()
        if last_refresh is None or now - last_refresh >= sources_refresh_seconds:
            last_refresh = now
            try:
                worker.refresh_sources()
            except Exception as error:  # no detiene los jobs: se reintenta en el siguiente ciclo
                logger.warning("No se pudo publicar el snapshot de fuentes: %s", error)
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
    sources_kwargs = settings.sources_kwargs()
    worker = Worker(
        store=store,
        tracker=MlflowTracker(settings.mlflow_tracking_uri),
        # Único por arranque: tras un reinicio, los jobs del proceso anterior son "de otro".
        worker_id=f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}",
        epoch_seconds=settings.epoch_seconds,
        verify_sources=functools.partial(verify_training_sources, **sources_kwargs),
        snapshot_sources=functools.partial(compute_sources_snapshot, **sources_kwargs),
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
        sources_refresh_seconds=settings.sources_refresh_seconds,
    )
    logger.info("trainer-worker detenido")


if __name__ == "__main__":
    main()
