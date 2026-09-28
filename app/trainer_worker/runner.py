"""D02-05 — Ejecución de un job: validación, run de MLflow y tarea controlada.

La tarea controlada NO entrena ni lee datos: recorre las épocas de `max_epochs` con
métricas sintéticas deterministas para demostrar estados, progreso, logs, cancelación,
fallo y vínculo al run de MLflow. El entrenamiento real (`task=training`) se conecta en
D03-03 con el trainer y el manifest oficial; este worker lo rechaza de forma explícita.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import ValidationError

from tracking.settings import P3_EXPERIMENT, configure_client_env
from trainer_worker.store import ClaimedJob, JobStore
from training.config import TrainingConfig

RUN_KIND_TAG = "p3.run_kind"
CONTROLLED_RUN_KIND = "controlled_task"


class JobCancelledError(Exception):
    """El usuario pidió cancelar el job."""


class WorkerStoppingError(Exception):
    """El proceso recibió SIGTERM/SIGINT mientras ejecutaba el job."""


class ControlledFailureError(Exception):
    """Fallo provocado a propósito por `controlled.fail_at_epoch`."""


class MlflowTracker:
    """Runs de MLflow para los jobs (experimento `p3-cnn-classifier`)."""

    def __init__(self, tracking_uri: str):
        self.tracking_uri = tracking_uri

    def _client(self):
        configure_client_env()
        from mlflow.tracking import MlflowClient

        return MlflowClient(tracking_uri=self.tracking_uri)

    def start(self, job: ClaimedJob, config: TrainingConfig) -> str:
        client = self._client()
        experiment = client.get_experiment_by_name(P3_EXPERIMENT)
        experiment_id = (
            experiment.experiment_id
            if experiment is not None
            else client.create_experiment(P3_EXPERIMENT)
        )
        run = client.create_run(
            experiment_id,
            run_name=f"job-{job.id}-{job.task}",
            tags={
                RUN_KIND_TAG: CONTROLLED_RUN_KIND,
                "p3.job_id": str(job.id),
                "p3.task": job.task,
                "p3.ticket": "D02-05",
                "dvc_release": job.dataset_version,
                "manifest_hash": job.manifest_hash,
            },
        )
        run_id = run.info.run_id
        for name, value in config.model_dump().items():
            client.log_param(run_id, name, value)
        return run_id

    def log_epoch(self, run_id: str, epoch: int, metrics: dict[str, float]) -> None:
        client = self._client()
        for name, value in metrics.items():
            client.log_metric(run_id, name, value, step=epoch)

    def end(self, run_id: str, status: str) -> None:
        self._client().set_terminated(run_id, status)


def controlled_metrics(seed: int, epoch: int) -> dict[str, float]:
    """Métricas sintéticas deterministas (prefijo `controlled_`: no son de entrenamiento)."""
    offset = (seed % 7) / 100
    return {
        "controlled_loss": round(1 / (epoch + 1) + offset, 6),
        "controlled_progress": round(1 - math.exp(-epoch / 3), 6),
    }


@dataclass
class Worker:
    store: JobStore
    tracker: MlflowTracker
    worker_id: str
    epoch_seconds: float = 0.5
    sleep: Callable[[float], None] = time.sleep
    after_epoch: Callable[[int], None] = field(default=lambda _epoch: None)
    _stop: threading.Event = field(default_factory=threading.Event)

    def request_stop(self) -> None:
        self._stop.set()

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def run_once(self) -> str | None:
        """Toma y ejecuta un job en cola. Devuelve su estado final, o `None` si no había."""
        job = self.store.claim_next(self.worker_id)
        if job is None:
            return None
        return self._execute(job)

    def _fail(self, job: ClaimedJob, message: str) -> str:
        self.store.log(job.id, "error", message)
        self.store.finish(job.id, self.worker_id, "failed", message)
        return "failed"

    def _execute(self, job: ClaimedJob) -> str:
        self.store.log(job.id, "info", f"Job {job.id} tomado por {self.worker_id}")
        try:
            config = TrainingConfig.model_validate(job.config)
        except ValidationError as error:
            fields = ", ".join(".".join(map(str, e["loc"])) for e in error.errors())
            return self._fail(job, f"TrainingConfig inválido en el worker ({fields}): {error}")

        if job.task != "controlled":
            return self._fail(
                job,
                "Este worker solo ejecuta tareas controladas; el training real con el trainer y "
                "el manifest oficial se conecta en D03-03.",
            )

        try:
            run_id = self.tracker.start(job, config)
        except Exception as error:
            return self._fail(
                job, f"No se pudo crear el run en MLflow: {type(error).__name__}: {error}"
            )
        self.store.set_run_id(job.id, self.worker_id, run_id)
        self.store.log(
            job.id,
            "info",
            f"Tarea controlada (sin datos ni entrenamiento real): run MLflow {run_id}, "
            f"{config.max_epochs} épocas",
        )

        try:
            self._run_controlled(job, config, run_id)
        except JobCancelledError:
            self.store.log(job.id, "warning", "Cancelado a petición del usuario")
            self.tracker.end(run_id, "KILLED")
            self.store.finish(job.id, self.worker_id, "cancelled")
            return "cancelled"
        except WorkerStoppingError:
            self.tracker.end(run_id, "KILLED")
            return self._fail(
                job,
                "Interrumpido: el worker se detuvo (SIGTERM) durante la ejecución. "
                "No se reintenta automáticamente para no duplicar el entrenamiento.",
            )
        except ControlledFailureError as error:
            self.tracker.end(run_id, "FAILED")
            return self._fail(job, f"Fallo controlado: {error}")
        except Exception as error:
            self.tracker.end(run_id, "FAILED")
            return self._fail(job, f"Fallo inesperado: {type(error).__name__}: {error}")

        self.tracker.end(run_id, "FINISHED")
        self.store.log(job.id, "info", "Tarea controlada terminada")
        self.store.finish(job.id, self.worker_id, "succeeded")
        return "succeeded"

    def _run_controlled(self, job: ClaimedJob, config: TrainingConfig, run_id: str) -> None:
        for epoch in range(1, config.max_epochs + 1):
            if self.store.cancel_requested(job.id):
                raise JobCancelledError
            if self.stopping:
                raise WorkerStoppingError
            if self.epoch_seconds:
                self.sleep(self.epoch_seconds)
            if job.controlled_fail_at_epoch == epoch:
                raise ControlledFailureError(
                    f"fail_at_epoch={epoch}: la tarea controlada falla a propósito "
                    f"en la época {epoch}"
                )
            metrics = controlled_metrics(config.seed, epoch)
            self.tracker.log_epoch(run_id, epoch, metrics)
            self.store.set_progress(job.id, self.worker_id, epoch)
            self.store.log(
                job.id,
                "info",
                f"Época {epoch}/{config.max_epochs} controlled_loss={metrics['controlled_loss']}",
            )
            self.after_epoch(epoch)
