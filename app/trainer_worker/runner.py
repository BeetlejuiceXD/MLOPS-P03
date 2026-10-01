"""Ejecución de un job: validación, run de MLflow y entrenamiento.

- `controlled` (D02-05): NO entrena ni lee datos; recorre las épocas con métricas
  sintéticas para demostrar estados, progreso, logs, cancelación, fallo y run.
- `training` (D03-03): verifica las fuentes reales (release aprobado + manifest
  congelado, `trainer_worker.sources`) ANTES de crear el run, entrena con el trainer
  real (D02-03) solo sobre train/val y deja run de MLflow con los tags/hashes reales,
  métricas por época, resumen y checkpoint verificado por SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import torch
from pydantic import ValidationError

from tracking.settings import P3_EXPERIMENT, configure_client_env
from trainer.engine import TrainingResult, train
from trainer_worker.sources import SourcesNotEligibleError, VerifiedSources, build_training_dataset
from trainer_worker.store import ClaimedJob, JobStore, SourceSnapshot
from training.class_map import CLASS_MAP
from training.config import TrainingConfig

RUN_KIND_TAG = "p3.run_kind"
CONTROLLED_RUN_KIND = "controlled_task"
TRAINING_RUN_KIND = "training"
CHECKPOINT_DIR = "checkpoint"
EPOCH_METRIC_FIELDS = (
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "val_macro_f1",
    "learning_rate",
)


def git_commit() -> str:
    """Commit del código que entrena: `GIT_COMMIT` (imagen/Compose) o el checkout local."""
    if os.environ.get("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _peak_memory_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class JobCancelledError(Exception):
    """El usuario pidió cancelar el job."""


class WorkerStoppingError(Exception):
    """El proceso recibió SIGTERM/SIGINT mientras ejecutaba el job."""


class StopDuringFinalizationError(BaseException):
    """Parada recibida mientras se persiste el estado terminal de un job.

    `request_stop` la lanza desde el handler de SIGTERM/SIGINT: interrumpe la
    transacción en el punto exacto donde llegó la señal (como KeyboardInterrupt), así el
    COMMIT no llega a hacerse. BaseException para que ningún `except Exception` la trague.
    """


class ControlledFailureError(Exception):
    """Fallo provocado a propósito por `controlled.fail_at_epoch`."""


class CheckpointVerificationError(Exception):
    """El checkpoint descargado del servidor no coincide con el guardado."""


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

    def start_training(
        self, job: ClaimedJob, config: TrainingConfig, verified: VerifiedSources
    ) -> str:
        """Run real de la campaña: tags de trazabilidad (#33) y params completos."""
        client = self._client()
        experiment = client.get_experiment_by_name(P3_EXPERIMENT)
        experiment_id = (
            experiment.experiment_id
            if experiment is not None
            else client.create_experiment(P3_EXPERIMENT)
        )
        classes = ",".join(sorted(CLASS_MAP))
        manifest = verified.manifest
        run = client.create_run(
            experiment_id,
            run_name=f"job-{job.id}-training",
            tags={
                RUN_KIND_TAG: TRAINING_RUN_KIND,
                "p3.job_id": str(job.id),
                "p3.task": job.task,
                "job_id": str(job.id),
                "git_commit": git_commit(),
                "dvc_release": manifest.dataset_version,
                "dvc_images_md5": manifest.images_md5,
                "dvc_annotations_md5": manifest.annotations_md5,
                "dvc_release_hash": manifest.dvc_release_hash,
                "manifest_version": manifest.manifest_version,
                "manifest_hash": manifest.manifest_hash,
                "classes": classes,
                "seed": str(config.seed),
            },
        )
        run_id = run.info.run_id
        for name, value in config.model_dump().items():
            client.log_param(run_id, name, value)
        client.log_param(run_id, "classes", classes)
        return run_id

    def log_training_summary(self, run_id: str, result: TrainingResult, duration: float) -> None:
        client = self._client()
        client.log_metric(run_id, "best_epoch", result.best.epoch)
        client.log_metric(run_id, "best_val_accuracy", result.best.val_accuracy)
        client.log_metric(run_id, "best_val_macro_f1", result.best.val_macro_f1)
        client.log_metric(run_id, "best_val_loss", result.best.val_loss)
        client.log_metric(run_id, "duration_seconds", duration)
        client.log_metric(run_id, "peak_memory_mb", _peak_memory_mb())
        client.set_tag(run_id, "stopped_early", str(result.stopped_early).lower())
        client.set_tag(run_id, "device", next(result.model.parameters()).device.type)

    def upload_checkpoint(
        self, run_id: str, result: TrainingResult, config: TrainingConfig, verified: VerifiedSources
    ) -> str:
        """Sube el mejor checkpoint y lo verifica descargándolo del servidor."""
        client = self._client()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            checkpoint = folder / "model.pt"
            torch.save(result.model.state_dict(), checkpoint)
            local_sha = _sha256(checkpoint)
            files = {
                "training_config.json": config.model_dump(),
                "class_map.json": dict(CLASS_MAP),
                "environment.json": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                },
                "sources.json": {
                    "dataset_version": verified.manifest.dataset_version,
                    "manifest_version": verified.manifest.manifest_version,
                    "manifest_hash": verified.manifest.manifest_hash,
                    "dvc_release_hash": verified.manifest.dvc_release_hash,
                    "best_epoch": result.best.epoch,
                    "checkpoint_sha256": local_sha,
                },
            }
            client.log_artifact(run_id, str(checkpoint), CHECKPOINT_DIR)
            for name, content in files.items():
                (folder / name).write_text(json.dumps(content, indent=2), encoding="utf-8")
                client.log_artifact(run_id, str(folder / name), CHECKPOINT_DIR)
        client.set_tag(run_id, "checkpoint_sha256", local_sha)
        served = self._served_sha256(run_id, f"{CHECKPOINT_DIR}/model.pt")
        if served != local_sha:
            raise CheckpointVerificationError(
                "El checkpoint descargado del servidor no coincide con el guardado"
            )
        return local_sha

    def _served_sha256(self, run_id: str, artifact_path: str) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            local = self._client().download_artifacts(run_id, artifact_path, tmp)
            return _sha256(Path(local))


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
    # Jobs cuyo cierre pendiente ya se avisó en su log (un aviso por job, no uno por intento).
    _close_warned: set[int] = field(default_factory=set, init=False, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event)
    # D03-03: verificador de fuentes reales (None = training real no configurado) y
    # función de entrenamiento (inyectable en tests; producción usa el trainer real).
    verify_sources: Callable[[str, str], VerifiedSources] | None = None
    train_fn: Callable[..., TrainingResult] = train
    # Snapshot de fuentes para el backend (None = no se publica nada).
    snapshot_sources: Callable[[], dict[str, SourceSnapshot]] | None = None
    # True solo mientras se persiste `succeeded` (UPDATE + último check + COMMIT).
    _finalizing: bool = field(default=False, init=False, repr=False)

    def request_stop(self) -> None:
        """Pide detener el worker (lo llama el handler de SIGTERM/SIGINT de main.py).

        Si llega mientras se persiste el éxito de un job, además corta esa transacción
        (StopDuringFinalizationError): una parada recibida antes de que el estado terminal
        quede persistido nunca termina en succeeded/FINISHED.
        """
        self._stop.set()
        if self._finalizing:
            self._finalizing = False  # una sola interrupción aunque lleguen dos señales
            raise StopDuringFinalizationError

    @property
    def stopping(self) -> bool:
        return self._stop.is_set()

    def run_once(self) -> str | None:
        """Toma y ejecuta un job en cola. Devuelve su estado final, o `None` si no había."""
        job = self.store.claim_next(self.worker_id)
        if job is None:
            return None
        return self._execute(job)

    def refresh_sources(self) -> None:
        """Publica el snapshot de fuentes (releases + manifest verificado) para el backend."""
        if self.snapshot_sources is None:
            return
        for name, snapshot in self.snapshot_sources().items():
            self.store.publish_source(name, snapshot)

    def retry_pending_run_closes(self) -> list[int]:
        """Aplica en MLflow los cierres pendientes (p. ej. MLflow estaba caído).

        Nunca reencola ni reejecuta: solo lleva el run al estado que el job ya tiene.
        """
        closed = []
        for job_id, run_id, run_status in self.store.pending_run_closes():
            if self._close_run(job_id, run_id, run_status):
                closed.append(job_id)
        return closed

    def _close_run(self, job_id: int, run_id: str, run_status: str) -> bool:
        try:
            self.tracker.end(run_id, run_status)
        except Exception as error:  # MLflow caído: queda pendiente y se reintenta
            if job_id not in self._close_warned:
                self._close_warned.add(job_id)
                self.store.log(
                    job_id,
                    "warning",
                    f"No se pudo cerrar el run {run_id} en MLflow ({type(error).__name__}); "
                    f"queda pendiente {run_status} y se reintenta.",
                )
            return False
        self.store.mark_run_closed(job_id, run_status)
        self.store.log(job_id, "info", f"Run de MLflow {run_id} marcado {run_status}")
        return True

    def _finish(
        self,
        job: ClaimedJob,
        run_id: str,
        status: str,
        run_status: str,
        error: str | None = None,
    ) -> str:
        """Primero el estado terminal en MariaDB (con el cierre del run pendiente en el
        mismo UPDATE); después MLflow. Si MLflow falla, el cierre queda para reintentarse."""
        if error is not None:
            self.store.log(job.id, "error", error)
        if not self.store.finish(job.id, self.worker_id, status, error, close_run=run_status):
            return self._lost(job)
        self._close_run(job.id, run_id, run_status)
        return status

    def _lost(self, job: ClaimedJob) -> str:
        """El job ya no está `running` para este worker (p. ej. otro lo recuperó por latido
        vencido): no se sobrescribe; se devuelve el estado que realmente tiene."""
        current = self.store.status(job.id)
        self.store.log(
            job.id,
            "warning",
            f"{self.worker_id} ya no tenía el job en ejecución (estado {current}); no se toca.",
        )
        return current

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

        if job.task == "controlled":
            prepared = self._prepare_controlled(job, config)
        else:
            prepared = self._prepare_training(job, config)
        if isinstance(prepared, str):
            return prepared  # rechazado antes de crear el run
        run_id, body, done_message = prepared

        try:
            body()
        except JobCancelledError:
            return self._cancelled(job, run_id)
        except WorkerStoppingError:
            return self._interrupted(job, run_id)
        except ControlledFailureError as error:
            return self._finish(job, run_id, "failed", "FAILED", f"Fallo controlado: {error}")
        except Exception as error:
            message = f"Fallo inesperado: {type(error).__name__}: {error}"
            return self._finish(job, run_id, "failed", "FAILED", message)

        # Estado terminal: el UPDATE exige cancel_requested=false; dentro de la misma
        # transacción, justo antes del COMMIT, se revisa si llegó una parada; y si la señal
        # llega DESPUÉS de ese check, request_stop corta la transacción antes del COMMIT.
        if self.stopping:
            return self._interrupted(job, run_id)
        committed: bool | None
        try:
            try:
                self._finalizing = True
                committed = self.store.finish(
                    job.id,
                    self.worker_id,
                    "succeeded",
                    close_run="FINISHED",
                    unless_cancel_requested=True,
                    abort_if=lambda: self.stopping,
                )
            finally:
                self._finalizing = False
        except StopDuringFinalizationError:
            committed = None  # la señal cortó el cierre: MariaDB dice si hubo COMMIT
        if committed is None:
            committed = self.store.status(job.id) == "succeeded"
            if committed:
                self.store.log(
                    job.id,
                    "info",
                    "La parada llegó con el COMMIT ya hecho: el éxito estaba persistido.",
                )
        if not committed:
            if self.store.cancel_requested(job.id):
                return self._cancelled(job, run_id)
            if self.stopping:
                return self._interrupted(job, run_id)
            return self._lost(job)
        self.store.log(job.id, "info", done_message)
        self._close_run(job.id, run_id, "FINISHED")
        return "succeeded"

    def _start_run(self, job: ClaimedJob, start: Callable[[], str]) -> str | None:
        """Crea el run y lo vincula al job; si MLflow falla, el job termina failed."""
        try:
            run_id = start()
        except Exception as error:
            self._fail(job, f"No se pudo crear el run en MLflow: {type(error).__name__}: {error}")
            return None
        self.store.set_run_id(job.id, self.worker_id, run_id)
        return run_id

    def _prepare_controlled(self, job: ClaimedJob, config: TrainingConfig):
        run_id = self._start_run(job, lambda: self.tracker.start(job, config))
        if run_id is None:
            return "failed"
        self.store.log(
            job.id,
            "info",
            f"Tarea controlada (sin datos ni entrenamiento real): run MLflow {run_id}, "
            f"{config.max_epochs} épocas",
        )
        return (
            run_id,
            lambda: self._run_controlled(job, config, run_id),
            "Tarea controlada terminada",
        )

    def _prepare_training(self, job: ClaimedJob, config: TrainingConfig):
        """Verifica las fuentes reales y prepara train/val ANTES de crear el run: una
        entrada no elegible nunca entrena ni deja un run de campaña."""
        if self.verify_sources is None:
            return self._fail(
                job,
                "Training real sin fuentes configuradas (release + manifest congelado de "
                "D03-01): no se entrena.",
            )
        try:
            verified = self.verify_sources(job.dataset_version, job.manifest_hash)
        except SourcesNotEligibleError as error:
            return self._fail(job, f"Fuentes no elegibles ({error.reason}): {error.detail}")
        try:
            dataset = build_training_dataset(verified)
        except Exception as error:
            message = f"No se pudieron cargar los crops train/val: {type(error).__name__}: {error}"
            return self._fail(job, message)
        self.store.log(
            job.id,
            "info",
            f"Fuentes verificadas: {verified.manifest.dataset_version} / "
            f"{verified.manifest.manifest_version} ({verified.manifest.manifest_hash[:12]}…); "
            f"train={len(dataset.train)} val={len(dataset.val)} crops (test no se carga)",
        )
        run_id = self._start_run(job, lambda: self.tracker.start_training(job, config, verified))
        if run_id is None:
            return "failed"
        self.store.log(
            job.id,
            "info",
            f"Entrenamiento real: run MLflow {run_id}, hasta {config.max_epochs} épocas",
        )
        return (
            run_id,
            lambda: self._run_training(job, config, run_id, verified, dataset),
            "Entrenamiento terminado",
        )

    def _run_training(self, job, config, run_id, verified, dataset) -> None:
        if self.store.cancel_requested(job.id):
            raise JobCancelledError
        if self.stopping:
            raise WorkerStoppingError

        def on_epoch_end(metrics) -> None:
            self.tracker.log_epoch(
                run_id, metrics.epoch, {f: getattr(metrics, f) for f in EPOCH_METRIC_FIELDS}
            )
            self.store.set_progress(job.id, self.worker_id, metrics.epoch)
            self.store.log(
                job.id,
                "info",
                f"Época {metrics.epoch}/{config.max_epochs} "
                f"train_loss={metrics.train_loss:.4f} val_loss={metrics.val_loss:.4f} "
                f"val_accuracy={metrics.val_accuracy:.4f} val_macro_f1={metrics.val_macro_f1:.4f}",
            )
            self.after_epoch(metrics.epoch)
            if self.store.cancel_requested(job.id):
                raise JobCancelledError
            if self.stopping:
                raise WorkerStoppingError

        started = time.monotonic()
        result = self.train_fn(config, dataset, on_epoch_end=on_epoch_end)
        duration = time.monotonic() - started
        self.tracker.log_training_summary(run_id, result, duration)
        sha = self.tracker.upload_checkpoint(run_id, result, config, verified)
        self.store.log(
            job.id,
            "info",
            f"Mejor época {result.best.epoch} (val_accuracy={result.best.val_accuracy:.4f}); "
            f"checkpoint verificado sha256={sha[:12]}…"
            + (" — early stopping" if result.stopped_early else ""),
        )

    def _cancelled(self, job: ClaimedJob, run_id: str) -> str:
        self.store.log(job.id, "warning", "Cancelado a petición del usuario")
        return self._finish(job, run_id, "cancelled", "KILLED")

    def _interrupted(self, job: ClaimedJob, run_id: str) -> str:
        return self._finish(
            job,
            run_id,
            "failed",
            "KILLED",
            "Interrumpido: el worker se detuvo (SIGTERM) durante la ejecución. "
            "No se reintenta automáticamente para no duplicar el entrenamiento.",
        )

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
