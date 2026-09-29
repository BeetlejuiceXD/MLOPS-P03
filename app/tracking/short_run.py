"""D02-06 — Corrida corta real del trainer de D02-03, instrumentada en el
MLflow persistente de D02-01 (experimento `p3-cnn-classifier`).

Componente de instrumentación/capacidad, no campaña: usa un fixture propio de
datos (`trainer.dataset`), nunca el manifest oficial (D03-01) ni el frozen
test. El run queda marcado con `p3.run_kind=short_run_instrumentation` y
`p3.data_provenance=fixture` para que Experiments/selección lo excluyan de la
campaña de verdad y nadie confunda sus hashes con los reales de v0.1.1.

Uso (con MLFLOW_TRACKING_URI definido):

    python -m tracking.short_run run --evidence evidencia.json
    python -m tracking.short_run check --evidence evidencia.json

`run` entrena de verdad (pesos actualizados por minibatch, D02-03), registra
TrainingConfig/tags/métricas por época/summary/checkpoint en MLflow, descarga
el checkpoint para confirmar su hash y guarda todo en el archivo de evidencia.
`check` vuelve a consultar el servidor y lo compara con esa evidencia.

Códigos de salida: 0 = verificado · 1 = el servidor responde pero algo no
coincide · 2 = servidor no disponible, entrenamiento falló o evidencia
ilegible. Nunca reporta éxito sin haber leído del servidor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch

from tracking.settings import TrackingSettings, configure_client_env
from trainer.dataset import build_fixture_dataset
from trainer.engine import train
from training.class_map import CLASS_MAP
from training.config import TrainingConfig

VERIFIED = 0
MISMATCH = 1
UNAVAILABLE = 2

RUN_NAME = "d02-06-short-run"
RUN_KIND_TAG = "p3.run_kind"
RUN_KIND = "short_run_instrumentation"
DATA_PROVENANCE_TAG = "p3.data_provenance"
DATA_PROVENANCE_FIXTURE = "fixture"  # NUNCA "v0.1.1" — ver #33/DAY-02

# Identidad de datos controlada para el fixture propio de trainer.dataset. NO
# son hashes reales de v0.1.1 (esos requieren D02-04/D03-01, fuera de alcance
# aquí) — por eso van marcados con DATA_PROVENANCE_TAG y con este prefijo.
FIXTURE_DVC_RELEASE = "fixture-d02-06"
FIXTURE_IMAGES_MD5 = "0" * 32
FIXTURE_ANNOTATIONS_MD5 = "1" * 32
FIXTURE_MANIFEST_VERSION = "fixture-d02-06-v1"
FIXTURE_MANIFEST_HASH = "2" * 64

CHECKPOINT_DIR = "checkpoint"
EPOCH_METRIC_FIELDS = (
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "val_macro_f1",
    "learning_rate",
)


def dvc_release_hash(images_md5: str, annotations_md5: str) -> str:
    """`sha256("<images_md5>:<annotations_md5>")`, hex minúsculo — contrato
    acordado con Hannah en #33 para el tag `dvc_release_hash`."""
    return hashlib.sha256(f"{images_md5}:{annotations_md5}".encode()).hexdigest()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _peak_memory_mb() -> float:
    # ru_maxrss: KB en Linux (el contenedor y el runner de CI lo son).
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


@dataclass(frozen=True)
class Evidence:
    tracking_uri: str
    experiment_name: str
    experiment_id: str
    run_id: str
    job_id: str
    config: dict
    classes: list[str]
    seed: int
    git_commit: str
    dvc_release: str
    dvc_images_md5: str
    dvc_annotations_md5: str
    dvc_release_hash: str
    manifest_version: str
    manifest_hash: str
    epochs: int
    best_epoch: int
    best_val_accuracy: float
    best_val_macro_f1: float
    best_val_loss: float
    checkpoint_artifact_path: str
    checkpoint_sha256: str
    duration_seconds: float
    device: str
    peak_memory_mb: float
    written_at: str


class TrackingUnavailableError(RuntimeError):
    """El servidor de MLflow no respondió."""


class ArtifactDownloadError(RuntimeError):
    """El servidor respondió, pero no devolvió el artefacto."""


def _connect(tracking_uri: str):
    configure_client_env()
    import mlflow
    from mlflow.tracking import MlflowClient

    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        client.search_experiments(max_results=1)
    except Exception as error:
        raise TrackingUnavailableError(
            f"MLflow no disponible en {tracking_uri}: {type(error).__name__}"
        ) from error
    return client


def _download_sha256(client, run_id: str, artifact_path: str) -> str:
    """Mismo patrón que `tracking.verify`: no usa `client.download_artifacts`,
    cuyo `RunsArtifactRepository` esconde el error real de S3/MinIO."""
    from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

    artifact_uri = client.get_run(run_id).info.artifact_uri
    try:
        with tempfile.TemporaryDirectory() as tmp:
            local = get_artifact_repository(artifact_uri).download_artifacts(artifact_path, tmp)
            return _sha256(Path(local))
    except Exception as error:
        raise ArtifactDownloadError(
            f"{artifact_uri}/{artifact_path}: {type(error).__name__}: {error}"
        ) from error


def run_short_training(
    settings: TrackingSettings, evidence_path: Path, *, seed: int = 7
) -> Evidence:
    client = _connect(settings.mlflow_tracking_uri)
    experiment = client.get_experiment_by_name(settings.experiment_name)
    experiment_id = (
        experiment.experiment_id
        if experiment is not None
        else client.create_experiment(settings.experiment_name)
    )

    job_id = f"d02-06-{int(time.time())}"
    config = TrainingConfig(
        seed=seed, pretrained=False, batch_size=8, max_epochs=10, patience=3, image_size=128
    )
    classes = sorted(CLASS_MAP)
    images_md5, annotations_md5 = FIXTURE_IMAGES_MD5, FIXTURE_ANNOTATIONS_MD5
    release_hash = dvc_release_hash(images_md5, annotations_md5)
    git_commit = _git_commit()

    run = client.create_run(
        experiment_id,
        run_name=RUN_NAME,
        tags={
            RUN_KIND_TAG: RUN_KIND,
            DATA_PROVENANCE_TAG: DATA_PROVENANCE_FIXTURE,
            "p3.ticket": "D02-06",
            "job_id": job_id,
            "git_commit": git_commit,
            "dvc_release": FIXTURE_DVC_RELEASE,
            "dvc_images_md5": images_md5,
            "dvc_annotations_md5": annotations_md5,
            "dvc_release_hash": release_hash,
            "manifest_version": FIXTURE_MANIFEST_VERSION,
            "manifest_hash": FIXTURE_MANIFEST_HASH,
            "classes": ",".join(classes),
            "seed": str(seed),
        },
    )
    run_id = run.info.run_id
    for name, value in config.model_dump().items():
        client.log_param(run_id, name, value)

    status = "FAILED"
    duration = 0.0
    try:
        dataset = build_fixture_dataset(seed=123)
        start = time.monotonic()
        result = train(config, dataset)
        duration = time.monotonic() - start

        for metrics in result.history:
            for field_name in EPOCH_METRIC_FIELDS:
                client.log_metric(
                    run_id, field_name, getattr(metrics, field_name), step=metrics.epoch
                )

        client.log_metric(run_id, "best_epoch", result.best.epoch)
        client.log_metric(run_id, "best_val_accuracy", result.best.val_accuracy)
        client.log_metric(run_id, "best_val_macro_f1", result.best.val_macro_f1)
        client.log_metric(run_id, "best_val_loss", result.best.val_loss)
        client.log_metric(run_id, "duration_seconds", duration)
        client.log_metric(run_id, "peak_memory_mb", _peak_memory_mb())
        device = _device()
        client.set_tag(run_id, "device", device)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            checkpoint_path = tmp_path / "model.pt"
            torch.save(result.model.state_dict(), checkpoint_path)
            checkpoint_sha_local = _sha256(checkpoint_path)
            client.log_artifact(run_id, str(checkpoint_path), CHECKPOINT_DIR)

            (tmp_path / "training_config.json").write_text(
                config.model_dump_json(indent=2), encoding="utf-8"
            )
            client.log_artifact(run_id, str(tmp_path / "training_config.json"), CHECKPOINT_DIR)

            (tmp_path / "class_map.json").write_text(
                json.dumps(dict(CLASS_MAP), indent=2), encoding="utf-8"
            )
            client.log_artifact(run_id, str(tmp_path / "class_map.json"), CHECKPOINT_DIR)

            environment = {"python": platform.python_version(), "torch": torch.__version__}
            (tmp_path / "environment.json").write_text(
                json.dumps(environment, indent=2), encoding="utf-8"
            )
            client.log_artifact(run_id, str(tmp_path / "environment.json"), CHECKPOINT_DIR)

        status = "FINISHED"
    except Exception as error:
        client.set_terminated(run_id, "FAILED")
        raise RuntimeError(f"Entrenamiento corto falló; run {run_id} quedó FAILED") from error
    client.set_terminated(run_id, status)

    checkpoint_artifact_path = f"{CHECKPOINT_DIR}/model.pt"
    served_sha = _download_sha256(client, run_id, checkpoint_artifact_path)
    if served_sha != checkpoint_sha_local:
        raise RuntimeError("El checkpoint descargado del servidor no coincide con el guardado")

    evidence = Evidence(
        tracking_uri=settings.mlflow_tracking_uri,
        experiment_name=settings.experiment_name,
        experiment_id=experiment_id,
        run_id=run_id,
        job_id=job_id,
        config=config.model_dump(),
        classes=classes,
        seed=seed,
        git_commit=git_commit,
        dvc_release=FIXTURE_DVC_RELEASE,
        dvc_images_md5=images_md5,
        dvc_annotations_md5=annotations_md5,
        dvc_release_hash=release_hash,
        manifest_version=FIXTURE_MANIFEST_VERSION,
        manifest_hash=FIXTURE_MANIFEST_HASH,
        epochs=len(result.history),
        best_epoch=result.best.epoch,
        best_val_accuracy=result.best.val_accuracy,
        best_val_macro_f1=result.best.val_macro_f1,
        best_val_loss=result.best.val_loss,
        checkpoint_artifact_path=checkpoint_artifact_path,
        checkpoint_sha256=served_sha,
        duration_seconds=duration,
        device=device,
        peak_memory_mb=_peak_memory_mb(),
        written_at=datetime.now(UTC).isoformat(),
    )
    evidence_path.write_text(json.dumps(asdict(evidence), indent=2) + "\n", encoding="utf-8")
    return evidence


def check_evidence(settings: TrackingSettings, evidence: Evidence) -> list[str]:
    """Devuelve las diferencias entre el servidor y la evidencia (vacía = OK)."""
    client = _connect(settings.mlflow_tracking_uri)
    problems: list[str] = []

    try:
        run = client.get_run(evidence.run_id)
    except Exception:
        return [f"El run {evidence.run_id} ya no existe"]

    if run.info.status != "FINISHED":
        problems.append(f"Estado del run: {run.info.status} (esperado FINISHED)")
    if run.data.tags.get(RUN_KIND_TAG) != RUN_KIND:
        problems.append(f"Falta el tag {RUN_KIND_TAG}={RUN_KIND}")
    if run.data.tags.get(DATA_PROVENANCE_TAG) != DATA_PROVENANCE_FIXTURE:
        problems.append(f"Falta el tag {DATA_PROVENANCE_TAG}={DATA_PROVENANCE_FIXTURE}")

    expected_dvc_hash = dvc_release_hash(evidence.dvc_images_md5, evidence.dvc_annotations_md5)
    if run.data.tags.get("dvc_release_hash") != expected_dvc_hash:
        problems.append("dvc_release_hash no coincide con sha256(images_md5:annotations_md5)")
    for tag_name, expected in (
        ("job_id", evidence.job_id),
        ("git_commit", evidence.git_commit),
        ("dvc_release", evidence.dvc_release),
        ("dvc_images_md5", evidence.dvc_images_md5),
        ("dvc_annotations_md5", evidence.dvc_annotations_md5),
        ("manifest_version", evidence.manifest_version),
        ("manifest_hash", evidence.manifest_hash),
        ("classes", ",".join(evidence.classes)),
        ("seed", str(evidence.seed)),
    ):
        if run.data.tags.get(tag_name) != expected:
            problems.append(
                f"Tag {tag_name} no coincide: {run.data.tags.get(tag_name)!r} != {expected!r}"
            )

    for name, expected in evidence.config.items():
        served = run.data.params.get(name)
        if served != str(expected):
            problems.append(f"Param {name} no coincide: {served!r} != {expected!r}")

    for field_name in EPOCH_METRIC_FIELDS:
        history = client.get_metric_history(evidence.run_id, field_name)
        if len(history) != evidence.epochs:
            problems.append(
                f"Métrica {field_name}: {len(history)} puntos, se esperaban {evidence.epochs}"
            )

    best_metrics = {
        "best_epoch": evidence.best_epoch,
        "best_val_accuracy": evidence.best_val_accuracy,
        "best_val_macro_f1": evidence.best_val_macro_f1,
        "best_val_loss": evidence.best_val_loss,
    }
    for name, expected in best_metrics.items():
        served = run.data.metrics.get(name)
        if served is None or round(served, 6) != round(float(expected), 6):
            problems.append(f"Summary {name} no coincide: {served} != {expected}")

    try:
        served_sha = _download_sha256(client, evidence.run_id, evidence.checkpoint_artifact_path)
    except ArtifactDownloadError as error:
        problems.append(f"No se pudo descargar el checkpoint: {error}")
    else:
        if served_sha != evidence.checkpoint_sha256:
            problems.append(f"SHA-256 del checkpoint: {served_sha} != {evidence.checkpoint_sha256}")

    return problems


def _load_evidence(path: Path) -> Evidence:
    return Evidence(**json.loads(path.read_text(encoding="utf-8")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tracking.short_run", description=__doc__)
    parser.add_argument("command", choices=["run", "check"])
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    try:
        settings = TrackingSettings()
    except Exception as error:
        print(f"Configuración inválida (¿falta MLFLOW_TRACKING_URI?): {error}", file=sys.stderr)
        return UNAVAILABLE

    try:
        if args.command == "run":
            evidence = run_short_training(settings, args.evidence, seed=args.seed)
            print(json.dumps(asdict(evidence), indent=2))
            print(f"Evidencia escrita en {args.evidence}")
            return VERIFIED

        try:
            evidence = _load_evidence(args.evidence)
        except (OSError, ValueError, TypeError) as error:
            print(f"Evidencia ilegible en {args.evidence}: {error}", file=sys.stderr)
            return UNAVAILABLE
        problems = check_evidence(settings, evidence)
    except TrackingUnavailableError as error:
        print(str(error), file=sys.stderr)
        return UNAVAILABLE
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return UNAVAILABLE

    if problems:
        print("NO coincide: " + "; ".join(problems), file=sys.stderr)
        return MISMATCH
    print(
        f"Run verificado en {settings.mlflow_tracking_uri}: "
        f"run_id={evidence.run_id} best_val_accuracy={evidence.best_val_accuracy} "
        f"checkpoint_sha256={evidence.checkpoint_sha256}"
    )
    return VERIFIED


if __name__ == "__main__":
    raise SystemExit(main())
