"""D02-01 — Verifica que MLflow persiste runs, métricas y artefactos.

Uso (con `MLFLOW_TRACKING_URI` definido):

    python -m tracking.verify write --evidence evidencia.json
    # reiniciar servicios (docker compose down && docker compose up -d)
    python -m tracking.verify check --evidence evidencia.json

`write` crea (o reutiliza) el experimento `p3-cnn-classifier`, registra un run de
verificación con una métrica por pasos y un artefacto, descarga el artefacto desde
el servidor y guarda IDs y SHA-256 en el archivo de evidencia.
`check` vuelve a consultar todo en el servidor y lo compara con esa evidencia.

El run lleva el tag `p3.run_kind=persistence_check`: no es un run de la campaña y
Experiments/selección deben excluirlo.

Códigos de salida: 0 = verificado · 1 = el servidor responde pero algo no coincide ·
2 = servidor no disponible o evidencia ilegible. Nunca reporta éxito sin haber
leído del servidor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from tracking.settings import TrackingSettings, configure_client_env

VERIFIED = 0
MISMATCH = 1
UNAVAILABLE = 2

RUN_NAME = "d02-01-persistence-check"
RUN_KIND_TAG = "p3.run_kind"
RUN_KIND = "persistence_check"
METRIC = "persistence_check_value"
METRIC_STEPS = (0.25, 0.5, 0.75)
ARTIFACT_DIR = "persistence-check"
ARTIFACT_NAME = "evidence.txt"


@dataclass(frozen=True)
class Evidence:
    tracking_uri: str
    experiment_name: str
    experiment_id: str
    run_id: str
    metric: str
    metric_history: list[float]
    artifact_path: str
    artifact_sha256: str
    written_at: str


class TrackingUnavailableError(RuntimeError):
    """El servidor de MLflow no respondió."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _connect(tracking_uri: str):
    """Crea el cliente y hace una lectura real; cualquier fallo = no disponible."""
    configure_client_env()
    import mlflow
    from mlflow.tracking import MlflowClient

    try:
        # Global además del cliente: las URIs `mlflow-artifacts:/` de los runs se
        # resuelven contra el tracking URI global al descargar artefactos.
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        client.search_experiments(max_results=1)
    except Exception as error:  # el cliente envuelve errores de red en varias clases
        raise TrackingUnavailableError(
            f"MLflow no disponible en {tracking_uri}: {type(error).__name__}"
        ) from error
    return client


class ArtifactDownloadError(RuntimeError):
    """El servidor respondió, pero no devolvió el artefacto."""


def _download_sha256(client, run_id: str, artifact_path: str) -> str:
    """Descarga desde el repositorio de artefactos del run y devuelve su SHA-256.

    No usa `client.download_artifacts`: su `RunsArtifactRepository` registra el error
    real (p. ej. de S3/MinIO) solo a nivel debug y lanza un mensaje genérico.
    """
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


def write_evidence(settings: TrackingSettings, evidence_path: Path) -> Evidence:
    client = _connect(settings.mlflow_tracking_uri)

    experiment = client.get_experiment_by_name(settings.experiment_name)
    experiment_id = (
        experiment.experiment_id
        if experiment is not None
        else client.create_experiment(settings.experiment_name)
    )
    run = client.create_run(
        experiment_id,
        run_name=RUN_NAME,
        tags={RUN_KIND_TAG: RUN_KIND, "p3.ticket": "D02-01"},
    )
    run_id = run.info.run_id
    for step, value in enumerate(METRIC_STEPS):
        client.log_metric(run_id, METRIC, value, step=step)

    written_at = datetime.now(UTC).isoformat()
    with tempfile.TemporaryDirectory() as tmp:
        artifact = Path(tmp) / ARTIFACT_NAME
        artifact.write_text(
            f"D02-01 persistence check\nrun_id={run_id}\nwritten_at={written_at}\n",
            encoding="utf-8",
        )
        local_sha = _sha256(artifact)
        client.log_artifact(run_id, str(artifact), ARTIFACT_DIR)
    client.set_terminated(run_id, "FINISHED")

    artifact_path = f"{ARTIFACT_DIR}/{ARTIFACT_NAME}"
    served_sha = _download_sha256(client, run_id, artifact_path)
    if served_sha != local_sha:
        raise RuntimeError("El artefacto descargado del servidor no coincide con el subido")

    evidence = Evidence(
        tracking_uri=settings.mlflow_tracking_uri,
        experiment_name=settings.experiment_name,
        experiment_id=experiment_id,
        run_id=run_id,
        metric=METRIC,
        metric_history=list(METRIC_STEPS),
        artifact_path=artifact_path,
        artifact_sha256=served_sha,
        written_at=written_at,
    )
    evidence_path.write_text(json.dumps(asdict(evidence), indent=2) + "\n", encoding="utf-8")
    return evidence


def check_evidence(settings: TrackingSettings, evidence: Evidence) -> list[str]:
    """Devuelve las diferencias entre el servidor y la evidencia (vacía = persiste)."""
    client = _connect(settings.mlflow_tracking_uri)
    problems: list[str] = []

    experiment = client.get_experiment_by_name(evidence.experiment_name)
    if experiment is None:
        return [f"El experimento {evidence.experiment_name} ya no existe"]
    if experiment.experiment_id != evidence.experiment_id:
        problems.append(
            f"experiment_id cambió: {evidence.experiment_id} → {experiment.experiment_id}"
        )

    try:
        run = client.get_run(evidence.run_id)
    except Exception:
        return [*problems, f"El run {evidence.run_id} ya no existe"]
    if run.info.experiment_id != evidence.experiment_id:
        problems.append("El run pertenece a otro experimento")
    if run.info.status != "FINISHED":
        problems.append(f"Estado del run: {run.info.status} (esperado FINISHED)")
    if run.data.tags.get(RUN_KIND_TAG) != RUN_KIND:
        problems.append(f"Falta el tag {RUN_KIND_TAG}={RUN_KIND}")

    history = sorted(
        client.get_metric_history(evidence.run_id, evidence.metric), key=lambda m: m.step
    )
    values = [metric.value for metric in history]
    if values != evidence.metric_history:
        problems.append(f"Historial de {evidence.metric}: {values} ≠ {evidence.metric_history}")

    try:
        served_sha = _download_sha256(client, evidence.run_id, evidence.artifact_path)
    except ArtifactDownloadError as error:
        problems.append(f"No se pudo descargar el artefacto: {error}")
    else:
        if served_sha != evidence.artifact_sha256:
            problems.append(f"SHA-256 del artefacto: {served_sha} ≠ {evidence.artifact_sha256}")
    return problems


def _load_evidence(path: Path) -> Evidence:
    return Evidence(**json.loads(path.read_text(encoding="utf-8")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tracking.verify", description=__doc__)
    parser.add_argument("command", choices=["write", "check"])
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        settings = TrackingSettings()
    except Exception as error:
        print(f"Configuración inválida (¿falta MLFLOW_TRACKING_URI?): {error}", file=sys.stderr)
        return UNAVAILABLE

    try:
        if args.command == "write":
            evidence = write_evidence(settings, args.evidence)
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
    except ArtifactDownloadError as error:
        print(f"NO persiste: el servidor no devuelve el artefacto: {error}", file=sys.stderr)
        return MISMATCH

    if problems:
        print("NO persiste: " + "; ".join(problems), file=sys.stderr)
        return MISMATCH
    print(
        f"Persistencia verificada en {settings.mlflow_tracking_uri}: "
        f"experiment_id={evidence.experiment_id} run_id={evidence.run_id} "
        f"artifact_sha256={evidence.artifact_sha256}"
    )
    return VERIFIED


if __name__ == "__main__":
    raise SystemExit(main())
