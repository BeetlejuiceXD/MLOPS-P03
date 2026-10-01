"""D03-04 — Smoke real Training → MLflow → checkpoint.

Recorre el camino que usa el portal (`/api`): comprueba que el release aprobado y el
manifest congelado están publicados, crea un job `task=training`, lo sigue hasta su
estado terminal y contrasta ese MISMO job con su run de MLflow (`p3-cnn-classifier`) y
con su checkpoint descargado del servidor.

    python -m tracking.smoke run --api http://localhost:8080/api \\
        --tracking-uri http://localhost:5000 --seed 7 --evidence smoke-d03-04.json

`verify_smoke` es la parte comprobable sin servicios (tests/test_smoke.py). Nunca lee
el frozen test: solo pide train/val al worker y rechaza un run con métricas de test.
Un smoke NO es una corrida de la campaña (#33): queda identificado con
`--label` en la evidencia y no se cuenta para selección.

Códigos de salida: 0 = smoke verificado · 1 = algo no coincide · 2 = servicios o
fuentes no disponibles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tracking.settings import P3_EXPERIMENT, configure_client_env

VERIFIED, MISMATCH, UNAVAILABLE = 0, 1, 2
TERMINAL = ("succeeded", "failed", "cancelled")
EPOCH_METRICS = (
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "val_macro_f1",
    "learning_rate",
)
REQUIRED_TAGS = (
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
    "checkpoint_sha256",
)
CHECKPOINT_FILES = ("model.pt", "training_config.json", "class_map.json")


@dataclass
class SmokeReport:
    problems: list[str] = field(default_factory=list)
    run_id: str | None = None
    run_status: str | None = None
    experiment: str | None = None
    epochs_logged: int | None = None
    best_epoch: int | None = None
    best_val_accuracy: float | None = None
    best_val_macro_f1: float | None = None
    checkpoint_sha256: str | None = None
    manifest_hash: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    loaded_classes: list[str] | None = None
    probabilities_sum: float | None = None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _download(client, run_id: str, artifact_path: str, dest: Path) -> Path:
    """Desde el repositorio de artefactos del run, pasando por el servidor (como
    `tracking.verify`): con MinIO, el host no resuelve URLs prefirmadas."""
    from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

    artifact_uri = client.get_run(run_id).info.artifact_uri
    dest.mkdir(parents=True, exist_ok=True)
    return Path(get_artifact_repository(artifact_uri).download_artifacts(artifact_path, str(dest)))


def _check_run(job: dict, run, client, report: SmokeReport) -> None:
    problems = report.problems
    config = job["config"]
    experiment = client.get_experiment(run.info.experiment_id).name
    report.experiment = experiment
    report.run_status = run.info.status
    if experiment != P3_EXPERIMENT:
        problems.append(f"el run está en el experimento {experiment!r}, no en {P3_EXPERIMENT}")
    if run.info.status != "FINISHED":
        problems.append(f"el run quedó {run.info.status}, no FINISHED")

    tags = run.data.tags
    report.tags = {name: tags[name] for name in REQUIRED_TAGS if name in tags}
    missing = [name for name in REQUIRED_TAGS if not tags.get(name)]
    if missing:
        problems.append(f"faltan tags de procedencia: {missing}")
    expected = {
        "p3.run_kind": "training",
        "job_id": str(job["id"]),
        "dvc_release": job["dataset_version"],
        "manifest_hash": job["manifest_hash"],
        "seed": str(config["seed"]),
    }
    for name, value in expected.items():
        if tags.get(name) != value:
            problems.append(f"tag {name}={tags.get(name)!r} no corresponde al job ({value!r})")

    params = run.data.params
    for name, value in config.items():
        if params.get(name) != str(value):
            problems.append(f"param {name}={params.get(name)!r} distinto de la config del job")

    metrics = run.data.metrics
    leaked = sorted(name for name in metrics if name.startswith("test"))
    if leaked:
        problems.append(f"el run tiene métricas de test (prohibidas antes de D06-01): {leaked}")

    histories = {name: client.get_metric_history(run.info.run_id, name) for name in EPOCH_METRICS}
    lengths = {name: len(points) for name, points in histories.items()}
    report.epochs_logged = lengths["val_accuracy"]
    epoch = (job.get("progress") or {}).get("epoch")
    if set(lengths.values()) != {epoch}:
        problems.append(f"épocas registradas {lengths} no coinciden con el progreso {epoch}")

    if "best_epoch" not in metrics or "best_val_accuracy" not in metrics:
        problems.append("falta el resumen best_epoch/best_val_accuracy")
        return
    report.best_epoch = int(metrics["best_epoch"])
    report.best_val_accuracy = metrics["best_val_accuracy"]
    report.best_val_macro_f1 = metrics.get("best_val_macro_f1")
    at_epoch = {point.step: point.value for point in histories["val_accuracy"]}
    if abs(at_epoch.get(report.best_epoch, float("nan")) - report.best_val_accuracy) > 1e-4:
        problems.append("best_val_accuracy no corresponde a la curva en best_epoch")


def _check_checkpoint(job: dict, run, client, workdir: Path, report: SmokeReport) -> None:
    problems = report.problems
    try:
        files = {
            name: _download(client, run.info.run_id, f"checkpoint/{name}", workdir)
            for name in CHECKPOINT_FILES
        }
    except Exception as error:  # el servidor respondió pero no entregó el checkpoint
        problems.append(f"no se pudo descargar el checkpoint: {type(error).__name__}: {error}")
        return

    report.checkpoint_sha256 = _sha256(files["model.pt"])
    if report.checkpoint_sha256 != run.data.tags.get("checkpoint_sha256"):
        problems.append("sha256 del model.pt descargado distinto del tag checkpoint_sha256")

    from training.class_map import CLASS_MAP
    from training.config import TrainingConfig

    stored_config = json.loads(files["training_config.json"].read_text(encoding="utf-8"))
    if stored_config != job["config"]:
        problems.append("training_config.json del checkpoint distinto de la config del job")
    class_map = json.loads(files["class_map.json"].read_text(encoding="utf-8"))
    if class_map != dict(CLASS_MAP):
        problems.append(f"class_map.json {class_map} distinto del class map congelado {CLASS_MAP}")

    try:
        report.loaded_classes, report.probabilities_sum = _load_and_classify(
            files["model.pt"], TrainingConfig(**stored_config), class_map
        )
    except Exception as error:
        problems.append(f"no se pudo cargar el checkpoint en el modelo de su config: {error}")


def _load_and_classify(model_path: Path, config, class_map: dict[str, int]):
    """Modelo de la config (sin bajar pesos: el state_dict los reemplaza), carga
    estricta y una predicción con el transform de evaluación compartido."""
    import torch
    from PIL import Image

    from training.model import build_model
    from training.preprocessing import build_eval_transform

    model = build_model(config.model_copy(update={"pretrained": False}))
    model.load_state_dict(torch.load(model_path, weights_only=True), strict=True)
    model.eval()
    image = Image.new("RGB", (config.image_size + 17, config.image_size + 5), (120, 90, 60))
    with torch.no_grad():
        logits = model(build_eval_transform(config)(image).unsqueeze(0))
    probabilities = torch.softmax(logits, dim=1)[0]
    if probabilities.shape[0] != len(class_map):
        raise ValueError(f"{probabilities.shape[0]} salidas para {len(class_map)} clases")
    classes = [name for name, _ in sorted(class_map.items(), key=lambda item: item[1])]
    return classes, float(probabilities.sum())


def verify_smoke(job: dict, *, tracking_uri: str, workdir: Path) -> SmokeReport:
    """Contrasta un job terminado con su run y su checkpoint en el servidor."""
    report = SmokeReport(run_id=job.get("mlflow_run_id"), manifest_hash=job.get("manifest_hash"))
    if job.get("task") != "training" or job.get("status") != "succeeded":
        report.problems.append(
            f"el job #{job.get('id')} es {job.get('task')}/{job.get('status')}, "
            "no un training succeeded"
        )
    if not report.run_id:
        report.problems.append("el job no tiene run de MLflow")
        return report

    configure_client_env()
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(report.run_id)
    _check_run(job, run, client, report)
    _check_checkpoint(job, run, client, Path(workdir), report)
    return report


# --- recorrido por la API del portal ---------------------------------------------------


class ApiError(RuntimeError):
    def __init__(self, status: int, body: object):
        super().__init__(f"HTTP {status}: {body}")
        self.status, self.body = status, body


def _http(method: str, url: str, body: object | None = None) -> object:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read() or b"null")
    except urllib.error.HTTPError as error:
        raise ApiError(error.code, json.loads(error.read() or b"null")) from error


def build_request(manifest: dict, args: argparse.Namespace) -> dict:
    from training.config import TrainingConfig

    config = TrainingConfig(
        seed=args.seed,
        max_epochs=args.max_epochs,
        patience=args.patience,
        image_size=args.image_size,
        batch_size=args.batch_size,
        pretrained=args.pretrained,
    ).model_dump()
    return {
        "task": "training",
        "dataset_version": manifest["dataset_version"],
        "manifest_hash": manifest["manifest_hash"],
        "config": config,
    }


def run_smoke(args: argparse.Namespace) -> int:
    api = args.api.rstrip("/")
    started = time.monotonic()
    try:
        releases = _http("GET", f"{api}/releases")
        manifest = _http("GET", f"{api}/manifest")
    except (ApiError, urllib.error.URLError) as error:
        print(f"Fuentes no disponibles en {api}: {error}", file=sys.stderr)
        return UNAVAILABLE
    approved = [r["dataset_version"] for r in releases["approved"]]
    if not manifest.get("frozen") or manifest["dataset_version"] not in approved:
        print(f"Sin manifest congelado sobre un release aprobado: {approved}", file=sys.stderr)
        return UNAVAILABLE
    print(
        f"Fuentes: {manifest['dataset_version']} / {manifest['manifest_version']} "
        f"({manifest['manifest_hash'][:12]}…) train={manifest['splits']['train']['crops']} "
        f"val={manifest['splits']['val']['crops']}",
        flush=True,
    )

    request = build_request(manifest, args)
    job = _http("POST", f"{api}/training/jobs", request)
    print(f"Job #{job['id']} creado ({job['status']})", flush=True)
    last = None
    deadline = time.monotonic() + args.timeout
    while job["status"] not in TERMINAL:
        if time.monotonic() > deadline:
            print(f"Timeout: el job #{job['id']} sigue {job['status']}", file=sys.stderr)
            return MISMATCH
        time.sleep(args.poll)
        job = _http("GET", f"{api}/training/jobs/{job['id']}")
        progress = (job.get("progress") or {}).get("epoch")
        if (job["status"], progress) != last:
            last = (job["status"], progress)
            total = (job.get("progress") or {}).get("total_epochs")
            print(f"  {job['status']} época {progress}/{total}", flush=True)
    logs = _http("GET", f"{api}/training/jobs/{job['id']}/logs")

    with tempfile.TemporaryDirectory() as tmp:
        report = verify_smoke(job, tracking_uri=args.tracking_uri, workdir=Path(tmp))
    evidence = {
        "label": args.label,
        "api": api,
        "tracking_uri": args.tracking_uri,
        "seconds": round(time.monotonic() - started, 1),
        "sources": {"approved_releases": approved, "manifest": manifest},
        "request": request,
        "job": job,
        "logs": [line["message"] for line in logs["lines"]],
        "report": asdict(report),
    }
    if args.evidence:
        Path(args.evidence).write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(asdict(report), indent=2, ensure_ascii=False))
    if report.problems:
        print("SMOKE CON PROBLEMAS:", *report.problems, sep="\n  - ", file=sys.stderr)
        return MISMATCH
    print(
        f"SMOKE OK: job #{job['id']} → run {report.run_id} → checkpoint "
        f"{report.checkpoint_sha256[:12]}… cargado ({report.loaded_classes})"
    )
    return VERIFIED


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D03-04 — smoke real Training → MLflow")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="crea el job desde la API y verifica job/run/checkpoint")
    run.add_argument("--api", default="http://localhost:8080/api")
    run.add_argument("--tracking-uri", default="http://localhost:5000")
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--max-epochs", type=int, default=10)
    run.add_argument("--patience", type=int, default=3)
    run.add_argument("--image-size", type=int, default=224)
    run.add_argument("--batch-size", type=int, default=16)
    run.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    run.add_argument("--timeout", type=float, default=3600)
    run.add_argument("--poll", type=float, default=3)
    run.add_argument("--label", default="smoke-d03-04 (no es corrida de campaña)")
    run.add_argument("--evidence", type=Path)
    args = parser.parse_args(argv)
    return run_smoke(args)


if __name__ == "__main__":
    sys.exit(main())
