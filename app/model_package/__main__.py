"""D05-01 — CLI del paquete smoke.

    # Construir desde el run real (descarga checkpoint/ por el servidor de MLflow):
    python -m model_package build --run-id <run_id> --tracking-uri http://mlflow:5000 \\
        --out /tmp/smoke-package
    # Construir desde un checkpoint/ ya descargado:
    python -m model_package build-from-dir --checkpoint-dir ./checkpoint --run-id <run_id> \\
        --out ./smoke-package
    # Cargar en un proceso limpio: identidad, inventario y predicción (JSON en stdout):
    python -m model_package predict --package ./smoke-package [--image foto.jpg]

Códigos de salida: 0 = ok · 1 = paquete rechazado o salida distinta de la de referencia ·
2 = servicios no disponibles.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from model_package.build import build_smoke_package
from model_package.format import MANIFEST_FILE, PackageError

OK, REJECTED, UNAVAILABLE = 0, 1, 2


def _print(payload: dict) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _build_from_run(args: argparse.Namespace) -> int:
    from tracking.settings import configure_client_env

    configure_client_env()
    from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=args.tracking_uri)
    try:
        run = client.get_run(args.run_id)
        experiment = client.get_experiment(run.info.experiment_id).name
        with tempfile.TemporaryDirectory() as tmp:
            repo = get_artifact_repository(run.info.artifact_uri)
            checkpoint = Path(repo.download_artifacts("checkpoint", tmp))
            if run.data.tags.get("p3.run_kind") != "training" or run.info.status != "FINISHED":
                raise PackageError(
                    f"el run {args.run_id} no es un training FINISHED "
                    f"({run.data.tags.get('p3.run_kind')}, {run.info.status})"
                )
            manifest = build_smoke_package(
                checkpoint,
                args.out,
                run_id=args.run_id,
                experiment=experiment,
                expected_sha256=run.data.tags.get("checkpoint_sha256"),
            )
    except PackageError:
        raise
    except Exception as error:
        print(f"MLflow no disponible o run ilegible: {type(error).__name__}: {error}")
        return UNAVAILABLE
    _print(json.loads(manifest.model_dump_json()))
    return OK


def _build_from_dir(args: argparse.Namespace) -> int:
    manifest = build_smoke_package(
        args.checkpoint_dir,
        args.out,
        run_id=args.run_id,
        experiment=args.experiment,
        expected_sha256=args.expected_sha256,
    )
    _print(json.loads(manifest.model_dump_json()))
    return OK


def _predict(args: argparse.Namespace) -> int:
    from PIL import Image

    from model_package.format import reference_image
    from model_package.loader import load_package

    package = load_package(args.package)
    reference = package.check_reference()
    image = Image.open(args.image) if args.image else reference_image()
    manifest = json.loads((Path(args.package) / MANIFEST_FILE).read_text(encoding="utf-8"))
    _print(
        {
            "identity": package.identity(),
            "inventory": manifest["files"],
            "input": str(args.image) if args.image else "reference_image()",
            "prediction": package.predict(image),
            "reference": reference,
        }
    )
    return OK if reference["matches"] else REJECTED


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D05-01 — paquete smoke y loader")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="desde el run de MLflow")
    build.add_argument("--run-id", required=True)
    build.add_argument("--tracking-uri", required=True)
    build.add_argument("--out", required=True, type=Path)
    build.set_defaults(handler=_build_from_run)

    from_dir = commands.add_parser("build-from-dir", help="desde un checkpoint/ descargado")
    from_dir.add_argument("--checkpoint-dir", required=True, type=Path)
    from_dir.add_argument("--run-id", required=True)
    from_dir.add_argument("--experiment", default="p3-cnn-classifier")
    from_dir.add_argument("--expected-sha256")
    from_dir.add_argument("--out", required=True, type=Path)
    from_dir.set_defaults(handler=_build_from_dir)

    predict = commands.add_parser("predict", help="carga el paquete y predice")
    predict.add_argument("--package", required=True, type=Path)
    predict.add_argument("--image", type=Path)
    predict.set_defaults(handler=_predict)

    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except PackageError as error:
        print(f"paquete rechazado: {error}", file=sys.stderr)
        return REJECTED


if __name__ == "__main__":
    sys.exit(main())
