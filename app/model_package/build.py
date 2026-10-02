"""D05-01 — Construcción del paquete smoke a partir del checkpoint de un run.

Entrada: el directorio `checkpoint/` que el trainer de D03-04 sube al run de MLflow
(`model.pt` de la mejor época, `training_config.json`, `class_map.json`, `sources.json`).
Los pesos se copian tal cual: no se reconstruyen ni se toma otra época.

La salida de referencia se calcula aquí con la CNN de D01-04 y el state_dict del checkpoint
cargado directamente (sin pasar por el loader), para que la recarga en un proceso limpio se
compare con una salida previa del mismo checkpoint.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import torch
from pydantic import ValidationError

from model_package.format import (
    CARD_FILE,
    CHECKPOINT_ARTIFACT,
    CLASS_MAP_FILE,
    CONFIG_FILE,
    DEPENDENCIES_FILE,
    FORMAT_NAME,
    FORMAT_NOTE,
    FORMAT_VERSION,
    MANIFEST_FILE,
    MODEL_NAME,
    PAYLOAD_FILES,
    PREPROCESSING_FILE,
    REFERENCE_FILE,
    REQUIRED_DEPENDENCIES,
    WEIGHTS_FILE,
    FileEntry,
    PackageError,
    PackageManifest,
    Source,
    package_id,
    preprocessing_descriptor,
    reference_image,
    reference_image_sha256,
    sha256_file,
)
from model_package.loader import installed_version
from training.class_map import CLASS_MAP
from training.config import TrainingConfig
from training.model import build_model
from training.preprocessing import build_eval_transform

APP_ROOT = Path(__file__).resolve().parents[1]
LOCKFILE = APP_ROOT / "uv.lock"


def _read(folder: Path, name: str) -> dict:
    path = folder / name
    if not path.is_file():
        raise PackageError(f"el checkpoint no trae {name}")
    return json.loads(path.read_text(encoding="utf-8"))


def _reference_output(weights: Path, config: TrainingConfig) -> dict:
    model = build_model(config.model_copy(update={"pretrained": False}))
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    with torch.no_grad():
        tensor = build_eval_transform(config)(reference_image()).unsqueeze(0)
        probabilities = torch.softmax(model(tensor), dim=1)[0].tolist()
    classes = sorted(CLASS_MAP, key=CLASS_MAP.__getitem__)
    by_class = {name: probabilities[CLASS_MAP[name]] for name in classes}
    return {
        "input": "model_package.format.reference_image(): degradado RGB determinista 251x237",
        "input_sha256": reference_image_sha256(),
        "computed_with": "CNN de D01-04 + state_dict del checkpoint de origen (sin el loader)",
        "predicted_class": max(by_class, key=by_class.__getitem__),
        "probabilities": by_class,
    }


def _dependencies(checkpoint: Path) -> dict:
    versions = {name: installed_version(name) for name in REQUIRED_DEPENDENCIES}
    missing = [name for name, version in versions.items() if version is None]
    if missing:
        raise PackageError(f"no se puede empaquetar sin {missing} instalados")
    training_env = checkpoint / "environment.json"
    return {
        "required": versions,
        "rule": "Recarga con las mismas MAJOR.MINOR; versiones exactas en app/uv.lock.",
        "lockfile": "app/uv.lock",
        "lockfile_sha256": sha256_file(LOCKFILE),
        "training_environment": (
            json.loads(training_env.read_text(encoding="utf-8")) if training_env.is_file() else None
        ),
    }


def _smoke_card(run_id: str, best_epoch: int) -> dict:
    return {
        "kind": "smoke",
        "model": MODEL_NAME,
        "mlflow_run_id": run_id,
        "best_epoch": best_epoch,
        "official_test_metrics": None,
        "statement": (
            "Paquete smoke para probar la recarga fuera del entrenamiento. No hay métricas "
            "oficiales de test: el frozen test, el package final y la model card oficial "
            "son de D06-02."
        ),
    }


def build_smoke_package(
    checkpoint: Path | str,
    out: Path | str,
    *,
    run_id: str,
    experiment: str,
    expected_sha256: str | None = None,
    created_at: str | None = None,
) -> PackageManifest:
    """`expected_sha256` es el tag `checkpoint_sha256` del run, si se conoce."""
    checkpoint, out = Path(checkpoint), Path(out)
    if out.exists():
        raise PackageError(f"{out} ya existe: un paquete no se sobrescribe")
    if experiment != MODEL_NAME:
        raise PackageError(f"el run es del experimento {experiment!r}, no de {MODEL_NAME}")
    weights = checkpoint / WEIGHTS_FILE
    if not weights.is_file():
        raise PackageError(f"el checkpoint no trae {WEIGHTS_FILE}")
    sha = sha256_file(weights)
    if expected_sha256 is not None and sha != expected_sha256:
        raise PackageError(
            f"{WEIGHTS_FILE} ({sha}) no es el checkpoint_sha256 del run ({expected_sha256})"
        )
    sources = _read(checkpoint, "sources.json")
    if sources.get("checkpoint_sha256") != sha:
        raise PackageError(
            f"sources.json declara checkpoint_sha256={sources.get('checkpoint_sha256')} "
            f"y {WEIGHTS_FILE} es {sha}"
        )
    try:
        config = TrainingConfig.model_validate(_read(checkpoint, CONFIG_FILE))
        source = Source(
            mlflow_run_id=run_id,
            experiment=experiment,
            checkpoint_artifact=CHECKPOINT_ARTIFACT,
            checkpoint_sha256=sha,
            best_epoch=sources.get("best_epoch"),
            dataset_version=sources.get("dataset_version"),
            manifest_version=sources.get("manifest_version"),
            manifest_hash=sources.get("manifest_hash"),
            dvc_release_hash=sources.get("dvc_release_hash"),
        )
    except ValidationError as error:
        raise PackageError(f"checkpoint o run no empaquetable: {error}") from error
    class_map = _read(checkpoint, CLASS_MAP_FILE)
    if class_map != dict(CLASS_MAP):
        raise PackageError(f"{CLASS_MAP_FILE} {class_map} no es el congelado {dict(CLASS_MAP)}")

    # Se arma en un directorio temporal y se mueve al final: si algo falla no queda
    # un paquete a medias en `out`.
    with tempfile.TemporaryDirectory(dir=out.parent if out.parent.exists() else None) as tmp:
        stage = Path(tmp) / "package"
        stage.mkdir()
        shutil.copyfile(weights, stage / WEIGHTS_FILE)
        documents = {
            CONFIG_FILE: config.model_dump(),
            CLASS_MAP_FILE: class_map,
            PREPROCESSING_FILE: preprocessing_descriptor(config),
            DEPENDENCIES_FILE: _dependencies(checkpoint),
            REFERENCE_FILE: _reference_output(stage / WEIGHTS_FILE, config),
            CARD_FILE: _smoke_card(run_id, source.best_epoch),
        }
        for name, content in documents.items():
            (stage / name).write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
        manifest = PackageManifest(
            format=FORMAT_NAME,
            format_version=FORMAT_VERSION,
            kind="smoke",
            package_id=package_id(run_id),
            created_at=created_at or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            source=source,
            files={
                name: FileEntry(
                    sha256=sha256_file(stage / name), size_bytes=(stage / name).stat().st_size
                )
                for name in PAYLOAD_FILES
            },
            note=FORMAT_NOTE,
        )
        (stage / MANIFEST_FILE).write_text(
            manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(stage), str(out))
    return manifest
