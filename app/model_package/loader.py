"""D05-01 — Loader del paquete smoke, sin estado del trainer.

Orden fijo: manifiesto → inventario (faltantes, extra, tamaño, SHA-256) → pesos = checkpoint
de origen → dependencias → config/arquitectura → class_map → preprocessing. Solo si todo
eso pasa se instancia la CNN de D01-04 y se cargan los pesos en modo estricto. Cualquier
discrepancia es `PackageError`: no se sirve predicción ni se busca otro artefacto.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import torch
from PIL import Image
from pydantic import ValidationError

from model_package.format import (
    CARD_FILE,
    CLASS_MAP_FILE,
    CONFIG_FILE,
    DEPENDENCIES_FILE,
    FORMAT_VERSION,
    MANIFEST_FILE,
    PAYLOAD_FILES,
    PREPROCESSING_FILE,
    REFERENCE_FILE,
    REFERENCE_TOLERANCE,
    REQUIRED_DEPENDENCIES,
    SUPPORTED_FORMAT_MAJOR,
    WEIGHTS_FILE,
    PackageError,
    PackageManifest,
    preprocessing_descriptor,
    reference_image,
    reference_image_sha256,
    release_version,
    sha256_file,
)
from training.class_map import CLASS_MAP
from training.config import TrainingConfig
from training.model import build_model
from training.preprocessing import build_eval_transform


def installed_version(name: str) -> str | None:
    """Versión instalada en ESTE proceso; None si la dependencia no está."""
    if name == "python":
        return ".".join(str(part) for part in sys.version_info[:3])
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _read_json(folder: Path, name: str) -> dict:
    try:
        return json.loads((folder / name).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise PackageError(f"{name} ilegible: {error}") from error


def _read_manifest(folder: Path) -> PackageManifest:
    if not folder.is_dir():
        raise PackageError(f"{folder} no es un directorio de paquete")
    if not (folder / MANIFEST_FILE).is_file():
        raise PackageError(f"falta {MANIFEST_FILE}: no es un paquete smoke")
    raw = _read_json(folder, MANIFEST_FILE)
    version = raw.get("format_version") if isinstance(raw, dict) else None
    if isinstance(version, str) and release_version(version)[:1] != (SUPPORTED_FORMAT_MAJOR,):
        raise PackageError(
            f"format_version {version} no soportado (este loader lee {FORMAT_VERSION})"
        )
    try:
        return PackageManifest.model_validate(raw)
    except ValidationError as error:
        raise PackageError(f"{MANIFEST_FILE} inválido: {error}") from error


def _check_inventory(folder: Path, manifest: PackageManifest) -> None:
    listed = set(manifest.files)
    if listed != set(PAYLOAD_FILES):
        raise PackageError(
            f"el inventario no es el del formato {FORMAT_VERSION}: "
            f"sobran {sorted(listed - set(PAYLOAD_FILES))}, "
            f"faltan {sorted(set(PAYLOAD_FILES) - listed)}"
        )
    present = {path.name for path in folder.iterdir()} - {MANIFEST_FILE}
    for extra in sorted(present - listed):
        raise PackageError(f"{extra} no está en el inventario del paquete")
    for name in PAYLOAD_FILES:
        path = folder / name
        if not path.is_file():
            raise PackageError(f"falta {name}")
        entry = manifest.files[name]
        size = path.stat().st_size
        if size != entry.size_bytes:
            raise PackageError(
                f"{name} truncado o de tamaño distinto: {size} B, se esperaban {entry.size_bytes} B"
            )
        actual = sha256_file(path)
        if actual != entry.sha256:
            raise PackageError(f"{name}: hash discordante ({actual}, inventario {entry.sha256})")
    if manifest.files[WEIGHTS_FILE].sha256 != manifest.source.checkpoint_sha256:
        raise PackageError(
            f"{WEIGHTS_FILE} no es el checkpoint de origen "
            f"({manifest.source.checkpoint_sha256} del run {manifest.source.mlflow_run_id})"
        )


def _check_dependencies(folder: Path) -> None:
    required = _read_json(folder, DEPENDENCIES_FILE).get("required")
    if not isinstance(required, dict):
        raise PackageError(f"{DEPENDENCIES_FILE} sin la sección required")
    for name in REQUIRED_DEPENDENCIES:
        wanted = required.get(name)
        if not isinstance(wanted, str) or not wanted:
            raise PackageError(f"{DEPENDENCIES_FILE} incompleto: falta la versión de {name}")
        have = installed_version(name)
        if have is None:
            raise PackageError(f"dependencia ausente: {name} (el paquete requiere {wanted})")
        if release_version(have)[:2] != release_version(wanted)[:2]:
            raise PackageError(
                f"{name} {have} incompatible con la del paquete ({wanted}): "
                "debe coincidir MAJOR.MINOR"
            )


def _read_config(folder: Path) -> TrainingConfig:
    try:
        return TrainingConfig.model_validate(_read_json(folder, CONFIG_FILE))
    except ValidationError as error:
        raise PackageError(f"{CONFIG_FILE} incompatible con la CNN de D01-04: {error}") from error


def _read_class_map(folder: Path) -> dict[str, int]:
    class_map = _read_json(folder, CLASS_MAP_FILE)
    if class_map != dict(CLASS_MAP):
        raise PackageError(
            f"{CLASS_MAP_FILE} {class_map} incompatible con el orden congelado "
            f"de #33 {dict(CLASS_MAP)}"
        )
    return class_map


def _check_preprocessing(folder: Path, config: TrainingConfig) -> None:
    described = _read_json(folder, PREPROCESSING_FILE)
    expected = preprocessing_descriptor(config)
    if described != expected:
        raise PackageError(
            f"{PREPROCESSING_FILE} incompatible con el transform de evaluación de D01-04 "
            f"para esta config: {described} != {expected}"
        )


def _build_model(folder: Path, config: TrainingConfig) -> torch.nn.Module:
    try:
        state = torch.load(folder / WEIGHTS_FILE, map_location="cpu", weights_only=True)
    except Exception as error:
        raise PackageError(f"{WEIGHTS_FILE} ilegible: {error}") from error
    # Sin pesos preentrenados descargados: el state_dict del paquete los reemplaza todos.
    model = build_model(config.model_copy(update={"pretrained": False}))
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise PackageError(
            f"los pesos no corresponden a la arquitectura de {CONFIG_FILE}: {error}"
        ) from error
    model.eval()
    return model


@dataclass(frozen=True)
class LoadedPackage:
    folder: Path
    manifest: PackageManifest
    config: TrainingConfig
    class_map: dict[str, int]
    model: torch.nn.Module

    @property
    def classes(self) -> list[str]:
        return sorted(self.class_map, key=self.class_map.__getitem__)

    def identity(self) -> dict:
        """Lo que un consumidor (D05-04, D06-02/04) necesita para saber qué cargó."""
        card = _read_json(self.folder, CARD_FILE)
        return {
            "format_version": self.manifest.format_version,
            "kind": self.manifest.kind,
            "package_id": self.manifest.package_id,
            "mlflow_run_id": self.manifest.source.mlflow_run_id,
            "checkpoint_sha256": self.manifest.source.checkpoint_sha256,
            "best_epoch": self.manifest.source.best_epoch,
            "classes": self.classes,
            "image_size": self.config.image_size,
            "official_test_metrics": card.get("official_test_metrics"),
        }

    def predict(self, image: Image.Image) -> dict:
        """Determinista: modo eval, sin gradientes y con el transform de evaluación."""
        tensor = build_eval_transform(self.config)(image.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0].tolist()
        by_class = {name: probabilities[self.class_map[name]] for name in self.classes}
        return {
            "predicted_class": max(by_class, key=by_class.__getitem__),
            "probabilities": by_class,
        }

    def check_reference(self) -> dict:
        """Compara con la salida que dio el checkpoint de origen al construir el paquete."""
        reference = _read_json(self.folder, REFERENCE_FILE)
        if reference.get("input_sha256") != reference_image_sha256():
            raise PackageError(f"{REFERENCE_FILE} no corresponde a la entrada conocida")
        current = self.predict(reference_image())["probabilities"]
        recorded = reference.get("probabilities", {})
        if set(recorded) != set(current):
            raise PackageError(f"{REFERENCE_FILE} con clases {sorted(recorded)}")
        diff = max(abs(current[name] - recorded[name]) for name in current)
        return {"matches": diff <= REFERENCE_TOLERANCE, "max_abs_diff": diff}


def load_package(folder: Path | str) -> LoadedPackage:
    folder = Path(folder)
    manifest = _read_manifest(folder)
    _check_inventory(folder, manifest)
    _check_dependencies(folder)
    config = _read_config(folder)
    class_map = _read_class_map(folder)
    _check_preprocessing(folder, config)
    model = _build_model(folder, config)
    return LoadedPackage(folder, manifest, config, class_map, model)
