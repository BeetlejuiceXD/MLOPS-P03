"""D05-01 — Formato versionado del paquete smoke.

Un paquete es un directorio con `package.json` (manifiesto + inventario con SHA-256 y
tamaño de cada archivo) y exactamente estos archivos:

    model.pt               pesos: el checkpoint de origen tal cual (mejor época del run)
    training_config.json   arquitectura/config de D01-04 (TrainingConfig)
    class_map.json         índice <-> clase (orden congelado de #33)
    preprocessing.json     transform de evaluación de D01-04, descrito
    dependencies.json      versiones necesarias para recargar + hash de app/uv.lock
    reference_output.json  salida del checkpoint de origen para una entrada conocida
    smoke_card.json        tarjeta smoke: sin métricas oficiales de test

`format_version` es la versión del FORMATO; no es el semver del modelo, que asigna D06-02
al package final.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Literal

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from training.config import TrainingConfig
from training.preprocessing import IMAGENET_MEAN, IMAGENET_STD

FORMAT_NAME = "p3-model-package"
FORMAT_VERSION = "1.0.0"
SUPPORTED_FORMAT_MAJOR = 1
MANIFEST_FILE = "package.json"
WEIGHTS_FILE = "model.pt"
CONFIG_FILE = "training_config.json"
CLASS_MAP_FILE = "class_map.json"
PREPROCESSING_FILE = "preprocessing.json"
DEPENDENCIES_FILE = "dependencies.json"
REFERENCE_FILE = "reference_output.json"
CARD_FILE = "smoke_card.json"
PAYLOAD_FILES = (
    WEIGHTS_FILE,
    CONFIG_FILE,
    CLASS_MAP_FILE,
    PREPROCESSING_FILE,
    DEPENDENCIES_FILE,
    REFERENCE_FILE,
    CARD_FILE,
)
# Lo que hace falta instalado para recargar: Python, la CNN (torch/torchvision) y la
# decodificación de imágenes (pillow).
REQUIRED_DEPENDENCIES = ("python", "torch", "torchvision", "pillow")
CHECKPOINT_ARTIFACT = "checkpoint/model.pt"
MODEL_NAME = "p3-cnn-classifier"
FORMAT_NOTE = (
    "format_version es la versión del formato del paquete smoke; no es el semver del "
    "modelo, que asigna D06-02 al package final."
)
REFERENCE_TOLERANCE = 1e-5

Sha256 = Field(pattern=r"^[0-9a-f]{64}$")


class PackageError(Exception):
    """El paquete no se puede cargar tal cual: nunca hay fallback a otro artefacto."""


class FileEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    sha256: str = Sha256
    size_bytes: int = Field(gt=0)


class Source(BaseModel):
    """De dónde salen los pesos: el run de MLflow y su checkpoint de la mejor época."""

    model_config = ConfigDict(extra="forbid", strict=True)

    mlflow_run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    experiment: Literal["p3-cnn-classifier"]
    checkpoint_artifact: Literal["checkpoint/model.pt"]
    checkpoint_sha256: str = Sha256
    best_epoch: int = Field(ge=1)
    dataset_version: str = Field(min_length=1)
    manifest_version: str = Field(min_length=1)
    manifest_hash: str = Sha256
    dvc_release_hash: str = Sha256


class PackageManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    format: Literal["p3-model-package"]
    format_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    kind: Literal["smoke"]
    package_id: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    source: Source
    files: dict[str, FileEntry]
    note: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocessing_descriptor(config: TrainingConfig) -> dict:
    """Descripción del transform de evaluación de D01-04 (`build_eval_transform`)."""
    return {
        "transform": "eval",
        "color_mode": "RGB",
        "image_size": config.image_size,
        "steps": ["resize", "to_tensor", "normalize"],
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
    }


def package_id(run_id: str) -> str:
    return f"{MODEL_NAME}-smoke-{run_id[:12]}"


def release_version(version: str) -> tuple[int, ...]:
    """'2.14.0+cu130' -> (2, 14, 0): la etiqueta local (+cpu, +cu130) no cambia la API."""
    numbers = re.findall(r"\d+", version.split("+", 1)[0])
    return tuple(int(n) for n in numbers[:3])


def reference_image() -> Image.Image:
    """Entrada conocida, sin aleatoriedad ni archivos: un degradado determinista de
    251x237 (tamaño distinto de image_size, para que el resize forme parte del cotejo)."""
    width, height = 251, 237
    pixels = bytes(
        channel
        for y in range(height)
        for x in range(width)
        for channel in ((x * 7 + y * 3) % 256, (x * 2 + y * 5) % 256, (x * y) % 256)
    )
    return Image.frombytes("RGB", (width, height), pixels)


def reference_image_sha256() -> str:
    return hashlib.sha256(reference_image().tobytes()).hexdigest()
