"""D05-04 — Motor de inferencia sobre el paquete de D05-01, independiente de HTTP.

El motor no reconstruye nada: carga el paquete con `load_package` (inventario, SHA-256,
class_map, preprocessing y dependencias validados ahí), coteja la salida de referencia y
recién entonces acepta imágenes. Cada predicción sale del mismo `LoadedPackage` que da su
identidad, así que nunca se atribuye a otro run.

Errores diferenciables para D05-07 (y D06-04, que recargará aquí el modelo de AWS):

    InputRejected     la imagen no se puede clasificar (vacía, no decodificable, formato
                      no admitido, demasiado grande)                      → HTTP 400/413/415
    PackageRejected   no hay paquete, falta, no pasa el loader, otro SHA o no reproduce
                      su salida de referencia                             → HTTP 503
    InferenceFailed   el modelo dio una salida incoherente con el class_map → HTTP 500

Ninguno de los tres emite predicción.
"""

from __future__ import annotations

import hashlib
import io
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from PIL import Image, UnidentifiedImageError

from model_package import LoadedPackage, PackageError, load_package

# Los formatos que acepta el upload del portal (D05-07: image/jpeg, image/png, image/webp).
ACCEPTED_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
MAX_IMAGE_BYTES = 10 * 1024 * 1024
PROBABILITY_SUM_TOLERANCE = 1e-4


class EngineError(Exception):
    kind: ClassVar[str] = "engine"


class InputRejected(EngineError):
    kind = "input"


class PackageRejected(EngineError):
    kind = "package"


class InferenceFailed(EngineError):
    kind = "inference"


@dataclass(frozen=True)
class Prediction:
    predicted_class: str
    probabilities: dict[str, float]
    model: dict
    input_sha256: str
    input_format: str
    input_size: tuple[int, int]

    def contract(self) -> dict:
        """Forma `inference_engine_prediction` de D05-07 (sin campos extra)."""
        return {
            "predicted_class": self.predicted_class,
            "probabilities": dict(self.probabilities),
            "model": dict(self.model),
        }


def model_identity(package: LoadedPackage) -> dict:
    """Identidad `InferenceModelIdentity` de D05-07. Un paquete smoke no tiene semver de
    modelo (lo asigna D06-02) ni objeto S3: el paquete local nunca se presenta como el
    official recargado de AWS (D06-06)."""
    identity = package.identity()
    return {
        "source": identity["kind"],
        "package_id": identity["package_id"],
        "format_version": identity["format_version"],
        "model_version": None,
        "mlflow_run_id": identity["mlflow_run_id"],
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "s3_object": None,
    }


def decode_image(data: bytes, *, max_bytes: int = MAX_IMAGE_BYTES) -> tuple[Image.Image, str]:
    """Decodifica completa la imagen (detecta archivos truncados): (imagen RGB, formato)."""
    if not data:
        raise InputRejected("imagen vacía")
    if len(data) > max_bytes:
        raise InputRejected(f"imagen de {len(data)} B: el máximo es {max_bytes} B")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in ACCEPTED_FORMATS:
                raise InputRejected(
                    f"formato {image.format} no admitido (se aceptan {sorted(ACCEPTED_FORMATS)})"
                )
            # convert() decodifica la imagen completa: un archivo truncado falla aquí.
            return image.convert("RGB"), image.format
    except InputRejected:
        raise
    except UnidentifiedImageError as error:
        raise InputRejected(
            "imagen no decodificable: no es un formato de imagen reconocido"
        ) from error
    except Image.DecompressionBombError as error:
        raise InputRejected(f"imagen no decodificable: {error}") from error
    except (OSError, ValueError, SyntaxError) as error:
        raise InputRejected(f"imagen no decodificable o truncada: {error}") from error


def _checked(raw: dict, classes: list[str]) -> tuple[str, dict[str, float]]:
    probabilities = raw.get("probabilities")
    if not isinstance(probabilities, dict) or list(probabilities) != classes:
        raise InferenceFailed(
            f"probabilidades {sorted(probabilities or {})} no corresponden al class_map {classes}"
        )
    values = [float(value) for value in probabilities.values()]
    if not all(math.isfinite(v) and 0.0 <= v <= 1.0 for v in values):
        raise InferenceFailed(f"probabilidades fuera de [0, 1]: {probabilities}")
    if abs(sum(values) - 1.0) > PROBABILITY_SUM_TOLERANCE:
        raise InferenceFailed(f"las probabilidades suman {sum(values)}, no ~1")
    best = max(classes, key=probabilities.__getitem__)
    if raw.get("predicted_class") != best:
        raise InferenceFailed(f"predicted_class {raw.get('predicted_class')} no es el argmax")
    return best, {name: float(probabilities[name]) for name in classes}


class InferenceEngine:
    """Una instancia sirve un paquete a la vez. `load` puede llamarse otra vez (D06-04) con
    el paquete recuperado de AWS: la lógica de predicción es la misma."""

    def __init__(self) -> None:
        self._package: LoadedPackage | None = None
        self._lock = threading.Lock()

    @classmethod
    def from_package(
        cls, folder: Path | str, *, expected_checkpoint_sha256: str | None = None
    ) -> InferenceEngine:
        engine = cls()
        engine.load(folder, expected_checkpoint_sha256=expected_checkpoint_sha256)
        return engine

    @property
    def loaded(self) -> bool:
        return self._package is not None

    def load(self, folder: Path | str, *, expected_checkpoint_sha256: str | None = None) -> dict:
        """Fail-closed: mientras carga y si el paquete se rechaza, el motor queda sin modelo;
        nunca sigue sirviendo el anterior bajo una petición de cambiarlo."""
        with self._lock:
            self._package = None
        folder = Path(folder)
        if not folder.exists():
            raise PackageRejected(f"paquete faltante: {folder} no existe")
        try:
            package = load_package(folder)
            sha = package.manifest.source.checkpoint_sha256
            if expected_checkpoint_sha256 is not None and sha != expected_checkpoint_sha256:
                raise PackageRejected(
                    f"el paquete trae el checkpoint {sha} del run "
                    f"{package.manifest.source.mlflow_run_id}; se esperaba "
                    f"{expected_checkpoint_sha256}"
                )
            reference = package.check_reference()
        except PackageError as error:
            raise PackageRejected(f"paquete rechazado: {error}") from error
        if not reference["matches"]:
            raise PackageRejected(
                "el paquete no reproduce su salida de referencia "
                f"(max_abs_diff {reference['max_abs_diff']})"
            )
        with self._lock:
            self._package = package
        return self.identity()

    def _current(self) -> LoadedPackage:
        package = self._package
        if package is None:
            raise PackageRejected("el motor no tiene un paquete cargado")
        return package

    def identity(self) -> dict:
        """Forma `inference_engine` de D05-07."""
        package = self._current()
        return {
            "model": model_identity(package),
            "classes": package.classes,
            "image_size": package.config.image_size,
        }

    def predict(self, data: bytes) -> Prediction:
        # Un solo paquete por predicción: el modelo que predice es el que da la identidad,
        # aunque otro hilo llame a `load` en medio.
        package = self._current()
        image, input_format = decode_image(data)
        try:
            raw = package.predict(image)
        except Exception as error:
            raise InferenceFailed(
                f"falló la inferencia: {type(error).__name__}: {error}"
            ) from error
        predicted, probabilities = _checked(raw, package.classes)
        return Prediction(
            predicted_class=predicted,
            probabilities=probabilities,
            model=model_identity(package),
            input_sha256=hashlib.sha256(data).hexdigest(),
            input_format=input_format,
            input_size=image.size,
        )
