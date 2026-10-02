"""D05-01 — Paquete smoke versionado y loader autocontenido (ver README.md)."""

from model_package.build import build_smoke_package
from model_package.format import (
    FORMAT_VERSION,
    MANIFEST_FILE,
    PAYLOAD_FILES,
    PackageError,
    reference_image,
)
from model_package.loader import LoadedPackage, load_package

__all__ = [
    "FORMAT_VERSION",
    "MANIFEST_FILE",
    "PAYLOAD_FILES",
    "LoadedPackage",
    "PackageError",
    "build_smoke_package",
    "load_package",
    "reference_image",
]
