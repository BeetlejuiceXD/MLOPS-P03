"""Reglas de qué no puede estar en el índice de Git (D01-06, review PR #39).

No empieza con `test_`, así pytest no lo recolecta. Lo usan `test_tracked_files.py`
(sobre el índice real del repo y sobre un repo temporal con `git add -f`).

`.gitignore` solo evita que un archivo entre por accidente; `git add -f` lo salta.
Por eso estas reglas revisan lo que realmente está versionado.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

WEIGHT_SUFFIXES = frozenset({".pt", ".pth", ".ckpt", ".onnx", ".h5", ".keras", ".safetensors"})
ENVIRONMENT_DIRS = frozenset({".venv", "node_modules"})
MLFLOW_STORE_DIRS = frozenset({"mlruns", "mlartifacts"})
# Contenido gestionado por DVC: en Git solo van los punteros `data/raw/*.dvc`.
DVC_MANAGED_PREFIXES = ("data/raw/images/", "data/raw/annotations/")
CROPS_PREFIX = "data/crops/"


def _reason(path: str) -> str | None:
    posix = PurePosixPath(path)
    parts = posix.parts
    name = posix.name

    if name == ".env" or (name.startswith(".env.") and not name.endswith(".example")):
        return "variables de entorno locales (.env)"
    if ".aws" in parts:
        return "credenciales o configuración de AWS"
    if path == ".dvc/config.local" or path.endswith("/.dvc/config.local"):
        return "credenciales locales de DVC"
    if MLFLOW_STORE_DIRS.intersection(parts):
        return "store local de MLflow"
    if path.startswith(CROPS_PREFIX):
        return "crops derivados (van por DVC)"
    if path.startswith(DVC_MANAGED_PREFIXES):
        return "datos gestionados por DVC"
    if ENVIRONMENT_DIRS.intersection(parts):
        return "entorno o dependencias instaladas"
    if ".terraform" in parts or name.endswith(".tfstate") or ".tfstate." in name:
        return "estado local de Terraform"
    if posix.suffix.lower() in WEIGHT_SUFFIXES:
        return "pesos de modelo (van a MLflow/S3)"
    return None


def forbidden_tracked_paths(paths: Iterable[str]) -> list[tuple[str, str]]:
    """Devuelve `(ruta, motivo)` de cada ruta versionada que no debería estarlo."""
    findings = []
    for path in paths:
        reason = _reason(path)
        if reason is not None:
            findings.append((path, reason))
    return findings


def tracked_paths(repo: Path) -> list[str]:
    """Rutas en el índice de Git (`git ls-files`), en formato POSIX relativo."""
    result = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=True,
    )
    return [name for name in result.stdout.split("\0") if name]
