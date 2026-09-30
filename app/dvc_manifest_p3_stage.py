"""Etapa DVC `manifest_p3` (D03-01): congela el candidato de D02-04 del release
`DATASET_VERSION` (v0.1.1 por defecto) y escribe train_val/test (cacheados en DVC)
más el resumen y el registro de congelación en `reports/` (versionados en git).

Si la auditoría bloquea la congelación, la etapa falla sin escribir nada."""

import os
import sys
from pathlib import Path

from policies.models import load_quality_policy
from presentation.manifest_freeze import (
    FreezeBlockedError,
    freeze_manifest,
    write_frozen_manifest,
)
from presentation.release_resolver import load_release_sources

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_VERSION = os.environ.get("DATASET_VERSION", "v0.1.1")


def freeze() -> None:
    frozen = freeze_manifest(
        DATASET_VERSION,
        repo_root=REPO_ROOT,
        reports_dir=REPO_ROOT / "reports",
        sources=load_release_sources(),
        policy=load_quality_policy(),
    )
    write_frozen_manifest(
        frozen,
        manifest_dir=REPO_ROOT / "data" / "manifest_p3",
        reports_dir=REPO_ROOT / "reports",
    )
    print(f"manifest congelado: {frozen.summary.manifest_version} {frozen.summary.manifest_hash}")


if __name__ == "__main__":
    try:
        freeze()
    except FreezeBlockedError as error:
        print(f"Congelación BLOQUEADA: {error}", file=sys.stderr)
        raise SystemExit(1) from error
