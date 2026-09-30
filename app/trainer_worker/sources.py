"""D03-03 — Fuentes reales de Training: release aprobado + manifest P3 congelado.

`verify_training_sources` comprueba, contra los archivos reales, que un job de
entrenamiento puede usar (dataset_version, manifest_hash). `build_training_dataset`
adapta los crops reales al `TrainingDataset` del trainer con train/val únicamente:
los píxeles de la partición test nunca se cargan para entrenar.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from crops.models import Crop

from policies.models import QualityPolicy
from presentation.contracts import FrozenManifest, ManifestSummary
from presentation.release_resolver import ReleaseSource, ResolvedRelease
from trainer.dataset import TrainingDataset


class SourcesNotEligibleError(Exception):
    """Las fuentes pedidas no son elegibles para entrenar; `reason` es estable."""

    def __init__(self, reason: str, detail: str):
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class VerifiedSources:
    release: ResolvedRelease
    manifest: FrozenManifest
    summary: ManifestSummary
    crops: tuple[Crop, ...]
    images_dir: Path
    image_files: dict[int, str]


def verify_training_sources(
    dataset_version: str,
    manifest_hash: str,
    *,
    manifest_path: Path,
    repo_root: Path,
    reports_dir: Path,
    sources: dict[str, ReleaseSource],
    policy: QualityPolicy,
) -> VerifiedSources:
    raise NotImplementedError


def build_training_dataset(verified: VerifiedSources) -> TrainingDataset:
    raise NotImplementedError
