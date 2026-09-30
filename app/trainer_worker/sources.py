"""D03-03 — Fuentes reales de Training: release aprobado + manifest P3 congelado.

`verify_training_sources` comprueba, contra los archivos reales, que un job de
entrenamiento puede usar (dataset_version, manifest_hash). `build_training_dataset`
adapta los crops reales al `TrainingDataset` del trainer con train/val únicamente:
los píxeles de la partición test nunca se cargan para entrenar.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml
from crops.models import Crop
from manifest.generator import ManifestValidationError, verify_manifest_assignment
from PIL import Image
from pydantic import ValidationError

from ingestion.loader import load_dataset
from policies.models import QualityPolicy
from presentation.contracts import (
    MANIFEST_CLASSES,
    FrozenManifest,
    ManifestSplitCounts,
    ManifestSplits,
    ManifestSummary,
    frozen_test_split_hash,
)
from presentation.crops_report import build_crops_report
from presentation.manifest_candidate import (
    ManifestBlockedError,
    _dvc_release_hash,
    _manifest_hash,
    build_manifest_candidate,
)
from presentation.release_resolver import (
    ReleaseRejectedError,
    ReleaseSource,
    ResolvedRelease,
    resolve_release,
)
from trainer.dataset import TrainingDataset, TrainingSample

SPLITS = ("train", "val", "test")


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


def _read_frozen_manifest(manifest_path: Path) -> FrozenManifest:
    """Lee el artefacto y exige que sea el que DVC versionó; no basta `frozen: true`."""
    if not manifest_path.is_file():
        raise SourcesNotEligibleError(
            "manifest_missing",
            f"No existe el manifest congelado de D03-01 en {manifest_path}",
        )
    content = manifest_path.read_bytes()
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as error:
        raise SourcesNotEligibleError("manifest_invalid", f"JSON inválido: {error}") from error
    if not isinstance(raw, dict) or raw.get("frozen") is not True:
        raise SourcesNotEligibleError(
            "manifest_not_frozen",
            "El manifest no está congelado (frozen debe ser true; D03-01 congela el candidato)",
        )

    dvc_file = manifest_path.with_name(manifest_path.name + ".dvc")
    if not dvc_file.is_file():
        raise SourcesNotEligibleError(
            "manifest_not_versioned",
            f"El manifest congelado debe estar versionado con DVC ({dvc_file.name} no existe)",
        )
    try:
        declared_md5 = yaml.safe_load(dvc_file.read_text(encoding="utf-8"))["outs"][0]["md5"]
    except (yaml.YAMLError, KeyError, IndexError, TypeError) as error:
        raise SourcesNotEligibleError(
            "manifest_not_versioned", f"{dvc_file.name} no declara el md5 del artefacto"
        ) from error
    if hashlib.md5(content).hexdigest() != declared_md5:
        raise SourcesNotEligibleError(
            "manifest_dvc_mismatch",
            "El artefacto no coincide con el md5 versionado en DVC (¿editado después?)",
        )

    try:
        return FrozenManifest.model_validate(raw)
    except ValidationError as error:
        raise SourcesNotEligibleError("manifest_invalid", str(error)) from error


def _check_self_consistency(manifest: FrozenManifest) -> None:
    assignments = {name: tuple(getattr(manifest.assignments, name)) for name in SPLITS}
    recomputed = _manifest_hash(
        dataset_version=manifest.dataset_version,
        seed=manifest.seed,
        target_ratios=manifest.target_ratios.model_dump(),
        assignments=assignments,
    )
    if recomputed != manifest.manifest_hash:
        raise SourcesNotEligibleError(
            "manifest_hash_mismatch",
            "manifest_hash no corresponde al contenido del artefacto (adulterado)",
        )
    if frozen_test_split_hash(manifest.assignments.test) != manifest.test_split_hash:
        raise SourcesNotEligibleError(
            "test_hash_mismatch", "test_split_hash no corresponde a la partición test declarada"
        )


def _split_counts(ids: list[int], crops: dict[int, Crop]) -> ManifestSplitCounts:
    per_class = dict.fromkeys(MANIFEST_CLASSES, 0)
    per_class.update(Counter(crops[crop_id].category_name for crop_id in ids))
    return ManifestSplitCounts(
        crops=len(ids),
        originals=len({crops[crop_id].image_id for crop_id in ids}),
        crops_per_class=per_class,
    )


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
    """Comprueba que (dataset_version, manifest_hash) son el release aprobado y el
    manifest congelado oficial, recalculando todo contra los archivos reales.

    Lanza `SourcesNotEligibleError` con un `reason` estable en cuanto algo no cuadra.
    """
    manifest = _read_frozen_manifest(manifest_path)
    if (dataset_version, manifest_hash) != (manifest.dataset_version, manifest.manifest_hash):
        raise SourcesNotEligibleError(
            "request_mismatch",
            f"Se pidió {dataset_version}/{manifest_hash[:12]}…, pero el manifest congelado es "
            f"{manifest.dataset_version}/{manifest.manifest_hash[:12]}…",
        )
    _check_self_consistency(manifest)

    try:
        release = resolve_release(
            dataset_version,
            repo_root=repo_root,
            reports_dir=reports_dir,
            sources=sources,
            policy=policy,
        )
    except ReleaseRejectedError as error:
        raise SourcesNotEligibleError(f"release_{error.reason}", str(error)) from error

    expected_release_hash = _dvc_release_hash(release.images_md5, release.annotations_md5)
    if (
        manifest.images_md5 != release.images_md5
        or manifest.annotations_md5 != release.annotations_md5
        or manifest.dvc_release_hash != expected_release_hash
    ):
        raise SourcesNotEligibleError(
            "identity_mismatch",
            "La identidad DVC del manifest no coincide con el release resuelto "
            f"({release.dataset_version})",
        )

    # Crops reales y grupos indivisibles (near-duplicates) recalculados desde los datos,
    # no desde el artefacto: así una asignación con fuga no se valida a sí misma.
    try:
        _, candidate = build_manifest_candidate(
            dataset_version,
            repo_root=repo_root,
            reports_dir=reports_dir,
            sources=sources,
            policy=policy,
        )
    except ManifestBlockedError as error:
        reason = (
            "insufficient_originals"
            if error.reason == "insufficient_originals_after_exclusions"
            else "candidate_unavailable"
        )
        raise SourcesNotEligibleError(reason, str(error)) from error
    _, crop_result = build_crops_report(
        dataset_version,
        repo_root=repo_root,
        reports_dir=reports_dir,
        sources=sources,
        policy=policy,
    )
    crops = {crop.crop_id: crop for crop in crop_result.crops}

    assignments = {name: list(getattr(manifest.assignments, name)) for name in SPLITS}
    unknown = [i for ids in assignments.values() for i in ids if i not in crops]
    if unknown:
        raise SourcesNotEligibleError(
            "invalid_partition", f"crop_id inexistentes en el release: {unknown[:5]}"
        )
    crops_per_class = {
        name: dict(Counter(crops[i].category_name for i in ids))
        for name, ids in assignments.items()
    }
    try:
        verify_manifest_assignment(
            assignments,
            crop_ids=sorted(crops),
            groups=candidate.groups,
            classes=list(MANIFEST_CLASSES),
            crops_per_class=crops_per_class,
        )
    except ManifestValidationError as error:
        raise SourcesNotEligibleError("invalid_partition", str(error)) from error

    try:
        summary = ManifestSummary(
            manifest_version=manifest.manifest_version,
            manifest_hash=manifest.manifest_hash,
            dataset_version=manifest.dataset_version,
            dvc_release_hash=manifest.dvc_release_hash,
            seed=manifest.seed,
            target_ratios=manifest.target_ratios,
            frozen=True,
            classes=sorted(MANIFEST_CLASSES),
            splits=ManifestSplits(
                **{name: _split_counts(ids, crops) for name, ids in assignments.items()}
            ),
        )
    except ValidationError as error:
        raise SourcesNotEligibleError("invalid_partition", str(error)) from error

    coco = load_dataset(repo_root / release.annotations_dir)
    return VerifiedSources(
        release=release,
        manifest=manifest,
        summary=summary,
        crops=tuple(crops[i] for i in sorted(crops)),
        images_dir=repo_root / release.images_dir,
        image_files={image.id: image.file_name for image in coco.images},
    )


def _crop_pixels(crop: Crop, images_dir: Path, image_files: dict[int, str]) -> Image.Image:
    """Recorta un crop de su imagen original (bbox en píxeles enteros, D01-07)."""
    x0, y0, x1, y1 = crop.bbox_pixels
    with Image.open(images_dir / image_files[crop.image_id]) as image:
        return image.convert("RGB").crop((x0, y0, x1, y1))


def build_training_dataset(verified: VerifiedSources) -> TrainingDataset:
    """Solo train/val llegan al trainer. La partición test no se recorta ni se carga:
    `TrainingDataset.test` queda vacío (Ale custodia el frozen test, #33)."""
    crops = {crop.crop_id: crop for crop in verified.crops}

    def samples(split: str) -> tuple[TrainingSample, ...]:
        return tuple(
            TrainingSample(
                sample_id=crop_id,
                group_id=crops[crop_id].image_id,
                label=crops[crop_id].category_name,
                image=_crop_pixels(crops[crop_id], verified.images_dir, verified.image_files),
            )
            for crop_id in getattr(verified.manifest.assignments, split)
        )

    return TrainingDataset(train=samples("train"), val=samples("val"), test=())
