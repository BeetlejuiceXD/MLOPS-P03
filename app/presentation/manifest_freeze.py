"""Tier 6 — auditoría y congelación del manifest P3 (D03-01).

Congela el candidato de D02-04 (`presentation.manifest_candidate`) SIN reconstruirlo:
el split es el mismo, solo cambia `frozen` a `True`. Como `manifest_hash` no incluye
`frozen`, la identidad del candidato y la del manifest congelado coinciden.

Artefactos:

- `data/p3/manifest.json` (+ `manifest.json.dvc` vía `dvc add`) — el artefacto oficial
  con el contrato `FrozenManifest` de `presentation.contracts`, el que lee Training
  (D03-03, `trainer_worker.sources`): identidad DVC, `test_split_hash` y `crop_id` por
  partición. Solo IDs: los píxeles de test nunca llegan al trainer.
- `reports/manifest_p3.json` — `ManifestSummary` (D01-05) con `frozen=true`.
- `reports/manifest_p3_freeze.json` — `FreezeRecord`: identidad del release, sha256 y
  md5 del artefacto, `test_split_hash`, conteos por clase y grupos auditados.

La auditoría es independiente del generador: recalcula los grupos near-duplicate
(con puentes sin crops), el hash del candidato, el hash del split test, las
proporciones, la presencia de clases y el mínimo de originales por clase, y exige que
las asignaciones cubran exactamente los crops reales del release. Cualquier violación
lanza `FreezeBlockedError` y no se escribe nada. Nada de esto evalúa modelos ni lee
métricas del test.
"""

import argparse
import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from crops.models import Crop
from pydantic import BaseModel, ConfigDict, ValidationError

from analyzers.duplicates import analyze_duplicates
from ingestion.loader import load_dataset, load_image_contents
from policies.duplicates import load_duplicate_config
from policies.models import QualityPolicy, load_quality_policy
from presentation.contracts import (
    MANIFEST_CLASSES,
    MANIFEST_SPLIT_TOLERANCE,
    FrozenManifest,
    ManifestSummary,
    frozen_test_split_hash,
)
from presentation.crops_report import build_crops_report
from presentation.manifest_candidate import (
    APP_ROOT,
    ManifestBlockedError,
    _dvc_release_hash,
    _manifest_hash,
    build_manifest_candidate,
)
from presentation.release_resolver import (
    ReleaseRejectedError,
    ReleaseSource,
    load_release_sources,
    resolve_release,
)
from splits.models import SplitsConfig

logger = logging.getLogger("manifest-freeze")

SPLITS = ("train", "val", "test")
MANIFEST_PATH = Path("data") / "p3" / "manifest.json"
SUMMARY_FILE = "manifest_p3.json"
RECORD_FILE = "manifest_p3_freeze.json"

FreezeBlockReason = Literal[
    "insufficient_originals",
    "invalid_partition",
    "release_hash_mismatch",
    "manifest_hash_mismatch",
    "artifact_hash_mismatch",
    "artifact_invalid",
    "dvc_mismatch",
    "test_hash_mismatch",
    "identity_mismatch",
    "candidate_drift",
    "coverage",
    "overlap",
    "group_crosses_splits",
    "ratio_out_of_tolerance",
    "class_missing",
    "summary_mismatch",
]


class FreezeBlockedError(ValueError):
    """El manifest no se congela: `reason` es estable para tests, CLI y Heri."""

    def __init__(
        self,
        reason: FreezeBlockReason,
        detail: str,
        *,
        classes_below_minimum: list[str] | None = None,
    ):
        super().__init__(f"[{reason}] {detail}")
        self.reason = reason
        self.classes_below_minimum = classes_below_minimum or []


class FreezeRecord(BaseModel):
    """Registro de congelación versionado en git junto a `ManifestSummary`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    manifest_version: str
    manifest_hash: str
    candidate_manifest_hash: str
    frozen: bool
    dataset_version: str
    dvc_images_md5: str
    dvc_annotations_md5: str
    dvc_release_hash: str
    seed: int
    manifest_sha256: str
    manifest_md5: str
    test_split_hash: str
    crops: dict[str, int]
    originals: dict[str, int]
    crops_per_class: dict[str, dict[str, int]]
    originals_per_class: dict[str, dict[str, int]]
    min_originals_per_class: int
    duplicate_pairs: int
    groups_total: int
    groups_multi_image: int
    groups_crossing_splits: int


@dataclass(frozen=True)
class FreezeResult:
    summary: ManifestSummary
    manifest: bytes
    record: FreezeRecord


def _canonical(payload) -> bytes:
    """JSON determinista (claves ordenadas, LF, newline final): mismos bytes y
    mismo sha256 en cualquier proceso o sistema operativo."""
    return (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _artifact(
    summary: ManifestSummary,
    assignments: Mapping[str, Sequence[int]],
    *,
    images_md5: str,
    annotations_md5: str,
) -> bytes:
    """Bytes del `FrozenManifest` (contrato de D03-03) con `crop_id` ordenados."""
    ids = {name: sorted(assignments[name]) for name in SPLITS}
    return _canonical(
        {
            "manifest_version": summary.manifest_version,
            "manifest_hash": summary.manifest_hash,
            "dataset_version": summary.dataset_version,
            "dvc_release_hash": summary.dvc_release_hash,
            "images_md5": images_md5,
            "annotations_md5": annotations_md5,
            "seed": summary.seed,
            "target_ratios": summary.target_ratios.model_dump(),
            "frozen": True,
            "test_split_hash": frozen_test_split_hash(ids["test"]),
            "assignments": ids,
        }
    )


def _image_groups(image_ids: set[int], duplicate_pairs: Sequence[Mapping]) -> list[set[int]]:
    """Componentes conexas por pares pHash, recalculadas aquí (no reutiliza el
    `_image_groups` del generador). Una imagen sin crops puede ser PUENTE: si
    A≈B≈C y B no tiene crops, A y C siguen en el mismo grupo."""
    parent: dict[int, int] = {}

    def find(i: int) -> int:
        parent.setdefault(i, i)
        root = i
        while parent[root] != root:
            root = parent[root]
        parent[i] = root
        return root

    for pair in duplicate_pairs:
        a, b = find(pair["image_id_a"]), find(pair["image_id_b"])
        if a != b:
            parent[max(a, b)] = min(a, b)
    groups: dict[int, set[int]] = {}
    for image_id in image_ids:
        groups.setdefault(find(image_id), set()).add(image_id)
    return list(groups.values())


def _audit(
    summary: ManifestSummary,
    manifest: bytes,
    *,
    crops: Sequence[Crop],
    duplicate_pairs: Sequence[Mapping],
    images_md5: str,
    annotations_md5: str,
    min_originals_per_class: int,
    expected_sha256: str | None = None,
) -> FreezeRecord:
    # 1) Integridad del artefacto (antes de interpretar su contenido) y contrato D03-03.
    if expected_sha256 is not None and _sha256(manifest) != expected_sha256:
        raise FreezeBlockedError(
            "artifact_hash_mismatch", "manifest.json no coincide con su sha256 registrado"
        )
    try:
        artifact = FrozenManifest.model_validate_json(manifest)
    except ValidationError as error:
        raise FreezeBlockedError("artifact_invalid", str(error)) from error

    # 2) Identidad: release, resumen publicado y hashes recalculados del contenido.
    if summary.dvc_release_hash != _dvc_release_hash(images_md5, annotations_md5):
        raise FreezeBlockedError(
            "release_hash_mismatch", "dvc_release_hash no corresponde a los md5 DVC del release"
        )
    if (artifact.images_md5, artifact.annotations_md5) != (images_md5, annotations_md5):
        raise FreezeBlockedError("identity_mismatch", "md5 DVC del artefacto != release")
    for key in ("manifest_version", "manifest_hash", "dataset_version", "dvc_release_hash", "seed"):
        if getattr(artifact, key) != getattr(summary, key):
            raise FreezeBlockedError("identity_mismatch", f"{key} difiere del resumen")
    ratios = summary.target_ratios.model_dump()
    if artifact.target_ratios.model_dump() != ratios:
        raise FreezeBlockedError("identity_mismatch", "target_ratios difiere del resumen")
    assignments = {name: tuple(sorted(getattr(artifact.assignments, name))) for name in SPLITS}
    recomputed = _manifest_hash(
        dataset_version=summary.dataset_version,
        seed=summary.seed,
        target_ratios=ratios,
        assignments=assignments,
    )
    if recomputed != summary.manifest_hash:
        raise FreezeBlockedError(
            "manifest_hash_mismatch", f"manifest_hash declarado != recalculado ({recomputed})"
        )
    if frozen_test_split_hash(assignments["test"]) != artifact.test_split_hash:
        raise FreezeBlockedError(
            "test_hash_mismatch", "test_split_hash no corresponde a la partición test"
        )

    # 3) Cobertura exacta de los crops reales del release.
    by_id = {crop.crop_id: crop for crop in crops}
    assigned = [crop_id for name in SPLITS for crop_id in assignments[name]]
    if len(assigned) != len(set(assigned)):
        raise FreezeBlockedError("overlap", "un crop_id aparece en más de una partición")
    if set(assigned) != set(by_id):
        missing, extra = set(by_id) - set(assigned), set(assigned) - set(by_id)
        raise FreezeBlockedError(
            "coverage", f"crops sin asignar {sorted(missing)[:5]}, desconocidos {sorted(extra)[:5]}"
        )

    # 4) Disjunción de originales y de grupos near-duplicate.
    split_of_image: dict[int, str] = {}
    for name in SPLITS:
        for crop_id in assignments[name]:
            image_id = by_id[crop_id].image_id
            if split_of_image.setdefault(image_id, name) != name:
                raise FreezeBlockedError(
                    "overlap", f"el original {image_id} está en dos particiones"
                )
    groups = _image_groups(set(split_of_image), duplicate_pairs)
    crossing = [g for g in groups if len({split_of_image[i] for i in g}) > 1]
    if crossing:
        raise FreezeBlockedError(
            "group_crosses_splits",
            f"{len(crossing)} grupo(s) cruzan particiones: {sorted(crossing[0])}",
        )

    # 5) Proporciones sobre crops, clases por partición y mínimo de originales.
    total = len(assigned)
    crops_count = {name: len(assignments[name]) for name in SPLITS}
    for name in SPLITS:
        if abs(crops_count[name] / total - ratios[name]) > MANIFEST_SPLIT_TOLERANCE + 1e-12:
            raise FreezeBlockedError(
                "ratio_out_of_tolerance",
                f"{name}: {crops_count[name] / total:.4f} fuera de ±5 pp de {ratios[name]}",
            )
    crops_per_class = {name: dict.fromkeys(MANIFEST_CLASSES, 0) for name in SPLITS}
    classes_of_image: dict[int, set[str]] = {}
    for name in SPLITS:
        for crop_id in assignments[name]:
            crop = by_id[crop_id]
            crops_per_class[name][crop.category_name] += 1
            classes_of_image.setdefault(crop.image_id, set()).add(crop.category_name)
    for name in SPLITS:
        absent = [c for c in MANIFEST_CLASSES if crops_per_class[name][c] == 0]
        if absent:
            raise FreezeBlockedError("class_missing", f"{name} no tiene crops de {absent}")
    originals_per_class = {
        name: {
            c: sum(1 for i, s in split_of_image.items() if s == name and c in classes_of_image[i])
            for c in MANIFEST_CLASSES
        }
        for name in SPLITS
    }
    below = [
        c
        for c in MANIFEST_CLASSES
        if sum(originals_per_class[name][c] for name in SPLITS) < min_originals_per_class
    ]
    if below:
        raise FreezeBlockedError(
            "insufficient_originals",
            f"clases con menos de {min_originals_per_class} originales: {below}",
            classes_below_minimum=below,
        )

    # 6) El resumen publicado debe describir exactamente estas asignaciones.
    originals = {name: sum(1 for s in split_of_image.values() if s == name) for name in SPLITS}
    for name in SPLITS:
        declared = getattr(summary.splits, name)
        if (
            declared.crops != crops_count[name]
            or declared.originals != originals[name]
            or dict(declared.crops_per_class) != crops_per_class[name]
        ):
            raise FreezeBlockedError("summary_mismatch", f"conteos de {name} no coinciden")

    return FreezeRecord(
        manifest_version=summary.manifest_version,
        manifest_hash=summary.manifest_hash,
        candidate_manifest_hash=recomputed,
        frozen=True,
        dataset_version=summary.dataset_version,
        dvc_images_md5=images_md5,
        dvc_annotations_md5=annotations_md5,
        dvc_release_hash=summary.dvc_release_hash,
        seed=summary.seed,
        manifest_sha256=_sha256(manifest),
        manifest_md5=hashlib.md5(manifest).hexdigest(),
        test_split_hash=artifact.test_split_hash,
        crops=crops_count,
        originals=originals,
        crops_per_class=crops_per_class,
        originals_per_class=originals_per_class,
        min_originals_per_class=min_originals_per_class,
        duplicate_pairs=len(duplicate_pairs),
        groups_total=len(groups),
        groups_multi_image=sum(1 for g in groups if len(g) > 1),
        groups_crossing_splits=0,
    )


def freeze_candidate(
    summary: ManifestSummary,
    assignments: Mapping[str, Sequence[int]],
    *,
    crops: Sequence[Crop],
    duplicate_pairs: Sequence[Mapping],
    images_md5: str,
    annotations_md5: str,
    min_originals_per_class: int,
) -> FreezeResult:
    """Audita el candidato y, solo si cumple todas las reglas, devuelve los bytes
    congelados. No escribe nada: `write_frozen_manifest` persiste el resultado."""
    frozen_summary = summary.model_copy(update={"frozen": True})
    manifest = _artifact(
        frozen_summary, assignments, images_md5=images_md5, annotations_md5=annotations_md5
    )
    record = _audit(
        frozen_summary,
        manifest,
        crops=crops,
        duplicate_pairs=duplicate_pairs,
        images_md5=images_md5,
        annotations_md5=annotations_md5,
        min_originals_per_class=min_originals_per_class,
    )
    # Última red: el contrato de D01-05 revalidado completo sobre lo que se publica.
    frozen_summary = ManifestSummary.model_validate(frozen_summary.model_dump())
    return FreezeResult(summary=frozen_summary, manifest=manifest, record=record)


@dataclass(frozen=True)
class _ReleaseInputs:
    crops: list[Crop]
    duplicate_pairs: list[dict]
    images_md5: str
    annotations_md5: str
    min_originals_per_class: int


def _release_inputs(
    version: str,
    *,
    repo_root: Path,
    reports_dir: Path,
    sources: dict[str, ReleaseSource],
    policy: QualityPolicy,
) -> _ReleaseInputs:
    crops_report, crop_result = build_crops_report(
        version, repo_root=repo_root, reports_dir=reports_dir, sources=sources, policy=policy
    )
    release = resolve_release(
        version, repo_root=repo_root, reports_dir=reports_dir, sources=sources, policy=policy
    )
    dataset_dir = repo_root / release.dataset_dir
    coco = load_dataset(dataset_dir / "annotations")
    duplicates = analyze_duplicates(
        load_image_contents(coco, dataset_dir / "images"), load_duplicate_config()
    )
    return _ReleaseInputs(
        crops=list(crop_result.crops),
        duplicate_pairs=list(duplicates.details["image_pairs"]),
        images_md5=release.images_md5,
        annotations_md5=release.annotations_md5,
        min_originals_per_class=int(crops_report.min_images_per_class),
    )


def freeze_manifest(
    version: str,
    *,
    repo_root: Path,
    reports_dir: Path,
    sources: dict[str, ReleaseSource],
    policy: QualityPolicy,
    manifest_config: SplitsConfig | None = None,
) -> FreezeResult:
    """Recupera el candidato de D02-04 sobre el release real y lo congela.
    Propaga `ReleaseRejectedError`; cualquier bloqueo es `FreezeBlockedError`."""
    try:
        summary, candidate = build_manifest_candidate(
            version,
            repo_root=repo_root,
            reports_dir=reports_dir,
            sources=sources,
            policy=policy,
            manifest_config=manifest_config,
        )
    except ManifestBlockedError as error:
        reason = (
            "insufficient_originals"
            if error.reason == "insufficient_originals_after_exclusions"
            else "invalid_partition"
        )
        raise FreezeBlockedError(
            reason, str(error), classes_below_minimum=error.classes_below_minimum
        ) from error
    inputs = _release_inputs(
        version, repo_root=repo_root, reports_dir=reports_dir, sources=sources, policy=policy
    )
    return freeze_candidate(
        summary,
        candidate.assignments,
        crops=inputs.crops,
        duplicate_pairs=inputs.duplicate_pairs,
        images_md5=inputs.images_md5,
        annotations_md5=inputs.annotations_md5,
        min_originals_per_class=inputs.min_originals_per_class,
    )


def write_frozen_manifest(frozen: FreezeResult, *, manifest_path: Path, reports_dir: Path) -> None:
    """Escribe los bytes exactos auditados (sin re-serializar: el sha256 y el md5
    registrados son los del archivo en disco). Después: `dvc add data/p3/manifest.json`."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(frozen.manifest)
    (reports_dir / SUMMARY_FILE).write_bytes(_canonical(frozen.summary.model_dump(mode="json")))
    (reports_dir / RECORD_FILE).write_bytes(_canonical(frozen.record.model_dump(mode="json")))


def _dvc_md5(manifest_path: Path) -> str | None:
    """md5 que declara `manifest.json.dvc` (lo que DVC versionó), o None si no hay."""
    dvc_file = manifest_path.with_name(manifest_path.name + ".dvc")
    try:
        return yaml.safe_load(dvc_file.read_text(encoding="utf-8"))["outs"][0]["md5"]
    except (OSError, yaml.YAMLError, KeyError, IndexError, TypeError):
        return None


def audit_frozen_on_disk(
    version: str,
    *,
    repo_root: Path,
    reports_dir: Path,
    sources: dict[str, ReleaseSource],
    policy: QualityPolicy,
    manifest_path: Path,
    frozen_reports_dir: Path,
) -> FreezeRecord:
    """Reauditoría repetible de un manifest ya congelado (p. ej. tras `dvc pull`):
    verifica que el artefacto es el versionado en DVC y el registrado, todas las
    reglas contra el release real y que regenerar el candidato de D02-04 dé los
    mismos bytes."""
    manifest = manifest_path.read_bytes()
    summary = ManifestSummary.model_validate_json(
        (frozen_reports_dir / SUMMARY_FILE).read_text(encoding="utf-8")
    )
    stored = FreezeRecord.model_validate_json(
        (frozen_reports_dir / RECORD_FILE).read_text(encoding="utf-8")
    )
    if _dvc_md5(manifest_path) != hashlib.md5(manifest).hexdigest():
        raise FreezeBlockedError(
            "dvc_mismatch", f"{manifest_path.name} no coincide con su .dvc (o no existe)"
        )
    inputs = _release_inputs(
        version, repo_root=repo_root, reports_dir=reports_dir, sources=sources, policy=policy
    )
    record = _audit(
        summary,
        manifest,
        crops=inputs.crops,
        duplicate_pairs=inputs.duplicate_pairs,
        images_md5=inputs.images_md5,
        annotations_md5=inputs.annotations_md5,
        min_originals_per_class=inputs.min_originals_per_class,
        expected_sha256=stored.manifest_sha256,
    )
    if record != stored:
        raise FreezeBlockedError(
            "summary_mismatch", "el registro guardado no coincide con la auditoría"
        )
    regenerated = freeze_manifest(
        version, repo_root=repo_root, reports_dir=reports_dir, sources=sources, policy=policy
    )
    if regenerated.manifest != manifest:
        raise FreezeBlockedError(
            "candidate_drift",
            "regenerar el candidato de D02-04 no reproduce el artefacto congelado",
        )
    return record


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="presentation.manifest_freeze")
    parser.add_argument("command", choices=("freeze", "audit"))
    parser.add_argument("version", help="vMAJOR.MINOR.PATCH, p. ej. v0.1.1")
    args = parser.parse_args(argv)

    repo_root = APP_ROOT.parent
    common = {
        "repo_root": repo_root,
        "reports_dir": repo_root / "reports",
        "sources": load_release_sources(),
        "policy": load_quality_policy(),
    }
    try:
        if args.command == "freeze":
            frozen = freeze_manifest(args.version, **common)
            write_frozen_manifest(
                frozen, manifest_path=repo_root / MANIFEST_PATH, reports_dir=repo_root / "reports"
            )
            record = frozen.record
        else:
            record = audit_frozen_on_disk(
                args.version,
                **common,
                manifest_path=repo_root / MANIFEST_PATH,
                frozen_reports_dir=repo_root / "reports",
            )
    except (ReleaseRejectedError, FreezeBlockedError) as error:
        print(json.dumps({"frozen": False, "reason": error.reason, "detail": str(error)}))
        return 1
    print(record.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
