"""Tier 6 — auditoría y congelación del manifest P3 (D03-01).

Congela el candidato de D02-04 (`presentation.manifest_candidate`) SIN reconstruirlo:
el split es el mismo, solo cambia `frozen` a `True`. Como `manifest_hash` no incluye
`frozen`, la identidad del candidato y la del manifest congelado coinciden.

Artefactos (etapa `manifest_p3` de `dvc.yaml`):

- `data/manifest_p3/train_val.json` — lo ÚNICO que consume el trainer (D03-03).
- `data/manifest_p3/test.json` — frozen test, custodia de Ale (#33). Archivo aparte
  para que el trainer no tenga que descargarlo ni leerlo nunca.
- `reports/manifest_p3.json` — `ManifestSummary` (D01-05) con `frozen=true`, lo que
  sirve `GET /api/manifest` y compara la compuerta de training del backend.
- `reports/manifest_p3_freeze.json` — `FreezeRecord`: identidad del release, sha256
  de `train_val.json` y de `test.json`, conteos por clase y grupos auditados.

La auditoría es independiente del generador: recalcula los grupos near-duplicate
(con puentes sin crops), el hash del candidato, las proporciones, la presencia de
clases y el mínimo de originales por clase, y contrasta cada entrada con los crops
reales del release. Cualquier violación lanza `FreezeBlockedError` y no se escribe
nada. Nada de esto evalúa modelos ni lee métricas del test.
"""

import argparse
import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from crops.models import Crop
from pydantic import BaseModel, ConfigDict

from analyzers.duplicates import analyze_duplicates
from ingestion.loader import load_dataset, load_image_contents
from policies.duplicates import load_duplicate_config
from policies.models import QualityPolicy, load_quality_policy
from presentation.contracts import MANIFEST_CLASSES, MANIFEST_SPLIT_TOLERANCE, ManifestSummary
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
TRAIN_VAL_FILE = "train_val.json"
TEST_FILE = "test.json"
SUMMARY_FILE = "manifest_p3.json"
RECORD_FILE = "manifest_p3_freeze.json"
PAYLOAD_SCHEMA_VERSION = "1.0"

FreezeBlockReason = Literal[
    "insufficient_originals",
    "invalid_partition",
    "release_hash_mismatch",
    "manifest_hash_mismatch",
    "train_val_hash_mismatch",
    "test_hash_mismatch",
    "identity_mismatch",
    "candidate_drift",
    "coverage",
    "crop_mismatch",
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
    train_val_sha256: str
    test_sha256: str
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
class FrozenManifest:
    summary: ManifestSummary
    train_val: bytes
    test: bytes
    record: FreezeRecord


def _canonical(payload) -> bytes:
    """JSON determinista (claves ordenadas, LF, newline final): mismos bytes y
    mismo sha256 en cualquier proceso o sistema operativo."""
    return (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _entry(crop: Crop, file_names: Mapping[int, str]) -> dict:
    return {
        "crop_id": crop.crop_id,
        "image_id": crop.image_id,
        "annotation_id": crop.annotation_id,
        "category_name": crop.category_name,
        "file_name": file_names[crop.image_id],
        "bbox_pixels": list(crop.bbox_pixels),
    }


def _payload(summary: ManifestSummary, splits: Mapping[str, list[dict]]) -> bytes:
    return _canonical(
        {
            "schema_version": PAYLOAD_SCHEMA_VERSION,
            "manifest_version": summary.manifest_version,
            "manifest_hash": summary.manifest_hash,
            "dataset_version": summary.dataset_version,
            "dvc_release_hash": summary.dvc_release_hash,
            "seed": summary.seed,
            "classes": list(MANIFEST_CLASSES),
            "splits": dict(splits),
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
    train_val: bytes,
    test: bytes,
    *,
    crops: Sequence[Crop],
    file_names: Mapping[int, str],
    duplicate_pairs: Sequence[Mapping],
    images_md5: str,
    annotations_md5: str,
    min_originals_per_class: int,
    expected_train_val_sha256: str | None = None,
    expected_test_sha256: str | None = None,
) -> FreezeRecord:
    # 1) Integridad de los archivos congelados (antes de interpretar su contenido).
    if expected_train_val_sha256 is not None and _sha256(train_val) != expected_train_val_sha256:
        raise FreezeBlockedError(
            "train_val_hash_mismatch", "train_val.json no coincide con su sha256"
        )
    if expected_test_sha256 is not None and _sha256(test) != expected_test_sha256:
        raise FreezeBlockedError("test_hash_mismatch", "test.json no coincide con su sha256")

    # 2) Identidad: release y hash del candidato.
    if summary.dvc_release_hash != _dvc_release_hash(images_md5, annotations_md5):
        raise FreezeBlockedError(
            "release_hash_mismatch", "dvc_release_hash no corresponde a los md5 DVC del release"
        )
    payloads = [json.loads(train_val), json.loads(test)]
    for payload in payloads:
        for key in ("manifest_version", "manifest_hash", "dataset_version", "dvc_release_hash"):
            if payload.get(key) != getattr(summary, key):
                raise FreezeBlockedError("identity_mismatch", f"{key} difiere entre archivos")
    entries = {**payloads[0]["splits"], **payloads[1]["splits"]}
    if set(payloads[0]["splits"]) != {"train", "val"} or set(payloads[1]["splits"]) != {"test"}:
        raise FreezeBlockedError("identity_mismatch", "train_val/test no contienen sus particiones")
    assignments = {name: tuple(sorted(e["crop_id"] for e in entries[name])) for name in SPLITS}
    ratios = summary.target_ratios.model_dump()
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

    # 3) Cobertura exacta y contenido idéntico a los crops reales del release.
    by_id = {crop.crop_id: crop for crop in crops}
    assigned = [crop_id for name in SPLITS for crop_id in assignments[name]]
    if len(assigned) != len(set(assigned)):
        raise FreezeBlockedError("overlap", "un crop_id aparece en más de una partición")
    if set(assigned) != set(by_id):
        missing, extra = set(by_id) - set(assigned), set(assigned) - set(by_id)
        raise FreezeBlockedError(
            "coverage", f"crops sin asignar {sorted(missing)[:5]}, desconocidos {sorted(extra)[:5]}"
        )
    for name in SPLITS:
        for entry in entries[name]:
            if entry != _entry(by_id[entry["crop_id"]], file_names):
                raise FreezeBlockedError("crop_mismatch", f"crop {entry['crop_id']} no coincide")

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
        train_val_sha256=_sha256(train_val),
        test_sha256=_sha256(test),
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
    file_names: Mapping[int, str],
    duplicate_pairs: Sequence[Mapping],
    images_md5: str,
    annotations_md5: str,
    min_originals_per_class: int,
) -> FrozenManifest:
    """Audita el candidato y, solo si cumple todas las reglas, devuelve los bytes
    congelados. No escribe nada: `write_frozen_manifest` persiste el resultado."""
    by_id = {crop.crop_id: crop for crop in crops}
    splits = {}
    for name in SPLITS:
        unknown = [c for c in assignments[name] if c not in by_id]
        if unknown:
            raise FreezeBlockedError("coverage", f"crops desconocidos en {name}: {unknown[:5]}")
        splits[name] = [_entry(by_id[c], file_names) for c in sorted(assignments[name])]
    frozen_summary = summary.model_copy(update={"frozen": True})
    train_val = _payload(frozen_summary, {"train": splits["train"], "val": splits["val"]})
    test = _payload(frozen_summary, {"test": splits["test"]})
    record = _audit(
        frozen_summary,
        train_val,
        test,
        crops=crops,
        file_names=file_names,
        duplicate_pairs=duplicate_pairs,
        images_md5=images_md5,
        annotations_md5=annotations_md5,
        min_originals_per_class=min_originals_per_class,
    )
    # Última red: el contrato de D01-05 revalidado completo sobre lo que se publica.
    frozen_summary = ManifestSummary.model_validate(frozen_summary.model_dump())
    return FrozenManifest(summary=frozen_summary, train_val=train_val, test=test, record=record)


@dataclass(frozen=True)
class _ReleaseInputs:
    crops: list[Crop]
    file_names: dict[int, str]
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
        file_names={image.id: image.file_name for image in coco.images},
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
) -> FrozenManifest:
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
        file_names=inputs.file_names,
        duplicate_pairs=inputs.duplicate_pairs,
        images_md5=inputs.images_md5,
        annotations_md5=inputs.annotations_md5,
        min_originals_per_class=inputs.min_originals_per_class,
    )


def write_frozen_manifest(frozen: FrozenManifest, *, manifest_dir: Path, reports_dir: Path) -> None:
    """Escribe los bytes exactos auditados (sin re-serializar: el sha256 registrado
    es el del archivo en disco)."""
    manifest_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / TRAIN_VAL_FILE).write_bytes(frozen.train_val)
    (manifest_dir / TEST_FILE).write_bytes(frozen.test)
    (reports_dir / SUMMARY_FILE).write_bytes(_canonical(frozen.summary.model_dump(mode="json")))
    (reports_dir / RECORD_FILE).write_bytes(_canonical(frozen.record.model_dump(mode="json")))


def audit_frozen_on_disk(
    version: str,
    *,
    repo_root: Path,
    reports_dir: Path,
    sources: dict[str, ReleaseSource],
    policy: QualityPolicy,
    manifest_dir: Path,
    frozen_reports_dir: Path,
) -> FreezeRecord:
    """Reauditoría repetible de un manifest ya congelado (p. ej. tras `dvc pull`):
    verifica el sha256 registrado de ambos archivos, todas las reglas contra el
    release real y que regenerar el candidato de D02-04 dé los mismos bytes."""
    train_val = (manifest_dir / TRAIN_VAL_FILE).read_bytes()
    test = (manifest_dir / TEST_FILE).read_bytes()
    summary = ManifestSummary.model_validate_json(
        (frozen_reports_dir / SUMMARY_FILE).read_text(encoding="utf-8")
    )
    stored = FreezeRecord.model_validate_json(
        (frozen_reports_dir / RECORD_FILE).read_text(encoding="utf-8")
    )
    inputs = _release_inputs(
        version, repo_root=repo_root, reports_dir=reports_dir, sources=sources, policy=policy
    )
    record = _audit(
        summary,
        train_val,
        test,
        crops=inputs.crops,
        file_names=inputs.file_names,
        duplicate_pairs=inputs.duplicate_pairs,
        images_md5=inputs.images_md5,
        annotations_md5=inputs.annotations_md5,
        min_originals_per_class=inputs.min_originals_per_class,
        expected_train_val_sha256=stored.train_val_sha256,
        expected_test_sha256=stored.test_sha256,
    )
    if record != stored:
        raise FreezeBlockedError(
            "summary_mismatch", "el registro guardado no coincide con la auditoría"
        )
    regenerated = freeze_manifest(
        version, repo_root=repo_root, reports_dir=reports_dir, sources=sources, policy=policy
    )
    if (regenerated.train_val, regenerated.test) != (train_val, test):
        raise FreezeBlockedError(
            "candidate_drift",
            "regenerar el candidato de D02-04 no reproduce los archivos congelados",
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
                frozen,
                manifest_dir=repo_root / "data" / "manifest_p3",
                reports_dir=repo_root / "reports",
            )
            record = frozen.record
        else:
            record = audit_frozen_on_disk(
                args.version,
                **common,
                manifest_dir=repo_root / "data" / "manifest_p3",
                frozen_reports_dir=repo_root / "reports",
            )
    except (ReleaseRejectedError, FreezeBlockedError) as error:
        print(json.dumps({"frozen": False, "reason": error.reason, "detail": str(error)}))
        return 1
    print(record.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
