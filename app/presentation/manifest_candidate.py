"""Tier 6 — manifest P3 candidato 70/20/10 (D02-04).

Orquesta, sobre datos reales: `release_resolver` (D01-02) para la identidad del
release, `crops_report` (D02-02) para los crops reconciliados y su gate de ≥300
originales por clase, `analyzers.duplicates` para los pares near-duplicate, y
`manifest.generator` (D02-04) para la partición en grupos indivisibles. Devuelve
`ManifestSummary` (D01-05, `presentation.contracts`): la misma forma que espera
`GET /api/manifest` en el backend, con `frozen=False` siempre — la congelación
oficial es D03-01, este ticket solo entrega un candidato auditable.

Si el gate de origen de `crops_report` está bloqueado (alguna clase por debajo de
`min_images_per_class` tras las exclusiones reales), no se genera ningún
candidato: se registra el bloqueo con `ManifestBlockedError`, como pide #33
("Si una exclusión futura deja cualquier clase por debajo de 300 originales, el
manifest no se congela y se registra el bloqueo").
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path

from manifest.generator import ManifestCandidateResult, ManifestValidationError, generate_manifest
from manifest.models import load_manifest_config
from pydantic import ValidationError

from analyzers.duplicates import analyze_duplicates
from ingestion.loader import load_dataset, load_image_contents
from policies.duplicates import load_duplicate_config
from policies.models import QualityPolicy, load_quality_policy
from presentation.contracts import (
    MANIFEST_CLASSES,
    ManifestSplitCounts,
    ManifestSplits,
    ManifestSummary,
    ManifestTargetRatios,
)
from presentation.crops_report import build_crops_report
from presentation.release_resolver import (
    ReleaseRejectedError,
    ReleaseSource,
    load_release_sources,
    resolve_release,
)
from splits.models import SplitsConfig

logger = logging.getLogger("manifest-candidate")

APP_ROOT = Path(__file__).resolve().parent.parent


class ManifestBlockedError(ValueError):
    """El origen no tiene ≥300 originales por clase tras las exclusiones reales
    (#33): no hay candidato que generar, y ninguno se congela con esta fuente."""

    def __init__(
        self,
        classes_below_minimum: list[str],
        detail: str,
        *,
        reason: str = "insufficient_originals_after_exclusions",
    ):
        super().__init__(detail)
        self.reason = reason
        self.classes_below_minimum = classes_below_minimum


def _dvc_release_hash(images_md5: str, annotations_md5: str) -> str:
    """`dvc_release_hash = sha256("<images_md5>:<annotations_md5>")` (#33, Esteban)."""
    return hashlib.sha256(f"{images_md5}:{annotations_md5}".encode()).hexdigest()


def _manifest_hash(
    *,
    dataset_version: str,
    seed: int,
    target_ratios: dict[str, float],
    assignments: dict[str, tuple[int, ...]],
) -> str:
    """Hash de contenido del candidato: cambia si cambia la versión, la semilla,
    los ratios objetivo o la asignación crop->partición. Serialización canónica
    (claves ordenadas, listas ya ordenadas) para que el mismo candidato dé el
    mismo hash sin importar el proceso (misma lección que D02-02, PR #50)."""
    canonical = json.dumps(
        {
            "dataset_version": dataset_version,
            "seed": seed,
            "target_ratios": target_ratios,
            "assignments": {name: list(ids) for name, ids in sorted(assignments.items())},
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _split_counts(candidate: ManifestCandidateResult, name: str) -> ManifestSplitCounts:
    # `crops_per_class[name]` (MappingProxyType, congelado por el dataclass) solo
    # trae las clases que SÍ tuvieron crops en esta partición; `ManifestSplitCounts`
    # exige las dos claves congeladas siempre presentes, con 0 explícito si faltan
    # (revisión de Heri en PR #54: una clave ausente no es lo mismo que un 0).
    counts = dict.fromkeys(MANIFEST_CLASSES, 0)
    counts.update(candidate.crops_per_class[name])
    return ManifestSplitCounts(
        crops=len(candidate.assignments[name]),
        originals=len(candidate.originals[name]),
        crops_per_class=counts,
    )


def build_manifest_candidate(
    version: str,
    *,
    repo_root: Path,
    reports_dir: Path,
    sources: dict[str, ReleaseSource],
    policy: QualityPolicy,
    manifest_config: SplitsConfig | None = None,
) -> tuple[ManifestSummary, ManifestCandidateResult]:
    """Resuelve `version`, corre crops+duplicados reales y arma el candidato de
    manifest. Propaga `ReleaseRejectedError` (D01-02) si el release no es
    elegible, y `ManifestBlockedError` si el gate de originales por clase está
    bloqueado tras las exclusiones reales de crops."""
    config = manifest_config if manifest_config is not None else load_manifest_config()

    crops_report, crop_result = build_crops_report(
        version, repo_root=repo_root, reports_dir=reports_dir, sources=sources, policy=policy
    )
    if crops_report.gate_status == "blocked":
        raise ManifestBlockedError(
            crops_report.classes_below_minimum,
            f"{version!r} no es elegible para un manifest: clases por debajo de "
            f"{crops_report.min_images_per_class:g} originales tras las exclusiones "
            f"reales: {crops_report.classes_below_minimum}",
        )

    release = resolve_release(
        version, repo_root=repo_root, reports_dir=reports_dir, sources=sources, policy=policy
    )
    dataset_dir = repo_root / release.dataset_dir
    coco = load_dataset(dataset_dir / "annotations")
    image_contents = load_image_contents(coco, dataset_dir / "images")
    duplicates = analyze_duplicates(image_contents, load_duplicate_config())

    try:
        candidate = generate_manifest(
            crop_result.crops, config, duplicate_pairs=duplicates.details["image_pairs"]
        )
    except ManifestValidationError as error:
        # Grupos indivisibles que no permiten cubrir cat/dog en val y test, o que no
        # caben dentro de ±5 pp sin partirse (#33/D02-04: "o demostrar la mejor
        # asignación factible"): no hay candidato válido que producir con este seed.
        raise ManifestBlockedError(
            [], f"no se pudo generar un candidato válido: {error}", reason="invalid_partition"
        ) from error

    target_ratios = {"train": config.train, "val": config.val, "test": config.test}
    try:
        summary = ManifestSummary(
            manifest_version=f"p3-{version}-s{config.seed}",
            manifest_hash=_manifest_hash(
                dataset_version=version,
                seed=config.seed,
                target_ratios=target_ratios,
                assignments=candidate.assignments,
            ),
            dataset_version=version,
            dvc_release_hash=_dvc_release_hash(release.images_md5, release.annotations_md5),
            seed=config.seed,
            target_ratios=ManifestTargetRatios(**target_ratios),
            frozen=False,
            classes=sorted({crop.category_name for crop in crop_result.crops}),
            splits=ManifestSplits(
                train=_split_counts(candidate, "train"),
                val=_split_counts(candidate, "val"),
                test=_split_counts(candidate, "test"),
            ),
        )
    except ValidationError as error:
        # La causa más probable con datos reales: ningún grupo indivisible cabe
        # dentro de ±5 pp sin romperlo (#33/D02-04: "o demostrar la mejor
        # asignación factible"). El propio contrato de Hannah (D01-05) no admite
        # una excepción de tolerancia, así que no se produce un candidato inválido.
        raise ManifestBlockedError(
            [],
            f"el candidato no cumple el contrato de manifest_summary: {error}",
            reason="invalid_partition",
        ) from error
    return summary, candidate


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="presentation.manifest_candidate")
    parser.add_argument("version", help="vMAJOR.MINOR.PATCH, p. ej. v0.1.1")
    args = parser.parse_args(argv)

    repo_root = APP_ROOT.parent
    try:
        summary, _ = build_manifest_candidate(
            args.version,
            repo_root=repo_root,
            reports_dir=repo_root / "reports",
            sources=load_release_sources(),
            policy=load_quality_policy(),
        )
    except ReleaseRejectedError as error:
        print(json.dumps({"resolved": False, "reason": error.reason, "detail": str(error)}))
        return 1
    except ManifestBlockedError as error:
        print(
            json.dumps(
                {
                    "resolved": False,
                    "reason": error.reason,
                    "classes_below_minimum": error.classes_below_minimum,
                    "detail": str(error),
                }
            )
        )
        return 1

    print(summary.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
