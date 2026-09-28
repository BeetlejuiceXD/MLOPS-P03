"""Tier 6 — reporte real de crops y exclusiones (D02-02).

Ejecuta `crops.engine.generate_crops` (D01-07) sobre el release que resuelve
`presentation.release_resolver` (D01-02): la fuente de datos real de P3, no un
fixture. Recuenta originales distintos por clase a partir de los crops
*válidos* y lo contrasta explícitamente contra el conteo del resolver (basado
en anotaciones crudas, sin pasar por el motor de crops): la verificación
manual de #33 no sustituye correr el motor real sobre `v0.1.1` (D02-02,
"Trabajo exacto").

`coco = load_dataset(...)` valida el COCO con `CocoDataset.model_validate`
antes de generar crops (a diferencia de los tests de D01-07, que alimentan al
motor con dicts crudos a propósito para probar sus cuatro motivos de
exclusión de forma aislada). Eso implica que, sobre datos reales que ya
pasaron esa validación estricta, dos motivos quedan clausurados de antemano:

- `degenerate_bbox`: `Annotation.bbox_is_x_y_width_height` ya rechaza
  width/height no positivos, forma inválida o valores no finitos.
- `missing_image` (el caso de `image_id` huérfano): `CocoDataset` ya exige
  que todo `image_id` esté declarado en `images`.

Lo que sí permanece alcanzable contra datos reales, porque `CocoDataset` no
lo valida:

- `out_of_bounds`: `CocoDataset` no compara `bbox` contra ninguna dimensión;
  el motor la compara contra el tamaño *real* decodificado, que puede no
  coincidir con `width`/`height` declarados.
- `unknown_category` por clase no congelada: `CocoDataset` solo exige que
  `category_id` esté declarado, no que su nombre sea `cat`/`dog`.
- `missing_image` por binario no decodificable (corrupción, no ausencia).

Si alguna clase queda por debajo de `policy.min_images_per_class` tras las
exclusiones reales, `gate_status="blocked"`: este reporte no decide si el
manifest se congela (eso es D03-01), pero dejarlo pasar en silencio sí sería
un defecto de este ticket.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Literal

from crops.engine import generate_crops
from crops.models import FROZEN_CLASSES, CropResult
from pydantic import BaseModel, ConfigDict, field_validator

from ingestion.loader import load_dataset, load_image_contents
from policies.models import QualityPolicy, load_quality_policy
from presentation.release_resolver import (
    ReleaseRejectedError,
    ReleaseSource,
    load_release_sources,
    resolve_release,
)

logger = logging.getLogger("crops-report")

APP_ROOT = Path(__file__).resolve().parent.parent
EXCLUSION_REASONS = ("degenerate_bbox", "out_of_bounds", "missing_image", "unknown_category")


class CropsReport(BaseModel):
    """Evidencia reproducible de una corrida real del motor de crops (D02-02)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_version: str
    annotations_md5: str
    images_md5: str
    quality_sha256: str
    policy_sha256: str
    min_images_per_class: float
    total_annotations: int
    total_crops: int
    total_exclusions: int
    # Recuento explícito: cada anotación de entrada termina en un crop o en una
    # exclusión, nunca en ambos ni en ninguno (P3-04-AC04).
    annotations_equal_crops_plus_exclusions: bool
    exclusions_by_reason: dict[str, int]
    crops_by_class: dict[str, int]
    originals_by_class: dict[str, int]
    classes_below_minimum: list[str]
    gate_status: Literal["ok", "blocked"]
    # Conteo del resolver (D01-02): a partir de TODAS las anotaciones, sin pasar
    # por el motor de crops. Sirve de contraste con `originals_by_class`.
    resolver_originals_per_class: dict[str, int]
    matches_resolver_originals_per_class: bool

    @field_validator(
        "exclusions_by_reason",
        "crops_by_class",
        "originals_by_class",
        "resolver_originals_per_class",
        mode="after",
    )
    @classmethod
    def _canonical_key_order(cls, value: dict[str, int]) -> dict[str, int]:
        """Orden alfabético fijo de claves (revisión de PR #50): `dict.fromkeys`
        sobre un `frozenset` de strings itera en un orden que depende del hash
        de esas strings, que Python aleatoriza por proceso salvo que se fije
        `PYTHONHASHSEED`. Sin esto, la misma entrada podía serializar distinto
        (mismos valores, JSON y hash distintos) según el proceso que lo corriera.
        Se fuerza aquí, en la validación del contrato, para que ninguna forma de
        construirlo pueda saltárselo."""
        return dict(sorted(value.items()))


def build_crops_report(
    version: str,
    *,
    repo_root: Path,
    reports_dir: Path,
    sources: dict[str, ReleaseSource],
    policy: QualityPolicy,
    allowed_categories: frozenset[str] | None = None,
) -> tuple[CropsReport, CropResult]:
    """Resuelve `version`, corre el motor de crops sobre sus datos reales y
    arma el reporte. Propaga `ReleaseRejectedError` si `version` no es
    elegible (mismas reglas que D01-02); no oculta ese rechazo detrás de un
    reporte vacío."""
    allowed = FROZEN_CLASSES if allowed_categories is None else allowed_categories
    release = resolve_release(
        version, repo_root=repo_root, reports_dir=reports_dir, sources=sources, policy=policy
    )

    dataset_dir = repo_root / release.dataset_dir
    coco = load_dataset(dataset_dir / "annotations")
    image_contents = load_image_contents(coco, dataset_dir / "images")

    result = generate_crops(coco.model_dump(), image_contents, allowed_categories=allowed)

    exclusions_by_reason = dict.fromkeys(EXCLUSION_REASONS, 0)
    for exclusion in result.exclusions:
        exclusions_by_reason[exclusion.reason] += 1

    crops_by_class: dict[str, int] = dict.fromkeys(sorted(allowed), 0)
    originals_by_image: dict[str, set[int]] = {name: set() for name in allowed}
    for crop in result.crops:
        crops_by_class[crop.category_name] = crops_by_class.get(crop.category_name, 0) + 1
        originals_by_image.setdefault(crop.category_name, set()).add(crop.image_id)
    originals_by_class = {name: len(ids) for name, ids in sorted(originals_by_image.items())}

    threshold = policy.min_images_per_class.threshold
    classes_below = sorted(name for name, count in originals_by_class.items() if count < threshold)

    report = CropsReport(
        dataset_version=release.dataset_version,
        annotations_md5=release.annotations_md5,
        images_md5=release.images_md5,
        quality_sha256=release.quality_sha256,
        policy_sha256=release.policy_sha256,
        min_images_per_class=threshold,
        total_annotations=len(coco.annotations),
        total_crops=len(result.crops),
        total_exclusions=len(result.exclusions),
        annotations_equal_crops_plus_exclusions=(
            len(coco.annotations) == len(result.crops) + len(result.exclusions)
        ),
        exclusions_by_reason=exclusions_by_reason,
        crops_by_class=crops_by_class,
        originals_by_class=originals_by_class,
        classes_below_minimum=classes_below,
        gate_status="blocked" if classes_below else "ok",
        resolver_originals_per_class=release.originals_per_class,
        matches_resolver_originals_per_class=(originals_by_class == release.originals_per_class),
    )
    return report, result


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="presentation.crops_report")
    parser.add_argument("version", help="vMAJOR.MINOR.PATCH, p. ej. v0.1.1")
    args = parser.parse_args(argv)

    repo_root = APP_ROOT.parent
    try:
        report, _ = build_crops_report(
            args.version,
            repo_root=repo_root,
            reports_dir=repo_root / "reports",
            sources=load_release_sources(),
            policy=load_quality_policy(),
        )
    except ReleaseRejectedError as error:
        print(json.dumps({"resolved": False, "reason": error.reason, "detail": str(error)}))
        return 1

    print(report.model_dump_json(indent=2))

    if report.gate_status == "blocked":
        logger.error(
            "Bloqueado: clases por debajo de %.0f originales: %s",
            report.min_images_per_class,
            report.classes_below_minimum,
        )
        return 1
    if not report.matches_resolver_originals_per_class:
        logger.warning(
            "El recuento de originales por crops válidos (%s) difiere del "
            "conteo del resolver sobre anotaciones crudas (%s); ver el reporte.",
            report.originals_by_class,
            report.resolver_originals_per_class,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
