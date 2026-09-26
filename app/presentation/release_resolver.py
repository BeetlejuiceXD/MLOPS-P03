"""Tier 6 — resolvedor de release: versión -> fuente de datos verificada (D01-02).

`resolve_release("v0.1.1", ...)` devuelve un descriptor con la fuente COCO/imágenes
del release, sus hashes DVC, la referencia y el hash del reporte de calidad, el hash
de la política y los originales distintos por clase; o lanza `ReleaseRejectedError`
con un `reason` estable. No modifica el catálogo ni los reportes de P2.

Un release es elegible solo si:
  1. tiene forma `vMAJOR.MINOR.PATCH`,
  2. está en `versions.json`,
  3. está en `release_sources.yaml` (allowlist con md5 DVC fijados),
  4. su reporte no está `failed` ni oculta un check bloqueante (`action: fail`) reprobado,
  5. los `.dvc` del repo declaran exactamente los md5 fijados,
  6. los datos existen y su cantidad de archivos coincide con `nfiles` de DVC,
  7. al menos `min_classes` clases tienen >= `min_images_per_class` originales distintos.

No recalcula el md5 de los directorios: eso lo garantiza DVC (`dvc status`), igual que en
`release.py`. Aquí solo se comprueba que lo declarado en git es lo autorizado.

Como `release.py`, recibe rutas/política directas y no lee variables de entorno.
"""

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ingestion.loader import load_dataset
from policies.models import QualityPolicy, load_quality_policy
from presentation.contracts import QualityReport, VersionsReport
from presentation.release import SEMVER_PATTERN

logger = logging.getLogger("release-resolver")

APP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCES_PATH = APP_ROOT / "ingestion" / "release_sources.yaml"
DEFAULT_MIN_CLASSES = 2

DVC_MD5_DIR = re.compile(r"^[0-9a-f]{32}\.dir$")

RejectionReason = Literal[
    "invalid_version",
    "not_in_catalog",
    "not_allowed",
    "quality_failed",
    "quality_mismatch",
    "identity_mismatch",
    "data_missing",
    "insufficient_classes",
]


class ReleaseRejectedError(ValueError):
    """El release no es elegible como fuente de P3; `reason` es estable para tests y UI."""

    def __init__(self, reason: RejectionReason, message: str):
        super().__init__(f"[{reason}] {message}")
        self.reason: RejectionReason = reason


class ReleaseSource(BaseModel):
    """Fuente autorizada de un release: carpeta de datos y md5 DVC fijados."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    dataset_dir: str
    annotations_md5: str
    images_md5: str

    @field_validator("dataset_dir")
    @classmethod
    def relative_dataset_dir(cls, value: str) -> str:
        parts = value.split("/")
        if not value or value.startswith(("/", "\\")) or ":" in value or "\\" in value:
            raise ValueError("dataset_dir debe ser una ruta relativa con '/' bajo la raíz del repo")
        if any(part in ("", ".", "..") for part in parts):
            raise ValueError("dataset_dir no puede contener segmentos vacíos, '.' ni '..'")
        return value

    @field_validator("annotations_md5", "images_md5")
    @classmethod
    def dvc_directory_md5(cls, value: str) -> str:
        if not DVC_MD5_DIR.fullmatch(value):
            raise ValueError("md5 de DVC debe ser 32 hex en minúsculas seguidos de '.dir'")
        return value


class ResolvedRelease(BaseModel):
    """Descriptor de la fuente resuelta. Rutas relativas a la raíz del repo, portables."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_version: str
    status: Literal["passed", "warning"]
    dataset_dir: str
    annotations_dir: str
    images_dir: str
    annotations_md5: str
    images_md5: str
    annotations_nfiles: Annotated[int, Field(ge=0)]
    images_nfiles: Annotated[int, Field(ge=0)]
    quality_file: str
    quality_sha256: str
    policy_sha256: str
    min_images_per_class: float
    min_classes: int
    originals_per_class: dict[str, int]


def load_release_sources(path: Path | None = None) -> dict[str, ReleaseSource]:
    """Lee la allowlist versionada; cada clave es una versión semver."""
    sources_path = path if path is not None else DEFAULT_SOURCES_PATH
    raw = yaml.safe_load(sources_path.read_text(encoding="utf-8"))
    allowed = raw["allowed_releases"]
    sources = {version: ReleaseSource.model_validate(entry) for version, entry in allowed.items()}
    for version in sources:
        if not SEMVER_PATTERN.fullmatch(version):
            raise ValueError(f"versión no semver en {sources_path.name}: {version!r}")
    return sources


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def policy_sha256(policy: QualityPolicy) -> str:
    """Hash de la política *aplicada* (forma canónica), no del archivo que la cargó:
    ignora comentarios y orden de claves, y cambia si cambia cualquier umbral/acción."""
    canonical = json.dumps(policy.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _catalog_entry(reports_dir: Path, version: str):
    catalog_path = reports_dir / "versions.json"
    if not catalog_path.exists():
        raise ReleaseRejectedError("not_in_catalog", f"no existe el catálogo {catalog_path.name}")
    catalog = VersionsReport.model_validate_json(catalog_path.read_text(encoding="utf-8"))
    for release in catalog.releases:
        if release.dataset_version == version:
            return release
    raise ReleaseRejectedError("not_in_catalog", f"{version!r} no está en el catálogo de releases")


def _check_quality(reports_dir: Path, version: str, quality_file: str) -> tuple[str, str]:
    quality_path = reports_dir / quality_file
    try:
        report = QualityReport.model_validate_json(quality_path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise ReleaseRejectedError(
            "quality_mismatch", f"reporte de calidad ilegible o inválido: {quality_file}"
        ) from error
    if report.dataset_version != version:
        raise ReleaseRejectedError(
            "quality_mismatch",
            f"{quality_file} describe {report.dataset_version!r}, no {version!r}",
        )
    blocking = [c.check_name for c in report.checks if c.action == "fail" and not c.passed]
    if report.status == "failed" or blocking:
        raise ReleaseRejectedError(
            "quality_failed",
            f"{version!r} no es elegible: status={report.status!r}, checks bloqueantes={blocking}",
        )
    return report.status, _sha256(quality_path)


def _declared_out(dvc_file: Path) -> dict:
    try:
        return yaml.safe_load(dvc_file.read_text(encoding="utf-8"))["outs"][0]
    except (OSError, KeyError, IndexError, TypeError, yaml.YAMLError) as error:
        raise ReleaseRejectedError(
            "identity_mismatch", f"no se pudo leer la identidad DVC en {dvc_file.name}"
        ) from error


def _check_identity(dataset_dir: Path, source: ReleaseSource) -> dict[str, dict]:
    outs = {}
    for name, pinned in (("annotations", source.annotations_md5), ("images", source.images_md5)):
        out = _declared_out(dataset_dir / f"{name}.dvc")
        if out.get("md5") != pinned:
            raise ReleaseRejectedError(
                "identity_mismatch",
                f"{name}.dvc declara md5={out.get('md5')!r}, se esperaba {pinned!r}",
            )
        outs[name] = out
    return outs


def _check_data_present(dataset_dir: Path, outs: dict[str, dict]) -> None:
    for name, out in outs.items():
        directory = dataset_dir / name
        actual = (
            sum(1 for item in directory.rglob("*") if item.is_file()) if directory.is_dir() else 0
        )
        if actual == 0 or actual != out.get("nfiles"):
            raise ReleaseRejectedError(
                "data_missing",
                f"{directory.name}/ tiene {actual} archivos y DVC declara {out.get('nfiles')} "
                "(¿falta `dvc pull`, o hay archivos ajenos?)",
            )


def _originals_per_class(dataset_dir: Path) -> dict[str, int]:
    coco = load_dataset(dataset_dir / "annotations")
    names = {category.id: category.name for category in coco.categories}
    images_by_class: dict[str, set[int]] = {name: set() for name in names.values()}
    for annotation in coco.annotations:
        images_by_class[names[annotation.category_id]].add(annotation.image_id)
    return {name: len(image_ids) for name, image_ids in sorted(images_by_class.items())}


def resolve_release(
    version: str,
    *,
    repo_root: Path,
    reports_dir: Path,
    sources: dict[str, ReleaseSource],
    policy: QualityPolicy,
    min_classes: int = DEFAULT_MIN_CLASSES,
) -> ResolvedRelease:
    """Resuelve `version` a su fuente verificada o lanza `ReleaseRejectedError`."""
    if not SEMVER_PATTERN.fullmatch(version):
        raise ReleaseRejectedError("invalid_version", f"versión inválida: {version!r}")

    entry = _catalog_entry(reports_dir, version)
    source = sources.get(version)
    if source is None:
        raise ReleaseRejectedError(
            "not_allowed", f"{version!r} está en el catálogo pero no es una fuente autorizada"
        )

    status, quality_sha256 = _check_quality(reports_dir, version, entry.quality_file)

    dataset_dir = repo_root / source.dataset_dir
    outs = _check_identity(dataset_dir, source)
    _check_data_present(dataset_dir, outs)

    threshold = policy.min_images_per_class.threshold
    originals = _originals_per_class(dataset_dir)
    eligible = [name for name, count in originals.items() if count >= threshold]
    if len(eligible) < min_classes:
        raise ReleaseRejectedError(
            "insufficient_classes",
            f"{len(eligible)} clases con >= {threshold:g} originales (se requieren "
            f"{min_classes}); conteos: {originals}",
        )

    return ResolvedRelease(
        dataset_version=version,
        status=status,
        dataset_dir=source.dataset_dir,
        annotations_dir=f"{source.dataset_dir}/annotations",
        images_dir=f"{source.dataset_dir}/images",
        annotations_md5=source.annotations_md5,
        images_md5=source.images_md5,
        annotations_nfiles=outs["annotations"]["nfiles"],
        images_nfiles=outs["images"]["nfiles"],
        quality_file=entry.quality_file,
        quality_sha256=quality_sha256,
        policy_sha256=policy_sha256(policy),
        min_images_per_class=threshold,
        min_classes=min_classes,
        originals_per_class=originals,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="presentation.release_resolver")
    parser.add_argument("version", help="vMAJOR.MINOR.PATCH")
    args = parser.parse_args(argv)

    repo_root = APP_ROOT.parent
    try:
        release = resolve_release(
            args.version,
            repo_root=repo_root,
            reports_dir=repo_root / "reports",
            sources=load_release_sources(),
            policy=load_quality_policy(),
        )
    except ReleaseRejectedError as error:
        print(json.dumps({"resolved": False, "reason": error.reason, "detail": str(error)}))
        return 1
    print(json.dumps({"resolved": True, **release.model_dump()}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
