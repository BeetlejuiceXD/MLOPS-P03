"""D02-02 — reporte real de crops y exclusiones, sobre el release resuelto (D01-02)
y el motor de crops (D01-07). No repite los tests de `test_crops.py` (esos prueban
el motor en aislamiento con dicts crudos); estos prueban la tubería real:
`resolve_release` -> `load_dataset` (CocoDataset estricto) -> `generate_crops`.

Por pasar por `CocoDataset.model_validate`, `degenerate_bbox` y el `image_id`
huérfano de `missing_image` quedan clausurados de antemano (ver el docstring de
`presentation/crops_report.py`): esos casos siguen probados en `test_crops.py`
sobre el motor en aislamiento. Aquí se ejercitan los que SÍ sobreviven a la
ingesta estricta: `out_of_bounds` (dimensión real vs. declarada) y
`unknown_category` (clase declarada pero fuera del contrato congelado).
"""

import hashlib
import io
import json
from pathlib import Path

import pytest
import yaml
from crops.models import FROZEN_CLASSES
from PIL import Image

from policies.models import load_quality_policy
from presentation.crops_report import build_crops_report, main
from presentation.release_resolver import ReleaseRejectedError, ReleaseSource, load_release_sources
from tests._dataset_fixtures import jpeg_bytes

REPO_ROOT = Path(__file__).resolve().parents[2]

QUALITY_YAML = """
min_images_per_class:
  threshold: {threshold}
  action: fail
max_imbalance_ratio:
  threshold: 999
  action: warn
max_small_object_ratio:
  threshold: 0.99
  width_px: 1
  height_px: 1
  action: warn
degenerate_boxes:
  threshold: 0
  action: fail
cross_split_leakage:
  threshold: 0
  action: fail
duplicate_similarity_threshold:
  threshold: 0.999
  action: warn
min_spatial_dispersion:
  threshold: 0.0
  action: warn
"""


def _policy(tmp_path, threshold=2):
    path = tmp_path / "quality.yaml"
    path.write_text(QUALITY_YAML.format(threshold=threshold), encoding="utf-8")
    return load_quality_policy(path)


def _write_dvc(dataset_dir, name, md5, nfiles):
    doc = {"outs": [{"md5": md5, "size": 1, "nfiles": nfiles, "hash": "md5", "path": name}]}
    (dataset_dir / f"{name}.dvc").write_text(yaml.safe_dump(doc), encoding="utf-8")


def _solid_jpeg(width, height):
    image = Image.new("RGB", (width, height), (30, 30, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _prepare_release(
    tmp_path,
    version,
    *,
    images,
    annotations,
    categories,
    image_bytes: dict[str, bytes] | None = None,
    status="warning",
    checks=None,
) -> ReleaseSource:
    """Arma un release completo (datos + `.dvc` + `quality.json` + catálogo),
    tan real para `resolve_release` como uno recuperado con `dvc pull`, con
    control total sobre `images`/`annotations`/`categories` para casos negativos.
    """
    dataset_dir = tmp_path / "data" / version
    (dataset_dir / "annotations").mkdir(parents=True)
    (dataset_dir / "images").mkdir(parents=True)
    (dataset_dir / "annotations" / "lote.json").write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": categories}),
        encoding="utf-8",
    )
    image_bytes = image_bytes or {}
    for image in images:
        data = image_bytes.get(image["file_name"], jpeg_bytes(image["id"]))
        (dataset_dir / "images" / image["file_name"]).write_bytes(data)

    annotations_md5 = f"{hashlib.md5(f'ann-{version}'.encode()).hexdigest()}.dir"
    images_md5 = f"{hashlib.md5(f'img-{version}'.encode()).hexdigest()}.dir"
    _write_dvc(dataset_dir, "annotations", annotations_md5, 1)
    _write_dvc(dataset_dir, "images", images_md5, len(images))

    reports_dir = tmp_path / "reports"
    release_dir = reports_dir / "releases" / version
    release_dir.mkdir(parents=True)
    checks = checks or [
        {
            "check_name": "min_images_per_class",
            "passed": True,
            "metric_value": 999.0,
            "details": {},
            "action": "fail",
        }
    ]
    (release_dir / "quality.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "dataset_version": version,
                "status": status,
                "checks": checks,
            }
        ),
        encoding="utf-8",
    )
    (release_dir / "splits.json").write_text("{}", encoding="utf-8")
    (reports_dir / "versions.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "releases": [
                    {
                        "dataset_version": version,
                        "quality_file": f"releases/{version}/quality.json",
                        "splits_file": f"releases/{version}/splits.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return ReleaseSource(
        dataset_dir=f"data/{version}", annotations_md5=annotations_md5, images_md5=images_md5
    )


def _annotation(id_, image_id, category_id, bbox):
    _x, _y, w, h = bbox
    return {
        "id": id_,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": list(bbox),
        "area": w * h,
        "iscrowd": 0,
        "segmentation": [],
    }


CATEGORIES = [{"id": 3, "name": "dog"}, {"id": 4, "name": "cat"}]


def _build(tmp_path, version, source):
    return build_crops_report(
        version,
        repo_root=tmp_path,
        reports_dir=tmp_path / "reports",
        sources={version: source},
        policy=_policy(tmp_path),
    )


def test_valid_release_reconciles_exactly_and_gate_is_ok(tmp_path):
    images = [{"id": i, "file_name": f"img{i}.jpg", "width": 64, "height": 64} for i in range(1, 5)]
    annotations = [
        _annotation(1, 1, 4, [0, 0, 10, 10]),
        _annotation(2, 2, 4, [5, 5, 10, 10]),
        _annotation(3, 3, 3, [0, 0, 10, 10]),
        _annotation(4, 4, 3, [5, 5, 10, 10]),
    ]
    source = _prepare_release(
        tmp_path, "v0.1.1", images=images, annotations=annotations, categories=CATEGORIES
    )

    report, result = _build(tmp_path, "v0.1.1", source)

    assert report.dataset_version == "v0.1.1"
    assert report.total_annotations == 4
    assert report.total_crops == 4
    assert report.total_exclusions == 0
    assert report.annotations_equal_crops_plus_exclusions is True
    assert report.exclusions_by_reason == {
        "degenerate_bbox": 0,
        "out_of_bounds": 0,
        "missing_image": 0,
        "unknown_category": 0,
    }
    assert report.crops_by_class == {"cat": 2, "dog": 2}
    assert report.originals_by_class == {"cat": 2, "dog": 2}
    assert report.resolver_originals_per_class == {"cat": 2, "dog": 2}
    assert report.matches_resolver_originals_per_class is True
    assert report.classes_below_minimum == []
    assert report.gate_status == "ok"
    assert len(result.crops) == 4


def test_out_of_bounds_from_real_pixel_mismatch_blocks_gate_and_flags_reconciliation(tmp_path):
    """El COCO declara 100x100 (CocoDataset no lo contrasta contra nada), pero
    el binario real es 20x20: la caja cabe en lo declarado, no en lo real."""
    images = [
        {"id": 1, "file_name": "dog_bad.jpg", "width": 100, "height": 100},
        {"id": 2, "file_name": "cat_ok.jpg", "width": 64, "height": 64},
    ]
    annotations = [
        _annotation(1, 1, 3, [50, 50, 10, 10]),  # dentro de 100x100, fuera de 20x20 real
        _annotation(2, 2, 4, [0, 0, 10, 10]),
    ]
    source = _prepare_release(
        tmp_path,
        "v0.1.1",
        images=images,
        annotations=annotations,
        categories=CATEGORIES,
        image_bytes={"dog_bad.jpg": _solid_jpeg(20, 20)},
    )

    report, _ = build_crops_report(
        "v0.1.1",
        repo_root=tmp_path,
        reports_dir=tmp_path / "reports",
        sources={"v0.1.1": source},
        policy=_policy(tmp_path, threshold=1),
    )

    assert report.exclusions_by_reason["out_of_bounds"] == 1
    assert report.total_crops == 1
    assert report.annotations_equal_crops_plus_exclusions is True
    assert report.crops_by_class["dog"] == 0
    assert report.originals_by_class["dog"] == 0
    # El resolver cuenta la anotación cruda (sin filtrar por validez del crop):
    # sigue viendo 1 original de "dog". El recuento por crops ve 0. Deben diferir.
    assert report.resolver_originals_per_class["dog"] == 1
    assert report.matches_resolver_originals_per_class is False
    assert report.classes_below_minimum == ["dog"]
    assert report.gate_status == "blocked"


def test_category_outside_frozen_classes_is_excluded_even_though_coco_declares_it_validly(tmp_path):
    categories = [*CATEGORIES, {"id": 7, "name": "horse"}]
    images = [
        {"id": 1, "file_name": "a.jpg", "width": 64, "height": 64},
        {"id": 2, "file_name": "b.jpg", "width": 64, "height": 64},
        {"id": 3, "file_name": "c.jpg", "width": 64, "height": 64},
    ]
    annotations = [
        _annotation(1, 1, 4, [0, 0, 10, 10]),
        _annotation(2, 2, 3, [0, 0, 10, 10]),
        _annotation(3, 3, 7, [0, 0, 10, 10]),  # "horse": estructuralmente válida, no congelada
    ]
    source = _prepare_release(
        tmp_path, "v0.1.1", images=images, annotations=annotations, categories=categories
    )

    report, _ = build_crops_report(
        "v0.1.1",
        repo_root=tmp_path,
        reports_dir=tmp_path / "reports",
        sources={"v0.1.1": source},
        policy=_policy(tmp_path, threshold=1),
    )

    assert report.exclusions_by_reason["unknown_category"] == 1
    assert report.total_crops == 2
    assert report.annotations_equal_crops_plus_exclusions is True
    assert "horse" not in report.crops_by_class
    assert "horse" not in report.originals_by_class
    assert set(report.crops_by_class) == FROZEN_CLASSES
    assert report.gate_status == "ok"


def test_undecodable_image_is_missing_image_not_a_crash(tmp_path):
    images = [
        {"id": 1, "file_name": "corrupt.jpg", "width": 64, "height": 64},
        {"id": 2, "file_name": "ok.jpg", "width": 64, "height": 64},
    ]
    annotations = [
        _annotation(1, 1, 3, [0, 0, 10, 10]),
        _annotation(2, 2, 4, [0, 0, 10, 10]),
    ]
    source = _prepare_release(
        tmp_path,
        "v0.1.1",
        images=images,
        annotations=annotations,
        categories=CATEGORIES,
        image_bytes={"corrupt.jpg": b"no es una imagen"},
    )

    report, _ = build_crops_report(
        "v0.1.1",
        repo_root=tmp_path,
        reports_dir=tmp_path / "reports",
        sources={"v0.1.1": source},
        policy=_policy(tmp_path, threshold=1),
    )

    assert report.exclusions_by_reason["missing_image"] == 1
    assert report.total_crops == 1


def test_reconciliation_flag_is_computed_not_hardcoded(tmp_path, monkeypatch):
    """`annotations_equal_crops_plus_exclusions` siempre da True en el motor real
    (D01-07 garantiza que ninguna anotación se pierde), así que ningún dataset
    real puede falsear este campo. Se fuerza un `CropResult` inconsistente para
    probar que el campo se calcula de verdad y no queda hardcodeado en True."""
    from crops.models import CropResult

    from presentation import crops_report as module

    images = [
        {"id": 1, "file_name": "a.jpg", "width": 64, "height": 64},
        {"id": 2, "file_name": "b.jpg", "width": 64, "height": 64},
    ]
    annotations = [_annotation(1, 1, 4, [0, 0, 10, 10]), _annotation(2, 2, 3, [0, 0, 10, 10])]
    source = _prepare_release(
        tmp_path, "v0.1.1", images=images, annotations=annotations, categories=CATEGORIES
    )
    monkeypatch.setattr(
        module, "generate_crops", lambda *a, **k: CropResult(crops=[], exclusions=[])
    )

    report, _ = build_crops_report(
        "v0.1.1",
        repo_root=tmp_path,
        reports_dir=tmp_path / "reports",
        sources={"v0.1.1": source},
        policy=_policy(tmp_path, threshold=1),
    )

    assert report.total_annotations == 2
    assert report.total_crops == 0
    assert report.total_exclusions == 0
    assert report.annotations_equal_crops_plus_exclusions is False


def test_release_rejected_by_the_resolver_is_not_silently_reported_as_empty(tmp_path):
    images = [{"id": 1, "file_name": "a.jpg", "width": 64, "height": 64}]
    annotations = [_annotation(1, 1, 4, [0, 0, 10, 10])]
    failing_check = [
        {
            "check_name": "degenerate_boxes",
            "passed": False,
            "metric_value": 1.0,
            "details": {},
            "action": "fail",
        }
    ]
    source = _prepare_release(
        tmp_path,
        "v0.1.1",
        images=images,
        annotations=annotations,
        categories=CATEGORIES,
        status="failed",
        checks=failing_check,
    )

    with pytest.raises(ReleaseRejectedError) as excinfo:
        _build(tmp_path, "v0.1.1", source)
    assert excinfo.value.reason == "quality_failed"


def test_cli_prints_report_and_exits_zero_when_gate_is_ok(tmp_path, monkeypatch, capsys):
    from presentation import crops_report as module

    images = [
        {"id": 1, "file_name": "a.jpg", "width": 64, "height": 64},
        {"id": 2, "file_name": "b.jpg", "width": 64, "height": 64},
    ]
    annotations = [_annotation(1, 1, 4, [0, 0, 10, 10]), _annotation(2, 2, 3, [0, 0, 10, 10])]
    source = _prepare_release(
        tmp_path, "v0.1.1", images=images, annotations=annotations, categories=CATEGORIES
    )
    monkeypatch.setattr(module, "APP_ROOT", tmp_path / "app")
    monkeypatch.setattr(module, "load_release_sources", lambda: {"v0.1.1": source})
    monkeypatch.setattr(module, "load_quality_policy", lambda: _policy(tmp_path, threshold=1))

    assert main(["v0.1.1"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["gate_status"] == "ok"
    assert payload["total_crops"] == 2


def test_cli_exits_one_when_the_gate_is_blocked(tmp_path, monkeypatch, capsys):
    """dog SÍ tiene una anotación cruda (pasa el propio gate del resolver, que no
    filtra por validez del crop), pero su única caja queda fuera de los límites
    reales: tras el motor de crops, dog cae a 0 y el gate de este reporte bloquea."""
    from presentation import crops_report as module

    images = [
        {"id": 1, "file_name": "dog_bad.jpg", "width": 100, "height": 100},
        {"id": 2, "file_name": "cat_ok.jpg", "width": 64, "height": 64},
    ]
    annotations = [
        _annotation(1, 1, 3, [50, 50, 10, 10]),  # dentro de 100x100, fuera de 20x20 real
        _annotation(2, 2, 4, [0, 0, 10, 10]),
    ]
    source = _prepare_release(
        tmp_path,
        "v0.1.1",
        images=images,
        annotations=annotations,
        categories=CATEGORIES,
        image_bytes={"dog_bad.jpg": _solid_jpeg(20, 20)},
    )
    monkeypatch.setattr(module, "APP_ROOT", tmp_path / "app")
    monkeypatch.setattr(module, "load_release_sources", lambda: {"v0.1.1": source})
    monkeypatch.setattr(module, "load_quality_policy", lambda: _policy(tmp_path, threshold=1))

    assert main(["v0.1.1"]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["gate_status"] == "blocked"
    assert payload["classes_below_minimum"] == ["dog"]


# --- contra el dataset real recuperado de DVC/S3 ----------------------------------

real_data = pytest.mark.skipif(
    not (REPO_ROOT / "data" / "raw" / "images").is_dir(),
    reason="datos reales no recuperados (dvc pull -r prod)",
)


@real_data
def test_real_v0_1_1_report_matches_the_manual_audit_from_issue_33():
    report, result = build_crops_report(
        "v0.1.1",
        repo_root=REPO_ROOT,
        reports_dir=REPO_ROOT / "reports",
        sources=load_release_sources(),
        policy=load_quality_policy(),
    )

    assert report.total_annotations == 668
    assert report.total_crops == 668
    assert report.total_exclusions == 0
    assert report.annotations_equal_crops_plus_exclusions is True
    assert report.exclusions_by_reason == {
        "degenerate_bbox": 0,
        "out_of_bounds": 0,
        "missing_image": 0,
        "unknown_category": 0,
    }
    assert report.crops_by_class == {"cat": 343, "dog": 325}
    assert report.originals_by_class == {"cat": 301, "dog": 300}
    assert report.resolver_originals_per_class == {"cat": 301, "dog": 300}
    assert report.matches_resolver_originals_per_class is True
    assert report.classes_below_minimum == []
    assert report.gate_status == "ok"
    assert len(result.crops) == 668
