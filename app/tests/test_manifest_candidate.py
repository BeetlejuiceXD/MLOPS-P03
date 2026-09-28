"""D02-04 — orquestación real del manifest P3: `release_resolver` (D01-02) +
`crops_report` (D02-02) + `analyzers.duplicates` + `manifest.generator` (D02-04).

Reutiliza el fixture-builder de D02-02 (`tests.test_crops_report._prepare_release`)
para no duplicar ~80 líneas de armado de release sintético: mismos `.dvc`,
catálogo y `quality.json` reales para `resolve_release`, con control total sobre
`images`/`annotations`/`categories`."""

import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from crops.models import FROZEN_CLASSES
from PIL import Image

from policies.models import load_quality_policy
from presentation.contracts import ManifestSummary
from presentation.manifest_candidate import ManifestBlockedError, build_manifest_candidate, main
from presentation.release_resolver import ReleaseRejectedError, load_release_sources
from splits.models import SplitsConfig
from tests.test_crops_report import CATEGORIES, _annotation, _policy, _prepare_release

APP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]


def _manifest_config(**overrides):
    values = {"train": 0.7, "val": 0.2, "test": 0.1, "seed": 42, **overrides}
    return SplitsConfig(**values)


def _balanced_release(tmp_path, version="v0.1.1", *, n_per_class=300, threshold=300):
    """Un release sintético a la escala real del umbral (300+300), sin duplicados
    ni exclusiones: suficientemente grande para que ±5 pp sea alcanzable de verdad,
    no solo en teoría."""
    images = [
        {"id": i, "file_name": f"cat{i}.jpg", "width": 64, "height": 64}
        for i in range(1, n_per_class + 1)
    ] + [
        {"id": n_per_class + i, "file_name": f"dog{i}.jpg", "width": 64, "height": 64}
        for i in range(1, n_per_class + 1)
    ]
    annotations = [_annotation(i, i, 4, [0, 0, 10, 10]) for i in range(1, n_per_class + 1)] + [
        _annotation(n_per_class + i, n_per_class + i, 3, [0, 0, 10, 10])
        for i in range(1, n_per_class + 1)
    ]
    source = _prepare_release(
        tmp_path, version, images=images, annotations=annotations, categories=CATEGORIES
    )
    return source, _policy(tmp_path, threshold=threshold)


def _build(tmp_path, version, source, policy, config=None):
    return build_manifest_candidate(
        version,
        repo_root=tmp_path,
        reports_dir=tmp_path / "reports",
        sources={version: source},
        policy=policy,
        manifest_config=config,
    )


# --- candidato válido, extremo a extremo --------------------------------------------


def test_valid_candidate_matches_the_contract_and_is_frozen_false(tmp_path):
    source, policy = _balanced_release(tmp_path)

    summary, candidate = _build(tmp_path, "v0.1.1", source, policy, _manifest_config())

    assert isinstance(summary, ManifestSummary)
    assert summary.frozen is False
    assert summary.seed == 42
    assert summary.dataset_version == "v0.1.1"
    assert summary.manifest_version == "p3-v0.1.1-s42"
    assert set(summary.classes) == set(FROZEN_CLASSES)
    total_crops = summary.splits.train.crops + summary.splits.val.crops + summary.splits.test.crops
    assert total_crops == 600
    total_originals = (
        summary.splits.train.originals
        + summary.splits.val.originals
        + summary.splits.test.originals
    )
    assert total_originals == 600
    # crop_id == annotation_id (D01-07): la unión de las tres particiones cubre
    # exactamente los 600 IDs de entrada, sin huecos ni repeticiones.
    all_ids = (
        set(candidate.assignments["train"])
        | set(candidate.assignments["val"])
        | set(candidate.assignments["test"])
    )
    assert all_ids == set(range(1, 601))


def test_dvc_release_hash_matches_the_frozen_formula_from_33(tmp_path):
    source, policy = _balanced_release(tmp_path)

    summary, _ = _build(tmp_path, "v0.1.1", source, policy, _manifest_config())

    expected = hashlib.sha256(f"{source.images_md5}:{source.annotations_md5}".encode()).hexdigest()
    assert summary.dvc_release_hash == expected


def test_manifest_hash_is_stable_across_repeated_runs(tmp_path):
    source, policy = _balanced_release(tmp_path)

    first, _ = _build(tmp_path, "v0.1.1", source, policy, _manifest_config())
    second, _ = _build(tmp_path, "v0.1.1", source, policy, _manifest_config())

    assert first.manifest_hash == second.manifest_hash
    assert first.model_dump_json() == second.model_dump_json()


def test_manifest_hash_is_sensitive_to_the_actual_assignment_not_hardcoded():
    """`manifest_hash` no puede depender solo de versión/ratios/seed: dos
    asignaciones distintas (mismos conteos, distinto reparto de IDs) deben dar
    hashes distintos. La seed del contrato está fija en 42 (#33), así que se
    prueba `_manifest_hash` directamente en vez de forzar otra seed real."""
    from presentation.manifest_candidate import _manifest_hash

    common = {
        "dataset_version": "v0.1.1",
        "seed": 42,
        "target_ratios": {"train": 0.7, "val": 0.2, "test": 0.1},
    }
    assignment_a = {"train": (1, 2, 3), "val": (4,), "test": (5,)}
    assignment_b = {"train": (1, 2, 4), "val": (3,), "test": (5,)}  # mismos conteos, otro reparto

    hash_a = _manifest_hash(**common, assignments=assignment_a)
    hash_b = _manifest_hash(**common, assignments=assignment_b)
    hash_a_again = _manifest_hash(**common, assignments=assignment_a)

    assert hash_a != hash_b
    assert hash_a == hash_a_again


def test_default_manifest_config_loads_the_real_yaml_when_not_overridden(tmp_path):
    """`manifest_config=None` debe usar `manifest/manifest.yaml` (70/20/10, seed 42),
    no un valor hardcodeado en el test."""
    source, policy = _balanced_release(tmp_path)

    summary, _ = build_manifest_candidate(
        "v0.1.1",
        repo_root=tmp_path,
        reports_dir=tmp_path / "reports",
        sources={"v0.1.1": source},
        policy=policy,
    )

    assert summary.seed == 42
    assert summary.target_ratios.train == 0.7


# --- bloqueos: gate de origen y release no elegible ---------------------------------


def test_gate_blocked_raises_without_producing_any_candidate(tmp_path):
    """dog empieza en exactamente 300 (admisible para el resolver), pero una única
    exclusión real (out_of_bounds) lo deja en 299: el candidato no se genera."""

    def _solid_jpeg(width, height):
        image = Image.new("RGB", (width, height), (30, 30, 30))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        return buffer.getvalue()

    images = [
        {"id": 1, "file_name": "dog_bad.jpg", "width": 100, "height": 100},
        {"id": 2, "file_name": "cat_ok.jpg", "width": 64, "height": 64},
    ]
    annotations = [
        _annotation(1, 1, 3, [50, 50, 10, 10]),  # dentro de 100x100, fuera del binario real 20x20
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
    policy = _policy(tmp_path, threshold=1)

    with pytest.raises(ManifestBlockedError) as excinfo:
        _build(tmp_path, "v0.1.1", source, policy, _manifest_config())
    assert excinfo.value.reason == "insufficient_originals_after_exclusions"
    assert excinfo.value.classes_below_minimum == ["dog"]


def test_release_rejected_by_the_resolver_propagates_not_swallowed(tmp_path):
    images = [
        {"id": i, "file_name": f"cat{i}.jpg", "width": 64, "height": 64} for i in range(1, 4)
    ] + [
        {"id": 100 + i, "file_name": f"dog{i}.jpg", "width": 64, "height": 64} for i in range(1, 4)
    ]
    annotations = [_annotation(i, i, 4, [0, 0, 10, 10]) for i in range(1, 4)] + [
        _annotation(100 + i, 100 + i, 3, [0, 0, 10, 10]) for i in range(1, 4)
    ]
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
    policy = _policy(tmp_path, threshold=1)

    with pytest.raises(ReleaseRejectedError) as excinfo:
        _build(tmp_path, "v0.1.1", source, policy, _manifest_config())
    assert excinfo.value.reason == "quality_failed"


def test_infeasible_tolerance_is_reported_not_silently_accepted(tmp_path):
    """Un solo grupo indivisible de 90 crops sobre 100 totales no puede acomodarse
    dentro de ±5 pp de 70/20/10 sin partirlo: el candidato no puede cumplir el
    contrato de Hannah (`ManifestSummary`), y `build_manifest_candidate` debe
    reportarlo, no fabricar un resumen que lo viole."""
    heavy_dog = [{"id": 1, "file_name": "dog_heavy.jpg", "width": 64, "height": 64}]
    other_images = [
        {"id": 100 + i, "file_name": f"cat{i}.jpg", "width": 64, "height": 64} for i in range(1, 11)
    ]
    annotations = [_annotation(i, 1, 3, [0, 0, 10, 10]) for i in range(1, 91)] + [
        _annotation(1000 + i, 100 + i, 4, [0, 0, 10, 10]) for i in range(1, 11)
    ]
    source = _prepare_release(
        tmp_path,
        "v0.1.1",
        images=heavy_dog + other_images,
        annotations=annotations,
        categories=CATEGORIES,
    )
    policy = _policy(tmp_path, threshold=1)

    with pytest.raises(ManifestBlockedError) as excinfo:
        _build(tmp_path, "v0.1.1", source, policy, _manifest_config())
    assert excinfo.value.reason == "invalid_partition"


# --- CLI -----------------------------------------------------------------------------


def test_cli_prints_summary_and_exits_zero(tmp_path, monkeypatch, capsys):
    from presentation import manifest_candidate as module

    source, policy = _balanced_release(tmp_path)
    monkeypatch.setattr(module, "APP_ROOT", tmp_path / "app")
    monkeypatch.setattr(module, "load_release_sources", lambda: {"v0.1.1": source})
    monkeypatch.setattr(module, "load_quality_policy", lambda: policy)

    assert main(["v0.1.1"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["frozen"] is False
    assert payload["seed"] == 42


def test_cli_exits_one_and_reports_the_reason_when_blocked(tmp_path, monkeypatch, capsys):
    from presentation import manifest_candidate as module

    images = [{"id": 1, "file_name": "a.jpg", "width": 64, "height": 64}]
    annotations = [_annotation(1, 1, 4, [0, 0, 10, 10])]  # solo cat: dog queda en 0
    source = _prepare_release(
        tmp_path, "v0.1.1", images=images, annotations=annotations, categories=CATEGORIES
    )
    policy = _policy(tmp_path, threshold=1)
    monkeypatch.setattr(module, "APP_ROOT", tmp_path / "app")
    monkeypatch.setattr(module, "load_release_sources", lambda: {"v0.1.1": source})
    monkeypatch.setattr(module, "load_quality_policy", lambda: policy)

    assert main(["v0.1.1"]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["resolved"] is False
    assert "reason" in payload


# --- contra el dataset real recuperado de DVC/S3 ------------------------------------

real_data = pytest.mark.skipif(
    not (REPO_ROOT / "data" / "raw" / "images").is_dir(),
    reason="datos reales no recuperados (dvc pull -r prod)",
)


@real_data
def test_real_v0_1_1_manifest_candidate_is_valid_and_reproducible():
    summary, candidate = build_manifest_candidate(
        "v0.1.1",
        repo_root=REPO_ROOT,
        reports_dir=REPO_ROOT / "reports",
        sources=load_release_sources(),
        policy=load_quality_policy(),
    )

    assert summary.frozen is False
    assert summary.seed == 42
    assert summary.dataset_version == "v0.1.1"
    assert summary.manifest_version == "p3-v0.1.1-s42"
    assert set(summary.classes) == {"cat", "dog"}
    total_crops = summary.splits.train.crops + summary.splits.val.crops + summary.splits.test.crops
    assert total_crops == 668
    total_originals = (
        summary.splits.train.originals
        + summary.splits.val.originals
        + summary.splits.test.originals
    )
    assert total_originals == 600
    for name in ("train", "val", "test"):
        split = getattr(summary.splits, name)
        assert split.crops_per_class.get("cat", 0) > 0
        assert split.crops_per_class.get("dog", 0) > 0

    again, _ = build_manifest_candidate(
        "v0.1.1",
        repo_root=REPO_ROOT,
        reports_dir=REPO_ROOT / "reports",
        sources=load_release_sources(),
        policy=load_quality_policy(),
    )
    assert again.manifest_hash == summary.manifest_hash
    assert set(candidate.assignments["train"]) & set(candidate.assignments["val"]) == set()
    assert set(candidate.assignments["train"]) & set(candidate.assignments["test"]) == set()
    assert set(candidate.assignments["val"]) & set(candidate.assignments["test"]) == set()


@real_data
def test_real_candidate_is_byte_identical_across_processes_regardless_of_hash_seed():
    """Misma lección que D02-02 (PR #50): se corre la tubería real completa en dos
    procesos con PYTHONHASHSEED distinto y se exige el mismo JSON byte a byte."""
    outputs = []
    for seed in ("0", "1"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run(
            [sys.executable, "-m", "presentation.manifest_candidate", "v0.1.1"],
            cwd=APP_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.append(result.stdout.strip())

    assert outputs[0], "el subproceso no produjo salida"
    assert outputs[0] == outputs[1]
