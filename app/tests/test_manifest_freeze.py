"""D03-01 — auditoría y congelación del manifest P3.

Congela el candidato de D02-04 sin reconstruirlo: `train_val.json` (lo único que
consume el trainer) y `test.json` (custodia de Ale) son archivos separados, y el
registro de congelación guarda el sha256 de ambos. La auditoría es independiente
del generador: recalcula grupos (con puentes near-duplicate sin crops), hashes,
proporciones, clases y el mínimo de originales, y bloquea la congelación ante
cualquier violación."""

import dataclasses
import json
from pathlib import Path

import pytest
import yaml
from crops.models import Crop
from manifest.frozen import FrozenManifestError, load_train_val

from policies.models import load_quality_policy
from presentation.contracts import (
    ManifestSplitCounts,
    ManifestSplits,
    ManifestSummary,
    ManifestTargetRatios,
)
from presentation.manifest_candidate import _dvc_release_hash, _manifest_hash
from presentation.manifest_freeze import (
    FreezeBlockedError,
    _image_groups,
    _release_inputs,
    audit_frozen_on_disk,
    freeze_candidate,
    freeze_manifest,
    write_frozen_manifest,
)
from presentation.release_resolver import load_release_sources
from tests.test_crops_report import CATEGORIES, _annotation, _policy, _prepare_release, _solid_jpeg
from tests.test_manifest_candidate import _balanced_release

REPO_ROOT = Path(__file__).resolve().parents[2]
IMAGES_MD5 = "a" * 32 + ".dir"
ANNOTATIONS_MD5 = "b" * 32 + ".dir"
SPLITS = ("train", "val", "test")


# --- fixtures sintéticos puros (sin release, sin imágenes) ---------------------------


def _crop(crop_id, image_id, name):
    return Crop(
        crop_id=crop_id,
        image_id=image_id,
        annotation_id=crop_id,
        category_id=4 if name == "cat" else 3,
        category_name=name,
        bbox_original=[0.0, 0.0, 10.0, 10.0],
        bbox_pixels=[0, 0, 10, 10],
        width=10,
        height=10,
    )


def _dataset(n_train=14, n_val=4, n_test=2):
    """Una imagen = un crop, clases alternadas: 70/20/10 exacto con 20 imágenes y
    ambas clases en cada partición."""
    crops, assignments, next_id = [], {name: [] for name in SPLITS}, 1
    for name, n in zip(SPLITS, (n_train, n_val, n_test), strict=True):
        for k in range(n):
            crops.append(_crop(next_id, next_id, "cat" if k % 2 == 0 else "dog"))
            assignments[name].append(next_id)
            next_id += 1
    return crops, assignments


def _summary(crops, assignments, *, dataset_version="v0.1.1"):
    """Resumen coherente con `assignments`, construido SIN validar (como lo haría un
    generador defectuoso): la auditoría no puede apoyarse en los validadores del
    contrato para detectar la violación."""
    by_id = {crop.crop_id: crop for crop in crops}
    splits = {}
    for name in SPLITS:
        per_class = dict.fromkeys(("cat", "dog"), 0)
        for crop_id in assignments[name]:
            per_class[by_id[crop_id].category_name] += 1
        splits[name] = ManifestSplitCounts.model_construct(
            crops=len(assignments[name]),
            originals=len({by_id[c].image_id for c in assignments[name]}),
            crops_per_class=per_class,
        )
    ratios = {"train": 0.7, "val": 0.2, "test": 0.1}
    return ManifestSummary.model_construct(
        manifest_version=f"p3-{dataset_version}-s42",
        manifest_hash=_manifest_hash(
            dataset_version=dataset_version,
            seed=42,
            target_ratios=ratios,
            assignments={name: tuple(sorted(ids)) for name, ids in assignments.items()},
        ),
        dataset_version=dataset_version,
        dvc_release_hash=_dvc_release_hash(IMAGES_MD5, ANNOTATIONS_MD5),
        seed=42,
        target_ratios=ManifestTargetRatios.model_construct(**ratios),
        frozen=False,
        classes=["cat", "dog"],
        splits=ManifestSplits.model_construct(**splits),
    )


def _freeze(crops, assignments, *, duplicate_pairs=(), min_originals=2, summary=None):
    return freeze_candidate(
        summary if summary is not None else _summary(crops, assignments),
        assignments,
        crops=crops,
        file_names={crop.image_id: f"img{crop.image_id}.jpg" for crop in crops},
        duplicate_pairs=list(duplicate_pairs),
        images_md5=IMAGES_MD5,
        annotations_md5=ANNOTATIONS_MD5,
        min_originals_per_class=min_originals,
    )


def _blocked(reason, **kwargs):
    with pytest.raises(FreezeBlockedError) as excinfo:
        _freeze(**kwargs)
    assert excinfo.value.reason == reason, str(excinfo.value)
    return excinfo.value


# --- congelación válida ---------------------------------------------------------------


def test_valid_candidate_freezes_with_the_same_identity():
    crops, assignments = _dataset()
    candidate = _summary(crops, assignments)

    frozen = _freeze(crops, assignments, summary=candidate)

    assert frozen.summary.frozen is True
    # Congelar no cambia la identidad: el hash del candidato no incluye `frozen`.
    assert frozen.summary.manifest_hash == candidate.manifest_hash
    assert frozen.record.candidate_manifest_hash == candidate.manifest_hash
    assert frozen.record.manifest_hash == candidate.manifest_hash
    assert frozen.record.frozen is True
    ManifestSummary.model_validate(frozen.summary.model_dump())


def test_train_val_and_test_payloads_are_separate_and_hashed():
    import hashlib

    crops, assignments = _dataset()

    frozen = _freeze(crops, assignments)

    train_val = json.loads(frozen.train_val)
    test = json.loads(frozen.test)
    assert set(train_val["splits"]) == {"train", "val"}
    assert set(test["splits"]) == {"test"}
    assert [e["crop_id"] for e in test["splits"]["test"]] == sorted(assignments["test"])
    assert frozen.record.test_sha256 == hashlib.sha256(frozen.test).hexdigest()
    assert frozen.record.train_val_sha256 == hashlib.sha256(frozen.train_val).hexdigest()
    for payload in (train_val, test):
        assert payload["manifest_hash"] == frozen.summary.manifest_hash
        assert payload["dataset_version"] == "v0.1.1"
        assert payload["dvc_release_hash"] == frozen.summary.dvc_release_hash
    entry = train_val["splits"]["train"][0]
    assert set(entry) == {
        "crop_id",
        "image_id",
        "annotation_id",
        "category_name",
        "file_name",
        "bbox_pixels",
    }


def test_freezing_is_byte_for_byte_deterministic():
    crops, assignments = _dataset()

    first = _freeze(crops, assignments)
    second = _freeze(list(reversed(crops)), {k: list(reversed(v)) for k, v in assignments.items()})

    assert first.train_val == second.train_val
    assert first.test == second.test
    assert first.record == second.record


def test_record_reports_originals_per_class_and_groups():
    crops, assignments = _dataset()

    record = _freeze(crops, assignments).record

    assert record.originals_per_class["train"] == {"cat": 7, "dog": 7}
    assert record.originals_per_class["test"] == {"cat": 1, "dog": 1}
    assert record.groups_total == 20
    assert record.groups_multi_image == 0
    assert record.groups_crossing_splits == 0


# --- negativas: nada se congela -------------------------------------------------------


def test_group_crossing_splits_blocks_freeze():
    crops, assignments = _dataset()
    train_image, test_image = assignments["train"][0], assignments["test"][0]

    _blocked(
        "group_crosses_splits",
        crops=crops,
        assignments=assignments,
        duplicate_pairs=[{"image_id_a": train_image, "image_id_b": test_image}],
    )


def test_near_duplicate_bridge_without_crops_still_joins_the_group():
    """A (train) ≈ B (sin crops) ≈ C (test): A y C son el mismo grupo aunque B no
    aparezca en ninguna partición (revisión de Heri en PR #54)."""
    crops, assignments = _dataset()
    train_image, test_image, bridge = assignments["train"][0], assignments["test"][0], 999

    _blocked(
        "group_crosses_splits",
        crops=crops,
        assignments=assignments,
        duplicate_pairs=[
            {"image_id_a": train_image, "image_id_b": bridge},
            {"image_id_a": bridge, "image_id_b": test_image},
        ],
    )


def test_missing_class_in_test_blocks_freeze():
    crops, assignments = _dataset()
    by_id = {crop.crop_id: crop for crop in crops}
    # Toda la partición test pasa a ser "cat" (mismo tamaño): falta dog en test.
    crops = [
        _crop(c.crop_id, c.image_id, "cat") if c.crop_id in assignments["test"] else c
        for c in by_id.values()
    ]

    _blocked("class_missing", crops=crops, assignments=assignments)


def test_299_originals_of_a_class_blocks_freeze():
    crops, assignments = _dataset()  # 10 cat, 10 dog originales

    error = _blocked(
        "insufficient_originals", crops=crops, assignments=assignments, min_originals=11
    )
    assert error.classes_below_minimum == ["cat", "dog"]


def test_exactly_the_minimum_originals_still_freezes():
    crops, assignments = _dataset()  # 10 cat, 10 dog: justo en el umbral

    assert _freeze(crops, assignments, min_originals=10).record.min_originals_per_class == 10


def test_ratio_outside_tolerance_blocks_freeze():
    crops, assignments = _dataset(n_train=10, n_val=6, n_test=4)  # 50/30/20

    _blocked("ratio_out_of_tolerance", crops=crops, assignments=assignments)


def test_overlapping_originals_block_freeze():
    crops, assignments = _dataset()
    # Un crop extra del mismo original que ya está en train, asignado a val.
    image_in_train = assignments["train"][0]
    crops = [*crops, _crop(500, image_in_train, "cat")]
    assignments = {**assignments, "val": [*assignments["val"], 500]}

    _blocked("overlap", crops=crops, assignments=assignments)


def test_unassigned_crop_blocks_freeze():
    crops, assignments = _dataset()
    crops = [*crops, _crop(500, 500, "cat")]

    _blocked("coverage", crops=crops, assignments=assignments)


def test_tampered_manifest_hash_blocks_freeze():
    crops, assignments = _dataset()
    summary = _summary(crops, assignments)
    summary = summary.model_copy(update={"manifest_hash": "f" * 64})

    _blocked("manifest_hash_mismatch", crops=crops, assignments=assignments, summary=summary)


def test_tampered_release_hash_blocks_freeze():
    crops, assignments = _dataset()
    summary = _summary(crops, assignments).model_copy(update={"dvc_release_hash": "e" * 64})

    _blocked("release_hash_mismatch", crops=crops, assignments=assignments, summary=summary)


def test_summary_counts_disagreeing_with_assignments_block_freeze():
    crops, assignments = _dataset()
    summary = _summary(crops, assignments)
    wrong = summary.splits.model_copy(
        update={"test": summary.splits.test.model_copy(update={"crops": 3})}
    )

    _blocked(
        "summary_mismatch",
        crops=crops,
        assignments=assignments,
        summary=summary.model_copy(update={"splits": wrong}),
    )


# --- escritura, relectura y auditoría desde disco ------------------------------------


def _write(tmp_path, frozen):
    manifest_dir, reports_dir = tmp_path / "data" / "manifest_p3", tmp_path / "reports"
    write_frozen_manifest(frozen, manifest_dir=manifest_dir, reports_dir=reports_dir)
    return manifest_dir, reports_dir


def test_written_files_are_exactly_the_frozen_bytes(tmp_path):
    crops, assignments = _dataset()
    frozen = _freeze(crops, assignments)

    manifest_dir, reports_dir = _write(tmp_path, frozen)

    assert (manifest_dir / "train_val.json").read_bytes() == frozen.train_val
    assert (manifest_dir / "test.json").read_bytes() == frozen.test
    summary = ManifestSummary.model_validate_json(
        (reports_dir / "manifest_p3.json").read_text(encoding="utf-8")
    )
    assert summary.frozen is True
    record = json.loads((reports_dir / "manifest_p3_freeze.json").read_text(encoding="utf-8"))
    assert record["test_sha256"] == frozen.record.test_sha256


def test_trainer_loader_returns_only_train_and_val(tmp_path):
    crops, assignments = _dataset()
    manifest_dir, reports_dir = _write(tmp_path, _freeze(crops, assignments))
    (manifest_dir / "test.json").unlink()  # el trainer nunca necesita el test

    loaded = load_train_val(
        manifest_dir / "train_val.json", reports_dir / "manifest_p3_freeze.json"
    )

    assert set(loaded["splits"]) == {"train", "val"}
    assert len(loaded["splits"]["train"]) == 14
    assert "test" not in json.dumps(loaded["splits"])


def test_trainer_loader_rejects_a_tampered_train_val(tmp_path):
    crops, assignments = _dataset()
    manifest_dir, reports_dir = _write(tmp_path, _freeze(crops, assignments))
    path = manifest_dir / "train_val.json"
    path.write_bytes(path.read_bytes().replace(b'"dog"', b'"cat"', 1))

    with pytest.raises(FrozenManifestError, match="sha256"):
        load_train_val(path, reports_dir / "manifest_p3_freeze.json")


def test_trainer_loader_rejects_an_unfrozen_record(tmp_path):
    crops, assignments = _dataset()
    manifest_dir, reports_dir = _write(tmp_path, _freeze(crops, assignments))
    record_path = reports_dir / "manifest_p3_freeze.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["frozen"] = False
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(FrozenManifestError, match="congelado"):
        load_train_val(manifest_dir / "train_val.json", record_path)


# --- extremo a extremo sobre un release sintético --------------------------------------


def _freeze_release(tmp_path, source, policy):
    return freeze_manifest(
        "v0.1.1",
        repo_root=tmp_path,
        reports_dir=tmp_path / "reports",
        sources={"v0.1.1": source},
        policy=policy,
    )


def test_end_to_end_freeze_keeps_the_candidate_and_audits_from_disk(tmp_path):
    source, policy = _balanced_release(tmp_path)

    frozen = _freeze_release(tmp_path, source, policy)
    again = _freeze_release(tmp_path, source, policy)
    manifest_dir = tmp_path / "data" / "manifest_p3"
    write_frozen_manifest(frozen, manifest_dir=manifest_dir, reports_dir=tmp_path / "out")

    assert frozen.train_val == again.train_val and frozen.test == again.test
    record = audit_frozen_on_disk(
        "v0.1.1",
        repo_root=tmp_path,
        reports_dir=tmp_path / "reports",
        sources={"v0.1.1": source},
        policy=policy,
        manifest_dir=manifest_dir,
        frozen_reports_dir=tmp_path / "out",
    )
    assert record == frozen.record


def test_audit_from_disk_detects_a_tampered_test_file(tmp_path):
    source, policy = _balanced_release(tmp_path)
    frozen = _freeze_release(tmp_path, source, policy)
    manifest_dir = tmp_path / "data" / "manifest_p3"
    write_frozen_manifest(frozen, manifest_dir=manifest_dir, reports_dir=tmp_path / "out")
    test_path = manifest_dir / "test.json"
    test_path.write_bytes(test_path.read_bytes().replace(b'"dog"', b'"cat"', 1))

    with pytest.raises(FreezeBlockedError) as excinfo:
        audit_frozen_on_disk(
            "v0.1.1",
            repo_root=tmp_path,
            reports_dir=tmp_path / "reports",
            sources={"v0.1.1": source},
            policy=policy,
            manifest_dir=manifest_dir,
            frozen_reports_dir=tmp_path / "out",
        )
    assert excinfo.value.reason == "test_hash_mismatch"


def test_audit_from_disk_detects_a_consistent_manifest_that_is_not_the_candidate(tmp_path):
    """Archivos, hashes y registro coherentes entre sí, pero con dos originales de la
    misma clase intercambiados entre train y val: no es el candidato de D02-04."""
    source, policy = _balanced_release(tmp_path)
    frozen = _freeze_release(tmp_path, source, policy)
    inputs = _release_inputs(
        "v0.1.1",
        repo_root=tmp_path,
        reports_dir=tmp_path / "reports",
        sources={"v0.1.1": source},
        policy=policy,
    )
    by_id = {crop.crop_id: crop for crop in inputs.crops}
    splits = {**json.loads(frozen.train_val)["splits"], **json.loads(frozen.test)["splits"]}
    assignments = {name: [e["crop_id"] for e in splits[name]] for name in SPLITS}
    split_of = {c: name for name in SPLITS for c in assignments[name]}
    crops_of_image = {}
    for crop in inputs.crops:
        crops_of_image.setdefault(crop.image_id, []).append(crop.crop_id)
    # Grupos near-duplicate completos (indivisibles) con la misma composición de
    # clases, uno en train y otro en val: intercambiarlos respeta todas las reglas.
    groups = {}
    for group in _image_groups(set(crops_of_image), inputs.duplicate_pairs):
        ids = [c for i in group for c in crops_of_image[i]]
        key = (split_of[ids[0]], tuple(sorted(by_id[c].category_name for c in ids)))
        groups.setdefault(key, []).append(ids)
    train_group, val_group = next(
        (groups[key][0], groups[("val", key[1])][0])
        for key in groups
        if key[0] == "train" and ("val", key[1]) in groups
    )
    assignments["train"] = [c for c in assignments["train"] if c not in train_group] + val_group
    assignments["val"] = [c for c in assignments["val"] if c not in val_group] + train_group
    swapped_hash = _manifest_hash(
        dataset_version="v0.1.1",
        seed=42,
        target_ratios={"train": 0.7, "val": 0.2, "test": 0.1},
        assignments={name: tuple(sorted(ids)) for name, ids in assignments.items()},
    )
    tampered = freeze_candidate(
        frozen.summary.model_copy(update={"manifest_hash": swapped_hash, "frozen": False}),
        assignments,
        crops=inputs.crops,
        file_names=inputs.file_names,
        duplicate_pairs=inputs.duplicate_pairs,
        images_md5=inputs.images_md5,
        annotations_md5=inputs.annotations_md5,
        min_originals_per_class=inputs.min_originals_per_class,
    )
    assert tampered.summary.manifest_hash != frozen.summary.manifest_hash
    manifest_dir = tmp_path / "data" / "manifest_p3"
    write_frozen_manifest(tampered, manifest_dir=manifest_dir, reports_dir=tmp_path / "out")

    with pytest.raises(FreezeBlockedError) as excinfo:
        audit_frozen_on_disk(
            "v0.1.1",
            repo_root=tmp_path,
            reports_dir=tmp_path / "reports",
            sources={"v0.1.1": source},
            policy=policy,
            manifest_dir=manifest_dir,
            frozen_reports_dir=tmp_path / "out",
        )
    assert excinfo.value.reason == "candidate_drift"


def test_end_to_end_299_originals_after_exclusions_blocks_freeze(tmp_path):
    """El release tiene 300+300 originales (D01-02 lo acepta), pero una caja de dog
    cae fuera del binario real: tras la exclusión de crops quedan 299 dog."""
    images = [
        {"id": i, "file_name": f"img{i}.jpg", "width": 64, "height": 64} for i in range(1, 601)
    ]
    images[599] = {"id": 600, "file_name": "dog_bad.jpg", "width": 100, "height": 100}
    annotations = [
        _annotation(i, i, 4 if i <= 300 else 3, [0, 0, 10, 10]) for i in range(1, 600)
    ] + [_annotation(600, 600, 3, [50, 50, 10, 10])]  # fuera del binario real 20x20
    source = _prepare_release(
        tmp_path,
        "v0.1.1",
        images=images,
        annotations=annotations,
        categories=CATEGORIES,
        image_bytes={"dog_bad.jpg": _solid_jpeg(20, 20)},
    )

    with pytest.raises(FreezeBlockedError) as excinfo:
        _freeze_release(tmp_path, source, _policy(tmp_path, threshold=300))
    assert excinfo.value.reason == "insufficient_originals"
    assert excinfo.value.classes_below_minimum == ["dog"]
    assert not (tmp_path / "data" / "manifest_p3").exists()


# --- etapa DVC --------------------------------------------------------------------------


def test_dvc_stage_versions_train_val_and_test_as_separate_outputs():
    stage = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"][
        "manifest_p3"
    ]

    assert stage["wdir"] == "app"
    assert stage["cmd"].endswith("dvc_manifest_p3_stage.py")
    outs = {next(iter(o)) if isinstance(o, dict) else o: o for o in stage["outs"]}
    # train_val y test: cacheados en DVC (remote prod), cada uno recuperable por separado.
    assert "../data/manifest_p3/train_val.json" in outs
    assert "../data/manifest_p3/test.json" in outs
    for path in ("../reports/manifest_p3.json", "../reports/manifest_p3_freeze.json"):
        assert outs[path][path]["cache"] is False  # pequeños, versionados en git
    assert {"../data/raw/annotations", "../data/raw/images"} <= set(stage["deps"])
    # Dependencias de código como archivos: una carpeta arrastra __pycache__ al md5
    # y la etapa aparecería modificada en cada clon limpio.
    code_deps = [d for d in stage["deps"] if not d.startswith("../")]
    assert "presentation/manifest_freeze.py" in code_deps
    assert all((REPO_ROOT / "app" / d).is_file() for d in code_deps), code_deps
    assert stage["params"] == [{"manifest/manifest.yaml": ["train", "val", "test", "seed"]}]


# --- contra el dataset real (v0.1.1 recuperado de DVC) ---------------------------------

real_data = pytest.mark.skipif(
    not (REPO_ROOT / "data" / "raw" / "images").is_dir(),
    reason="datos reales no recuperados (dvc pull -r prod)",
)


@real_data
def test_real_v0_1_1_freezes_the_d02_04_candidate_unchanged():
    frozen = freeze_manifest(
        "v0.1.1",
        repo_root=REPO_ROOT,
        reports_dir=REPO_ROOT / "reports",
        sources=load_release_sources(),
        policy=load_quality_policy(),
    )

    # Mismo candidato que D02-04 (PR #54): la congelación no reconstruye el split.
    assert frozen.summary.manifest_hash == (
        "0c03c3951554b5dbf096c37468f0fb5f04e9c62c033604acabf366fb11375c43"
    )
    assert frozen.summary.frozen is True
    assert frozen.record.originals_per_class == {
        "train": {"cat": 198, "dog": 225},
        "val": {"cat": 69, "dog": 57},
        "test": {"cat": 34, "dog": 18},
    }
    assert sum(frozen.record.crops.values()) == 668
    assert sum(frozen.record.originals.values()) == 600
    assert frozen.record.groups_crossing_splits == 0


def test_frozen_manifest_dataclass_is_immutable():
    crops, assignments = _dataset()
    frozen = _freeze(crops, assignments)

    with pytest.raises(dataclasses.FrozenInstanceError):
        frozen.test = b"{}"
