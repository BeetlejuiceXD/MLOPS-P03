"""D03-03 — Fuentes reales de Training: release aprobado + manifest P3 congelado (D03-01).

El manifest congelado se arma aquí como lo hará D03-01: a partir del candidato real de
D02-04 (`build_manifest_candidate`) sobre un release sintético completo (datos, `.dvc`,
catálogo y quality reales para el resolver), con `frozen: true`, `test_split_hash` y
su propio `.dvc`. Estos fixtures solo acreditan el componente; el recorrido con v0.1.1
real y el manifest oficial se evidencia aparte (D03-03/D03-04).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from presentation.contracts import FrozenManifest, frozen_test_split_hash
from presentation.manifest_candidate import _manifest_hash, build_manifest_candidate
from splits.models import SplitsConfig
from tests.test_crops_report import CATEGORIES, _annotation, _policy, _prepare_release
from trainer_worker import sources
from trainer_worker.sources import (
    SourcesNotEligibleError,
    build_training_dataset,
    verify_training_sources,
)

VERSION = "v0.1.1"
N_PER_CLASS = 20  # 40 originales; el umbral de la política de prueba se ajusta a 20


def _release(tmp_path):
    """20 cat + 20 dog, un crop por imagen, y la imagen 1 con un SEGUNDO crop cat:
    el grupo indivisible de dos crops permite probar una fuga real entre particiones."""
    images = [
        {"id": i, "file_name": f"img{i}.jpg", "width": 64, "height": 64}
        for i in range(1, 2 * N_PER_CLASS + 1)
    ]
    annotations = [
        _annotation(i, i, 4 if i <= N_PER_CLASS else 3, [2, 3, 20, 12])
        for i in range(1, 2 * N_PER_CLASS + 1)
    ]
    annotations.append(_annotation(1000, 1, 4, [30, 30, 10, 16]))
    source = _prepare_release(
        tmp_path, VERSION, images=images, annotations=annotations, categories=CATEGORIES
    )
    return source, _policy(tmp_path, threshold=N_PER_CLASS)


def _freeze(tmp_path, source, policy, *, mutate=None, recompute=True, frozen=True):
    """Congela el candidato real como lo haría D03-01 y escribe artefacto + `.dvc`.

    `mutate(doc)` altera el documento antes de escribirlo; con `recompute=True` se
    recalculan `manifest_hash`/`test_split_hash` (un artefacto "coherente" pero
    manipulado), con `False` quedan los hashes originales (adulterado)."""
    summary, candidate = build_manifest_candidate(
        VERSION,
        repo_root=tmp_path,
        reports_dir=tmp_path / "reports",
        sources={VERSION: source},
        policy=policy,
        manifest_config=SplitsConfig(train=0.7, val=0.2, test=0.1, seed=42),
    )
    doc = {
        "manifest_version": summary.manifest_version,
        "manifest_hash": summary.manifest_hash,
        "dataset_version": VERSION,
        "dvc_release_hash": summary.dvc_release_hash,
        "images_md5": source.images_md5,
        "annotations_md5": source.annotations_md5,
        "seed": 42,
        "target_ratios": {"train": 0.7, "val": 0.2, "test": 0.1},
        "frozen": frozen,
        "test_split_hash": frozen_test_split_hash(list(candidate.assignments["test"])),
        "assignments": {name: sorted(ids) for name, ids in candidate.assignments.items()},
    }
    if mutate is not None:
        mutate(doc)
        if recompute:
            doc["manifest_hash"] = _manifest_hash(
                dataset_version=doc["dataset_version"],
                seed=doc["seed"],
                target_ratios=doc["target_ratios"],
                assignments={k: tuple(v) for k, v in doc["assignments"].items()},
            )
            doc["test_split_hash"] = frozen_test_split_hash(doc["assignments"]["test"])
    path = tmp_path / "data" / "p3" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_manifest_dvc(path)
    return path, doc


def _write_manifest_dvc(path: Path, md5: str | None = None) -> None:
    digest = md5 or hashlib.md5(path.read_bytes()).hexdigest()
    dvc = {"outs": [{"md5": digest, "size": path.stat().st_size, "hash": "md5", "path": path.name}]}
    path.with_name(path.name + ".dvc").write_text(yaml.safe_dump(dvc), encoding="utf-8")


def _verify(tmp_path, source, policy, path, doc, **overrides):
    kwargs = {
        "dataset_version": doc["dataset_version"],
        "manifest_hash": doc["manifest_hash"],
    } | overrides
    return verify_training_sources(
        kwargs["dataset_version"],
        kwargs["manifest_hash"],
        manifest_path=path,
        repo_root=tmp_path,
        reports_dir=tmp_path / "reports",
        sources={VERSION: source},
        policy=policy,
    )


def _reason(excinfo) -> str:
    return excinfo.value.reason


# --- camino elegible -----------------------------------------------------------------


def test_frozen_manifest_over_the_real_release_is_eligible(tmp_path):
    source, policy = _release(tmp_path)
    path, doc = _freeze(tmp_path, source, policy)

    verified = _verify(tmp_path, source, policy, path, doc)

    assert verified.release.dataset_version == VERSION
    assert isinstance(verified.manifest, FrozenManifest)
    assert verified.manifest.manifest_hash == doc["manifest_hash"]
    # El resumen publicado sale del artefacto verificado, congelado y con los conteos reales.
    assert verified.summary.frozen is True
    assert verified.summary.manifest_hash == doc["manifest_hash"]
    total = sum(len(ids) for ids in doc["assignments"].values())
    assert (
        verified.summary.splits.train.crops
        + verified.summary.splits.val.crops
        + verified.summary.splits.test.crops
        == total
        == 2 * N_PER_CLASS + 1
    )


def test_dataset_feeds_only_train_and_val_and_never_opens_test_pixels(tmp_path, monkeypatch):
    source, policy = _release(tmp_path)
    path, doc = _freeze(tmp_path, source, policy)
    verified = _verify(tmp_path, source, policy, path, doc)

    cropped: list[int] = []
    original_crop = sources._crop_pixels

    def spy(crop, images_dir, image_files):
        cropped.append(crop.crop_id)
        return original_crop(crop, images_dir, image_files)

    monkeypatch.setattr(sources, "_crop_pixels", spy)
    dataset = build_training_dataset(verified)

    assert [s.sample_id for s in dataset.train] == doc["assignments"]["train"]
    assert [s.sample_id for s in dataset.val] == doc["assignments"]["val"]
    assert dataset.test == ()  # el trainer no recibe la partición test
    assert set(cropped).isdisjoint(doc["assignments"]["test"])
    assert set(cropped) == set(doc["assignments"]["train"]) | set(doc["assignments"]["val"])


def test_samples_keep_label_group_and_the_real_crop_geometry(tmp_path):
    source, policy = _release(tmp_path)
    path, doc = _freeze(tmp_path, source, policy)
    verified = _verify(tmp_path, source, policy, path, doc)
    crops = {crop.crop_id: crop for crop in verified.crops}

    dataset = build_training_dataset(verified)

    for sample in dataset.train + dataset.val:
        crop = crops[sample.sample_id]
        assert sample.label == crop.category_name
        assert sample.group_id == crop.image_id
        assert isinstance(sample.image, Image.Image) and sample.image.mode == "RGB"
        assert sample.image.size == (crop.width, crop.height)


# --- rechazos antes de entrenar --------------------------------------------------------


def test_missing_manifest_artifact_is_rejected(tmp_path):
    source, policy = _release(tmp_path)
    path, doc = _freeze(tmp_path, source, policy)
    path.unlink()

    with pytest.raises(SourcesNotEligibleError) as excinfo:
        _verify(tmp_path, source, policy, path, doc)
    assert _reason(excinfo) == "manifest_missing"


def test_a_frozen_flag_alone_is_not_enough_candidate_not_frozen_is_rejected(tmp_path):
    source, policy = _release(tmp_path)
    path, doc = _freeze(tmp_path, source, policy, frozen=False)

    with pytest.raises(SourcesNotEligibleError) as excinfo:
        _verify(tmp_path, source, policy, path, doc)
    assert _reason(excinfo) == "manifest_not_frozen"


def test_manifest_not_versioned_with_dvc_is_rejected(tmp_path):
    source, policy = _release(tmp_path)
    path, doc = _freeze(tmp_path, source, policy)
    path.with_name(path.name + ".dvc").unlink()

    with pytest.raises(SourcesNotEligibleError) as excinfo:
        _verify(tmp_path, source, policy, path, doc)
    assert _reason(excinfo) == "manifest_not_versioned"


def test_artifact_edited_after_versioning_is_rejected(tmp_path):
    """El archivo ya no es el que DVC versionó (md5 distinto al del `.dvc`)."""
    source, policy = _release(tmp_path)
    path, doc = _freeze(tmp_path, source, policy)
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    with pytest.raises(SourcesNotEligibleError) as excinfo:
        _verify(tmp_path, source, policy, path, doc)
    assert _reason(excinfo) == "manifest_dvc_mismatch"


def test_adulterated_assignment_with_the_old_hash_is_rejected(tmp_path):
    source, policy = _release(tmp_path)

    def swap(doc):
        doc["assignments"]["train"][0], doc["assignments"]["val"][0] = (
            doc["assignments"]["val"][0],
            doc["assignments"]["train"][0],
        )

    path, doc = _freeze(tmp_path, source, policy, mutate=swap, recompute=False)

    with pytest.raises(SourcesNotEligibleError) as excinfo:
        _verify(tmp_path, source, policy, path, doc)
    assert _reason(excinfo) == "manifest_hash_mismatch"


def test_wrong_test_split_hash_is_rejected(tmp_path):
    source, policy = _release(tmp_path)

    def bad_test_hash(doc):
        doc["test_split_hash"] = "0" * 64

    path, doc = _freeze(tmp_path, source, policy, mutate=bad_test_hash, recompute=False)

    with pytest.raises(SourcesNotEligibleError) as excinfo:
        _verify(tmp_path, source, policy, path, doc)
    assert _reason(excinfo) == "test_hash_mismatch"


def test_leak_of_an_indivisible_group_across_splits_is_rejected(tmp_path):
    """Hashes recalculados (artefacto "coherente"), pero los dos crops de la imagen 1
    quedan en train y test: fuga real, detectada contra los grupos de los datos."""
    source, policy = _release(tmp_path)

    def leak(doc):
        assignments = doc["assignments"]
        for ids in assignments.values():
            for crop_id in (1, 1000):
                if crop_id in ids:
                    ids.remove(crop_id)
        assignments["train"].append(1)
        assignments["test"].append(1000)
        for ids in assignments.values():
            ids.sort()

    path, doc = _freeze(tmp_path, source, policy, mutate=leak)

    with pytest.raises(SourcesNotEligibleError) as excinfo:
        _verify(tmp_path, source, policy, path, doc)
    assert _reason(excinfo) == "invalid_partition"
    assert "Leakage" in excinfo.value.detail


def test_artifact_of_another_data_identity_is_rejected(tmp_path):
    source, policy = _release(tmp_path)

    def other_images(doc):
        doc["images_md5"] = "f" * 32 + ".dir"

    path, doc = _freeze(tmp_path, source, policy, mutate=other_images)

    with pytest.raises(SourcesNotEligibleError) as excinfo:
        _verify(tmp_path, source, policy, path, doc)
    assert _reason(excinfo) == "identity_mismatch"


@pytest.mark.parametrize("field", ["dataset_version", "manifest_hash"])
def test_request_for_other_sources_than_the_frozen_manifest_is_rejected(tmp_path, field):
    source, policy = _release(tmp_path)
    path, doc = _freeze(tmp_path, source, policy)
    other = {"dataset_version": "v9.9.9", "manifest_hash": "a" * 64}[field]

    with pytest.raises(SourcesNotEligibleError) as excinfo:
        _verify(tmp_path, source, policy, path, doc, **{field: other})
    assert _reason(excinfo) == "request_mismatch"


def test_release_not_approved_is_rejected_even_with_a_valid_manifest(tmp_path):
    source, policy = _release(tmp_path)
    path, doc = _freeze(tmp_path, source, policy)
    quality = tmp_path / "reports" / "releases" / VERSION / "quality.json"
    data = json.loads(quality.read_text(encoding="utf-8"))
    data["status"] = "failed"
    quality.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(SourcesNotEligibleError) as excinfo:
        _verify(tmp_path, source, policy, path, doc)
    assert _reason(excinfo).startswith("release_")
