"""D01-02 — resolvedor de release: versión -> fuente de datos verificada.

Los tests de componente construyen un repo sintético en `tmp_path`; no acreditan
procedencia real. Los tests `real_data` corren contra `data/raw` recuperado con DVC y se
omiten cuando esos datos no están en el disco (CI sin `dvc pull`).
"""

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from policies.models import load_quality_policy
from presentation.release_resolver import (
    ReleaseRejectedError,
    ReleaseSource,
    load_release_sources,
    main,
    policy_sha256,
    resolve_all_releases,
    resolve_release,
)
from tests._dataset_fixtures import write_coco_dataset

REPO_ROOT = Path(__file__).resolve().parents[2]

QUALITY_YAML = """
min_images_per_class:
  threshold: 3
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


def _write_policy(tmp_path, text):
    path = tmp_path / "quality.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _policy(tmp_path):
    return load_quality_policy(_write_policy(tmp_path, QUALITY_YAML))


def _check(name, *, passed, action):
    return {
        "check_name": name,
        "passed": passed,
        "metric_value": 1.0,
        "details": {},
        "action": action,
    }


def _write_dvc(dataset_dir, name, md5, nfiles):
    doc = {"outs": [{"md5": md5, "size": 1, "nfiles": nfiles, "hash": "md5", "path": name}]}
    (dataset_dir / f"{name}.dvc").write_text(yaml.safe_dump(doc), encoding="utf-8")


def _add_release(
    tmp_path,
    version,
    *,
    cats=3,
    dogs=3,
    status="warning",
    checks=None,
    report_version=None,
    md5_seed=None,
):
    """Registra `version` en el catálogo y crea sus datos + .dvc bajo `data/<version>`."""
    reports_dir = tmp_path / "reports"
    release_dir = reports_dir / "releases" / version
    release_dir.mkdir(parents=True, exist_ok=True)
    checks = checks or [_check("spatial_bias", passed=False, action="warn")]
    report = {
        "schema_version": "1.0",
        "dataset_version": report_version or version,
        "status": status,
        "checks": checks,
    }
    (release_dir / "quality.json").write_text(json.dumps(report), encoding="utf-8")
    (release_dir / "splits.json").write_text("{}", encoding="utf-8")

    catalog_path = reports_dir / "versions.json"
    catalog = (
        json.loads(catalog_path.read_text(encoding="utf-8"))
        if catalog_path.exists()
        else {"schema_version": "1.0", "releases": []}
    )
    catalog["releases"].append(
        {
            "dataset_version": version,
            "quality_file": f"releases/{version}/quality.json",
            "splits_file": f"releases/{version}/splits.json",
        }
    )
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")

    dataset_dir = write_coco_dataset(tmp_path / "data" / version, cats=cats, dogs=dogs)
    seed = md5_seed or version
    annotations_md5 = f"{hashlib.md5(f'ann-{seed}'.encode()).hexdigest()}.dir"
    images_md5 = f"{hashlib.md5(f'img-{seed}'.encode()).hexdigest()}.dir"
    _write_dvc(dataset_dir, "annotations", annotations_md5, 1)
    _write_dvc(dataset_dir, "images", images_md5, cats + dogs)
    return ReleaseSource(
        dataset_dir=f"data/{version}",
        annotations_md5=annotations_md5,
        images_md5=images_md5,
    )


def _resolve(tmp_path, version, sources):
    return resolve_release(
        version,
        repo_root=tmp_path,
        reports_dir=tmp_path / "reports",
        sources=sources,
        policy=_policy(tmp_path),
    )


def _rejected(tmp_path, version, sources):
    with pytest.raises(ReleaseRejectedError) as excinfo:
        _resolve(tmp_path, version, sources)
    return excinfo.value.reason


def test_resolves_allowed_release_with_hashes_policy_and_counts(tmp_path):
    source = _add_release(tmp_path, "v0.1.1", cats=4, dogs=3)

    release = _resolve(tmp_path, "v0.1.1", {"v0.1.1": source})

    assert release.dataset_version == "v0.1.1"
    assert release.status == "warning"
    assert release.dataset_dir == "data/v0.1.1"
    assert release.annotations_dir == "data/v0.1.1/annotations"
    assert release.images_dir == "data/v0.1.1/images"
    assert release.annotations_md5 == source.annotations_md5
    assert release.images_md5 == source.images_md5
    assert release.images_nfiles == 7
    assert release.originals_per_class == {"cat": 4, "dog": 3}
    assert release.quality_file == "releases/v0.1.1/quality.json"
    quality_bytes = (tmp_path / "reports" / "releases" / "v0.1.1" / "quality.json").read_bytes()
    assert release.quality_sha256 == hashlib.sha256(quality_bytes).hexdigest()
    assert release.policy_sha256 == policy_sha256(_policy(tmp_path))
    assert release.min_images_per_class == 3
    assert release.min_classes == 2


def test_policy_hash_ignores_comments_but_changes_with_any_threshold(tmp_path):
    base = _policy(tmp_path)
    commented = load_quality_policy(_write_policy(tmp_path, "# otro comentario\n" + QUALITY_YAML))
    stricter = load_quality_policy(
        _write_policy(tmp_path, QUALITY_YAML.replace("threshold: 3", "threshold: 4"))
    )

    assert policy_sha256(commented) == policy_sha256(base)
    assert policy_sha256(stricter) != policy_sha256(base)


def test_unknown_release_is_rejected(tmp_path):
    source = _add_release(tmp_path, "v0.1.1")

    assert _rejected(tmp_path, "v9.9.9", {"v0.1.1": source}) == "not_in_catalog"


@pytest.mark.parametrize("version", ["", "latest", "0.1.1", "v0.1", "../v0.1.1", "v0.1.1/../x"])
def test_malformed_version_is_rejected(tmp_path, version):
    source = _add_release(tmp_path, "v0.1.1")

    assert _rejected(tmp_path, version, {"v0.1.1": source}) == "invalid_version"


def test_release_in_catalog_but_not_allowed_is_rejected(tmp_path):
    source = _add_release(tmp_path, "v0.1.1")
    _add_release(tmp_path, "v0.2.0")

    assert _rejected(tmp_path, "v0.2.0", {"v0.1.1": source}) == "not_allowed"


def test_failed_release_is_rejected_even_if_allowed(tmp_path):
    source = _add_release(
        tmp_path,
        "v0.1.0",
        status="failed",
        checks=[_check("min_images_per_class", passed=False, action="fail")],
    )

    assert _rejected(tmp_path, "v0.1.0", {"v0.1.0": source}) == "quality_failed"


def test_failed_status_is_rejected_on_its_own_without_a_failing_blocking_check(tmp_path):
    source = _add_release(
        tmp_path,
        "v0.1.0",
        status="failed",
        checks=[_check("spatial_bias", passed=False, action="warn")],
    )

    assert _rejected(tmp_path, "v0.1.0", {"v0.1.0": source}) == "quality_failed"


def test_warning_release_with_only_warn_actions_is_accepted(tmp_path):
    source = _add_release(
        tmp_path,
        "v0.1.1",
        status="warning",
        checks=[
            _check("min_images_per_class", passed=True, action="fail"),
            _check("spatial_bias", passed=False, action="warn"),
        ],
    )

    assert _resolve(tmp_path, "v0.1.1", {"v0.1.1": source}).status == "warning"


def test_warning_status_hiding_a_failed_blocking_check_is_rejected(tmp_path):
    source = _add_release(
        tmp_path,
        "v0.1.1",
        status="warning",
        checks=[_check("degenerate_boxes", passed=False, action="fail")],
    )

    assert _rejected(tmp_path, "v0.1.1", {"v0.1.1": source}) == "quality_failed"


def test_quality_report_of_another_version_is_rejected(tmp_path):
    source = _add_release(tmp_path, "v0.1.1", report_version="v0.1.0")

    assert _rejected(tmp_path, "v0.1.1", {"v0.1.1": source}) == "quality_mismatch"


def test_dvc_identity_mismatch_is_rejected(tmp_path):
    source = _add_release(tmp_path, "v0.1.1")
    tampered = source.model_copy(update={"images_md5": f"{'0' * 32}.dir"})

    assert _rejected(tmp_path, "v0.1.1", {"v0.1.1": tampered}) == "identity_mismatch"


def test_annotations_identity_mismatch_is_rejected(tmp_path):
    source = _add_release(tmp_path, "v0.1.1")
    tampered = source.model_copy(update={"annotations_md5": f"{'f' * 32}.dir"})

    assert _rejected(tmp_path, "v0.1.1", {"v0.1.1": tampered}) == "identity_mismatch"


def test_missing_data_is_rejected(tmp_path):
    source = _add_release(tmp_path, "v0.1.1")
    for image in (tmp_path / "data" / "v0.1.1" / "images").iterdir():
        image.unlink()

    assert _rejected(tmp_path, "v0.1.1", {"v0.1.1": source}) == "data_missing"


def test_extra_files_that_break_dvc_nfiles_are_rejected(tmp_path):
    source = _add_release(tmp_path, "v0.1.1")
    (tmp_path / "data" / "v0.1.1" / "images" / "intruder.jpg").write_bytes(b"x")

    assert _rejected(tmp_path, "v0.1.1", {"v0.1.1": source}) == "data_missing"


def test_release_without_two_classes_at_minimum_is_rejected(tmp_path):
    source = _add_release(tmp_path, "v0.1.1", cats=5, dogs=2)

    assert _rejected(tmp_path, "v0.1.1", {"v0.1.1": source}) == "insufficient_classes"


def test_release_with_exactly_the_minimum_per_class_is_accepted(tmp_path):
    source = _add_release(tmp_path, "v0.1.1", cats=3, dogs=3)

    assert _resolve(tmp_path, "v0.1.1", {"v0.1.1": source}).originals_per_class == {
        "cat": 3,
        "dog": 3,
    }


def test_changing_version_changes_the_resolved_source(tmp_path):
    source_a = _add_release(tmp_path, "v0.1.1", cats=4, dogs=3)
    source_b = _add_release(tmp_path, "v0.2.0", cats=6, dogs=5)
    sources = {"v0.1.1": source_a, "v0.2.0": source_b}

    release_a = _resolve(tmp_path, "v0.1.1", sources)
    release_b = _resolve(tmp_path, "v0.2.0", sources)

    assert release_a.dataset_version != release_b.dataset_version
    assert release_a.dataset_dir != release_b.dataset_dir
    assert release_a.annotations_md5 != release_b.annotations_md5
    assert release_a.images_md5 != release_b.images_md5
    assert release_a.quality_sha256 != release_b.quality_sha256 or (
        release_a.quality_file != release_b.quality_file
    )
    assert release_a.originals_per_class == {"cat": 4, "dog": 3}
    assert release_b.originals_per_class == {"cat": 6, "dog": 5}


def _resolve_all(tmp_path, sources):
    return resolve_all_releases(
        repo_root=tmp_path,
        reports_dir=tmp_path / "reports",
        sources=sources,
        policy=_policy(tmp_path),
    )


def test_resolve_all_lists_approved_releases_sorted_by_semver(tmp_path):
    source_new = _add_release(tmp_path, "v0.10.0", cats=6, dogs=5)
    source_old = _add_release(tmp_path, "v0.2.0", cats=4, dogs=3)

    listing = _resolve_all(tmp_path, {"v0.10.0": source_new, "v0.2.0": source_old})

    assert [r.dataset_version for r in listing.approved] == ["v0.2.0", "v0.10.0"]
    assert listing.rejected == []
    assert listing.approved[0].originals_per_class == {"cat": 4, "dog": 3}
    assert listing.approved[1].originals_per_class == {"cat": 6, "dog": 5}


def test_resolve_all_reports_rejected_releases_with_reason_instead_of_hiding_them(tmp_path):
    good = _add_release(tmp_path, "v0.1.1")
    failed = _add_release(
        tmp_path,
        "v0.1.0",
        status="failed",
        checks=[_check("min_images_per_class", passed=False, action="fail")],
    )
    _add_release(tmp_path, "v0.3.0")  # en el catálogo pero fuera de la allowlist

    listing = _resolve_all(tmp_path, {"v0.1.1": good, "v0.1.0": failed})

    assert [r.dataset_version for r in listing.approved] == ["v0.1.1"]
    assert {r.dataset_version: r.reason for r in listing.rejected} == {
        "v0.1.0": "quality_failed",
        "v0.3.0": "not_allowed",
    }
    assert [r.dataset_version for r in listing.rejected] == ["v0.1.0", "v0.3.0"]
    assert all(r.detail for r in listing.rejected)


def test_resolve_all_flags_allowed_version_missing_from_catalog(tmp_path):
    source = _add_release(tmp_path, "v0.1.1")

    listing = _resolve_all(tmp_path, {"v0.1.1": source, "v0.9.0": source})

    assert [r.dataset_version for r in listing.approved] == ["v0.1.1"]
    assert {r.dataset_version: r.reason for r in listing.rejected} == {"v0.9.0": "not_in_catalog"}


def test_resolve_all_without_catalog_or_sources_is_empty_not_an_error(tmp_path):
    listing = _resolve_all(tmp_path, {})

    assert listing.approved == []
    assert listing.rejected == []


def test_cli_all_prints_listing_json_and_exits_zero_even_with_rejections(
    tmp_path, monkeypatch, capsys
):
    from presentation import release_resolver

    good = _add_release(tmp_path, "v0.1.1", cats=4, dogs=3)
    _add_release(tmp_path, "v0.3.0")  # rechazada: fuera de la allowlist
    monkeypatch.setattr(release_resolver, "APP_ROOT", tmp_path / "app")
    monkeypatch.setattr(release_resolver, "load_release_sources", lambda: {"v0.1.1": good})
    monkeypatch.setattr(release_resolver, "load_quality_policy", lambda: _policy(tmp_path))

    assert main(["--all"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert [r["dataset_version"] for r in payload["approved"]] == ["v0.1.1"]
    assert payload["approved"][0]["originals_per_class"] == {"cat": 4, "dog": 3}
    assert [(r["dataset_version"], r["reason"]) for r in payload["rejected"]] == [
        ("v0.3.0", "not_allowed")
    ]


def test_cli_requires_exactly_one_of_version_or_all(capsys):
    for argv in ([], ["v0.1.1", "--all"]):
        with pytest.raises(SystemExit) as excinfo:
            main(argv)
        assert excinfo.value.code == 2
    capsys.readouterr()


def test_registry_rejects_unsafe_dataset_dir(tmp_path):
    for bad in ("../outside", "/abs/path", "data/../../x", ""):
        with pytest.raises(ValueError, match="dataset_dir"):
            ReleaseSource(
                dataset_dir=bad, annotations_md5="a" * 32 + ".dir", images_md5="b" * 32 + ".dir"
            )


def test_registry_rejects_malformed_md5(tmp_path):
    with pytest.raises(ValueError, match="md5"):
        ReleaseSource(
            dataset_dir="data/raw", annotations_md5="nothex.dir", images_md5="b" * 32 + ".dir"
        )


def test_load_release_sources_reads_yaml(tmp_path):
    path = tmp_path / "sources.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "allowed_releases": {
                    "v0.1.1": {
                        "dataset_dir": "data/raw",
                        "annotations_md5": "a" * 32 + ".dir",
                        "images_md5": "b" * 32 + ".dir",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    sources = load_release_sources(path)

    assert set(sources) == {"v0.1.1"}
    assert sources["v0.1.1"].dataset_dir == "data/raw"


# --- contra los artefactos versionados en el repo (no requieren `dvc pull`) -------------


def test_committed_sources_pin_exactly_v0_1_1_and_match_committed_dvc_files():
    sources = load_release_sources()

    assert set(sources) == {"v0.1.1"}
    source = sources["v0.1.1"]
    for name, pinned in (("annotations", source.annotations_md5), ("images", source.images_md5)):
        dvc_file = REPO_ROOT / source.dataset_dir / f"{name}.dvc"
        declared = yaml.safe_load(dvc_file.read_text(encoding="utf-8"))["outs"][0]["md5"]
        assert declared == pinned, f"{dvc_file} ya no coincide con el md5 fijado para v0.1.1"


def test_committed_catalog_v0_1_0_failed_is_not_eligible_and_v0_1_1_warning_is_reported():
    reports_dir = REPO_ROOT / "reports"
    sources = load_release_sources()

    with pytest.raises(ReleaseRejectedError) as excinfo:
        resolve_release(
            "v0.1.0",
            repo_root=REPO_ROOT,
            reports_dir=reports_dir,
            sources=sources,
            policy=load_quality_policy(),
        )
    assert excinfo.value.reason == "not_allowed"

    v010 = json.loads((reports_dir / "releases" / "v0.1.0" / "quality.json").read_text("utf-8"))
    v011 = json.loads((reports_dir / "releases" / "v0.1.1" / "quality.json").read_text("utf-8"))
    assert v010["status"] == "failed"
    assert v011["status"] == "warning"


# --- contra los datos reales recuperados de DVC/S3 ---------------------------------------

real_data = pytest.mark.skipif(
    not (REPO_ROOT / "data" / "raw" / "images").is_dir(),
    reason="datos reales no recuperados (dvc pull -r prod)",
)


@real_data
def test_real_all_lists_v0_1_1_approved_and_v0_1_0_rejected(capsys):
    assert main(["--all"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert [r["dataset_version"] for r in payload["approved"]] == ["v0.1.1"]
    assert payload["approved"][0]["originals_per_class"] == {"cat": 301, "dog": 300}
    assert {r["dataset_version"]: r["reason"] for r in payload["rejected"]} == {
        "v0.1.0": "not_allowed"
    }


@real_data
def test_real_v0_1_1_resolves_with_expected_counts_and_hashes():
    release = resolve_release(
        "v0.1.1",
        repo_root=REPO_ROOT,
        reports_dir=REPO_ROOT / "reports",
        sources=load_release_sources(),
        policy=load_quality_policy(),
    )

    assert release.status == "warning"
    assert release.originals_per_class == {"cat": 301, "dog": 300}
    assert release.images_nfiles == 600
    assert release.annotations_nfiles == 10
    assert release.images_md5 == "951150dd4fb053f4665089fcb37a1c87.dir"
    assert release.annotations_md5 == "c7cb86ae7ece94ef7b853620e464a4d7.dir"
