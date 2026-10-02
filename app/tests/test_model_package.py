"""D05-01 — Paquete smoke y loader autocontenido.

Pesos de FIXTURE: el checkpoint de estos tests sale de `build_model` (D01-04) con una seed,
sin entrenar. Es una prueba de componente del formato y del loader; no acredita el
roundtrip del checkpoint real de D03-04, que se evidencia aparte con el run real.

Cada negativa debe fallar con `PackageError` ANTES de servir una predicción, y el loader
nunca carga otro artefacto por fallback.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from model_package import (
    FORMAT_VERSION,
    MANIFEST_FILE,
    PAYLOAD_FILES,
    PackageError,
    build_smoke_package,
    load_package,
    reference_image,
)
from training.class_map import CLASS_MAP
from training.config import TrainingConfig
from training.model import build_model
from training.preprocessing import build_eval_transform

APP_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "0123456789abcdef0123456789abcdef"
CONFIG = TrainingConfig(seed=7, trainable_layers="last_block")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_checkpoint(folder: Path, config: TrainingConfig, *, seed: int) -> Path:
    """Misma forma que sube el trainer de D03-04 a `checkpoint/` en MLflow."""
    folder.mkdir(parents=True)
    model = build_model(config.model_copy(update={"pretrained": False, "seed": seed}))
    # Como un checkpoint entrenado: la cabeza ya no es la inicialización de build_model,
    # así que solo reproduce la salida quien cargue ESTOS pesos.
    generator = torch.Generator().manual_seed(seed + 100)
    with torch.no_grad():
        for parameter in model.fc.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.5)
    torch.save(model.state_dict(), folder / "model.pt")
    sha = _sha256(folder / "model.pt")
    files = {
        "training_config.json": config.model_dump(),
        "class_map.json": dict(CLASS_MAP),
        "environment.json": {"python": "3.12.3", "torch": torch.__version__},
        "sources.json": {
            "dataset_version": "v0.1.1",
            "manifest_version": "p3-v0.1.1-s42",
            "manifest_hash": "d" * 64,
            "dvc_release_hash": "e" * 64,
            "best_epoch": 3,
            "checkpoint_sha256": sha,
        },
    }
    for name, content in files.items():
        (folder / name).write_text(json.dumps(content, indent=2), encoding="utf-8")
    return folder


@pytest.fixture(scope="session")
def checkpoint(tmp_path_factory) -> Path:
    return _write_checkpoint(tmp_path_factory.mktemp("ckpt") / "checkpoint", CONFIG, seed=7)


@pytest.fixture(scope="session")
def built(checkpoint, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("pkg") / "package"
    build_smoke_package(
        checkpoint,
        out,
        run_id=RUN_ID,
        experiment="p3-cnn-classifier",
        expected_sha256=_sha256(checkpoint / "model.pt"),
        created_at="2026-10-02T06:00:00Z",
    )
    return out


@pytest.fixture
def package(built, tmp_path) -> Path:
    """Copia propia para que cada negativa pueda alterarla."""
    return Path(shutil.copytree(built, tmp_path / "package"))


def _manifest(package: Path) -> dict:
    return json.loads((package / MANIFEST_FILE).read_text(encoding="utf-8"))


def _rewrite(package: Path, name: str, content: dict | bytes) -> None:
    """Cambia un archivo y actualiza su entrada del inventario, para que la
    comprobación que se prueba sea la de contenido y no la del hash."""
    path = package / name
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(json.dumps(content, indent=2), encoding="utf-8")
    manifest = _manifest(package)
    manifest["files"][name] = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
    (package / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _json(package: Path, name: str) -> dict:
    return json.loads((package / name).read_text(encoding="utf-8"))


@pytest.fixture
def no_model_built(monkeypatch):
    """Falla si el loader llega a instanciar la CNN: la negativa debe cortar antes."""

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("el loader instanció el modelo antes de rechazar el paquete")

    monkeypatch.setattr("model_package.loader.build_model", _forbidden)


# --- Formato -----------------------------------------------------------------------


def test_build_writes_versioned_manifest_with_full_inventory(built, checkpoint):
    manifest = _manifest(built)
    assert manifest["format"] == "p3-model-package"
    assert manifest["format_version"] == FORMAT_VERSION == "1.0.0"
    assert manifest["kind"] == "smoke"
    assert set(manifest["files"]) == set(PAYLOAD_FILES)
    assert sorted(p.name for p in built.iterdir()) == sorted([MANIFEST_FILE, *PAYLOAD_FILES])
    for name, entry in manifest["files"].items():
        assert entry == {
            "sha256": _sha256(built / name),
            "size_bytes": (built / name).stat().st_size,
        }


def test_manifest_links_run_id_origin_checkpoint_and_sha(built, checkpoint):
    source = _manifest(built)["source"]
    assert source["mlflow_run_id"] == RUN_ID
    assert source["checkpoint_artifact"] == "checkpoint/model.pt"
    assert source["checkpoint_sha256"] == _sha256(checkpoint / "model.pt")
    assert source["checkpoint_sha256"] == _manifest(built)["files"]["model.pt"]["sha256"]
    assert source["best_epoch"] == 3
    assert source["manifest_hash"] == "d" * 64
    # El paquete describe el artefacto de origen: los pesos son esos bytes, sin rehacer.
    assert (built / "model.pt").read_bytes() == (checkpoint / "model.pt").read_bytes()


def test_format_version_is_not_the_model_semver(built):
    manifest = _manifest(built)
    assert "semver" not in manifest
    assert "D06-02" in manifest["note"]


def test_package_carries_architecture_class_map_preprocessing_and_dependencies(built):
    assert _json(built, "training_config.json") == CONFIG.model_dump()
    assert _json(built, "class_map.json") == {"cat": 0, "dog": 1}
    preprocessing = _json(built, "preprocessing.json")
    assert preprocessing["image_size"] == CONFIG.image_size
    assert preprocessing["mean"] == [0.485, 0.456, 0.406]
    assert preprocessing["std"] == [0.229, 0.224, 0.225]
    deps = _json(built, "dependencies.json")
    for name in ("python", "torch", "torchvision", "pillow"):
        assert deps["required"][name]
    assert len(deps["lockfile_sha256"]) == 64


def test_smoke_card_says_there_are_no_official_test_metrics(built):
    card = _json(built, "smoke_card.json")
    assert card["kind"] == "smoke"
    assert card["official_test_metrics"] is None
    assert "test" in card["statement"]
    assert "D06-02" in card["statement"]


# --- Roundtrip ---------------------------------------------------------------------


def test_load_exposes_readable_identity(built, checkpoint):
    loaded = load_package(built)
    assert loaded.identity() == {
        "format_version": "1.0.0",
        "kind": "smoke",
        "package_id": f"p3-cnn-classifier-smoke-{RUN_ID[:12]}",
        "mlflow_run_id": RUN_ID,
        "checkpoint_sha256": _sha256(checkpoint / "model.pt"),
        "best_epoch": 3,
        "classes": ["cat", "dog"],
        "image_size": 224,
        "official_test_metrics": None,
    }


def test_prediction_matches_the_output_recorded_from_the_origin_checkpoint(built):
    loaded = load_package(built)
    reference = _json(built, "reference_output.json")
    prediction = loaded.predict(reference_image())
    assert prediction["predicted_class"] == reference["predicted_class"]
    for name, value in reference["probabilities"].items():
        assert prediction["probabilities"][name] == pytest.approx(value, abs=1e-6)
    assert sorted(prediction["probabilities"]) == ["cat", "dog"]
    assert sum(prediction["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert loaded.check_reference() == {"matches": True, "max_abs_diff": pytest.approx(0, abs=1e-6)}


def test_roundtrip_equals_the_in_memory_model_of_the_checkpoint(built, checkpoint):
    """Otra entrada conocida: el modelo del paquete da lo mismo que la CNN de D01-04
    con el state_dict del checkpoint cargado directamente."""
    image = Image.new("RGB", (300, 260), (30, 160, 90))
    model = build_model(CONFIG.model_copy(update={"pretrained": False}))
    model.load_state_dict(torch.load(checkpoint / "model.pt", weights_only=True), strict=True)
    model.eval()
    with torch.no_grad():
        expected = torch.softmax(model(build_eval_transform(CONFIG)(image).unsqueeze(0)), 1)[0]
    prediction = load_package(built).predict(image)
    assert prediction["probabilities"]["cat"] == pytest.approx(float(expected[0]), abs=1e-6)
    assert prediction["probabilities"]["dog"] == pytest.approx(float(expected[1]), abs=1e-6)


def test_prediction_is_deterministic(built):
    loaded = load_package(built)
    assert loaded.predict(reference_image()) == loaded.predict(reference_image())


def test_clean_process_loads_the_package_and_reproduces_the_output(built):
    """Proceso nuevo, sin estado del trainer ni del test: solo el directorio del paquete."""
    result = subprocess.run(
        [sys.executable, "-m", "model_package", "predict", "--package", str(built)],
        cwd=APP_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["identity"]["mlflow_run_id"] == RUN_ID
    assert report["reference"]["matches"] is True
    reference = _json(built, "reference_output.json")
    assert report["prediction"]["predicted_class"] == reference["predicted_class"]


# --- Negativas: fallan antes de servir una predicción --------------------------------


def test_truncated_weights_are_rejected(package, no_model_built):
    data = (package / "model.pt").read_bytes()
    (package / "model.pt").write_bytes(data[: len(data) // 2])
    with pytest.raises(PackageError, match=r"model\.pt.*truncado"):
        load_package(package)


def test_weights_with_a_different_hash_are_rejected(package, no_model_built):
    data = bytearray((package / "model.pt").read_bytes())
    data[-100] ^= 0xFF
    (package / "model.pt").write_bytes(bytes(data))
    with pytest.raises(PackageError, match=r"model\.pt.*hash"):
        load_package(package)


def test_weights_that_are_not_the_origin_checkpoint_are_rejected(package, tmp_path, no_model_built):
    other = _write_checkpoint(tmp_path / "other", CONFIG, seed=8)
    _rewrite(package, "model.pt", (other / "model.pt").read_bytes())
    with pytest.raises(PackageError, match="checkpoint de origen"):
        load_package(package)


def test_incompatible_architecture_config_is_rejected(package, no_model_built):
    config = _json(package, "training_config.json")
    config["architecture"] = "resnet50"
    _rewrite(package, "training_config.json", config)
    with pytest.raises(PackageError, match="training_config"):
        load_package(package)


def test_config_that_does_not_match_the_weights_is_rejected_without_predicting(package):
    config = _json(package, "training_config.json")
    config["hidden_layers"] = 1
    _rewrite(package, "training_config.json", config)
    with pytest.raises(PackageError, match="no corresponden a la arquitectura"):
        load_package(package)


def test_incompatible_class_map_is_rejected(package, no_model_built):
    _rewrite(package, "class_map.json", {"cat": 1, "dog": 0})
    with pytest.raises(PackageError, match="class_map"):
        load_package(package)


def test_incompatible_preprocessing_is_rejected(package, no_model_built):
    preprocessing = _json(package, "preprocessing.json")
    preprocessing["image_size"] = 128
    _rewrite(package, "preprocessing.json", preprocessing)
    with pytest.raises(PackageError, match="preprocessing"):
        load_package(package)


def test_missing_installed_dependency_is_rejected(package, monkeypatch, no_model_built):
    real = __import__("model_package.loader", fromlist=["installed_version"]).installed_version
    monkeypatch.setattr(
        "model_package.loader.installed_version",
        lambda name: None if name == "torchvision" else real(name),
    )
    with pytest.raises(PackageError, match="dependencia ausente: torchvision"):
        load_package(package)


def test_dependency_missing_from_the_package_is_rejected(package, no_model_built):
    deps = _json(package, "dependencies.json")
    del deps["required"]["pillow"]
    _rewrite(package, "dependencies.json", deps)
    with pytest.raises(PackageError, match="pillow"):
        load_package(package)


def test_incompatible_dependency_version_is_rejected(package, no_model_built):
    deps = _json(package, "dependencies.json")
    deps["required"]["torch"] = "1.13.1"
    _rewrite(package, "dependencies.json", deps)
    with pytest.raises(PackageError, match=r"torch.*incompatible"):
        load_package(package)


def test_incompatible_minor_dependency_version_is_rejected(package, no_model_built):
    deps = _json(package, "dependencies.json")
    major, minor = deps["required"]["torch"].split(".")[:2]
    deps["required"]["torch"] = f"{major}.{int(minor) + 1}.0"
    _rewrite(package, "dependencies.json", deps)
    with pytest.raises(PackageError, match=r"torch.*incompatible"):
        load_package(package)


def test_manifest_inventory_must_be_exactly_the_format_files(package, no_model_built):
    manifest = _manifest(package)
    manifest["files"]["notas.txt"] = manifest["files"].pop("smoke_card.json")
    (package / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PackageError, match=r"inventario.*notas\.txt.*smoke_card\.json"):
        load_package(package)


def test_output_different_from_the_reference_is_reported_and_fails_the_cli(package):
    reference = _json(package, "reference_output.json")
    reference["probabilities"] = dict.fromkeys(reference["probabilities"], 0.5)
    _rewrite(package, "reference_output.json", reference)
    check = load_package(package).check_reference()
    assert check["matches"] is False
    assert check["max_abs_diff"] > 1e-3
    result = subprocess.run(
        [sys.executable, "-m", "model_package", "predict", "--package", str(package)],
        cwd=APP_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert json.loads(result.stdout)["reference"]["matches"] is False


def test_missing_file_is_rejected_without_fallback(package, no_model_built):
    (package / "preprocessing.json").unlink()
    with pytest.raises(PackageError, match=r"falta preprocessing\.json"):
        load_package(package)


def test_file_outside_the_inventory_is_rejected(package, no_model_built):
    (package / "model_v2.pt").write_bytes(b"x")
    with pytest.raises(PackageError, match=r"model_v2\.pt.*inventario"):
        load_package(package)


def test_unsupported_format_version_is_rejected(package, no_model_built):
    manifest = _manifest(package)
    manifest["format_version"] = "2.0.0"
    (package / MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(PackageError, match=r"format_version 2\.0\.0"):
        load_package(package)


def test_missing_manifest_is_rejected(package, no_model_built):
    (package / MANIFEST_FILE).unlink()
    with pytest.raises(PackageError, match=MANIFEST_FILE):
        load_package(package)


# --- Builder ------------------------------------------------------------------------


def test_build_refuses_a_checkpoint_whose_sha_is_not_the_recorded_one(checkpoint, tmp_path):
    with pytest.raises(PackageError, match="checkpoint_sha256"):
        build_smoke_package(
            checkpoint,
            tmp_path / "out",
            run_id=RUN_ID,
            experiment="p3-cnn-classifier",
            expected_sha256="0" * 64,
        )
    assert not (tmp_path / "out").exists()


def test_build_refuses_sources_that_do_not_match_the_weights(checkpoint, tmp_path):
    copy = Path(shutil.copytree(checkpoint, tmp_path / "ckpt"))
    sources = json.loads((copy / "sources.json").read_text(encoding="utf-8"))
    sources["checkpoint_sha256"] = "f" * 64
    (copy / "sources.json").write_text(json.dumps(sources), encoding="utf-8")
    with pytest.raises(PackageError, match=r"sources\.json"):
        build_smoke_package(copy, tmp_path / "out", run_id=RUN_ID, experiment="p3-cnn-classifier")
    assert not (tmp_path / "out").exists()


def test_build_never_overwrites_an_existing_package(checkpoint, built):
    with pytest.raises(PackageError, match="ya existe"):
        build_smoke_package(checkpoint, built, run_id=RUN_ID, experiment="p3-cnn-classifier")


def test_build_refuses_another_experiment(checkpoint, tmp_path):
    with pytest.raises(PackageError, match="experimento 'otro', no de p3-cnn-classifier"):
        build_smoke_package(checkpoint, tmp_path / "out", run_id=RUN_ID, experiment="otro")
    assert not (tmp_path / "out").exists()
