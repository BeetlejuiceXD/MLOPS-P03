"""D05-04 — Motor de inferencia sobre el paquete de D05-01.

Pesos de FIXTURE (los mismos de `test_model_package.py`): prueba de componente del motor y
de su contrato. El paquete smoke real se acredita aparte con su run_id y SHA; estos tests
no cierran el ticket.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest
import torch
from PIL import Image

from inference_engine import (
    InferenceEngine,
    InferenceFailed,
    InputRejected,
    PackageRejected,
    decode_image,
)
from inference_engine.server import make_server
from model_package import build_smoke_package, load_package
from tests.test_model_package import CONFIG, _rewrite, _sha256, _write_checkpoint
from training.preprocessing import build_eval_transform

APP_ROOT = Path(__file__).resolve().parents[1]
RUN_A = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
RUN_B = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"


def _build(tmp_path_factory, run_id: str, seed: int) -> Path:
    checkpoint = _write_checkpoint(
        tmp_path_factory.mktemp(f"ckpt-{seed}") / "checkpoint", CONFIG, seed=seed
    )
    out = tmp_path_factory.mktemp(f"pkg-{seed}") / "package"
    build_smoke_package(
        checkpoint,
        out,
        run_id=run_id,
        experiment="p3-cnn-classifier",
        expected_sha256=_sha256(checkpoint / "model.pt"),
        created_at="2026-10-02T06:00:00Z",
    )
    return out


@pytest.fixture(scope="session")
def package_a(tmp_path_factory) -> Path:
    return _build(tmp_path_factory, RUN_A, seed=7)


@pytest.fixture(scope="session")
def package_b(tmp_path_factory) -> Path:
    return _build(tmp_path_factory, RUN_B, seed=11)


@pytest.fixture(scope="session")
def engine(package_a) -> InferenceEngine:
    return InferenceEngine.from_package(package_a)


@pytest.fixture
def package_copy(package_a, tmp_path) -> Path:
    return Path(shutil.copytree(package_a, tmp_path / "package"))


def _image_bytes(fmt: str = "PNG", size=(320, 200), mode="RGB") -> bytes:
    width, height = size
    pixels = bytes(
        channel
        for y in range(height)
        for x in range(width)
        for channel in ((x * 5) % 256, (y * 3) % 256, (x + y) % 256)
    )
    buffer = io.BytesIO()
    Image.frombytes("RGB", size, pixels).convert(mode).save(buffer, format=fmt)
    return buffer.getvalue()


def _sha_of(package: Path) -> str:
    return json.loads((package / "package.json").read_text(encoding="utf-8"))["source"][
        "checkpoint_sha256"
    ]


# --- Positivos ---------------------------------------------------------------------


def test_identity_has_the_d05_07_engine_shape(engine, package_a):
    assert engine.identity() == {
        "model": {
            "source": "smoke",
            "package_id": f"p3-cnn-classifier-smoke-{RUN_A[:12]}",
            "format_version": "1.0.0",
            "model_version": None,
            "mlflow_run_id": RUN_A,
            "checkpoint_sha256": _sha_of(package_a),
            # D06-06: un paquete local (smoke) nunca trae objeto S3; official lo exige.
            "s3_object": None,
        },
        "classes": ["cat", "dog"],
        "image_size": CONFIG.image_size,
    }


def test_prediction_maps_probabilities_to_the_class_map_and_sums_one(engine):
    prediction = engine.predict(_image_bytes())
    assert list(prediction.probabilities) == ["cat", "dog"]
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    assert prediction.predicted_class == max(
        prediction.probabilities, key=prediction.probabilities.__getitem__
    )
    assert prediction.model == engine.identity()["model"]
    assert set(prediction.contract()) == {"predicted_class", "probabilities", "model"}


def test_prediction_uses_the_package_transform_and_weights(engine, package_a):
    """Mismo resultado que la CNN del paquete con el transform de evaluación de su config,
    calculado a mano: el motor no reconstruye el preprocessing desde defaults."""
    data = _image_bytes("JPEG")
    loaded = load_package(package_a)
    image = Image.open(io.BytesIO(data)).convert("RGB")
    with torch.no_grad():
        expected = torch.softmax(
            loaded.model(build_eval_transform(loaded.config)(image).unsqueeze(0)), dim=1
        )[0].tolist()
    got = engine.predict(data).probabilities
    assert [got["cat"], got["dog"]] == pytest.approx(expected, abs=1e-6)


def test_same_input_and_reloaded_package_give_the_same_output(engine, package_a):
    data = _image_bytes("WEBP")
    first = engine.predict(data)
    again = engine.predict(data)
    reloaded = InferenceEngine.from_package(package_a).predict(data)
    assert first.contract() == again.contract() == reloaded.contract()
    assert first.input_sha256 == reloaded.input_sha256


def test_accepts_jpeg_png_webp_and_non_rgb_modes(engine):
    for data in (
        _image_bytes("JPEG"),
        _image_bytes("PNG"),
        _image_bytes("WEBP"),
        _image_bytes("PNG", mode="L"),
    ):
        prediction = engine.predict(data)
        assert prediction.input_format in {"JPEG", "PNG", "WEBP"}


def test_clean_process_cli_reproduces_the_in_process_output(engine, package_a, tmp_path):
    image = tmp_path / "input.png"
    image.write_bytes(_image_bytes())
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "inference_engine",
            "predict",
            "--package",
            str(package_a),
            "--image",
            str(image),
            "--expected-sha256",
            _sha_of(package_a),
        ],
        cwd=APP_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert out["engine"] == engine.identity()
    [prediction] = out["predictions"]
    assert prediction["input_sha256"] == engine.predict(image.read_bytes()).input_sha256
    expected = engine.predict(image.read_bytes()).contract()
    assert prediction["predicted_class"] == expected["predicted_class"]
    assert prediction["probabilities"] == pytest.approx(expected["probabilities"], abs=1e-6)
    assert prediction["model"] == expected["model"]


# --- Negativos de entrada ----------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "match"),
    [
        (b"", "vacía"),
        (b"esto no es una imagen", "no decodificable"),
        (_image_bytes("PNG")[:200], "no decodificable"),
        (_image_bytes("GIF"), "formato GIF no admitido"),
    ],
    ids=["vacia", "texto", "png-truncado", "gif"],
)
def test_undecodable_inputs_are_rejected_without_prediction(engine, monkeypatch, data, match):
    def _forbidden(*_args, **_kwargs):
        raise AssertionError("el motor llegó a predecir con una entrada inválida")

    monkeypatch.setattr("model_package.loader.LoadedPackage.predict", _forbidden)
    with pytest.raises(InputRejected, match=match):
        engine.predict(data)


def test_oversized_input_is_rejected():
    with pytest.raises(InputRejected, match="máximo"):
        decode_image(b"\x89PNG" + b"0" * 64, max_bytes=16)


# --- Negativos de paquete ----------------------------------------------------------


def test_missing_package_is_rejected(tmp_path):
    with pytest.raises(PackageRejected, match="faltante"):
        InferenceEngine.from_package(tmp_path / "no-existe")


def test_wrong_expected_sha_is_rejected(package_a):
    with pytest.raises(PackageRejected, match="se esperaba"):
        InferenceEngine.from_package(package_a, expected_checkpoint_sha256="0" * 64)


def test_tampered_weights_are_rejected(package_copy):
    weights = package_copy / "model.pt"
    data = bytearray(weights.read_bytes())
    data[-1] ^= 0xFF
    weights.write_bytes(bytes(data))
    with pytest.raises(PackageRejected, match="hash discordante"):
        InferenceEngine.from_package(package_copy)


def test_class_map_missing_a_class_is_rejected(package_copy):
    _rewrite(package_copy, "class_map.json", {"cat": 0})
    with pytest.raises(PackageRejected, match="class_map"):
        InferenceEngine.from_package(package_copy)


def test_package_that_does_not_reproduce_its_reference_is_rejected(package_copy):
    reference = json.loads((package_copy / "reference_output.json").read_text(encoding="utf-8"))
    reference["probabilities"] = {"cat": 0.5, "dog": 0.5}
    _rewrite(package_copy, "reference_output.json", reference)
    with pytest.raises(PackageRejected, match="referencia"):
        InferenceEngine.from_package(package_copy)


def test_engine_without_package_does_not_predict():
    engine = InferenceEngine()
    with pytest.raises(PackageRejected, match="no tiene un paquete"):
        engine.predict(_image_bytes())
    with pytest.raises(PackageRejected):
        engine.identity()


# --- Negativos de inferencia -------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ({"predicted_class": "dog", "probabilities": {"dog": 1.0}}, "class_map"),
        ({"predicted_class": "dog", "probabilities": {"dog": 0.5, "cat": 0.5}}, "class_map"),
        ({"predicted_class": "dog", "probabilities": {"cat": 0.2, "dog": 0.2}}, "suman"),
        ({"predicted_class": "dog", "probabilities": {"cat": float("nan"), "dog": 1.0}}, "fuera"),
        ({"predicted_class": "cat", "probabilities": {"cat": 0.1, "dog": 0.9}}, "argmax"),
    ],
    ids=["clase-ausente", "orden", "no-suma-1", "nan", "no-argmax"],
)
def test_incoherent_model_output_is_not_emitted(package_a, monkeypatch, raw, match):
    engine = InferenceEngine.from_package(package_a)
    monkeypatch.setattr("model_package.loader.LoadedPackage.predict", lambda *_a: raw)
    with pytest.raises(InferenceFailed, match=match):
        engine.predict(_image_bytes())


def test_model_crash_is_an_inference_error(package_a, monkeypatch):
    engine = InferenceEngine.from_package(package_a)

    def _boom(*_args):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr("model_package.loader.LoadedPackage.predict", _boom)
    with pytest.raises(InferenceFailed, match="RuntimeError"):
        engine.predict(_image_bytes())


# --- Atribución y recarga (D06-04) -------------------------------------------------


def test_each_output_is_attributed_to_the_package_that_produced_it(package_a, package_b):
    data = _image_bytes()
    a = InferenceEngine.from_package(package_a).predict(data)
    b = InferenceEngine.from_package(package_b).predict(data)
    assert a.model["mlflow_run_id"] == RUN_A
    assert b.model["mlflow_run_id"] == RUN_B
    assert a.model["checkpoint_sha256"] == _sha_of(package_a)
    assert b.model["checkpoint_sha256"] == _sha_of(package_b)
    assert a.probabilities != b.probabilities


def test_same_instance_loads_another_package_with_the_same_logic(package_a, package_b):
    engine = InferenceEngine.from_package(package_a)
    data = _image_bytes()
    engine.load(package_b, expected_checkpoint_sha256=_sha_of(package_b))
    assert (
        engine.predict(data).contract()
        == InferenceEngine.from_package(package_b).predict(data).contract()
    )


def test_failed_reload_leaves_the_engine_without_model(package_a, tmp_path):
    engine = InferenceEngine.from_package(package_a)
    with pytest.raises(PackageRejected):
        engine.load(tmp_path / "no-existe")
    assert not engine.loaded
    with pytest.raises(PackageRejected):
        engine.predict(_image_bytes())


# --- Adaptador HTTP (contrato de D05-07) -------------------------------------------


@pytest.fixture
def served(request):
    servers = []

    def _serve(engine: InferenceEngine, **kwargs) -> str:
        server = make_server(engine, "127.0.0.1", 0, **kwargs)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return f"http://127.0.0.1:{server.server_address[1]}"

    yield _serve
    for server in servers:
        server.shutdown()
        server.server_close()


def _call(url: str, data: bytes | None = None, content_type: str = "image/png"):
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": content_type} if data is not None else {}
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())


def test_http_identity_and_predict_follow_the_contract(engine, served):
    url = served(engine)
    assert _call(f"{url}/identity") == (200, engine.identity())
    data = _image_bytes()
    status, body = _call(f"{url}/predict", data)
    assert status == 200
    assert body == json.loads(json.dumps(engine.predict(data).contract()))


@pytest.mark.parametrize(
    ("data", "content_type", "status"),
    [
        (b"no es imagen", "image/png", 400),
        (b"", "image/png", 400),
        (b"{}", "application/json", 415),
    ],
    ids=["no-decodificable", "vacia", "content-type"],
)
def test_http_rejects_bad_inputs_with_4xx_and_error(engine, served, data, content_type, status):
    got, body = _call(f"{served(engine)}/predict", data, content_type)
    assert got == status
    assert set(body) == {"error"}


def test_http_rejects_oversized_bodies(engine, served):
    status, body = _call(f"{served(engine, max_bytes=100)}/predict", _image_bytes())
    assert status == 413
    assert "máximo" in body["error"]


def test_http_without_package_is_503(served):
    url = served(InferenceEngine())
    assert _call(f"{url}/identity")[0] == 503
    status, body = _call(f"{url}/predict", _image_bytes())
    assert status == 503
    assert set(body) == {"error"}


def test_http_incoherent_output_is_5xx(package_a, served, monkeypatch):
    engine = InferenceEngine.from_package(package_a)
    monkeypatch.setattr(
        "model_package.loader.LoadedPackage.predict",
        lambda *_a: {"predicted_class": "dog", "probabilities": {"dog": 1.0}},
    )
    status, body = _call(f"{served(engine)}/predict", _image_bytes())
    assert status == 500
    assert "class_map" in body["error"]
