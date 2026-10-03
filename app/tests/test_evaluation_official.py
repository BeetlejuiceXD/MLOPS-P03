"""D06-01 — Primera y única evaluación oficial del frozen test (custodia: Ale).

Ensayo con datos SINTÉTICOS: un manifest congelado inventado (20 crops de test que no
existen en el release), imágenes generadas y un paquete con pesos de fixture. Ningún ID,
etiqueta ni métrica del frozen test oficial: la ejecución real se hace una sola vez,
después del acta de D05-08, con el comando documentado en `evaluation/README.md`.

La base es SQLite con las tablas de `evaluation.store` (las mismas columnas que MariaDB,
comprobado en `test_evaluation_producer.py`).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from PIL import Image
from sqlalchemy import create_engine, select

from evaluation import official, store
from evaluation.official import ClosingAct, TestCrop, audit_stored, run_evaluation
from evaluation.producer import EvaluationRefusedError
from evaluation.store import EvaluationStore, p3_evaluation, p3_model_selection
from model_package import build_smoke_package, load_package
from presentation.contracts import frozen_test_split_hash
from presentation.manifest_candidate import _dvc_release_hash, _manifest_hash
from tests.test_model_package import CONFIG, _sha256, _write_checkpoint

APP_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "a" * 32
OTHER_RUN = "b" * 32
OUTCOME_HASH = "1" * 64
CLOSED_AT = datetime(2026, 10, 2, 20, 0, 0, 456000, tzinfo=UTC)
EVALUATED_AT = datetime(2026, 10, 2, 21, 0, 0, tzinfo=UTC)

# Manifest sintético: los IDs de test NO son los del frozen test oficial.
TRAIN_IDS = (1, 2, 3)
VAL_IDS = (4, 5)
TEST_IDS = tuple(range(100, 120))
IMAGES_MD5 = "synthetic-images-md5"
ANNOTATIONS_MD5 = "synthetic-annotations-md5"
DVC_RELEASE_HASH = _dvc_release_hash(IMAGES_MD5, ANNOTATIONS_MD5)
MANIFEST_HASH = _manifest_hash(
    dataset_version="v0.1.1",
    seed=42,
    target_ratios={"train": 0.7, "val": 0.2, "test": 0.1},
    assignments={"train": TRAIN_IDS, "val": VAL_IDS, "test": TEST_IDS},
)
TEST_SPLIT_HASH = frozen_test_split_hash(TEST_IDS)


# --- Fixtures sintéticas ------------------------------------------------------------------


def _write_manifest(folder: Path, **overrides) -> Path:
    content = {
        "manifest_version": "p3-v0.1.1-s42",
        "manifest_hash": MANIFEST_HASH,
        "dataset_version": "v0.1.1",
        "dvc_release_hash": DVC_RELEASE_HASH,
        "images_md5": IMAGES_MD5,
        "annotations_md5": ANNOTATIONS_MD5,
        "seed": 42,
        "target_ratios": {"train": 0.7, "val": 0.2, "test": 0.1},
        "frozen": True,
        "test_split_hash": TEST_SPLIT_HASH,
        "assignments": {"train": TRAIN_IDS, "val": VAL_IDS, "test": TEST_IDS},
    }
    content.update(overrides)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "manifest.json"
    data = json.dumps(content, indent=2).encode("utf-8")
    path.write_bytes(data)
    (folder / "manifest.json.dvc").write_text(
        f"outs:\n- md5: {hashlib.md5(data).hexdigest()}\n  size: {len(data)}\n"
        "  hash: md5\n  path: manifest.json\n",
        encoding="utf-8",
    )
    return path


def _build_package(tmp_path_factory, run_id: str, seed: int) -> Path:
    checkpoint = _write_checkpoint(
        tmp_path_factory.mktemp(f"ckpt-{seed}") / "checkpoint", CONFIG, seed=seed
    )
    sources = json.loads((checkpoint / "sources.json").read_text(encoding="utf-8"))
    sources.update(manifest_hash=MANIFEST_HASH, dvc_release_hash=DVC_RELEASE_HASH)
    (checkpoint / "sources.json").write_text(json.dumps(sources, indent=2), encoding="utf-8")
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
def package(tmp_path_factory) -> Path:
    return _build_package(tmp_path_factory, CANDIDATE, seed=7)


@pytest.fixture(scope="session")
def package_other_run(tmp_path_factory) -> Path:
    return _build_package(tmp_path_factory, OTHER_RUN, seed=11)


def _checkpoint_sha(package: Path) -> str:
    return load_package(package).manifest.source.checkpoint_sha256


def _image(crop_id: int) -> Image.Image:
    """Imagen sintética determinista por crop (no es un crop del release)."""
    width, height = 40 + crop_id % 7, 30 + crop_id % 5
    pixels = bytes(
        (x * 9 + crop_id * 13) % 256 if c == 0 else (y * 7 + crop_id * c * 31) % 256
        for y in range(height)
        for x in range(width)
        for c in range(3)
    )
    return Image.frombytes("RGB", (width, height), pixels)


@pytest.fixture(scope="session")
def package_predictions(package) -> dict[int, str]:
    loaded = load_package(package)
    return {i: loaded.predict(_image(i))["predicted_class"] for i in TEST_IDS}


def _flip(name: str) -> str:
    return "dog" if name == "cat" else "cat"


def _crops(labels: dict[int, str], ids=TEST_IDS) -> list[TestCrop]:
    return [TestCrop(crop_id=i, true_class=labels[i], image=_image(i)) for i in ids]


def _labels_with_errors(package_predictions, errors: int) -> dict[int, str]:
    """Verdad sintética = predicción del paquete salvo en `errors` crops: así se controla
    exactamente cuántos aciertos hay (umbral 0.85 sin redondeo)."""
    wrong = set(TEST_IDS[:errors])
    return {
        i: _flip(predicted) if i in wrong else predicted
        for i, predicted in package_predictions.items()
    }


@pytest.fixture
def engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'evaluation.db'}")
    store.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(p3_model_selection.insert().values(id=1, status="open"))
    return engine


OUTCOME = {
    "reference": {
        "dataset_version": "v0.1.1",
        "manifest_hash": MANIFEST_HASH,
        "dvc_release_hash": DVC_RELEASE_HASH,
    },
    "candidate": {"run_id": CANDIDATE, "campaign_row": 3, "best_epoch": 3},
    "outcome_hash": OUTCOME_HASH,
}


def _set_selection(engine, status: str, outcome=OUTCOME, closed_at=CLOSED_AT) -> None:
    with engine.begin() as conn:
        conn.execute(
            p3_model_selection.update()
            .where(p3_model_selection.c.id == 1)
            .values(
                status=status,
                outcome=None if outcome is None else json.dumps(outcome),
                outcome_hash=None if outcome is None else outcome["outcome_hash"],
                closed_at=None if closed_at is None else closed_at.replace(tzinfo=None),
            )
        )


def _selection_row(engine):
    with engine.connect() as conn:
        return tuple(conn.execute(select(p3_model_selection)).one())


def _rows(engine) -> list[str]:
    with engine.connect() as conn:
        return [row.namespace for row in conn.execute(select(p3_evaluation)).all()]


@pytest.fixture
def act(package) -> ClosingAct:
    return ClosingAct(
        candidate_run_id=CANDIDATE,
        checkpoint_sha256=_checkpoint_sha(package),
        manifest_hash=MANIFEST_HASH,
        test_split_hash=TEST_SPLIT_HASH,
        outcome_hash=OUTCOME_HASH,
    )


class Loader:
    """Cargador de crops de test que registra si se llamó (el frozen test se abre solo
    después de que el preflight completo pasa)."""

    def __init__(self, crops):
        self.crops = crops
        self.calls = 0

    def __call__(self, manifest):
        self.calls += 1
        assert manifest.test_split_hash == TEST_SPLIT_HASH
        return list(self.crops)


class Forbidden:
    calls = 0

    def __call__(self, manifest):
        raise AssertionError("se intentó abrir la partición test antes de pasar el preflight")


@pytest.fixture
def predict_calls(monkeypatch) -> list[int]:
    """Cuenta las inferencias por imagen (una sola por crop)."""
    calls: list[int] = []
    original = official.LoadedPackage.predict

    def spy(self, image):
        calls.append(image.size)
        return original(self, image)

    monkeypatch.setattr(official.LoadedPackage, "predict", spy)
    return calls


def _run(engine, act, package, tmp_path, loader, namespace="synthetic", **overrides):
    kwargs = {
        "act": act,
        "package_dir": package,
        "manifest_path": overrides.pop("manifest_path", None)
        or _write_manifest(tmp_path / "data" / "p3"),
        "load_crops": loader,
        "out_dir": overrides.pop("out_dir", tmp_path / "evidence"),
        "namespace": namespace,
        "clock": lambda: EVALUATED_AT,
        "commit": "c" * 40,
    }
    kwargs.update(overrides)
    return run_evaluation(EvaluationStore(engine), **kwargs)


def _refused(reason: str, call) -> EvaluationRefusedError:
    with pytest.raises(EvaluationRefusedError) as caught:
        call()
    assert caught.value.reason == reason, caught.value
    return caught.value


# --- Preflight negativo: nada del test se abre --------------------------------------------


@pytest.mark.parametrize("status", ["open", "candidate"])
def test_selection_not_closed_refuses_before_package_manifest_or_test(
    engine, act, tmp_path, status
):
    _set_selection(engine, status, OUTCOME if status == "candidate" else None, None)
    _refused(
        "model_selection_open",
        lambda: _run(engine, act, tmp_path / "no-package", tmp_path, Forbidden()),
    )
    assert _rows(engine) == []
    assert not (tmp_path / "evidence").exists()


def test_closed_record_without_close_timestamp_is_refused(engine, act, tmp_path):
    _set_selection(engine, "closed", OUTCOME, None)
    _refused(
        "incomplete_selection",
        lambda: _run(engine, act, tmp_path / "no-package", tmp_path, Forbidden()),
    )


@pytest.mark.parametrize(
    "field, value",
    [
        ("candidate_run_id", OTHER_RUN),
        ("outcome_hash", "2" * 64),
        ("manifest_hash", "3" * 64),
    ],
)
def test_act_that_does_not_match_the_persisted_close_is_refused(
    engine, act, package, tmp_path, field, value
):
    _set_selection(engine, "closed")
    _refused(
        "act_mismatch",
        lambda: _run(engine, replace(act, **{field: value}), package, tmp_path, Forbidden()),
    )
    assert _rows(engine) == []


def test_previous_official_result_stops_for_audit_without_a_second_evaluation(
    engine, act, package, tmp_path
):
    _set_selection(engine, "closed")
    with engine.begin() as conn:
        conn.execute(
            p3_evaluation.insert().values(
                namespace="official",
                candidate_run_id=CANDIDATE,
                evaluated_at=EVALUATED_AT.replace(tzinfo=None),
                evaluation="{}",
                predictions="{}",
                created_at=EVALUATED_AT.replace(tzinfo=None),
            )
        )
    _refused(
        "official_already_recorded",
        lambda: _run(engine, act, package, tmp_path, Forbidden(), namespace="official"),
    )
    assert _rows(engine) == ["official"]
    assert not (tmp_path / "evidence").exists()


def test_package_with_another_checkpoint_is_refused(engine, act, package, tmp_path):
    _set_selection(engine, "closed")
    _refused(
        "package_rejected",
        lambda: _run(
            engine, replace(act, checkpoint_sha256="f" * 64), package, tmp_path, Forbidden()
        ),
    )


def test_missing_package_is_refused(engine, act, tmp_path):
    _set_selection(engine, "closed")
    _refused(
        "package_rejected",
        lambda: _run(engine, act, tmp_path / "no-package", tmp_path, Forbidden()),
    )


def test_package_of_a_run_other_than_the_closed_candidate_is_refused(
    engine, act, package_other_run, tmp_path
):
    _set_selection(engine, "closed")
    substitute = replace(act, checkpoint_sha256=_checkpoint_sha(package_other_run))
    _refused(
        "package_not_candidate",
        lambda: _run(engine, substitute, package_other_run, tmp_path, Forbidden()),
    )


def test_package_whose_best_epoch_is_not_the_closed_one_is_refused(engine, act, package, tmp_path):
    outcome = json.loads(json.dumps(OUTCOME))
    outcome["candidate"]["best_epoch"] = 5
    _set_selection(engine, "closed", outcome)
    _refused("package_not_candidate", lambda: _run(engine, act, package, tmp_path, Forbidden()))


def test_act_test_split_hash_different_from_the_manifest_is_refused(engine, act, package, tmp_path):
    _set_selection(engine, "closed")
    _refused(
        "test_split_hash_mismatch",
        lambda: _run(
            engine, replace(act, test_split_hash="4" * 64), package, tmp_path, Forbidden()
        ),
    )


def test_manifest_edited_after_dvc_versioning_is_refused(engine, act, package, tmp_path):
    _set_selection(engine, "closed")
    manifest_path = _write_manifest(tmp_path / "data" / "p3")
    manifest_path.write_bytes(manifest_path.read_bytes() + b"\n")
    _refused(
        "manifest_rejected",
        lambda: _run(engine, act, package, tmp_path, Forbidden(), manifest_path=manifest_path),
    )


def test_manifest_whose_test_ids_do_not_give_its_hash_is_refused(engine, act, package, tmp_path):
    _set_selection(engine, "closed")
    manifest_path = _write_manifest(
        tmp_path / "data" / "p3",
        assignments={"train": TRAIN_IDS, "val": VAL_IDS, "test": TEST_IDS[:-1]},
    )
    _refused(
        "manifest_rejected",
        lambda: _run(engine, act, package, tmp_path, Forbidden(), manifest_path=manifest_path),
    )


# --- Positivo único -----------------------------------------------------------------------


def test_single_run_covers_exactly_the_test_ids_and_is_recalculated_independently(
    engine, act, package, tmp_path, package_predictions, predict_calls
):
    _set_selection(engine, "closed")
    labels = _labels_with_errors(package_predictions, errors=2)
    loader = Loader(_crops(labels))
    result = _run(engine, act, package, tmp_path, loader)

    assert loader.calls == 1
    # Una inferencia por crop (más la salida de referencia que coteja el paquete al cargar).
    assert len(predict_calls) == len(TEST_IDS) + 1
    stored = EvaluationStore(engine).read("synthetic")
    predictions = stored["predictions"]["predictions"]
    assert [p["crop_id"] for p in predictions] == sorted(TEST_IDS)
    assert {p["crop_id"]: p["predicted_class"] for p in predictions} == package_predictions
    assert {p["crop_id"]: p["true_class"] for p in predictions} == labels
    evaluation = stored["evaluation"]
    assert sum(map(sum, evaluation["confusion_matrix"]["rows"])) == len(TEST_IDS)
    assert evaluation["selection"]["candidate_run_id"] == CANDIDATE
    assert evaluation["manifest_hash"] == MANIFEST_HASH
    assert stored["predictions"]["test_split_hash"] == TEST_SPLIT_HASH
    assert datetime.fromisoformat(evaluation["evaluated_at"]) > CLOSED_AT

    audit = result.audit
    assert audit["matches"] is True
    assert audit["correct"] == 18
    assert audit["n_test"] == 20
    assert audit["target"] == {"accuracy": 0.85, "correct": 18, "n_test": 20, "met": True}
    assert sorted(audit["error_crop_ids"]) == sorted(TEST_IDS[:2])
    assert audit == audit_stored(stored)


def test_evidence_keeps_attempt_preflight_trace_and_audit(
    engine, act, package, tmp_path, package_predictions
):
    _set_selection(engine, "closed")
    _run(
        engine, act, package, tmp_path, Loader(_crops(_labels_with_errors(package_predictions, 0)))
    )
    evidence = tmp_path / "evidence"
    attempt = json.loads((evidence / "attempt.json").read_text(encoding="utf-8"))
    assert attempt["commit"] == "c" * 40
    assert attempt["act"]["candidate_run_id"] == CANDIDATE
    assert attempt["namespace"] == "synthetic"
    preflight = json.loads((evidence / "preflight.json").read_text(encoding="utf-8"))
    assert preflight["selection"]["closed_at"] == "2026-10-02T20:00:00.456Z"
    assert preflight["package"]["checkpoint_sha256"] == act.checkpoint_sha256
    assert preflight["manifest"]["test_split_hash"] == TEST_SPLIT_HASH
    trace = [
        json.loads(line)
        for line in (evidence / "trace.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [line["crop_id"] for line in trace] == list(TEST_IDS)
    assert all(len(line["input_sha256"]) == 64 for line in trace)
    audit = json.loads((evidence / "audit.json").read_text(encoding="utf-8"))
    assert audit["matches"] is True


def test_official_namespace_is_written_once_and_a_second_run_is_refused(
    engine, act, package, tmp_path, package_predictions
):
    _set_selection(engine, "closed")
    crops = _crops(_labels_with_errors(package_predictions, 1))
    _run(engine, act, package, tmp_path, Loader(crops), namespace="official")
    stored = EvaluationStore(engine).read("official")
    _refused(
        "official_already_recorded",
        lambda: _run(
            engine,
            act,
            package,
            tmp_path,
            Forbidden(),
            namespace="official",
            out_dir=tmp_path / "otra-carpeta",
            clock=lambda: EVALUATED_AT + timedelta(hours=1),
        ),
    )
    assert EvaluationStore(engine).read("official") == stored


def test_an_interrupted_attempt_is_not_restarted_as_a_new_first_evaluation(
    engine, act, package, tmp_path, package_predictions
):
    _set_selection(engine, "closed")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "attempt.json").write_text('{"started_at": "antes"}', encoding="utf-8")
    _refused(
        "attempt_already_started",
        lambda: _run(engine, act, package, tmp_path, Forbidden(), out_dir=evidence),
    )
    assert json.loads((evidence / "attempt.json").read_text(encoding="utf-8")) == {
        "started_at": "antes"
    }


# --- Negativos de integridad: nada se predice ni se guarda --------------------------------


@pytest.mark.parametrize(
    "ids, reason",
    [
        (TEST_IDS[:-1], "crop_ids_not_test_split"),
        ((*TEST_IDS, 999), "crop_ids_not_test_split"),
        ((*TEST_IDS, TEST_IDS[0]), "duplicate_crop_id"),
    ],
    ids=["faltante", "sobrante", "duplicada"],
)
def test_crops_that_are_not_exactly_the_test_partition_are_refused_without_predicting(
    engine, act, package, tmp_path, package_predictions, predict_calls, ids, reason
):
    _set_selection(engine, "closed")
    labels = {**package_predictions, 999: "cat"}
    _refused(reason, lambda: _run(engine, act, package, tmp_path, Loader(_crops(labels, ids))))
    assert len(predict_calls) == 1  # solo la salida de referencia del paquete
    assert _rows(engine) == []
    # La traza del intento se conserva para auditarlo.
    assert (tmp_path / "evidence" / "attempt.json").exists()


@pytest.mark.parametrize("label", ["Cat", "bird", "0"])
def test_label_outside_the_class_map_is_refused(
    engine, act, package, tmp_path, package_predictions, label
):
    _set_selection(engine, "closed")
    labels = {**package_predictions, TEST_IDS[0]: label}
    _refused(
        "label_outside_class_map",
        lambda: _run(engine, act, package, tmp_path, Loader(_crops(labels))),
    )
    assert _rows(engine) == []


@pytest.mark.parametrize(
    "errors, met",
    [(4, False), (3, True)],
    ids=["16-de-20-no-cumple", "17-de-20-cumple-exacto"],
)
def test_target_uses_counts_without_rounding_and_never_reopens_the_selection(
    engine, act, package, tmp_path, package_predictions, errors, met
):
    _set_selection(engine, "closed")
    before = _selection_row(engine)
    result = _run(
        engine,
        act,
        package,
        tmp_path,
        Loader(_crops(_labels_with_errors(package_predictions, errors))),
    )
    assert result.audit["target"]["met"] is met
    assert result.audit["target"]["correct"] == 20 - errors
    # El candidato sigue cerrado: ni se reabre ni cambia, cumpla o no el objetivo.
    assert _selection_row(engine) == before


# --- Auditoría independiente --------------------------------------------------------------


def test_audit_detects_a_stored_result_that_does_not_match_its_predictions(
    engine, act, package, tmp_path, package_predictions
):
    _set_selection(engine, "closed")
    _run(
        engine, act, package, tmp_path, Loader(_crops(_labels_with_errors(package_predictions, 2)))
    )
    stored = EvaluationStore(engine).read("synthetic")
    tampered = json.loads(json.dumps(stored))
    tampered["evaluation"]["metrics"]["accuracy"] = 0.95
    assert audit_stored(tampered)["matches"] is False
    tampered = json.loads(json.dumps(stored))
    rows = tampered["evaluation"]["confusion_matrix"]["rows"]
    rows[0][0], rows[0][1] = rows[0][1], rows[0][0]
    assert audit_stored(tampered)["matches"] is False


# --- CLI ----------------------------------------------------------------------------------


def test_cli_preflight_with_the_selection_open_exits_without_touching_the_test(
    engine, act, package, tmp_path
):
    url = str(engine.url)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "evaluation.official",
            "preflight",
            "--package",
            str(package),
            "--manifest",
            str(tmp_path / "no-existe.json"),
            "--candidate-run-id",
            act.candidate_run_id,
            "--checkpoint-sha256",
            act.checkpoint_sha256,
            "--manifest-hash",
            act.manifest_hash,
            "--test-split-hash",
            act.test_split_hash,
            "--outcome-hash",
            act.outcome_hash,
        ],
        cwd=APP_ROOT,
        env={**os.environ, "DATABASE_URL": url, "PYTHONIOENCODING": "utf-8"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    assert result.returncode == 2, result.stderr
    assert "model_selection_open" in result.stderr
