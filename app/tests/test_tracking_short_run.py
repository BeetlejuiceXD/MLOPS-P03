"""D02-06 — `tracking.short_run`: entrenamiento corto real (D02-03) instrumentado
en MLflow (D02-01), sin servidor real ni descarga de pesos preentrenados.

Mismo patrón que `test_tracking_verify.py`: usa el store de archivos local de
MLflow (rápido, sin red) para la lógica de escritura/comprobación. La prueba
contra los servicios reales (MariaDB + MinIO + MLflow) la hace el job
`Corrida corta instrumentada (D02-06)` de `.github/workflows/ci.yml`.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest
import torch

from tracking import short_run
from tracking.settings import P3_EXPERIMENT
from training.class_map import CLASS_MAP


def _closed_port_uri() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


@pytest.fixture
def file_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    return uri


def _run(tmp_path: Path, seed: int = 1) -> tuple[Path, dict]:
    evidence_path = tmp_path / "evidence.json"
    exit_code = short_run.main(["run", "--evidence", str(evidence_path), "--seed", str(seed)])
    assert exit_code == short_run.VERIFIED
    return evidence_path, json.loads(evidence_path.read_text(encoding="utf-8"))


def _client(uri: str):
    from mlflow.tracking import MlflowClient

    return MlflowClient(tracking_uri=uri)


def _latest_run(uri: str):
    """El run más reciente del experimento — para casos donde `run` falla
    antes de escribir evidencia (no hay `run_id` a mano de otra forma)."""
    client = _client(uri)
    experiment = client.get_experiment_by_name(P3_EXPERIMENT)
    runs = client.search_runs(
        [experiment.experiment_id], order_by=["start_time DESC"], max_results=1
    )
    return runs[0]


def test_run_then_check_round_trip(file_store, tmp_path):
    evidence_path, evidence = _run(tmp_path)

    assert evidence["experiment_name"] == P3_EXPERIMENT
    assert len(evidence["run_id"]) == 32
    assert len(evidence["checkpoint_sha256"]) == 64
    assert evidence["epochs"] >= 1

    assert short_run.main(["check", "--evidence", str(evidence_path)]) == short_run.VERIFIED


def test_run_is_tagged_as_short_run_instrumentation_not_campaign(file_store, tmp_path):
    _, evidence = _run(tmp_path)
    run = _client(file_store).get_run(evidence["run_id"])

    assert run.data.tags[short_run.RUN_KIND_TAG] == short_run.RUN_KIND
    assert run.info.status == "FINISHED"


def test_fixture_data_is_marked_by_provenance_not_v011(file_store, tmp_path):
    _, evidence = _run(tmp_path)
    run = _client(file_store).get_run(evidence["run_id"])

    assert run.data.tags[short_run.DATA_PROVENANCE_TAG] == short_run.DATA_PROVENANCE_FIXTURE
    assert evidence["dvc_release"] != "v0.1.1"
    assert evidence["manifest_version"] != "v0.1.1"


def test_dvc_release_hash_is_sha256_of_images_and_annotations_md5(file_store, tmp_path):
    _, evidence = _run(tmp_path)
    run = _client(file_store).get_run(evidence["run_id"])

    expected = short_run.dvc_release_hash(
        evidence["dvc_images_md5"], evidence["dvc_annotations_md5"]
    )
    assert evidence["dvc_release_hash"] == expected
    assert run.data.tags["dvc_release_hash"] == expected
    assert expected == expected.lower()
    assert len(expected) == 64


def test_all_contracted_tags_are_present(file_store, tmp_path):
    _, evidence = _run(tmp_path)
    run = _client(file_store).get_run(evidence["run_id"])

    for tag_name in (
        "git_commit",
        "dvc_release",
        "dvc_images_md5",
        "dvc_annotations_md5",
        "dvc_release_hash",
        "manifest_version",
        "manifest_hash",
        "classes",
        "seed",
        "job_id",
    ):
        assert tag_name in run.data.tags, f"falta el tag {tag_name}"
    assert run.data.tags["classes"] == ",".join(sorted(CLASS_MAP))


def test_effective_training_config_is_logged_as_params(file_store, tmp_path):
    _, evidence = _run(tmp_path)
    run = _client(file_store).get_run(evidence["run_id"])

    for name, value in evidence["config"].items():
        assert run.data.params[name] == str(value)


def test_epoch_metrics_are_logged_with_one_point_per_epoch(file_store, tmp_path):
    _, evidence = _run(tmp_path)
    client = _client(file_store)

    for field_name in short_run.EPOCH_METRIC_FIELDS:
        history = client.get_metric_history(evidence["run_id"], field_name)
        assert len(history) == evidence["epochs"]


def test_summary_fields_match_the_best_epoch_in_history(file_store, tmp_path):
    _, evidence = _run(tmp_path)
    client = _client(file_store)

    accuracy_history = {
        m.step: m.value for m in client.get_metric_history(evidence["run_id"], "val_accuracy")
    }
    assert round(accuracy_history[evidence["best_epoch"]], 6) == round(
        evidence["best_val_accuracy"], 6
    )


def test_checkpoint_is_recoverable_with_matching_hash(file_store, tmp_path):
    _, evidence = _run(tmp_path)
    client = _client(file_store)

    served_sha = short_run._download_sha256(
        client, evidence["run_id"], evidence["checkpoint_artifact_path"]
    )
    assert served_sha == evidence["checkpoint_sha256"]


def test_resource_measurements_are_present_and_positive(file_store, tmp_path):
    _, evidence = _run(tmp_path)

    assert evidence["duration_seconds"] > 0
    assert evidence["peak_memory_mb"] > 0
    assert evidence["device"] in ("cpu", "cuda")


def test_no_test_split_metric_is_ever_logged(file_store, tmp_path):
    # trainer.engine.train() solo ve dataset.train/dataset.val (D02-03); este
    # componente nunca toca ni publica nada del split test, que Ale custodia
    # hasta MODEL SELECTION CLOSED (#33).
    _, evidence = _run(tmp_path)
    client = _client(file_store)
    run = client.get_run(evidence["run_id"])

    logged_metric_names = set(run.data.metrics) | set(short_run.EPOCH_METRIC_FIELDS)
    assert not any("test" in name.lower() for name in logged_metric_names)


def test_reuses_the_existing_experiment_across_runs(file_store, tmp_path):
    _, first = _run(tmp_path, seed=1)
    _, second = _run(tmp_path, seed=2)

    assert first["experiment_id"] == second["experiment_id"]
    assert first["run_id"] != second["run_id"]
    assert first["job_id"] != second["job_id"]


def test_checkpoint_hash_mismatch_is_a_failure_not_a_success(file_store, tmp_path, capsys):
    evidence_path, evidence = _run(tmp_path)
    run = _client(file_store).get_run(evidence["run_id"])
    stored = (
        Path(run.info.artifact_uri.removeprefix("file://")) / evidence["checkpoint_artifact_path"]
    )
    stored.write_bytes(b"contenido alterado")

    assert short_run.main(["check", "--evidence", str(evidence_path)]) == short_run.MISMATCH
    assert "checkpoint" in capsys.readouterr().err.lower()


def test_check_without_server_is_unavailable_not_success(tmp_path, monkeypatch):
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(
        json.dumps(
            {
                "tracking_uri": _closed_port_uri(),
                "experiment_name": P3_EXPERIMENT,
                "experiment_id": "0",
                "run_id": "0" * 32,
                "job_id": "d02-06-0",
                "config": {},
                "classes": ["cat", "dog"],
                "seed": 1,
                "git_commit": "unknown",
                "dvc_release": short_run.FIXTURE_DVC_RELEASE,
                "dvc_images_md5": short_run.FIXTURE_IMAGES_MD5,
                "dvc_annotations_md5": short_run.FIXTURE_ANNOTATIONS_MD5,
                "dvc_release_hash": "x",
                "manifest_version": short_run.FIXTURE_MANIFEST_VERSION,
                "manifest_hash": short_run.FIXTURE_MANIFEST_HASH,
                "epochs": 1,
                "best_epoch": 1,
                "best_val_accuracy": 0.5,
                "best_val_macro_f1": 0.5,
                "best_val_loss": 0.5,
                "checkpoint_artifact_path": "checkpoint/model.pt",
                "checkpoint_sha256": "x",
                "duration_seconds": 1.0,
                "device": "cpu",
                "peak_memory_mb": 1.0,
                "written_at": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MLFLOW_TRACKING_URI", _closed_port_uri())

    assert short_run.main(["check", "--evidence", str(evidence_path)]) == short_run.UNAVAILABLE


# --- Bloqueante 1: device es el dispositivo real del modelo, no una suposición ---


def test_device_of_reflects_actual_model_placement_not_cuda_availability(monkeypatch):
    # CUDA "disponible" simulado (sin GPU real): _device_of no debe confiar en
    # torch.cuda.is_available(), solo en donde quedaron los parámetros del
    # modelo. No se llama a train()/optimizer aquí a propósito: eso sí
    # dispara la inicialización real de CUDA y truena sin driver.
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    model = torch.nn.Linear(4, 2)  # se queda en CPU por default

    assert short_run._device_of(model) == "cpu"


# --- Bloqueante 2: FINISHED solo despues de verificar el checkpoint --------------


def test_run_leaves_status_failed_if_training_itself_raises(file_store, tmp_path, monkeypatch):
    def _boom(*args, **kwargs):
        raise RuntimeError("entrenamiento simulado roto")

    monkeypatch.setattr(short_run, "train", _boom)
    evidence_path = tmp_path / "evidence.json"

    exit_code = short_run.main(["run", "--evidence", str(evidence_path), "--seed", "1"])

    assert exit_code == short_run.UNAVAILABLE
    assert not evidence_path.exists()
    run = _latest_run(file_store)
    assert run.info.status == "FAILED"


def test_run_leaves_status_failed_if_param_logging_raises(file_store, tmp_path, monkeypatch):
    # El bug reportado en la reauditoria: log_param corria antes del try, asi
    # que si fallaba, ningun `except` lo atrapaba y el run quedaba RUNNING
    # para siempre (set_terminated nunca se llamaba).
    from mlflow.tracking import MlflowClient

    def _boom(self, *args, **kwargs):
        raise RuntimeError("log_param simulado roto")

    monkeypatch.setattr(MlflowClient, "log_param", _boom)
    evidence_path = tmp_path / "evidence.json"

    exit_code = short_run.main(["run", "--evidence", str(evidence_path), "--seed", "1"])

    assert exit_code == short_run.UNAVAILABLE
    assert not evidence_path.exists()
    run = _latest_run(file_store)
    assert run.info.status == "FAILED"


def test_run_leaves_status_failed_if_metric_logging_raises(file_store, tmp_path, monkeypatch):
    from mlflow.tracking import MlflowClient

    def _boom(self, *args, **kwargs):
        raise RuntimeError("log_metric simulado roto")

    monkeypatch.setattr(MlflowClient, "log_metric", _boom)
    evidence_path = tmp_path / "evidence.json"

    exit_code = short_run.main(["run", "--evidence", str(evidence_path), "--seed", "1"])

    assert exit_code == short_run.UNAVAILABLE
    assert not evidence_path.exists()
    run = _latest_run(file_store)
    assert run.info.status == "FAILED"


def test_run_leaves_status_failed_if_artifact_logging_raises(file_store, tmp_path, monkeypatch):
    from mlflow.tracking import MlflowClient

    def _boom(self, *args, **kwargs):
        raise RuntimeError("log_artifact simulado roto")

    monkeypatch.setattr(MlflowClient, "log_artifact", _boom)
    evidence_path = tmp_path / "evidence.json"

    exit_code = short_run.main(["run", "--evidence", str(evidence_path), "--seed", "1"])

    assert exit_code == short_run.UNAVAILABLE
    assert not evidence_path.exists()
    run = _latest_run(file_store)
    assert run.info.status == "FAILED"


def test_run_leaves_status_failed_if_checkpoint_verification_fails(
    file_store, tmp_path, monkeypatch
):
    # El corazon del bloqueante 2: la descarga/verificacion inmediata del
    # checkpoint pasa DESPUES de que el run pudo haberse marcado FINISHED en
    # el codigo viejo. Si esto falla, el run debe quedar FAILED, no FINISHED.
    monkeypatch.setattr(short_run, "_download_sha256", lambda *a, **k: "hash-no-coincide")
    evidence_path = tmp_path / "evidence.json"

    exit_code = short_run.main(["run", "--evidence", str(evidence_path), "--seed", "1"])

    assert exit_code == short_run.UNAVAILABLE
    assert not evidence_path.exists()
    run = _latest_run(file_store)
    assert run.info.status == "FAILED"


# --- Bloqueante 3: classes tambien como param de MLflow -------------------------


def test_classes_are_logged_as_mlflow_param(file_store, tmp_path):
    _, evidence = _run(tmp_path)
    run = _client(file_store).get_run(evidence["run_id"])

    assert run.data.params["classes"] == ",".join(evidence["classes"])
