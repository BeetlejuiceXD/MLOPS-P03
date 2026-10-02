"""D04-05 — Productor de la evaluación: motor de D03-05 + guardas de D04-04 + exportación.

Todo con predicciones SINTÉTICAS conocidas en el namespace `synthetic`: 6 crops
inventados (no son el frozen test), matriz esperada [[2, 1], [1, 2]]. Ningún modelo,
ningún dato del test oficial. La base es SQLite con las mismas columnas que las
migraciones de MariaDB (comprobado abajo); la prueba con MariaDB real es el job de CI
"Jobs persistentes" (`.github/scripts/evaluation_e2e.py`).

`backend/tests/fixtures/evaluation-synthetic.json` es la salida EXACTA de
`build_evaluation` con estas entradas: el backend la consume en sus tests, así que el
recorrido Python → API queda amarrado a un mismo archivo.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select

from evaluation import producer, store
from evaluation.producer import (
    ClosedSelection,
    EvaluationRefusedError,
    SamplePrediction,
    TestPartition,
    build_evaluation,
    confusion_rows,
    produce_evaluation,
)
from evaluation.store import EvaluationStore, p3_evaluation, p3_model_selection
from presentation.contracts import frozen_test_split_hash

REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS = REPO_ROOT / "backend/src/data/db/migrations"
BACKEND_FIXTURE = REPO_ROOT / "backend/tests/fixtures/evaluation-synthetic.json"

CANDIDATE = "a" * 32
MANIFEST_HASH = "d" * 64
CLOSED_AT = datetime(2026, 10, 1, 13, 0, 0, 456000, tzinfo=UTC)
EVALUATED_AT = datetime(2026, 10, 1, 14, 0, 0, tzinfo=UTC)
TEST_IDS = (3, 7, 12, 20, 21, 30)

# (crop_id, real, predicha, p_cat, p_dog). Filas reales, columnas predichas:
# cat: 2 bien, 1 → dog; dog: 1 → cat, 2 bien.
KNOWN = [
    (12, "cat", "dog", 0.3, 0.7),  # desordenado a propósito: la exportación ordena
    (3, "cat", "cat", 0.9, 0.1),
    (7, "cat", "cat", 0.8, 0.2),
    (20, "dog", "dog", 0.25, 0.75),
    (21, "dog", "cat", 0.6, 0.4),
    (30, "dog", "dog", 0.1, 0.9),
]


def _samples(rows=KNOWN) -> list[SamplePrediction]:
    return [
        SamplePrediction(
            crop_id=crop_id,
            true_class=true,
            predicted_class=predicted,
            probabilities={"cat": p_cat, "dog": p_dog},
        )
        for crop_id, true, predicted, p_cat, p_dog in rows
    ]


SELECTION = ClosedSelection(
    candidate_run_id=CANDIDATE, closed_at=CLOSED_AT, manifest_hash=MANIFEST_HASH
)
PARTITION = TestPartition(
    manifest_hash=MANIFEST_HASH,
    test_split_hash=frozen_test_split_hash(list(TEST_IDS)),
    crop_ids=TEST_IDS,
)


def _build(samples=None, **overrides):
    kwargs = {
        "namespace": "synthetic",
        "model_run_id": CANDIDATE,
        "selection": SELECTION,
        "partition": PARTITION,
        "evaluated_at": EVALUATED_AT,
    }
    kwargs.update(overrides)
    return build_evaluation(_samples() if samples is None else samples, **kwargs)


def _refused(reason: str, call) -> EvaluationRefusedError:
    """Exige el rechazo POR ESTE motivo: otro motivo no demuestra que la regla exista, y
    otra excepción (un crash) tampoco es un rechazo: ambos son un veredicto del test."""
    try:
        call()
    except EvaluationRefusedError as error:
        assert error.reason == reason, error
        return error
    except Exception as error:  # el crash es el veredicto, no se propaga
        pytest.fail(f"se esperaba el rechazo {reason}; llegó {type(error).__name__}: {error}")
    pytest.fail(f"no se rechazó (se esperaba {reason})")


def _accepted(call):
    """Camino feliz: un rechazo inesperado es un veredicto del test, no un crash."""
    try:
        return call()
    except EvaluationRefusedError as error:
        pytest.fail(f"rechazo inesperado {error.reason}: {error.detail}")


# --- Cálculo y exportación con predicciones conocidas ---------------------------------


def test_known_predictions_give_the_expected_matrix_metrics_and_export():
    record = _accepted(_build)
    evaluation = record.evaluation.model_dump(mode="json")
    exported = record.predictions.model_dump(mode="json")

    assert record.namespace == "synthetic"
    assert evaluation["state"] == "ready"
    # D05-05: el resultado lleva su namespace; la UI lo muestra y la API oficial lo exige.
    assert evaluation["namespace"] == "synthetic"
    assert evaluation["confusion_matrix"] == {"labels": ["cat", "dog"], "rows": [[2, 1], [1, 2]]}
    assert evaluation["n_test"] == 6
    assert evaluation["metrics"]["accuracy"] == 4 / 6
    assert evaluation["metrics"]["macro_f1"] == pytest.approx(2 / 3, abs=1e-12)
    assert evaluation["majority_baseline_accuracy"] == 0.5
    for stats in evaluation["metrics"]["per_class"]:
        assert stats["support"] == 3
        assert stats["precision"] == pytest.approx(2 / 3, abs=1e-12)
        assert stats["recall"] == pytest.approx(2 / 3, abs=1e-12)

    # Trazabilidad: candidato, cierre, manifest y partición test en ambos payloads.
    assert evaluation["selection"] == {
        "candidate_run_id": CANDIDATE,
        "metric": "val_accuracy",
        "closed_at": "2026-10-01T13:00:00.456Z",
    }
    assert evaluation["manifest_hash"] == MANIFEST_HASH
    assert evaluation["evaluated_at"] == "2026-10-01T14:00:00.000Z"
    assert exported["namespace"] == "synthetic"
    assert exported["candidate_run_id"] == CANDIDATE
    assert exported["manifest_hash"] == MANIFEST_HASH
    assert exported["test_split_hash"] == frozen_test_split_hash(list(TEST_IDS))
    assert exported["evaluated_at"] == evaluation["evaluated_at"]
    assert exported["classes"] == ["cat", "dog"]
    assert exported["n_test"] == 6

    # Por muestra: IDs ordenados, verdad, predicción y probabilidades tal cual.
    assert [s["crop_id"] for s in exported["predictions"]] == list(TEST_IDS)
    by_id = {s["crop_id"]: s for s in exported["predictions"]}
    assert by_id[12] == {
        "crop_id": 12,
        "true_class": "cat",
        "predicted_class": "dog",
        "probabilities": {"cat": 0.3, "dog": 0.7},
    }
    assert by_id[21]["predicted_class"] == "cat"


def test_export_rebuilds_exactly_the_reported_matrix():
    record = _accepted(_build)
    rows = confusion_rows(record.predictions)
    assert rows == [[2, 1], [1, 2]]
    assert rows == record.evaluation.confusion_matrix.rows


def test_export_matrix_rows_are_true_class_and_columns_predicted_class():
    """Matriz asimétrica: filas = real, columnas = predicha (la simétrica no lo distingue)."""
    skewed = [(3, "dog", "cat", 0.9, 0.1), *KNOWN[:1], *KNOWN[2:]]
    record = _accepted(lambda: _build(_samples(skewed)))
    rows = confusion_rows(record.predictions)
    assert rows == [[1, 1], [2, 2]]
    assert rows == record.evaluation.confusion_matrix.rows


def test_backend_fixture_is_the_exact_output_of_the_producer():
    record = _accepted(_build)
    fixture = json.loads(BACKEND_FIXTURE.read_text(encoding="utf-8"))
    assert fixture == {
        "evaluation": record.evaluation.model_dump(mode="json"),
        "predictions": record.predictions.model_dump(mode="json"),
    }


def test_partition_comes_from_the_frozen_manifest_test_assignments():
    manifest = type(
        "Manifest",
        (),
        {
            "manifest_hash": MANIFEST_HASH,
            "test_split_hash": "f" * 64,
            "assignments": type("A", (), {"test": [30, 3, 21, 7, 20, 12]})(),
        },
    )()
    partition = TestPartition.from_manifest(manifest)
    assert partition == TestPartition(
        manifest_hash=MANIFEST_HASH, test_split_hash="f" * 64, crop_ids=(3, 7, 12, 20, 21, 30)
    )


# --- Datos incompatibles -----------------------------------------------------------------


def test_predictions_of_a_run_other_than_the_closed_candidate_are_refused():
    error = _refused("not_selected_candidate", lambda: _build(model_run_id="b" * 32))
    assert CANDIDATE in error.detail


def test_manifest_other_than_the_one_of_the_selection_is_refused():
    other = replace(PARTITION, manifest_hash="e" * 64)
    _refused("manifest_mismatch", lambda: _build(partition=other))


def test_test_split_hash_that_does_not_match_the_ids_is_refused():
    other = replace(PARTITION, test_split_hash="0" * 64)
    _refused("test_split_hash_mismatch", lambda: _build(partition=other))


def test_missing_crop_of_the_test_partition_is_refused():
    error = _refused("crop_ids_not_test_split", lambda: _build(_samples(KNOWN[:-1])))
    assert "faltan 1" in error.detail


def test_crop_outside_the_test_partition_is_refused():
    extra = [*KNOWN, (99, "dog", "dog", 0.1, 0.9)]
    error = _refused("crop_ids_not_test_split", lambda: _build(_samples(extra)))
    assert "sobran 1" in error.detail


def test_duplicated_crop_is_refused():
    duplicated = [*KNOWN[:-1], (3, "dog", "dog", 0.1, 0.9)]
    _refused("duplicate_crop_id", lambda: _build(_samples(duplicated)))


def test_unknown_namespace_is_refused():
    _refused("unknown_namespace", lambda: _build(namespace="test"))


@pytest.mark.parametrize("namespace", ["official", "synthetic"])
def test_evaluation_and_export_declare_the_same_namespace(namespace):
    """D05-05: el namespace viaja en los dos JSON; uno sintético nunca dice official."""
    record = _build(namespace=namespace)
    assert record.evaluation.namespace == namespace
    assert record.predictions.namespace == namespace


def test_evaluation_not_after_the_close_is_refused():
    _refused("evaluated_before_close", lambda: _build(evaluated_at=CLOSED_AT))
    # Igual al milisegundo (lo que guardan MariaDB y el contrato) tampoco es "después".
    _refused(
        "evaluated_before_close",
        lambda: _build(evaluated_at=CLOSED_AT + timedelta(microseconds=900)),
    )


@pytest.mark.parametrize(
    "row",
    [
        (3, "cat", "dog", 0.9, 0.1),  # predicha ≠ argmax
        (3, "cat", "cat", 0.9, 0.3),  # no suman ~1
        (3, "Cat", "cat", 0.9, 0.1),  # clase real fuera de cat/dog
    ],
    ids=["not-argmax", "sum", "unknown-class"],
)
def test_invalid_sample_is_refused(row):
    # Reemplaza al crop 3 (KNOWN[1]); los demás quedan igual.
    _refused("invalid_prediction", lambda: _build(_samples([KNOWN[0], row, *KNOWN[2:]])))


def test_probability_of_an_undeclared_class_is_refused():
    sample = SamplePrediction(
        crop_id=3, true_class="cat", predicted_class="cat", probabilities={"cat": 1.0}
    )
    others = _samples([KNOWN[0], *KNOWN[2:]])
    _refused("invalid_prediction", lambda: _build([sample, *others]))


# --- Guardas de D04-04 sobre la base y persistencia ------------------------------------


@pytest.fixture
def engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'evaluation.db'}")
    store.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(p3_model_selection.insert().values(id=1, status="open"))
    return engine


def _set_selection(engine, status: str, outcome: dict | None = None, closed_at=None) -> None:
    with engine.begin() as conn:
        conn.execute(
            p3_model_selection.update()
            .where(p3_model_selection.c.id == 1)
            .values(
                status=status,
                outcome=None if outcome is None else json.dumps(outcome),
                outcome_hash=None if outcome is None else "1" * 64,
                closed_at=closed_at,
            )
        )


OUTCOME = {
    "reference": {
        "dataset_version": "v0.1.1",
        "manifest_hash": MANIFEST_HASH,
        "dvc_release_hash": "e" * 64,
    },
    "candidate": {"run_id": CANDIDATE, "campaign_row": 1},
    "outcome_hash": "1" * 64,
}


def _close(engine) -> None:
    _set_selection(engine, "closed", OUTCOME, CLOSED_AT.replace(tzinfo=None))


def _rows(engine) -> dict[str, int]:
    with engine.connect() as conn:
        result = conn.execute(
            select(p3_evaluation.c.namespace, func.count()).group_by(p3_evaluation.c.namespace)
        ).all()
    return dict(result)


def _produce(evaluation_store, namespace="synthetic", **overrides):
    kwargs = {
        "namespace": namespace,
        "model_run_id": CANDIDATE,
        "partition": PARTITION,
        "clock": lambda: EVALUATED_AT,
    }
    kwargs.update(overrides)
    return produce_evaluation(evaluation_store, _samples(), **kwargs)


@pytest.mark.parametrize("status", ["open", "candidate"])
def test_nothing_is_evaluated_or_written_before_model_selection_closed(engine, status):
    _set_selection(engine, status, OUTCOME if status == "candidate" else None)
    evaluation_store = EvaluationStore(engine)
    _refused("model_selection_open", lambda: _produce(evaluation_store))
    _refused("model_selection_open", lambda: _produce(evaluation_store, namespace="official"))
    assert _rows(engine) == {}


def test_guard_runs_before_reading_any_prediction(engine):
    """La guarda va primero: con la selección abierta no se toca ni una predicción."""

    class Untouchable:
        def __iter__(self):
            raise AssertionError("se leyeron predicciones antes de la guarda")

    _refused(
        "model_selection_open",
        lambda: produce_evaluation(
            EvaluationStore(engine),
            Untouchable(),
            namespace="synthetic",
            model_run_id=CANDIDATE,
            partition=PARTITION,
            clock=lambda: EVALUATED_AT,
        ),
    )


def test_closed_record_without_candidate_keeps_the_test_blocked(engine):
    _set_selection(engine, "closed", {"reference": OUTCOME["reference"]}, CLOSED_AT)
    _refused("incomplete_selection", lambda: _produce(EvaluationStore(engine)))
    _set_selection(engine, "closed", OUTCOME, None)
    _refused("incomplete_selection", lambda: _produce(EvaluationStore(engine)))
    assert _rows(engine) == {}


def test_closed_selection_is_read_from_the_persisted_record(engine):
    _close(engine)
    assert EvaluationStore(engine).closed_selection() == SELECTION


def test_synthetic_run_is_persisted_and_read_back_identically(engine):
    _close(engine)
    evaluation_store = EvaluationStore(engine)
    record = _accepted(lambda: _produce(evaluation_store))
    assert _rows(engine) == {"synthetic": 1}
    stored = evaluation_store.read("synthetic")
    assert stored == {
        "evaluation": record.evaluation.model_dump(mode="json"),
        "predictions": record.predictions.model_dump(mode="json"),
    }
    with engine.connect() as conn:
        row = conn.execute(select(p3_evaluation)).one()
    assert row.candidate_run_id == CANDIDATE
    assert row.evaluated_at == EVALUATED_AT.replace(tzinfo=None)
    # Isolation: el namespace sintético nunca aparece como oficial.
    assert evaluation_store.read("official") is None


def test_synthetic_run_can_be_repeated_and_replaces_the_previous_one(engine):
    _close(engine)
    evaluation_store = EvaluationStore(engine)
    _accepted(lambda: _produce(evaluation_store))
    later = EVALUATED_AT + timedelta(hours=1)
    _accepted(lambda: _produce(evaluation_store, clock=lambda: later))
    assert _rows(engine) == {"synthetic": 1}
    stored = evaluation_store.read("synthetic")
    assert stored["evaluation"]["evaluated_at"] == "2026-10-01T15:00:00.000Z"


def test_official_evaluation_is_recorded_once_and_never_overwritten(engine):
    _close(engine)
    evaluation_store = EvaluationStore(engine)
    first = _accepted(lambda: _produce(evaluation_store, namespace="official"))
    later = EVALUATED_AT + timedelta(hours=1)
    _refused(
        "official_already_recorded",
        lambda: _produce(evaluation_store, namespace="official", clock=lambda: later),
    )
    assert _rows(engine) == {"official": 1}
    stored = evaluation_store.read("official")
    assert stored["evaluation"] == first.evaluation.model_dump(mode="json")


def test_incompatible_data_writes_nothing(engine):
    _close(engine)
    evaluation_store = EvaluationStore(engine)
    _refused("not_selected_candidate", lambda: _produce(evaluation_store, model_run_id="b" * 32))
    assert _rows(engine) == {}


def test_producer_reads_no_files(monkeypatch, engine):
    """El productor recibe predicciones; no abre el manifest ni el test por su cuenta."""
    _close(engine)

    def forbidden(*args, **kwargs):
        raise AssertionError("el productor intentó abrir un archivo")

    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr(Path, "read_text", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    _accepted(lambda: _produce(EvaluationStore(engine)))
    assert not hasattr(producer, "load_manifest")


# --- Mismas columnas que MariaDB ----------------------------------------------------------


def _migration_columns(table: str) -> set[str]:
    columns: set[str] = set()
    for migration in sorted(MIGRATIONS.glob("*.sql")):
        sql = migration.read_text(encoding="utf-8")
        block = re.search(rf"CREATE TABLE `{table}` \((.*?)\n\);", sql, re.S)
        if block:
            columns |= set(re.findall(r"^\t`([a-z_]+)`", block.group(1), re.M))
        columns |= set(re.findall(rf"ALTER TABLE `{table}` ADD `([a-z_]+)`", sql))
    return columns


@pytest.mark.parametrize("table", [p3_evaluation, p3_model_selection], ids=lambda t: t.name)
def test_tables_have_the_same_columns_as_the_backend_migrations(table):
    assert {column.name for column in table.columns} == _migration_columns(table.name)
