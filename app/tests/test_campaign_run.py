"""D04-03 — `campaign.run.summarize`: agrega las 12 filas sin red (datos ya
producidos por `run`/`verify_smoke`). Cubre los "Tests requeridos" del
ticket y los 4 bloqueantes de la reauditoría de #72: selección de reintentos
por el criterio de #33 (no "el más reciente"), endurecimiento del auditor
(fila→request→job.config, reporte/FINISHED, IDs, unicidad de run_id), y
evidencia de recursos/commit. La identidad de datos, curvas y checkpoint por
run ya los garantiza `verify_smoke` (D03-04, `tests/test_smoke.py`) — aquí
no se repiten.
"""

from __future__ import annotations

from campaign.matrix import MATRIX
from campaign.run import summarize


def _ok(
    row: int,
    run_id: str = "run-x",
    *,
    val_accuracy: float = 0.90,
    val_macro_f1: float = 0.90,
    val_loss: float = 0.10,
    job_id: int | None = None,
    config: dict | None = None,
) -> dict:
    config = config or {"seed": row}
    return {
        "row": row,
        "change": "x",
        "request": {"config": config},
        "job": {
            "id": job_id if job_id is not None else row,
            "status": "succeeded",
            "config": config,
        },
        "report": {
            "run_id": run_id,
            "run_status": "FINISHED",
            "best_val_accuracy": val_accuracy,
            "best_val_macro_f1": val_macro_f1,
            "problems": [],
        },
        "details": {
            "best_val_loss": val_loss,
            "duration_seconds": 12.5,
            "peak_memory_mb": 512.0,
            "device": "cpu",
            "git_commit": "abc123",
        },
    }


def _failed(row: int, problems: list[str]) -> dict:
    result = _ok(row, run_id=f"run-{row}")
    result["report"]["problems"] = problems
    return result


def _all_twelve_ok() -> list[dict]:
    return [_ok(row.index, run_id=f"run-{row.index}") for row in MATRIX]


def test_twelve_clean_runs_meet_the_minimum():
    summary = summarize(_all_twelve_ok())

    assert summary["valid_count"] == 12
    assert summary["meets_minimum"] is True
    assert summary["min_required"] == 10


def test_exactly_ten_clean_runs_meets_the_minimum_two_missing():
    results = [_ok(row.index, run_id=f"run-{row.index}") for row in MATRIX[:10]]

    summary = summarize(results)

    assert summary["valid_count"] == 10
    assert summary["meets_minimum"] is True
    not_run = [r for r in summary["rows"] if not r["valid"]]
    assert len(not_run) == 2
    assert {r["row"] for r in not_run} == {11, 12}


def test_nine_clean_runs_does_not_meet_the_minimum():
    results = [_ok(row.index, run_id=f"run-{row.index}") for row in MATRIX[:9]]

    summary = summarize(results)

    assert summary["valid_count"] == 9
    assert summary["meets_minimum"] is False


def test_a_row_with_verify_smoke_problems_does_not_count_even_if_succeeded():
    results = [_ok(row.index, run_id=f"run-{row.index}") for row in MATRIX[:11]] + [
        _failed(12, ["el run quedó RUNNING, no FINISHED"])
    ]

    summary = summarize(results)

    assert summary["valid_count"] == 11
    row_12 = next(r for r in summary["rows"] if r["row"] == 12)
    assert row_12["valid"] is False
    assert "el run quedó RUNNING, no FINISHED" in row_12["problems"]


def test_a_row_that_never_ran_is_reported_as_missing_not_silently_dropped():
    results = [_ok(row.index, run_id=f"run-{row.index}") for row in MATRIX if row.index != 7]

    summary = summarize(results)

    row_7 = next(r for r in summary["rows"] if r["row"] == 7)
    assert row_7["valid"] is False
    assert row_7["status"] == "no ejecutada"
    assert summary["valid_count"] == 11


def test_a_failed_job_status_is_excluded_from_the_minimum():
    results = [_ok(row.index, run_id=f"run-{row.index}") for row in MATRIX[:11]] + [
        {
            "row": 12,
            "change": "x",
            "request": {"config": {}},
            "job": {"id": 12, "status": "failed", "config": {}},
            "report": {"run_id": None, "run_status": None, "problems": []},
            "details": {},
        }
    ]

    summary = summarize(results)

    assert summary["valid_count"] == 11
    row_12 = next(r for r in summary["rows"] if r["row"] == 12)
    assert row_12["valid"] is False
    assert row_12["job_status"] == "failed"


def test_resources_and_commit_are_surfaced_per_row():
    results = [_ok(row.index, run_id=f"run-{row.index}") for row in MATRIX[:10]]

    summary = summarize(results)

    row_1 = next(r for r in summary["rows"] if r["row"] == 1)
    assert row_1["resources"] == {
        "duration_seconds": 12.5,
        "peak_memory_mb": 512.0,
        "device": "cpu",
        "git_commit": "abc123",
    }


# --- Bloqueante 1 (#33): seleccion de reintentos por metricas, no por "el mas reciente" ---


def test_retry_selection_picks_higher_accuracy_even_if_it_ran_first():
    first_attempt = _ok(1, run_id="run-good", val_accuracy=0.95)
    second_attempt = _ok(1, run_id="run-worse", val_accuracy=0.80)  # corrio despues, pero es peor
    results = [first_attempt, second_attempt] + [
        _ok(row.index, run_id=f"run-{row.index}") for row in MATRIX[1:10]
    ]

    summary = summarize(results)

    row_1 = next(r for r in summary["rows"] if r["row"] == 1)
    assert row_1["run_id"] == "run-good"
    assert row_1["retries"] == 1


def test_retry_selection_breaks_accuracy_tie_with_macro_f1():
    best = _ok(1, run_id="run-best", val_accuracy=0.90, val_macro_f1=0.95)
    worse = _ok(1, run_id="run-worse", val_accuracy=0.90, val_macro_f1=0.60)
    results = [worse, best] + [_ok(row.index, run_id=f"run-{row.index}") for row in MATRIX[1:10]]

    summary = summarize(results)

    row_1 = next(r for r in summary["rows"] if r["row"] == 1)
    assert row_1["run_id"] == "run-best"


def test_retry_selection_breaks_full_tie_with_lower_run_id():
    # Empate total en accuracy/macro-F1/val_loss a 4 decimales -> gana el run_id menor.
    a = _ok(1, run_id="run-aaa", val_accuracy=0.90, val_macro_f1=0.90, val_loss=0.10)
    b = _ok(1, run_id="run-bbb", val_accuracy=0.90, val_macro_f1=0.90, val_loss=0.10)
    results = [b, a] + [_ok(row.index, run_id=f"run-{row.index}") for row in MATRIX[1:10]]

    summary = summarize(results)

    row_1 = next(r for r in summary["rows"] if r["row"] == 1)
    assert row_1["run_id"] == "run-aaa"


def test_retry_selection_skips_invalid_attempts_and_picks_the_only_valid_one():
    broken = _failed(1, ["checkpoint no recuperable"])
    broken["report"]["run_id"] = "run-broken"
    good = _ok(1, run_id="run-good")
    results = [broken, good] + [_ok(row.index, run_id=f"run-{row.index}") for row in MATRIX[1:10]]

    summary = summarize(results)

    row_1 = next(r for r in summary["rows"] if r["row"] == 1)
    assert row_1["valid"] is True
    assert row_1["run_id"] == "run-good"
    assert row_1["retries"] == 1


def test_all_attempts_invalid_reports_the_most_recent_without_hiding_it():
    first = _failed(1, ["problema A"])
    second = _failed(1, ["problema B"])
    results = [first, second] + [_ok(row.index, run_id=f"run-{row.index}") for row in MATRIX[1:10]]

    summary = summarize(results)

    row_1 = next(r for r in summary["rows"] if r["row"] == 1)
    assert row_1["valid"] is False
    assert row_1["retries"] == 1
    assert "problema B" in row_1["problems"]


# --- Bloqueante 2: auditor endurecido -------------------------------------------


def test_job_config_mismatch_versus_request_invalidates_the_row():
    mismatched = _ok(1, run_id="run-1", config={"seed": 999})
    mismatched["request"]["config"] = {"seed": 1}  # lo que se pidio != lo que el job guardo

    results = [mismatched] + [_ok(row.index, run_id=f"run-{row.index}") for row in MATRIX[1:10]]

    summary = summarize(results)

    row_1 = next(r for r in summary["rows"] if r["row"] == 1)
    assert row_1["valid"] is False
    assert any("job.config" in p for p in row_1["problems"])


def test_succeeded_job_without_finished_run_status_invalidates_the_row():
    not_finished = _ok(1, run_id="run-1")
    not_finished["report"]["run_status"] = "RUNNING"

    results = [not_finished] + [_ok(row.index, run_id=f"run-{row.index}") for row in MATRIX[1:10]]

    summary = summarize(results)

    row_1 = next(r for r in summary["rows"] if r["row"] == 1)
    assert row_1["valid"] is False
    assert any("RUNNING" in p for p in row_1["problems"])


def test_succeeded_job_without_run_id_invalidates_the_row():
    no_run_id = _ok(1, run_id="run-1")
    no_run_id["report"]["run_id"] = None

    results = [no_run_id] + [_ok(row.index, run_id=f"run-{row.index}") for row in MATRIX[1:10]]

    summary = summarize(results)

    row_1 = next(r for r in summary["rows"] if r["row"] == 1)
    assert row_1["valid"] is False
    assert any("run_id" in p for p in row_1["problems"])


def test_missing_job_id_invalidates_the_row():
    no_job_id = _ok(1, run_id="run-1")
    no_job_id["job"]["id"] = None

    results = [no_job_id] + [_ok(row.index, run_id=f"run-{row.index}") for row in MATRIX[1:10]]

    summary = summarize(results)

    row_1 = next(r for r in summary["rows"] if r["row"] == 1)
    assert row_1["valid"] is False
    assert any("job_id" in p for p in row_1["problems"])


def test_duplicate_run_id_across_rows_invalidates_the_later_row():
    row_1_result = _ok(1, run_id="run-shared")
    row_2_result = _ok(2, run_id="run-shared")  # mismo run_id que la fila 1: no puede ser
    results = [row_1_result, row_2_result] + [
        _ok(row.index, run_id=f"run-{row.index}") for row in MATRIX[2:11]
    ]

    summary = summarize(results)

    row_1 = next(r for r in summary["rows"] if r["row"] == 1)
    row_2 = next(r for r in summary["rows"] if r["row"] == 2)
    # la primera fila en reclamarlo (orden de la matriz) se queda con el run_id
    assert row_1["valid"] is True
    assert row_2["valid"] is False
    assert any("duplicado" in p for p in row_2["problems"])
    assert summary["valid_count"] == 10  # 12 intentos - fila 2 invalidada por duplicado
