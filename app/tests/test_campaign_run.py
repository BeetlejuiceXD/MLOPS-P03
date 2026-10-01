"""D04-03 — `campaign.run.summarize`: agrega las 12 filas sin red (datos ya
producidos por `run`/`verify_smoke`). Esto cubre los dos "Tests requeridos"
del ticket: auditoría de estados/diversidad, y exclusión de fallidos/
duplicados del mínimo + correspondencia matriz/run. La identidad de datos,
ausencia de métricas de test, curvas y checkpoint por run ya los garantiza
`verify_smoke` (D03-04, `tests/test_smoke.py`) — aquí no se repiten.
"""

from __future__ import annotations

from campaign.matrix import MATRIX
from campaign.run import summarize


def _ok(row: int, run_id: str = "run-x") -> dict:
    return {
        "row": row,
        "change": "x",
        "job": {"id": row, "status": "succeeded"},
        "report": {"run_id": run_id, "problems": []},
    }


def _failed(row: int, problems: list[str]) -> dict:
    return {
        "row": row,
        "change": "x",
        "job": {"id": row, "status": "succeeded"},
        "report": {"run_id": f"run-{row}", "problems": problems},
    }


def _all_twelve_ok() -> list[dict]:
    return [_ok(row.index, run_id=f"run-{row.index}") for row in MATRIX]


def test_twelve_clean_runs_meet_the_minimum():
    summary = summarize(_all_twelve_ok())

    assert summary["valid_count"] == 12
    assert summary["meets_minimum"] is True
    assert summary["min_required"] == 10


def test_exactly_ten_clean_runs_meets_the_minimum_two_missing():
    results = [_ok(row.index) for row in MATRIX[:10]]

    summary = summarize(results)

    assert summary["valid_count"] == 10
    assert summary["meets_minimum"] is True
    not_run = [r for r in summary["rows"] if not r["valid"]]
    assert len(not_run) == 2
    assert {r["row"] for r in not_run} == {11, 12}


def test_nine_clean_runs_does_not_meet_the_minimum():
    results = [_ok(row.index) for row in MATRIX[:9]]

    summary = summarize(results)

    assert summary["valid_count"] == 9
    assert summary["meets_minimum"] is False


def test_a_row_with_verify_smoke_problems_does_not_count_even_if_succeeded():
    results = [_ok(row.index) for row in MATRIX[:11]] + [
        _failed(12, ["el run quedó RUNNING, no FINISHED"])
    ]

    summary = summarize(results)

    assert summary["valid_count"] == 11
    row_12 = next(r for r in summary["rows"] if r["row"] == 12)
    assert row_12["valid"] is False
    assert row_12["problems"] == ["el run quedó RUNNING, no FINISHED"]


def test_a_row_that_never_ran_is_reported_as_missing_not_silently_dropped():
    results = [_ok(row.index) for row in MATRIX if row.index != 7]

    summary = summarize(results)

    row_7 = next(r for r in summary["rows"] if r["row"] == 7)
    assert row_7["valid"] is False
    assert row_7["status"] == "no ejecutada"
    assert summary["valid_count"] == 11


def test_a_failed_job_status_is_excluded_from_the_minimum():
    results = [_ok(row.index) for row in MATRIX[:11]] + [
        {
            "row": 12,
            "change": "x",
            "job": {"id": 12, "status": "failed"},
            "report": {"run_id": None, "problems": []},
        }
    ]

    summary = summarize(results)

    assert summary["valid_count"] == 11
    row_12 = next(r for r in summary["rows"] if r["row"] == 12)
    assert row_12["valid"] is False
    assert row_12["job_status"] == "failed"


def test_a_retried_row_counts_once_using_the_latest_attempt_and_documents_the_retry():
    # Primer intento falla (timeout), el reintento sale limpio: debe contar
    # UNA sola vez, no duplicar la fila 1 en el minimo.
    results = [
        {
            "row": 1,
            "change": "Baseline",
            "job": {"id": 100, "status": "running"},
            "problems": ["timeout: sigue running"],
        },
        _ok(1, run_id="run-retry"),
    ] + [_ok(row.index) for row in MATRIX[1:11]]

    summary = summarize(results)

    row_1 = next(r for r in summary["rows"] if r["row"] == 1)
    assert row_1["valid"] is True
    assert row_1["retries"] == 1
    assert row_1["run_id"] == "run-retry"
    assert summary["valid_count"] == 11  # fila 1 cuenta una vez + filas 2..11


def test_all_twelve_matrix_rows_are_always_reported_even_with_empty_results():
    summary = summarize([])

    assert {r["row"] for r in summary["rows"]} == {row.index for row in MATRIX}
    assert summary["valid_count"] == 0
    assert summary["meets_minimum"] is False
