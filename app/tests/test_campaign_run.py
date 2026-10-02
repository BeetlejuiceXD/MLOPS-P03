"""D04-03 — `campaign.run.summarize`: agrega las 12 filas sin red (datos ya
producidos por `run`/`verify_smoke`). Cubre los "Tests requeridos" del
ticket y la reauditoría de #72: selección de reintentos por `start_time`
(#33 — nunca por métrica, eso sería p-hacking entre corridas reales),
endurecimiento del auditor (fila→request→job.config, job.mlflow_run_id↔
report.run_id, task/run_kind=training, reporte completo, unicidad de
run_id), y el código de salida en la frontera de `main()`. La identidad de
datos, curvas y checkpoint por run ya los garantiza `verify_smoke` (D03-04,
`tests/test_smoke.py`) — aquí no se repiten.
"""

from __future__ import annotations

import json

from campaign.matrix import MATRIX, to_training_config_kwargs
from campaign.run import MISMATCH, VERIFIED, _parse_rows, main, summarize
from training.config import TrainingConfig


def _frozen_config(row_index: int) -> dict:
    row = next(r for r in MATRIX if r.index == row_index)
    return TrainingConfig(**to_training_config_kwargs(row)).model_dump()


def _ok(
    row: int,
    run_id: str = "run-x",
    *,
    start_time: int = 1_000_000,
    job_id: int | None = None,
    config: dict | None = None,
) -> dict:
    config = config if config is not None else _frozen_config(row)
    resolved_job_id = job_id if job_id is not None else row
    return {
        "row": row,
        "change": "x",
        "request": {"task": "training", "config": config},
        "job": {
            "id": resolved_job_id,
            "task": "training",
            "status": "succeeded",
            "config": config,
            "mlflow_run_id": run_id,
        },
        "report": {
            "run_id": run_id,
            "run_status": "FINISHED",
            "best_epoch": 10,
            "best_val_accuracy": 0.90,
            "best_val_macro_f1": 0.90,
            "checkpoint_sha256": "a" * 64,
            "manifest_hash": "b" * 64,
            "tags": {"p3.run_kind": "training", "job_id": str(resolved_job_id)},
            "problems": [],
        },
        "details": {
            "start_time": start_time,
            "duration_seconds": 12.5,
            "peak_memory_mb": 512.0,
            "device": "cpu",
            "git_commit": "abc123",
            "run_kind": "training",
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
            "request": {"task": "training", "config": {}},
            "job": {"id": 12, "status": "failed", "config": {}, "mlflow_run_id": None},
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
        "run_kind": "training",
    }


# --- Reintentos (#33): se elige por start_time, NUNCA por metrica, NUNCA ----------
# --- usando run_id como sustituto de un start_time faltante ----------------------


def test_retry_selection_picks_earlier_start_time_not_better_metric():
    # El intento 2 corrio DESPUES pero "parece mejor" si uno mirara metricas
    # (no las tiene en este fixture a proposito: #33 prohibe usarlas aqui).
    earlier = _ok(1, run_id="run-earlier", start_time=1_000_000)
    later = _ok(1, run_id="run-later", start_time=2_000_000)
    results = [later, earlier] + [_ok(row.index, run_id=f"run-{row.index}") for row in MATRIX[1:10]]

    summary = summarize(results)

    row_1 = next(r for r in summary["rows"] if r["row"] == 1)
    assert row_1["run_id"] == "run-earlier"
    assert row_1["retries"] == 1


def test_retry_selection_breaks_start_time_tie_with_lower_run_id():
    a = _ok(1, run_id="run-aaa", start_time=1_000_000)
    b = _ok(1, run_id="run-bbb", start_time=1_000_000)
    results = [b, a] + [_ok(row.index, run_id=f"run-{row.index}") for row in MATRIX[1:10]]

    summary = summarize(results)

    row_1 = next(r for r in summary["rows"] if r["row"] == 1)
    assert row_1["run_id"] == "run-aaa"


def test_single_valid_attempt_does_not_need_start_time():
    # Caso real de la campana: un solo intento por fila, sin reintentos. No
    # hay nada que elegir, asi que falta de start_time no debe invalidarla.
    only_attempt = _ok(1, run_id="run-1", start_time=None)
    results = [only_attempt] + [_ok(row.index, run_id=f"run-{row.index}") for row in MATRIX[1:10]]

    summary = summarize(results)

    row_1 = next(r for r in summary["rows"] if r["row"] == 1)
    assert row_1["valid"] is True
    assert row_1["run_id"] == "run-1"


def test_multiple_valid_attempts_with_missing_start_time_are_not_resolved_by_run_id():
    # Dos intentos validos, a uno le falta start_time: NO se debe caer en
    # comparar run_id como si fuera el tiempo. Debe quedar documentado como
    # problema, no resuelto en silencio.
    with_time = _ok(1, run_id="run-aaa", start_time=1_000_000)
    without_time = _ok(1, run_id="run-zzz", start_time=None)
    results = [with_time, without_time] + [
        _ok(row.index, run_id=f"run-{row.index}") for row in MATRIX[1:10]
    ]

    summary = summarize(results)

    row_1 = next(r for r in summary["rows"] if r["row"] == 1)
    assert row_1["valid"] is False
    assert any("start_time" in p for p in row_1["problems"])


def test_multiple_valid_attempts_with_empty_string_start_time_are_not_resolved_by_run_id():
    # start_time="" (cadena vacia, no None) debe tratarse igual que ausente:
    # no se puede elegir representante sin inventar el orden.
    a = _ok(1, run_id="run-aaa", start_time="")
    b = _ok(1, run_id="run-bbb", start_time="")
    results = [b, a] + [_ok(row.index, run_id=f"run-{row.index}") for row in MATRIX[1:10]]

    summary = summarize(results)

    row_1 = next(r for r in summary["rows"] if r["row"] == 1)
    assert row_1["valid"] is False
    assert any("start_time" in p for p in row_1["problems"])


def test_multiple_valid_attempts_with_wrong_type_start_time_are_not_resolved_by_run_id():
    a = _ok(1, run_id="run-aaa", start_time=[1, 2, 3])
    b = _ok(1, run_id="run-bbb", start_time=1_000_000)
    results = [b, a] + [_ok(row.index, run_id=f"run-{row.index}") for row in MATRIX[1:10]]

    summary = summarize(results)

    row_1 = next(r for r in summary["rows"] if r["row"] == 1)
    assert row_1["valid"] is False
    assert any("start_time" in p for p in row_1["problems"])


def test_multiple_valid_attempts_with_non_numeric_string_start_time_are_not_resolved_by_run_id():
    a = _ok(1, run_id="run-aaa", start_time="no-es-un-numero")
    b = _ok(1, run_id="run-bbb", start_time=1_000_000)
    results = [b, a] + [_ok(row.index, run_id=f"run-{row.index}") for row in MATRIX[1:10]]

    summary = summarize(results)

    row_1 = next(r for r in summary["rows"] if r["row"] == 1)
    assert row_1["valid"] is False
    assert any("start_time" in p for p in row_1["problems"])


def test_a_numeric_string_start_time_is_accepted_as_valid():
    # "1000000" (string numerico) si es interpretable -> valido, no se trata
    # como invalido.
    a = _ok(1, run_id="run-earlier", start_time="1000000")
    b = _ok(1, run_id="run-later", start_time=2_000_000)
    results = [b, a] + [_ok(row.index, run_id=f"run-{row.index}") for row in MATRIX[1:10]]

    summary = summarize(results)

    row_1 = next(r for r in summary["rows"] if r["row"] == 1)
    assert row_1["valid"] is True
    assert row_1["run_id"] == "run-earlier"


def test_retry_selection_skips_invalid_attempts_and_picks_the_only_valid_one():
    broken = _failed(1, ["checkpoint no recuperable"])
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


# --- Auditor endurecido (bloqueante 2 de la reauditoria) -------------------------


def test_config_that_does_not_match_the_frozen_matrix_row_invalidates_the_row():
    wrong = _ok(1, run_id="run-1", config={**_frozen_config(1), "seed": 999})

    results = [wrong] + [_ok(row.index, run_id=f"run-{row.index}") for row in MATRIX[1:10]]

    summary = summarize(results)

    row_1 = next(r for r in summary["rows"] if r["row"] == 1)
    assert row_1["valid"] is False
    assert any("fila OFAT congelada" in p for p in row_1["problems"])


def test_job_config_mismatch_versus_request_invalidates_the_row():
    mismatched = _ok(1, run_id="run-1")
    mismatched["job"]["config"] = {**_frozen_config(1), "seed": 999}

    results = [mismatched] + [_ok(row.index, run_id=f"run-{row.index}") for row in MATRIX[1:10]]

    summary = summarize(results)

    row_1 = next(r for r in summary["rows"] if r["row"] == 1)
    assert row_1["valid"] is False
    assert any("job.config" in p for p in row_1["problems"])


def test_job_mlflow_run_id_mismatch_versus_report_run_id_invalidates_the_row():
    mismatched = _ok(1, run_id="run-1")
    mismatched["job"]["mlflow_run_id"] = "otro-run-distinto"

    results = [mismatched] + [_ok(row.index, run_id=f"run-{row.index}") for row in MATRIX[1:10]]

    summary = summarize(results)

    row_1 = next(r for r in summary["rows"] if r["row"] == 1)
    assert row_1["valid"] is False
    assert any("mlflow_run_id" in p for p in row_1["problems"])


def test_request_task_other_than_training_invalidates_the_row():
    wrong_task = _ok(1, run_id="run-1")
    wrong_task["request"]["task"] = "controlled"

    results = [wrong_task] + [_ok(row.index, run_id=f"run-{row.index}") for row in MATRIX[1:10]]

    summary = summarize(results)

    row_1 = next(r for r in summary["rows"] if r["row"] == 1)
    assert row_1["valid"] is False
    assert any("request.task" in p for p in row_1["problems"])


def test_job_task_other_than_training_invalidates_the_row():
    # Independiente de request.task: aqui es job.task (lo que el backend
    # guardo), que podria diferir del request si hay un bug de la API.
    wrong_job_task = _ok(1, run_id="run-1")
    wrong_job_task["job"]["task"] = "controlled_task"

    results = [wrong_job_task] + [_ok(row.index, run_id=f"run-{row.index}") for row in MATRIX[1:10]]

    summary = summarize(results)

    row_1 = next(r for r in summary["rows"] if r["row"] == 1)
    assert row_1["valid"] is False
    assert any("job.task" in p for p in row_1["problems"])


def test_job_id_mismatch_versus_run_tag_invalidates_the_row():
    # El tag job_id que el run guarda en MLflow debe coincidir con job.id;
    # si no, algo esta mal enlazado entre el job y el run.
    wrong_job_id_tag = _ok(1, run_id="run-1", job_id=1)
    wrong_job_id_tag["report"]["tags"]["job_id"] = "999"  # no coincide con job.id=1

    results = [wrong_job_id_tag] + [
        _ok(row.index, run_id=f"run-{row.index}") for row in MATRIX[1:10]
    ]

    summary = summarize(results)

    row_1 = next(r for r in summary["rows"] if r["row"] == 1)
    assert row_1["valid"] is False
    assert any("tag job_id" in p for p in row_1["problems"])


def test_run_kind_mismatch_reported_by_verify_smoke_invalidates_the_row():
    # p3.run_kind lo valida `verify_smoke` (D03-04) contra los tags reales
    # sin filtrar, no `campaign.run` — así que un run_kind incorrecto llega
    # aquí como un problema ya puesto en report["problems"], no como un tag
    # que campaign.run reinterprete.
    wrong_kind = _failed(
        1, ["tag p3.run_kind='controlled_task' no corresponde al job ('training')"]
    )

    results = [wrong_kind] + [_ok(row.index, run_id=f"run-{row.index}") for row in MATRIX[1:10]]

    summary = summarize(results)

    row_1 = next(r for r in summary["rows"] if r["row"] == 1)
    assert row_1["valid"] is False
    assert any("p3.run_kind" in p for p in row_1["problems"])


def test_succeeded_job_without_finished_run_status_invalidates_the_row():
    not_finished = _ok(1, run_id="run-1")
    not_finished["report"]["run_status"] = "RUNNING"

    results = [not_finished] + [_ok(row.index, run_id=f"run-{row.index}") for row in MATRIX[1:10]]

    summary = summarize(results)

    row_1 = next(r for r in summary["rows"] if r["row"] == 1)
    assert row_1["valid"] is False
    assert any("RUNNING" in p for p in row_1["problems"])


def test_incomplete_report_invalidates_the_row():
    incomplete = _ok(1, run_id="run-1")
    incomplete["report"]["best_val_accuracy"] = None
    incomplete["report"]["checkpoint_sha256"] = None

    results = [incomplete] + [_ok(row.index, run_id=f"run-{row.index}") for row in MATRIX[1:10]]

    summary = summarize(results)

    row_1 = next(r for r in summary["rows"] if r["row"] == 1)
    assert row_1["valid"] is False
    assert any("reporte incompleto" in p for p in row_1["problems"])


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


# --- Codigo de salida en la frontera de main() (bloqueante 5) --------------------


def test_main_audit_returns_mismatch_exit_code_when_below_minimum(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    results = [_ok(row.index, run_id=f"run-{row.index}") for row in MATRIX[:9]]
    evidence_path.write_text(json.dumps({"results": results}), encoding="utf-8")

    exit_code = main(["audit", "--evidence", str(evidence_path)])

    assert exit_code == MISMATCH


def test_main_audit_returns_verified_exit_code_when_minimum_met(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    results = [_ok(row.index, run_id=f"run-{row.index}") for row in MATRIX[:10]]
    evidence_path.write_text(json.dumps({"results": results}), encoding="utf-8")

    exit_code = main(["audit", "--evidence", str(evidence_path)])

    assert exit_code == VERIFIED


# --- --rows: corrida de verificación de una sola fila antes de la matriz completa -


def test_parse_rows_default_none_means_all_twelve():
    assert _parse_rows(None) is None


def test_parse_rows_single_value():
    assert _parse_rows("1") == {1}


def test_parse_rows_multiple_values_with_spaces():
    assert _parse_rows("1, 3,5") == {1, 3, 5}
