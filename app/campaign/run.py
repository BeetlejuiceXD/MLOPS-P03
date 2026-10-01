"""D04-03 — Ejecuta la matriz OFAT congelada (12 filas, #33) como jobs reales
de `task=training`, uno por fila, contra los servicios reales (API + MLflow).
Cada job se audita individualmente con `tracking.smoke.verify_smoke` (D03-04)
antes de contarlo — así ningún run con problemas de identidad, curvas,
best_* o checkpoint cuenta para el mínimo.

    python -m campaign.run run --api http://localhost:8080/api \\
        --tracking-uri http://localhost:5000 --evidence reports/campaign_p3.json

    python -m campaign.run audit --evidence reports/campaign_p3.json

`run` reutiliza el ciclo de vida (crear job → poll → `verify_smoke`) del mismo
modo que `tracking.smoke.run_smoke`, sobre las 12 filas en vez de un solo
`--seed`. Igual que en D03-04, la parte que habla HTTP con servicios reales no
se prueba con fixtures (se evidencia en el PR); `summarize` — el mínimo de 10,
exclusión de fallidos/duplicados y cobertura de las 12 filas — sí, en
`tests/test_campaign_run.py`.

Códigos de salida: 0 = ≥10 filas válidas · 1 = menos de 10 o fuentes
inconsistentes · 2 = servicios no disponibles.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from campaign.matrix import MATRIX, CampaignRow, to_training_config_kwargs
from tracking import smoke
from trainer.metrics import EpochMetrics, is_better
from training.config import TrainingConfig

VERIFIED, MISMATCH, UNAVAILABLE = 0, 1, 2
MIN_VALID_RUNS = 10


def build_request(manifest: dict, row: CampaignRow) -> dict:
    config = TrainingConfig(**to_training_config_kwargs(row)).model_dump()
    return {
        "task": "training",
        "dataset_version": manifest["dataset_version"],
        "manifest_hash": manifest["manifest_hash"],
        "config": config,
    }


def _run_details(tracking_uri: str, run_id: str) -> dict:
    """Evidencia que `verify_smoke` (D03-04) no captura porque no la
    necesitaba: duración, memoria pico, dispositivo, commit y val_loss del
    mejor checkpoint. Se usa aquí para el desempate de reintentos (#33) y
    como evidencia de recursos/commit del PR (ambas, bloqueantes de la
    auditoría de #72)."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)
    return {
        "duration_seconds": run.data.metrics.get("duration_seconds"),
        "peak_memory_mb": run.data.metrics.get("peak_memory_mb"),
        "device": run.data.tags.get("device"),
        "git_commit": run.data.tags.get("git_commit"),
        "best_val_loss": run.data.metrics.get("best_val_loss"),
    }


def _run_one(
    api: str,
    tracking_uri: str,
    manifest: dict,
    releases: dict,
    row: CampaignRow,
    *,
    timeout: float,
    poll: float,
) -> dict:
    request = build_request(manifest, row)
    job = smoke._http("POST", f"{api}/training/jobs", request)
    print(f"Fila {row.index} ({row.change}): job #{job['id']} creado", flush=True)

    deadline = time.monotonic() + timeout
    last = None
    while job["status"] not in smoke.TERMINAL:
        if time.monotonic() > deadline:
            return {
                "row": row.index,
                "change": row.change,
                "job": job,
                "problems": [f"timeout: sigue {job['status']}"],
            }
        time.sleep(poll)
        job = smoke._http("GET", f"{api}/training/jobs/{job['id']}")
        progress = (job.get("progress") or {}).get("epoch")
        if (job["status"], progress) != last:
            last = (job["status"], progress)
            print(f"  fila {row.index}: {job['status']} época {progress}", flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        report = smoke.verify_smoke(
            job,
            tracking_uri=tracking_uri,
            workdir=Path(tmp),
            sources={"releases": releases, "manifest": manifest},
        )

    details = {}
    if report.run_id:
        try:
            details = _run_details(tracking_uri, report.run_id)
        except Exception as error:
            details = {"error": f"no se pudo leer duracion/memoria/device/commit: {error}"}

    return {
        "row": row.index,
        "change": row.change,
        "request": request,
        "job": job,
        "report": asdict(report),
        "details": details,
    }


def _attempt_problems(attempt: dict) -> list[str]:
    """Chequeos endurecidos sobre UN intento, además de lo que ya reporta
    `verify_smoke`: fila→request→job.config, reporte/FINISHED, e IDs
    presentes. La unicidad de `run_id` es entre filas, no por intento — se
    audita aparte en `summarize`."""
    report = attempt.get("report") or {}
    problems = list(attempt.get("problems") or []) + list(report.get("problems") or [])

    request_config = (attempt.get("request") or {}).get("config")
    job_config = (attempt.get("job") or {}).get("config")
    if request_config is not None and job_config is not None and request_config != job_config:
        problems.append("job.config no coincide con el config enviado en el request")

    succeeded = attempt["job"]["status"] == "succeeded"
    if succeeded and report.get("run_status") != "FINISHED":
        problems.append(f"run_status={report.get('run_status')!r}, se esperaba FINISHED")
    if succeeded and not report.get("run_id"):
        problems.append("corrida exitosa sin run_id")
    if not attempt["job"].get("id"):
        problems.append("sin job_id")

    return problems


def _retry_key(attempt: dict) -> dict:
    report = attempt.get("report") or {}
    details = attempt.get("details") or {}
    return {
        "run_id": report.get("run_id") or "",
        "best_val_accuracy": report.get("best_val_accuracy"),
        "best_val_macro_f1": report.get("best_val_macro_f1"),
        "best_val_loss": details.get("best_val_loss"),
    }


def _better_retry(candidate: dict, current: dict) -> bool:
    """¿`candidate` desplaza a `current` como el intento que cuenta para la
    fila? Mismo criterio de selección que #33: accuracy -> macro-F1 ->
    val_loss (reutilizando `trainer.metrics.is_better`), y como última
    instancia run_id menor — el desempate que D02-03 dejó fuera de su
    alcance porque ahí aún no existían corridas reales con run_id."""
    neutral = {"epoch": 0, "train_loss": 0.0, "train_accuracy": 0.0, "learning_rate": 0.0}
    candidate_metrics = EpochMetrics(
        **neutral,
        val_accuracy=candidate["best_val_accuracy"],
        val_macro_f1=candidate["best_val_macro_f1"],
        val_loss=candidate["best_val_loss"],
    )
    current_metrics = EpochMetrics(
        **neutral,
        val_accuracy=current["best_val_accuracy"],
        val_macro_f1=current["best_val_macro_f1"],
        val_loss=current["best_val_loss"],
    )
    if is_better(candidate_metrics, current_metrics):
        return True
    if is_better(current_metrics, candidate_metrics):
        return False
    return candidate["run_id"] < current["run_id"]  # empate total: run_id menor, determinista


def summarize(results: list[dict]) -> dict:
    """Agregación pura (sin I/O): ≥10 filas válidas, sin duplicados ni filas
    faltantes, cada problema documentado. `verify_smoke` ya garantiza por run
    que no hay métricas de test y que la identidad/config/curvas/checkpoint
    coinciden; aquí además se audita fila→request→job.config, reporte/
    FINISHED, IDs, unicidad de `run_id` entre filas, y se elige entre
    reintentos con el criterio de selección de #33 (no "el más reciente")."""
    by_row: dict[int, list[dict]] = {}
    for result in results:
        by_row.setdefault(result["row"], []).append(result)

    rows_report = []
    valid_rows: list[int] = []
    claimed_run_ids: dict[str, int] = {}  # run_id -> primera fila que lo reclamó

    for row in MATRIX:
        attempts = by_row.get(row.index, [])
        if not attempts:
            rows_report.append(
                {"row": row.index, "change": row.change, "status": "no ejecutada", "valid": False}
            )
            continue

        retries = len(attempts) - 1
        enriched = [{**attempt, "problems": _attempt_problems(attempt)} for attempt in attempts]
        candidates = [
            a for a in enriched if a["job"]["status"] == "succeeded" and not a["problems"]
        ]

        if candidates:
            chosen = candidates[0]
            chosen_key = _retry_key(chosen)
            for candidate in candidates[1:]:
                candidate_key = _retry_key(candidate)
                if _better_retry(candidate_key, chosen_key):
                    chosen, chosen_key = candidate, candidate_key
            valid = True
        else:
            chosen = enriched[-1]  # ninguno valido: se documenta el mas reciente, no se oculta
            valid = False

        run_id = (chosen.get("report") or {}).get("run_id")
        if valid and run_id:
            if run_id in claimed_run_ids:
                valid = False
                chosen = {
                    **chosen,
                    "problems": [
                        *chosen["problems"],
                        f"run_id duplicado: ya usado por la fila {claimed_run_ids[run_id]}",
                    ],
                }
            else:
                claimed_run_ids[run_id] = row.index

        if valid:
            valid_rows.append(row.index)

        details = chosen.get("details") or {}
        rows_report.append(
            {
                "row": row.index,
                "change": row.change,
                "job_id": chosen["job"].get("id"),
                "job_status": chosen["job"]["status"],
                "run_id": run_id,
                "valid": valid,
                "problems": chosen["problems"],
                "retries": retries,
                "resources": {
                    "duration_seconds": details.get("duration_seconds"),
                    "peak_memory_mb": details.get("peak_memory_mb"),
                    "device": details.get("device"),
                    "git_commit": details.get("git_commit"),
                },
            }
        )

    return {
        "valid_count": len(valid_rows),
        "valid_rows": valid_rows,
        "min_required": MIN_VALID_RUNS,
        "meets_minimum": len(valid_rows) >= MIN_VALID_RUNS,
        "rows": rows_report,
    }


def _print_summary(summary: dict) -> int:
    print(f"\nFilas válidas: {summary['valid_count']}/12 (mínimo {summary['min_required']})")
    for row in summary["rows"]:
        label = f"fila {row['row']:2d} ({row['change']})"
        if not row["valid"]:
            detail = row["problems"] if row.get("problems") else [row.get("status", "sin detalle")]
            print(f"  {label}: {'; '.join(detail)}")
            continue
        retry_note = f" (tras {row['retries']} reintento(s))" if row["retries"] else ""
        resources = row.get("resources") or {}
        duration = resources.get("duration_seconds")
        memory = resources.get("peak_memory_mb")
        res_note = (
            f" [{duration:.1f}s, {memory:.0f}MB, {resources.get('device')}]"
            if duration is not None and memory is not None
            else ""
        )
        print(f"  {label}: OK — job #{row['job_id']} run {row['run_id']}{retry_note}{res_note}")


def run_campaign(args: argparse.Namespace) -> int:
    api = args.api.rstrip("/")
    try:
        approved, manifest, releases = smoke._sources(api)
    except (smoke.ApiError, OSError) as error:
        print(f"Fuentes no disponibles en {api}: {error}", file=sys.stderr)
        return UNAVAILABLE
    if not manifest.get("frozen") or manifest["dataset_version"] not in approved:
        print(f"Sin manifest congelado sobre un release aprobado: {approved}", file=sys.stderr)
        return UNAVAILABLE

    results = [
        _run_one(
            api, args.tracking_uri, manifest, releases, row, timeout=args.timeout, poll=args.poll
        )
        for row in MATRIX
    ]
    evidence = {"manifest": manifest, "results": results, "summary": summarize(results)}
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return _print_summary(evidence["summary"])


def audit_campaign(args: argparse.Namespace) -> int:
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    summary = summarize(evidence["results"])
    return _print_summary(summary)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D04-03 - campaña de 12 runs OFAT (#33)")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="ejecuta las 12 filas como jobs reales y las audita")
    run.add_argument("--api", default="http://localhost:8080/api")
    run.add_argument("--tracking-uri", default="http://localhost:5000")
    run.add_argument("--timeout", type=float, default=3600)
    run.add_argument("--poll", type=float, default=5)
    run.add_argument("--evidence", type=Path, default=Path("reports/campaign_p3.json"))

    audit = sub.add_parser("audit", help="re-audita una evidencia ya escrita, sin red")
    audit.add_argument("--evidence", type=Path, default=Path("reports/campaign_p3.json"))

    args = parser.parse_args(argv)
    return run_campaign(args) if args.command == "run" else audit_campaign(args)


if __name__ == "__main__":
    sys.exit(main())
