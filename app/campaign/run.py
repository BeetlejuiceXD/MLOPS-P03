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
    return {
        "row": row.index,
        "change": row.change,
        "request": request,
        "job": job,
        "report": asdict(report),
    }


def summarize(results: list[dict]) -> dict:
    """Agregación pura (sin I/O): ≥10 filas válidas, sin duplicados ni filas
    faltantes, cada problema documentado. `verify_smoke` ya garantiza por run
    que no hay métricas de test y que la identidad/config/curvas/checkpoint
    coinciden — aquí solo se audita la cobertura de la matriz."""
    by_row: dict[int, list[dict]] = {}
    for result in results:
        by_row.setdefault(result["row"], []).append(result)

    rows_report = []
    valid_rows: list[int] = []
    for row in MATRIX:
        attempts = by_row.get(row.index, [])
        if not attempts:
            rows_report.append(
                {"row": row.index, "change": row.change, "status": "no ejecutada", "valid": False}
            )
            continue

        retries = len(attempts) - 1
        chosen = attempts[-1]  # el intento mas reciente decide; los anteriores quedan documentados
        problems = list(chosen.get("problems") or []) + list(
            (chosen.get("report") or {}).get("problems") or []
        )
        succeeded = chosen["job"]["status"] == "succeeded"
        valid = succeeded and not problems
        if valid:
            valid_rows.append(row.index)
        rows_report.append(
            {
                "row": row.index,
                "change": row.change,
                "job_id": chosen["job"].get("id"),
                "job_status": chosen["job"]["status"],
                "run_id": (chosen.get("report") or {}).get("run_id"),
                "valid": valid,
                "problems": problems,
                "retries": retries,
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
        print(f"  {label}: OK — job #{row['job_id']} run {row['run_id']}{retry_note}")
    return VERIFIED if summary["meets_minimum"] else MISMATCH


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
