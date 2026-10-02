#!/usr/bin/env python3
"""D104 — verifica un MLflow/backend (productor o restaurado) contra
reports/campaign_p3.json y el estado de selección esperado. Solo lectura;
no usa ninguna dependencia del proyecto (solo librería estándar) para que
funcione en un clean clone antes de montar el entorno de `app/`.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

VERIFIED, FAILED = 0, 1

EXPECTED_CANDIDATE_RUN_ID = "2d56233c886142b7824e1551b90e8327"
EXPECTED_ROW1_REPRESENTATIVE = "684c6a694c34408fb07c780045656da8"
EXPECTED_ROW1_RETRIES = 4


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.loads(response.read())


def main() -> int:
    parser = argparse.ArgumentParser(description="D104 — verifica un MLflow/backend restaurado")
    parser.add_argument("--api", default="http://localhost:3100")
    parser.add_argument("--tracking-uri", default="http://localhost:5000")
    parser.add_argument("--evidence", type=Path, default=Path("reports/campaign_p3.json"))
    args = parser.parse_args()

    problems: list[str] = []
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    summary_rows = {r["row"]: r for r in evidence["summary"]["rows"]}

    # --- 1. Campaña: 12/12 filas, cada run_id esperado existe y está FINISHED ---
    found = 0
    for row in range(1, 13):
        expected_run_id = summary_rows.get(row, {}).get("run_id")
        if not expected_run_id:
            problems.append(f"fila {row}: sin representante en campaign_p3.json")
            continue
        try:
            run = _get(f"{args.tracking_uri}/api/2.0/mlflow/runs/get?run_id={expected_run_id}")["run"]
        except Exception as error:
            problems.append(f"fila {row}: run {expected_run_id} no encontrado ({error})")
            continue
        if run["info"]["status"] != "FINISHED":
            problems.append(f"fila {row}: status={run['info']['status']}, se esperaba FINISHED")
        tags = {t["key"]: t["value"] for t in run["data"].get("tags", [])}
        metric_names = {m["key"] for m in run["data"].get("metrics", [])}
        test_metrics = [m for m in metric_names if m.lower().startswith("test")]
        if test_metrics:
            problems.append(f"fila {row}: métricas test_* encontradas: {test_metrics}")
        found += 1
    print(f"CAMPAIGN: {found}/12 filas con run_id verificado")

    # --- 2. Fila 1: representante y reintentos ---
    row1 = summary_rows.get(1, {})
    if row1.get("run_id") != EXPECTED_ROW1_REPRESENTATIVE:
        problems.append(
            f"fila 1: representante en evidencia es {row1.get('run_id')}, "
            f"se esperaba {EXPECTED_ROW1_REPRESENTATIVE}"
        )
    if row1.get("retries") != EXPECTED_ROW1_RETRIES:
        problems.append(f"fila 1: retries={row1.get('retries')}, se esperaba {EXPECTED_ROW1_RETRIES}")
    print(f"ROW 1: representante={row1.get('run_id')} retries={row1.get('retries')}")

    # --- 3. Selección: status=candidate, candidate correcto, closed_at=null ---
    try:
        selection = _get(f"{args.api}/selection")
    except Exception as error:
        problems.append(f"GET /selection falló: {error}")
        selection = {}

    if selection.get("status") != "candidate":
        problems.append(f"selection.status={selection.get('status')!r}, se esperaba 'candidate'")
    candidate = (selection.get("candidate") or {}).get("run_id")
    if candidate != EXPECTED_CANDIDATE_RUN_ID:
        problems.append(f"selection.candidate={candidate!r}, se esperaba {EXPECTED_CANDIDATE_RUN_ID!r}")
    if selection.get("closed_at") is not None:
        problems.append(f"selection.closed_at={selection.get('closed_at')!r}, se esperaba null")
    print(f"SELECTION: status={selection.get('status')} candidate={candidate} closed_at={selection.get('closed_at')}")

    # --- 4. Artifact del candidato recuperable ---
    try:
        run = _get(f"{args.tracking_uri}/api/2.0/mlflow/runs/get?run_id={EXPECTED_CANDIDATE_RUN_ID}")["run"]
        tags = {t["key"]: t["value"] for t in run["data"].get("tags", [])}
        print(f"CANDIDATE CHECKPOINT SHA (tag): {tags.get('checkpoint_sha256')}")
    except Exception as error:
        problems.append(f"no se pudo leer el run del candidato: {error}")

    print()
    if problems:
        print("PROBLEMAS ENCONTRADOS:")
        for p in problems:
            print(" -", p)
        return FAILED
    print("VERIFY: PASS — campaña, fila 1 y selección coinciden con reports/campaign_p3.json")
    return VERIFIED


if __name__ == "__main__":
    sys.exit(main())