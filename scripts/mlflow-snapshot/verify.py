#!/usr/bin/env python3
"""D104 — verifica un MLflow/backend RESTAURADO (o el productor) contra la campaña.

`reports/campaign_p3.json` es el CONTRATO: dice qué 16 intentos (12 filas OFAT) deben
existir y con qué valores. Nunca se usa como evidencia de que algo existe: cada
afirmación se comprueba consultando el MLflow y la API reales:

- runs/search paginado: exactamente los 16 run IDs esperados, ninguno extra;
- cada intento: FINISHED, tag job_id correcto, métricas de validation y su historial
  completo por época (best_* = curva en best_epoch), sin ninguna métrica test_*;
- fila 1: 5 intentos, representante job 1 (el de menor start_time real) y 4 reintentos;
- checkpoint/model.pt presente en los 16 runs; el del candidato se DESCARGA y su
  SHA-256 se calcula sobre los bytes;
- GET /selection: status=candidate, closed_at=null, candidato 2d56233c… (fila 3, job 7).

Solo lectura; solo librería estándar. Cualquier discrepancia → exit 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

VERIFIED, FAILED = 0, 1

EXPERIMENT = "p3-cnn-classifier"
EXPECTED_ATTEMPTS = 16
EXPECTED_ROWS = 12
ROW1_ATTEMPTS = 5
ROW1_REPRESENTATIVE_JOB = 1
ROW1_REPRESENTATIVE_RUN = "684c6a694c34408fb07c780045656da8"
CANDIDATE_RUN = "2d56233c886142b7824e1551b90e8327"
CANDIDATE_ROW = 3
CANDIDATE_JOB = 7
CANDIDATE_CHECKPOINT_SHA256 = "0c6b589bdd8ba639ed6890386db5bdd20555adc3452df7e9657c2b4a4b9d563b"
CHECKPOINT_PATH = "checkpoint/model.pt"
EXPECTED_JOB_COUNT = 16

# Curvas por época que registra el trainer (tracking/short_run.py, EPOCH_METRIC_FIELDS).
HISTORY_METRICS = ("train_loss", "train_accuracy", "val_loss", "val_accuracy", "val_macro_f1")
# Resumen de validation que se coteja contra el contrato y contra la curva en best_epoch.
BEST_SUMMARY = (
    ("best_val_accuracy", "val_accuracy"),
    ("best_val_macro_f1", "val_macro_f1"),
    ("best_val_loss", "val_loss"),
)
VALUE_TOLERANCE = 1e-9
PAGE_SIZE = 100

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Attempt:
    row: int
    job_id: int
    run_id: str
    epochs_logged: int
    best_epoch: int
    best_val_accuracy: float
    best_val_macro_f1: float
    best_val_loss: float
    checkpoint_sha256: str


@dataclass(frozen=True)
class Contract:
    attempts: dict[str, Attempt]  # por run_id
    representatives: dict[int, str]  # fila → run_id
    row1_retries: int


class ContractError(Exception):
    pass


def load_contract(evidence: dict) -> Contract:
    """Expectativa versionada. Si el propio contrato no describe 16 intentos / 12 filas
    con la forma esperada, se rechaza: no se verifica contra una expectativa rota."""
    attempts: dict[str, Attempt] = {}
    for result in evidence.get("results", []):
        report, details, job = result["report"], result["details"], result["job"]
        if report["run_id"] != job.get("mlflow_run_id"):
            raise ContractError(f"job {job['id']}: report.run_id != job.mlflow_run_id")
        attempt = Attempt(
            row=int(result["row"]),
            job_id=int(job["id"]),
            run_id=report["run_id"],
            epochs_logged=int(report["epochs_logged"]),
            best_epoch=int(report["best_epoch"]),
            best_val_accuracy=float(report["best_val_accuracy"]),
            best_val_macro_f1=float(report["best_val_macro_f1"]),
            best_val_loss=float(details["best_val_loss"]),
            checkpoint_sha256=report["checkpoint_sha256"],
        )
        if attempt.run_id in attempts:
            raise ContractError(f"run {attempt.run_id} repetido en el contrato")
        attempts[attempt.run_id] = attempt
    rows = evidence.get("summary", {}).get("rows", [])
    representatives = {int(r["row"]): r["run_id"] for r in rows}
    row1 = next((r for r in rows if int(r["row"]) == 1), {})

    by_row: dict[int, list[Attempt]] = defaultdict(list)
    for attempt in attempts.values():
        by_row[attempt.row].append(attempt)
    problems = []
    if len(attempts) != EXPECTED_ATTEMPTS:
        problems.append(f"{len(attempts)} intentos, se esperaban {EXPECTED_ATTEMPTS}")
    if sorted(by_row) != list(range(1, EXPECTED_ROWS + 1)):
        problems.append(f"filas {sorted(by_row)}, se esperaban 1..{EXPECTED_ROWS}")
    if sorted(representatives) != list(range(1, EXPECTED_ROWS + 1)):
        problems.append(f"representantes para filas {sorted(representatives)}")
    for row, items in by_row.items():
        want = ROW1_ATTEMPTS if row == 1 else 1
        if len(items) != want:
            problems.append(f"fila {row}: {len(items)} intentos, se esperaban {want}")
        if representatives.get(row) not in {a.run_id for a in items}:
            problems.append(f"fila {row}: el representante no es uno de sus intentos")
    if representatives.get(1) != ROW1_REPRESENTATIVE_RUN:
        problems.append(f"fila 1: representante {representatives.get(1)}")
    rep1 = attempts.get(ROW1_REPRESENTATIVE_RUN)
    if rep1 is None or rep1.job_id != ROW1_REPRESENTATIVE_JOB:
        problems.append(f"fila 1: el representante no es el job {ROW1_REPRESENTATIVE_JOB}")
    candidate = attempts.get(CANDIDATE_RUN)
    if candidate is None or (candidate.row, candidate.job_id) != (CANDIDATE_ROW, CANDIDATE_JOB):
        problems.append(
            f"candidato {CANDIDATE_RUN} no es fila {CANDIDATE_ROW} / job {CANDIDATE_JOB}"
        )
    elif candidate.checkpoint_sha256 != CANDIDATE_CHECKPOINT_SHA256:
        problems.append("el contrato declara otro checkpoint para el candidato")
    if problems:
        raise ContractError("; ".join(problems))
    return Contract(attempts, representatives, int(row1.get("retries", -1)))


class Http:
    """Acceso real (solo GET/POST de lectura). Los tests lo sustituyen por un fake."""

    def __init__(self, tracking_uri: str, api: str, timeout: float = 30.0):
        self.tracking_uri = tracking_uri.rstrip("/")
        self.api = api.rstrip("/")
        self.timeout = timeout

    def _open(self, url: str, body: dict | None = None):
        data = None if body is None else json.dumps(body).encode()
        headers = {"Content-Type": "application/json"} if body is not None else {}
        request = urllib.request.Request(url, data=data, headers=headers)
        return urllib.request.urlopen(request, timeout=self.timeout)

    def mlflow_get(self, path: str, params: dict) -> dict:
        url = f"{self.tracking_uri}{path}?{urllib.parse.urlencode(params)}"
        with self._open(url) as response:
            return json.loads(response.read())

    def mlflow_post(self, path: str, body: dict) -> dict:
        with self._open(f"{self.tracking_uri}{path}", body) as response:
            return json.loads(response.read())

    def mlflow_sha256(self, run_id: str, path: str) -> tuple[str, int]:
        query = urllib.parse.urlencode({"path": path, "run_uuid": run_id})
        digest, size = hashlib.sha256(), 0
        with self._open(f"{self.tracking_uri}/get-artifact?{query}") as response:
            while chunk := response.read(1 << 20):
                digest.update(chunk)
                size += len(chunk)
        return digest.hexdigest(), size

    def api_get(self, path: str) -> dict:
        with self._open(f"{self.api}{path}") as response:
            return json.loads(response.read())


def _search_runs(http, experiment_id: str) -> list[dict]:
    runs, token = [], None
    while True:
        body = {
            "experiment_ids": [experiment_id],
            "max_results": PAGE_SIZE,
            "run_view_type": "ALL",
        }
        if token:
            body["page_token"] = token
        page = http.mlflow_post("/api/2.0/mlflow/runs/search", body)
        runs.extend(page.get("runs", []))
        token = page.get("next_page_token")
        if not token:
            return runs


def _history(http, run_id: str, key: str) -> list[dict]:
    points, token = [], None
    while True:
        params = {"run_id": run_id, "metric_key": key, "max_results": 1000}
        if token:
            params["page_token"] = token
        page = http.mlflow_get("/api/2.0/mlflow/metrics/get-history", params)
        points.extend(page.get("metrics", []))
        token = page.get("next_page_token")
        if not token:
            return points


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= VALUE_TOLERANCE


def _check_attempt(http, run: dict, attempt: Attempt, problems: list[str]) -> None:
    tag = f"fila {attempt.row} job {attempt.job_id} ({attempt.run_id})"
    info, data = run["info"], run.get("data", {})
    if info.get("status") != "FINISHED":
        problems.append(f"{tag}: status={info.get('status')}, se esperaba FINISHED")
    if info.get("lifecycle_stage", "active") != "active":
        problems.append(f"{tag}: lifecycle_stage={info.get('lifecycle_stage')}")
    tags = {t["key"]: t["value"] for t in data.get("tags", [])}
    if tags.get("job_id") != str(attempt.job_id):
        problems.append(f"{tag}: tag job_id={tags.get('job_id')!r}")
    if tags.get("checkpoint_sha256") != attempt.checkpoint_sha256:
        problems.append(f"{tag}: tag checkpoint_sha256 distinto del contrato")
    metrics = {m["key"]: float(m["value"]) for m in data.get("metrics", [])}
    test_keys = sorted(k for k in metrics if k.lower().startswith("test"))
    if test_keys:
        problems.append(f"{tag}: métricas de frozen test presentes {test_keys}")

    expected_best = {
        "best_val_accuracy": attempt.best_val_accuracy,
        "best_val_macro_f1": attempt.best_val_macro_f1,
        "best_val_loss": attempt.best_val_loss,
    }
    for key in ("best_epoch", *expected_best):
        if key not in metrics:
            problems.append(f"{tag}: falta la métrica {key}")
    if "best_epoch" in metrics and metrics["best_epoch"] != attempt.best_epoch:
        problems.append(f"{tag}: best_epoch={metrics['best_epoch']}, contrato {attempt.best_epoch}")
    for key, want in expected_best.items():
        if key in metrics and not _close(metrics[key], want):
            problems.append(f"{tag}: {key}={metrics[key]}, contrato {want}")

    want_steps = list(range(1, attempt.epochs_logged + 1))
    curves: dict[str, dict[int, float]] = {}
    for key in HISTORY_METRICS:
        points = _history(http, attempt.run_id, key)
        steps = sorted(int(p["step"]) for p in points)
        if steps != want_steps:
            problems.append(
                f"{tag}: historial de {key} con épocas {steps[:3]}…({len(steps)}), "
                f"se esperaban 1..{attempt.epochs_logged}"
            )
            continue
        curves[key] = {int(p["step"]): float(p["value"]) for p in points}
    for summary, curve in BEST_SUMMARY:
        at_best = curves.get(curve, {}).get(attempt.best_epoch)
        if curve in curves and (at_best is None or not _close(at_best, expected_best[summary])):
            problems.append(f"{tag}: {curve}[{attempt.best_epoch}]={at_best} != {summary}")

    try:
        listing = http.mlflow_get(
            "/api/2.0/mlflow/artifacts/list", {"run_id": attempt.run_id, "path": "checkpoint"}
        )
    except Exception as error:
        problems.append(f"{tag}: no se pudo listar checkpoint/ ({error})")
        return
    files = {f["path"]: f for f in listing.get("files", []) if not f.get("is_dir")}
    if CHECKPOINT_PATH not in files:
        problems.append(f"{tag}: falta el artefacto {CHECKPOINT_PATH}")

def _check_training_jobs(http, contract: Contract, problems: list[str]) -> str:
    """Los `training_jobs` ORIGINALES (tabla `image_repo.training_jobs`) deben
    existir en el stack verificado y coincidir, job por job, con el run_id que
    el contrato espera — no basta con que el run exista en MLflow, tiene que
    estar correctamente referenciado desde el job que lo originó."""
    try:
        jobs = http.api_get("/training/jobs").get("jobs", [])
    except Exception as error:
        problems.append(f"GET /training/jobs falló: {error}")
        return "TRAINING_JOBS: no se pudo consultar"

    by_id = {int(j["id"]): j for j in jobs if isinstance(j, dict) and "id" in j}
    if len(by_id) != EXPECTED_JOB_COUNT:
        problems.append(f"training_jobs: {len(by_id)} jobs, se esperaban {EXPECTED_JOB_COUNT}")

    job1_run = (by_id.get(ROW1_REPRESENTATIVE_JOB) or {}).get("mlflow_run_id")
    if job1_run != ROW1_REPRESENTATIVE_RUN:
        problems.append(
            f"training_jobs: job {ROW1_REPRESENTATIVE_JOB} -> {job1_run!r}, "
            f"se esperaba {ROW1_REPRESENTATIVE_RUN!r}"
        )

    job7_run = (by_id.get(CANDIDATE_JOB) or {}).get("mlflow_run_id")
    if job7_run != CANDIDATE_RUN:
        problems.append(
            f"training_jobs: job {CANDIDATE_JOB} -> {job7_run!r}, se esperaba {CANDIDATE_RUN!r}"
        )

    for run_id, attempt in contract.attempts.items():
        actual_run_id = (by_id.get(attempt.job_id) or {}).get("mlflow_run_id")
        if actual_run_id != run_id:
            problems.append(
                f"training_jobs: job {attempt.job_id} -> run {actual_run_id}, pero el contrato dice "
                f"{run_id}"
            )

    return (
        f"TRAINING_JOBS: {len(by_id)}/{EXPECTED_JOB_COUNT} jobs, "
        f"job{ROW1_REPRESENTATIVE_JOB}->{job1_run}, job{CANDIDATE_JOB}->{job7_run}"
    )


def verify(http, evidence: dict) -> tuple[list[str], list[str]]:
    """Devuelve (líneas de reporte, problemas). Sin problemas ⇒ PASS."""
    lines: list[str] = []
    problems: list[str] = []
    try:
        contract = load_contract(evidence)
    except (ContractError, KeyError, TypeError, ValueError) as error:
        return lines, [f"contrato reports/campaign_p3.json inválido: {error}"]

    try:
        experiment = http.mlflow_get(
            "/api/2.0/mlflow/experiments/get-by-name", {"experiment_name": EXPERIMENT}
        )["experiment"]
        runs = _search_runs(http, experiment["experiment_id"])
    except Exception as error:
        return lines, [f"no se pudo consultar el MLflow restaurado: {error}"]

    real = {run["info"]["run_id"]: run for run in runs}
    expected = set(contract.attempts)
    missing = sorted(expected - set(real))
    extra = sorted(set(real) - expected)
    if len(real) != len(runs):
        problems.append("runs/search devolvió run IDs repetidos")
    if missing:
        problems.append(f"faltan {len(missing)} intentos esperados: {missing}")
    if extra:
        problems.append(f"runs inesperados en {EXPERIMENT}: {extra}")
    lines.append(
        f"RUNS: {len(real)} reales en {EXPERIMENT} ({len(runs)} filas de búsqueda), "
        f"{len(expected & set(real))}/{EXPECTED_ATTEMPTS} esperados, {len(extra)} inesperados"
    )

    by_row: dict[int, list[str]] = defaultdict(list)
    for run_id in sorted(expected & set(real)):
        attempt = contract.attempts[run_id]
        by_row[attempt.row].append(run_id)
        _check_attempt(http, real[run_id], attempt, problems)
    lines.append(
        f"CAMPAIGN: {sum(1 for r in by_row if by_row[r])}/{EXPECTED_ROWS} filas con intentos"
    )
    for row in range(1, EXPECTED_ROWS + 1):
        want = ROW1_ATTEMPTS if row == 1 else 1
        if len(by_row.get(row, [])) != want:
            problems.append(
                f"fila {row}: {len(by_row.get(row, []))} intentos reales, se esperaban {want}"
            )

    row1 = by_row.get(1, [])
    if row1:
        earliest = min(row1, key=lambda r: int(real[r]["info"].get("start_time", 0)))
        retries = [r for r in row1 if r != earliest]
        if earliest != ROW1_REPRESENTATIVE_RUN:
            problems.append(
                f"fila 1: el intento más temprano real es {earliest}, no el representante"
            )
        if contract.row1_retries != len(retries) or len(retries) != ROW1_ATTEMPTS - 1:
            problems.append(
                f"fila 1: {len(retries)} reintentos reales, contrato {contract.row1_retries}"
            )
        lines.append(f"ROW 1: representante={earliest} reintentos={len(retries)} {sorted(retries)}")

    try:
        digest, size = http.mlflow_sha256(CANDIDATE_RUN, CHECKPOINT_PATH)
    except Exception as error:
        problems.append(f"candidato: no se pudo descargar {CHECKPOINT_PATH} ({error})")
    else:
        lines.append(f"CANDIDATE CHECKPOINT: sha256(bytes)={digest} size={size}")
        if size == 0 or digest != CANDIDATE_CHECKPOINT_SHA256:
            problems.append(
                f"candidato: SHA-256 de los bytes {digest} != {CANDIDATE_CHECKPOINT_SHA256}"
            )
    lines.append(_check_training_jobs(http, contract, problems))
    try:
        selection = http.api_get("/selection")
    except Exception as error:
        problems.append(f"GET /selection falló: {error}")
        selection = {}
    candidate = selection.get("candidate") or {}
    if selection.get("status") != "candidate":
        problems.append(f"selection.status={selection.get('status')!r}, se esperaba 'candidate'")
    if "closed_at" not in selection or selection.get("closed_at") is not None:
        problems.append(
            f"selection.closed_at={selection.get('closed_at', 'ausente')!r}, se esperaba null"
        )
    if candidate.get("run_id") != CANDIDATE_RUN:
        problems.append(
            f"selection.candidate={candidate.get('run_id')!r}, se esperaba {CANDIDATE_RUN}"
        )
    if candidate.get("campaign_row") != CANDIDATE_ROW:
        problems.append(f"selection.candidate.campaign_row={candidate.get('campaign_row')!r}")
    lines.append(
        f"SELECTION: status={selection.get('status')} candidate={candidate.get('run_id')} "
        f"row={candidate.get('campaign_row')} closed_at={selection.get('closed_at')}"
    )
    return lines, problems


def main(argv: list[str] | None = None, http=None) -> int:
    parser = argparse.ArgumentParser(description="D104 — verifica un MLflow/backend restaurado")
    parser.add_argument("--api", default="http://localhost:3100")
    parser.add_argument("--tracking-uri", default="http://localhost:5000")
    parser.add_argument("--evidence", type=Path, default=REPO_ROOT / "reports/campaign_p3.json")
    args = parser.parse_args(argv)

    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    lines, problems = verify(http or Http(args.tracking_uri, args.api), evidence)
    for line in lines:
        print(line)
    print()
    if problems:
        print(f"VERIFY: FAIL — {len(problems)} problema(s):")
        for problem in problems:
            print(" -", problem)
        return FAILED
    print(
        "VERIFY: PASS — 16 intentos reales (12 filas, fila 1 con 5), historiales completos, "
        "sin test_*, checkpoint del candidato por bytes y selección candidate"
    )
    return VERIFIED


if __name__ == "__main__":
    sys.exit(main())
