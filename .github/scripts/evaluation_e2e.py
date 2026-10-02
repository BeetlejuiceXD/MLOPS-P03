"""D04-05 — Recorrido real productor Python → MariaDB → API de evaluación (namespace synthetic).

Requiere `docker compose` con mariadb y backend (migración 0007 aplicada) y la imagen de
trainer-worker, donde corre el productor (`app/evaluation`) contra la MariaDB real. Va en
el job "Jobs persistentes" de CI, sobre una base DESECHABLE: reinicia `p3_model_selection`
y borra `p3_evaluation`. Solo biblioteca estándar.

Todo es SINTÉTICO: 6 crops inventados (no son el frozen test), candidato `a…a`, manifest
`d…d`. El cierre de la selección se escribe directo en la base para el recorrido; no es
MODEL SELECTION CLOSED de la campaña (D05-02) ni evaluación oficial (D06-01).

Escenarios:
  1. Selección abierta → el productor se niega (`model_selection_open`) en ambos
     namespaces y no escribe; la API responde blocked (200) y predictions 409.
  2. Selección cerrada (sintética) → predicciones de otro run (`not_selected_candidate`)
     o sin un crop de la partición (`crop_ids_not_test_split`): rechazadas, nada escrito.
  3. Corrida sintética válida → una sola fila `synthetic` en MariaDB.
  4. Aislamiento (D05-05): GET /evaluation → 200 `pending` del namespace official (cerrada,
     sin resultado; la sintética no se sirve como oficial) y /evaluation/predictions → 404.

Deja la fila sintética y la selección cerrada para `backend/tests/evaluation.mariadb.test.ts`,
que la lee con el repositorio real y limpia al final. Sale con 1 al primer fallo.
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = "http://localhost:3100"
REPO = Path(__file__).resolve().parents[2]
CANDIDATE = "a" * 32
COMPOSE_RUN = ["docker", "compose", "run", "--rm", "--no-deps", "-T", "trainer-worker"]

# Corre DENTRO de trainer-worker (misma imagen y DATABASE_URL que el worker).
PRODUCER = r"""
import json, os, sys
from datetime import datetime
from sqlalchemy import create_engine
from evaluation.producer import (
    EvaluationRefusedError, SamplePrediction, TestPartition, produce_evaluation,
)
from evaluation.store import EvaluationStore
from presentation.contracts import frozen_test_split_hash

args = json.loads(sys.argv[1])
ids = (3, 7, 12, 20, 21, 30)
rows = [
    (3, "cat", "cat", 0.9, 0.1), (7, "cat", "cat", 0.8, 0.2), (12, "cat", "dog", 0.3, 0.7),
    (20, "dog", "dog", 0.25, 0.75), (21, "dog", "cat", 0.6, 0.4), (30, "dog", "dog", 0.1, 0.9),
]
rows = [row for row in rows if row[0] not in args.get("drop", [])]
samples = [
    SamplePrediction(crop_id=i, true_class=t, predicted_class=p,
                     probabilities={"cat": c, "dog": d})
    for i, t, p, c, d in rows
]
partition = TestPartition(
    manifest_hash="d" * 64, test_split_hash=frozen_test_split_hash(list(ids)), crop_ids=ids
)
store = EvaluationStore(create_engine(os.environ["DATABASE_URL"]))
try:
    record = produce_evaluation(
        store, samples, namespace=args["namespace"], model_run_id=args["model_run_id"],
        partition=partition,
        clock=lambda: datetime.fromisoformat("2026-10-01T14:00:00+00:00"),
    )
    print(json.dumps({"ok": record.namespace}))
except EvaluationRefusedError as error:
    print(json.dumps({"refused": error.reason, "detail": error.detail}))
"""

OUTCOME = {
    "reference": {
        "dataset_version": "v0.1.1",
        "manifest_hash": "d" * 64,
        "dvc_release_hash": "e" * 64,
    },
    "ranking": [],
    "excluded": [],
    "candidate": {
        "run_id": CANDIDATE,
        "campaign_row": 1,
        "start_time": "2026-10-01T10:00:00Z",
        "best_epoch": 2,
        "val_accuracy": 0.86,
        "val_macro_f1": 0.85,
        "val_loss": 0.38,
    },
    "campaign_rows": [1],
    "ready_to_close": True,
    "outcome_hash": "1" * 64,
}


def fail(message: str) -> None:
    print(f"::error::{message}")
    sys.exit(1)


def check(condition: bool, message: str) -> None:
    if not condition:
        fail(message)
    print(f"OK  {message}", flush=True)


def http(url: str) -> tuple[int, object]:
    try:
        with urllib.request.urlopen(f"{API}{url}", timeout=15) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"null")


def sql(query: str) -> list[list[str]]:
    command = f'mariadb -uroot -p"$MARIADB_ROOT_PASSWORD" image_repo -N -B -e "{query}"'
    result = subprocess.run(
        ["docker", "compose", "exec", "-T", "mariadb", "sh", "-c", command],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )
    print(f"SQL> {query}\n{result.stdout}", flush=True)
    return [line.split("\t") for line in result.stdout.splitlines()]


def produce(namespace: str, model_run_id: str = CANDIDATE, drop: list[int] | None = None):
    args = json.dumps({"namespace": namespace, "model_run_id": model_run_id, "drop": drop or []})
    result = subprocess.run(
        [*COMPOSE_RUN, "python", "-c", PRODUCER, args],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout, result.stderr, sep="\n", flush=True)
        fail(f"el productor terminó con código {result.returncode} ({args})")
    print(f"productor {args} → {result.stdout.strip()}", flush=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


def namespaces() -> list[str]:
    return [row[0] for row in sql("SELECT namespace FROM p3_evaluation ORDER BY namespace")]


def main() -> None:
    sql(
        "UPDATE p3_model_selection SET status='open', outcome=NULL, outcome_hash=NULL, "
        "proposed_at=NULL, closed_at=NULL WHERE id=1; DELETE FROM p3_evaluation"
    )

    # 1. Selección abierta: nada se produce ni se sirve.
    for namespace in ("synthetic", "official"):
        result = produce(namespace)
        check(result.get("refused") == "model_selection_open", f"{namespace}: bloqueado")
    check(namespaces() == [], "selección abierta: p3_evaluation sigue vacía")
    status, body = http("/evaluation")
    print(f"    respuesta: {json.dumps(body, ensure_ascii=False)}")
    check(status == 200 and body.get("state") == "blocked", "GET /evaluation → blocked")
    # Acceso directo prematuro a resultados del test: rechazado por la API, no solo por la UI.
    status, body = http("/evaluation/predictions")
    print(f"    respuesta: {json.dumps(body, ensure_ascii=False)}")
    check(status == 409, "GET /evaluation/predictions → 409 antes del cierre")

    # 2. Cierre sintético; datos incompatibles rechazados.
    outcome = json.dumps(OUTCOME).replace('"', '\\"')
    sql(
        f"UPDATE p3_model_selection SET status='closed', outcome='{outcome}', "
        f"outcome_hash='{'1' * 64}', proposed_at='2026-10-01 12:00:00.000', "
        "closed_at='2026-10-01 13:00:00.456' WHERE id=1"
    )
    result = produce("official", model_run_id="b" * 32)
    check(result.get("refused") == "not_selected_candidate", "otro run → not_selected_candidate")
    result = produce("synthetic", drop=[30])
    check(result.get("refused") == "crop_ids_not_test_split", "crop faltante → rechazado")
    check(namespaces() == [], "datos incompatibles: nada escrito")

    # 3. Corrida sintética válida.
    check(produce("synthetic") == {"ok": "synthetic"}, "corrida sintética guardada")
    check(namespaces() == ["synthetic"], "una sola fila, namespace synthetic")

    # 4. Aislamiento: la API oficial no la sirve.
    status, body = http("/evaluation")
    print(f"    respuesta: {json.dumps(body, ensure_ascii=False)}")
    check(
        status == 200
        and body.get("state") == "pending"
        and body.get("namespace") == "official"
        and "metrics" not in body
        and "confusion_matrix" not in body,
        "GET /evaluation → 200 pending official, sin resultados (la sintética no es oficial)",
    )
    status, _ = http("/evaluation/predictions?format=csv")
    check(status == 404, "GET /evaluation/predictions → 404")
    print("Recorrido de evaluación sintético completo.")


if __name__ == "__main__":
    main()
