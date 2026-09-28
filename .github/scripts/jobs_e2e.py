"""D02-05 — Prueba de punta a punta de jobs persistentes con servicios reales.

Requiere `docker compose` con mariadb, minio, mlflow, backend y trainer-worker arriba
(ver el job "Jobs persistentes" de .github/workflows/ci.yml). Solo biblioteca estándar.

Escenarios (todos con la TAREA CONTROLADA, que no entrena ni lee datos):
  1. TrainingConfig inválido → 400 y no se encola.
  2. Training real → 409 (release/manifest oficiales aún no disponibles en la API).
  3. Éxito: queued → running → succeeded, progreso N/N, logs y run de MLflow FINISHED.
  4. Fallo controlado en la época 3 → failed con error, no éxito; run FAILED.
  5. Cancelación en ejecución → cancel_requested → cancelled; run KILLED.
  6. SIGTERM al worker durante un job → failed "Interrumpido"; no se reejecuta.
  7. Worker matado (SIGKILL) y reiniciado → el job huérfano pasa a failed tras el
     latido vencido, conserva progreso y run (que se cierra FAILED) y no se reejecuta.
  8. `docker compose down` + `up` (sin borrar volúmenes) → jobs, estados, progreso,
     errores, run IDs y logs idénticos; un job nuevo sigue funcionando.

Sale con 1 al primer fallo. Imprime la evidencia (IDs y estados, sin secretos).
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API = "http://localhost:3100"
MLFLOW = "http://localhost:5000"
REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "contracts/p3/fixtures"
EVIDENCE: dict[str, object] = {}


def fixture(contract: str, name: str) -> dict:
    return json.loads((FIXTURES / contract / f"{name}.json").read_text())["payload"]


def fail(message: str) -> None:
    print(f"::error::{message}")
    print(json.dumps(EVIDENCE, indent=2, ensure_ascii=False))
    sys.exit(1)


def check(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def http(method: str, url: str, body: object | None = None) -> tuple[int, object]:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"null")


def compose(*args: str) -> None:
    print(f"$ docker compose {' '.join(args)}", flush=True)
    subprocess.run(["docker", "compose", *args], cwd=REPO, check=True)


def wait_for(description: str, predicate, timeout: float = 120, every: float = 1.0):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            last = predicate()
        except (urllib.error.URLError, ConnectionError, TimeoutError) as error:
            last = error
        if last and not isinstance(last, Exception):
            return last
        time.sleep(every)
    fail(f"Tiempo agotado esperando: {description} (último: {last})")
    return None


def get_job(job_id: int) -> dict:
    status, body = http("GET", f"{API}/training/jobs/{job_id}")
    check(status == 200, f"GET job {job_id} respondió {status}: {body}")
    return body  # type: ignore[return-value]


def job_in(job_id: int, *statuses: str):
    return lambda: (job := get_job(job_id))["status"] in statuses and job


def running_with_progress(job_id: int, epoch: int):
    def predicate():
        job = get_job(job_id)
        progress = job.get("progress") or {}
        return job["status"] == "running" and progress.get("epoch", 0) >= epoch and job

    return predicate


def logs(job_id: int) -> list[str]:
    status, body = http("GET", f"{API}/training/jobs/{job_id}/logs")
    check(status == 200, f"logs del job {job_id}: {status}")
    return [line["message"] for line in body["lines"]]  # type: ignore[index]


def create(max_epochs: int = 10, fail_at: int | None = None, task: str = "controlled"):
    request = fixture("create_training_job_request", "valid-ok")
    request["task"] = task
    request["config"] = dict(request["config"], max_epochs=max_epochs, seed=7)
    if fail_at is not None:
        request["controlled"] = {"fail_at_epoch": fail_at}
    return http("POST", f"{API}/training/jobs", request)


def mlflow_run(run_id: str) -> dict:
    status, body = http("GET", f"{MLFLOW}/api/2.0/mlflow/runs/get?run_id={run_id}")
    check(status == 200, f"MLflow no encuentra el run {run_id}: {status}")
    return body["run"]  # type: ignore[index]


def run_tags(run: dict) -> dict:
    return {tag["key"]: tag["value"] for tag in run["data"].get("tags", [])}


def main() -> None:
    wait_for("API del backend", lambda: http("GET", f"{API}/health")[0] == 200, timeout=180)

    # 1. Inválido: 400 y nada encolado.
    _, before = http("GET", f"{API}/training/jobs")
    request = fixture("create_training_job_request", "invalid-invalid-config")
    status, body = http("POST", f"{API}/training/jobs", request)
    check(status == 400, f"config inválido debía dar 400 y dio {status}: {body}")
    check("learning_rate" in body["error"], f"el 400 debía nombrar el campo: {body}")
    _, after = http("GET", f"{API}/training/jobs")
    check(len(after["jobs"]) == len(before["jobs"]), "un job inválido quedó encolado")
    EVIDENCE["invalid_config"] = {"status": status, "error": body["error"]}

    # 2. Training real: 409.
    status, body = create(task="training")
    check(status == 409, f"training real debía dar 409 y dio {status}: {body}")
    EVIDENCE["real_training"] = {"status": status, "error": body["error"]}

    # 3. Éxito.
    status, job = create(max_epochs=10)
    check(status == 201 and job["status"] == "queued", f"crear job: {status} {job}")
    check(job["mlflow_run_id"] is None, "un job en cola ya tenía run_id")
    ok_id = job["id"]
    done = wait_for(f"job {ok_id} succeeded", job_in(ok_id, "succeeded", "failed", "cancelled"))
    check(
        done["status"] == "succeeded", f"job {ok_id} terminó en {done['status']}: {done['error']}"
    )
    check(done["progress"] == {"epoch": 10, "total_epochs": 10}, f"progreso: {done['progress']}")
    check(any("Época 10/10" in line for line in logs(ok_id)), "faltan logs por época")
    run = mlflow_run(done["mlflow_run_id"])
    tags = run_tags(run)
    check(run["info"]["status"] == "FINISHED", f"run MLflow: {run['info']['status']}")
    check(tags.get("p3.run_kind") == "controlled_task", f"tags: {tags}")
    check(tags.get("p3.job_id") == str(ok_id), f"tags: {tags}")
    EVIDENCE["succeeded"] = {
        "job_id": ok_id,
        "mlflow_run_id": done["mlflow_run_id"],
        "progress": done["progress"],
        "run_status": run["info"]["status"],
        "config": done["config"],
    }

    # 4. Fallo controlado.
    _, job = create(max_epochs=10, fail_at=3)
    failed = wait_for("fallo controlado", job_in(job["id"], "succeeded", "failed", "cancelled"))
    check(failed["status"] == "failed", f"el fallo controlado terminó en {failed['status']}")
    check("Fallo controlado" in failed["error"], f"error: {failed['error']}")
    check(mlflow_run(failed["mlflow_run_id"])["info"]["status"] == "FAILED", "run no FAILED")
    EVIDENCE["failed"] = {"job_id": failed["id"], "error": failed["error"]}

    # 5. Cancelación en ejecución.
    _, job = create(max_epochs=30)
    wait_for("job en ejecución", running_with_progress(job["id"], 2))
    status, cancelling = http("POST", f"{API}/training/jobs/{job['id']}/cancel")
    check(
        status == 200 and cancelling["cancel_requested"] is True, f"cancel: {status} {cancelling}"
    )
    cancelled = wait_for("cancelado", job_in(job["id"], "cancelled", "succeeded", "failed"))
    check(cancelled["status"] == "cancelled", f"la cancelación terminó en {cancelled['status']}")
    check(mlflow_run(cancelled["mlflow_run_id"])["info"]["status"] == "KILLED", "run no KILLED")
    status, _ = http("POST", f"{API}/training/jobs/{job['id']}/cancel")
    check(status == 409, f"cancelar un job terminado debía dar 409 y dio {status}")
    EVIDENCE["cancelled"] = {"job_id": cancelled["id"], "progress": cancelled["progress"]}

    # 6. SIGTERM durante un job.
    _, job = create(max_epochs=60)
    wait_for("job en ejecución (SIGTERM)", running_with_progress(job["id"], 2))
    compose("stop", "trainer-worker")
    stopped = wait_for("interrumpido por SIGTERM", job_in(job["id"], "failed", "succeeded"), 30)
    check(stopped["status"] == "failed" and "SIGTERM" in stopped["error"], f"SIGTERM: {stopped}")
    compose("up", "-d", "trainer-worker")
    time.sleep(5)
    check(get_job(job["id"])["status"] == "failed", "el job interrumpido se reejecutó")
    EVIDENCE["sigterm"] = {"job_id": stopped["id"], "error": stopped["error"]}

    # 7. SIGKILL + reinicio: recuperación sin duplicar.
    _, job = create(max_epochs=60)
    running = wait_for("job en ejecución (SIGKILL)", running_with_progress(job["id"], 2))
    compose("kill", "-s", "SIGKILL", "trainer-worker")
    orphan = get_job(job["id"])
    check(orphan["status"] == "running", f"tras SIGKILL el job debía seguir running: {orphan}")
    compose("up", "-d", "trainer-worker")
    recovered = wait_for("job huérfano recuperado", job_in(job["id"], "failed", "succeeded"), 150)
    check(recovered["status"] == "failed", f"el huérfano terminó en {recovered['status']}")
    check("no se reintenta" in recovered["error"], f"error: {recovered['error']}")
    check(recovered["mlflow_run_id"] == running["mlflow_run_id"], "cambió el run del job")
    orphan_run = wait_for(
        "run huérfano cerrado en MLflow",
        lambda: (
            (run := mlflow_run(recovered["mlflow_run_id"]))["info"]["status"] == "FAILED" and run
        ),
        30,
    )
    check(
        recovered["progress"]["epoch"] >= running["progress"]["epoch"]
        and recovered["progress"]["epoch"] < 60,
        f"progreso inesperado tras recuperar: {recovered['progress']}",
    )
    EVIDENCE["sigkill_recovery"] = {
        "job_id": recovered["id"],
        "progress": recovered["progress"],
        "error": recovered["error"],
        "run_status": orphan_run["info"]["status"],
    }

    # 8. down + up: todo persiste.
    _, snapshot = http("GET", f"{API}/training/jobs")
    log_counts = {job["id"]: len(logs(job["id"])) for job in snapshot["jobs"]}
    compose("down")
    compose("up", "-d", "--wait", "--wait-timeout", "300", "mariadb", "minio", "mlflow")
    compose("up", "-d", "backend", "trainer-worker")
    wait_for("API tras reinicio", lambda: http("GET", f"{API}/health")[0] == 200, timeout=180)
    _, restored = http("GET", f"{API}/training/jobs")
    check(restored == snapshot, "los jobs cambiaron tras down/up")
    for job_id, count in log_counts.items():
        check(len(logs(job_id)) == count, f"los logs del job {job_id} cambiaron tras down/up")
    check(
        mlflow_run(EVIDENCE["succeeded"]["mlflow_run_id"])["info"]["status"] == "FINISHED",
        "el run del job exitoso no sobrevivió al reinicio",
    )
    _, job = create(max_epochs=10)
    final = wait_for("job después del reinicio", job_in(job["id"], "succeeded", "failed"))
    check(
        final["status"] == "succeeded",
        f"tras el reinicio un job nuevo terminó en {final['status']}",
    )
    EVIDENCE["after_restart"] = {
        "jobs_identical": True,
        "jobs": [(j["id"], j["status"], j["mlflow_run_id"]) for j in restored["jobs"]],
        "new_job": {"id": final["id"], "status": final["status"]},
    }

    print(json.dumps(EVIDENCE, indent=2, ensure_ascii=False))
    print("Jobs persistentes verificados con MariaDB, MLflow, backend y trainer-worker reales.")


if __name__ == "__main__":
    main()
