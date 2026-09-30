"""D02-05 — Prueba de punta a punta de jobs persistentes con servicios reales.

Requiere `docker compose` con mariadb, minio, mlflow, backend y trainer-worker arriba
(ver el job "Jobs persistentes" de .github/workflows/ci.yml). Solo biblioteca estándar.

Escenarios (todos con la TAREA CONTROLADA, que no entrena ni lee datos):
  1. TrainingConfig inválido → 400 y no se encola.
  2. D03-03: trainer-worker publica las fuentes reales verificadas. En CI no hay datos
     (sin `dvc pull`) ni manifest congelado: GET /releases → 200 sin aprobados (v0.1.1
     rechazado con motivo), GET /manifest → 503 con el motivo, y training real → 409.
  3. Éxito: queued → running → succeeded, progreso N/N, logs y run de MLflow FINISHED.
  4. Fallo controlado en la época 3 → failed con error, no éxito; run FAILED.
  5. Cancelación en ejecución → cancel_requested → cancelled; run KILLED.
  6. SIGTERM al worker durante un job → failed "Interrumpido"; no se reejecuta.
  7. Worker matado (SIGKILL) y reiniciado CON MLflow CAÍDO → el job huérfano pasa a failed
     tras el latido vencido, conserva progreso y run y deja el cierre pendiente
     (mlflow_close_status=FAILED en MariaDB); al volver MLflow el mismo run queda FAILED
     y el pendiente se borra. No se reejecuta.
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


def sql(query: str) -> list[list[str]]:
    """Consulta directa a MariaDB (evidencia "antes y después"); la contraseña la pone el
    propio contenedor desde su entorno, no pasa por este script."""
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


JOBS_QUERY = (
    "SELECT id, task, status, progress_epoch, total_epochs, mlflow_run_id, "
    "IFNULL(mlflow_close_status, 'NULL'), cancel_requested FROM training_jobs ORDER BY id"
)


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


MLFLOW_TERMINAL = ("FINISHED", "FAILED", "KILLED")


def wait_run_status(run_id: str, expected: str, timeout: float = 60) -> dict:
    """Espera acotada a que ESE run de MLflow llegue a `expected`.

    El worker escribe primero el estado terminal en MariaDB y después cierra el run en
    MLflow, así que justo después de ver el job terminado el run puede seguir RUNNING un
    instante. RUNNING nunca se acepta: si no llega a `expected` antes del timeout, falla;
    si llega a OTRO estado terminal, falla de inmediato.
    """
    deadline = time.monotonic() + timeout
    last = "sin respuesta"
    while time.monotonic() < deadline:
        try:
            run = mlflow_run(run_id)
        except (urllib.error.URLError, ConnectionError, TimeoutError) as error:
            last = f"error {type(error).__name__}"
        else:
            last = run["info"]["status"]
            if last == expected:
                return run
            if last in MLFLOW_TERMINAL:
                fail(f"el run {run_id} terminó en {last}, se esperaba {expected}")
        time.sleep(1)
    fail(f"Tiempo agotado ({timeout:.0f}s): el run {run_id} sigue en {last} ≠ {expected}")
    return {}


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

    # 2. Fuentes reales publicadas por el worker (D03-03) y training real cerrado: 409.
    def manifest_published():
        status, body = http("GET", f"{API}/manifest")
        return status == 503 and "todavía no publicó" not in body.get("error", "")

    wait_for("snapshot de fuentes publicado por trainer-worker", manifest_published, timeout=180)
    status, manifest = http("GET", f"{API}/manifest")
    check("manifest_missing" in manifest["error"], f"el 503 debía dar el motivo: {manifest}")
    status, releases = http("GET", f"{API}/releases")
    check(status == 200, f"GET /releases debía dar 200 y dio {status}: {releases}")
    check(releases["approved"] == [], f"sin datos no puede haber releases aprobados: {releases}")
    rejected = {r["dataset_version"]: r["reason"] for r in releases["rejected"]}
    check(rejected.get("v0.1.1") == "data_missing", f"v0.1.1 debía estar data_missing: {rejected}")
    EVIDENCE["sources"] = {"manifest": manifest["error"], "rejected": rejected}

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
    run = wait_run_status(done["mlflow_run_id"], "FINISHED")
    tags = run_tags(run)
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
    wait_run_status(failed["mlflow_run_id"], "FAILED")
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
    wait_run_status(cancelled["mlflow_run_id"], "KILLED")
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

    # 7. SIGKILL + reinicio con MLflow caído: recuperación sin duplicar y cierre pendiente.
    _, job = create(max_epochs=60)
    running = wait_for("job en ejecución (SIGKILL)", running_with_progress(job["id"], 2))
    compose("kill", "-s", "SIGKILL", "trainer-worker")
    orphan = get_job(job["id"])
    check(orphan["status"] == "running", f"tras SIGKILL el job debía seguir running: {orphan}")
    compose("stop", "mlflow")
    compose("up", "-d", "--no-deps", "trainer-worker")
    recovered = wait_for("job huérfano recuperado", job_in(job["id"], "failed", "succeeded"), 150)
    check(recovered["status"] == "failed", f"el huérfano terminó en {recovered['status']}")
    check("no se reintenta" in recovered["error"], f"error: {recovered['error']}")
    check(recovered["mlflow_run_id"] == running["mlflow_run_id"], "cambió el run del job")
    check(
        recovered["progress"]["epoch"] >= running["progress"]["epoch"]
        and recovered["progress"]["epoch"] < 60,
        f"progreso inesperado tras recuperar: {recovered['progress']}",
    )
    pending = sql(f"SELECT status, mlflow_close_status FROM training_jobs WHERE id = {job['id']}")
    check(pending == [["failed", "FAILED"]], f"el cierre del run debía quedar pendiente: {pending}")
    compose("up", "-d", "--wait", "--wait-timeout", "300", "mlflow")
    orphan_run = wait_for(
        "run huérfano cerrado en MLflow al volver",
        lambda: (
            (run := mlflow_run(recovered["mlflow_run_id"]))["info"]["status"] == "FAILED" and run
        ),
        150,
    )
    closed = sql(
        f"SELECT IFNULL(mlflow_close_status, 'NULL') FROM training_jobs WHERE id = {job['id']}"
    )
    check(closed == [["NULL"]], f"el pendiente debía borrarse: {closed}")
    check(get_job(job["id"]) == recovered, "el job cambió al cerrar el run (¿se reejecutó?)")
    EVIDENCE["sigkill_recovery_mlflow_down"] = {
        "job_id": recovered["id"],
        "progress": recovered["progress"],
        "error": recovered["error"],
        "mlflow_close_status_while_down": pending[0][1],
        "run_status_after_mlflow_back": orphan_run["info"]["status"],
    }

    # 8. down + up: todo persiste (API y consulta directa a MariaDB, antes y después).
    _, snapshot = http("GET", f"{API}/training/jobs")
    rows_before = sql(JOBS_QUERY)
    log_counts = {job["id"]: len(logs(job["id"])) for job in snapshot["jobs"]}
    compose("down")
    compose("up", "-d", "--wait", "--wait-timeout", "300", "mariadb", "minio", "mlflow")
    compose("up", "-d", "backend", "trainer-worker")
    wait_for("API tras reinicio", lambda: http("GET", f"{API}/health")[0] == 200, timeout=180)
    _, restored = http("GET", f"{API}/training/jobs")
    check(restored == snapshot, "los jobs cambiaron tras down/up")
    check(sql(JOBS_QUERY) == rows_before, "las filas de training_jobs cambiaron tras down/up")
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
        "sql_rows": rows_before,
        "jobs": [(j["id"], j["status"], j["mlflow_run_id"]) for j in restored["jobs"]],
        "new_job": {"id": final["id"], "status": final["status"]},
    }

    print(json.dumps(EVIDENCE, indent=2, ensure_ascii=False))
    print("Jobs persistentes verificados con MariaDB, MLflow, backend y trainer-worker reales.")


if __name__ == "__main__":
    main()
