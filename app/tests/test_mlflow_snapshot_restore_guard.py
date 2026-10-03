"""#104 — el restore aislado es FAIL-CLOSED.

Dos niveles:
- `evaluate()` (decisión pura) con configuraciones Compose resueltas y recursos Docker;
- `restore.sh` completo con un `docker` falso que registra cada llamada: los casos
  rechazados salen con código 3 ANTES de cualquier operación que modifique Docker
  (up, exec, cp, mc…); los válidos sí llegan a ellas.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts/mlflow-snapshot"
SPEC = importlib.util.spec_from_file_location("restore_guard", SCRIPTS / "restore_guard.py")
guard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = guard
SPEC.loader.exec_module(guard)

SNAPSHOT_ID = "67ccb30fc1e855c234fb0a90df44f5aa.dir"
PRODUCER = "mlops-p03"
PROJECT = "mlops-p03-restore"
MARK = {guard.MARKER_LABEL: guard.MARKER_VALUE, guard.SNAPSHOT_LABEL: SNAPSHOT_ID}


def compose_config(project: str = PROJECT, snapshot: str = SNAPSHOT_ID) -> dict:
    """Forma de `docker compose -f base -f restore-isolated -p <p> config --format json`."""
    labels = {guard.MARKER_LABEL: guard.MARKER_VALUE, guard.SNAPSHOT_LABEL: snapshot}
    return {
        "name": project,
        "services": {
            "mariadb": {
                "volumes": [
                    {"type": "volume", "source": "mariadb_data", "target": "/var/lib/mysql"}
                ]
            },
            "minio": {"volumes": [{"type": "volume", "source": "minio_data", "target": "/data"}]},
        },
        "volumes": {
            key: {"name": f"{project}_{key}", "labels": dict(labels)}
            for key in ("mariadb_data", "minio_data", "torch_cache")
        },
    }


def facts(**overrides) -> guard.Facts:
    base = {
        "project": PROJECT,
        "snapshot_id": SNAPSHOT_ID,
        "producer_projects": {PRODUCER},
        "producer_volumes": {f"{PRODUCER}_mariadb_data", f"{PRODUCER}_minio_data"},
        "config": compose_config(),
    }
    base.update(overrides)
    return guard.Facts(**base)


def marked_volumes(snapshot: str = SNAPSHOT_ID) -> dict[str, dict[str, str]]:
    return {
        f"{PROJECT}_{key}": {
            guard.MARKER_LABEL: guard.MARKER_VALUE,
            guard.SNAPSHOT_LABEL: snapshot,
            guard.PROJECT_LABEL: PROJECT,
        }
        for key in ("mariadb_data", "minio_data")
    }


# --- decisión pura -----------------------------------------------------------------


def test_proyecto_nuevo_aislado_y_marcado_se_permite():
    assert guard.evaluate(facts()) == []


def test_project_igual_al_productor_falla():
    problems = guard.evaluate(facts(project=PRODUCER, config=compose_config(PRODUCER)))
    assert any("PRODUCTOR" in p for p in problems)


def test_productor_con_restore_en_el_nombre_tambien_falla():
    """No basta con que el nombre contenga 'restore': si es el productor, aborta."""
    problems = guard.evaluate(facts(producer_projects={PROJECT}))
    assert any("PRODUCTOR" in p for p in problems)


@pytest.mark.parametrize("name", ["", "Mlops-Restore", "otro-proyecto", "a b restore"])
def test_nombre_de_proyecto_invalido_falla(name):
    assert any("inválido" in p for p in guard.evaluate(facts(project=name)))


def test_volumen_del_productor_falla():
    config = compose_config()
    config["volumes"]["mariadb_data"]["name"] = f"{PRODUCER}_mariadb_data"
    problems = guard.evaluate(facts(config=config))
    assert any("volumen del productor" in p for p in problems)


def test_volumen_montado_por_contenedores_de_otro_proyecto_falla():
    problems = guard.evaluate(facts(volume_users={f"{PROJECT}_mariadb_data": {PRODUCER}}))
    assert any("lo montan contenedores" in p and PRODUCER in p for p in problems)


def test_volumen_external_falla():
    config = compose_config()
    config["volumes"]["minio_data"]["external"] = True
    assert any("external" in p for p in guard.evaluate(facts(config=config)))


def test_volumen_con_nombre_fijo_falla():
    config = compose_config()
    config["volumes"]["minio_data"]["name"] = "mlflow-minio-compartido"
    assert any("sin name: fijo" in p for p in guard.evaluate(facts(config=config)))


def test_bind_mount_en_lugar_de_volumen_falla():
    config = compose_config()
    config["services"]["mariadb"]["volumes"] = [
        {
            "type": "bind",
            "source": "/var/lib/docker/volumes/mlops-p03_mariadb_data/_data",
            "target": "/var/lib/mysql",
        }
    ]
    assert any("se exige un volumen" in p for p in guard.evaluate(facts(config=config)))


def test_configuracion_sin_marca_falla():
    config = compose_config()
    del config["volumes"]["mariadb_data"]["labels"]
    assert any("no marca" in p for p in guard.evaluate(facts(config=config)))


def test_configuracion_de_otro_proyecto_falla():
    assert any(
        "configuración resuelta" in p
        for p in guard.evaluate(facts(config=compose_config("otro-restore")))
    )


def test_recursos_existentes_sin_marca_fallan():
    volumes = marked_volumes()
    for labels in volumes.values():
        del labels[guard.MARKER_LABEL]
    problems = guard.evaluate(facts(containers=["abc123"], volumes=volumes))
    assert any("sin marca" in p for p in problems)


def test_marca_de_otro_snapshot_falla():
    problems = guard.evaluate(
        facts(containers=["abc123"], volumes=marked_volumes("0" * 32 + ".dir"))
    )
    assert any("marca incorrecta" in p for p in problems)


def test_contenedores_sin_sus_volumenes_son_ambiguos():
    problems = guard.evaluate(facts(containers=["abc123"], networks=[f"{PROJECT}_default"]))
    assert any("ambiguos" in p for p in problems)


def test_volumen_del_proyecto_etiquetado_por_otro_proyecto_falla():
    volumes = marked_volumes()
    volumes[f"{PROJECT}_mariadb_data"][guard.PROJECT_LABEL] = PRODUCER
    assert any("pertenece al proyecto" in p for p in guard.evaluate(facts(volumes=volumes)))


def test_segundo_restore_marcado_con_el_mismo_snapshot_se_permite():
    assert (
        guard.evaluate(
            facts(
                containers=["abc123"],
                networks=[f"{PROJECT}_default"],
                volumes=marked_volumes(),
                volume_users={name: {PROJECT} for name in marked_volumes()},
            )
        )
        == []
    )


def test_snapshot_incompleto_o_alterado_falla(tmp_path):
    snapshot = make_snapshot(tmp_path / "snap")
    assert guard.check_snapshot(snapshot) == []
    (snapshot / "db/mlflow.sql").write_text("alterado", encoding="utf-8")
    assert any("alterado" in p for p in guard.check_snapshot(snapshot))
    shutil.rmtree(snapshot / "artifacts")
    assert any("artifacts" in p for p in guard.check_snapshot(snapshot))
    (snapshot / "manifest.json").unlink()
    assert any("manifest.json" in p for p in guard.check_snapshot(snapshot))


# --- restore.sh completo con docker falso -------------------------------------------

FAKE_DOCKER = textwrap.dedent(
    """\
    #!{python}
    import json, os, sys
    args = sys.argv[1:]
    state = json.load(open(os.environ["FAKE_DOCKER_STATE"]))
    with open(os.environ["FAKE_DOCKER_LOG"], "a") as log:
        log.write(" ".join(args) + "\\n")

    def opt(name):
        return args[args.index(name) + 1] if name in args else None

    if args[0] == "compose" and "config" in args:
        print(json.dumps(state["restore_config"] if "-p" in args else state["producer_config"]))
    elif args[:2] == ["ps", "-a"] and (opt("--filter") or "").startswith("volume="):
        name = opt("--filter").split("=", 1)[1]
        print("\\n".join(state.get("volume_users", {{}}).get(name, [])))
    elif args[:2] == ["ps", "-a"]:
        print("\\n".join(state.get("containers", [])))
    elif args[:2] == ["network", "ls"]:
        print("\\n".join(state.get("networks", [])))
    elif args[:2] == ["volume", "ls"]:
        volumes = state.get("volumes", {{}})
        label = opt("--filter")
        if label:
            key, value = label.split("=", 1)[1].split("=", 1)
            volumes = {{n: l for n, l in volumes.items() if l.get(key) == value}}
        print("\\n".join(volumes))
    elif args[:2] == ["volume", "inspect"]:
        print(json.dumps(state["volumes"][args[2]]))
    else:
        with open(os.environ["FAKE_DOCKER_LOG"], "a") as log:
            log.write("MUTATION " + " ".join(args) + "\\n")
    """
)

MUTATING = ("MUTATION",)


def make_snapshot(path: Path) -> Path:
    (path / "db").mkdir(parents=True)
    (path / "artifacts/mlflow-artifacts/1").mkdir(parents=True)
    files = {}
    for name, body in (
        ("db/mlflow.sql", "CREATE DATABASE mlflow;\n"),
        ("db/image_repo_p3_model_selection.sql", "-- p3_model_selection\n"),
        ("db/image_repo_training_jobs.sql", "-- training_jobs\n"),
    ):
        (path / name).write_text(body, encoding="utf-8")
        files[name] = {"sha256": hashlib.sha256(body.encode()).hexdigest()}
    (path / "manifest.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    return path


@pytest.fixture
def sandbox(tmp_path):
    """Copia mínima del repo: scripts, compose, puntero DVC, .env falso y snapshot."""
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copytree(SCRIPTS, repo / "scripts/mlflow-snapshot")
    for name in ("docker-compose.yml", "docker-compose.restore-isolated.yml"):
        shutil.copy(REPO / name, repo / name)
    (repo / "data").mkdir()
    shutil.copy(REPO / "data/mlflow_snapshot.dvc", repo / "data/mlflow_snapshot.dvc")
    (repo / ".env").write_text(
        "MARIADB_ROOT_PASSWORD=fake\nMINIO_ROOT_USER=fake\nMINIO_ROOT_PASSWORD=fakefake\n",
        encoding="utf-8",
    )
    make_snapshot(repo / "data/mlflow_snapshot")
    docker = tmp_path / "docker"
    docker.write_text(FAKE_DOCKER.format(python=sys.executable), encoding="utf-8")
    docker.chmod(docker.stat().st_mode | stat.S_IEXEC)
    return repo, docker, tmp_path


def run_restore(sandbox, state: dict, project: str = PROJECT, extra_env: dict | None = None):
    repo, docker, tmp = sandbox
    state = {
        "producer_config": {
            "name": PRODUCER,
            "volumes": {"mariadb_data": {"name": f"{PRODUCER}_mariadb_data"}},
        },
        "restore_config": compose_config(project),
        **state,
    }
    (tmp / "state.json").write_text(json.dumps(state), encoding="utf-8")
    log = tmp / "docker.log"
    log.write_text("", encoding="utf-8")
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith(("MARIADB", "MINIO"))},
        "DOCKER": str(docker),
        "PYTHON": sys.executable,
        "FAKE_DOCKER_STATE": str(tmp / "state.json"),
        "FAKE_DOCKER_LOG": str(log),
        **(extra_env or {}),
    }
    result = subprocess.run(
        ["bash", str(repo / "scripts/mlflow-snapshot/restore.sh"), "data/mlflow_snapshot", project],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    calls = log.read_text(encoding="utf-8").splitlines()
    return result, calls


def assert_aborted_before_mutation(result, calls):
    assert result.returncode == guard.REFUSED, result.stdout + result.stderr
    assert "RESTORE GUARD: ABORT" in result.stderr
    assert not [c for c in calls if c.startswith(MUTATING)], calls
    assert "[1/6]" not in result.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="requiere bash")
class TestRestoreSh:
    def test_proyecto_nuevo_llega_a_las_operaciones(self, sandbox):
        result, calls = run_restore(sandbox, {})
        assert result.returncode == 0, result.stderr
        assert "RESTORE GUARD: OK" in result.stdout
        mutations = [c for c in calls if c.startswith(MUTATING)]
        assert mutations[0].startswith(
            "MUTATION compose -f docker-compose.yml -f docker-compose.restore-isolated.yml "
            f"-p {PROJECT} up"
        )
        # la guarda corrió antes que la primera mutación
        assert calls.index(mutations[0]) > max(i for i, c in enumerate(calls) if "config" in c)

    def test_project_del_productor_aborta_antes_de_mutar(self, sandbox):
        assert_aborted_before_mutation(
            *run_restore(sandbox, {}, extra_env={"P3_PRODUCER_PROJECT": PROJECT})
        )

    def test_volumen_compartido_con_el_productor_aborta(self, sandbox):
        config = compose_config()
        config["volumes"]["minio_data"]["name"] = f"{PRODUCER}_minio_data"
        assert_aborted_before_mutation(*run_restore(sandbox, {"restore_config": config}))

    def test_volumen_external_aborta(self, sandbox):
        config = compose_config()
        config["volumes"]["mariadb_data"]["external"] = True
        assert_aborted_before_mutation(*run_restore(sandbox, {"restore_config": config}))

    def test_recursos_sin_marca_abortan(self, sandbox):
        volumes = {f"{PROJECT}_mariadb_data": {guard.PROJECT_LABEL: PROJECT}}
        assert_aborted_before_mutation(
            *run_restore(sandbox, {"containers": ["c1"], "volumes": volumes})
        )

    def test_marca_incorrecta_aborta(self, sandbox):
        volumes = marked_volumes("f" * 32 + ".dir")
        assert_aborted_before_mutation(
            *run_restore(sandbox, {"containers": ["c1"], "volumes": volumes})
        )

    def test_snapshot_alterado_aborta(self, sandbox):
        repo = sandbox[0]
        (repo / "data/mlflow_snapshot/db/mlflow.sql").write_text(
            "DROP DATABASE x;", encoding="utf-8"
        )
        assert_aborted_before_mutation(*run_restore(sandbox, {}))

    def test_segundo_restore_marcado_se_permite(self, sandbox):
        state = {
            "containers": ["c1"],
            "networks": [f"{PROJECT}_default"],
            "volumes": marked_volumes(),
            "volume_users": {name: [PROJECT] for name in marked_volumes()},
        }
        result, calls = run_restore(sandbox, state)
        assert result.returncode == 0, result.stderr
        assert "segundo restore" in result.stdout
        assert any(c.startswith("MUTATION") for c in calls)


@pytest.mark.skipif(shutil.which("docker") is None, reason="requiere docker compose")
def test_el_override_real_resuelve_una_configuracion_aislada_y_marcada():
    """La configuración que de verdad resuelve Compose pasa la guarda; sin
    P3_RESTORE_SNAPSHOT_ID ni siquiera resuelve."""
    env = {
        **os.environ,
        "MARIADB_ROOT_PASSWORD": "x",
        "MINIO_ROOT_USER": "x",
        "MINIO_ROOT_PASSWORD": "xxxxxxxx",
        "P3_RESTORE_SNAPSHOT_ID": SNAPSHOT_ID,
    }
    cmd = [
        "docker",
        "compose",
        "--project-directory",
        str(REPO),
        "-f",
        str(REPO / "docker-compose.yml"),
        "-f",
        str(REPO / "docker-compose.restore-isolated.yml"),
        "-p",
        PROJECT,
        "config",
        "--format",
        "json",
    ]
    resolved = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    if resolved.returncode != 0 and "unknown flag" in resolved.stderr:
        pytest.skip("docker sin el plugin compose v2")
    assert resolved.returncode == 0, resolved.stderr
    assert guard.evaluate(facts(config=json.loads(resolved.stdout))) == []
    env.pop("P3_RESTORE_SNAPSHOT_ID")
    unresolved = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    assert unresolved.returncode != 0
    assert "P3_RESTORE_SNAPSHOT_ID" in unresolved.stderr


def test_no_hay_llamadas_a_docker_antes_de_la_guarda_en_restore_sh():
    """Ninguna línea ejecutable anterior a la guarda llama a docker."""
    lines = [
        line
        for line in (SCRIPTS / "restore.sh").read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    ]
    calls = [i for i, line in enumerate(lines) if "restore_guard.py" in line]
    assert len(calls) == 1
    before = lines[: calls[0]]
    assert not [
        line for line in before if "$DOCKER" in line or "$COMPOSE" in line or "docker " in line
    ]
    assert any("$COMPOSE" in line for line in lines[calls[0] :])
