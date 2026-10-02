#!/usr/bin/env python3
"""#104 — guarda FAIL-CLOSED del restore aislado. Solo LECTURA.

restore.sh la ejecuta antes de cualquier operación que modifique Docker (up, exec,
cp, mc). Si encuentra un problema sale con código 3 y el restore aborta sin tocar nada.

Comprueba, sin confiar en que el nombre por defecto sea distinto:

1. Proyecto de restore: nombre válido, contiene "restore" y NO es el proyecto del
   productor. El productor se resuelve con `docker compose config` del compose base (con
   y sin COMPOSE_PROJECT_NAME del entorno), más P3_PRODUCER_PROJECT si se define.
2. Configuración Compose RESUELTA del restore: MariaDB (/var/lib/mysql) y MinIO (/data)
   montan volúmenes con nombre `<proyecto>_<volumen>`, no bind mounts, no `external`, no
   un `name:` fijo, ninguno de los volúmenes del productor, y con la marca del restore.
3. Recursos Docker existentes (label com.docker.compose.project): si el proyecto ya
   tiene contenedores, redes o volúmenes, solo se permite continuar si TODOS sus volúmenes
   llevan la marca `org.p3.restore.marker` y el mismo snapshot esperado. Sin marca, con
   otra marca o con recursos ambiguos (contenedores sin sus volúmenes) → aborta.
4. Ningún contenedor de OTRO proyecto monta los volúmenes del restore.
5. El snapshot local está completo y coincide con su manifest.json (SHA-256 de los dumps).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REFUSED = 3
MARKER_LABEL = "org.p3.restore.marker"
MARKER_VALUE = "mlflow-snapshot-restore"
SNAPSHOT_LABEL = "org.p3.restore.snapshot"
PROJECT_LABEL = "com.docker.compose.project"
PROJECT_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
# Servicio → ruta del almacenamiento que el restore escribe.
DATA_MOUNTS = {"mariadb": "/var/lib/mysql", "minio": "/data"}
OVERRIDE = "docker-compose.restore-isolated.yml"
SNAPSHOT_FILES = ("db/mlflow.sql", "db/image_repo_p3_model_selection.sql")


@dataclass
class Facts:
    """Todo lo que la guarda sabe del entorno; se junta con comandos de solo lectura."""

    project: str
    snapshot_id: str
    producer_projects: set[str]
    producer_volumes: set[str]
    config: dict
    containers: list[str] = field(default_factory=list)
    networks: list[str] = field(default_factory=list)
    # nombre → labels, de los volúmenes con label del proyecto o con el nombre esperado
    volumes: dict[str, dict[str, str]] = field(default_factory=dict)
    # nombre de volumen del restore → proyectos de los contenedores que lo montan
    volume_users: dict[str, set[str]] = field(default_factory=dict)
    snapshot_problems: list[str] = field(default_factory=list)


def evaluate(facts: Facts) -> list[str]:
    """Decisión pura: lista vacía ⇒ se puede restaurar."""
    problems: list[str] = list(facts.snapshot_problems)
    project = facts.project

    if not PROJECT_RE.match(project) or "restore" not in project:
        problems.append(
            f"proyecto {project!r} inválido: minúsculas/dígitos/-/_ y debe contener 'restore'"
        )
    if project.lower() in {p.lower() for p in facts.producer_projects}:
        problems.append(f"el proyecto {project!r} es el del PRODUCTOR: nunca se restaura ahí")
    if facts.config.get("name") != project:
        problems.append(
            f"la configuración resuelta es del proyecto {facts.config.get('name')!r}, "
            f"no de {project!r}"
        )

    expected_volumes: dict[str, str] = {}
    services = facts.config.get("services", {})
    declared = facts.config.get("volumes", {}) or {}
    for service, target in DATA_MOUNTS.items():
        mounts = [
            m for m in services.get(service, {}).get("volumes", []) if m.get("target") == target
        ]
        if len(mounts) != 1:
            problems.append(f"{service}: no hay exactamente un montaje en {target}")
            continue
        mount = mounts[0]
        if mount.get("type") != "volume":
            problems.append(f"{service}: {target} es {mount.get('type')!r}, se exige un volumen")
            continue
        key = mount.get("source", "")
        definition = declared.get(key)
        if definition is None:
            problems.append(f"{service}: el volumen {key!r} no está declarado")
            continue
        name = definition.get("name", "")
        if definition.get("external"):
            problems.append(f"{service}: el volumen {key!r} es external (podría ser del productor)")
        if name != f"{project}_{key}":
            problems.append(
                f"{service}: el volumen {key!r} se llama {name!r}, se exige {project}_{key} "
                "(sin name: fijo)"
            )
        if name in facts.producer_volumes:
            problems.append(f"{service}: {name!r} es un volumen del productor")
        labels = definition.get("labels") or {}
        if (
            labels.get(MARKER_LABEL) != MARKER_VALUE
            or labels.get(SNAPSHOT_LABEL) != facts.snapshot_id
        ):
            problems.append(
                f"{service}: la configuración no marca {name!r} con el snapshot esperado"
            )
        expected_volumes[service] = name

    for name, users in facts.volume_users.items():
        others = sorted(u for u in users if u != project)
        if others:
            problems.append(f"el volumen {name!r} lo montan contenedores de {others}")

    existing = facts.containers or facts.networks or facts.volumes
    if existing:
        for name, labels in sorted(facts.volumes.items()):
            if labels.get(MARKER_LABEL) != MARKER_VALUE:
                problems.append(f"recurso existente sin marca de restore: volumen {name!r}")
            elif labels.get(SNAPSHOT_LABEL) != facts.snapshot_id:
                problems.append(
                    f"marca incorrecta en {name!r}: snapshot {labels.get(SNAPSHOT_LABEL)!r}, "
                    f"se esperaba {facts.snapshot_id!r}"
                )
            if labels.get(PROJECT_LABEL) not in (None, project):
                problems.append(f"{name!r} pertenece al proyecto {labels.get(PROJECT_LABEL)!r}")
        for service, name in expected_volumes.items():
            if name not in facts.volumes:
                problems.append(
                    f"recursos ambiguos: el proyecto {project!r} ya existe pero falta el "
                    f"volumen marcado de {service} ({name!r})"
                )
    return problems


# --- recolección (solo lectura) ----------------------------------------------------


def _run(docker: str, *args: str, env: dict | None = None) -> str:
    result = subprocess.run([docker, *args], capture_output=True, text=True, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args[:3])}… falló: {result.stderr.strip()[:300]}")
    return result.stdout


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def check_snapshot(snapshot_dir: Path) -> list[str]:
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.is_file():
        return [f"snapshot incompleto: falta {manifest_path} (¿dvc pull -r prod?)"]
    problems = []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in SNAPSHOT_FILES:
        path = snapshot_dir / name
        want = manifest.get("files", {}).get(name, {}).get("sha256")
        if not path.is_file():
            problems.append(f"snapshot incompleto: falta {name}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != want:
            problems.append(f"snapshot alterado: {name} no coincide con manifest.json")
    if not (snapshot_dir / "artifacts/mlflow-artifacts").is_dir():
        problems.append("snapshot incompleto: falta artifacts/mlflow-artifacts")
    return problems


def collect(project: str, repo: Path, snapshot_dir: Path, snapshot_id: str, docker: str) -> Facts:
    base = ["compose", "--project-directory", str(repo), "-f", str(repo / "docker-compose.yml")]
    producers: set[str] = set()
    producer_volumes: set[str] = set()
    env_without = {k: v for k, v in os.environ.items() if k != "COMPOSE_PROJECT_NAME"}
    for env in (None, env_without):
        producer = json.loads(_run(docker, *base, "config", "--format", "json", env=env))
        producers.add(producer["name"])
        producer_volumes |= {
            v.get("name", f"{producer['name']}_{k}")
            for k, v in (producer.get("volumes") or {}).items()
        }
    producers |= {p for p in os.environ.get("P3_PRODUCER_PROJECT", "").split(",") if p}
    producer_volumes |= {f"{p}_{k}" for p in producers for k in ("mariadb_data", "minio_data")}

    restore_env = {**os.environ, "P3_RESTORE_SNAPSHOT_ID": snapshot_id}
    config = json.loads(
        _run(
            docker,
            *base,
            "-f",
            str(repo / OVERRIDE),
            "-p",
            project,
            "config",
            "--format",
            "json",
            env=restore_env,
        )
    )
    label = f"label={PROJECT_LABEL}={project}"
    containers = _lines(_run(docker, "ps", "-a", "--filter", label, "--format", "{{.ID}}"))
    networks = _lines(_run(docker, "network", "ls", "--filter", label, "--format", "{{.Name}}"))
    names = set(_lines(_run(docker, "volume", "ls", "--filter", label, "--format", "{{.Name}}")))
    data_names = {
        (config.get("volumes") or {}).get(m.get("source"), {}).get("name")
        for service, target in DATA_MOUNTS.items()
        for m in config.get("services", {}).get(service, {}).get("volumes", [])
        if m.get("target") == target
    } - {None}
    all_volumes = set(_lines(_run(docker, "volume", "ls", "--format", "{{.Name}}")))
    names |= data_names & all_volumes
    volumes = {
        name: json.loads(_run(docker, "volume", "inspect", name, "--format", "{{json .Labels}}"))
        or {}
        for name in sorted(names)
    }
    users = {
        name: set(
            _lines(
                _run(
                    docker,
                    "ps",
                    "-a",
                    "--filter",
                    f"volume={name}",
                    "--format",
                    '{{.Label "com.docker.compose.project"}}',
                )
            )
        )
        for name in sorted(data_names & all_volumes)
    }
    return Facts(
        project=project,
        snapshot_id=snapshot_id,
        producer_projects=producers,
        producer_volumes=producer_volumes,
        config=config,
        containers=containers,
        networks=networks,
        volumes=volumes,
        volume_users=users,
        snapshot_problems=check_snapshot(snapshot_dir),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="#104 — guarda fail-closed del restore aislado")
    parser.add_argument("--project", required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--snapshot-id", required=True)
    parser.add_argument("--docker", default=os.environ.get("DOCKER", "docker"))
    args = parser.parse_args(argv)

    if not re.fullmatch(r"[0-9a-f]{32}\.dir", args.snapshot_id):
        print(f"RESTORE GUARD: ABORT — snapshot id {args.snapshot_id!r} inválido", file=sys.stderr)
        return REFUSED
    try:
        facts = collect(args.project, args.repo, args.snapshot_dir, args.snapshot_id, args.docker)
    except (RuntimeError, OSError, ValueError, KeyError) as error:
        print(f"RESTORE GUARD: ABORT — no se pudo comprobar el entorno: {error}", file=sys.stderr)
        return REFUSED
    problems = evaluate(facts)
    if problems:
        print("RESTORE GUARD: ABORT — no se toca nada:", file=sys.stderr)
        for problem in problems:
            print(f" - {problem}", file=sys.stderr)
        return REFUSED
    state = "segundo restore sobre entorno marcado" if facts.volumes else "proyecto nuevo"
    print(
        f"RESTORE GUARD: OK — proyecto {facts.project} ({state}); productor(es) "
        f"{sorted(facts.producer_projects)} excluido(s); snapshot {facts.snapshot_id}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
