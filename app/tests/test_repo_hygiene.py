"""D01-06: controles de higiene del repositorio (rúbrica 7.3).

Comprueban que Git excluya secretos, datos derivados, stores de MLflow y pesos de
modelos de P3, que no haya secretos ni artefactos grandes versionados y que el CI no
pueda esconder un fallo (`continue-on-error`, `|| true`, `CI OK` que no espere a todos).
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
CI_WORKFLOW = WORKFLOWS_DIR / "ci.yml"

# Ningún archivo versionado debería pasar de esto: los datos van por DVC y los pesos a
# MLflow/S3. El archivo versionado más grande del baseline ronda 230 KB.
MAX_TRACKED_BYTES = 1024 * 1024

WEIGHT_SUFFIXES = {".pt", ".pth", ".ckpt", ".onnx", ".h5", ".keras", ".safetensors"}

# Patrones armados por partes para que este archivo no se detecte a sí mismo.
SECRET_PATTERNS = {
    "AWS access key id": re.compile("AK" + r"IA[0-9A-Z]{16}"),
    "AWS temporary key id": re.compile("AS" + r"IA[0-9A-Z]{16}"),
    "llave privada PEM": re.compile("-----BEGIN [A-Z ]*" + "PRIVATE KEY-----"),
    "API key de Anthropic": re.compile("sk-" + r"ant-[A-Za-z0-9_-]{20,}"),
    "token de GitHub": re.compile("gh" + r"[pousr]_[A-Za-z0-9]{36,}"),
}


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _in_git_work_tree() -> bool:
    try:
        result = _git("rev-parse", "--is-inside-work-tree")
    except FileNotFoundError:
        return False
    return result.returncode == 0 and result.stdout.strip() == "true"


requires_git = pytest.mark.skipif(
    not _in_git_work_tree(), reason="se necesita un clon de Git (en CI siempre existe)"
)


def _tracked_files() -> list[Path]:
    result = _git("ls-files", "-z")
    assert result.returncode == 0, result.stderr
    return [REPO_ROOT / name for name in result.stdout.split("\0") if name]


def _is_ignored(relative_path: str) -> bool:
    # --no-index: evalúa las reglas aunque la ruta no exista ni esté en el índice.
    return _git("check-ignore", "--no-index", "-q", relative_path).returncode == 0


@requires_git
@pytest.mark.parametrize(
    "path",
    [
        # Secretos y credenciales locales.
        ".env",
        ".env.local",
        ".dvc/config.local",
        ".aws/credentials",
        "app/.aws/config",
        # Stores locales de MLflow (runs, métricas y checkpoints).
        "mlruns/0/meta.yaml",
        "mlartifacts/1/abc/artifacts/model.pt",
        "app/mlruns/0/meta.yaml",
        # Pesos y checkpoints en cualquier carpeta.
        "models/p3-cnn-classifier/1.0.0/model.pt",
        "app/checkpoints/best.pth",
        "app/outputs/epoch_03.ckpt",
        "export/model.onnx",
        "export/model.h5",
        "export/model.keras",
        "export/model.safetensors",
        # Datos derivados de P3 que se versionan con DVC, no con Git.
        "data/crops/000001_000017.png",
        # Entornos y dependencias.
        "app/.venv/pyvenv.cfg",
        "backend/node_modules/zod/package.json",
        "frontend/node_modules/react/package.json",
    ],
)
def test_secrets_data_and_weights_are_ignored(path: str) -> None:
    assert _is_ignored(path), f"{path} no está excluido por .gitignore"


@requires_git
@pytest.mark.parametrize(
    "path",
    [
        ".env.example",
        "app/pyproject.toml",
        "app/uv.lock",
        "backend/package-lock.json",
        "frontend/package-lock.json",
        "terraform/environments/dev/.terraform.lock.hcl",
        "data/raw/images.dvc",
        "data/raw/annotations.dvc",
    ],
)
def test_templates_lockfiles_and_dvc_pointers_stay_versioned(path: str) -> None:
    assert not _is_ignored(path), f"{path} debe versionarse y .gitignore lo excluye"
    assert (REPO_ROOT / path).is_file(), f"{path} no existe en el repositorio"


@requires_git
def test_no_model_weights_or_large_files_are_tracked() -> None:
    offenders = []
    for path in _tracked_files():
        if not path.is_file():
            continue
        if path.suffix.lower() in WEIGHT_SUFFIXES:
            offenders.append(f"{path.relative_to(REPO_ROOT)} (pesos de modelo)")
        elif path.stat().st_size > MAX_TRACKED_BYTES:
            offenders.append(f"{path.relative_to(REPO_ROOT)} ({path.stat().st_size} bytes)")
    assert not offenders, "Artefactos que no deben ir en Git: " + ", ".join(offenders)


@requires_git
def test_no_secrets_in_tracked_files() -> None:
    findings = []
    for path in _tracked_files():
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binarios: no pueden contener estas claves como texto
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                # Solo la ruta y el tipo: nunca imprimir el valor encontrado.
                findings.append(f"{path.relative_to(REPO_ROOT)}: {label}")
    assert not findings, "Posibles secretos versionados: " + "; ".join(findings)


def _load_workflow(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _workflow_files() -> list[Path]:
    return sorted([*WORKFLOWS_DIR.glob("*.yml"), *WORKFLOWS_DIR.glob("*.yaml")])


def test_ci_workflows_exist() -> None:
    assert CI_WORKFLOW.is_file(), "falta .github/workflows/ci.yml"


@pytest.mark.parametrize("workflow", _workflow_files(), ids=lambda p: p.name)
def test_workflows_do_not_hide_failures(workflow: Path) -> None:
    data = _load_workflow(workflow)
    problems = []
    for job_id, job in (data.get("jobs") or {}).items():
        if job.get("continue-on-error"):
            problems.append(f"job {job_id}: continue-on-error")
        for index, step in enumerate(job.get("steps") or []):
            name = step.get("name", f"paso {index}")
            if step.get("continue-on-error"):
                problems.append(f"{job_id}/{name}: continue-on-error")
            run = step.get("run") or ""
            if re.search(r"\|\|\s*(true|:)\b", run) or "set +e" in run:
                problems.append(f"{job_id}/{name}: ignora el código de salida")
    assert not problems, f"{workflow.name} oculta fallos: " + "; ".join(problems)


def test_ci_ok_waits_for_every_job_and_fails_on_any_failure() -> None:
    jobs = _load_workflow(CI_WORKFLOW)["jobs"]
    summary = jobs["ci-ok"]
    needs = summary.get("needs") or []
    needs = [needs] if isinstance(needs, str) else needs
    expected = sorted(job_id for job_id in jobs if job_id != "ci-ok")
    assert sorted(needs) == expected, "CI OK debe depender de todos los demás jobs"
    # Sin always(), un job fallido deja a CI OK en "skipped" y GitHub lo da por bueno.
    assert "always()" in str(summary.get("if", ""))


def test_ci_runs_the_required_checks() -> None:
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    for command in (
        "ruff check .",
        "ruff format --check .",
        "pytest -q",
        "npm run lint",
        "npm run typecheck",
        "npm test",
        "npm run build",
        "docker compose config --quiet",
    ):
        assert command in text, f"el CI no ejecuta `{command}`"
