"""D01-06 (review PR #39): lo que está en el índice de Git, no solo lo que .gitignore dice.

`.gitignore` no impide `git add -f`. Estos tests revisan `git ls-files` y fallan si
se versiona algo local o sensible (`.env`, `.aws/`, `.dvc/config.local`), contenido
de stores de MLflow, crops, datos gestionados por DVC, entornos o estado de Terraform,
manteniendo las excepciones legítimas (`.env.example`, lockfiles, punteros `.dvc`).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tests._repo_hygiene_rules import forbidden_tracked_paths, tracked_paths

REPO_ROOT = Path(__file__).resolve().parents[2]
HAS_GIT = shutil.which("git") is not None


def _in_git_work_tree(path: Path) -> bool:
    if not HAS_GIT:
        return False
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


@pytest.mark.parametrize(
    ("path", "reason_fragment"),
    [
        (".env", "variables de entorno"),
        ("app/.env", "variables de entorno"),
        (".env.local", "variables de entorno"),
        ("frontend/.env.production", "variables de entorno"),
        (".aws/credentials", "AWS"),
        ("app/.aws/config", "AWS"),
        (".dvc/config.local", "DVC"),
        ("mlruns/0/meta.yaml", "MLflow"),
        ("app/mlruns/1/abc/metrics/val_loss", "MLflow"),
        ("mlartifacts/1/abc/artifacts/best.pt", "MLflow"),
        ("data/crops/000001_000017.png", "crops"),
        ("data/crops/manifest-preview.json", "crops"),
        ("data/raw/images/cat_0001.jpg", "DVC"),
        ("data/raw/annotations/instances.json", "DVC"),
        ("app/.venv/lib/site.py", "entorno"),
        ("backend/node_modules/zod/index.js", "entorno"),
        ("terraform/environments/dev/.terraform/providers/x", "Terraform"),
        ("terraform/environments/dev/terraform.tfstate", "Terraform"),
        ("terraform/environments/dev/terraform.tfstate.backup", "Terraform"),
        ("models/p3-cnn-classifier/1.0.0/model.pt", "pesos"),
        ("app/outputs/epoch_03.ckpt", "pesos"),
    ],
)
def test_rules_flag_local_or_sensitive_paths(path: str, reason_fragment: str) -> None:
    findings = forbidden_tracked_paths([path])
    assert [p for p, _ in findings] == [path], f"{path} debería marcarse como prohibido"
    assert reason_fragment in findings[0][1]


@pytest.mark.parametrize(
    "path",
    [
        ".env.example",
        "frontend/.env.example",
        ".env.production.example",
        "app/uv.lock",
        "backend/package-lock.json",
        "frontend/package-lock.json",
        "terraform/environments/dev/.terraform.lock.hcl",
        "terraform/environments/dev/terraform.tfvars.example",
        "data/raw/images.dvc",
        "data/raw/annotations.dvc",
        "data/raw/.gitignore",
        ".dvc/config",
        ".dvc/.gitignore",
        "dvc.lock",
        "app/tests/_dataset_fixtures.py",
        "docs/environment.md",
    ],
)
def test_rules_keep_legitimate_files(path: str) -> None:
    assert forbidden_tracked_paths([path]) == []


@pytest.mark.skipif(not HAS_GIT, reason="se necesita git")
def test_forced_add_is_detected_in_a_real_index(tmp_path: Path) -> None:
    """Prueba negativa: `git add -f` salta .gitignore y el control lo detecta."""
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    git("init", "-q")
    shutil.copy(REPO_ROOT / ".gitignore", repo / ".gitignore")

    files = {
        # Legítimos: deben seguir permitidos.
        ".env.example": "MINIO_ROOT_USER=\n",
        "app/uv.lock": "version = 1\n",
        "data/raw/images.dvc": "outs:\n- md5: 0\n",
        # Locales o sensibles: .gitignore los excluye, pero se fuerzan al índice.
        ".env": "MINIO_ROOT_PASSWORD=fake\n",
        ".aws/credentials": "[default]\n",
        ".dvc/config.local": "['remote \"prod\"']\n",
        "mlruns/0/meta.yaml": "name: Default\n",
        "data/crops/000001_000017.png": "not-a-real-png",
    }
    for relative, content in files.items():
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    git("add", ".gitignore", ".env.example", "app/uv.lock", "data/raw/images.dvc")
    assert forbidden_tracked_paths(tracked_paths(repo)) == []

    forced = [".env", ".aws/credentials", ".dvc/config.local", "mlruns/0/meta.yaml"]
    forced.append("data/crops/000001_000017.png")
    git("add", "-f", *forced)

    flagged = sorted(path for path, _ in forbidden_tracked_paths(tracked_paths(repo)))
    assert flagged == sorted(forced)


@pytest.mark.skipif(not _in_git_work_tree(REPO_ROOT), reason="se necesita un clon de Git")
def test_repository_index_has_no_local_or_sensitive_files() -> None:
    findings = forbidden_tracked_paths(tracked_paths(REPO_ROOT))
    detail = "; ".join(f"{path} ({reason})" for path, reason in findings)
    assert not findings, f"Archivos que no deben versionarse: {detail}"
