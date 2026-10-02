"""D06-03 — Raíz de Terraform del bucket de modelos P3 (`terraform/p3-models`).

`terraform test` (CI, provider simulado) comprueba los valores del bucket y del permiso
operacional. Aquí se comprueba lo que un test de Terraform no puede afirmar: que NO exista
ninguna regla que borre versiones, que nada destruya el bucket, que no haya credenciales y
que CI valide y pruebe la raíz.
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "terraform" / "p3-models"
WORKFLOW = REPO / ".github" / "workflows" / "terraform-oidc.yml"


def _tf_sources() -> str:
    files = sorted(ROOT.glob("*.tf"))
    assert files, "terraform/p3-models no tiene archivos .tf"
    return "\n".join(path.read_text(encoding="utf-8") for path in files)


def test_root_has_its_own_state_and_pinned_provider():
    sources = _tf_sources()
    assert 'key          = "p3-models/terraform.tfstate"' in sources
    assert 'version = "~> 6.0"' in sources
    lock = (ROOT / ".terraform.lock.hcl").read_text(encoding="utf-8")
    assert lock == (REPO / "terraform/environments/dev/.terraform.lock.hcl").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("pattern", "why"),
    [
        (r"aws_s3_bucket_lifecycle_configuration", "un lifecycle podría expirar versiones"),
        (r"force_destroy\s*=\s*true", "destruir el bucket se llevaría las versiones"),
        (r"noncurrent_version_expiration", "las versiones anteriores se conservan"),
        (r"aws_s3_bucket_acl", "sin ACL: el bucket usa BucketOwnerEnforced"),
        (r"(?i)aws_access_key_id|aws_secret_access_key|secret_key\s*=", "nunca credenciales"),
    ],
)
def test_nothing_deletes_versions_or_carries_secrets(pattern, why):
    assert not re.search(pattern, _tf_sources()), why


def test_bucket_name_is_not_invented():
    # El nombre real sale del recurso (bucket_prefix), nunca de una variable con el nombre.
    sources = _tf_sources()
    assert re.search(r'resource "aws_s3_bucket" "models"', sources)
    assert not re.search(r'^\s*bucket\s*=\s*"', sources, re.MULTILINE)


def test_ci_validates_and_tests_the_root():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    assert "p3-models" in workflow
    assert 'terraform -chdir="terraform/p3-models" test' in workflow
