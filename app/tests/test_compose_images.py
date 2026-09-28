"""D02-01 — Las imágenes de docker-compose.yml deben tener una versión fija.

Sin versión (o con `latest`) un clon limpio descarga lo que el registro sirva ese
día, o nada: en septiembre de 2026 `quay.io/minio/minio` sin tag pasó a pedir login
y `docker compose up` dejó de arrancar en máquinas sin la imagen en caché (M1).
"""

from pathlib import Path

import pytest
import yaml

COMPOSE = Path(__file__).resolve().parents[2] / "docker-compose.yml"


def _images() -> list[tuple[str, str]]:
    services = yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["services"]
    return [(name, spec["image"]) for name, spec in services.items() if "image" in spec]


def _tag(image: str) -> str | None:
    if "@sha256:" in image:
        return "digest"
    last = image.rsplit("/", 1)[-1]
    return last.split(":", 1)[1] if ":" in last else None


def test_compose_declares_images():
    assert _images(), "docker-compose.yml no declara imágenes"


@pytest.mark.parametrize(("service", "image"), _images())
def test_image_is_pinned(service, image):
    tag = _tag(image)
    assert tag is not None, f"{service}: la imagen {image} no fija versión"
    assert tag != "latest", f"{service}: {image} usa latest"


@pytest.mark.parametrize(
    ("image", "expected"),
    [
        ("quay.io/minio/minio", None),
        ("quay.io/minio/minio:latest", "latest"),
        ("localhost:5000/minio", None),
        ("mariadb:11", "11"),
        ("quay.io/minio/minio:RELEASE.2025-09-07T16-13-09Z", "RELEASE.2025-09-07T16-13-09Z"),
        ("minio/minio@sha256:" + "a" * 64, "digest"),
    ],
)
def test_tag_parser(image, expected):
    assert _tag(image) == expected
