"""D05-07 — El portal queda conectado al motor real de D05-04 desde docker-compose.

El motor sirve el paquete smoke de D05-01 (fuera de Git) y el backend lo llama por
INFERENCE_ENGINE_URL. D06-06 sustituye solo la fuente: otra URL o paquete, mismo contrato.
"""

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
SERVICES = yaml.safe_load((REPO / "docker-compose.yml").read_text(encoding="utf-8"))["services"]


def test_el_motor_de_d05_04_es_un_servicio_del_compose():
    engine = SERVICES["inference-engine"]
    assert engine["build"] == "./app"
    assert engine["command"][:4] == ["python", "-m", "inference_engine", "serve"]
    command = engine["command"]
    assert command[command.index("--package") + 1] == "/package"
    assert command[command.index("--port") + 1] == "8090"


def test_el_paquete_se_monta_de_solo_lectura_y_fuera_de_git():
    volumes = SERVICES["inference-engine"]["volumes"]
    assert "${INFERENCE_PACKAGE_DIR:-./data/inference-package}:/package:ro" in volumes
    ignored = (REPO / "data/.gitignore").read_text(encoding="utf-8").splitlines()
    assert "/inference-package" in ignored


def test_el_motor_solo_arranca_con_el_perfil_inference():
    """Sin paquete no hay motor: el resto del stack (y la CI) arranca igual, y el
    portal responde 503 con el motivo en vez de inventar una predicción."""
    assert SERVICES["inference-engine"]["profiles"] == ["inference"]
    assert "inference-engine" not in SERVICES["backend"].get("depends_on", {})


def test_el_backend_apunta_al_motor_del_compose_por_defecto():
    url = SERVICES["backend"]["environment"]["INFERENCE_ENGINE_URL"]
    assert url == "${INFERENCE_ENGINE_URL:-http://inference-engine:8090}"
