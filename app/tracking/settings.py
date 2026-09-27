"""D02-01 — Configuración del cliente de MLflow.

Único módulo de `tracking/` que lee variables de entorno (misma regla que
`storage/settings.py`). `MLFLOW_TRACKING_URI` apunta al servidor persistente:
`http://mlflow:5000` dentro de Compose y `http://localhost:5000` desde el host.
"""

import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

P3_EXPERIMENT = "p3-cnn-classifier"

# Sin esto, el cliente de MLflow reintenta durante minutos contra un servidor
# caído; la verificación debe fallar rápido y con un mensaje claro.
CLIENT_ENV_DEFAULTS = {
    "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "0",
    "MLFLOW_HTTP_REQUEST_TIMEOUT": "15",
    "MLFLOW_DISABLE_AGENT_HINT": "1",
}


class TrackingSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    mlflow_tracking_uri: str = Field(min_length=1)
    experiment_name: str = P3_EXPERIMENT


def configure_client_env() -> None:
    """Fija timeouts cortos y sin reintentos, salvo que el entorno ya los defina."""
    for name, value in CLIENT_ENV_DEFAULTS.items():
        os.environ.setdefault(name, value)
