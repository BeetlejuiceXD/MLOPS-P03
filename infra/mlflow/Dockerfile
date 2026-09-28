# D02-01 — Servidor MLflow de P3: backend store en MariaDB, artefactos en MinIO.
FROM python:3.12-slim

WORKDIR /srv/mlflow
RUN useradd --system --no-create-home mlflow

COPY requirements.txt ./
# --only-binary: solo wheels publicadas, sin compilar dependencias de terceros
# (mismo criterio que `uv sync --no-build` en app/).
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt

COPY bootstrap.py ./

USER mlflow
EXPOSE 5000
HEALTHCHECK --interval=10s --timeout=5s --start-period=60s --retries=12 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:5000/health', timeout=4)"]
CMD ["python", "bootstrap.py"]
