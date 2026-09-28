"""D02-01 — Arranque del servidor MLflow de P3.

1. Crea la base de datos del backend store en MariaDB si no existe (la del
   portal ya existe; `docker-entrypoint-initdb.d` solo corre con volumen nuevo).
2. Crea el bucket de artefactos en MinIO si no existe.
3. Reemplaza el proceso por `mlflow server` (backend persistente + artefactos
   servidos por el propio servidor, sin exponer MinIO al cliente).

Todo sale de variables de entorno definidas en docker-compose.yml; no imprime
contraseñas.
"""

import os
import re
import sys
import time
from urllib.parse import quote

import boto3
import pymysql
from botocore.exceptions import ClientError, EndpointConnectionError

ENV = os.environ
DB_NAME = ENV["MLFLOW_DB_NAME"]
BUCKET = ENV["MLFLOW_ARTIFACTS_BUCKET"]

if not re.fullmatch(r"[A-Za-z0-9_]+", DB_NAME):
    sys.exit(f"MLFLOW_DB_NAME inválido: {DB_NAME!r}")


def retry(action, what, attempts=30, delay=2.0):
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except (pymysql.err.OperationalError, EndpointConnectionError, ConnectionError) as error:
            if attempt == attempts:
                sys.exit(f"{what}: sin conexión tras {attempts} intentos ({type(error).__name__})")
            time.sleep(delay)


def ensure_database():
    connection = pymysql.connect(
        host=ENV["MLFLOW_DB_HOST"],
        port=int(ENV["MLFLOW_DB_PORT"]),
        user=ENV["MLFLOW_DB_USER"],
        password=ENV["MLFLOW_DB_PASSWORD"],
    )
    try:
        with connection.cursor() as cursor:
            # utf8mb4_bin: claves de métricas/params distinguen mayúsculas, como en MLflow.
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_bin"
            )
    finally:
        connection.close()
    print(f"Base de datos {DB_NAME} lista en {ENV['MLFLOW_DB_HOST']}", flush=True)


def ensure_bucket():
    s3 = boto3.client("s3", endpoint_url=ENV["MLFLOW_S3_ENDPOINT_URL"])
    try:
        s3.head_bucket(Bucket=BUCKET)
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") not in {"404", "NoSuchBucket"}:
            raise
        s3.create_bucket(Bucket=BUCKET)
    print(f"Bucket {BUCKET} listo en {ENV['MLFLOW_S3_ENDPOINT_URL']}", flush=True)


def main():
    retry(ensure_database, "MariaDB")
    retry(ensure_bucket, "MinIO")
    backend = (
        f"mysql+pymysql://{ENV['MLFLOW_DB_USER']}:{quote(ENV['MLFLOW_DB_PASSWORD'], safe='')}"
        f"@{ENV['MLFLOW_DB_HOST']}:{ENV['MLFLOW_DB_PORT']}/{DB_NAME}"
    )
    args = [
        "mlflow",
        "server",
        "--host",
        "0.0.0.0",
        "--port",
        "5000",
        "--workers",
        ENV.get("MLFLOW_WORKERS", "2"),
        "--backend-store-uri",
        backend,
        "--serve-artifacts",
        "--artifacts-destination",
        f"s3://{BUCKET}",
        # `mlflow:5000` (worker en Compose) no es localhost ni IP privada: sin esto
        # MLflow 3 responde 403 por protección contra DNS rebinding.
        "--allowed-hosts",
        ENV["MLFLOW_ALLOWED_HOSTS"],
    ]
    print("Iniciando mlflow server (backend MariaDB, artefactos en MinIO)", flush=True)
    os.execvp(args[0], args)


if __name__ == "__main__":
    main()
