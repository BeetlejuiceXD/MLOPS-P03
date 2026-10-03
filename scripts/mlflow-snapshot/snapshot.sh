#!/usr/bin/env bash
# D104 — snapshot reproducible del MLflow productor: metadata (MariaDB) +
# artifacts (MinIO) + el estado de selección mínimo necesario para D05-02.
#
# Solo LECTURA sobre el productor: mysqldump y mc mirror son exports, nunca
# tocan los datos de origen. No usa docker compose down -v en ningún
# momento, ni antes, ni después.
#
# Alcance exacto (congelado en #104, ampliado para incluir los
# training_jobs originales a pedido del PM):
#   - dump COMPLETO de la base `mlflow`
#   - dump de SOLO las tablas `image_repo.p3_model_selection` e
#     `image_repo.training_jobs` (nunca `image_repo` completa)
#   - espejo 1:1 del bucket MinIO `mlflow-artifacts`
#
# Uso: ./scripts/mlflow-snapshot/snapshot.sh [directorio_salida]
# Por defecto: data/mlflow_snapshot/ (relativo a la raíz del repo).

set -euo pipefail

OUT_DIR="${1:-data/mlflow_snapshot}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [ -z "${MARIADB_ROOT_PASSWORD:-}" ]; then
  export $(grep -E "^MARIADB_ROOT_PASSWORD=" .env | xargs)
fi
if [ -z "${MINIO_ROOT_USER:-}" ] || [ -z "${MINIO_ROOT_PASSWORD:-}" ]; then
  export $(grep -E "^(MINIO_ROOT_USER|MINIO_ROOT_PASSWORD)=" .env | xargs)
fi

mkdir -p "$OUT_DIR/db" "$OUT_DIR/artifacts"

echo "==> Confirmando que el stack productor está sano (no se toca su estado)"
docker compose ps mariadb minio mlflow

echo "==> [1/4] Dump completo de la base 'mlflow' (incluye CREATE DATABASE)"
docker compose exec -T mariadb mariadb-dump -uroot -p"$MARIADB_ROOT_PASSWORD" \
  --databases mlflow --single-transaction --routines --triggers \
  > "$OUT_DIR/db/mlflow.sql"

echo "==> [2/4] Dump de SOLO image_repo.p3_model_selection (nunca la base completa)"
docker compose exec -T mariadb mariadb-dump -uroot -p"$MARIADB_ROOT_PASSWORD" \
  --single-transaction image_repo p3_model_selection \
  > "$OUT_DIR/db/image_repo_p3_model_selection.sql"

echo "==> [3/4] Dump de SOLO image_repo.training_jobs (nunca la base completa)"
docker compose exec -T mariadb mariadb-dump -uroot -p"$MARIADB_ROOT_PASSWORD" \
  --single-transaction image_repo training_jobs \
  > "$OUT_DIR/db/image_repo_training_jobs.sql"

echo "==> [4/4] Espejo 1:1 del bucket mlflow-artifacts"
docker compose exec minio mc alias set snapshot-src http://localhost:9000 \
  "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" >/dev/null
docker compose exec minio rm -rf /tmp/mlflow-artifacts-mirror
docker compose exec minio mc mirror --overwrite snapshot-src/mlflow-artifacts /tmp/mlflow-artifacts-mirror
rm -rf "$OUT_DIR/artifacts/mlflow-artifacts"
docker compose cp minio:/tmp/mlflow-artifacts-mirror "$OUT_DIR/artifacts/mlflow-artifacts"
docker compose exec minio rm -rf /tmp/mlflow-artifacts-mirror

echo "==> Generando manifest.json del snapshot (metadata para auditar, sin secretos)"
python3 - "$OUT_DIR" <<'PYEOF'
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

out_dir = Path(sys.argv[1])


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


artifacts_dir = out_dir / "artifacts" / "mlflow-artifacts"
artifact_files = list(artifacts_dir.rglob("*")) if artifacts_dir.exists() else []
artifact_files = [p for p in artifact_files if p.is_file()]
total_bytes = sum(p.stat().st_size for p in artifact_files)

manifest = {
    "created_at": datetime.now(UTC).isoformat(),
    "git_commit": git_commit(),
    "scope": {
        "mariadb_mlflow_db": "full",
        "mariadb_image_repo": "p3_model_selection and training_jobs only",
        "minio_bucket": "mlflow-artifacts",
    },
    "files": {
        "db/mlflow.sql": {
            "sha256": sha256_file(out_dir / "db/mlflow.sql"),
            "size_bytes": (out_dir / "db/mlflow.sql").stat().st_size,
        },
        "db/image_repo_p3_model_selection.sql": {
            "sha256": sha256_file(out_dir / "db/image_repo_p3_model_selection.sql"),
            "size_bytes": (out_dir / "db/image_repo_p3_model_selection.sql").stat().st_size,
        },
        "db/image_repo_training_jobs.sql": {
            "sha256": sha256_file(out_dir / "db/image_repo_training_jobs.sql"),
            "size_bytes": (out_dir / "db/image_repo_training_jobs.sql").stat().st_size,
        },
    },
    "artifacts": {
        "file_count": len(artifact_files),
        "total_bytes": total_bytes,
    },
}
(out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps(manifest, indent=2))
PYEOF

echo
echo "==> Snapshot escrito en: $OUT_DIR"
du -sh "$OUT_DIR"