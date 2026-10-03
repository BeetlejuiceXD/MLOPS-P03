#!/usr/bin/env bash
# D104 — restaura un snapshot en un stack AISLADO (proyecto Compose propio,
# volúmenes propios, puertos propios). NUNCA toca el stack productor.
#
# Uso: ./scripts/mlflow-snapshot/restore.sh [directorio_snapshot] [proyecto]
# Por defecto: data/mlflow_snapshot/ y proyecto "mlops-p03-restore".
#
# FAIL-CLOSED: antes de la PRIMERA operación que modifica Docker se ejecuta
# restore_guard.py (solo lectura). Si el proyecto es el del productor, si algún
# volumen de datos no es aislado, si hay recursos sin la marca de este restore o
# con otro snapshot, o si el snapshot local no coincide con su manifest, aborta
# (código 3) sin haber levantado, copiado ni importado nada.

set -euo pipefail

SNAPSHOT_DIR="${1:-data/mlflow_snapshot}"
PROJECT="${2:-mlops-p03-restore}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
DOCKER="${DOCKER:-docker}"
PYTHON="${PYTHON:-python3}"

if [ -z "${MARIADB_ROOT_PASSWORD:-}" ]; then
  export $(grep -E "^MARIADB_ROOT_PASSWORD=" .env | xargs)
fi
if [ -z "${MINIO_ROOT_USER:-}" ] || [ -z "${MINIO_ROOT_PASSWORD:-}" ]; then
  export $(grep -E "^(MINIO_ROOT_USER|MINIO_ROOT_PASSWORD)=" .env | xargs)
fi

# Identidad del snapshot = el md5 del puntero DVC versionado. Etiqueta los volúmenes
# del restore (docker-compose.restore-isolated.yml) y es la marca que exige un
# segundo restore sobre el mismo proyecto.
P3_RESTORE_SNAPSHOT_ID="$(sed -n 's/^- md5: *//p' data/mlflow_snapshot.dvc | head -n 1)"
export P3_RESTORE_SNAPSHOT_ID

echo "==> [0/8] Guarda fail-closed (solo lectura): proyecto '$PROJECT', snapshot $P3_RESTORE_SNAPSHOT_ID"
"$PYTHON" scripts/mlflow-snapshot/restore_guard.py \
  --project "$PROJECT" --repo "$REPO_ROOT" \
  --snapshot-dir "$SNAPSHOT_DIR" --snapshot-id "$P3_RESTORE_SNAPSHOT_ID" --docker "$DOCKER"

COMPOSE="$DOCKER compose -f docker-compose.yml -f docker-compose.restore-isolated.yml -p $PROJECT"

echo "==> Proyecto aislado: $PROJECT (puertos: mariadb 3307, minio 9010/9011, mlflow 5001, backend 3101)"

echo "==> [1/8] Levantando mariadb + minio aislados (volúmenes propios y marcados)"
$COMPOSE up -d --wait mariadb minio

echo "==> [2/8] Levantando backend aislado (crea el esquema de image_repo vía migraciones)"
$COMPOSE up -d --build --wait backend

echo "==> [3/8] Importando dump de 'mlflow' (crea la base desde cero)"
$COMPOSE exec -T mariadb mariadb -uroot -p"$MARIADB_ROOT_PASSWORD" \
  < "$SNAPSHOT_DIR/db/mlflow.sql"

echo "==> [4/8] Importando image_repo.p3_model_selection sobre el esquema ya creado por backend"
$COMPOSE exec -T mariadb mariadb -uroot -p"$MARIADB_ROOT_PASSWORD" image_repo \
  < "$SNAPSHOT_DIR/db/image_repo_p3_model_selection.sql"

echo "==> [5/8] Importando image_repo.training_jobs (los jobs ORIGINALES de la campaña)"
$COMPOSE exec -T mariadb mariadb -uroot -p"$MARIADB_ROOT_PASSWORD" image_repo \
  < "$SNAPSHOT_DIR/db/image_repo_training_jobs.sql"

echo "==> [6/8] Levantando mlflow aislado (la base ya existe, no hay migración nueva que correr)"
$COMPOSE up -d --build --wait mlflow

echo "==> [7/8] Levantando trainer-worker aislado (necesario para /selection/campaign: publica el manifest)"
$COMPOSE up -d --build --wait trainer-worker

echo "==> [8/8] Restaurando artifacts a MinIO aislado"
$COMPOSE exec minio mc alias set restore-dst http://localhost:9000 \
  "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" >/dev/null
$COMPOSE exec minio mc mb --ignore-existing restore-dst/mlflow-artifacts
$COMPOSE exec minio rm -rf /tmp/restore-artifacts
$COMPOSE cp "$SNAPSHOT_DIR/artifacts/mlflow-artifacts" minio:/tmp/restore-artifacts
$COMPOSE exec minio mc mirror --overwrite /tmp/restore-artifacts restore-dst/mlflow-artifacts
$COMPOSE exec minio rm -rf /tmp/restore-artifacts

echo
echo "==> Restore completo en el proyecto '$PROJECT'."
echo "    MLflow aislado: http://localhost:5001"
echo "    Backend aislado: http://localhost:3101"
echo "    Verifica con: python3 scripts/mlflow-snapshot/verify.py --api http://localhost:3101 --tracking-uri http://localhost:5001"

