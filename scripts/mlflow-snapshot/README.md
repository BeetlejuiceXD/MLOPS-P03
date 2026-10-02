# D104 — Snapshot/restore reproducible de MLflow con DVC

Hace reproducible, **sin reentrenar**, el estado real de MLflow que usó D05-02: la campaña
de D04-03 (12 filas OFAT, 16 intentos) y la selección con su candidato. El snapshot pesado
(~684 MiB) vive en DVC/S3 (`prod`), **nunca en Git**: Git solo lleva el puntero
`data/mlflow_snapshot.dvc`, los scripts y esta guía.

## Flujo

```
PRODUCTOR (stack de quien entrenó la campaña: MariaDB mlflow + MinIO mlflow-artifacts)
    │  ./scripts/mlflow-snapshot/snapshot.sh            (solo lectura sobre el productor)
    ▼
data/mlflow_snapshot/        db/mlflow.sql · db/image_repo_p3_model_selection.sql
    │                        artifacts/mlflow-artifacts/ · manifest.json
    │  dvc add data/mlflow_snapshot
    ▼
data/mlflow_snapshot.dvc     (puntero md5 → es la identidad del snapshot)
    │  dvc push -r prod
    ▼
Git (.dvc + scripts + docs)  +  AWS S3 del remote DVC prod (mlops-p2-dvc-cache)
    │
    ▼  segunda máquina / clean clone
git pull origin main
dvc pull -r prod data/mlflow_snapshot.dvc
    │
    ▼
./scripts/mlflow-snapshot/restore.sh                  (guarda fail-closed + stack AISLADO)
    │
    ▼
python3 scripts/mlflow-snapshot/verify.py --api http://localhost:3101 --tracking-uri http://localhost:5001
```

### 1. Productor → snapshot (solo quien tiene la campaña)

```bash
# en la raíz del repo, con el stack productor arriba y sano
./scripts/mlflow-snapshot/snapshot.sh                 # escribe data/mlflow_snapshot/
python3 scripts/mlflow-snapshot/verify.py             # contra el productor (:3100 / :5000)
dvc add data/mlflow_snapshot
dvc push -r prod
dvc status -c -r prod                                 # "Cache and remote 'prod' are in sync."
git add data/mlflow_snapshot.dvc data/.gitignore && git commit
```

`snapshot.sh` solo exporta: `mariadb-dump` de la base `mlflow` completa y de **solo** la
tabla `image_repo.p3_model_selection`, y `mc mirror` del bucket `mlflow-artifacts`. Nunca
ejecuta `docker compose down -v` ni escribe en el productor.

### 2. Segunda máquina / clean clone → restore aislado

```bash
git clone https://github.com/BeetlejuiceXD/MLOPS-P03.git && cd MLOPS-P03
cp .env.example .env                                  # y completa los valores locales
dvc pull -r prod data/mlflow_snapshot.dvc             # ~684 MiB desde S3 (perfil con MLOPS-S3-DVC)
./scripts/mlflow-snapshot/restore.sh                  # proyecto mlops-p03-restore
python3 scripts/mlflow-snapshot/verify.py \
  --api http://localhost:3101 --tracking-uri http://localhost:5001
```

Argumentos de `restore.sh`: `[directorio_snapshot] [proyecto]` (por defecto
`data/mlflow_snapshot` y `mlops-p03-restore`).

## Variables necesarias

| Variable | De dónde sale | Para qué |
|---|---|---|
| `MARIADB_ROOT_PASSWORD`, `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD` | `.env` local (nunca en Git) | Levantar e importar en el stack aislado. Los scripts las leen en tiempo de ejecución y no las imprimen. |
| `AWS_PROFILE` (o la cadena de credenciales por defecto) | SSO con el permiso `MLOPS-S3-DVC` | `dvc pull/push -r prod`. |
| `P3_RESTORE_SNAPSHOT_ID` | La exporta `restore.sh` (md5 de `data/mlflow_snapshot.dvc`) | Marca los volúmenes del restore. Sin ella, `docker-compose.restore-isolated.yml` no resuelve. |
| `P3_PRODUCER_PROJECT` (opcional) | Tú, si tu productor usa un nombre de proyecto no estándar | Nombres extra (separados por coma) que la guarda trata como productor. |
| `DOCKER`, `PYTHON` (opcional) | Por defecto `docker` y `python3` | Binarios que usa `restore.sh`. |

Requisitos: Docker con Compose v2, `bash` y `python3` (la guarda y `verify.py` usan solo la
librería estándar).

## Cómo se identifica el productor

El productor es el proyecto que resuelve `docker compose -f docker-compose.yml config` en
la raíz del repo: `COMPOSE_PROJECT_NAME` si está definido (en el entorno o en `.env`) o, si
no, el nombre de la carpeta del repo (p. ej. `mlops-p03`). Para verlo:

```bash
docker compose config --format json | python3 -c "import json,sys; print(json.load(sys.stdin)['name'])"
docker compose ls                                     # proyectos con contenedores
```

La guarda resuelve ese nombre **con y sin** `COMPOSE_PROJECT_NAME` del entorno, agrega los de
`P3_PRODUCER_PROJECT`, y trata como del productor sus volúmenes (`<productor>_mariadb_data`,
`<productor>_minio_data` y los que declare su configuración).

## Cómo elegir el proyecto aislado

Usa el default `mlops-p03-restore` o pasa otro como segundo argumento. Debe contener
`restore`, usar solo minúsculas, dígitos, `-` y `_`, y **no** puede ser el del productor.
`docker-compose.restore-isolated.yml` le da puertos propios (MariaDB 3307, MinIO 9010/9011,
MLflow 5001, backend 3101) para que convivan ambos stacks en la misma máquina.

## Protecciones fail-closed

Antes de la **primera** operación que modifica Docker (`up`, `exec`, `cp`, `mc`),
`restore.sh` ejecuta `restore_guard.py`, que solo lee. Si encuentra cualquier problema
imprime `RESTORE GUARD: ABORT`, sale con código **3** y no se levanta, importa ni copia
nada. Aborta si:

- el proyecto es el del productor, no contiene `restore` o tiene un nombre inválido;
- en la configuración Compose **resuelta** del restore, MariaDB (`/var/lib/mysql`) o MinIO
  (`/data`) no montan un volumen propio `<proyecto>_<volumen>`: bind mount, volumen
  `external`, `name:` fijo o un volumen del productor;
- la configuración no marca esos volúmenes con `org.p3.restore.marker` y el snapshot esperado;
- algún contenedor de **otro** proyecto monta los volúmenes del restore;
- el proyecto ya tiene recursos (contenedores, redes o volúmenes con
  `com.docker.compose.project=<proyecto>`) y alguno de sus volúmenes no tiene la marca,
  tiene la de otro snapshot o falta (recursos ambiguos);
- el snapshot local no está completo o sus dumps no coinciden con `manifest.json`.

No depende de que el nombre por defecto sea distinto: todo se comprueba contra lo que
Docker y Compose reportan. Los tests están en `app/tests/test_mlflow_snapshot_restore_guard.py`
(ejecutan `restore.sh` con un `docker` falso y comprueban que los casos rechazados no
llegan a ninguna operación).

## Segundo restore

Se permite repetir el restore sobre el **mismo** proyecto aislado solo si sus volúmenes
fueron creados por este mecanismo (`org.p3.restore.marker=mlflow-snapshot-restore`) para el
**mismo** snapshot (`org.p3.restore.snapshot=<md5 del .dvc>`). Los dumps llevan
`DROP TABLE IF EXISTS` y `mc mirror --overwrite`, así que el resultado es el mismo.

Si cambió el snapshot o el proyecto aislado se creó antes de esta marca (p. ej. con la
primera versión de este script), la guarda aborta. Revisa que de verdad sea el proyecto
**aislado** (`docker compose ls`, `docker volume ls --filter label=com.docker.compose.project=<proyecto>`)
y solo entonces bórralo a mano: `docker compose -p <proyecto-restore> down -v`.

## Lo que NUNCA se hace

- **Nunca** usar el productor como destino del restore.
- **Nunca** `docker compose down -v` sobre el productor: borra sus volúmenes y la campaña.
- El restore **no entrena** ni reentrena modelos: solo importa la base y los artefactos.
- **No** ejecuta el frozen test. `verify.py` falla si cualquiera de los 16 intentos tiene una
  métrica `test_*`.
- El snapshot pesado permanece en DVC/S3; en Git solo están el puntero, los scripts y docs.

## Qué comprueba `verify.py`

`reports/campaign_p3.json` es el **contrato** (qué debe existir); la evidencia se obtiene del
MLflow y la API restaurados. Sale con código 1 ante cualquier discrepancia:

- `runs/search` paginado sobre `p3-cnn-classifier`: exactamente los 16 run IDs esperados,
  únicos, sin runs extra ni sustitutos;
- 12 filas: fila 1 con 5 intentos (representante job 1 `684c6a69…`, el de menor
  `start_time` real, y 4 reintentos), filas 2–12 con uno;
- cada intento `FINISHED` y activo, tag `job_id` y `checkpoint_sha256` del contrato,
  métricas `best_*` iguales al contrato, historial por época completo (1..`epochs_logged`)
  de `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy` y `val_macro_f1`, y la curva
  en `best_epoch` igual a cada `best_*`;
- ninguna métrica `test_*` en los 16 intentos;
- `checkpoint/model.pt` presente en los 16; el del candidato se **descarga** y el SHA-256
  de sus bytes debe ser `0c6b589bdd8ba639ed6890386db5bdd20555adc3452df7e9657c2b4a4b9d563b`;
- `GET /selection`: `status=candidate`, `closed_at=null`, candidato
  `2d56233c886142b7824e1551b90e8327` (fila 3, job 7).

## Nota sobre el escaneo de secretos

Un `grep -i "secret"` sobre `db/mlflow.sql` encuentra coincidencias — son el
**esquema** de la tabla nativa `secrets` de MLflow 3.x (usada para
credenciales de proveedores GenAI externos, no usada en este proyecto:
solo tracking de entrenamiento). Confirmado vacía por dos vías
independientes antes de publicar:

    grep -c "INSERT INTO \`secrets\`" data/mlflow_snapshot/db/mlflow.sql   # 0
    docker compose exec mariadb mariadb -uroot -p"$MARIADB_ROOT_PASSWORD" \
      -e "SELECT COUNT(*) FROM mlflow.secrets;"                            # 0

Si en el futuro esta tabla llegara a tener filas, excluir sus DATOS del
dump (sin tocar el resto) con:

    mariadb-dump ... --ignore-table=mlflow.secrets --databases mlflow > ...
    mariadb-dump ... --no-data mlflow secrets >> ...   # solo el esquema, sin filas
