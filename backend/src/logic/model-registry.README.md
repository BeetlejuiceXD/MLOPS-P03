# D04-06 — Registro y adaptador de almacenamiento de modelos

Registra cada versión del modelo con una identidad comprobable y guarda su objeto en el
bucket de modelos. Una versión solo queda `published` cuando su objeto existe en el
storage con el VersionId registrado y su contenido coincide con lo declarado.

- `model-registry.ts`: reglas puras. Contiene la clave `models/p3-cnn-classifier/<semver>/model.pt`,
  el SHA-256, el orden por semver y la conversión al contrato `model_version`.
- `model-registry.service.ts`: `register → upload → verify` (más `audit` y `list`) y las
  interfaces `ModelObjectStore` y `ModelRegistryRepository`.
- `model-registry.repository.ts` y `../data/repositories/model-registry.repository.ts`:
  tabla `p3_model_registry` (migración 0008), con transiciones por `UPDATE` condicional.
- `../data/storage/model-object.storage.ts`: adaptador S3-compatible. Sube con el
  metadato `x-amz-meta-sha256` y hace `head`/`get` por VersionId. Funciona igual con
  MinIO y con AWS S3.

> Este ticket solo lo prueba con **MinIO local**, en el namespace `local_test` y con
> objetos de prueba. **No hay ningún modelo final publicado en AWS**. Crear el bucket con
> Terraform, publicar la versión oficial y recargarla corresponde a D06-03/D06-04. No
> hay endpoint nuevo: `GET /api/models` y `POST /api/models/:semver/publish` son de D06.

## Estados

```text
register ──▶ draft ──upload──▶ draft + VersionId ──verify──▶ published
               │                       │
               └──────── failed ◀──────┘   (terminal)
```

| Paso | Qué comprueba | Si falla |
|---|---|---|
| `register` | Identidad del contrato `model_version`: semver, run de MLflow, `manifest_hash`, `dvc_release`, `dvc_release_hash` y `sha256`, más `size_bytes` > 0. El bucket sale de `MODEL_S3_BUCKET` y la clave, del semver. | 400 sin escribir. Si el semver ya existe en el namespace, 409: el semver es inmutable. |
| `upload` | Los bytes deben medir `size_bytes` y tener el `sha256` registrado **antes** de subirlos. Guarda el VersionId que devuelve el storage. | `failed` (`size_mismatch`, `sha256_mismatch`) y no se sube nada. Si no hay VersionId (bucket sin versioning), `failed` con `version_id_missing`. |
| `verify` | `head` por VersionId: existe, mismo VersionId, mismo tamaño, metadato `sha256` igual. `get` por VersionId: el SHA-256 del contenido es el registrado. | `failed` con `object_missing`, `version_mismatch`, `size_mismatch` o `sha256_mismatch`. Nunca `published`. |
| `audit` | Repite `verify` sobre una versión ya `published`, por ejemplo tras un reinicio. | Devuelve el motivo y **no** cambia el registro. |

- Un error del storage (red, credenciales, permisos) **no** se interpreta como objeto
  ausente. Responde 503 y la versión sigue como estaba. Solo `NotFound`, `NoSuchKey` y
  `NoSuchVersion` significan que el objeto no existe.
- `published` y `failed` son definitivos. Para reintentar se registra otro semver.
- Sobrescribir la misma clave crea **otra** versión en S3. La registrada no cambia y se
  sigue leyendo por su VersionId.
- `namespace`: `local_test` para roundtrips de prueba y `official` para la publicación
  real (D06-03). `GET /api/models` solo deberá servir `official`, así que una prueba
  local nunca aparece como modelo publicado.

## Bucket y configuración

| Variable | Valor local/CI | Nota |
|---|---|---|
| `MODEL_S3_BUCKET` | `p3-models-local` (MinIO) | En AWS es el bucket propio de modelos, que todavía no existe. |
| Prefijo | `models/p3-cnn-classifier/<semver>/` | Lo exige el contrato `model_version`. |

`ensureLocalModelBucket` (solo local/CI) crea el bucket y activa versioning.
`assertModelBucketVersioned` rechaza cualquier bucket sin versioning `Enabled`. En AWS,
el adaptador **no** crea ni configura el bucket: solo lo comprueba.

Requisitos del bucket de AWS, para el ticket de Terraform que corresponda:

- versioning `Enabled`;
- SSE `AES256` por defecto;
- public access block completo;
- política `DenyInsecureTransport` (solo HTTPS);
- conservación de versiones (sin lifecycle que borre versiones no actuales del prefijo);
- `force_destroy = false`.

## Permisos

Son dos cosas distintas y una no implica la otra:

| Tipo | Quién | Alcance |
|---|---|---|
| Infraestructura (Terraform) | `MLOPS-P3-Terraform` (asignado a Hanna) | Crear y configurar el bucket. **No** sirve para publicar ni leer modelos. |
| Acceso operativo mínimo | `MLOPS-S3-MODELS` (todavía no existe) | Para publicar y recargar: `s3:PutObject`, `s3:GetObject`, `s3:GetObjectVersion` y `s3:ListBucket` (con condición de prefijo), solo sobre `arn:aws:s3:::<MODEL_S3_BUCKET>/models/p3-cnn-classifier/*`, y `s3:GetBucketVersioning` sobre el bucket. Sin `DeleteObject`/`DeleteObjectVersion`. |

`MLOPS-S3-DVC` es el acceso al remoto de DVC, no al bucket de modelos. Tener
`MLOPS-S3-DVC` o `MLOPS-P3-Terraform` **no** demuestra que se pueda publicar.

### Estado administrativo de AWS (según Heri, sin verificar en AWS)

- [ ] Bucket exclusivo de modelos P3: no existe. Lo crea Hanna con Terraform en su ticket.
- [ ] `MLOPS-S3-MODELS`: no existe. Lo coordina Heri y Hanna verifica después el acceso operativo.
- [ ] Publicación y recarga oficial: pendientes (D06-03/D06-04).

## Pruebas

- `tests/model-registry.test.ts`: tests de componente con storage y repositorio en
  memoria, más el adaptador MinIO con un cliente simulado.
- `tests/model-registry.minio.test.ts`: MinIO y MariaDB **reales** del job de CI "Jobs
  persistentes". La fase `write` registra, sube, verifica y provoca las negativas. Luego
  `docker compose restart mariadb minio`, y la fase `check` relee desde otro proceso. Usa
  credenciales de MinIO del runner, nunca de AWS. Para correrlo a mano con Compose
  levantado:

```bash
cd backend
export P3_MODEL_REGISTRY_TEST=1 DATABASE_URL=mysql://root:<pwd>@127.0.0.1:3306/image_repo \
  MINIO_ENDPOINT=127.0.0.1 MINIO_PORT=9000 MINIO_ACCESS_KEY=<user> MINIO_SECRET_KEY=<pwd> \
  MODEL_S3_BUCKET=p3-models-local P3_MODEL_REGISTRY_EVIDENCE=/tmp/model-registry.json
P3_MODEL_REGISTRY_PHASE=write npx vitest run tests/model-registry.minio.test.ts
docker compose -f ../docker-compose.yml restart mariadb minio
P3_MODEL_REGISTRY_PHASE=check npx vitest run tests/model-registry.minio.test.ts
```

- `.github/scripts/run_model_registry_mutations.py`: mutation testing en una copia aislada.
