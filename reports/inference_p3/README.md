# D05-07 — Inference con el motor real de D05-04: request/response

Recorrido real del stack de compose (rama `d05-07-inference-cola-anotacion`): portal →
API (`/api/inference`) → motor de D05-04 (`inference-engine`, perfil `inference`) →
MariaDB. Las respuestas se guardaron tal cual con `curl`. Nada entrena, nada lee el
frozen test, nada toca AWS.

## Paquete y motor

- **Paquete smoke (D05-01)**, construido con `python -m model_package build` desde el run
  real `c46e4c3ab2bb4ee18c37571adbb65d92` del MLflow local (`p3-cnn-classifier`, FINISHED):
  `02-package.json` (manifiesto con inventario y SHA-256) y `02-smoke-card.json`
  (`kind: smoke`, `official_test_metrics: null`). Vive en `data/inference-package`, fuera de Git.
- **Identidad del motor** (`03-engine.json`, `GET /api/inference/engine`): `source: smoke`,
  `package_id` `p3-cnn-classifier-smoke-c46e4c3ab2bb`, `model_version: null`, run
  `c46e4c3a…` y `checkpoint_sha256` `e93de23e…5b97`. Es el mismo SHA del `model.pt` del
  paquete y del checkpoint de su run.
- Sin el motor levantado: `01-engine-sin-motor.json`, **503** con el motivo.

No es el modelo seleccionado, evaluado ni publicado: es el smoke de D05-01. D06-06 cambia
solo la fuente.

## Imágenes de entrada

`gato.jpg` = `data/raw/images/cat.0.jpg` y `perro.jpg` = `data/raw/images/dog.0.jpg`. Todas
sus anotaciones están en el split **train** del manifest congelado `p3-v0.1.1-s42`, ninguna
en val o test (se comprobó contra `data/p3/manifest.json` y las anotaciones COCO: 668
anotaciones, 422 originales solo de train).

## Recorrido

| Archivo | Request | HTTP | Resultado |
|---|---|---|---|
| `04-predict-archivo.json` | `POST /api/inference`, multipart `image=gato.jpg` | 201 | Inferencia 1: `cat` 0.99795 / `dog` 0.00205, entrada con SHA-256 e identidad del modelo |
| `05-portal-imagen.json` | `POST /api/images`, multipart `image=perro.jpg` (foto del portal) | 201 | Imagen 1 (499×375) |
| `06b-portal-anotacion.json` | `POST /api/images/1/annotations` con `06b-body-anotacion.json` | 201 | Anotación 1 (`dog`, caja 49,37,399,300) |
| `07b-predict-crop.json` | `POST /api/inference` con `07b-body-crop.json` `{annotation_id: 1}` | 201 | Inferencia 2 del recorte 399×300: `dog` 0.99995 |
| `08-cola-archivo.json` | `POST /api/inference/1/annotation-queue` | 201 | Elemento 1: crea la imagen 2 del portal `pending`, `human_label: null`, `suggestion.source: model` |
| `09-cola-archivo-reintento.json` | Mismo request (reintento) | **200** | El **mismo** elemento 1, sin duplicar |
| `10b-cola-crop.json` | `POST /api/inference/2/annotation-queue` | 201 | Elemento 2: referencia imagen 1 y anotación 1 |
| `10c-cola-crop-reintento.json` | Mismo request (reintento) | **200** | El **mismo** elemento 2 |
| `11b-anotaciones-imagen.json` | `GET /api/images/1/annotations` | 200 | La anotación humana sigue igual (`dog`, misma caja); la predicción no la sobrescribe |
| `12b-cola-tras-reinicio.json` | `GET /api/inference/annotation-queue` tras `docker compose restart backend` | 200 | Los 2 elementos persisten, `pending`, `human_label: null` |
| `13-inferencia-tras-reinicio.json`, `13b-crop-tras-reinicio.json` | `GET /api/inference/1` y `/2` tras reiniciar | 200 | Mismas inferencias, ya ligadas a sus elementos de la cola |

## Negativos (ninguno crea inferencia ni elemento)

| Archivo | Request | HTTP | Respuesta |
|---|---|---|---|
| `15-neg-tipo.json` | `image=neg-texto.txt` (`text/plain`) | 400 | Solo JPEG, PNG o WebP de hasta 5120 KiB |
| `16-neg-contenido.json` | `image=neg-corrupta.jpg` (cabecera JPEG + basura) | 400 | No es una imagen válida |
| `17-neg-tamano.json` | Archivo de 6 MiB | 413 | Excede el tamaño máximo |
| `18-neg-anotacion-inexistente.json` | `{annotation_id: 999999999}` | 404 | La anotación no existe |
| `19-neg-motor-caido.json`, `20-engine-motor-caido.json` | Con `inference-engine` detenido | 503 | Motor de inferencia (D05-04) no disponible |
| `mariadb-caida/` | Con MariaDB **realmente detenida** (código `ee674df`), ver abajo | 503 | Sin SQL ni detalle interno; nada parcial |

`14b-lista-antes.json`, `22b-lista-despues.json` y `23b-cola-final.json`: estado antes y
después de los negativos anteriores (2 inferencias, 2 elementos).

## MariaDB caída con el código corregido (`mariadb-caida/`)

Corrida con el backend en `ee674df` (cola atómica en una transacción y guarda de 503). Se
crea la inferencia 4 con la base arriba (`00-predict-archivo.json`), se toma el estado,
se **detiene MariaDB** (`docker compose stop mariadb`), se repiten las operaciones y se vuelve
a tomar el estado con la base arriba:

| Archivo | Request con MariaDB detenida | HTTP | Respuesta |
|---|---|---|---|
| `04-neg-predict-mariadb-caida.json` | `POST /api/inference` (archivo) | 503 | No se pudo guardar la inferencia: la base de datos no está disponible. No se guardó nada. |
| `05-neg-cola-mariadb-caida.json` | `POST /api/inference/4/annotation-queue` | 503 | No se pudo leer la inferencia: la base de datos o el almacenamiento no están disponibles. |
| `06-neg-lista-mariadb-caida.json` | `GET /api/inference` | 503 | No se pudo leer las inferencias: … |
| `07-neg-cola-lectura-mariadb-caida.json` | `GET /api/inference/annotation-queue` | 503 | No se pudo leer la cola de anotación: … |

Ninguna respuesta contiene SQL, parámetros ni host. Antes (`01`–`03`) y después (`08`–`10`),
las respuestas son **idénticas**: 4 inferencias, 3 elementos en la cola y una sola imagen
`pending` (la #2). `11-inferencia-sin-elemento.json`: la inferencia 4 sigue sin elemento
(`annotation_queue_item_id: null`). No quedó nada parcial.

La respuesta anterior a la corrección (que traía la consulta SQL) se quitó del repo.

La atomicidad del envío de un archivo a la cola (imagen pending + elemento en una sola
transacción) también se prueba contra MariaDB real en `backend/tests/inference.mariadb.test.ts`:
si el elemento viola el índice único, la imagen insertada en la misma transacción se
revierte. Si esos inserts se sacan de la transacción, el test falla.
