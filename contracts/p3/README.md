# Contratos P3 — Platform & Portal (D01-05)

Interfaces estables entre el portal (`frontend/`), la API (`backend/`) y el worker/ML (`app/`)
para que cada área avance en paralelo. Las reglas salen del **protocolo congelado de D01-03
(#33)**; si el protocolo cambia, cambian aquí primero.

- Esquemas Zod: `backend/src/logic/p3.contracts.ts` y su espejo `frontend/src/p3/contracts.ts`.
- Fixtures compartidos: `contracts/p3/fixtures/<contrato>/{valid,invalid}-*.json`, con la forma
  `{ "contract", "valid", "why", "payload" }`. **Backend y frontend validan los mismos archivos**
  (`backend/tests/p3-contracts.test.ts`, `frontend/tests/p3-contracts.test.ts`): si los dos
  esquemas divergen, uno de los dos tests falla.

> Los fixtures son **datos de ejemplo** para construir y probar componentes. No son resultados
> reales ni evidencia de aceptación de la rúbrica: los hashes, run IDs, métricas y conteos son
> ilustrativos. La integración real se verifica en los tickets de D02–D06.

## Endpoints

| Método y ruta | Respuesta (contrato) | Dueño de la implementación |
|---|---|---|
| `GET /api/releases` | `releases_response` (forma de `release_resolver --all`, #40) | Hannah, sobre el resolver de Ale |
| `GET /api/manifest` | `manifest_summary` del manifest P3 70/20/10 vigente | Hannah, sobre el manifest de Ale |
| `POST /api/training/jobs` | body `create_training_job_request` → `training_job` (201) | Hannah (D02-05) |
| `GET /api/training/jobs` | `{ jobs: training_job[] }` | Hannah (D02-05) |
| `GET /api/training/jobs/:id` | `training_job` | Hannah (D02-05) |
| `GET /api/training/jobs/:id/logs` | `job_logs` | Hannah (D02-05) |
| `POST /api/training/jobs/:id/cancel` | `training_job` (`queued` → `cancelled`; `running` → `cancel_requested`) | Hannah (D02-05) |
| `GET /api/experiments/runs` | `experiment_runs_response` (adaptador de MLflow, sin datos fijos; 503 con el motivo si MLflow no responde) | Hannah (D04-01) |
| `GET /api/experiments/runs/:runId` | `experiment_run_detail` (run + artefactos; 409 si el run es auxiliar o incompleto, 404 si no es de P3) | Hannah (D04-01) |
| `GET /api/experiments/runs/:runId/artifacts/<ruta>` | bytes del artefacto vía MLflow (404 `artifact_missing`) | Hannah (D04-01) |
| `GET /api/selection` | `selection_state` (D05-03): estado persistido de la selección (`open` \| `candidate` \| `closed`), candidato, ranking solo validation y runs excluidos con su motivo | Ale (D04-04); lectura en Experiments: Hannah (D05-03) |
| `POST /api/selection/candidate` | Recalcula y guarda el candidato **preparatorio** (409 si ya está cerrada; 503 sin runs/manifest) | Ale (D04-04) |
| `POST /api/selection/close` | body `{ "candidate_run_id" }` → MODEL SELECTION CLOSED, definitivo (409 si no es el candidato, hay < 10 filas comparables o la campaña cambió) | Ale (D04-04); cierre oficial en D05-02 |
| `GET /api/evaluation` | `evaluation_response` (`blocked` hasta MODEL SELECTION CLOSED; cerrada y sin evaluación oficial → `pending` (200, D05-05); `ready` con su `namespace`; guardada pero incoherente → 503 con el motivo) | Ale (D04-05, D05-05) |
| `GET /api/evaluation/predictions[?format=json\|csv]` | `evaluation_predictions` por muestra (409 antes del cierre, sin leer nada; 404 sin evaluación oficial; `csv` = mismo contenido como descarga) | Ale (D04-05) |
| `GET /api/evaluation/details` | `evaluation_details`: procedencia completa (run, checkpoint, commit, release, manifest, `test_split_hash`, `closed_at`→`evaluated_at`), umbral 0.85 con conteos y ejemplos por `crop_id`. 409 antes del cierre, 404 sin evaluación, 503 si el run cerrado en MLflow no cuadra | Ale (D06-05) |
| `GET /api/models` | `models_response` | Hannah (D06) |
| `POST /api/models/:semver/publish` | `model_version` | Hannah (D06) |
| `POST /api/inference` | multipart (`image`, `model_version`) → `inference_result` | Hannah + motor de Esteban (D06) |
| `POST /api/inference/:id/annotation-queue` | `annotation_queue_item` | Hannah (D06) |

Errores: `api_error` (`{ "error": "..." }`) con 400 (validación), 404 (no existe) y 409
(estado no permitido, p. ej. evaluar test antes del cierre de selección).

## Reglas que el contrato hace cumplir

| Contrato | Regla (origen) |
|---|---|
| `training_config` | Campos y rangos congelados en #33, sin restricciones adicionales: `learning_rate` en (1e-5, 1e-2]; `seed` obligatoria y entera (cualquier valor); `patience` y `max_epochs` con rangos independientes; `hidden_dim` fijo en 128; campos desconocidos rechazados. |
| `manifest_summary` | Seed 42; clases `cat`, `dog`; cada split ±5 pp de 70/20/10 medido en crops; cada clase presente en val y test; `crops_per_class` suma `crops`. |
| `training_job` | Máquina de estados `queued → running → succeeded \| failed \| cancelled`; sin `mlflow_run_id` en `queued` (no se fabrican IDs); `succeeded` exige run; `failed` exige error; `total_epochs = max_epochs`. |
| `create_training_job_request` | `task` obligatoria: `controlled` (tarea sintética de D02-05, sin datos ni entrenamiento) o `training` (D03-03: la API exige release elegible y manifest oficial congelado, si no responde 409); `controlled.fail_at_epoch` solo con `controlled` y ≤ `max_epochs`. |
| `experiment_runs_response` | Solo `p3-cnn-classifier`; tags de trazabilidad obligatorios (`git_commit`, `dvc_release`, `dvc_images_md5`, `dvc_annotations_md5`, `dvc_release_hash`, `manifest_version`, `manifest_hash`, `classes`, `seed`, `job_id`); `tags.seed = params.seed`; `best_epoch` es la de mayor `val_accuracy` (restaurar el mejor, no el último) y `best_val_accuracy`, `best_val_macro_f1` y `best_val_loss` son los de `history[best_epoch]` (tolerancia 1e-4); sin métricas de test. |
| `experiment_runs_response` (D04-01) | `runs` = solo `p3.run_kind=training` con provenance, params y curvas completas tal como están en MLflow. `campaign_eligible` ⇔ sin `ineligible_reasons`; elegible exige `FINISHED`, resumen y `checkpoint_sha256` (FINISHED solo no basta). El `status` es el de MLflow tal cual (`RUNNING`, `SCHEDULED`, `FINISHED`, `FAILED`, `KILLED`); no se sustituye. Un training con métricas `test*` va a `excluded`. `excluded` = auxiliares (`controlled_task`, `short_run_instrumentation`, `persistence_check`), runs sin `p3.run_kind` y training con provenance/curvas incompletas, cada uno con su motivo. Un `run_id` no aparece dos veces. Pertenecer a la matriz OFAT de la campaña es de D04-03/D04-04. |
| `experiment_run_detail` | El mismo run del listado más `artifacts`: rutas relativas al run (sin `/` inicial ni `..`), directorios sin tamaño. |
| `selection_state` (D05-03) | `open` sin candidato ni ranking; `candidate`/`closed` exigen candidato y `proposed_at`; `closed` ⇔ `closed_at`; candidato = `ranking[0]` (no se reordena en la UI); `ready_to_close` exige `campaign_rows` ≥ `min_comparable_runs`; ranking solo con métricas de validation (`test_*` rechazado). |
| `evaluation_predictions` | `namespace` `official` o `synthetic` (la API solo sirve `official`); una fila por crop en orden estrictamente creciente de `crop_id`, `n_test` filas; probabilidad para cada clase declarada y ninguna otra, suma ≈ 1 (±1e-3), `predicted_class` = argmax. El backend exige además `test_split_hash` = sha256 del JSON de los `crop_id` ordenados, la matriz reconstruida igual a la de `evaluation_response` y mismo candidato/`closed_at`/manifest que el cierre persistido. |
| `evaluation_details` | `final` ⇔ `namespace` = `official` (un `synthetic` nunca es final; `local_test` no es un namespace de evaluación); `evaluated_at` > `closed_at`; `target.met` ⇔ `correct · 100 ≥ 85 · n_test` con enteros, sin redondear; aciertos + errores = `n_test`; un ejemplo de error tiene clase real ≠ predicha y uno de acierto, igual. |
| `evaluation_response` | Tres estados: `blocked` (selección abierta), `pending` (cerrada, sin evaluación: solo la identidad de la selección, sin resultados) y `ready`. `ready` y `pending` declaran `namespace` `official` o `synthetic` (`local_test` es del registro de modelos, no de Evaluation); la API oficial rechaza (503) una evaluación que no declare `official`. `blocked` hasta MODEL SELECTION CLOSED; selección solo por `val_accuracy`; `evaluated_at > closed_at`; matriz filas=reales/columnas=predichas que suma `n_test`; `accuracy = traza / n_test` exacto; `support` = suma de la fila; precision/recall/F1 coherentes con la matriz; baseline de clase mayoritaria. |
| `models_response` | `semver` propio del modelo (no del dataset); `s3_key` bajo `models/p3-cnn-classifier/<semver>/`; `sha256` hex de 64; `published` exige `version_id` y `published_at`. |
| `models_response` (D04-06) | El registro `p3_model_registry` solo marca `published` si el objeto existe en el bucket con el VersionId registrado y su tamaño, metadato `sha256` y SHA-256 del contenido coinciden; objeto ausente o hash incorrecto → `failed` con motivo. Las pruebas locales (MinIO) van en el namespace `local_test`, separado de `official`. Detalle en `backend/src/logic/model-registry.README.md`. |
| `inference_result` | Probabilidad para cada clase declarada y ninguna otra; suma ≈ 1 (±1e-3); `predicted_class` = argmax. |

## Estados de las cinco páginas

| Página | Ruta | Vacío | Bloqueado | Error / incompatible |
|---|---|---|---|---|
| Training | `/ml/training` | Sin jobs | Sin release aprobado o manifest no congelado | Error HTTP o respuesta fuera de contrato |
| Experiments | `/ml/experiments` | Sin corridas | — | ídem |
| Evaluation | `/ml/evaluation` | `pending`: selección cerrada, evaluación oficial aún no existe | MODEL SELECTION CLOSED no existe | ídem; el 503 muestra su motivo. Un `ready` `synthetic` se rotula como recorrido de prueba |
| Models | `/ml/models` | Sin versiones | — | ídem |
| Inference | `/ml/inference` | — | No hay ninguna versión publicada | ídem |

Una respuesta que no cumple el contrato **nunca se muestra**: la página enseña el estado de
error ("La respuesta del servidor no tiene el formato esperado.").
