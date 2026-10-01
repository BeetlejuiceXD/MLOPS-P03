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
| `GET /api/evaluation` | `evaluation_response` (`blocked` hasta MODEL SELECTION CLOSED) | Ale |
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
| `experiment_runs_response` (D04-01) | `runs` = solo `p3.run_kind=training` con provenance, params y curvas completas tal como están en MLflow. `campaign_eligible` ⇔ sin `ineligible_reasons`; elegible exige `FINISHED`, resumen y `checkpoint_sha256` (FINISHED solo no basta). `excluded` = auxiliares (`controlled_task`, `short_run_instrumentation`, `persistence_check`), runs sin `p3.run_kind` y training con provenance/curvas incompletas, cada uno con su motivo. Un `run_id` no aparece dos veces. Pertenecer a la matriz OFAT de la campaña es de D04-03/D04-04. |
| `experiment_run_detail` | El mismo run del listado más `artifacts`: rutas relativas al run (sin `/` inicial ni `..`), directorios sin tamaño. |
| `evaluation_response` | `blocked` hasta MODEL SELECTION CLOSED; selección solo por `val_accuracy`; `evaluated_at > closed_at`; matriz filas=reales/columnas=predichas que suma `n_test`; `accuracy = traza / n_test` exacto; `support` = suma de la fila; precision/recall/F1 coherentes con la matriz; baseline de clase mayoritaria. |
| `models_response` | `semver` propio del modelo (no del dataset); `s3_key` bajo `models/p3-cnn-classifier/<semver>/`; `sha256` hex de 64; `published` exige `version_id` y `published_at`. |
| `inference_result` | Probabilidad para cada clase declarada y ninguna otra; suma ≈ 1 (±1e-3); `predicted_class` = argmax. |

## Estados de las cinco páginas

| Página | Ruta | Vacío | Bloqueado | Error / incompatible |
|---|---|---|---|---|
| Training | `/ml/training` | Sin jobs | Sin release aprobado o manifest no congelado | Error HTTP o respuesta fuera de contrato |
| Experiments | `/ml/experiments` | Sin corridas | — | ídem |
| Evaluation | `/ml/evaluation` | — | MODEL SELECTION CLOSED no existe | ídem |
| Models | `/ml/models` | Sin versiones | — | ídem |
| Inference | `/ml/inference` | — | No hay ninguna versión publicada | ídem |

Una respuesta que no cumple el contrato **nunca se muestra**: la página enseña el estado de
error ("La respuesta del servidor no tiene el formato esperado.").
