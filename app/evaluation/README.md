# D03-05 — Motor de métricas de evaluación (P3)

`evaluation.metrics` calcula, a partir de etiquetas reales y predichas **ya
obtenidas**:

- matriz de confusión;
- accuracy y macro-F1;
- precision, recall, F1 y support por clase;
- baseline de clase mayoritaria;
- criterio de aceptación 0.85.

Es una función pura: no lee archivos, manifest, DVC ni MLflow, y no conoce
particiones. Quien llama decide sobre qué datos se evalúa.

> Ale custodia el frozen test (#33). Este ticket solo se probó con predicciones
> controladas. El test oficial se abre en D06-01, después de MODEL SELECTION CLOSED
> (D05-02). `tests/test_evaluation_metrics.py::test_metrics_engine_has_no_io_or_manifest_access`
> falla si el motor empieza a leer archivos o datos.

## Uso

```python
from evaluation.metrics import compute_metrics

report = compute_metrics(y_true=["cat", "dog", ...], y_pred=["cat", "cat", ...])
report.rows  # ((tp_cat, cat→dog), (dog→cat, tp_dog))
report.accuracy, report.macro_f1  # sin redondear
report.per_class  # (ClassMetrics(cat), ClassMetrics(dog))
report.meets_acceptance()  # accuracy >= 0.85 con conteos enteros

response = report.to_ready_response(
    namespace=...,  # "official" (D06-01) o "synthetic" (recorridos de prueba); visible en la UI
    candidate_run_id=...,  # run_id de MLflow del candidato cerrado (D05-02)
    closed_at=...,  # ISO 8601 con zona: momento de MODEL SELECTION CLOSED
    manifest_hash=...,  # manifest congelado (D03-01)
    evaluated_at=...,  # debe ser posterior a closed_at
)
response.model_dump_json()  # payload `ready` de GET /api/evaluation
```

## Reglas

| Regla | Detalle |
|---|---|
| Orientación | Filas = clase **real**, columnas = clase **predicha**, siempre en el orden congelado `(cat, dog)`, aunque una clase no aparezca en la entrada. |
| accuracy | `traza / n`, sin redondear. |
| support / predicted | Suma de la fila / suma de la columna de la clase. |
| Divisiones 0/0 | Una clase nunca predicha tiene precision 0, y una sin soporte tiene recall 0, sin error. Ambas clases cuentan siempre en el macro-F1. |
| F1 | `2·TP / (predicted + support)`, que equivale a `2PR/(P+R)`. |
| Baseline | Soporte de la clase mayoritaria / n. |
| Aceptación | `correct · 20 >= 17 · n`, es decir `>= 0.85` exacto. 0.8499 **no** pasa, aunque redondeado a 2 decimales sería "0.85". |
| Entradas inválidas | `MetricsInputError`: longitudes distintas, entradas vacías, clases fuera de cat/dog (incluye `"Cat"`, índices `0/1` y `None`) o un string en lugar de una secuencia. |

Se calcula directo de la matriz, sin `sklearn.metrics.f1_score`. Usado sin
`labels=` explícito, scikit-learn calcula el macro-F1 solo sobre las clases
presentes en la entrada. Los tests usan scikit-learn **con** `labels=` como segundo
cálculo independiente.

## Contrato: `presentation.contracts.EVALUATION_RESPONSE`

Es el espejo en Python de `evaluationResponseSchema`
(`backend/src/logic/p3.contracts.ts`, D01-05). Tiene las mismas ramas del
`superRefine` y las mismas tolerancias: 1e-9 para accuracy y baseline, 1e-4 para
precision/recall/F1/macro-F1. `tests/test_evaluation_contract.py` lo valida contra
los fixtures compartidos de `contracts/p3/fixtures/evaluation_response/`.

`to_ready_response` valida con ese mismo modelo, así que un payload que el motor
entrega ya pasó las reglas que aplicarán el backend y la pantalla Evaluation.

## Tests y mutation testing

```bash
cd app
uv run pytest tests/test_evaluation_metrics.py tests/test_evaluation_contract.py
uv run python ../.github/scripts/run_evaluation_metrics_mutations.py   # ~2-3 min
```

El runner trabaja sobre una copia aislada en un directorio temporal y nunca
modifica el árbol de trabajo.

# D04-05 — Productor de la evaluación, API y exportación por muestra

`evaluation.producer` conecta el motor anterior con las guardas de D04-04 y la
persistencia; el backend sirve el resultado con `GET /api/evaluation` y
`GET /api/evaluation/predictions`.

> El frozen test **no** se ejecutó en este ticket. Todo se probó con predicciones
> sintéticas conocidas (6 crops inventados, matriz `[[2, 1], [1, 2]]`) en el namespace
> `synthetic`. La evaluación oficial es de D06-01, después de MODEL SELECTION CLOSED (D05-02).

## Flujo

```text
predicciones por crop ──▶ produce_evaluation ──▶ p3_evaluation (MariaDB) ──▶ API
                            │ 1. guarda: p3_model_selection = closed      │ requireClosed
                            │ 2. compatibilidad (run, manifest, IDs)       │ + coherencia
                            │ 3. motor D03-05 + exportación                │ (503 si no)
```

```python
from evaluation.producer import SamplePrediction, TestPartition, produce_evaluation
from evaluation.store import EvaluationStore

record = produce_evaluation(
    EvaluationStore(engine),  # MariaDB (DATABASE_URL del worker)
    samples,  # SamplePrediction(crop_id, true_class, predicted_class, probabilities)
    namespace="official",  # o "synthetic" para recorridos de prueba
    model_run_id=run_id,  # run de MLflow que produjo las predicciones
    partition=TestPartition.from_manifest(frozen_manifest),  # IDs + hashes del test (D03-01)
)
```

## Guardas y rechazos (`EvaluationRefusedError.reason`)

| Motivo | Cuándo |
|---|---|
| `model_selection_open` | `p3_model_selection` no está `closed`. Se comprueba **antes** de leer predicciones. |
| `incomplete_selection` | Cerrada, pero sin candidato, manifest o `closed_at`. |
| `not_selected_candidate` | Las predicciones son de un run distinto del candidato cerrado. |
| `manifest_mismatch` | La partición es de otro manifest que el de la selección. |
| `test_split_hash_mismatch` | Los `crop_id` de la partición no dan su `test_split_hash`. |
| `duplicate_crop_id` / `crop_ids_not_test_split` | Crops repetidos, faltantes o ajenos a la partición test. |
| `evaluated_before_close` | `evaluated_at` no es posterior al cierre (al milisegundo). |
| `invalid_prediction` | Probabilidades que no suman ~1, clase fuera de cat/dog o `predicted_class` ≠ argmax. |
| `official_already_recorded` | Ya existe la evaluación oficial: se escribe una sola vez. `synthetic` sí se reemplaza. |

## Exportación (`evaluation_predictions`)

Una fila por crop, ordenada por `crop_id`, con la clase real, la predicha, la
probabilidad de cada clase, el candidato, el manifest, el `test_split_hash` y
`evaluated_at`. `GET /api/evaluation/predictions?format=csv` entrega lo mismo como
descarga (`p3-evaluation-predictions-official.csv`), con los hashes en cada fila.

Antes de servir, el backend contrasta lo guardado con el cierre persistido y entre
sí: candidato, `closed_at`, manifest, `evaluated_at`, `n_test`, clases,
`test_split_hash` recalculado y matriz reconstruida desde las predicciones. Si algo no
coincide responde 503 con el motivo. `server.ts` solo monta el namespace `official`.

## Tests y mutation testing

```bash
cd app && uv run pytest tests/test_evaluation_producer.py tests/test_evaluation_predictions_contract.py
cd backend && npx vitest run tests/evaluation.test.ts
python .github/scripts/run_evaluation_api_mutations.py   # copia aislada, Python + TS
```

`backend/tests/fixtures/evaluation-synthetic.json` es la salida exacta del productor
con las entradas sintéticas; el test Python lo exige igual y los tests del backend lo
consumen. Con MariaDB real (job de CI "Jobs persistentes"),
`.github/scripts/evaluation_e2e.py` corre el productor dentro de la imagen de
trainer-worker y `backend/tests/evaluation.mariadb.test.ts` lee esa fila con el
repositorio real.

# D06-01 — Primera y única evaluación oficial del frozen test

`evaluation.official` ejecuta la evaluación `official` **una sola vez**, después del acta
de MODEL SELECTION CLOSED (D05-08). Reutiliza lo ya acreditado: la guarda y el
`EvaluationStore` de D04-05, el motor de métricas de D03-05, el paquete de D05-01 (cargado
con su SHA y su salida de referencia) y las fuentes verificadas de Training (D03-03).

> Las pruebas (`tests/test_evaluation_official.py`) usan un manifest **sintético** (IDs
> 100–119 que no existen en el release), imágenes generadas y pesos de fixture. Ningún ID,
> etiqueta ni métrica del frozen test oficial se lee antes del cierre persistido.

## Orden

| Paso | Qué comprueba | Si falla |
|---|---|---|
| 1. Cierre | `p3_model_selection` está `closed`, con candidato, manifest y `closed_at` | `model_selection_open` / `incomplete_selection` |
| 2. Acta | run, `outcome_hash` y `manifest_hash` del acta = lo persistido | `act_mismatch` |
| 3. Sin repetición | no existe evaluación `official`; si existe, se audita esa | `official_already_recorded` |
| 4. Paquete | carga (el loader exige el class_map congelado), SHA del checkpoint = acta, salida de referencia, run, mejor época, manifest y release del cierre | `package_rejected` / `package_not_candidate` |
| 5. Manifest | el versionado en DVC (md5), su hash y su `test_split_hash` recalculados, igual al cierre y al acta | `manifest_rejected` / `manifest_mismatch` / `test_split_hash_mismatch` |
| 6. Intento | escribe `attempt.json` **antes** de abrir el test; si ya existía, no se repite | `attempt_already_started` |
| 7. Crops | todos y solo los IDs de test, sin duplicados, clase dentro del class_map; **antes** de predecir | `crop_ids_not_test_split` / `duplicate_crop_id` / `label_outside_class_map` |
| 8. Inferencia y guardado | una predicción por crop; `produce_evaluation` en `official` (una sola escritura) | motivos de D04-05 |
| 9. Auditoría | relee lo guardado y recalcula con scikit-learn: matriz, accuracy, macro-F1, por clase, baseline, argmax y cronología | `audit_mismatch` |

Los pasos 1–5 son el `preflight`: no leen ningún crop ni escriben nada. El umbral se compara
con enteros (`correct·100 ≥ 85·n_test`). Un resultado por debajo del 85 % se registra igual
y la selección no se toca.

## Ejecución (una sola vez, tras el acta)

```bash
cd app
export DATABASE_URL=...            # la MariaDB donde quedó el cierre de D05-08
ACTA="--candidate-run-id <run> --checkpoint-sha256 <sha> --manifest-hash <hash> \
      --test-split-hash <hash> --outcome-hash <hash>"
uv run python -m evaluation.official preflight --package <paquete del candidato> $ACTA
uv run python -m evaluation.official run --package <paquete del candidato> $ACTA
```

`run` deja la evidencia en `reports/evaluation_p3/official/`:

- `attempt.json`: inicio, commit y acta;
- `preflight.json`: cierre, paquete y manifest cotejados;
- `trace.jsonl`: una línea por crop con el SHA-256 de la entrada, la verdad, la predicción y
  las probabilidades;
- `evaluation.json` y `predictions.json`: lo que quedó en `p3_evaluation`;
- `audit.json`: el recálculo independiente, el umbral y los `crop_id` con error.

Si la ejecución se interrumpe, `attempt.json` y la traza se conservan y el intento se
audita; no se borra para "empezar de nuevo".
