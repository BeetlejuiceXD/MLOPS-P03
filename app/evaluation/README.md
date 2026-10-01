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
report.rows                        # ((tp_cat, cat→dog), (dog→cat, tp_dog))
report.accuracy, report.macro_f1   # sin redondear
report.per_class                   # (ClassMetrics(cat), ClassMetrics(dog))
report.meets_acceptance()          # accuracy >= 0.85 con conteos enteros

response = report.to_ready_response(
    candidate_run_id=...,   # run_id de MLflow del candidato cerrado (D05-02)
    closed_at=...,          # ISO 8601 con zona: momento de MODEL SELECTION CLOSED
    manifest_hash=...,      # manifest congelado (D03-01)
    evaluated_at=...,       # debe ser posterior a closed_at
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
