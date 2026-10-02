# D05-01 — Paquete smoke y loader autocontenido

Recarga un checkpoint real **fuera del proceso de entrenamiento**, sin memoria ni estado
del trainer. El paquete es un directorio versionado; el loader lo valida completo antes
de instanciar la CNN.

## Formato `p3-model-package` 1.0.0 (`kind: smoke`)

| Archivo | Contenido |
|---|---|
| `package.json` | Manifiesto: `format`, `format_version`, `kind`, `package_id`, `source` y el **inventario** (`sha256` + `size_bytes` de cada archivo) |
| `model.pt` | Los pesos del checkpoint de origen **tal cual**, es decir `checkpoint/model.pt` de la mejor época del run de D03-04. No se reconstruyen ni se toma otra época |
| `training_config.json` | Arquitectura/config de D01-04 (`TrainingConfig`) |
| `class_map.json` | `{"cat": 0, "dog": 1}`, el orden congelado de #33 |
| `preprocessing.json` | El transform de evaluación de D01-04, descrito: RGB, resize a `image_size`, to_tensor, normalize ImageNet |
| `dependencies.json` | Versiones de `python`, `torch`, `torchvision` y `pillow` para recargar, el `sha256` de `app/uv.lock` y el `environment.json` del training |
| `reference_output.json` | Salida del checkpoint de origen para una entrada conocida (`reference_image()`), calculada al construir con la CNN de D01-04 y el `state_dict` cargado directo, sin el loader |
| `smoke_card.json` | Tarjeta smoke con `official_test_metrics: null`: todavía no hay métricas oficiales de test |

`source` en `package.json` liga el paquete a su origen con estos campos:
- `mlflow_run_id`, `experiment` y `checkpoint_artifact` (`checkpoint/model.pt`);
- `checkpoint_sha256` y `best_epoch`;
- `dataset_version`, `manifest_version`, `manifest_hash` y `dvc_release_hash`.

`format_version` es la versión del **formato**, no el semver del modelo. Ese semver lo
asigna D06-02 al package final.

## Loader

```python
from model_package import load_package

package = load_package("smoke-package")
package.identity()  # format_version, kind, package_id, run_id, sha, best_epoch, classes…
package.predict(pil_image)  # {"predicted_class": "dog", "probabilities": {"cat": …, "dog": …}}
package.check_reference()  # {"matches": True, "max_abs_diff": …}
```

El loader hace las comprobaciones en este orden, y **todas pasan antes de instanciar el
modelo**:
1. Manifiesto válido y `format_version` 1.x.
2. Inventario exacto: no puede faltar ni sobrar ningún archivo.
3. Tamaño de cada archivo (detecta pesos truncados) y SHA-256 de cada uno.
4. `model.pt` debe ser el `checkpoint_sha256` del origen.
5. Dependencias instaladas, con el mismo MAJOR.MINOR.
6. `training_config.json` válido para la CNN de D01-04.
7. `class_map` igual al congelado.
8. `preprocessing` igual al transform de evaluación de D01-04 para esa config.

Si todo pasa, construye la CNN con `build_model`, sin descargar pesos preentrenados, y
carga los pesos con `strict=True`. Cualquier discrepancia lanza `PackageError` y no se
sirve ninguna predicción. **No hay fallback**: el loader nunca busca otro archivo ni
otro checkpoint.

## CLI

```bash
# Desde el run real (el checkpoint se descarga por el servidor de MLflow):
python -m model_package build --run-id <run_id> --tracking-uri http://mlflow:5000 --out /tmp/smoke-package
# Proceso limpio: identidad, inventario, predicción y cotejo con la salida de referencia
python -m model_package predict --package /tmp/smoke-package
```

`build` exige que el run sea `p3.run_kind=training`, que esté `FINISHED` y que pertenezca
a `p3-cnn-classifier`. También exige que el `model.pt` descargado coincida con el tag
`checkpoint_sha256` y con `sources.json`. Nunca sobrescribe un paquete existente.

## Consumidores

- **D05-04:** usa `identity()` y `predict()`.
- **D05-06:** registra el paquete con el script de `local_test`.
- **D06-02:** reutiliza el formato para el package final, con su semver y sus métricas
  oficiales.

Los tests (`tests/test_model_package.py`) usan pesos de **fixture**. Prueban el
componente, no acreditan el checkpoint real.
