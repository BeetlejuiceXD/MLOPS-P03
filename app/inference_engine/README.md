# D05-04 — Motor de inferencia

Clasifica imágenes con un paquete de D05-01 (`app/model_package/`). No reconstruye nada:
la CNN, los pesos, la transformación de evaluación y el `class_map` salen del paquete, que
`load_package` valida completo antes de instanciar el modelo.

## Interfaz (independiente de HTTP)

```python
from inference_engine import InferenceEngine

engine = InferenceEngine.from_package("smoke-package", expected_checkpoint_sha256="0c6b…563b")
engine.identity()  # {"model": {...}, "classes": ["cat", "dog"], "image_size": 224}
prediction = engine.predict(image_bytes)
prediction.contract()  # {"predicted_class", "probabilities", "model"}
prediction.input_sha256  # trazabilidad de la entrada
engine.load("otro-paquete")  # D06-04: misma instancia, misma lógica de predicción
```

Al cargar, el motor:
1. Rechaza un paquete que no existe.
2. Llama a `load_package` (inventario, SHA-256, dependencias, config, `class_map` y
   preprocessing).
3. Compara el `checkpoint_sha256` con el esperado, si se pasa.
4. Exige que el paquete reproduzca su `reference_output.json`.

Si algo falla, el motor queda **sin modelo**: no sigue sirviendo el paquete anterior.

`model` es la identidad `InferenceModelIdentity` de D05-07. En un paquete smoke,
`source` es `smoke` y `model_version` es `null`, porque el semver lo asigna D06-02. La
identidad sale del mismo `LoadedPackage` que predijo, así que una predicción nunca se
atribuye a otro run, aunque se recargue el motor a la mitad.

## Errores

| Excepción | Cuándo | HTTP |
|---|---|---|
| `InputRejected` | Imagen vacía, no decodificable o truncada, formato fuera de JPEG/PNG/WEBP, más de 10 MiB | 400 (415 si el `Content-Type` no es `image/*`, 413 si excede el tamaño) |
| `PackageRejected` | Sin paquete, paquete faltante, rechazado por el loader, otro SHA o no reproduce su referencia | 503 |
| `InferenceFailed` | El modelo falla, o su salida no tiene las clases del `class_map` en orden, no suma ~1, tiene valores fuera de [0, 1] o la clase no es el argmax | 500 |

Ninguno emite predicción.

## HTTP

`python -m inference_engine serve --package <dir> [--port 8090] [--expected-sha256 <sha>]`
expone el contrato que consume D05-07 (`backend/src/logic/inference-engine.ts`):

- `GET /identity` devuelve `inference_engine`.
- `POST /predict` recibe los bytes de la imagen con `Content-Type: image/*` y devuelve
  `inference_engine_prediction`, sin campos extra.
- `GET /health` sirve como chequeo de vida.

Solo usa la biblioteca estándar. Cada predicción queda en el log con el SHA-256 de la
entrada, la clase, las probabilidades y el run.

## CLI de evidencia

`python -m inference_engine predict --package <dir> --image a.jpg [--image b.jpg]
[--expected-sha256 <sha>]` escribe JSON con la identidad del motor y, por cada imagen,
el SHA-256, el formato, el tamaño, la clase, las probabilidades y la identidad. Los
códigos de salida son: 0 ok, 1 paquete rechazado, 3 alguna imagen rechazada.

Los tests (`tests/test_inference_engine.py`) usan pesos de **fixture**. El paquete smoke
real se acredita aparte, con su run_id y su SHA.
