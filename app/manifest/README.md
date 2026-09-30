# D02-04 — Manifest P3 candidato (70/20/10 sobre crops)

`generate_manifest(crops, config, duplicate_pairs=...)` parte los crops reales de
`crops.engine.generate_crops` (D01-07/D02-02) en train/val/test, midiendo las
proporciones en **crops**, no en imágenes, con grupos indivisibles (mismo original +
near-duplicates). Es una función pura, sin I/O — la orquestación real con release,
crops y duplicados reales vive en `presentation.manifest_candidate`.

```bash
cd app
uv run python -m presentation.manifest_candidate v0.1.1
```

## Por qué es un paquete propio, no una extensión de `splits/`

`splits/stratified.py` (Tier 4, P2) ya resuelve un problema muy parecido: agrupar
por original+duplicados y asignar por costo con semilla reproducible. D02-04 lo
adapta, no lo reutiliza directamente, por dos diferencias deliberadas que #33
congela para P3:

- El tamaño que se balancea es el número de **crops** de cada grupo, no el número
  de imágenes: una imagen con 5 crops pesa 5, no 1.
- Cada crop tiene una sola clase (ya resuelta por el motor de crops), no un
  conjunto de categorías por imagen como en P2.

`splits/splits.yaml` (70/15/15) nunca se toca: el manifest de P3 tiene su propia
configuración (`manifest/manifest.yaml`, 70/20/10, seed 42) e identidad propia.

## Contrato de salida: `ManifestSummary` (D01-05)

`presentation.manifest_candidate.build_manifest_candidate` arma un
`presentation.contracts.ManifestSummary`: el espejo exacto en Python de
`manifestSummarySchema` (`backend/src/logic/p3.contracts.ts`), validado contra los
mismos fixtures reales de `contracts/p3/fixtures/manifest_summary/` que usan los
tests de backend y frontend (`app/tests/test_manifest_contract.py`). `frozen` es
siempre `False` desde este ticket — la congelación oficial (`frozen: true`) es de
D03-01.

## Bloqueos: cuándo no hay candidato

`build_manifest_candidate` no produce nunca un `ManifestSummary` que viole el
contrato. Dos rutas de rechazo, ambas como `ManifestBlockedError`:

| `reason` | Cuándo |
|---|---|
| `insufficient_originals_after_exclusions` | El gate de `crops_report` (D02-02) está bloqueado: alguna clase quedó bajo el mínimo tras las exclusiones reales (#33). |
| `invalid_partition` | El generador no pudo cubrir `cat`/`dog` en val y test, o ningún grupo indivisible cabe dentro de ±5 pp de 70/20/10 sin partirse. |

`ReleaseRejectedError` (D01-02) se sigue propagando tal cual si la versión no es
elegible.

## Grupos indivisibles

Todos los crops de la misma imagen original, unidos con cualquier otra imagen
marcada como near-duplicate (`analyzers.duplicates`, mismo umbral pHash de la
política — 0.94 hoy). No se repite la deduplicación por bytes idénticos/`file_name`
de `splits.stratified` (un bug de ingesta distinto, P2-22/23/24): un pHash de bytes
idénticos ya da similitud 1.0, por encima de cualquier umbral razonable.

## Manifest congelado (D03-01)

`presentation.manifest_freeze` congela el candidato de D02-04 **sin reconstruirlo**:
mismo split, `frozen: true`. `manifest_hash` no incluye `frozen`, así que la
identidad del candidato y la del manifest congelado es la misma.

```bash
cd app
# Congelar (falla sin escribir nada si la auditoría bloquea)
uv run python -m presentation.manifest_freeze freeze v0.1.1
cd .. && dvc add data/p3/manifest.json && dvc push -r prod data/p3/manifest.json

# Reauditar lo congelado (p. ej. tras un dvc pull): .dvc, sha256, reglas y regeneración
cd app && uv run python -m presentation.manifest_freeze audit v0.1.1
```

| Archivo | Dónde vive | Quién lo usa |
|---|---|---|
| `data/p3/manifest.json` | caché DVC (remote `prod`); `manifest.json.dvc` en git | Training (D03-03, `trainer_worker.sources.verify_training_sources`) |
| `reports/manifest_p3.json` | git | `ManifestSummary` (D01-05) con `frozen: true` |
| `reports/manifest_p3_freeze.json` | git | identidad del release, sha256/md5 del artefacto, `test_split_hash`, conteos por clase y grupos |

`data/p3/manifest.json` sigue el contrato `FrozenManifest` de
`presentation/contracts.py`: identidad DVC del release, `seed`, `target_ratios`,
`frozen: true`, `test_split_hash` (sha256 de los `crop_id` de test ordenados) y los
`crop_id` de cada partición. Solo IDs: Training usa los de test para verificar
cobertura y fuga, y nunca recorta ni carga sus píxeles.

La auditoría (`FreezeBlockedError.reason`) es independiente del generador y bloquea
la congelación ante: artefacto que no cumple el contrato o que no coincide con su
`.dvc` o con el sha256 registrado, resumen publicado sin `frozen: true`, hash del release, del candidato o del split test
adulterado, identidad distinta entre artefacto y resumen, crops sin asignar o
desconocidos, originales o grupos near-duplicate que cruzan particiones (incluidos
los puentes sin crops), proporciones fuera de ±5 pp sobre crops, clase ausente en
cualquier partición, menos de 300 originales por clase tras exclusiones, un resumen
cuyos conteos no coinciden con las asignaciones, o un artefacto coherente que no es
el candidato regenerado (`candidate_drift`).

**Mutation testing (D03-01).** `.github/scripts/run_manifest_freeze_mutations.py`
aplica 19 mutantes (M01–M19) a `presentation/manifest_freeze.py`, uno por uno. Con cada
uno corre `tests/test_manifest_freeze.py` completo y restaura el archivo al terminar.
El test contra v0.1.1 real se excluye para que el resultado no dependa de `data/raw`.
La base sin mutantes tiene que estar en verde.

Del reporte JUnit de pytest el script saca qué tests fallan y cómo:

- **aserción**: veredicto explícito del test: `AssertionError`, `pytest.fail` o `pytest.raises` que no se cumplió.
- **excepción**: el test terminó con una excepción no esperada.

Con eso clasifica cada mutante:

| Estado | Cuándo |
|---|---|
| **KILLED** | Al menos un test falló por aserción y no hubo errores de colección o setup. |
| **KILLED (solo excepción)** | Hubo fallos, pero ninguno fue por aserción. |
| **SURVIVED** | Todos los tests pasaron. |
| **ERROR** | Hubo un error de colección o setup, `rc` no fue 0 ni 1, o el texto a mutar no aparece exactamente una vez. |

Un ERROR nunca cuenta como muerto. El script sale con 0 solo si todos los mutantes quedan KILLED.

```bash
cd app && uv run python ../.github/scripts/run_manifest_freeze_mutations.py   # ~20-30 min
```

## Qué no hace (todavía)

- `build_manifest_candidate` (D02-04) sigue entregando `frozen=False`: congelar y
  persistir es responsabilidad de `presentation.manifest_freeze` (D03-01).
- No distribuye las etiquetas del test al trainer: `build_manifest_candidate`
  puede validar la integridad de la partición test (IDs, cobertura), pero no
  evalúa modelos ni expone resultados del frozen test.
