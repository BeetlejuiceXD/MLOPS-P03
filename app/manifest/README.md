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

## Qué no hace (todavía)

- No decide si el manifest se congela — D03-01 hace la auditoría y congelación
  oficial; este ticket entrega un candidato auditable, siempre con `frozen=False`.
- No persiste el candidato en `reports/` ni agrega una etapa a `dvc.yaml`: sigue el
  mismo patrón que `presentation.release_resolver`/`presentation.crops_report`
  (CLI manual, reproducible, no integrado al pipeline DVC).
- No distribuye las etiquetas del test al trainer: `build_manifest_candidate`
  puede validar la integridad de la partición test (IDs, cobertura), pero no
  evalúa modelos ni expone resultados del frozen test.
