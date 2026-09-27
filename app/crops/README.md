# D01-07 — Motor de crops COCO deterministas

`generate_crops(coco, image_contents)` convierte cada anotación COCO válida en un
`Crop` (una muestra del clasificador: recorte + etiqueta + procedencia), o la excluye
con un motivo auditable. Es una función pura: recibe el COCO crudo (`dict`) y los
bytes de imagen ya cargados, igual que `analyzers.duplicates`/`splits.stratified`; no
abre archivos, no usa MinIO/S3 ni variables de entorno.

```python
from crops.engine import generate_crops

# coco: dict con "images"/"annotations"/"categories", en la forma cruda del COCO.
# image_contents: dict[image_id, bytes], no necesita cubrir todos los image_id.
result = generate_crops(coco, image_contents)
result.crops  # list[Crop]: una muestra por anotación válida
result.exclusions  # list[CropExclusion]: cada anotación descartada, con motivo
```

## Por qué no usa `ingestion.models.CocoDataset`

`CocoDataset.model_validate` (Tier 1) es estricto a propósito: una sola caja
degenerada o una categoría desconocida rechaza el lote **completo**, antes de que
`analyzers`/`policies` lo vean (ver `ingestion/README.md`). Eso es correcto para
detectar un bug de ingesta, pero no sirve para producir crops: una caja mala real
no debe impedir generar los cientos de crops válidos del resto del dataset.

Por eso `generate_crops` recibe el COCO crudo y evalúa cada anotación de forma
independiente. No debilita las garantías de P2: no se toca `ingestion/models.py` ni
`ingestion/loader.py`, y una caja que `CocoDataset` habría rechazado aquí simplemente
termina en `exclusions`, con su motivo, nunca reparada en silencio.

## Motivos de exclusión

Ninguna anotación se pierde en silencio: `len(crops) + len(exclusions) ==
len(coco["annotations"])` siempre.

| Motivo | Regla | Depende de la imagen |
|---|---|---|
| `unknown_category` | `category_id` no está en `categories` | no |
| `degenerate_bbox` | forma inválida, `width`/`height` no positivos, o valores no finitos | no |
| `missing_image` | `image_id` no declarado, sin binario en `image_contents`, o binario no decodificable | — |
| `out_of_bounds` | la caja se sale de la imagen — **total o parcialmente** — de sus límites reales | sí |

Los límites se verifican contra el tamaño **real** de la imagen decodificada, no
contra `width`/`height` declarados en el COCO: un metadato desactualizado no debe
validar una caja que en realidad se sale de la imagen.

Una caja **parcialmente** fuera de imagen se excluye igual que una totalmente fuera
(decisión de protocolo D01-03, issue #33): no se recorta al borde, para no alterar la
geometría anotada por el equipo de anotación.

## Redondeo y procedencia

- `x0, y0 = floor(x), floor(y)`; `x1, y1 = ceil(x + width), ceil(y + height)`. Con `width`/`height`
  positivos (ya filtrados por `degenerate_bbox`) y el límite superior ya verificado contra la
  imagen real, el crop siempre mide al menos 1 px por lado; `Crop.width`/`Crop.height`
  (`Field(gt=0)`) son la red de seguridad del modelo, no una regla de negocio aparte.
- `Crop.bbox_original` conserva `[x, y, width, height]` tal cual el COCO (auditoría);
  `Crop.bbox_pixels` es `[x0, y0, x1, y1]` en píxeles enteros.
- `Crop.crop_id` es hoy el mismo valor que `annotation_id`: una caja válida produce
  exactamente un crop. Se expone como campo propio para no acoplar a los llamadores
  a esa igualdad si en el futuro un crop deja de ser 1:1 con su anotación (por
  ejemplo, augmentation).

## Qué no hace (todavía)

- No decide splits ni pertenece a un manifest 70/20/10 — eso es P3-06 (D02+), que
  consume `result.crops` como entrada.
- No persiste crops en disco ni produce bytes de imagen recortada: solo geometría y
  metadatos. `data/crops/` está excluido del índice de Git (`tests/_repo_hygiene_rules.py`,
  D01-06) hasta que se decida su ubicación/seguimiento — mismo criterio que
  `splits/README.md` aplicó a `splits.json` antes de P2-45.
- No se ha corrido contra el dataset real de producción como parte de la aceptación
  de este ticket (motor probado con fixtures); `tests/test_crops.py` incluye un caso
  marcado que sí corre contra `data/raw` cuando está recuperado con DVC, como
  evidencia adicional, no como sustituto del alcance de D01-07.
