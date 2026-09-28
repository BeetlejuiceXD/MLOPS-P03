"""D02-04 — manifest P3 candidato: partición 70/20/10 por grupos indivisibles,
medida en CROPS (no en imágenes). Función pura, sin I/O.

Adapta el algoritmo ya probado de `splits.stratified.split_dataset` (P2, Tier 4):
mismo enfoque de unión de grupos indivisibles + asignación greedy con reparación
por búsqueda local, y el mismo criterio de desempate por semilla. Dos diferencias
deliberadas frente a P2:

- El tamaño que se balancea es el número de **crops** de cada grupo, no el número
  de imágenes: un grupo de una imagen con 5 crops "pesa" 5 al calcular
  proporciones, no 1 (#33/D02-04: "proporciones sobre crops").
- Cada crop tiene una sola clase (ya resuelta por `crops.engine.generate_crops`,
  D01-07), no un conjunto de categorías por imagen como en P2: el conteo de clase
  por grupo es un `Counter` de una sola entrada por crop, no un conjunto.

Los "grupos indivisibles" son: todos los crops de la misma imagen original, unidos
con cualquier otra imagen marcada como near-duplicate (`analyzers.duplicates`,
mismo umbral pHash de la política — 0.94 hoy). No se repite la deduplicación por
bytes idénticos/`file_name` de `splits.stratified` (P2-22/23/24, un bug de ingesta
distinto): un pHash de bytes idénticos ya da distancia 0 y similitud 1.0, por
encima de cualquier umbral razonable, así que ese caso queda cubierto igual.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isclose
from random import Random
from types import MappingProxyType

from crops.models import Crop

from splits.models import SplitsConfig

SPLIT_NAMES = ("train", "val", "test")


class ManifestValidationError(ValueError):
    """Un candidato de manifest no cumple sus invariantes (fuga, cobertura, clase
    ausente en val/test, etc.). No se congela nada al lanzarla: es una señal de
    que este candidato no se puede aceptar, no de que el pipeline deba abortar."""


@dataclass(frozen=True)
class ManifestCandidateResult:
    """Resultado en memoria de `generate_manifest`; no es el contrato `ManifestSummary`
    (D01-05) — eso lo arma `presentation.manifest_candidate` con la identidad de
    release/DVC. `assignments`/`originals` están en crop_id/image_id."""

    assignments: Mapping[str, tuple[int, ...]]
    originals: Mapping[str, tuple[int, ...]]
    groups: tuple[tuple[int, ...], ...]
    crops_per_class: Mapping[str, Mapping[str, int]]
    seed: int

    def __post_init__(self):
        object.__setattr__(
            self,
            "assignments",
            MappingProxyType({name: tuple(ids) for name, ids in self.assignments.items()}),
        )
        object.__setattr__(
            self,
            "originals",
            MappingProxyType({name: tuple(ids) for name, ids in self.originals.items()}),
        )
        object.__setattr__(self, "groups", tuple(tuple(group) for group in self.groups))
        object.__setattr__(
            self,
            "crops_per_class",
            MappingProxyType(
                {
                    name: MappingProxyType(dict(counts))
                    for name, counts in self.crops_per_class.items()
                }
            ),
        )


def verify_manifest_assignment(
    assignments: Mapping[str, Sequence[int]],
    *,
    crop_ids: Sequence[int],
    groups: Sequence[Sequence[int]],
    classes: Sequence[str],
    crops_per_class: Mapping[str, Mapping[str, int]],
) -> None:
    """Rechaza omisiones, IDs extra/repetidos, splits vacíos, grupos separados
    (fuga) y una clase ausente en val/test. `groups` son grupos de **crop_id**
    (ya expandidos desde los grupos de imagen); el llamador no debe reconstruirlos
    desde una asignación potencialmente contaminada."""
    if set(assignments) != set(SPLIT_NAMES):
        raise ManifestValidationError("Expected exactly train, val and test assignments")
    if any(type(i) is not int or i < 0 for i in crop_ids):
        raise ManifestValidationError("Expected non-negative integer crop IDs")
    expected = set(crop_ids)
    if len(expected) != len(crop_ids):
        raise ManifestValidationError("Duplicate crop IDs in input")
    owner = _assignment_owners(assignments, expected)
    _verify_groups(groups, expected, owner)
    _verify_class_presence_in_val_and_test(classes, crops_per_class)


def _assignment_owners(assignments, expected):
    owner = {}
    for name in SPLIT_NAMES:
        if not assignments[name]:
            raise ManifestValidationError("All three splits must be non-empty")
        for crop_id in assignments[name]:
            if type(crop_id) is not int or crop_id not in expected:
                raise ManifestValidationError(f"Unknown crop ID in assignment: {crop_id}")
            if crop_id in owner:
                raise ManifestValidationError(f"Crop ID assigned more than once: {crop_id}")
            owner[crop_id] = name
    if set(owner) != expected:
        raise ManifestValidationError("Assignment does not cover all input crops")
    return owner


def _verify_groups(groups, expected, owner):
    grouped = set()
    for group in groups:
        if not group:
            raise ManifestValidationError("Groups must be non-empty")
        for crop_id in group:
            if type(crop_id) is not int or crop_id not in expected or crop_id in grouped:
                raise ManifestValidationError("Groups must partition the input crop IDs")
            grouped.add(crop_id)
        if len({owner[crop_id] for crop_id in group}) != 1:
            raise ManifestValidationError("Leakage: a group crosses splits")
    if grouped != expected:
        raise ManifestValidationError("Groups do not cover all input crops")


def _verify_class_presence_in_val_and_test(classes, crops_per_class):
    for name in ("val", "test"):
        for class_name in classes:
            if crops_per_class[name].get(class_name, 0) == 0:
                raise ManifestValidationError(f"Class {class_name!r} must be present in {name}")


def _image_groups(
    image_ids: set[int], duplicate_pairs: Sequence[Mapping]
) -> tuple[tuple[int, ...], ...]:
    """Componentes conexas de `image_ids`, unidas por los pares pHash. Una imagen
    sin ningún crop válido (ya excluida por el motor de D01-07) puede seguir
    participando como nodo PUENTE de la unión: si A≈B≈C y B no tiene crops, A y C
    deben quedar en el mismo grupo igual (revisión de Heri en PR #54). El nodo
    puente en sí nunca aparece en el resultado: la proyección final solo recorre
    `image_ids`."""
    parent: dict[int, int] = {}

    def find(i):
        parent.setdefault(i, i)
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for pair in duplicate_pairs:
        if not isinstance(pair, Mapping) or not {"image_id_a", "image_id_b"} <= pair.keys():
            raise ManifestValidationError("Each duplicate pair requires image_id_a and image_id_b")
        a, b = pair["image_id_a"], pair["image_id_b"]
        if a == b:
            raise ManifestValidationError("A duplicate pair must reference two distinct images")
        union(a, b)

    components: dict[int, list[int]] = {}
    for image_id in image_ids:
        components.setdefault(find(image_id), []).append(image_id)
    return tuple(sorted(tuple(sorted(group)) for group in components.values()))


def generate_manifest(
    crops: Sequence[Crop],
    config: SplitsConfig,
    *,
    duplicate_pairs: Sequence[Mapping],
) -> ManifestCandidateResult:
    """Genera el candidato de manifest P3 a partir de crops ya válidos (D01-07/D02-02).

    `duplicate_pairs` es `analyze_duplicates(image_contents, ...).details["image_pairs"]`
    sobre las MISMAS imágenes de origen de esos crops (mismo contrato que
    `splits.stratified.split_dataset`). No decodifica imágenes ni recalcula pHash.
    """
    config = SplitsConfig.model_validate(config.model_dump())
    if not crops:
        raise ManifestValidationError("No hay crops de entrada para generar un manifest")

    crops_by_image: dict[int, list[Crop]] = {}
    for crop in crops:
        crops_by_image.setdefault(crop.image_id, []).append(crop)
    image_ids = set(crops_by_image)

    groups = _image_groups(image_ids, duplicate_pairs)
    if len(groups) < 3:
        raise ManifestValidationError(
            "Need at least three independent original-image groups for three non-empty splits"
        )

    classes = sorted({crop.category_name for crop in crops})
    total_crops = len(crops)
    class_totals = Counter(crop.category_name for crop in crops)
    group_crop_counts = {
        group: Counter(
            crop.category_name for image_id in group for crop in crops_by_image[image_id]
        )
        for group in groups
    }
    group_sizes = {group: sum(group_crop_counts[group].values()) for group in groups}

    ratios = {name: getattr(config, name) for name in SPLIT_NAMES}
    total_ratio = sum(ratios.values())
    ratios = {name: value / total_ratio for name, value in ratios.items()}
    targets = {name: ratios[name] * total_crops for name in SPLIT_NAMES}
    target_sizes = {name: int(targets[name]) for name in SPLIT_NAMES}
    rng = Random(config.seed)  # Solo desempate de asignación; nunca secretos ni criptografía.

    def choose_tie(items):
        return items[rng.randrange(len(items))] if len(items) > 1 else items[0]

    remainder_order = list(SPLIT_NAMES)
    rng.shuffle(remainder_order)
    remainder_order.sort(key=lambda name: -(targets[name] - target_sizes[name]))
    for name in remainder_order[: total_crops - sum(target_sizes.values())]:
        target_sizes[name] += 1
    target_classes = {
        name: {c: ratios[name] * class_totals[c] for c in classes} for name in SPLIT_NAMES
    }

    image_assignments = {name: [] for name in SPLIT_NAMES}
    counts = {name: Counter() for name in SPLIT_NAMES}

    def cost(name, group, direction=1):
        size_error = (
            sum(len(crops_by_image[i]) for i in image_assignments[name]) - target_sizes[name]
        )
        delta = ((size_error + direction * group_sizes[group]) ** 2 - size_error**2) / total_crops
        for c, count in group_crop_counts[group].items():
            error = counts[name][c] - target_classes[name][c]
            delta += ((error + direction * count) ** 2 - error**2) / class_totals[c]
        return delta

    _assign_initial(
        groups, class_totals, group_crop_counts, image_assignments, counts, cost, choose_tie
    )
    _improve_assignment(groups, image_assignments, counts, group_crop_counts, cost, choose_tie)

    originals = {name: tuple(sorted(ids)) for name, ids in image_assignments.items()}
    assignments = {
        name: tuple(sorted(crop.crop_id for image_id in ids for crop in crops_by_image[image_id]))
        for name, ids in originals.items()
    }
    crops_per_class = {name: dict(counts[name]) for name in SPLIT_NAMES}
    crop_groups = tuple(
        tuple(sorted(crop.crop_id for image_id in group for crop in crops_by_image[image_id]))
        for group in groups
    )

    verify_manifest_assignment(
        assignments,
        crop_ids=sorted(crop.crop_id for crop in crops),
        groups=crop_groups,
        classes=classes,
        crops_per_class=crops_per_class,
    )

    return ManifestCandidateResult(
        assignments=assignments,
        originals=originals,
        groups=crop_groups,
        crops_per_class=crops_per_class,
        seed=config.seed,
    )


def _assign_initial(groups, class_totals, group_crop_counts, assignments, counts, cost, choose_tie):
    pending = list(groups)
    remaining = class_totals.copy()
    while pending:

        def priority(group):
            rarity = min((remaining[c] for c in group_crop_counts[group]), default=float("inf"))
            return rarity, -sum(group_crop_counts[group].values())

        best_priority = min(priority(group) for group in pending)
        group = choose_tie([g for g in pending if priority(g) == best_priority])
        empty = [name for name in SPLIT_NAMES if not assignments[name]]
        candidates = empty if len(pending) == len(empty) else list(SPLIT_NAMES)

        costs = {name: cost(name, group) for name in candidates}
        best_cost = min(costs.values())
        name = choose_tie(
            [
                name
                for name in candidates
                if isclose(costs[name], best_cost, abs_tol=1e-12, rel_tol=0)
            ]
        )
        assignments[name].extend(group)
        counts[name].update(group_crop_counts[group])
        remaining.subtract(group_crop_counts[group])
        pending.remove(group)


def _improving_moves(groups, assignments, cost):
    moves = []
    for group in groups:
        source = next(name for name in SPLIT_NAMES if group[0] in assignments[name])
        if len(assignments[source]) == len(group):
            continue  # este grupo es todo el contenido de `source`; moverlo lo vaciaría.
        for destination in SPLIT_NAMES:
            if destination != source:
                delta = cost(source, group, -1) + cost(destination, group)
                if delta < -1e-12:
                    moves.append((delta, group, source, destination))
    return moves


def _improve_assignment(groups, assignments, counts, group_crop_counts, cost, choose_tie):
    while True:
        moves = _improving_moves(groups, assignments, cost)
        if not moves:
            break
        best_delta = min(move[0] for move in moves)
        _, group, source, destination = choose_tie(
            [move for move in moves if isclose(move[0], best_delta, abs_tol=1e-12, rel_tol=0)]
        )
        for image_id in group:
            assignments[source].remove(image_id)
        assignments[destination].extend(group)
        counts[source].subtract(group_crop_counts[group])
        counts[destination].update(group_crop_counts[group])
