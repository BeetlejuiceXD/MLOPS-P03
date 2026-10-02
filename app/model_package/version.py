"""D06-02 — Semver del MODELO, separado del `format_version` del paquete
(D05-01) y de la versión del dataset/manifest (#33). Un cambio de versión
de formato o de release de datos no implica, por sí solo, un cambio de
versión del modelo — son tres ejes independientes.

Esta es solo la representación (`major.minor.patch` + parseo/formato); qué
incrementar y cuándo es una decisión del acta (D05-08), no algo que este
módulo decida por su cuenta.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PATTERN = re.compile(r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)$")


@dataclass(frozen=True)
class ModelVersion:
    major: int
    minor: int
    patch: int

    def __post_init__(self) -> None:
        for name, value in (("major", self.major), ("minor", self.minor), ("patch", self.patch)):
            if value < 0:
                raise ValueError(f"ModelVersion.{name} no puede ser negativo: {value}")

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    @classmethod
    def parse(cls, value: str) -> ModelVersion:
        match = _PATTERN.match(value)
        if not match:
            raise ValueError(f"{value!r} no es un semver major.minor.patch válido")
        return cls(major=int(match["major"]), minor=int(match["minor"]), patch=int(match["patch"]))
