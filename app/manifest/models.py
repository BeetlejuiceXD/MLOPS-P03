"""Configuración del manifest P3 (D02-04): mismo modelo que `splits.models.SplitsConfig`
(proporciones que suman 1.0 y semilla reproducible), pero cargado de `manifest.yaml`
propio — nunca de `splits/splits.yaml`, que sigue siendo el split 70/15/15 de P2
(#33: "P2 conserva su split 70/15/15... P3 genera un manifest derivado 70/20/10 con
identidad propia")."""

from pathlib import Path

import yaml

from splits.models import SplitsConfig


def load_manifest_config(path: Path | None = None) -> SplitsConfig:
    """Lee `manifest.yaml` completo y lo valida; nunca se usa el dict crudo."""
    config_path = path if path is not None else Path(__file__).with_name("manifest.yaml")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return SplitsConfig.model_validate(raw)
