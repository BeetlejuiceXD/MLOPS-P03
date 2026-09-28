"""Frozen JSON v1.0 contracts; validation only, without pipeline execution or I/O."""

from math import isclose
from re import fullmatch
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    ValidationInfo,
    field_validator,
    model_validator,
)

from analyzers.base import AnalyzerResult

Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")]
Count = Annotated[int, Field(ge=0)]
Ratio = Annotated[float, Field(ge=0, le=1)]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class QualityCheck(AnalyzerResult):
    """Analyzer output plus the policy action supplied by the upstream gate."""

    model_config = ContractModel.model_config

    check_name: Identifier
    metric_value: float
    details: dict[str, JsonValue] = Field(default_factory=dict)
    action: Literal["warn", "fail"]


class QualityReport(ContractModel):
    schema_version: Literal["1.0"]
    dataset_version: Identifier
    status: Literal["passed", "warning", "failed"]
    checks: list[QualityCheck] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_check_names(self) -> Self:
        names = [check.check_name for check in self.checks]
        if len(names) != len(set(names)):
            raise ValueError("check_name must be unique within a quality report")
        return self


class SplitSummary(ContractModel):
    image_count: Count
    ratio: Ratio


class DatasetSplits(ContractModel):
    train: SplitSummary
    validation: SplitSummary
    test: SplitSummary


class SplitsReport(ContractModel):
    schema_version: Literal["1.0"]
    dataset_version: Identifier
    total_images: int = Field(gt=0)
    splits: DatasetSplits
    class_distribution: dict[str, dict[str, Count]] = Field(
        default_factory=dict, exclude_if=lambda value: not value
    )
    leakage: dict[str, JsonValue] = Field(default_factory=dict, exclude_if=lambda value: not value)

    @model_validator(mode="after")
    def consistent_split_summary(self) -> Self:
        splits = (self.splits.train, self.splits.validation, self.splits.test)
        if sum(split.image_count for split in splits) != self.total_images:
            raise ValueError("split counts must sum to total_images")
        for split in splits:
            if not isclose(
                split.ratio, split.image_count / self.total_images, rel_tol=0, abs_tol=1e-6
            ):
                raise ValueError("each ratio must match image_count / total_images within 1e-6")
        return self


class DatasetRelease(ContractModel):
    dataset_version: Identifier
    quality_file: str
    splits_file: str

    @field_validator("quality_file", "splits_file")
    @classmethod
    def relative_report_reference(cls, value: str, info: ValidationInfo) -> str:
        expected = "quality.json" if info.field_name == "quality_file" else "splits.json"
        parts = value.split("/")
        if (
            parts[-1] != expected
            or any(part in ("", ".", "..") for part in parts)
            or any(fullmatch(r"[A-Za-z0-9._-]+", part) is None for part in parts)
        ):
            raise ValueError(f"reference must be a relative POSIX path ending in {expected}")
        return value


class VersionsReport(ContractModel):
    schema_version: Literal["1.0"]
    releases: list[DatasetRelease]

    @model_validator(mode="after")
    def unique_dataset_versions(self) -> Self:
        versions = [release.dataset_version for release in self.releases]
        if len(versions) != len(set(versions)):
            raise ValueError("dataset_version must be unique within the release catalog")
        return self


# --- Manifest P3 70/20/10 (D02-04) -------------------------------------------------
#
# Espejo exacto de `manifestSummarySchema` en `backend/src/logic/p3.contracts.ts`
# (D01-05, protocolo #33): mismos nombres de campo, mismos literales y las mismas
# reglas, sin `schema_version` (esa API no la versiona así). `test_manifest_contract
# _matches_typescript_fixtures.py` valida este modelo contra los fixtures reales de
# `contracts/p3/fixtures/manifest_summary/`, compartidos con los tests de TS.

DatasetVersion = Annotated[str, StringConstraints(pattern=r"^v\d+\.\d+\.\d+$")]
Sha256Hex = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
ManifestClassName = Literal["cat", "dog"]
MANIFEST_CLASSES = ("cat", "dog")
MANIFEST_SEED = 42
MANIFEST_TARGET_RATIOS = {"train": 0.7, "val": 0.2, "test": 0.1}
MANIFEST_SPLIT_TOLERANCE = 0.05


class ManifestSplitCounts(ContractModel):
    crops: Count
    originals: Count
    crops_per_class: dict[ManifestClassName, Count]

    @model_validator(mode="after")
    def crops_per_class_sums_to_crops(self) -> Self:
        if sum(self.crops_per_class.values()) != self.crops:
            raise ValueError("crops_per_class debe sumar crops")
        return self


class ManifestTargetRatios(ContractModel):
    train: Literal[0.7]
    val: Literal[0.2]
    test: Literal[0.1]


class ManifestSplits(ContractModel):
    train: ManifestSplitCounts
    val: ManifestSplitCounts
    test: ManifestSplitCounts


class ManifestSummary(ContractModel):
    """`GET /api/manifest` (D01-05): la implementación es de Hannah, el contenido es
    de Ale. `frozen=False` es lo único que este ticket (D02-04) puede producir; la
    congelación oficial (`frozen=True`) es de D03-01."""

    manifest_version: Annotated[str, Field(min_length=1)]
    manifest_hash: Sha256Hex
    dataset_version: DatasetVersion
    dvc_release_hash: Sha256Hex
    seed: Literal[42]
    target_ratios: ManifestTargetRatios
    frozen: bool
    classes: list[ManifestClassName]
    splits: ManifestSplits

    @model_validator(mode="after")
    def classes_are_exactly_the_frozen_set(self) -> Self:
        if set(self.classes) != set(MANIFEST_CLASSES) or len(self.classes) != len(MANIFEST_CLASSES):
            raise ValueError(f"classes debe ser exactamente {MANIFEST_CLASSES}")
        return self

    @model_validator(mode="after")
    def each_split_within_tolerance_of_its_target(self) -> Self:
        splits = {"train": self.splits.train, "val": self.splits.val, "test": self.splits.test}
        total = sum(split.crops for split in splits.values())
        targets = {
            "train": self.target_ratios.train,
            "val": self.target_ratios.val,
            "test": self.target_ratios.test,
        }
        for name, split in splits.items():
            if (
                total == 0
                or abs(split.crops / total - targets[name]) > MANIFEST_SPLIT_TOLERANCE + 1e-12
            ):
                raise ValueError(
                    f"{name} fuera de ±5 pp de {targets[name] * 100:.0f}% (medido en crops)"
                )
        return self

    @model_validator(mode="after")
    def val_and_test_contain_every_class(self) -> Self:
        for name, split in (("val", self.splits.val), ("test", self.splits.test)):
            for class_name in MANIFEST_CLASSES:
                if split.crops_per_class.get(class_name, 0) == 0:
                    raise ValueError(f"La clase {class_name} debe estar presente en {name}")
        return self
