"""Machine-readable feature definitions and deterministic feature-set manifests."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _source_revision() -> str:
    configured = os.environ.get("STOCK_AI_CODE_COMMIT")
    if configured:
        return configured
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        revision = completed.stdout.strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet"],
            cwd=Path(__file__).resolve().parent,
            check=False,
            capture_output=True,
            timeout=2,
        ).returncode
        return f"{revision}-dirty" if dirty else revision
    except (OSError, subprocess.SubprocessError):
        return "UNSET"


class FeatureDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    family: str = Field(min_length=1)
    version: int = Field(ge=1)
    stage: str = Field(pattern=r"^v[0-3](_core|_extended)?$")
    inputs: tuple[str, ...]
    parameters: Mapping[str, int | float | str]
    formula: str = Field(min_length=1)
    implementation: str = "stock_ai.features.engine@0.1.0"
    warmup_period: int = Field(ge=0)
    output_unit: str = Field(min_length=1)
    normalization: tuple[str, ...] = ("raw",)
    availability_rule: str = "completed_daily_bar_only"
    required_capabilities: tuple[str, ...] = ("daily_adjusted_ohlcv",)

    @field_validator("parameters", mode="after")
    @classmethod
    def freeze_parameters(
        cls, value: Mapping[str, int | float | str]
    ) -> Mapping[str, int | float | str]:
        return MappingProxyType(dict(value))

    @field_serializer("parameters")
    def serialize_parameters(
        self, value: Mapping[str, int | float | str]
    ) -> dict[str, int | float | str]:
        return dict(value)

    @property
    def definition_hash(self) -> str:
        return _stable_hash(self.model_dump(mode="json"))


class FeatureSetManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    feature_set_id: str = Field(min_length=1)
    feature_set_version: str = Field(min_length=1)
    feature_names: tuple[str, ...]
    feature_definition_hashes: Mapping[str, str]
    preprocessing_version: str = "raw-v1"
    created_at: datetime = datetime.fromisoformat("2026-08-24T00:00:00+00:00")
    code_commit: str = Field(default_factory=_source_revision)
    required_capabilities: tuple[str, ...]
    training_only_fit_rules: tuple[str, ...] = (
        "imputation_fit_on_training_fold_only",
        "scaling_fit_on_training_fold_only",
        "feature_selection_fit_on_training_fold_only",
    )

    @field_validator("feature_definition_hashes", mode="after")
    @classmethod
    def freeze_definition_hashes(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        return MappingProxyType(dict(value))

    @field_serializer("feature_definition_hashes")
    def serialize_definition_hashes(self, value: Mapping[str, str]) -> dict[str, str]:
        return dict(value)

    @model_validator(mode="after")
    def hashes_match_features(self) -> Self:
        if len(self.feature_names) != len(set(self.feature_names)):
            raise ValueError("manifest feature names must be unique")
        if set(self.feature_names) != set(self.feature_definition_hashes):
            raise ValueError("every manifest feature must have exactly one definition hash")
        return self

    @property
    def manifest_hash(self) -> str:
        return _stable_hash(self.model_dump(mode="json"))


class FeatureRegistry:
    def __init__(self, definitions: Iterable[FeatureDefinition]) -> None:
        entries = tuple(definitions)
        by_name = {definition.name: definition for definition in entries}
        if len(by_name) != len(entries):
            raise ValueError("feature names must be unique")
        self._definitions = by_name

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._definitions)

    def definition(self, name: str) -> FeatureDefinition:
        try:
            return self._definitions[name]
        except KeyError as exc:
            raise KeyError(f"unknown feature: {name}") from exc

    def manifest(
        self, feature_set_id: str, version: str, names: Iterable[str]
    ) -> FeatureSetManifest:
        selected_names = tuple(names)
        definitions = [self.definition(name) for name in selected_names]
        capabilities = tuple(
            sorted(
                {capability for item in definitions for capability in item.required_capabilities}
            )
        )
        return FeatureSetManifest(
            feature_set_id=feature_set_id,
            feature_set_version=version,
            feature_names=selected_names,
            feature_definition_hashes={item.name: item.definition_hash for item in definitions},
            required_capabilities=capabilities,
            code_commit=_source_revision(),
        )
