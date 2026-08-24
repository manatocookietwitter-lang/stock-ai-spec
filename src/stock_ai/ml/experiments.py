"""Append-only experiment records, including rejected and negative results."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from types import MappingProxyType

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)


class ExperimentRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = Field(min_length=1)
    created_at: datetime
    hypothesis: str = Field(min_length=1)
    data_snapshot_id: str = Field(min_length=1)
    feature_set_version: str = Field(min_length=1)
    preprocessing_version: str = Field(min_length=1)
    feature_definition_hashes: Mapping[str, str]
    code_commit: str = Field(min_length=1)
    config_hash: str = Field(min_length=1)
    model_type: str = Field(min_length=1)
    parameters: Mapping[str, int | float | str]
    seed: int | None
    fold_results: tuple[Mapping[str, int | float | str], ...]
    trial_results: tuple[Mapping[str, int | float | str], ...] = ()
    aggregate_results: Mapping[str, int | float | str]
    decision: str = Field(pattern="^(adopted|rejected|research_only)$")
    rejection_reason: str | None = None
    locked_holdout_accessed: bool = False

    @field_validator("created_at")
    @classmethod
    def aware_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("experiment timestamps must be timezone-aware")
        return value

    @field_validator("feature_definition_hashes", mode="after")
    @classmethod
    def freeze_hashes(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        return MappingProxyType(dict(value))

    @field_validator("parameters", "aggregate_results", mode="after")
    @classmethod
    def freeze_values(
        cls, value: Mapping[str, int | float | str]
    ) -> Mapping[str, int | float | str]:
        return MappingProxyType(dict(value))

    @field_validator("fold_results", "trial_results", mode="after")
    @classmethod
    def freeze_audit_rows(
        cls, value: tuple[Mapping[str, int | float | str], ...]
    ) -> tuple[Mapping[str, int | float | str], ...]:
        return tuple(MappingProxyType(dict(row)) for row in value)

    @field_serializer("feature_definition_hashes", "parameters", "aggregate_results")
    def serialize_mapping(self, value: Mapping[str, object]) -> dict[str, object]:
        return dict(value)

    @field_serializer("fold_results", "trial_results")
    def serialize_audit_rows(
        self, value: tuple[Mapping[str, int | float | str], ...]
    ) -> tuple[dict[str, int | float | str], ...]:
        return tuple(dict(row) for row in value)

    @model_validator(mode="after")
    def valid_audit_record(self) -> ExperimentRecord:
        if self.decision == "rejected" and not self.rejection_reason:
            raise ValueError("rejected experiments require a rejection reason")
        values: list[object] = [
            *self.parameters.values(),
            *(value for fold in self.fold_results for value in fold.values()),
            *(value for trial in self.trial_results for value in trial.values()),
            *self.aggregate_results.values(),
        ]
        if any(isinstance(value, float) and not math.isfinite(value) for value in values):
            raise ValueError("experiment metrics and parameters must be finite")
        return self


class ExperimentRegistry:
    """A minimal append-only JSONL registry for deterministic local research."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, record: ExperimentRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            existing_ids = {
                str(json.loads(line)["experiment_id"])
                for line in self.path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            if record.experiment_id in existing_ids:
                raise ValueError(f"experiment ID already exists: {record.experiment_id}")
        payload = json.dumps(record.model_dump(mode="json"), sort_keys=True, allow_nan=False) + "\n"
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
