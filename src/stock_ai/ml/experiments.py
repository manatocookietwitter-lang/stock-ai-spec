"""Append-only experiment records, including rejected and negative results."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExperimentRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = Field(min_length=1)
    created_at: datetime
    hypothesis: str = Field(min_length=1)
    data_snapshot_id: str = Field(min_length=1)
    feature_set_version: str = Field(min_length=1)
    model_type: str = Field(min_length=1)
    parameters: dict[str, int | float | str]
    seed: int | None
    fold_results: tuple[dict[str, int | float], ...]
    aggregate_results: dict[str, int | float | str]
    decision: str = Field(pattern="^(adopted|rejected|research_only)$")
    rejection_reason: str | None = None

    @field_validator("created_at")
    @classmethod
    def aware_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("experiment timestamps must be timezone-aware")
        return value


class ExperimentRegistry:
    """A minimal append-only JSONL registry for deterministic local research."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, record: ExperimentRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record.model_dump(mode="json"), sort_keys=True) + "\n")
