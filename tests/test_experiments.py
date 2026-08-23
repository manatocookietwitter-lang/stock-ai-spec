from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from stock_ai.ml.experiments import ExperimentRecord, ExperimentRegistry


def test_rejected_experiment_is_preserved_append_only(tmp_path: Path) -> None:
    path = tmp_path / "experiments.jsonl"
    registry = ExperimentRegistry(path)
    record = ExperimentRecord(
        experiment_id="E2-fixture-ridge",
        created_at=datetime(2026, 8, 24, 12, 0, tzinfo=ZoneInfo("Asia/Tokyo")),
        hypothesis="V1 Ridge improves rank IC over V0 on deterministic fixture mechanics",
        data_snapshot_id="fixture-snapshot",
        feature_set_version="1.0.0",
        model_type="ridge",
        parameters={"alpha": 5.0},
        seed=None,
        fold_results=({"fold": 0, "rank_ic": -0.10},),
        aggregate_results={"mean_rank_ic": -0.10},
        decision="rejected",
        rejection_reason=(
            "negative fixture result is retained; fixtures do not measure profitability"
        ),
    )
    registry.append(record)
    registry.append(record.model_copy(update={"experiment_id": "E2-fixture-ridge-repeat"}))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["decision"] == "rejected"
    assert json.loads(lines[1])["experiment_id"] == "E2-fixture-ridge-repeat"
