from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

import stock_ai.ml.advanced as advanced_module
from stock_ai.cli import app
from stock_ai.ml.advanced import AdvancedResearchConfig, bounded_optuna_search
from stock_ai.ml.checkpoint import ResearchCheckpointStore, read_checkpoint_status


def _frame_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype="uint64").tobytes()
    ).hexdigest()


def _provenance() -> dict[str, object]:
    return {
        "data_snapshot_id": "d" * 64,
        "feature_snapshot_id": "f" * 64,
        "feature_manifest_hash": "m" * 64,
        "feature_names": ["feature"],
        "code_commit": "commit",
        "config_hash": "c" * 64,
        "locked_holdout_accessed": False,
    }


def test_fold_checkpoint_is_atomic_authenticated_and_read_only_status(tmp_path: Path) -> None:
    evidence = {"horizon": 5, "model": "lightgbm", "seed": 17, "fold": 2}
    frame = pd.DataFrame(
        {
            "symbol": ["1000", "2000"],
            "trading_date": pd.to_datetime(["2025-01-01", "2025-01-01"]),
            "prediction": [0.1, -0.2],
        }
    )
    store = ResearchCheckpointStore(tmp_path, provenance=_provenance())
    checkpoint_path = store.path
    store.begin_fold(evidence)
    artifact = store.publish_fold(evidence, frame, frame_hash=_frame_hash)
    assert artifact.is_dir()
    with pytest.raises(RuntimeError, match="another worker"):
        ResearchCheckpointStore(tmp_path, provenance=_provenance())
    store.close()

    progress_path = checkpoint_path / "progress.json"
    before = progress_path.read_bytes()
    status = read_checkpoint_status(checkpoint_path)
    assert status["unit_counts"] == {"SUCCEEDED": 1}
    assert progress_path.read_bytes() == before
    result = CliRunner().invoke(
        app,
        ["research", "checkpoint-status", "--checkpoint-path", str(checkpoint_path)],
    )
    assert result.exit_code == 0
    assert '"SUCCEEDED": 1' in result.stdout
    assert progress_path.read_bytes() == before

    with ResearchCheckpointStore(tmp_path, provenance=_provenance()) as resumed:
        observed = resumed.load_fold(evidence, frame_hash=_frame_hash)
    assert observed is not None
    pd.testing.assert_frame_equal(observed, frame)


def test_running_fold_becomes_interrupted_and_tamper_fails_closed(tmp_path: Path) -> None:
    interrupted = {"horizon": 20, "model": "xgboost", "seed": 29, "fold": 1}
    complete = {"horizon": 20, "model": "xgboost", "seed": 29, "fold": 2}
    frame = pd.DataFrame({"symbol": ["1000"], "prediction": [0.1]})
    first = ResearchCheckpointStore(tmp_path, provenance=_provenance())
    first.begin_fold(interrupted)
    first.begin_fold(complete)
    artifact = first.publish_fold(complete, frame, frame_hash=_frame_hash)
    first.close()

    with ResearchCheckpointStore(tmp_path, provenance=_provenance()) as resumed:
        status = read_checkpoint_status(resumed.path)
        assert status["unit_counts"] == {"INTERRUPTED": 1, "SUCCEEDED": 1}
        tampered = frame.copy()
        tampered["prediction"] = 999.0
        tampered.to_parquet(artifact / "oof.parquet", index=False)
        with pytest.raises(RuntimeError, match="Parquet hash mismatch"):
            resumed.load_fold(complete, frame_hash=_frame_hash)


def test_generate_oof_predictions_reuses_every_completed_fold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dates = pd.date_range("2025-01-01", periods=36, freq="D")
    rows = []
    for day_number, trading_date in enumerate(dates):
        for symbol_number, symbol in enumerate(("1000", "2000", "3000")):
            rows.append(
                {
                    "symbol": symbol,
                    "trading_date": trading_date,
                    "label_end": pd.Timestamp(trading_date) + timedelta(days=1),
                    "target": float(day_number + symbol_number) / 100.0,
                    "feature": float(day_number - symbol_number),
                }
            )
    frame = pd.DataFrame(rows)
    config = AdvancedResearchConfig(
        horizons=(1,),
        model_families=("lightgbm",),
        seeds=(17,),
        initial_train_periods=20,
        validation_periods=5,
        step_periods=5,
        holdout_periods=2,
        estimator_count=5,
        tuning_trials=1,
        tuning_timeout_seconds=10,
        run_ablations=False,
        run_diagnostics=False,
    )
    fit_calls = 0

    def fake_fit_predict(**kwargs: object) -> np.ndarray:
        nonlocal fit_calls
        fit_calls += 1
        validation_x = kwargs["validation_x"]
        assert isinstance(validation_x, pd.DataFrame)
        return validation_x.iloc[:, 0].to_numpy(dtype=float)

    monkeypatch.setattr(advanced_module, "_fit_predict", fake_fit_predict)
    with ResearchCheckpointStore(tmp_path, provenance=_provenance()) as store:
        first = advanced_module.generate_oof_predictions(
            frame,
            feature_names=("feature",),
            target_column="target",
            label_end_column="label_end",
            horizon=1,
            family="lightgbm",
            task="regression",
            seed=17,
            parameters={},
            config=config,
            checkpoint_store=store,
        )
    assert fit_calls > 0

    def should_not_fit(**_kwargs: object) -> np.ndarray:
        raise AssertionError("an authenticated completed fold was recomputed")

    monkeypatch.setattr(advanced_module, "_fit_predict", should_not_fit)
    with ResearchCheckpointStore(tmp_path, provenance=_provenance()) as resumed:
        second = advanced_module.generate_oof_predictions(
            frame,
            feature_names=("feature",),
            target_column="target",
            label_end_column="label_end",
            horizon=1,
            family="lightgbm",
            task="regression",
            seed=17,
            parameters={},
            config=config,
            checkpoint_store=resumed,
        )
    pd.testing.assert_frame_equal(second, first)


def test_optuna_trials_persist_and_are_not_repeated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["1000", "2000"],
            "trading_date": pd.to_datetime(["2025-01-01", "2025-01-01"]),
            "label_end": pd.to_datetime(["2025-01-02", "2025-01-02"]),
            "target": [0.1, -0.1],
            "feature": [1.0, 2.0],
        }
    )
    config = AdvancedResearchConfig(
        horizons=(1,),
        model_families=("lightgbm",),
        seeds=(17,),
        initial_train_periods=20,
        validation_periods=5,
        step_periods=5,
        holdout_periods=2,
        estimator_count=5,
        tuning_trials=2,
        tuning_timeout_seconds=30,
        run_ablations=False,
        run_diagnostics=False,
    )
    objective_calls = 0

    def fake_oof(*_args: object, **_kwargs: object) -> pd.DataFrame:
        nonlocal objective_calls
        objective_calls += 1
        return pd.DataFrame(
            {
                "trading_date": pd.to_datetime(
                    ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02"]
                ),
                "target": [0.0, 1.0, 0.0, 1.0],
                "prediction": [0.1, 0.9, 0.2, 0.8],
            }
        )

    monkeypatch.setattr(advanced_module, "generate_oof_predictions", fake_oof)
    with ResearchCheckpointStore(tmp_path, provenance=_provenance()) as store:
        first = bounded_optuna_search(
            frame,
            feature_names=("feature",),
            target_column="target",
            label_end_column="label_end",
            horizon=1,
            family="lightgbm",
            config=config,
            checkpoint_store=store,
        )
    assert objective_calls == 2
    assert len(first.trials) == 2

    objective_calls = 0
    with ResearchCheckpointStore(tmp_path, provenance=_provenance()) as resumed:
        second = bounded_optuna_search(
            frame,
            feature_names=("feature",),
            target_column="target",
            label_end_column="label_end",
            horizon=1,
            family="lightgbm",
            config=config,
            checkpoint_store=resumed,
        )
    assert objective_calls == 0
    assert second.best_parameters == first.best_parameters
    assert second.trials == first.trials
    assert second.trials_completed == 2
    assert (resumed.path / "optuna.sqlite3").is_file()


def test_checkpoint_identity_changes_for_any_research_hash(tmp_path: Path) -> None:
    original = _provenance()
    changed = {**original, "data_snapshot_id": "x" * 64}
    with ResearchCheckpointStore(tmp_path, provenance=original) as first:
        first_id = first.checkpoint_id
    with ResearchCheckpointStore(tmp_path, provenance=changed) as second:
        second_id = second.checkpoint_id
    assert first_id != second_id
    assert datetime.fromisoformat(
        str(read_checkpoint_status(tmp_path / first_id)["updated_at"])
    ).tzinfo is not None
    assert datetime.now(UTC).tzinfo is not None
