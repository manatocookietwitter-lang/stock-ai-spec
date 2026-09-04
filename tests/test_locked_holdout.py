from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

import stock_ai.cli as cli_module
import stock_ai.ml.holdout as holdout_module
import stock_ai.ml.selection as selection_module
from stock_ai.cli import _holdout_component_audit_row, app
from stock_ai.data.contracts import CapabilityStatus
from stock_ai.features import V0_MANIFEST
from stock_ai.ml.advanced import (
    AdvancedModelMetrics,
    AdvancedResearchConfig,
    EnsembleResult,
    UncertaintyCalibration,
)
from stock_ai.ml.holdout import (
    HoldoutComponentCheckpoint,
    HoldoutComponentResult,
    HoldoutComponentStatus,
    HoldoutEnsembleResult,
    HoldoutEvaluationStatus,
    LockedHoldoutEvaluation,
    LockedHoldoutLedger,
    LockedHoldoutReport,
    evaluate_locked_holdout,
    read_locked_holdout_status,
)
from stock_ai.ml.production import write_production_dataset_snapshot
from stock_ai.ml.selection import (
    DevelopmentFeatureSelectionArtifact,
    DevelopmentSelectionArtifact,
    FeatureFamilySelection,
    FrozenModelComponent,
    HorizonDevelopmentSelection,
    HorizonFeatureSelection,
    write_development_selection,
)


def test_locked_holdout_resumes_components_and_never_refits_successes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_path, dataset_id, build_id = _production_fixture(tmp_path)
    selection = _selection_fixture(build_id=build_id, dataset_id=dataset_id)
    selection_path = write_development_selection(selection, tmp_path / "selections")
    calls: list[str] = []
    fail_once = True

    def fake_fit(
        training: pd.DataFrame,
        prediction_frame: pd.DataFrame,
        **kwargs: Any,
    ) -> np.ndarray:
        nonlocal fail_once
        del training
        task = str(kwargs["task"])
        horizon = int(kwargs["horizon"])
        key = f"h{horizon}:{task}"
        calls.append(key)
        if fail_once and len(calls) == 3:
            fail_once = False
            raise RuntimeError("simulated interruption")
        target = prediction_frame[f"target_return_{horizon}d"].to_numpy(dtype=float)
        if task == "large_loss":
            return np.where(target <= -0.08, 0.8, 0.05)
        if task == "quantile":
            return target - 0.01
        if task == "ranking":
            return target * 2.0
        return target + 0.001

    monkeypatch.setattr(holdout_module, "fit_predict_frozen_model", fake_fit)
    evaluation_root = tmp_path / "holdout"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        evaluate_locked_holdout(
            selection_path=selection_path,
            build_manifest_path=build_path,
            evaluation_root=evaluation_root,
            evaluator_code_commit="holdout-test",
        )
    evaluation_directory = evaluation_root / selection.selection_id
    before = (evaluation_directory / "ledger.json").read_bytes()
    status = read_locked_holdout_status(evaluation_directory)
    after = (evaluation_directory / "ledger.json").read_bytes()
    assert before == after
    assert status["holdout_accessed"] is True
    first_successes = sum(
        item["status"] == "SUCCEEDED" for item in status["components"]  # type: ignore[index]
    )
    assert first_successes == 2

    calls_before_resume = len(calls)
    completed = evaluate_locked_holdout(
        selection_path=selection_path,
        build_manifest_path=build_path,
        evaluation_root=evaluation_root,
        evaluator_code_commit="holdout-test",
    )
    assert completed.resumed
    assert completed.report.locked_holdout_accessed
    assert completed.report.selection_was_frozen_before_access
    assert not completed.report.model_choices_changed_after_access
    assert tuple(item.horizon for item in completed.report.ensemble_results) == (1, 5, 20)
    assert len(calls) - calls_before_resume == len(status["components"]) - 2  # type: ignore[arg-type]
    for path in (evaluation_directory / "predictions").glob("*.parquet"):
        assert "target" not in pd.read_parquet(path).columns

    calls_before_idempotent = len(calls)
    same = evaluate_locked_holdout(
        selection_path=selection_path,
        build_manifest_path=build_path,
        evaluation_root=evaluation_root,
        evaluator_code_commit="holdout-test",
    )
    assert same.report.report_id == completed.report.report_id
    assert len(calls) == calls_before_idempotent


def test_completed_holdout_rejects_prediction_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_path, dataset_id, build_id = _production_fixture(tmp_path)
    selection = _selection_fixture(build_id=build_id, dataset_id=dataset_id)
    selection_path = write_development_selection(selection, tmp_path / "selections")

    def fake_fit(
        training: pd.DataFrame,
        prediction_frame: pd.DataFrame,
        **kwargs: Any,
    ) -> np.ndarray:
        del training
        horizon = int(kwargs["horizon"])
        target = prediction_frame[f"target_return_{horizon}d"].to_numpy(dtype=float)
        if kwargs["task"] == "large_loss":
            return np.full(len(target), 0.05)
        return target

    monkeypatch.setattr(holdout_module, "fit_predict_frozen_model", fake_fit)
    evaluation_root = tmp_path / "holdout"
    evaluate_locked_holdout(
        selection_path=selection_path,
        build_manifest_path=build_path,
        evaluation_root=evaluation_root,
        evaluator_code_commit="holdout-test",
    )
    prediction_path = next(
        (evaluation_root / selection.selection_id / "predictions").glob("*.parquet")
    )
    frame = pd.read_parquet(prediction_path)
    frame.loc[0, "prediction"] = 999.0
    frame.to_parquet(prediction_path, index=False)

    with pytest.raises(RuntimeError, match="Parquet hash mismatch"):
        evaluate_locked_holdout(
            selection_path=selection_path,
            build_manifest_path=build_path,
            evaluation_root=evaluation_root,
            evaluator_code_commit="holdout-test",
        )


def test_holdout_rejects_changed_evaluator_identity_after_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_path, dataset_id, build_id = _production_fixture(tmp_path)
    selection = _selection_fixture(build_id=build_id, dataset_id=dataset_id)
    selection_path = write_development_selection(selection, tmp_path / "selections")

    def always_fail(*_: object, **__: object) -> np.ndarray:
        raise RuntimeError("stop after ledger publication")

    monkeypatch.setattr(holdout_module, "fit_predict_frozen_model", always_fail)
    with pytest.raises(RuntimeError, match="stop after ledger"):
        evaluate_locked_holdout(
            selection_path=selection_path,
            build_manifest_path=build_path,
            evaluation_root=tmp_path / "holdout",
            evaluator_code_commit="first-code",
        )
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        evaluate_locked_holdout(
            selection_path=selection_path,
            build_manifest_path=build_path,
            evaluation_root=tmp_path / "holdout",
            evaluator_code_commit="changed-code",
        )
    with pytest.raises(RuntimeError, match="authorization provenance mismatch"):
        evaluate_locked_holdout(
            selection_path=selection_path,
            build_manifest_path=build_path,
            evaluation_root=tmp_path / "alternate-holdout-root",
            evaluator_code_commit="first-code",
        )


def test_selection_schema_rejects_incoherent_frozen_choices() -> None:
    selection = _selection_fixture(build_id="a" * 64, dataset_id="b" * 64)
    horizon = selection.horizons[0]
    component = horizon.expected_return_component

    for field, value, message in (
        ("source_config_hash", "f" * 64, "source config hash"),
        (
            "source_config",
            AdvancedResearchConfig.model_validate(
                {**component.source_config.model_dump(), "horizons": (5,)}
            ),
            "source config horizon",
        ),
        (
            "source_config",
            AdvancedResearchConfig.model_validate(
                {**component.source_config.model_dump(), "model_families": ("xgboost",)}
            ),
            "source config family",
        ),
        (
            "source_config",
            AdvancedResearchConfig.model_validate(
                {**component.source_config.model_dump(), "seeds": (29,)}
            ),
            "source config seed",
        ),
    ):
        payload = component.model_dump(mode="python")
        payload[field] = value
        if field == "source_config":
            payload["source_config_hash"] = value.config_hash
        with pytest.raises(ValueError, match=message):
            FrozenModelComponent.model_validate(payload)

    family = horizon.feature_families[0]
    for updates, message in (
        ({"seeds": (17, 29)}, "at least three"),
        ({"seed_votes_selected": (99,)}, "undeclared seed"),
        ({"selected": True}, "frozen rule"),
        ({"selected_features": ("unexpected",)}, "rejected feature"),
        (
            {"selected": True, "seed_votes_selected": (17, 29)},
            "must retain its exact features",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            FeatureFamilySelection.model_validate(
                {**family.model_dump(mode="python"), **updates}
            )

    ranking_component = horizon.ensemble_components[0]
    horizon_five = selection.horizons[1]
    wrong_ensemble = EnsembleResult.model_validate(
        {**horizon.ensemble.model_dump(), "horizon": 5}
    )
    wrong_uncertainty = UncertaintyCalibration.model_validate(
        {**horizon.uncertainty.model_dump(), "horizon": 5}
    )
    duplicate_features = (horizon.feature_names[0], horizon.feature_names[0])
    cases: tuple[tuple[dict[str, object], str], ...] = (
        ({"horizon": 2}, "must be 1, 5, or 20"),
        ({"feature_names_hash": "f" * 64}, "feature identity"),
        (
            {
                "feature_names": duplicate_features,
                "feature_names_hash": selection_module._stable_hash(duplicate_features),
            },
            "features must be unique",
        ),
        ({"feature_families": horizon.feature_families[:-1]}, "ordered F1..F12"),
        (
            {
                "feature_families": (
                    horizon_five.feature_families[0],
                    *horizon.feature_families[1:],
                )
            },
            "evidence horizon mismatch",
        ),
        (
            {"expected_return_component": horizon_five.expected_return_component},
            "component horizon mismatch",
        ),
        ({"expected_return_component": ranking_component}, "must be a regression"),
        ({"rank_component": horizon.downside_quantile_component}, "stackable model"),
        ({"downside_quantile_component": component}, "must be a quantile"),
        ({"large_loss_component": component}, "must be a classifier"),
        ({"ensemble": wrong_ensemble}, "ensemble or uncertainty horizon"),
        ({"uncertainty": wrong_uncertainty}, "ensemble or uncertainty horizon"),
        (
            {"ensemble_components": tuple(reversed(horizon.ensemble_components))},
            "ensemble component identity",
        ),
        ({"ensemble_adopted": False}, "ensemble adoption"),
    )
    for updates, message in cases:
        with pytest.raises(ValueError, match=message):
            HorizonDevelopmentSelection.model_validate(
                {**horizon.model_dump(mode="python"), **updates}
            )


def test_selection_artifact_schema_requires_complete_unique_provenance() -> None:
    selection = _selection_fixture(build_id="a" * 64, dataset_id="b" * 64)
    cases: tuple[tuple[dict[str, object], str], ...] = (
        ({"created_at": datetime(2026, 1, 2)}, "timezone-aware"),
        ({"horizons": selection.horizons[:-1]}, "ordered 1d/5d/20d"),
        ({"seeds": (17, 29)}, "at least three"),
        (
            {"source_report_ids": (selection.source_report_ids[0],) * 2},
            "report identities",
        ),
        ({"candidate_campaign_ids": ()}, "candidate campaign identities"),
        (
            {"ablation_campaign_ids": ("5" * 64, "5" * 64)},
            "ablation campaign identities",
        ),
        ({"source_code_commits": ()}, "source code commit identities"),
        ({"adoption_blocking_reasons": ()}, "cannot claim live adoption"),
        ({"selection_id": "f" * 64}, "content identity"),
    )
    for updates, message in cases:
        with pytest.raises(ValueError, match=message):
            DevelopmentSelectionArtifact.model_validate(
                {**selection.model_dump(mode="python"), **updates}
            )


def test_feature_selection_schema_requires_exact_votes_and_shared_seeds() -> None:
    selection = _selection_fixture(build_id="a" * 64, dataset_id="b" * 64)
    artifact = _feature_selection_fixture(selection)
    horizon = artifact.horizons[0]
    horizon_five = artifact.horizons[1]
    duplicate_features = (horizon.feature_names[0], horizon.feature_names[0])
    cases: tuple[tuple[dict[str, object], str], ...] = (
        ({"horizon": 2}, "must be 1, 5, or 20"),
        ({"feature_names_hash": "f" * 64}, "feature identity"),
        (
            {
                "feature_names": duplicate_features,
                "feature_names_hash": selection_module._stable_hash(duplicate_features),
            },
            "features must be unique",
        ),
        ({"feature_families": horizon.feature_families[:-1]}, "ordered F1..F12"),
        (
            {
                "feature_families": (
                    horizon_five.feature_families[0],
                    *horizon.feature_families[1:],
                )
            },
            "family horizon mismatch",
        ),
        (
            {
                "feature_names": horizon.feature_names[:-1],
                "feature_names_hash": selection_module._stable_hash(
                    horizon.feature_names[:-1]
                ),
            },
            "do not match the frozen family votes",
        ),
    )
    for updates, message in cases:
        with pytest.raises(ValueError, match=message):
            HorizonFeatureSelection.model_validate(
                {**horizon.model_dump(mode="python"), **updates}
            )

    different_seed_family = FeatureFamilySelection.model_validate(
        {
            **horizon.feature_families[0].model_dump(mode="python"),
            "seeds": (17, 29, 47),
        }
    )
    different_seed_horizon = horizon.model_copy(
        update={
            "feature_families": (
                different_seed_family,
                *horizon.feature_families[1:],
            )
        }
    )
    artifact_cases: tuple[tuple[dict[str, object], str], ...] = (
        ({"created_at": datetime(2026, 1, 2)}, "timezone-aware"),
        ({"horizons": artifact.horizons[:-1]}, "ordered 1d/5d/20d"),
        ({"seeds": (17, 29)}, "at least three"),
        (
            {"horizons": (different_seed_horizon, *artifact.horizons[1:])},
            "same frozen seeds",
        ),
        ({"ablation_campaign_ids": ()}, "ablation campaign identities"),
        ({"source_report_ids": ()}, "source report identities"),
        ({"source_code_commits": ()}, "source code commit identities"),
        ({"feature_selection_id": "f" * 64}, "content identity"),
    )
    for updates, message in artifact_cases:
        with pytest.raises(ValueError, match=message):
            DevelopmentFeatureSelectionArtifact.model_validate(
                {**artifact.model_dump(mode="python"), **updates}
            )


def test_holdout_schema_and_registry_rows_preserve_task_specific_metrics() -> None:
    selection = _selection_fixture(build_id="a" * 64, dataset_id="b" * 64)
    component = selection.horizons[0].expected_return_component
    regression_metrics = _metrics_for(component, mean_squared_error=0.04)
    result = HoldoutComponentResult(
        component_key="h1:regression",
        component=component,
        metrics=regression_metrics,
        prediction_sha256="c" * 64,
    )
    row = _holdout_component_audit_row(result)
    assert row["mean_squared_error"] == pytest.approx(0.04)
    assert "pinball_loss" not in row

    quantile = selection.horizons[0].downside_quantile_component
    quantile_row = _holdout_component_audit_row(
        HoldoutComponentResult(
            component_key="h1:quantile",
            component=quantile,
            metrics=_metrics_for(quantile, pinball_loss=0.02),
            prediction_sha256="d" * 64,
        )
    )
    assert quantile_row["pinball_loss"] == pytest.approx(0.02)
    assert "mean_squared_error" not in quantile_row

    with pytest.raises(ValueError, match="horizon mismatch"):
        HoldoutComponentResult(
            component_key="bad",
            component=component,
            metrics=_metrics_for(selection.horizons[1].expected_return_component),
            prediction_sha256="e" * 64,
        )
    with pytest.raises(ValueError, match="model identity mismatch"):
        HoldoutComponentResult(
            component_key="bad",
            component=component,
            metrics=_metrics_for(component, task="ranking"),
            prediction_sha256="e" * 64,
        )
    valid_ensemble = _holdout_ensemble(1)
    with pytest.raises(ValueError, match="names and weights"):
        HoldoutEnsembleResult.model_validate(
            {**valid_ensemble.model_dump(), "weights": ()}
        )
    with pytest.raises(ValueError, match="non-negative simplex"):
        HoldoutEnsembleResult.model_validate(
            {**valid_ensemble.model_dump(), "weights": (1.2,)}
        )


def _metrics_for(
    component: FrozenModelComponent,
    **updates: object,
) -> AdvancedModelMetrics:
    return AdvancedModelMetrics.model_validate(
        {
            "horizon": component.horizon,
            "model_family": component.model_family,
            "task": component.task,
            "seed": component.seed,
            "folds": 1,
            "rows": 3,
            "dates": 1,
            **updates,
        }
    )


def _holdout_ensemble(horizon: int) -> HoldoutEnsembleResult:
    return HoldoutEnsembleResult(
        horizon=horizon,
        adopted_on_development=True,
        component_names=("component",),
        weights=(1.0,),
        rows=3,
        dates=1,
        mean_squared_error_rank_space=0.1,
        mean_daily_rank_ic=0.2,
        rank_icir=0.3,
        empirical_coverage_80=0.8,
        empirical_coverage_90=0.9,
        disagreement_error_correlation=None,
    )


def test_holdout_ledger_and_report_fail_closed_on_incoherent_state(tmp_path: Path) -> None:
    selection = _selection_fixture(build_id="a" * 64, dataset_id="b" * 64)
    component = selection.horizons[0].expected_return_component
    checkpoint = HoldoutComponentCheckpoint(
        component_id="c" * 64,
        component_key="h1:regression",
        horizon=1,
        component_name=component.component_name,
        status=HoldoutComponentStatus.SUCCEEDED,
        attempts=1,
        prediction_path=str((tmp_path / "prediction.parquet").resolve()),
        parquet_sha256="d" * 64,
        frame_sha256="e" * 64,
        rows=3,
        started_at=datetime(2026, 1, 3, tzinfo=UTC),
        completed_at=datetime(2026, 1, 3, 0, 1, tzinfo=UTC),
    )
    ledger = _ledger_fixture(selection, checkpoint)
    with pytest.raises(ValueError, match="ledger identity mismatch"):
        LockedHoldoutLedger.model_validate(
            {**ledger.model_dump(mode="python"), "ledger_id": "f" * 64}
        )
    with pytest.raises(ValueError, match="identities must be unique"):
        _ledger_fixture(selection, checkpoint, components=[checkpoint, checkpoint])
    with pytest.raises(ValueError, match="requires its immutable report"):
        _ledger_fixture(selection, checkpoint, status=HoldoutEvaluationStatus.COMPLETED)
    pending = checkpoint.model_copy(update={"status": HoldoutComponentStatus.PENDING})
    with pytest.raises(ValueError, match="unfinished components"):
        _ledger_fixture(
            selection,
            checkpoint,
            status=HoldoutEvaluationStatus.COMPLETED,
            report_path=str(tmp_path / "report.json"),
            completed_at=datetime(2026, 1, 3, 0, 2, tzinfo=UTC),
            components=[pending],
        )

    report = _report_fixture(selection, ledger, component)
    with pytest.raises(ValueError, match="timezone-aware"):
        _report_fixture(
            selection,
            ledger,
            component,
            created_at=datetime(2026, 1, 3),
        )
    with pytest.raises(ValueError, match="ordered 1d/5d/20d"):
        _report_fixture(
            selection,
            ledger,
            component,
            ensemble_results=(_holdout_ensemble(1), _holdout_ensemble(5)),
        )
    with pytest.raises(ValueError, match="cannot claim live adoption"):
        _report_fixture(selection, ledger, component, component_results=())
    with pytest.raises(ValueError, match="feature definition hashes"):
        _report_fixture(selection, ledger, component, feature_definition_hashes={})
    with pytest.raises(ValueError, match="content identity mismatch"):
        LockedHoldoutReport.model_validate(
            {**report.model_dump(mode="python"), "report_id": "f" * 64}
        )

    holdout_module._require_completed_report(
        report,
        ledger=ledger,
        selection=selection,
        build_id=selection.build_id,
    )
    for field, value in (
        ("data_snapshot_id", "f" * 64),
        ("evaluator_code_commit", "changed"),
        ("locked_holdout_start", "1999-01-01"),
        ("feature_definition_hashes", {"bad": "hash"}),
        ("component_results", ()),
    ):
        with pytest.raises(RuntimeError, match="report provenance mismatch"):
            holdout_module._require_completed_report(
                report.model_copy(update={field: value}),
                ledger=ledger,
                selection=selection,
                build_id=selection.build_id,
            )

    bad_horizon = selection.horizons[0].model_copy(
        update={"feature_names": ("outside-v2",)}
    )
    bad_selection = selection.model_copy(
        update={"horizons": (bad_horizon, *selection.horizons[1:])}
    )
    with pytest.raises(RuntimeError, match="outside the frozen V2 manifest"):
        holdout_module._expected_feature_definition_hashes(bad_selection)


def test_selection_helpers_reject_missing_metrics_and_invalid_metadata(tmp_path: Path) -> None:
    selection = _selection_fixture(build_id="a" * 64, dataset_id="b" * 64)
    with pytest.raises(ValueError, match="no development metric"):
        selection_module._best_component([], task="quantile", reports={})
    for component in (
        selection.horizons[0].expected_return_component,
        selection.horizons[0].downside_quantile_component,
        selection.horizons[0].large_loss_component,
    ):
        with pytest.raises(ValueError, match="no finite development selection metric"):
            selection_module._best_component(
                [_metrics_for(component)], task=component.task, reports={}
            )
    with pytest.raises(ValueError, match="exactly one model result"):
        selection_module._component_from_name(
            [],
            component_name="missing",
            reports={},
            metric_name="mean_daily_rank_ic",
            metric_value=0.0,
        )
    with pytest.raises(ValueError, match="at least six dates"):
        selection_module._chronological_meta_masks(
            pd.Series(pd.date_range("2026-01-01", periods=5)),
            pd.Series(pd.date_range("2026-01-02", periods=5)),
        )
    dates = pd.Series(pd.date_range("2026-01-01", periods=6))
    with pytest.raises(ValueError, match="meta stages are empty"):
        selection_module._chronological_meta_masks(
            dates,
            pd.Series([pd.Timestamp("2027-01-01")] * 6),
        )
    assert selection_module._finite_correlation(
        np.array([1.0, np.nan]), np.array([2.0, 3.0])
    ) is None

    path = write_development_selection(selection, tmp_path / "selections")
    assert selection_module.load_development_selection(path) == selection
    with pytest.raises(RuntimeError, match="not content-addressed"):
        selection_module.load_development_selection(tmp_path / "plain.json")
    addressed = tmp_path / ("9" * 64) / f"{'9' * 64}.json"
    with pytest.raises(RuntimeError, match="missing or invalid"):
        selection_module.load_development_selection(addressed)
    addressed.parent.mkdir(parents=True)
    addressed.write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="metadata is invalid"):
        selection_module.load_development_selection(addressed)

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selection_path"] = str(tmp_path / "wrong.json")
    payload["metadata_hash"] = selection_module._stable_hash(
        {key: value for key, value in payload.items() if key != "metadata_hash"}
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="path metadata mismatch"):
        selection_module.load_development_selection(path)


def test_holdout_prediction_and_report_artifacts_are_strictly_authenticated(
    tmp_path: Path,
) -> None:
    selection = _selection_fixture(build_id="a" * 64, dataset_id="b" * 64)
    component = selection.horizons[0].expected_return_component
    checkpoint = HoldoutComponentCheckpoint(
        component_id="c" * 64,
        component_key="h1:regression",
        horizon=1,
        component_name=component.component_name,
    )
    frame = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "trading_date": pd.to_datetime(["2026-01-01", "2026-01-01"]),
            "prediction": [0.1, 0.2],
        }
    )
    holdout_module._publish_prediction(checkpoint, frame, tmp_path / "predictions")
    holdout_module._publish_prediction(checkpoint, frame, tmp_path / "predictions")
    assert holdout_module._load_prediction(checkpoint, expected=frame).equals(frame)
    with pytest.raises(RuntimeError, match="identity already exists with conflicts"):
        holdout_module._publish_prediction(
            checkpoint,
            frame.assign(prediction=[0.3, 0.4]),
            tmp_path / "predictions",
        )
    with pytest.raises(RuntimeError, match="lacks prediction provenance"):
        holdout_module._load_prediction_artifact(
            HoldoutComponentCheckpoint(
                component_id="d" * 64,
                component_key="missing",
                horizon=1,
                component_name="missing",
            )
        )
    with pytest.raises(RuntimeError, match="row count"):
        holdout_module._load_prediction(checkpoint, expected=frame.iloc[:1])
    with pytest.raises(RuntimeError, match="symbol identity"):
        holdout_module._load_prediction(
            checkpoint, expected=frame.assign(symbol=["B", "A"])
        )
    with pytest.raises(RuntimeError, match="date identity"):
        holdout_module._load_prediction(
            checkpoint,
            expected=frame.assign(
                trading_date=pd.to_datetime(["2026-01-02", "2026-01-02"])
            ),
        )
    with pytest.raises(RuntimeError, match="logical content mismatch"):
        holdout_module._load_prediction_artifact(
            checkpoint.model_copy(update={"frame_sha256": "f" * 64})
        )

    wrong_schema = frame.rename(columns={"prediction": "score"})
    wrong_state = checkpoint.model_copy(update={"component_id": "e" * 64})
    holdout_module._publish_prediction(wrong_state, wrong_schema, tmp_path / "wrong-schema")
    with pytest.raises(RuntimeError, match="schema mismatch"):
        holdout_module._load_prediction_artifact(wrong_state)
    nonfinite = frame.assign(prediction=[0.1, np.inf])
    nonfinite_state = checkpoint.model_copy(update={"component_id": "f" * 64})
    holdout_module._publish_prediction(nonfinite_state, nonfinite, tmp_path / "nonfinite")
    with pytest.raises(RuntimeError, match="non-finite"):
        holdout_module._load_prediction_artifact(nonfinite_state)

    successful = checkpoint.model_copy(update={"status": HoldoutComponentStatus.SUCCEEDED})
    ledger = _ledger_fixture(selection, successful)
    report = _report_fixture(selection, ledger, component)
    report_path = holdout_module.write_locked_holdout_report(report, tmp_path / "reports")
    assert holdout_module.load_locked_holdout_report(report_path) == report
    assert holdout_module.write_locked_holdout_report(report, tmp_path / "reports") == report_path
    with pytest.raises(RuntimeError, match="not content-addressed"):
        holdout_module.load_locked_holdout_report(tmp_path / "report.json")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["report_path"] = str(tmp_path / "wrong.json")
    payload["metadata_hash"] = holdout_module._stable_hash(
        {key: value for key, value in payload.items() if key != "metadata_hash"}
    )
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="path metadata mismatch"):
        holdout_module.load_locked_holdout_report(report_path)

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing or invalid"):
        holdout_module._read_json(invalid_json, description="fixture")
    not_object = tmp_path / "list.json"
    not_object.write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not a JSON object"):
        holdout_module._read_json(not_object, description="fixture")
    with pytest.raises(RuntimeError, match="metadata hash mismatch"):
        holdout_module._authenticate_envelope({}, description="fixture")


def test_selection_and_holdout_cli_publish_auditable_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    selection = _selection_fixture(build_id="a" * 64, dataset_id="b" * 64)
    component = selection.horizons[0].expected_return_component
    checkpoint = HoldoutComponentCheckpoint(
        component_id="c" * 64,
        component_key="h1:regression",
        horizon=1,
        component_name=component.component_name,
        status=HoldoutComponentStatus.SUCCEEDED,
        prediction_path=str(tmp_path / "prediction.parquet"),
        parquet_sha256="d" * 64,
        frame_sha256="e" * 64,
        rows=3,
    )
    ledger = _ledger_fixture(selection, checkpoint)
    report = _report_fixture(selection, ledger, component)
    feature_selection = _feature_selection_fixture(selection)
    ablation_path = tmp_path / "ablation.json"
    candidate_path = tmp_path / "candidate.json"
    build_path = tmp_path / "build.json"
    for path in (ablation_path, candidate_path, build_path):
        path.write_text("{}", encoding="utf-8")
    selection_path = tmp_path / "selection.json"
    selection_path.write_text("{}", encoding="utf-8")
    report_path = tmp_path / "report.json"
    runner = CliRunner()

    monkeypatch.setattr(cli_module, "freeze_development_features", lambda **_: feature_selection)
    monkeypatch.setattr(
        cli_module,
        "write_development_feature_selection",
        lambda *_: selection_path,
    )
    frozen_features = runner.invoke(
        app,
        [
            "research",
            "freeze-features",
            "--ablation-campaign",
            str(ablation_path),
            "--feature-selection-root",
            str(tmp_path / "feature-selections"),
            "--experiment-registry",
            str(tmp_path / "feature-experiments.jsonl"),
        ],
    )
    assert frozen_features.exit_code == 0, frozen_features.output
    assert "feature_choices_complete=true" in frozen_features.output
    assert "holdout_accessed=false" in frozen_features.output

    candidate_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        cli_module,
        "load_development_feature_selection",
        lambda _: feature_selection,
    )
    monkeypatch.setattr(cli_module, "load_campaign_build_id", lambda _: selection.build_id)
    monkeypatch.setattr(
        cli_module,
        "research_campaign",
        lambda **kwargs: candidate_calls.append(kwargs),
    )
    candidates = runner.invoke(
        app,
        [
            "research",
            "candidate-campaigns",
            "--feature-selection",
            str(selection_path),
            "--build-manifest",
            str(build_path),
            "--code-commit",
            "candidate-test",
            "--holdout-periods",
            str(feature_selection.holdout_periods),
            "--campaign-root",
            str(tmp_path / "candidate-campaigns"),
        ],
    )
    assert candidates.exit_code == 0, candidates.output
    assert len(candidate_calls) == 3
    assert [call["horizons"] for call in candidate_calls] == ["1", "5", "20"]
    assert all(call["seeds"] == "17,29,43" for call in candidate_calls)
    assert all(call["run_ablations"] is False for call in candidate_calls)
    assert [
        tuple(str(call["feature_names"]).split(",")) for call in candidate_calls
    ] == [horizon.feature_names for horizon in feature_selection.horizons]
    campaign_root = tmp_path / "candidate-status"
    h1_manifest = campaign_root / feature_selection.feature_selection_id / "h1.json"
    h1_manifest.parent.mkdir(parents=True)
    h1_manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        cli_module,
        "read_campaign_status",
        lambda path: {"manifest": str(path), "current_batch": {"seed": 17}},
    )
    candidate_status = runner.invoke(
        app,
        [
            "research",
            "candidate-status",
            "--feature-selection",
            str(selection_path),
            "--campaign-root",
            str(campaign_root),
        ],
    )
    assert candidate_status.exit_code == 0, candidate_status.output
    candidate_status_payload = json.loads(candidate_status.stdout)
    assert candidate_status_payload["locked_holdout_accessed"] is False
    assert candidate_status_payload["horizons"][0]["campaign"]["current_batch"] == {
        "seed": 17
    }
    assert [item["status"] for item in candidate_status_payload["horizons"][1:]] == [
        "NOT_STARTED",
        "NOT_STARTED",
    ]
    blocked_candidates = runner.invoke(
        app,
        [
            "research",
            "candidate-campaigns",
            "--feature-selection",
            str(selection_path),
            "--build-manifest",
            str(build_path),
            "--code-commit",
            "candidate-test",
            "--seeds",
            "17,29",
        ],
    )
    assert blocked_candidates.exit_code == 2
    assert "at least three unique seeds" in blocked_candidates.output

    monkeypatch.setattr(cli_module, "freeze_development_selection", lambda **_: selection)
    monkeypatch.setattr(cli_module, "write_development_selection", lambda *_: selection_path)
    frozen = runner.invoke(
        app,
        [
            "research",
            "freeze-selection",
            "--ablation-campaign",
            str(ablation_path),
            "--candidate-campaign",
            str(candidate_path),
            "--selection-root",
            str(tmp_path / "selections"),
            "--experiment-registry",
            str(tmp_path / "selection-experiments.jsonl"),
        ],
    )
    assert frozen.exit_code == 0, frozen.output
    assert "choices_complete=true" in frozen.output
    assert "holdout_accessed=false" in frozen.output

    monkeypatch.setattr(
        cli_module,
        "freeze_development_selection",
        lambda **_: (_ for _ in ()).throw(ValueError("incomplete evidence")),
    )
    blocked_selection = runner.invoke(
        app,
        [
            "research",
            "freeze-selection",
            "--ablation-campaign",
            str(ablation_path),
            "--candidate-campaign",
            str(candidate_path),
        ],
    )
    assert blocked_selection.exit_code == 2
    assert "development selection blocked" in blocked_selection.output

    monkeypatch.setattr(
        cli_module,
        "freeze_development_selection_from_features",
        lambda **_: selection,
    )
    finalized = runner.invoke(
        app,
        [
            "research",
            "finalize-selection",
            "--feature-selection",
            str(selection_path),
            "--candidate-campaign",
            str(candidate_path),
            "--selection-root",
            str(tmp_path / "selections-final"),
            "--experiment-registry",
            str(tmp_path / "final-selection-experiments.jsonl"),
        ],
    )
    assert finalized.exit_code == 0, finalized.output
    assert "choices_complete=true" in finalized.output

    monkeypatch.setattr(
        cli_module,
        "evaluate_locked_holdout",
        lambda **_: LockedHoldoutEvaluation(
            report=report,
            report_path=report_path,
            resumed=True,
        ),
    )
    evaluated = runner.invoke(
        app,
        [
            "research",
            "holdout-evaluate",
            "--selection",
            str(selection_path),
            "--build-manifest",
            str(build_path),
            "--code-commit",
            "holdout-test",
            "--evaluation-root",
            str(tmp_path / "holdout"),
            "--experiment-registry",
            str(tmp_path / "holdout-experiments.jsonl"),
        ],
    )
    assert evaluated.exit_code == 0, evaluated.output
    assert "adoption_eligible=false" in evaluated.output
    assert "RESEARCH ONLY" in evaluated.output

    monkeypatch.setattr(
        cli_module,
        "evaluate_locked_holdout",
        lambda **_: (_ for _ in ()).throw(RuntimeError("wrong build")),
    )
    blocked_holdout = runner.invoke(
        app,
        [
            "research",
            "holdout-evaluate",
            "--selection",
            str(selection_path),
            "--build-manifest",
            str(build_path),
            "--code-commit",
            "holdout-test",
        ],
    )
    assert blocked_holdout.exit_code == 2
    assert "locked holdout evaluation blocked" in blocked_holdout.output

    evaluation_directory = tmp_path / "evaluation"
    evaluation_directory.mkdir()
    monkeypatch.setattr(
        cli_module,
        "read_locked_holdout_status",
        lambda _: {"status": "RUNNING", "holdout_accessed": True},
    )
    status = runner.invoke(
        app,
        [
            "research",
            "holdout-status",
            "--evaluation-directory",
            str(evaluation_directory),
        ],
    )
    assert status.exit_code == 0
    assert '"status": "RUNNING"' in status.output


def _feature_selection_fixture(
    selection: DevelopmentSelectionArtifact,
) -> DevelopmentFeatureSelectionArtifact:
    values: dict[str, object] = {
        "feature_selection_id": "0" * 64,
        "created_at": datetime(2026, 1, 2, tzinfo=UTC),
        "build_id": selection.build_id,
        "data_snapshot_id": selection.data_snapshot_id,
        "feature_snapshot_id": selection.feature_snapshot_id,
        "feature_manifest_hash": selection.feature_manifest_hash,
        "ablation_campaign_ids": selection.ablation_campaign_ids,
        "source_report_ids": selection.source_report_ids,
        "source_code_commits": selection.source_code_commits,
        "seeds": selection.seeds,
        "holdout_periods": (
            selection.horizons[0].expected_return_component.source_config.holdout_periods
        ),
        "locked_holdout_start": selection.locked_holdout_start,
        "horizons": tuple(
            HorizonFeatureSelection(
                horizon=horizon.horizon,
                feature_names=horizon.feature_names,
                feature_names_hash=horizon.feature_names_hash,
                feature_families=horizon.feature_families,
            )
            for horizon in selection.horizons
        ),
        "adoption_eligible": False,
    }
    provisional = DevelopmentFeatureSelectionArtifact.model_construct(**values)
    values["feature_selection_id"] = selection_module._stable_hash(
        selection_module._feature_selection_identity(provisional)
    )
    return DevelopmentFeatureSelectionArtifact.model_validate(values)


def _ledger_fixture(
    selection: DevelopmentSelectionArtifact,
    checkpoint: HoldoutComponentCheckpoint,
    **updates: object,
) -> LockedHoldoutLedger:
    values: dict[str, object] = {
        "ledger_id": "0" * 64,
        "selection_id": selection.selection_id,
        "build_id": selection.build_id,
        "data_snapshot_id": selection.data_snapshot_id,
        "evaluator_code_commit": "holdout-test",
        "locked_holdout_start": selection.locked_holdout_start,
        "holdout_accessed": True,
        "status": HoldoutEvaluationStatus.RUNNING,
        "started_at": datetime(2026, 1, 3, tzinfo=UTC),
        "updated_at": datetime(2026, 1, 3, tzinfo=UTC),
        "components": [checkpoint],
        **updates,
    }
    provisional = LockedHoldoutLedger.model_construct(**values)
    values["ledger_id"] = holdout_module._stable_hash(
        holdout_module._ledger_identity(provisional)
    )
    return LockedHoldoutLedger.model_validate(values)


def _report_fixture(
    selection: DevelopmentSelectionArtifact,
    ledger: LockedHoldoutLedger,
    component: FrozenModelComponent,
    **updates: object,
) -> LockedHoldoutReport:
    values: dict[str, object] = {
        "report_id": "0" * 64,
        "created_at": datetime(2026, 1, 3, tzinfo=UTC),
        "selection_id": selection.selection_id,
        "ledger_id": ledger.ledger_id,
        "build_id": selection.build_id,
        "data_snapshot_id": selection.data_snapshot_id,
        "evaluator_code_commit": ledger.evaluator_code_commit,
        "locked_holdout_start": selection.locked_holdout_start,
        "locked_holdout_end": "2025-04-01",
        "feature_definition_hashes": holdout_module._expected_feature_definition_hashes(
            selection
        ),
        "component_results": (
            HoldoutComponentResult(
                component_key=ledger.components[0].component_key,
                component=component,
                metrics=_metrics_for(component, mean_squared_error=0.04),
                prediction_sha256=ledger.components[0].parquet_sha256,
            ),
        ),
        "ensemble_results": tuple(_holdout_ensemble(horizon) for horizon in (1, 5, 20)),
        "adoption_blocking_reasons": ("research only",),
        **updates,
    }
    provisional = LockedHoldoutReport.model_construct(**values)
    values["report_id"] = holdout_module._stable_hash(
        holdout_module._report_identity(provisional)
    )
    return LockedHoldoutReport.model_validate(values)


def _production_fixture(tmp_path: Path) -> tuple[Path, str, str]:
    dates = pd.date_range("2025-01-06", periods=50, freq="B")
    rows: list[dict[str, object]] = []
    feature_names = V0_MANIFEST.feature_names
    for date_index, trading_date in enumerate(dates):
        for symbol_index, symbol in enumerate(("A", "B", "C")):
            row: dict[str, object] = {
                "symbol": symbol,
                "trading_date": trading_date,
                "as_of": trading_date.tz_localize("Asia/Tokyo")
                + timedelta(hours=11, minutes=30),
                "historical_revision_policy": "SINGLE_VINTAGE_AS_REVISED",
                "historical_revision_status": CapabilityStatus.PARTIAL.value,
            }
            for feature_index, feature_name in enumerate(feature_names):
                row[feature_name] = float(
                    (date_index + 1) * (symbol_index + 1) + feature_index
                )
            for horizon in (1, 5, 20):
                target = (symbol_index - 1) * 0.01 + date_index * 0.0001
                endpoint = trading_date + timedelta(days=horizon + 1)
                row[f"target_return_{horizon}d"] = target
                row[f"target_topix_excess_{horizon}d"] = target - 0.001
                row[f"target_sector_excess_{horizon}d"] = target - 0.002
                row[f"target_beta_residual_{horizon}d"] = target - 0.003
                row[f"target_large_loss_{horizon}d"] = int(target <= -0.08)
                row[f"label_end_date_{horizon}d"] = endpoint
                row[f"label_available_at_{horizon}d"] = endpoint.tz_localize("Asia/Tokyo")
                row[f"label_status_{horizon}d"] = "AVAILABLE"
                row[f"label_status_topix_excess_{horizon}d"] = "AVAILABLE"
                row[f"label_status_sector_excess_{horizon}d"] = "AVAILABLE"
                row[f"label_status_beta_residual_{horizon}d"] = "AVAILABLE"
            rows.append(row)
    dataset = pd.DataFrame(rows)
    source_as_of = datetime(2025, 4, 1, tzinfo=UTC)
    source_ids = (("fixture", "source-v1"),)
    snapshot = write_production_dataset_snapshot(
        dataset,
        tmp_path / "datasets",
        source_snapshot_as_of=source_as_of,
        source_snapshot_ids=source_ids,
        as_of=datetime(2026, 1, 1, tzinfo=UTC),
        created_at=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
        label_1230_status=CapabilityStatus.BLOCKED_BY_DATA_CAPABILITY,
    )
    snapshot_ids = {
        "v0_snapshot_id": "0" * 64,
        "v1_snapshot_id": "1" * 64,
        "v2_snapshot_id": "2" * 64,
        "dataset_snapshot_id": snapshot.snapshot_id,
    }
    identity = {
        "source_snapshot_as_of": source_as_of.isoformat(),
        "source_snapshot_ids": source_ids,
        **snapshot_ids,
    }
    build_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    build_directory = tmp_path / "builds" / build_id
    build_directory.mkdir(parents=True)
    build_path = (build_directory / f"{build_id}.json").resolve()
    payload: dict[str, object] = {
        "build_id": build_id,
        "created_at": datetime(2026, 1, 1, 0, 2, tzinfo=UTC).isoformat(),
        "source_snapshot_as_of": source_as_of.isoformat(),
        "source_snapshot_ids": [list(item) for item in source_ids],
        **snapshot_ids,
        "v0_parquet_path": str(tmp_path / "unused-v0.parquet"),
        "v1_parquet_path": str(tmp_path / "unused-v1.parquet"),
        "v2_parquet_path": str(tmp_path / "unused-v2.parquet"),
        "dataset_parquet_path": str(snapshot.parquet_path.resolve()),
        "manifest_path": str(build_path),
    }
    payload["metadata_hash"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    build_path.write_text(json.dumps(payload), encoding="utf-8")
    return build_path, snapshot.snapshot_id, build_id


def _selection_fixture(*, build_id: str, dataset_id: str) -> DevelopmentSelectionArtifact:
    feature_names = V0_MANIFEST.feature_names
    horizons: list[HorizonDevelopmentSelection] = []
    report_ids: list[str] = []
    for horizon in (1, 5, 20):
        config = AdvancedResearchConfig(
            horizons=(horizon,),
            model_families=("lightgbm",),
            seeds=(17,),
            holdout_periods=10,
            estimator_count=5,
            tuning_trials=1,
            tuning_timeout_seconds=10,
            run_ablations=False,
            run_diagnostics=False,
        )
        components: dict[str, FrozenModelComponent] = {}
        for task, metric_name, metric_value in (
            ("regression", "mean_daily_rank_ic", 0.10),
            ("ranking", "mean_daily_rank_ic", 0.09),
            ("quantile", "pinball_loss", 0.02),
            ("large_loss", "brier_score", 0.05),
        ):
            report_id = hashlib.sha256(f"{horizon}:{task}".encode()).hexdigest()
            report_ids.append(report_id)
            name = f"lightgbm:{task}:seed=17"
            components[task] = FrozenModelComponent(
                component_name=name,
                horizon=horizon,
                model_family="lightgbm",
                task=task,  # type: ignore[arg-type]
                seed=17,
                parameters={},
                source_config=config,
                source_report_id=report_id,
                source_config_hash=config.config_hash,
                selection_metric_name=metric_name,
                selection_metric_value=metric_value,
            )
        family_votes = tuple(
            FeatureFamilySelection(
                horizon=horizon,
                family_id=f"F{number}",
                family_name=f"family-{number}",
                seeds=(17, 29, 43),
                seed_votes_selected=(),
                selected=False,
                selected_features=(),
                evidence_reports=3,
                blocked_reports=0,
            )
            for number in range(1, 13)
        )
        ensemble = EnsembleResult(
            horizon=horizon,
            component_names=(
                components["ranking"].component_name,
                components["regression"].component_name,
            ),
            weights=(0.5, 0.5),
            mean_daily_rank_ic=0.20,
            mean_pairwise_correlation=0.8,
            mean_disagreement=0.1,
            uncertainty_error_correlation=0.2,
            meta_fit_rows=20,
            meta_evaluation_rows=10,
        )
        horizons.append(
            HorizonDevelopmentSelection(
                horizon=horizon,
                feature_names=feature_names,
                feature_names_hash=selection_module._stable_hash(feature_names),
                feature_families=family_votes,
                expected_return_component=components["regression"],
                rank_component=components["regression"],
                downside_quantile_component=components["quantile"],
                large_loss_component=components["large_loss"],
                ensemble_components=(components["ranking"], components["regression"]),
                ensemble=ensemble,
                ensemble_adopted=True,
                ensemble_best_component_rank_ic=0.10,
                uncertainty=UncertaintyCalibration(
                    horizon=horizon,
                    residual_quantile_80=0.2,
                    residual_quantile_90=0.3,
                    empirical_coverage_80=0.8,
                    empirical_coverage_90=0.9,
                    disagreement_error_correlation=0.2,
                    calibration_rows=20,
                    evaluation_rows=10,
                ),
            )
        )
    values: dict[str, object] = {
        "selection_id": "0" * 64,
        "created_at": datetime(2026, 1, 2, tzinfo=UTC),
        "build_id": build_id,
        "data_snapshot_id": dataset_id,
        "feature_snapshot_id": "2" * 64,
        "feature_manifest_hash": "3" * 64,
        "candidate_campaign_ids": ("4" * 64,),
        "ablation_campaign_ids": ("5" * 64,),
        "source_report_ids": tuple(report_ids),
        "source_code_commits": ("candidate-test",),
        "seeds": (17, 29, 43),
        "locked_holdout_start": "2025-03-03",
        "locked_holdout_accessed": False,
        "horizons": tuple(horizons),
        "adoption_eligible": False,
        "adoption_blocking_reasons": ("development only",),
    }
    provisional = DevelopmentSelectionArtifact.model_construct(**values)
    values["selection_id"] = selection_module._stable_hash(
        selection_module._selection_identity(provisional)
    )
    return DevelopmentSelectionArtifact.model_validate(values)
