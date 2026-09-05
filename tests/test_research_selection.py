from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pandas as pd
import pytest

import stock_ai.ml.advanced as advanced_module
from stock_ai.data.contracts import CapabilityStatus
from stock_ai.features import V0_MANIFEST, V2_EXTENDED_MANIFEST
from stock_ai.ml.advanced import (
    AdvancedModelMetrics,
    AdvancedResearchConfig,
    AdvancedResearchReport,
    AdvancedResearchRun,
    FeatureAblationResult,
    TuningResult,
    UncertaintyCalibration,
    feature_family_ablation_plan,
    load_authenticated_advanced_oof_slice,
    write_advanced_research_run,
)
from stock_ai.ml.campaign import (
    CampaignBatchStatus,
    create_campaign_manifest,
    load_campaign_manifest,
    write_campaign_manifest,
)
from stock_ai.ml.selection import (
    freeze_development_features,
    freeze_development_selection,
    freeze_development_selection_from_features,
    load_development_feature_selection,
    load_development_selection,
    write_development_feature_selection,
    write_development_selection,
)


def test_complete_development_selection_is_content_addressed_and_holdout_closed(
    tmp_path: Path,
) -> None:
    ablation_path, candidate_path = _campaign_fixtures(tmp_path)
    frozen_at = datetime(2026, 9, 4, 12, 0, tzinfo=UTC)

    selection = freeze_development_selection(
        ablation_campaign_paths=(ablation_path,),
        candidate_campaign_paths=(candidate_path,),
        created_at=frozen_at,
    )

    assert selection.locked_holdout_accessed is False
    assert selection.feature_selection_complete
    assert selection.model_selection_complete
    assert selection.hyperparameter_selection_complete
    assert selection.ensemble_selection_complete
    assert tuple(item.horizon for item in selection.horizons) == (1, 5, 20)
    expected_features = tuple(
        name
        for name in V2_EXTENDED_MANIFEST.feature_names
        if name in {*V0_MANIFEST.feature_names, *_family_features("F1")}
    )
    assert all(item.feature_names == expected_features for item in selection.horizons)
    assert all(
        next(family for family in item.feature_families if family.family_id == "F1").selected
        for item in selection.horizons
    )
    assert all(item.expected_return_component.task == "regression" for item in selection.horizons)
    assert all(item.downside_quantile_component.task == "quantile" for item in selection.horizons)
    assert all(item.large_loss_component.task == "large_loss" for item in selection.horizons)

    path = write_development_selection(selection, tmp_path / "selections")
    assert load_development_selection(path) == selection
    assert write_development_selection(selection, tmp_path / "selections") == path

    feature_selection = freeze_development_features(
        ablation_campaign_paths=(ablation_path,),
        created_at=datetime(2026, 9, 4, 11, 0, tzinfo=UTC),
    )
    assert not feature_selection.locked_holdout_accessed
    assert tuple(item.horizon for item in feature_selection.horizons) == (1, 5, 20)
    assert all(item.feature_names == expected_features for item in feature_selection.horizons)
    feature_path = write_development_feature_selection(
        feature_selection,
        tmp_path / "feature-selections",
    )
    assert load_development_feature_selection(feature_path) == feature_selection
    assert (
        write_development_feature_selection(
            feature_selection,
            tmp_path / "feature-selections",
        )
        == feature_path
    )
    finalized = freeze_development_selection_from_features(
        feature_selection_path=feature_path,
        candidate_campaign_paths=(candidate_path,),
        created_at=frozen_at,
    )
    assert finalized == selection

    candidate_manifest = load_campaign_manifest(candidate_path)
    candidate_oof_path = candidate_manifest.batches[0].oof_path
    assert candidate_oof_path is not None
    sliced_report, sliced = load_authenticated_advanced_oof_slice(
        Path(candidate_oof_path), tasks=("regression", "ranking")
    )
    assert sliced_report.report_id == candidate_manifest.batches[0].report_id
    assert set(sliced["task"]) == {"regression", "ranking"}
    assert tuple(sliced.columns) == (
        "symbol",
        "trading_date",
        "target",
        "label_end",
        "prediction",
        "task",
    )
    with pytest.raises(ValueError, match="unique non-empty task subset"):
        load_authenticated_advanced_oof_slice(Path(candidate_oof_path), tasks=())


def test_selection_fails_closed_for_incomplete_candidate_matrix(tmp_path: Path) -> None:
    ablation_path, candidate_path = _campaign_fixtures(tmp_path)
    payload = json.loads(candidate_path.read_text(encoding="utf-8"))
    payload["batches"] = payload["batches"][:-1]
    # Recompute by parsing/re-emitting the mutated plan is intentionally impossible: the
    # campaign identity itself must fail before selection can infer a smaller matrix.
    candidate_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="identity mismatch"):
        freeze_development_selection(
            ablation_campaign_paths=(ablation_path,),
            candidate_campaign_paths=(candidate_path,),
        )


def test_selection_metadata_tamper_is_rejected(tmp_path: Path) -> None:
    ablation_path, candidate_path = _campaign_fixtures(tmp_path)
    selection = freeze_development_selection(
        ablation_campaign_paths=(ablation_path,),
        candidate_campaign_paths=(candidate_path,),
    )
    path = write_development_selection(selection, tmp_path / "selections")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selection"]["locked_holdout_start"] = "1999-01-01"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="metadata hash mismatch"):
        load_development_selection(path)


def _campaign_fixtures(tmp_path: Path) -> tuple[Path, Path]:
    build_id, build_path = _build_marker(tmp_path)
    full_features = V2_EXTENDED_MANIFEST.feature_names
    selected_features = tuple(
        name
        for name in full_features
        if name in {*V0_MANIFEST.feature_names, *_family_features("F1")}
    )
    ablation = _campaign(
        tmp_path,
        name="ablation",
        build_id=build_id,
        build_path=build_path,
        families=("lightgbm",),
        features=full_features,
        run_ablations=True,
    )
    candidate = _campaign(
        tmp_path,
        name="candidate",
        build_id=build_id,
        build_path=build_path,
        families=("lightgbm", "xgboost", "catboost"),
        features=selected_features,
        run_ablations=False,
    )
    return ablation, candidate


def _campaign(
    tmp_path: Path,
    *,
    name: str,
    build_id: str,
    build_path: Path,
    families: tuple[str, ...],
    features: tuple[str, ...],
    run_ablations: bool,
) -> Path:
    root = tmp_path / name
    common = {
        "target_family": "return",
        "seeds": (17, 29, 43),
        "tuning_trials": 1,
        "tuning_timeout_seconds": 10,
        "estimator_count": 5,
        "initial_train_periods": 20,
        "validation_periods": 5,
        "step_periods": 5,
        "holdout_periods": 5,
        "run_ablations": run_ablations,
        "run_diagnostics": False,
        "max_materialized_oof_rows": 10_000,
        "max_model_fits": 100,
        "feature_names": features,
    }
    manifest = create_campaign_manifest(
        build_id=build_id,
        build_manifest_path=build_path,
        code_commit=f"{name}-commit",
        report_root=root / "reports",
        experiment_registry=root / "experiments.jsonl",
        horizons=(1, 5, 20),
        model_families=families,
        common_config=common,
        checkpoint_root=root / "checkpoints",
        now=datetime(2026, 9, 4, 9, 0, tzinfo=UTC),
    )
    for batch in manifest.batches:
        config = AdvancedResearchConfig.model_validate(
            {
                **{key: value for key, value in common.items() if key != "feature_names"},
                "horizons": (batch.horizon,),
                "model_families": (batch.model_family,),
                "seeds": (cast(int, batch.seed),),
            }
        )
        run = _run_fixture(
            config=config,
            feature_names=features,
            run_ablations=run_ablations,
            code_commit=manifest.code_commit,
        )
        _, oof_path = write_advanced_research_run(run, Path(manifest.report_root))
        batch.status = CampaignBatchStatus.SUCCEEDED
        batch.attempts = 1
        batch.report_id = run.report.report_id
        batch.oof_path = str(oof_path.resolve())
        batch.completed_at = datetime(2026, 9, 4, 10, 0, tzinfo=UTC)
    path = root / "campaign.json"
    write_campaign_manifest(manifest, path)
    return path


def _build_marker(tmp_path: Path) -> tuple[str, Path]:
    source_as_of = datetime(2026, 9, 1, tzinfo=UTC).isoformat()
    source_ids = (("fixture", "source"),)
    identity = {
        "source_snapshot_as_of": source_as_of,
        "source_snapshot_ids": source_ids,
        "v0_snapshot_id": "0" * 64,
        "v1_snapshot_id": "1" * 64,
        "v2_snapshot_id": "c" * 64,
        "dataset_snapshot_id": "b" * 64,
    }
    build_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    directory = tmp_path / "builds" / build_id
    directory.mkdir(parents=True)
    path = (directory / f"{build_id}.json").resolve()
    payload: dict[str, object] = {
        "build_id": build_id,
        "created_at": datetime(2026, 9, 1, 0, 1, tzinfo=UTC).isoformat(),
        "source_snapshot_as_of": source_as_of,
        "source_snapshot_ids": [list(item) for item in source_ids],
        **{key: value for key, value in identity.items() if key.endswith("snapshot_id")},
        "manifest_path": str(path),
    }
    payload["metadata_hash"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")
    return build_id, path


def _run_fixture(
    *,
    config: AdvancedResearchConfig,
    feature_names: tuple[str, ...],
    run_ablations: bool,
    code_commit: str,
) -> AdvancedResearchRun:
    horizon = config.horizons[0]
    family = config.model_families[0]
    seed = config.seeds[0]
    oof = _oof_fixture(horizon=horizon, family=family, seed=seed)
    oof_hash = advanced_module._frame_hash(oof)
    family_bias = {"lightgbm": 0.003, "xgboost": 0.002, "catboost": 0.001}[family]
    seed_bias = {17: 0.0003, 29: 0.0002, 43: 0.0001}[seed]
    metrics = tuple(
        AdvancedModelMetrics(
            horizon=horizon,
            model_family=family,  # type: ignore[arg-type]
            task=task,  # type: ignore[arg-type]
            seed=seed,
            folds=2,
            rows=len(oof) // 4,
            dates=80,
            mean_squared_error=0.01 + task_index * 0.001,
            mean_daily_rank_ic=0.02 + family_bias + seed_bias - task_index * 0.001,
            rank_ic_standard_deviation=0.01,
            rank_icir=2.0,
            ndcg_at_5=0.5,
            ndcg_at_10=0.5,
            ndcg_at_20=0.5,
            precision_at_5=0.5,
            precision_at_10=0.5,
            precision_at_20=0.5,
            top_5_mean_target=0.01,
            top_10_mean_target=0.01,
            top_20_mean_target=0.01,
            pinball_loss=(0.01 + family_bias if task == "quantile" else None),
            lower_tail_rate=(0.1 if task == "quantile" else None),
            brier_score=(0.1 + family_bias if task == "large_loss" else None),
            log_loss=(0.2 if task == "large_loss" else None),
            expected_calibration_error=(0.05 if task == "large_loss" else None),
        )
        for task_index, task in enumerate(("regression", "ranking", "quantile", "large_loss"))
    )
    ablations = _ablation_results(horizon=horizon, seed=seed) if run_ablations else ()
    tuning = TuningResult(
        horizon=horizon,
        model_family=family,  # type: ignore[arg-type]
        trials_completed=1,
        timeout_seconds=10,
        best_value=0.02,
        best_parameters={"depth": 3, "learning_rate": 0.05},
        trials=(),
    )
    values = {
        "report_id": "0" * 64,
        "created_at": datetime(2026, 9, 4, 10, 0, tzinfo=UTC),
        "code_commit": code_commit,
        "hypothesis": config.hypothesis,
        "config": config,
        "config_hash": config.config_hash,
        "data_snapshot_id": "b" * 64,
        "feature_snapshot_id": "c" * 64,
        "feature_set_id": V2_EXTENDED_MANIFEST.feature_set_id,
        "feature_set_version": V2_EXTENDED_MANIFEST.feature_set_version,
        "preprocessing_version": V2_EXTENDED_MANIFEST.preprocessing_version,
        "feature_manifest_hash": V2_EXTENDED_MANIFEST.manifest_hash,
        "feature_definition_hashes": {
            name: V2_EXTENDED_MANIFEST.feature_definition_hashes[name] for name in feature_names
        },
        "feature_names": feature_names,
        "prediction_semantics": "return",
        "locked_holdout_start": "2026-08-01",
        "locked_holdout_accessed": False,
        "historical_revision_policy": "SINGLE_VINTAGE_AS_REVISED",
        "historical_revision_status": CapabilityStatus.PARTIAL,
        "adoption_eligible": False,
        "adoption_blocking_reasons": ("development only",),
        "cost_scenarios_bps": (0, 10, 25, 50),
        "cost_evaluation_status": CapabilityStatus.BLOCKED_BY_DATA_CAPABILITY,
        "tax_policy_version": "NOT_APPLIED_MODEL_RESEARCH",
        "decision_engine_version": "decision-engine-v1",
        "library_versions": (("fixture", "1"),),
        "stage_boundaries": (),
        "tuning_results": (tuning,),
        "fold_results": (),
        "model_metrics": metrics,
        "ablations": ablations,
        "ensembles": (),
        "uncertainty": (
            UncertaintyCalibration(
                horizon=horizon,
                residual_quantile_80=0.2,
                residual_quantile_90=0.3,
                empirical_coverage_80=0.8,
                empirical_coverage_90=0.9,
                disagreement_error_correlation=0.1,
                calibration_rows=6,
                evaluation_rows=6,
            ),
        ),
        "feature_diagnostics": (),
        "seed_stability": (),
        "oof_rows": len(oof),
        "oof_sha256": oof_hash,
    }
    provisional = AdvancedResearchReport.model_validate(values)
    report = provisional.model_copy(
        update={
            "report_id": advanced_module._stable_hash(
                advanced_module._report_identity(provisional)
            )
        }
    )
    return AdvancedResearchRun(report=report, oof_predictions=oof)


def _oof_fixture(*, horizon: int, family: str, seed: int) -> pd.DataFrame:
    dates = pd.date_range("2025-01-06", periods=80, freq="B")
    family_offset = {"lightgbm": 0.03, "xgboost": 0.02, "catboost": 0.01}[family]
    seed_offset = {17: 0.003, 29: 0.002, 43: 0.001}[seed]
    rows: list[dict[str, object]] = []
    for task_index, task in enumerate(("regression", "ranking", "quantile", "large_loss")):
        for date_index, trading_date in enumerate(dates):
            for symbol_index, symbol in enumerate(("A", "B", "C")):
                target = (symbol_index - 1) * 0.01 + date_index * 0.0001
                prediction = target + family_offset + seed_offset + task_index * 0.00001
                if task == "large_loss":
                    prediction = 0.05 + symbol_index * 0.01
                rows.append(
                    {
                        "symbol": symbol,
                        "trading_date": trading_date,
                        "as_of": trading_date.tz_localize("Asia/Tokyo")
                        + timedelta(hours=11, minutes=30),
                        "target": target,
                        "label_end": trading_date
                        + timedelta(days=int(horizon) + 1),
                        "horizon": horizon,
                        "model_family": family,
                        "task": task,
                        "seed": seed,
                        "fold": date_index // 4,
                        "prediction": prediction,
                        "retained_feature_count": 2,
                    }
                )
    return pd.DataFrame(rows).sort_values(
        ["horizon", "model_family", "task", "seed", "trading_date", "symbol"],
        kind="stable",
    ).reset_index(drop=True)


def _ablation_results(*, horizon: int, seed: int) -> tuple[FeatureAblationResult, ...]:
    results = [
        FeatureAblationResult(
            family_id="F0",
            family_name="FeatureSet V0 baseline",
            horizon=horizon,
            status=CapabilityStatus.AVAILABLE,
            added_features=V0_MANIFEST.feature_names,
            mean_daily_rank_ic=0.01,
            selection_rank_ic=0.01,
            selected_on_tuning_period=True,
        )
    ]
    for item in feature_family_ablation_plan():
        if item.family_id == "F0":
            continue
        selected = item.family_id == "F1" and seed in (17, 29)
        results.append(
            item.model_copy(
                update={
                    "horizon": horizon,
                    "status": CapabilityStatus.AVAILABLE,
                    "selection_rank_ic": 0.02 if selected else 0.009,
                    "selected_on_tuning_period": selected,
                }
            )
        )
    return tuple(results)


def _family_features(family_id: str) -> tuple[str, ...]:
    return next(
        item.added_features
        for item in feature_family_ablation_plan()
        if item.family_id == family_id
    )
