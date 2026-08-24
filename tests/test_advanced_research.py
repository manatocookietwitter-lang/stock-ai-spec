from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import stock_ai.ml.advanced as advanced_module
from stock_ai.decision import (
    CostPolicy,
    DailyPortfolioDecisionEngine,
    DecisionCandidate,
    DecisionEngineConfig,
    SimpleJapanTaxEngine,
    TaxPolicy,
    TransactionCostEngine,
)
from stock_ai.domain import (
    Account,
    AccountBucket,
    AccountType,
    CashState,
    PortfolioState,
    Security,
    TaxState,
    WithholdingMode,
)
from stock_ai.features import V0_MANIFEST, V1_CORE_MANIFEST, V2_EXTENDED_MANIFEST
from stock_ai.ml.advanced import (
    AdvancedResearchConfig,
    AdvancedResearchExecutionError,
    AdvancedResearchRun,
    FoldPreprocessor,
    TrialAudit,
    TuningResult,
    TuningSearchError,
    bounded_optuna_search,
    build_decision_compatible_oof_predictions,
    calibrate_oof_uncertainty,
    feature_family_ablation_plan,
    fit_oof_ensemble,
    generate_oof_predictions,
    load_advanced_research_run,
    run_advanced_research,
    write_advanced_research_run,
)
from stock_ai.ml.research_metrics import (
    cross_sectional_relevance,
    evaluate_cross_sectional_predictions,
)


def test_cross_sectional_metrics_are_date_grouped_and_exact() -> None:
    dates = pd.Series(["2026-01-05"] * 5 + ["2026-01-06"] * 5)
    target = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 8.0, 6.0, 4.0, 2.0])
    prediction = target.copy()

    metrics = evaluate_cross_sectional_predictions(
        dates=dates,
        target=target,
        prediction=prediction,
    )

    assert metrics.mean_daily_rank_ic == pytest.approx(1.0)
    assert metrics.ndcg_at_5 == pytest.approx(1.0)
    assert metrics.precision_at_5 == pytest.approx(1.0)
    assert metrics.mean_squared_error == 0.0
    assert cross_sectional_relevance(target.iloc[:5], dates.iloc[:5]).tolist() == [0, 1, 2, 3, 4]


def test_fold_preprocessor_is_fit_only_on_training_values() -> None:
    train = pd.DataFrame(
        {
            "a": [0.0, 1.0, 2.0, 3.0],
            "b": [0.0, 2.0, 4.0, 6.0],
            "c": [1.0, 0.0, 1.0, 0.0],
        }
    )
    unfitted = FoldPreprocessor(
        ("a", "b", "c"),
        lower_quantile=0.0,
        upper_quantile=1.0,
        correlation_threshold=0.99,
    )
    with pytest.raises(RuntimeError, match="has not been fitted"):
        _ = unfitted.retained_features
    with pytest.raises(RuntimeError, match="must be fitted"):
        unfitted.transform(train)
    processor = FoldPreprocessor(
        ("a", "b", "c"),
        lower_quantile=0.0,
        upper_quantile=1.0,
        correlation_threshold=0.99,
    ).fit(train)
    before = processor.transform(pd.DataFrame({"a": [np.nan], "b": [3.0], "c": [1.0]}))
    processor.transform(pd.DataFrame({"a": [1e12], "b": [-1e12], "c": [1e12]}))
    after = processor.transform(pd.DataFrame({"a": [np.nan], "b": [3.0], "c": [1.0]}))

    pdt.assert_frame_equal(before, after)
    assert "b" not in processor.retained_features


def test_all_failed_optuna_trials_remain_available_for_failure_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _small_model_frame()
    config = _config(model_families=("lightgbm",))

    def forced_failure(**_: object) -> np.ndarray:
        raise RuntimeError("forced trial failure")

    monkeypatch.setattr(advanced_module, "_fit_predict", forced_failure)
    with pytest.raises(TuningSearchError) as captured:
        bounded_optuna_search(
            frame,
            feature_names=("f1", "f2"),
            target_column="target",
            label_end_column="label_end",
            horizon=1,
            family="lightgbm",
            config=config,
        )

    error = captured.value
    assert error.horizon == 1
    assert error.model_family == "lightgbm"
    assert len(error.trials) == config.tuning_trials
    assert {trial.state for trial in error.trials} == {"FAIL"}
    assert all(trial.parameters for trial in error.trials)
    assert all("forced trial failure" in (trial.failure_reason or "") for trial in error.trials)


def test_later_tuning_failure_preserves_prior_completed_trial_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    complete = TrialAudit(
        number=0,
        state="COMPLETE",
        parameters={"max_depth": 3},
        value=0.1,
        duration_seconds=0.2,
    )
    failed = TrialAudit(
        number=0,
        state="FAIL",
        parameters={"max_depth": 4},
        duration_seconds=0.3,
        failure_reason="forced later family failure",
    )

    def controlled_search(*_: object, family: str, horizon: int, **__: object) -> TuningResult:
        if family == "lightgbm":
            return TuningResult(
                horizon=horizon,
                model_family="lightgbm",
                trials_completed=1,
                timeout_seconds=10,
                best_value=0.1,
                best_parameters={"max_depth": 3},
                trials=(complete,),
            )
        raise TuningSearchError(
            "forced later family failure",
            horizon=horizon,
            model_family="xgboost",
            trials=(failed,),
        )

    monkeypatch.setattr(advanced_module, "bounded_optuna_search", controlled_search)
    with pytest.raises(AdvancedResearchExecutionError) as captured:
        run_advanced_research(
            _production_dataset(),
            data_snapshot_id="a" * 64,
            created_at=datetime(2026, 8, 24, 12, 0, tzinfo=UTC),
            code_commit="goal3-partial-audit-test",
            config=_config(model_families=("lightgbm", "xgboost")),
            feature_snapshot_id="b" * 64,
            feature_manifest_hash=V2_EXTENDED_MANIFEST.manifest_hash,
            feature_names=_research_features(),
        )

    assert [
        (horizon, family, trial.state)
        for horizon, family, trial in captured.value.trial_contexts
    ] == [(1, "lightgbm", "COMPLETE"), (1, "xgboost", "FAIL")]


def test_post_tuning_failure_preserves_completed_trials_and_generated_folds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_generate = advanced_module.generate_oof_predictions
    call_count = 0

    def fail_after_one_oof(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            raise RuntimeError("forced post-tuning OOF failure")
        return original_generate(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(advanced_module, "generate_oof_predictions", fail_after_one_oof)
    with pytest.raises(AdvancedResearchExecutionError) as captured:
        run_advanced_research(
            _production_dataset(),
            data_snapshot_id="a" * 64,
            created_at=datetime(2026, 8, 24, 12, 0, tzinfo=UTC),
            code_commit="goal3-partial-fold-audit-test",
            config=_config(model_families=("lightgbm",)),
            feature_snapshot_id="b" * 64,
            feature_manifest_hash=V2_EXTENDED_MANIFEST.manifest_hash,
            feature_names=_research_features(),
        )

    error = captured.value
    assert "forced post-tuning OOF failure" in str(error)
    assert {trial.state for _, _, trial in error.trial_contexts} == {"COMPLETE"}
    assert error.fold_results
    assert {fold.task for fold in error.fold_results} == {"regression"}


def test_oof_progress_sink_preserves_completed_fold_before_later_fold_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_fit = advanced_module._fit_predict
    fit_count = 0

    def fail_second_fold(**kwargs: object) -> np.ndarray:
        nonlocal fit_count
        fit_count += 1
        if fit_count == 2:
            raise RuntimeError("forced second-fold failure")
        return original_fit(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(advanced_module, "_fit_predict", fail_second_fold)
    completed_folds: list[pd.DataFrame] = []
    with pytest.raises(RuntimeError, match="second-fold failure"):
        generate_oof_predictions(
            _small_model_frame(),
            feature_names=("f1", "f2"),
            target_column="target",
            label_end_column="label_end",
            horizon=1,
            family="lightgbm",
            task="regression",
            seed=17,
            parameters={},
            config=_config(model_families=("lightgbm",)),
            progress_sink=completed_folds,
        )

    assert len(completed_folds) == 1
    assert completed_folds[0]["fold"].unique().tolist() == [0]
    assert completed_folds[0]["task"].unique().tolist() == ["regression"]


def test_nonfinite_later_fold_does_not_poison_prior_completed_fold_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_fit = advanced_module._fit_predict
    fit_count = 0

    def nonfinite_second_fold(**kwargs: object) -> np.ndarray:
        nonlocal fit_count
        fit_count += 1
        validation_x = kwargs["validation_x"]
        assert isinstance(validation_x, pd.DataFrame)
        if fit_count == 2:
            return np.full(len(validation_x), np.nan)
        return original_fit(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(advanced_module, "_fit_predict", nonfinite_second_fold)
    completed_folds: list[pd.DataFrame] = []
    with pytest.raises(RuntimeError, match="non-finite fold predictions"):
        generate_oof_predictions(
            _small_model_frame(),
            feature_names=("f1", "f2"),
            target_column="target",
            label_end_column="label_end",
            horizon=1,
            family="lightgbm",
            task="regression",
            seed=17,
            parameters={},
            config=_config(model_families=("lightgbm",)),
            progress_sink=completed_folds,
        )

    assert len(completed_folds) == 1
    assert np.isfinite(completed_folds[0]["prediction"].to_numpy(dtype=float)).all()


@pytest.mark.parametrize("family", ["lightgbm", "xgboost", "catboost"])
@pytest.mark.parametrize("task", ["regression", "ranking", "quantile", "large_loss"])
def test_all_goal3_model_adapters_emit_finite_oof(family: str, task: str) -> None:
    frame = _small_model_frame()
    config = _config(model_families=(family,))

    oof = generate_oof_predictions(
        frame,
        feature_names=("f1", "f2"),
        target_column="target",
        label_end_column="label_end",
        horizon=1,
        family=family,  # type: ignore[arg-type]
        task=task,  # type: ignore[arg-type]
        seed=17,
        parameters={},
        config=config,
    )

    assert len(oof) > 0
    assert np.isfinite(oof["prediction"]).all()
    assert not oof.duplicated(
        ["symbol", "trading_date", "horizon", "model_family", "task", "seed"]
    ).any()


def test_advanced_run_never_uses_locked_holdout_and_is_authenticated(tmp_path: Path) -> None:
    dataset = _production_dataset()
    config = _config(model_families=("lightgbm",))
    created_at = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)
    run = run_advanced_research(
        dataset,
        data_snapshot_id="a" * 64,
        created_at=created_at,
        code_commit="goal3-test",
        config=config,
        feature_snapshot_id="b" * 64,
        feature_manifest_hash=V2_EXTENDED_MANIFEST.manifest_hash,
        feature_names=_research_features(),
    )
    holdout_start = pd.Timestamp(run.report.locked_holdout_start)
    mutated = dataset.copy()
    holdout = pd.to_datetime(mutated["trading_date"]) >= holdout_start
    mutated.loc[holdout, "target_return_1d"] = 999.0
    rerun = run_advanced_research(
        mutated,
        data_snapshot_id="a" * 64,
        created_at=created_at,
        code_commit="goal3-test",
        config=config,
        feature_snapshot_id="b" * 64,
        feature_manifest_hash=V2_EXTENDED_MANIFEST.manifest_hash,
        feature_names=_research_features(),
    )

    pdt.assert_frame_equal(run.oof_predictions, rerun.oof_predictions)
    assert run.report.report_id == rerun.report.report_id
    assert not run.report.locked_holdout_accessed
    assert not run.report.adoption_eligible
    assert run.report.tuning_results[0].trials_completed == 1
    assert len(run.report.tuning_results[0].trials) == 1
    assert run.report.tuning_results[0].trials[0].state == "COMPLETE"
    assert run.report.config == config
    assert run.report.feature_snapshot_id == "b" * 64
    evaluation_start = pd.Timestamp(run.report.stage_boundaries[0].model_evaluation_start)
    assert pd.to_datetime(run.oof_predictions["trading_date"]).min() >= evaluation_start
    outer_mutated = dataset.copy()
    first_outer_date = pd.to_datetime(outer_mutated["trading_date"]).eq(evaluation_start)
    outer_mutated.loc[first_outer_date, "target_return_1d"] = 777.0
    outer_rerun = run_advanced_research(
        outer_mutated,
        data_snapshot_id="a" * 64,
        created_at=created_at,
        code_commit="goal3-test",
        config=config,
        feature_snapshot_id="b" * 64,
        feature_manifest_hash=V2_EXTENDED_MANIFEST.manifest_hash,
        feature_names=_research_features(),
    )
    assert (
        outer_rerun.report.tuning_results[0].best_parameters
        == run.report.tuning_results[0].best_parameters
    )
    assert outer_rerun.report.tuning_results[0].best_value == pytest.approx(
        run.report.tuning_results[0].best_value
    )
    original_first_outer = run.oof_predictions.loc[
        pd.to_datetime(run.oof_predictions["trading_date"]).eq(evaluation_start), "prediction"
    ].reset_index(drop=True)
    mutated_first_outer = outer_rerun.oof_predictions.loc[
        pd.to_datetime(outer_rerun.oof_predictions["trading_date"]).eq(evaluation_start),
        "prediction",
    ].reset_index(drop=True)
    pdt.assert_series_equal(original_first_outer, mutated_first_outer)
    assert len(run.report.model_metrics) == 4
    assert 0.0 <= run.report.uncertainty[0].empirical_coverage_90 <= 1.0
    assert run.report.uncertainty[0].calibration_rows > 0
    assert run.report.uncertainty[0].evaluation_rows > 0
    assert len(run.report.feature_diagnostics) == len(_research_features())
    assert len(run.report.seed_stability) == 2
    assert sum(run.report.ensembles[0].weights) == pytest.approx(1.0)
    metadata_path, parquet_path = write_advanced_research_run(run, tmp_path)
    assert metadata_path.exists()
    loaded_report, loaded_oof = load_advanced_research_run(parquet_path)
    assert loaded_report == run.report
    pdt.assert_frame_equal(loaded_oof, run.oof_predictions)
    repeated_metadata, repeated_parquet = write_advanced_research_run(run, tmp_path)
    assert (repeated_metadata, repeated_parquet) == (metadata_path, parquet_path)
    later_run = AdvancedResearchRun(
        report=run.report.model_copy(update={"created_at": created_at + timedelta(seconds=1)}),
        oof_predictions=run.oof_predictions,
    )
    assert write_advanced_research_run(later_run, tmp_path) == (metadata_path, parquet_path)
    parquet_path.write_bytes(parquet_path.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        load_advanced_research_run(parquet_path)


@pytest.mark.parametrize(
    "override",
    [
        {"horizons": (1, 1)},
        {"horizons": ()},
        {"model_families": ("lightgbm", "lightgbm")},
        {"model_families": ()},
        {"seeds": ()},
        {"ablation_families": ("F13",)},
    ],
)
def test_advanced_config_rejects_ambiguous_research_boundaries(
    override: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "initial_train_periods": 20,
        "holdout_periods": 5,
        **override,
    }
    with pytest.raises(ValueError):
        AdvancedResearchConfig.model_validate(values)


def test_feature_family_plan_blocks_unavailable_families_without_guessing() -> None:
    plan = {item.family_id: item for item in feature_family_ablation_plan()}

    assert len(plan) == 12
    assert plan["F2"].added_features
    assert plan["F11"].status.value == "BLOCKED_BY_DATA_CAPABILITY"
    assert plan["F12"].status.value == "AVAILABLE"
    assert any(name.startswith("candle.") for name in plan["F12"].added_features)
    assert not set(plan["F8"].added_features) & set(plan["F9"].added_features)


def test_incremental_ablation_selects_on_tuning_and_reports_on_later_outer_rows() -> None:
    frame = _small_model_frame()
    frame["price.return_1d"] = frame["f1"]
    frame["price.return_5d"] = frame["f2"]
    frame["risk.downside_vol_20d"] = frame["f1"] * frame["f2"]
    frame["candle.body_pct_close"] = frame["f1"] - frame["f2"]
    feature_names = (
        "price.return_1d",
        "price.return_5d",
        "risk.downside_vol_20d",
        "candle.body_pct_close",
    )
    config = _config(model_families=("lightgbm",)).model_copy(
        update={"ablation_families": ("F6", "F12")}
    )
    dates = pd.DatetimeIndex(pd.to_datetime(frame["trading_date"]).sort_values().unique())
    outer_start = dates[30]
    selection = frame.loc[
        (pd.to_datetime(frame["trading_date"]) < outer_start)
        & (pd.to_datetime(frame["label_end"]) < outer_start)
    ].copy()

    original = advanced_module._run_ablations(
        frame,
        selection_frame=selection,
        target_column="target",
        label_end_column="label_end",
        horizon=1,
        feature_names=feature_names,
        parameters={},
        config=config,
        validation_not_before=outer_start,
    )
    mutated = frame.copy()
    outer_rows = pd.to_datetime(mutated["trading_date"]) >= outer_start
    mutated.loc[outer_rows, "target"] *= -100.0
    rerun = advanced_module._run_ablations(
        mutated,
        selection_frame=selection,
        target_column="target",
        label_end_column="label_end",
        horizon=1,
        feature_names=feature_names,
        parameters={},
        config=config,
        validation_not_before=outer_start,
    )

    assert original[0].family_id == "F0"
    assert original[0].added_features == tuple(
        name for name in V0_MANIFEST.feature_names if name in feature_names
    )
    assert [item.selection_rank_ic for item in original] == [
        item.selection_rank_ic for item in rerun
    ]
    assert [item.selected_on_tuning_period for item in original] == [
        item.selected_on_tuning_period for item in rerun
    ]
    assert any(
        before.mean_daily_rank_ic != after.mean_daily_rank_ic
        for before, after in zip(original, rerun, strict=True)
    )


def test_decision_oof_uncertainty_uses_strictly_prior_targets() -> None:
    original = _decision_oof_fixture()
    before = build_decision_compatible_oof_predictions(
        original,
        model_version="model-v1",
        feature_version="feature-v2",
        data_snapshot_id="snapshot-v1",
    )
    mutated = original.copy()
    final_date = pd.to_datetime(mutated["trading_date"]).max()
    later_regression = mutated["task"].eq("regression") & pd.to_datetime(
        mutated["trading_date"]
    ).eq(final_date)
    mutated.loc[later_regression, "target"] = 999.0
    after = build_decision_compatible_oof_predictions(
        mutated,
        model_version="model-v1",
        feature_version="feature-v2",
        data_snapshot_id="snapshot-v1",
    )

    assert before == after
    not_mature = original.copy()
    horizon_20_dates = pd.DatetimeIndex(
        pd.to_datetime(not_mature.loc[not_mature["horizon"].eq(20), "trading_date"])
        .sort_values()
        .unique()
    )
    not_yet_available = (
        not_mature["horizon"].eq(20)
        & not_mature["task"].eq("regression")
        & pd.to_datetime(not_mature["trading_date"]).eq(horizon_20_dates[10])
    )
    not_mature.loc[not_yet_available, "target"] = -888.0
    maturity_safe = build_decision_compatible_oof_predictions(
        not_mature,
        model_version="model-v1",
        feature_version="feature-v2",
        data_snapshot_id="snapshot-v1",
    )
    assert before == maturity_safe
    with pytest.raises(ValueError, match="absolute returns"):
        build_decision_compatible_oof_predictions(
            original,
            model_version="model-v1",
            feature_version="feature-v2",
            data_snapshot_id="snapshot-v1",
            target_family="topix_excess",
        )

    prediction = before.predictions[0]
    bucket = AccountBucket(
        bucket_id="bucket-1",
        account_id="account-1",
        account_type=AccountType.TAXABLE_SPECIFIED,
        withholding_mode=WithholdingMode.WITHHOLDING,
        fee_policy_id="cost-v1",
        tax_policy_id="tax-v1",
    )
    portfolio = PortfolioState(
        portfolio_id="portfolio-1",
        as_of=prediction.as_of,
        accounts=(Account(account_id="account-1", broker="paper", display_name="Paper"),),
        account_buckets=(bucket,),
        positions=(),
        cash=(CashState(account_bucket_id=bucket.bucket_id, available_cash=Decimal("1000000")),),
        tax_states=(
            TaxState(account_bucket_id=bucket.bucket_id, tax_year=prediction.as_of.year),
        ),
    )
    candidate = DecisionCandidate(
        security=Security(
            symbol=prediction.symbol,
            company_name=prediction.symbol,
            sector="Test",
        ),
        account_bucket_id=bucket.bucket_id,
        price=Decimal("1000"),
        average_daily_trading_value=Decimal("100000000"),
        prediction=prediction,
    )
    engine = DailyPortfolioDecisionEngine(
        config=DecisionEngineConfig(
            maximum_positions=1,
            maximum_symbol_weight=Decimal("1"),
            maximum_sector_weight=Decimal("1"),
            minimum_cash_ratio=Decimal("0"),
            minimum_improvement_yen=Decimal("0"),
        ),
        cost_engine=TransactionCostEngine(
            CostPolicy(
                policy_id="cost-v1",
                version="1",
                zero_commission_confirmed=True,
            )
        ),
        tax_engine=SimpleJapanTaxEngine(
            TaxPolicy(policy_id="tax-v1", version="1", effective_from=date(2020, 1, 1))
        ),
    )
    proposal = engine.propose(
        portfolio=portfolio,
        candidates=(candidate,),
        generated_at=prediction.as_of,
        model_bundle_version=prediction.model_version,
    )
    assert proposal.is_order_instruction is False


def test_decision_oof_requires_complete_aligned_horizons() -> None:
    incomplete = _decision_oof_fixture().loc[lambda frame: frame["horizon"] != 20]
    with pytest.raises(ValueError, match="complete 1d/5d/20d"):
        build_decision_compatible_oof_predictions(
            incomplete,
            model_version="model-v1",
            feature_version="feature-v2",
            data_snapshot_id="snapshot-v1",
        )


def test_advanced_research_fails_closed_before_unbounded_materialization() -> None:
    config = _config(model_families=("lightgbm", "xgboost")).model_copy(
        update={"max_materialized_oof_rows": 1_000}
    )
    with pytest.raises(ValueError, match="BLOCKED_BY_RESOURCE_CAPABILITY"):
        run_advanced_research(
            _production_dataset(),
            data_snapshot_id="a" * 64,
            created_at=datetime(2026, 8, 24, 12, 0, tzinfo=UTC),
            code_commit="goal3-test",
            config=config,
            feature_snapshot_id="b" * 64,
            feature_manifest_hash=V2_EXTENDED_MANIFEST.manifest_hash,
            feature_names=_research_features(),
        )


def test_stacking_and_uncertainty_use_purged_disjoint_meta_periods() -> None:
    original = _stacking_oof_fixture()
    ensemble = fit_oof_ensemble(original, horizon=5)
    calibration = calibrate_oof_uncertainty(original, horizon=5, ensemble=ensemble)
    dates = pd.DatetimeIndex(pd.to_datetime(original["trading_date"]).sort_values().unique())

    calibration_mutated = original.copy()
    calibration_rows = pd.to_datetime(calibration_mutated["trading_date"]).isin(dates[15:18])
    calibration_mutated.loc[calibration_rows, "target"] *= -100
    calibration_ensemble = fit_oof_ensemble(calibration_mutated, horizon=5)
    assert calibration_ensemble.weights == pytest.approx(ensemble.weights)

    evaluation_mutated = original.copy()
    evaluation_rows = pd.to_datetime(evaluation_mutated["trading_date"]).isin(dates[23:])
    evaluation_mutated.loc[evaluation_rows, "target"] *= -100
    evaluation_ensemble = fit_oof_ensemble(evaluation_mutated, horizon=5)
    evaluation_calibration = calibrate_oof_uncertainty(
        evaluation_mutated, horizon=5, ensemble=evaluation_ensemble
    )
    assert evaluation_ensemble.weights == pytest.approx(ensemble.weights)
    assert evaluation_calibration.residual_quantile_80 == pytest.approx(
        calibration.residual_quantile_80
    )
    assert evaluation_calibration.residual_quantile_90 == pytest.approx(
        calibration.residual_quantile_90
    )


def _config(*, model_families: tuple[str, ...]) -> AdvancedResearchConfig:
    return AdvancedResearchConfig.model_validate(
        {
            "horizons": (1,),
            "model_families": model_families,
            "seeds": (17,),
            "initial_train_periods": 20,
            "validation_periods": 5,
            "step_periods": 5,
            "holdout_periods": 5,
            "estimator_count": 5,
            "tuning_trials": 1,
            "tuning_timeout_seconds": 10,
            "large_loss_threshold": -0.02,
            "run_ablations": False,
        }
    )


def _small_model_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-01", periods=40)
    rows: list[dict[str, object]] = []
    for date_number, trading_date in enumerate(dates):
        for symbol_number in range(6):
            rows.append(
                {
                    "symbol": f"S{symbol_number}",
                    "trading_date": trading_date,
                    "f1": float(rng.normal() + symbol_number / 5),
                    "f2": float(rng.normal() + date_number / 100),
                    "target": float((symbol_number - 2.5) / 50 + rng.normal(0.0, 0.01)),
                    "label_end": dates[min(date_number + 1, len(dates) - 1)],
                }
            )
    return pd.DataFrame(rows)


def _research_features() -> tuple[str, ...]:
    return (
        "price.return_1d",
        "price.return_5d",
        "risk.realized_vol_20d",
        "volume.ratio_20d",
    )


def _production_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-01", periods=45)
    rows: list[dict[str, object]] = []
    for date_number, trading_date in enumerate(dates):
        for symbol_number in range(6):
            row: dict[str, object] = {
                "symbol": f"P{symbol_number}",
                "trading_date": trading_date,
                "historical_revision_policy": "SINGLE_VINTAGE_AS_REVISED",
                "historical_revision_status": "PARTIAL",
                "as_of": trading_date.tz_localize("Asia/Tokyo").replace(hour=11, minute=30),
                "target_return_1d": float((symbol_number - 2.5) / 45 + rng.normal(0.0, 0.012)),
                "label_end_date_1d": dates[min(date_number + 1, len(dates) - 1)],
                "label_status_1d": (
                    "AVAILABLE" if date_number < len(dates) - 1 else "HORIZON_NOT_MATURE"
                ),
            }
            for feature_number, name in enumerate(_research_features()):
                row[name] = float(
                    rng.normal() + symbol_number / (feature_number + 2) + date_number / 100
                )
            rows.append(row)
    frame = pd.DataFrame(rows)
    assert set(_research_features()) <= set(V1_CORE_MANIFEST.feature_names)
    return frame


def _decision_oof_fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.bdate_range("2025-01-06", periods=25)
    for horizon in (1, 5, 20):
        for date_number, trading_date in enumerate(dates):
            as_of = trading_date.tz_localize("Asia/Tokyo").replace(hour=11, minute=30)
            for symbol_number, symbol in enumerate(("A", "B")):
                target = float(date_number / 100 + symbol_number / 50)
                for component in range(2):
                    rows.append(
                        {
                            "symbol": symbol,
                            "trading_date": trading_date,
                            "as_of": as_of,
                            "horizon": horizon,
                            "task": "regression",
                            "target": target,
                            "label_end": trading_date + pd.offsets.BDay(horizon),
                            "prediction": target + (component - 0.5) / 100,
                        }
                    )
                rows.extend(
                    [
                        {
                            "symbol": symbol,
                            "trading_date": trading_date,
                            "as_of": as_of,
                            "horizon": horizon,
                            "task": "quantile",
                            "target": target,
                            "label_end": trading_date + pd.offsets.BDay(horizon),
                            "prediction": target - 0.03,
                        },
                        {
                            "symbol": symbol,
                            "trading_date": trading_date,
                            "as_of": as_of,
                            "horizon": horizon,
                            "task": "large_loss",
                            "target": target,
                            "label_end": trading_date + pd.offsets.BDay(horizon),
                            "prediction": 0.10 + symbol_number / 10,
                        },
                    ]
                )
    return pd.DataFrame(rows)


def _stacking_oof_fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.bdate_range("2024-01-02", periods=30)
    for date_number, trading_date in enumerate(dates):
        for symbol_number in range(6):
            target = float(symbol_number + ((date_number % 3) - 1) * symbol_number / 10)
            for task_number, task in enumerate(("regression", "ranking")):
                rows.append(
                    {
                        "symbol": f"S{symbol_number}",
                        "trading_date": trading_date,
                        "label_end": trading_date + pd.offsets.BDay(5),
                        "horizon": 5,
                        "model_family": "lightgbm",
                        "task": task,
                        "seed": 17,
                        "fold": date_number // 5,
                        "target": target,
                        "prediction": float(
                            symbol_number * (1 if task_number == 0 else 0.8)
                            + ((date_number + task_number) % 4) / 10
                        ),
                    }
                )
    return pd.DataFrame(rows)
