from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from typer.testing import CliRunner

import stock_ai.ml.morning as morning_ml
from stock_ai.cli import app
from stock_ai.data import (
    CapabilityStatus,
    MorningCapabilityReport,
    MorningDataError,
    MorningFreezeMetadata,
    assert_morning_freeze_coverage,
    build_morning_universe,
    morning_capabilities,
    morning_capability_rows,
    validate_morning_bars,
)
from stock_ai.decision import DecisionCandidate
from stock_ai.domain import (
    MorningDecisionAudit,
    Position,
    Prediction,
    PredictionUncertainty,
    ProposalAction,
    Security,
)
from stock_ai.features import MORNING_CORE_MANIFEST, build_morning_features
from stock_ai.fixtures import morning_research_fixture
from stock_ai.ml import (
    MorningDatasetSnapshot,
    MorningModelDisposition,
    MorningModelResult,
    MorningResearchConfig,
    MorningResearchReport,
    build_morning_decision_predictions,
    build_morning_supervised_dataset,
    fit_morning_research_bundle,
    infer_current_morning_predictions,
    load_morning_dataset_snapshot,
    load_morning_research_run,
    propose_from_morning_batch,
    run_morning_research,
    write_morning_dataset_snapshot,
    write_morning_research_run,
)
from stock_ai.ml.morning import _frame_hash, _report_identity, _stable_hash

from .conftest import AS_OF, decision_engine, portfolio

JST = ZoneInfo("Asia/Tokyo")
CUTOFFS = (time(9, 0), time(9, 5), time(9, 15), time(9, 30), time(10), time(11), time(11, 30))


@pytest.mark.parametrize(
    ("update", "message"),
    (
        ({"horizons": ()}, "unique subset"),
        ({"model_families": ()}, "non-empty and unique"),
        ({"seeds": ()}, "seeds must be non-empty"),
        ({"validation_periods": 5, "step_periods": 4}, "cannot overlap"),
        ({"enable_neural_challenger": True}, "MLP research requires"),
        ({"mlp_hidden_units": ()}, "hidden units"),
    ),
)
def test_morning_research_config_rejects_unsafe_contracts(
    update: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        MorningResearchConfig.model_validate(update)


def test_capability_and_universe_contracts_fail_closed() -> None:
    blocked = morning_capabilities(provider=None, available_fields=())
    assert set(blocked.capabilities.values()) == {CapabilityStatus.BLOCKED_BY_DATA_CAPABILITY}
    with pytest.raises(MorningDataError, match="BLOCKED_BY_DATA_CAPABILITY"):
        blocked.require("intraday_bars")

    available = _core_capability_report()
    assert available.capabilities["intraday_bars"] is CapabilityStatus.AVAILABLE
    assert available.capabilities["quotes"] is CapabilityStatus.BLOCKED_BY_DATA_CAPABILITY
    universe = build_morning_universe(current_holdings=("A", "B"), candidates=("B", "C"))
    assert [member.symbol for member in universe] == ["A", "B", "C"]
    assert universe[0].role.value == "HOLDING"
    assert universe[1].role.value == "HOLDING_AND_CANDIDATE"
    assert morning_capability_rows(available)[0].name == "morning_ohlc"
    assert available.model_dump(mode="json")["capabilities"]["intraday_bars"] == "AVAILABLE"
    available.require("morning_ohlc", "intraday_bars")
    with pytest.raises(MorningDataError, match="cannot be empty"):
        build_morning_universe(current_holdings=(), candidates=())

    freeze = MorningFreezeMetadata(
        as_of=AS_OF,
        provider="deterministic-fixture",
        source_snapshot_ids=("source-1",),
        source_record_ids=tuple(f"source-{member.symbol}" for member in universe),
        universe=universe,
        capability_report=available,
    )
    assert freeze.is_order_instruction is False
    freeze_bars = pd.DataFrame(
        [
            _bar_row(member.symbol, AS_OF, 100.0, 10.0, f"source-{member.symbol}")
            for member in universe
        ]
    )
    assert len(assert_morning_freeze_coverage(freeze_bars, freeze)) == len(universe)
    with pytest.raises(MorningDataError, match="universe mismatch"):
        assert_morning_freeze_coverage(freeze_bars.iloc[:-1], freeze)
    wrong_provider = freeze_bars.copy()
    wrong_provider["provider"] = "other-provider"
    with pytest.raises(MorningDataError, match="provider does not match"):
        assert_morning_freeze_coverage(wrong_provider, freeze)
    with pytest.raises(ValueError, match="exactly 11:30"):
        freeze.model_copy(update={"as_of": AS_OF - timedelta(minutes=1)}).model_validate(
            {
                **freeze.model_dump(),
                "as_of": AS_OF - timedelta(minutes=1),
            }
        )
    for override, match in (
        ({"source_snapshot_ids": ()}, "requires source"),
        ({"source_snapshot_ids": ("x", "x")}, "must be unique"),
        ({"is_order_instruction": True}, "never be order"),
        ({"universe": ()}, "must be non-empty"),
    ):
        with pytest.raises(ValueError, match=match):
            MorningFreezeMetadata.model_validate({**freeze.model_dump(), **override})


def test_morning_bar_quality_matrix_is_fail_closed() -> None:
    timestamp = datetime(2025, 1, 6, 9, tzinfo=JST)
    valid = pd.DataFrame([_bar_row("A", timestamp, 100.0, 10.0, "source")])
    with pytest.raises(MorningDataError, match="missing columns"):
        validate_morning_bars(valid.drop(columns="price"))
    with pytest.raises(MorningDataError, match="cannot be empty"):
        validate_morning_bars(valid.iloc[0:0])
    with pytest.raises(MorningDataError, match="unique"):
        validate_morning_bars(pd.concat([valid, valid], ignore_index=True))
    cases = (
            ("symbol", "", "symbol cannot be blank"),
        ("provider", "", "provider cannot be blank"),
            ("source_record_id", "", "source_record_id cannot be blank"),
        ("price", "not-numeric", "finite numeric"),
        ("price", 0.0, "prices must be positive"),
        ("volume", -1.0, "cannot be negative"),
    )
    for column, value, match in cases:
        malformed = valid.copy()
        malformed[column] = value
        with pytest.raises(MorningDataError, match=match):
            validate_morning_bars(malformed)
    naive = valid.copy()
    naive["timestamp"] = timestamp.replace(tzinfo=None)
    with pytest.raises(MorningDataError, match="timezone-aware"):
        validate_morning_bars(naive)
    quotes = valid.assign(bid=101.0, ask=100.0, spread=1.0)
    with pytest.raises(MorningDataError, match="ask cannot be below bid"):
        validate_morning_bars(quotes)
    negative_bid = valid.assign(bid=-1.0)
    with pytest.raises(MorningDataError, match="bid cannot be negative"):
        validate_morning_bars(negative_bid)


def test_morning_cli_is_explicitly_blocked_or_fixture_only(tmp_path: Path) -> None:
    runner = CliRunner()
    capability = runner.invoke(app, ["research", "morning-capabilities"])
    assert capability.exit_code == 0, capability.output
    assert "intraday_bars BLOCKED_BY_DATA_CAPABILITY" in capability.output
    fixture = runner.invoke(
        app,
        ["research", "morning-fixture", "--output-dir", str(tmp_path)],
    )
    assert fixture.exit_code == 0, fixture.output
    assert "research_only=true" in fixture.output
    assert "current_inference=4" in fixture.output
    assert "order_instruction=false" in fixture.output
    assert "live_provider_used=false" in fixture.output


def test_morning_bar_validation_rejects_post_freeze_and_late_receipt() -> None:
    bars, _, _, _, _ = _morning_inputs(periods=2, symbols=("A",))
    post_freeze = bars.iloc[[0]].copy()
    post_freeze["timestamp"] = datetime(2025, 1, 6, 11, 31, tzinfo=JST)
    with pytest.raises(MorningDataError, match="09:00 through 11:30"):
        validate_morning_bars(post_freeze)

    late = bars.iloc[[0]].copy()
    late["available_at"] = datetime(2025, 1, 6, 11, 31, tzinfo=JST)
    with pytest.raises(MorningDataError, match="not available"):
        validate_morning_bars(late)


def test_feature_builder_binds_freeze_provider_universe_and_profile_capability() -> None:
    bars, context, market, sectors, freezes = _morning_inputs(periods=2, symbols=("A", "B"))
    wrong_provider = morning_capabilities(
        provider="claimed-other-provider",
        available_fields=(
            "timestamp",
            "price",
            "volume",
            "trading_value",
            "historical_same_time_sessions",
        ),
    )
    with pytest.raises(MorningDataError, match="provider lineage"):
        build_morning_features(
            bars,
            daily_context=context,
            market_bars=market,
            sector_bars=sectors,
            capability_report=wrong_provider,
            freeze_metadata=freezes,
        )

    with pytest.raises(MorningDataError, match="universe mismatch"):
        build_morning_features(
            bars.loc[bars["symbol"].eq("A")],
            daily_context=context.loc[context["symbol"].eq("A")],
            market_bars=market,
            sector_bars=sectors,
            capability_report=_core_capability_report(),
            freeze_metadata=freezes,
        )

    no_profile = morning_capabilities(
        provider="deterministic-fixture",
        available_fields=("timestamp", "price", "volume", "trading_value"),
    )
    with pytest.raises(MorningDataError, match="intraday_volume_profile"):
        build_morning_features(
            bars,
            daily_context=context,
            market_bars=market,
            sector_bars=sectors,
            capability_report=no_profile,
            freeze_metadata=freezes,
        )

    swapped = context.copy()
    swapped[["is_current_holding", "is_candidate"]] = swapped[
        ["is_candidate", "is_current_holding"]
    ].to_numpy()
    with pytest.raises(MorningDataError, match="context roles"):
        build_morning_features(
            bars,
            daily_context=swapped,
            market_bars=market,
            sector_bars=sectors,
            capability_report=_core_capability_report(),
            freeze_metadata=freezes,
        )

    neither = context.copy()
    neither.loc[neither["symbol"].eq("B"), ["is_current_holding", "is_candidate"]] = False
    with pytest.raises(MorningDataError, match="holding or candidate"):
        build_morning_features(
            bars,
            daily_context=neither,
            market_bars=market,
            sector_bars=sectors,
            capability_report=_core_capability_report(),
            freeze_metadata=freezes,
        )

    blocked_report = morning_capabilities(provider="deterministic-fixture", available_fields=())
    mismatched_freezes = tuple(
        item.model_copy(update={"capability_report": blocked_report}) for item in freezes
    )
    with pytest.raises(MorningDataError, match="capability report"):
        build_morning_features(
            bars,
            daily_context=context,
            market_bars=market,
            sector_bars=sectors,
            capability_report=_core_capability_report(),
            freeze_metadata=mismatched_freezes,
        )


def test_morning_core_formulas_profiles_and_future_mutation_are_exact() -> None:
    bars, context, market, sectors, freezes = _morning_inputs(periods=30, symbols=("A", "B"))
    original = build_morning_features(
        bars,
        daily_context=context,
        market_bars=market,
        sector_bars=sectors,
        capability_report=_core_capability_report(),
        freeze_metadata=freezes,
    )
    assert set(MORNING_CORE_MANIFEST.feature_names) <= set(original.frame.columns)
    selected = original.frame.loc[
        (original.frame["symbol"] == "A")
        & (original.frame["trading_date"] == original.frame["trading_date"].unique()[25])
    ].iloc[0]
    day_number = 25
    signal = 0.002 + day_number * 0.0001
    selected_date = pd.Timestamp(selected["trading_date"])
    session = bars.loc[
        bars["symbol"].eq("A")
        & bars["timestamp"].map(
            lambda value: pd.Timestamp(value).date() == selected_date.date()
        )
    ].sort_values("timestamp")
    prices = session["price"].to_numpy(dtype=float)
    expected_vwap = float(session["trading_value"].sum() / session["volume"].sum())
    assert selected["morning.return_0900_1130"] == pytest.approx(signal)
    assert selected["morning.topix_relative_1130"] == pytest.approx(signal - signal * 0.2)
    assert selected["morning.sector_relative_1130"] == pytest.approx(signal - signal * 0.4)
    assert selected["morning.close_location"] == pytest.approx(1.0)
    assert selected["morning.high"] == pytest.approx(prices[-1])
    assert selected["morning.low"] == pytest.approx(prices[0])
    assert selected["morning.range_pct_open"] == pytest.approx(signal)
    assert selected["morning.realized_volatility"] == pytest.approx(
        np.sqrt(np.square(np.diff(np.log(prices))).sum())
    )
    assert selected["morning.vwap"] == pytest.approx(expected_vwap)
    assert selected["morning.price_to_vwap"] == pytest.approx(prices[-1] / expected_vwap - 1)
    assert selected["morning.drop_from_high"] == pytest.approx(0.0)
    assert selected["morning.rebound_from_low"] == pytest.approx(signal)
    assert selected["morning.cumulative_volume_0930"] == pytest.approx(
        session["volume"].iloc[:4].sum()
    )
    assert selected["morning.volume_progress_1130_20d"] == pytest.approx(
        (1.0 + day_number * 0.01) / np.mean([1.0 + value * 0.01 for value in range(5, 25)])
    )
    assert selected["morning.monitored_volume_rank_pct"] == pytest.approx(0.5)
    assert np.isnan(selected["morning.candidate_volume_rank_pct"])
    selected_b = original.frame.loc[
        original.frame["symbol"].eq("B")
        & original.frame["trading_date"].eq(selected_date)
    ].iloc[0]
    assert selected_b["morning.monitored_volume_rank_pct"] == pytest.approx(1.0)
    assert selected_b["morning.candidate_volume_rank_pct"] == pytest.approx(1.0)
    assert selected_b["morning.prior_expected_return_5d"] == pytest.approx(0.002)

    cutoff = original.frame["trading_date"].unique()[20]
    future_bars = bars.copy()
    future_mask = future_bars["timestamp"].map(
        lambda value: pd.Timestamp(value).date() > pd.Timestamp(cutoff).date()
    )
    future_bars.loc[future_mask, "price"] *= 10.0
    mutated = build_morning_features(
        future_bars,
        daily_context=context,
        market_bars=market,
        sector_bars=sectors,
        capability_report=_core_capability_report(),
        freeze_metadata=freezes,
    )
    columns = ["symbol", "trading_date", *MORNING_CORE_MANIFEST.feature_names]
    pdt.assert_frame_equal(
        original.frame.loc[original.frame["trading_date"] <= cutoff, columns].reset_index(
            drop=True
        ),
        mutated.frame.loc[mutated.frame["trading_date"] <= cutoff, columns].reset_index(drop=True),
    )


def test_microstructure_features_require_declared_fields_and_are_exact() -> None:
    bars, context, market, sectors, freezes = _morning_inputs(periods=3, symbols=("A",))
    for frame in (bars, market, sectors):
        frame["bid"] = frame["price"] - 0.05
        frame["ask"] = frame["price"] + 0.05
        frame["spread"] = 0.10
        frame["quote_state"] = "NORMAL"
        frame["bid_size"] = 300.0
        frame["ask_size"] = 100.0
        frame["trade_count"] = 2.0
        frame["seconds_since_last_trade"] = 5.0
    report = morning_capabilities(
        provider="deterministic-fixture",
        available_fields=(
            "timestamp",
            "price",
            "volume",
            "trading_value",
            "historical_same_time_sessions",
            "bid",
            "ask",
            "spread",
            "quote_state",
            "bid_size",
            "ask_size",
            "trade_count",
            "seconds_since_last_trade",
        ),
    )
    freezes = tuple(item.model_copy(update={"capability_report": report}) for item in freezes)
    output = build_morning_features(
        bars,
        daily_context=context,
        market_bars=market,
        sector_bars=sectors,
        capability_report=report,
        freeze_metadata=freezes,
    )
    assert len(output.available_microstructure_features) == 5
    row = output.frame.iloc[-1]
    assert row["morning.micro_spread_bps"] == pytest.approx(
        0.10 / float(bars.iloc[-1]["price"]) * 10_000
    )
    assert row["morning.micro_price_to_midpoint"] == pytest.approx(0.0)
    assert row["morning.micro_order_book_imbalance"] == pytest.approx(0.5)
    assert row["morning.micro_trade_frequency_per_minute"] == pytest.approx(14 / 150)
    assert row["morning.micro_no_trade_seconds"] == 5.0
    missing = bars.drop(columns=["bid"])
    with pytest.raises(MorningDataError, match="declared morning microstructure"):
        build_morning_features(
            missing,
            daily_context=context,
            market_bars=market,
            sector_bars=sectors,
            capability_report=report,
            freeze_metadata=freezes,
        )
    null_trade_count = bars.copy()
    null_trade_count["trade_count"] = np.nan
    with pytest.raises(MorningDataError, match="values are missing"):
        build_morning_features(
            null_trade_count,
            daily_context=context,
            market_bars=market,
            sector_bars=sectors,
            capability_report=report,
            freeze_metadata=freezes,
        )
    assert output.manifest.feature_names == (
        *MORNING_CORE_MANIFEST.feature_names,
        *output.available_microstructure_features,
    )
    assert output.manifest.feature_set_version.startswith("morning-microstructure-v1-")


def test_morning_vwap_zero_volume_is_missing_not_synthetic() -> None:
    bars, context, market, sectors, freezes = _morning_inputs(periods=2, symbols=("A",))
    bars["volume"] = 0.0
    bars["trading_value"] = 0.0
    output = build_morning_features(
        bars,
        daily_context=context,
        market_bars=market,
        sector_bars=sectors,
        capability_report=_core_capability_report(),
        freeze_metadata=freezes,
    )
    assert output.frame["morning.vwap"].isna().all()
    assert output.frame["morning.price_to_vwap"].isna().all()


def test_morning_dataset_requires_post_freeze_labels_and_blanks_unmatured() -> None:
    features, labels = _feature_and_label_fixture(periods=35, symbols=("A", "B"))
    calendar = _label_calendar(labels)
    bad = labels.copy()
    bad["label_entry_at_1d"] = bad["trading_date"].map(
        lambda value: datetime.combine(pd.Timestamp(value).date(), time(11, 30), tzinfo=JST)
    )
    with pytest.raises(ValueError, match="strictly after"):
        build_morning_supervised_dataset(
            features,
            bad,
            publication_as_of=datetime(2026, 1, 1, tzinfo=JST),
            trading_calendar=calendar,
        )
    impossible = labels.copy()
    impossible["label_available_at_20d"] = impossible["label_entry_at_20d"]
    with pytest.raises(ValueError, match="before their label endpoint"):
        build_morning_supervised_dataset(
            features,
            impossible,
            publication_as_of=datetime(2027, 1, 1, tzinfo=JST),
            trading_calendar=calendar,
        )
    before_endpoint = labels.copy()
    before_endpoint["label_available_at_1d"] = pd.to_datetime(
        before_endpoint["label_end_date_1d"]
    ).map(lambda value: datetime.combine(value.date(), time(0), tzinfo=JST))
    with pytest.raises(ValueError, match="before their label endpoint"):
        build_morning_supervised_dataset(
            features,
            before_endpoint,
            publication_as_of=datetime(2027, 1, 1, tzinfo=JST),
            trading_calendar=calendar,
        )

    cutoff = datetime.combine(
        pd.Timestamp(labels["trading_date"].iloc[20]).date(), time(16), tzinfo=JST
    )
    dataset = build_morning_supervised_dataset(
        features, labels, publication_as_of=cutoff, trading_calendar=calendar
    )
    immature = dataset["label_available_at_20d"].map(pd.Timestamp) > pd.Timestamp(cutoff)
    assert dataset.loc[immature, "target_return_20d"].isna().all()
    assert dataset.loc[immature, "label_status_20d"].eq("HORIZON_NOT_MATURE").all()


def test_snapshot_rejects_post_freeze_sources_and_persists_partial_f14(
    tmp_path: Path,
) -> None:
    features, labels = _feature_and_label_fixture(periods=35, symbols=("A", "B"))
    calendar = _label_calendar(labels)
    dataset = build_morning_supervised_dataset(
        features,
        labels,
        publication_as_of=datetime(2027, 1, 1, tzinfo=JST),
        trading_calendar=calendar,
    )
    post_freeze = dataset.copy()
    post_freeze.loc[post_freeze.index[0], "available_at"] = pd.Timestamp(
        post_freeze.loc[post_freeze.index[0], "as_of"]
    ).to_pydatetime() + timedelta(minutes=1)
    with pytest.raises(ValueError, match="received after"):
        write_morning_dataset_snapshot(
            post_freeze,
            tmp_path / "late",
            created_at=datetime(2026, 8, 24, 12, tzinfo=JST),
            publication_as_of=datetime(2027, 1, 1, tzinfo=JST),
            capability_report=_core_capability_report(),
            manifest=MORNING_CORE_MANIFEST,
            trading_calendar=calendar,
        )
    pre_endpoint = dataset.copy()
    first = pre_endpoint.index[0]
    end_date = pd.Timestamp(pre_endpoint.loc[first, "label_end_date_1d"]).date()
    pre_endpoint.loc[first, "label_available_at_1d"] = datetime.combine(
        end_date, time(0), tzinfo=JST
    )
    with pytest.raises(ValueError, match="before their endpoint"):
        write_morning_dataset_snapshot(
            pre_endpoint,
            tmp_path / "pre-endpoint",
            created_at=datetime(2026, 8, 24, 12, tzinfo=JST),
            publication_as_of=datetime(2027, 1, 1, tzinfo=JST),
            capability_report=_core_capability_report(),
            manifest=MORNING_CORE_MANIFEST,
            trading_calendar=calendar,
        )

    bars, context, market, sectors, freezes = _morning_inputs(periods=35, symbols=("A", "B"))
    for frame in (bars, market, sectors):
        frame["bid"] = frame["price"] - 0.05
        frame["ask"] = frame["price"] + 0.05
        frame["spread"] = 0.10
        frame["quote_state"] = "NORMAL"
    quote_report = morning_capabilities(
        provider="deterministic-fixture",
        available_fields=(
            "timestamp",
            "price",
            "volume",
            "trading_value",
            "historical_same_time_sessions",
            "bid",
            "ask",
            "spread",
            "quote_state",
        ),
    )
    with pytest.raises(ValueError, match="exactly match provider capabilities"):
        write_morning_dataset_snapshot(
            dataset,
            tmp_path / "capability-mismatch",
            created_at=datetime(2026, 8, 24, 12, tzinfo=JST),
            publication_as_of=datetime(2027, 1, 1, tzinfo=JST),
            capability_report=quote_report,
            manifest=MORNING_CORE_MANIFEST,
            trading_calendar=calendar,
        )
    quote_freezes = tuple(
        item.model_copy(update={"capability_report": quote_report}) for item in freezes
    )
    output = build_morning_features(
        bars,
        daily_context=context,
        market_bars=market,
        sector_bars=sectors,
        capability_report=quote_report,
        freeze_metadata=quote_freezes,
    )
    quote_dataset = build_morning_supervised_dataset(
        output.frame,
        labels,
        publication_as_of=datetime(2027, 1, 1, tzinfo=JST),
        trading_calendar=calendar,
    )
    snapshot = write_morning_dataset_snapshot(
        quote_dataset,
        tmp_path / "partial-f14",
        created_at=datetime(2026, 8, 24, 12, tzinfo=JST),
        publication_as_of=datetime(2027, 1, 1, tzinfo=JST),
        capability_report=quote_report,
        manifest=output.manifest,
        trading_calendar=calendar,
    )
    observed, observed_frame = load_morning_dataset_snapshot(snapshot.parquet_path)
    assert observed.feature_names == output.manifest.feature_names
    assert set(output.available_microstructure_features) <= set(observed_frame.columns)

    inconsistent = dataset.copy()
    inconsistent.loc[inconsistent.index[0], "revision_target_5d"] += 1.0
    with pytest.raises(ValueError, match="revision targets"):
        write_morning_dataset_snapshot(
            inconsistent,
            tmp_path / "bad-revision",
            created_at=datetime(2026, 8, 24, 12, tzinfo=JST),
            publication_as_of=datetime(2027, 1, 1, tzinfo=JST),
            capability_report=_core_capability_report(),
            manifest=MORNING_CORE_MANIFEST,
            trading_calendar=calendar,
        )


def test_morning_labels_use_an_explicit_jpx_session_calendar() -> None:
    features, labels = _feature_and_label_fixture(periods=35, symbols=("A",))
    friday = pd.Timestamp("2025-01-10")
    selected_features = features.loc[pd.to_datetime(features["trading_date"]).eq(friday)]
    selected_labels = labels.loc[pd.to_datetime(labels["trading_date"]).eq(friday)]
    jpx_calendar = pd.DatetimeIndex(
        value
        for value in pd.bdate_range(friday, periods=30)
        if value != pd.Timestamp("2025-01-13")
    )
    with pytest.raises(ValueError, match="fixed JPX session calendar"):
        build_morning_supervised_dataset(
            selected_features,
            selected_labels,
            publication_as_of=datetime(2027, 1, 1, tzinfo=JST),
            trading_calendar=jpx_calendar,
        )


def test_morning_research_compares_no_update_gbdt_and_disabled_neural(
    tmp_path: Path,
) -> None:
    features, labels = _feature_and_label_fixture(periods=75, symbols=("A", "B", "C", "D"))
    calendar = _label_calendar(labels)
    dataset = build_morning_supervised_dataset(
        features,
        labels,
        publication_as_of=datetime(2027, 1, 1, tzinfo=JST),
        trading_calendar=calendar,
    )
    config = MorningResearchConfig(
        horizons=(1,),
        model_families=("ridge", "lightgbm", "mlp"),
        seeds=(17,),
        initial_train_periods=30,
        validation_periods=10,
        step_periods=10,
        holdout_periods=10,
        lightgbm_estimators=10,
        enable_neural_challenger=False,
        max_model_fits=50,
    )
    snapshot = write_morning_dataset_snapshot(
        dataset,
        tmp_path / "datasets",
        created_at=datetime(2026, 8, 24, 12, tzinfo=JST),
        publication_as_of=datetime(2027, 1, 1, tzinfo=JST),
        capability_report=_core_capability_report(),
        manifest=MORNING_CORE_MANIFEST,
        trading_calendar=calendar,
    )
    authenticated, authenticated_dataset = load_morning_dataset_snapshot(snapshot.parquet_path)
    assert authenticated.snapshot_id == snapshot.snapshot_id
    pdt.assert_frame_equal(authenticated_dataset, dataset, check_dtype=False)
    snapshot_payload = authenticated.model_dump(mode="python")
    invalid_snapshot_cases = (
        ({"created_at": datetime(2026, 8, 24, 12)}, "timezone-aware"),
        ({"source_record_ids": ()}, "source record IDs"),
        ({"trading_calendar_dates": authenticated.trading_calendar_dates[:20]}, "fixed trading"),
    )
    for update, message in invalid_snapshot_cases:
        with pytest.raises(ValueError, match=message):
            MorningDatasetSnapshot.model_validate({**snapshot_payload, **update})
    with pytest.raises(ValueError, match="snapshot identity"):
        run_morning_research(
            authenticated.model_copy(update={"snapshot_id": "forged-snapshot"}),
            authenticated_dataset,
            created_at=datetime(2026, 8, 24, 12, tzinfo=JST),
            code_commit="fixture-commit",
            config=config,
        )
    forged_range = authenticated.model_copy(update={"data_end": "1900-01-01"})
    forged_range = forged_range.model_copy(
        update={
            "snapshot_id": _stable_hash(morning_ml._morning_snapshot_identity(forged_range))
        }
    )
    with pytest.raises(ValueError, match="range metadata"):
        run_morning_research(
            forged_range,
            authenticated_dataset,
            created_at=datetime(2026, 8, 24, 12, tzinfo=JST),
            code_commit="fixture-commit",
            config=config,
        )
    run = run_morning_research(
        authenticated,
        authenticated_dataset,
        created_at=datetime(2026, 8, 24, 12, tzinfo=JST),
        code_commit="fixture-commit",
        config=config,
    )
    assert {result.model_family for result in run.report.results} == {"ridge", "lightgbm"}
    assert all(
        result.holdings_rows > 0 and result.candidate_rows > 0 for result in run.report.results
    )
    assert run.report.locked_holdout_accessed is False
    assert run.report.adoption_eligible is False
    capabilities = {
        item.model_name: item.disposition.value
        for item in run.report.neural_and_sequence_capabilities
    }
    assert capabilities["small_mlp"] == "DISABLED"
    assert capabilities["tcn"] == "BLOCKED_BY_DATA_CAPABILITY"
    metadata_path, parquet_path = write_morning_research_run(run, tmp_path)
    observed, observed_oof = load_morning_research_run(parquet_path)
    assert observed.report_id == run.report.report_id
    pdt.assert_frame_equal(observed_oof, run.oof_predictions, check_dtype=False)
    assert write_morning_research_run(run, tmp_path) == (metadata_path, parquet_path)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["report"]["data_snapshot_id"] = "tampered"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="metadata hash"):
        load_morning_research_run(parquet_path)


def test_delayed_label_availability_cannot_change_earlier_oof_predictions(
    tmp_path: Path,
) -> None:
    features, labels = _feature_and_label_fixture(periods=75, symbols=("A", "B", "C", "D"))
    calendar = _label_calendar(labels)
    dataset = build_morning_supervised_dataset(
        features,
        labels,
        publication_as_of=datetime(2027, 1, 1, tzinfo=JST),
        trading_calendar=calendar,
    )
    delayed = pd.to_datetime(dataset["trading_date"]) <= pd.Timestamp("2025-01-17")
    dataset.loc[delayed, "label_available_at_1d"] = datetime(2026, 12, 31, 16, tzinfo=JST)
    mutated = dataset.copy()
    mutated.loc[delayed, "target_return_1d"] += 10.0
    mutated.loc[delayed, "revision_target_1d"] += 10.0
    config = MorningResearchConfig(
        horizons=(1,),
        model_families=("ridge",),
        seeds=(17,),
        initial_train_periods=30,
        validation_periods=10,
        step_periods=10,
        holdout_periods=10,
        max_model_fits=20,
    )
    runs = []
    for name, frame in (("base", dataset), ("mutated", mutated)):
        snapshot = write_morning_dataset_snapshot(
            frame,
            tmp_path / name,
            created_at=datetime(2026, 8, 24, 12, tzinfo=JST),
            publication_as_of=datetime(2027, 1, 1, tzinfo=JST),
            capability_report=_core_capability_report(),
            manifest=MORNING_CORE_MANIFEST,
            trading_calendar=calendar,
        )
        authenticated, authenticated_frame = load_morning_dataset_snapshot(snapshot.parquet_path)
        runs.append(
            run_morning_research(
                authenticated,
                authenticated_frame,
                created_at=datetime(2026, 8, 24, 12, tzinfo=JST),
                code_commit="fixture-commit",
                config=config,
            )
        )
    identity = ["symbol", "trading_date", "horizon", "model_family", "seed", "fold"]
    comparison = runs[0].oof_predictions.merge(
        runs[1].oof_predictions,
        on=identity,
        suffixes=("_base", "_mutated"),
        validate="one_to_one",
    )
    assert np.allclose(
        comparison["predicted_revision_base"], comparison["predicted_revision_mutated"]
    )


def test_model_fit_resource_bound_is_checked_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    features, labels = _feature_and_label_fixture(periods=75, symbols=("A", "B", "C"))
    calendar = _label_calendar(labels)
    dataset = build_morning_supervised_dataset(
        features,
        labels,
        publication_as_of=datetime(2027, 1, 1, tzinfo=JST),
        trading_calendar=calendar,
    )
    snapshot = write_morning_dataset_snapshot(
        dataset,
        tmp_path,
        created_at=datetime(2026, 8, 24, 12, tzinfo=JST),
        publication_as_of=datetime(2027, 1, 1, tzinfo=JST),
        capability_report=_core_capability_report(),
        manifest=MORNING_CORE_MANIFEST,
        trading_calendar=calendar,
    )
    authenticated, authenticated_frame = load_morning_dataset_snapshot(snapshot.parquet_path)
    called = False

    def unexpected_fit(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal called
        called = True
        raise AssertionError("fit must not start")

    monkeypatch.setattr(morning_ml, "_generate_morning_oof", unexpected_fit)
    with pytest.raises(ValueError, match="model-fit bound"):
        run_morning_research(
            authenticated,
            authenticated_frame,
            created_at=datetime(2026, 8, 24, 12, tzinfo=JST),
            code_commit="fixture-commit",
            config=MorningResearchConfig(
                horizons=(1,),
                model_families=("ridge", "lightgbm"),
                seeds=(17,),
                initial_train_periods=30,
                validation_periods=5,
                step_periods=5,
                holdout_periods=5,
                max_model_fits=10,
            ),
        )
    assert called is False


def test_small_mlp_is_an_explicit_research_only_challenger(tmp_path: Path) -> None:
    features, labels = _feature_and_label_fixture(periods=90, symbols=("A", "B", "C"))
    calendar = _label_calendar(labels)
    dataset = build_morning_supervised_dataset(
        features,
        labels,
        publication_as_of=datetime(2027, 1, 1, tzinfo=JST),
        trading_calendar=calendar,
    )
    snapshot = write_morning_dataset_snapshot(
        dataset,
        tmp_path / "mlp-dataset",
        created_at=datetime(2026, 8, 24, 12, tzinfo=JST),
        publication_as_of=datetime(2027, 1, 1, tzinfo=JST),
        capability_report=_core_capability_report(),
        manifest=MORNING_CORE_MANIFEST,
        trading_calendar=calendar,
    )
    authenticated, authenticated_dataset = load_morning_dataset_snapshot(snapshot.parquet_path)
    run = run_morning_research(
        authenticated,
        authenticated_dataset,
        created_at=datetime(2026, 8, 24, 12, tzinfo=JST),
        code_commit="fixture-commit",
        config=MorningResearchConfig(
            horizons=(1, 5, 20),
            model_families=("mlp",),
            seeds=(17, 23, 31),
            initial_train_periods=20,
            validation_periods=5,
            step_periods=10,
            holdout_periods=5,
            mlp_hidden_units=(4,),
            mlp_max_iterations=300,
            enable_neural_challenger=True,
            max_model_fits=60,
        ),
    )
    assert {result.model_family for result in run.report.results} == {"mlp"}
    assert len(run.report.results) == 9
    assert run.report.neural_and_sequence_capabilities[0].disposition.value in {
        "RESEARCH",
        "REJECTED",
    }
    assert run.report.adoption_eligible is False


def test_current_research_inference_flows_through_decision_engine(tmp_path: Path) -> None:
    bars, context, market, sectors, labels, freezes, trading_calendar = (
        morning_research_fixture(periods=76)
    )
    capability = _core_capability_report()
    features = build_morning_features(
        bars,
        daily_context=context,
        market_bars=market,
        sector_bars=sectors,
        capability_report=capability,
        freeze_metadata=freezes,
    )
    current_date = pd.to_datetime(features.frame["trading_date"]).max()
    dataset = build_morning_supervised_dataset(
        features.frame.loc[pd.to_datetime(features.frame["trading_date"]) < current_date],
        labels.loc[pd.to_datetime(labels["trading_date"]) < current_date],
        publication_as_of=datetime(2027, 1, 1, tzinfo=JST),
        trading_calendar=trading_calendar,
    )
    snapshot = write_morning_dataset_snapshot(
        dataset,
        tmp_path,
        created_at=datetime(2026, 8, 24, 12, tzinfo=JST),
        publication_as_of=datetime(2027, 1, 1, tzinfo=JST),
        capability_report=capability,
        manifest=features.manifest,
        trading_calendar=trading_calendar,
    )
    authenticated, authenticated_dataset = load_morning_dataset_snapshot(snapshot.parquet_path)
    run = run_morning_research(
        authenticated,
        authenticated_dataset,
        created_at=datetime(2026, 8, 24, 12, tzinfo=JST),
        code_commit="fixture-commit",
        config=MorningResearchConfig(
            horizons=(1, 5, 20),
            model_families=("ridge",),
            seeds=(17,),
            initial_train_periods=20,
            validation_periods=5,
            step_periods=10,
            holdout_periods=5,
            max_model_fits=100,
        ),
    )
    with pytest.raises(ValueError, match="snapshot identity"):
        fit_morning_research_bundle(
            run,
            authenticated.model_copy(update={"snapshot_id": "forged-snapshot"}),
            authenticated_dataset,
            selected_family="ridge",
            selected_seed=17,
        )
    fitted = fit_morning_research_bundle(
        run,
        authenticated,
        authenticated_dataset,
        selected_family="ridge",
        selected_seed=17,
    )
    with pytest.raises(ValueError, match="authenticated refit path"):
        morning_ml.MorningFittedResearchBundle(
            _construction_token=object(),
            report=fitted.report,
            snapshot=fitted.snapshot,
            selected_family=fitted.selected_family,
            selected_seed=fitted.selected_seed,
            training_as_of=fitted.training_as_of,
            models=fitted.models,
            calibration_oof=fitted.calibration_oof,
        )
    current_freeze = next(item for item in freezes if item.as_of.date() == current_date.date())
    batch = infer_current_morning_predictions(
        fitted,
        features,
        current_freeze,
        minimum_calibration_rows=20,
    )
    assert len(batch.predictions) == 4
    batch_payload = batch.model_dump(mode="python")
    invalid_batch_cases = (
        ({"predictions": (), "blocked": ()}, "predictions or an explicit block"),
        ({"universe_symbols": batch.universe_symbols[:-1]}, "roles must cover"),
        ({"predictions": batch.predictions[:-1]}, "predictions must cover"),
        ({"frozen_market": batch.frozen_market[:-1]}, "market data must cover"),
        (
            {"frozen_market": (*batch.frozen_market, batch.frozen_market[0])},
            "market symbols must be unique",
        ),
        (
            {
                "predictions": (
                    batch.predictions[0].model_copy(
                        update={
                            "as_of": batch.as_of + timedelta(days=1),
                            "morning_revision": batch.predictions[0].morning_revision.model_copy(
                                update={"as_of": batch.as_of + timedelta(days=1)}
                            ),
                        }
                    ),
                    *batch.predictions[1:],
                )
            },
            "share the exact freeze",
        ),
        (
            {"source_snapshot_ids": (*batch.source_snapshot_ids, batch.source_snapshot_ids[0])},
            "source snapshot IDs",
        ),
        ({"capability_statuses": ()}, "requires capability statuses"),
        ({"is_research_only": False}, "must remain research-only"),
        ({"blocked": ("blocked",)}, "cannot retain predictions"),
    )
    for update, message in invalid_batch_cases:
        with pytest.raises(ValueError, match=message):
            type(batch).model_validate({**batch_payload, **update})
    current_rows = features.frame.loc[
        pd.to_datetime(features.frame["trading_date"]).eq(current_date)
    ].set_index("symbol")
    holding_price = Decimal(str(current_rows.loc["7203", "reference_price_1130"]))
    second_holding_price = Decimal(str(current_rows.loc["8306", "reference_price_1130"]))
    state = portfolio(
        (
            Position(
                symbol="7203",
                account_bucket_id="bucket",
                shares=100,
                average_acquisition_price=Decimal("900"),
                market_price=holding_price,
            ),
            Position(
                symbol="8306",
                account_bucket_id="bucket",
                shares=100,
                average_acquisition_price=Decimal("1000"),
                market_price=second_holding_price,
            ),
        ),
        cash=Decimal("100000"),
    ).model_copy(update={"as_of": current_freeze.as_of})
    state = state.model_copy(
        update={
            "tax_states": tuple(
                item.model_copy(update={"tax_year": current_freeze.as_of.year})
                for item in state.tax_states
            )
        }
    )
    proposal = propose_from_morning_batch(
        decision_engine(),
        portfolio=state,
        batch=batch,
        features=features,
        freeze_metadata=current_freeze,
        securities={
            str(symbol): Security(
                symbol=str(symbol),
                company_name=f"Company {symbol}",
                sector=str(row["sector"]),
            )
            for symbol, row in current_rows.iterrows()
        },
        account_bucket_ids_by_symbol={symbol: ("bucket",) for symbol in batch.universe_symbols},
        generated_at=current_freeze.as_of + timedelta(minutes=1),
    )
    assert proposal.is_order_instruction is False
    assert proposal.is_research_only is True
    assert proposal.morning_audit is not None
    assert proposal.morning_audit.research_report_id == batch.research_report_id
    assert proposal.morning_audit.freeze_evidence_hash == batch.freeze_evidence_hash
    assert dict(proposal.morning_audit.average_daily_trading_values) == {
        item.symbol: item.average_daily_trading_value for item in batch.frozen_market
    }
    audit_payload = proposal.morning_audit.model_dump(mode="python")
    invalid_audit_cases = (
        (
            {"as_of": proposal.morning_audit.as_of + timedelta(minutes=1)},
            "exact 11:30",
        ),
        (
            {
                "source_record_ids": (
                    *proposal.morning_audit.source_record_ids,
                    proposal.morning_audit.source_record_ids[0],
                )
            },
            "source record IDs",
        ),
        ({"universe_roles": {}}, "requires the frozen universe roles"),
        ({"capability_statuses": {}}, "requires capability statuses"),
        ({"reference_prices": {}}, "market data must cover"),
        ({"is_research_only": False}, "must remain research-only"),
    )
    for update, message in invalid_audit_cases:
        with pytest.raises(ValueError, match=message):
            MorningDecisionAudit.model_validate({**audit_payload, **update})
    assert {line.symbol for line in proposal.lines} == {
        prediction.symbol for prediction in batch.predictions
    }
    with pytest.raises(ValueError, match="exact freeze universe"):
        propose_from_morning_batch(
            decision_engine(),
            portfolio=state,
            batch=batch,
            features=features,
            freeze_metadata=current_freeze,
            securities={
                str(symbol): Security(
                    symbol=str(symbol),
                    company_name=f"Company {symbol}",
                    sector=str(row["sector"]),
                )
                for symbol, row in current_rows.iterrows()
            },
            account_bucket_ids_by_symbol={
                symbol: ("bucket",) for symbol in batch.universe_symbols[:-1]
            },
            generated_at=current_freeze.as_of + timedelta(minutes=2),
        )

    changed_market = features.frame.copy()
    changed_market.loc[
        pd.to_datetime(changed_market["trading_date"]).eq(current_date)
        & changed_market["symbol"].eq("7203"),
        "reference_price_1130",
    ] += 1.0
    with pytest.raises(ValueError, match="feature evidence"):
        propose_from_morning_batch(
            decision_engine(),
            portfolio=state,
            batch=batch,
            features=replace(features, frame=changed_market),
            freeze_metadata=current_freeze,
            securities={
                str(symbol): Security(
                    symbol=str(symbol),
                    company_name=f"Company {symbol}",
                    sector=str(row["sector"]),
                )
                for symbol, row in current_rows.iterrows()
            },
            account_bucket_ids_by_symbol={symbol: ("bucket",) for symbol in batch.universe_symbols},
            generated_at=current_freeze.as_of + timedelta(minutes=2),
        )
    with pytest.raises(ValueError, match="holding roles"):
        propose_from_morning_batch(
            decision_engine(),
            portfolio=state.model_copy(update={"positions": state.positions[:1]}),
            batch=batch,
            features=features,
            freeze_metadata=current_freeze,
            securities={
                str(symbol): Security(
                    symbol=str(symbol),
                    company_name=f"Company {symbol}",
                    sector=str(row["sector"]),
                )
                for symbol, row in current_rows.iterrows()
            },
            account_bucket_ids_by_symbol={symbol: ("bucket",) for symbol in batch.universe_symbols},
            generated_at=current_freeze.as_of + timedelta(minutes=2),
        )
    wrong_price_state = state.model_copy(
        update={
            "positions": (
                state.positions[0].model_copy(
                    update={"market_price": state.positions[0].market_price + Decimal("1")}
                ),
                *state.positions[1:],
            )
        }
    )
    with pytest.raises(ValueError, match="holding prices"):
        propose_from_morning_batch(
            decision_engine(),
            portfolio=wrong_price_state,
            batch=batch,
            features=features,
            freeze_metadata=current_freeze,
            securities={
                str(symbol): Security(
                    symbol=str(symbol),
                    company_name=f"Company {symbol}",
                    sector=str(row["sector"]),
                )
                for symbol, row in current_rows.iterrows()
            },
            account_bucket_ids_by_symbol={symbol: ("bucket",) for symbol in batch.universe_symbols},
            generated_at=current_freeze.as_of + timedelta(minutes=2),
        )
    changed_frozen = batch.frozen_market[0].model_copy(
        update={"reference_price": batch.frozen_market[0].reference_price + Decimal("1")}
    )
    with pytest.raises(ValueError, match="frozen batch"):
        propose_from_morning_batch(
            decision_engine(),
            portfolio=state,
            batch=batch.model_copy(
                update={"frozen_market": (changed_frozen, *batch.frozen_market[1:])}
            ),
            features=features,
            freeze_metadata=current_freeze,
            securities={
                str(symbol): Security(
                    symbol=str(symbol),
                    company_name=f"Company {symbol}",
                    sector=str(row["sector"]),
                )
                for symbol, row in current_rows.iterrows()
            },
            account_bucket_ids_by_symbol={symbol: ("bucket",) for symbol in batch.universe_symbols},
            generated_at=current_freeze.as_of + timedelta(minutes=2),
        )
    with pytest.raises(ValueError, match="current-feature"):
        propose_from_morning_batch(
            decision_engine(),
            portfolio=state,
            batch=batch.model_copy(update={"evidence_kind": "HISTORICAL_OOF_REPLAY"}),
            features=features,
            freeze_metadata=current_freeze,
            securities={
                str(symbol): Security(
                    symbol=str(symbol),
                    company_name=f"Company {symbol}",
                    sector=str(row["sector"]),
                )
                for symbol, row in current_rows.iterrows()
            },
            account_bucket_ids_by_symbol={symbol: ("bucket",) for symbol in batch.universe_symbols},
            generated_at=current_freeze.as_of + timedelta(minutes=2),
        )

    incomplete_frame = features.frame.copy()
    current_mask = pd.to_datetime(incomplete_frame["trading_date"]).eq(current_date)
    incomplete_frame.loc[current_mask, "morning.volume_progress_1130_20d"] = np.nan
    blocked = infer_current_morning_predictions(
        fitted,
        replace(features, frame=incomplete_frame),
        current_freeze,
        minimum_calibration_rows=20,
    )
    assert blocked.predictions == ()
    assert "profile" in blocked.blocked[0]

    authentic_bundle_id = fitted.bundle_id
    object.__setattr__(fitted, "bundle_id", "forged-bundle")
    with pytest.raises(ValueError, match="bundle identity"):
        infer_current_morning_predictions(
            fitted,
            features,
            current_freeze,
            minimum_calibration_rows=20,
        )
    object.__setattr__(fitted, "bundle_id", authentic_bundle_id)

    fitted_model = fitted.models[5].named_steps["model"]
    fitted_model.coef_[0] += 1.0
    with pytest.raises(ValueError, match="model state was mutated"):
        infer_current_morning_predictions(
            fitted,
            features,
            current_freeze,
            minimum_calibration_rows=20,
        )


def test_decision_prediction_calibration_uses_only_matured_past_targets() -> None:
    oof = _decision_oof_fixture()
    freeze, snapshot, report = _decision_artifacts(oof)
    original = build_morning_decision_predictions(
        oof,
        report=report,
        snapshot=snapshot,
        freeze_metadata=freeze,
        selected_family="ridge",
        selected_seed=17,
        minimum_calibration_rows=8,
    )
    assert original.predictions
    first = original.predictions[0]
    mutated = oof.copy()
    unavailable = pd.to_datetime(mutated["label_available_at"], utc=True) >= pd.Timestamp(
        first.as_of
    ).tz_convert("UTC")
    mutated.loc[unavailable, "target"] += 50.0
    _, _, mutated_report = _decision_artifacts(mutated)
    rerun = build_morning_decision_predictions(
        mutated,
        report=mutated_report,
        snapshot=snapshot,
        freeze_metadata=freeze,
        selected_family="ridge",
        selected_seed=17,
        minimum_calibration_rows=8,
    )
    rerun_map = {(item.symbol, item.as_of): item for item in rerun.predictions}
    observed = rerun_map[(first.symbol, first.as_of)]
    assert observed.expected_return_1d == first.expected_return_1d
    assert observed.expected_return_5d == first.expected_return_5d
    assert observed.expected_return_20d == first.expected_return_20d
    assert observed.downside_quantile == first.downside_quantile
    assert observed.large_loss_probability == first.large_loss_probability
    assert observed.uncertainty == first.uncertainty
    assert first.morning_revision is not None
    assert first.morning_revision.calibration_history_rows >= 8

    same_day = oof.copy()
    calibration_row = same_day.index[same_day["horizon"].eq(5)][0]
    same_day.loc[calibration_row, "label_end"] = pd.Timestamp(first.as_of.date())
    same_day.loc[calibration_row, "label_available_at"] = first.as_of - timedelta(hours=1)
    changed = same_day.copy()
    changed.loc[calibration_row, "target"] += 50.0
    batches = []
    for frame in (same_day, changed):
        _, _, frame_report = _decision_artifacts(frame)
        batches.append(
            build_morning_decision_predictions(
                frame,
                report=frame_report,
                snapshot=snapshot,
                freeze_metadata=freeze,
                selected_family="ridge",
                selected_seed=17,
                minimum_calibration_rows=8,
            )
        )
    first_by_symbol = {item.symbol: item for item in batches[0].predictions}
    changed_by_symbol = {item.symbol: item for item in batches[1].predictions}
    for symbol in first_by_symbol:
        assert changed_by_symbol[symbol].uncertainty == first_by_symbol[symbol].uncertainty
        assert (
            changed_by_symbol[symbol].large_loss_probability
            == first_by_symbol[symbol].large_loss_probability
        )


def test_decision_prediction_requires_exact_horizon_and_prior_bundle_coherence() -> None:
    oof = _decision_oof_fixture()
    freeze, snapshot, _ = _decision_artifacts(oof)
    current = pd.to_datetime(oof["trading_date"]).dt.date == freeze.as_of.date()
    one_day = oof["horizon"].eq(1)
    wrong_prior = oof.copy()
    wrong_prior.loc[current & one_day, "prior_model_version"] = "wrong-prior"
    _, _, wrong_report = _decision_artifacts(wrong_prior)
    blocked = build_morning_decision_predictions(
        wrong_prior,
        report=wrong_report,
        snapshot=snapshot,
        freeze_metadata=freeze,
        selected_family="ridge",
        selected_seed=17,
        minimum_calibration_rows=8,
    )
    assert blocked.predictions == ()
    assert "PROVENANCE_COHERENCE" in blocked.blocked[0]

    duplicate = pd.concat(
        [oof, oof.loc[current & one_day & oof["symbol"].eq("A")]], ignore_index=True
    )
    _, _, duplicate_report = _decision_artifacts(duplicate)
    duplicate_batch = build_morning_decision_predictions(
        duplicate,
        report=duplicate_report,
        snapshot=snapshot,
        freeze_metadata=freeze,
        selected_family="ridge",
        selected_seed=17,
        minimum_calibration_rows=8,
    )
    assert duplicate_batch.predictions == ()
    assert any("incomplete" in reason for reason in duplicate_batch.blocked)

    other_capability = morning_capabilities(
        provider="other-provider",
        available_fields=(
            "timestamp",
            "price",
            "volume",
            "trading_value",
            "historical_same_time_sessions",
        ),
    )
    wrong_freeze = freeze.model_copy(
        update={"provider": "other-provider", "capability_report": other_capability}
    )
    with pytest.raises(ValueError, match="freeze lineage"):
        build_morning_decision_predictions(
            oof,
            report=_decision_artifacts(oof)[2],
            snapshot=snapshot,
            freeze_metadata=wrong_freeze,
            selected_family="ridge",
            selected_seed=17,
            minimum_calibration_rows=8,
        )


def test_morning_revision_can_change_actions_only_through_decision_engine() -> None:
    oof = _decision_oof_fixture()
    freeze, snapshot, _ = _decision_artifacts(oof)
    prior_a = _prior_prediction("A", expected=0.02, as_of=freeze.as_of - timedelta(hours=2))
    prior_b = _prior_prediction("B", expected=-0.01, as_of=freeze.as_of - timedelta(hours=2))
    current = pd.to_datetime(oof["trading_date"]).dt.date == freeze.as_of.date()
    for symbol, prior_value, final_value in (("A", 0.02, -0.40), ("B", -0.01, 0.40)):
        selected = current & oof["symbol"].eq(symbol)
        oof.loc[selected, "prior_prediction"] = prior_value
        oof.loc[selected, "final_prediction"] = final_value
        oof.loc[selected, "prior_downside_quantile"] = 0.0
        oof.loc[selected, "prior_large_loss_probability"] = 0.0
        oof.loc[selected, "prior_uncertainty"] = 0.0
    _, _, report = _decision_artifacts(oof)
    batch = build_morning_decision_predictions(
        oof,
        report=report,
        snapshot=snapshot,
        freeze_metadata=freeze,
        selected_family="ridge",
        selected_seed=17,
        minimum_calibration_rows=8,
    )
    revised = {prediction.symbol: prediction for prediction in batch.predictions}
    revised_a = revised["A"]
    revised_b = revised["B"]
    state = portfolio(
        (
            Position(
                symbol="A",
                account_bucket_id="bucket",
                shares=100,
                average_acquisition_price=Decimal("900"),
                market_price=Decimal("1000"),
            ),
        ),
        cash=Decimal("0"),
    ).model_copy(update={"as_of": freeze.as_of})
    state = state.model_copy(
        update={
            "tax_states": tuple(
                item.model_copy(update={"tax_year": freeze.as_of.year})
                for item in state.tax_states
            )
        }
    )
    engine = decision_engine()
    prior_proposal = engine.propose(
        portfolio=state,
        candidates=(
            _decision_candidate("A", prior_a),
            _decision_candidate("B", prior_b),
        ),
        generated_at=freeze.as_of + timedelta(minutes=1),
        model_bundle_version=prior_a.model_version,
    )
    revised_proposal = engine.propose(
        portfolio=state,
        candidates=(
            _decision_candidate("A", revised_a),
            _decision_candidate("B", revised_b),
        ),
        generated_at=freeze.as_of + timedelta(minutes=2),
        model_bundle_version=revised_a.model_version,
        morning_audit=MorningDecisionAudit(
            as_of=freeze.as_of,
            provider=batch.provider,
            source_snapshot_ids=batch.source_snapshot_ids,
            source_record_ids=batch.source_record_ids,
            universe_roles={member.symbol: member.role.value for member in batch.universe},
            capability_statuses=dict(batch.capability_statuses),
            research_report_id=batch.research_report_id,
            freeze_evidence_hash=batch.freeze_evidence_hash,
            reference_prices={"A": Decimal("1000"), "B": Decimal("1000")},
            average_daily_trading_values={
                "A": Decimal("100000000"),
                "B": Decimal("100000000"),
            },
        ),
    )
    prior_actions = {line.symbol: line.action for line in prior_proposal.lines}
    revised_actions = {line.symbol: line.action for line in revised_proposal.lines}
    assert prior_actions == {"A": ProposalAction.HOLD, "B": ProposalAction.SKIP}
    assert revised_actions == {"A": ProposalAction.SELL, "B": ProposalAction.BUY}
    assert revised_proposal.is_order_instruction is False
    assert revised_proposal.morning_audit is not None
    mismatched_audit = revised_proposal.morning_audit.model_copy(
        update={"reference_prices": {"A": Decimal("999"), "B": Decimal("1000")}}
    )
    with pytest.raises(ValueError, match="market data must match"):
        engine.propose(
            portfolio=state,
            candidates=(
                _decision_candidate("A", revised_a),
                _decision_candidate("B", revised_b),
            ),
            generated_at=freeze.as_of + timedelta(minutes=3),
            model_bundle_version=revised_a.model_version,
            morning_audit=mismatched_audit,
        )


def _core_capability_report() -> MorningCapabilityReport:
    return morning_capabilities(
        provider="deterministic-fixture",
        available_fields=(
            "timestamp",
            "price",
            "volume",
            "trading_value",
            "historical_same_time_sessions",
        ),
    )


def _morning_inputs(
    *, periods: int, symbols: tuple[str, ...]
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    tuple[MorningFreezeMetadata, ...],
]:
    dates = pd.bdate_range("2025-01-06", periods=periods)
    stock_rows: list[dict[str, object]] = []
    market_rows: list[dict[str, object]] = []
    sector_rows: list[dict[str, object]] = []
    context_rows: list[dict[str, object]] = []
    sectors = {
        symbol: ("Sector-X" if number % 2 == 0 else "Sector-Y")
        for number, symbol in enumerate(symbols)
    }
    for day_number, trading_date in enumerate(dates):
        day = trading_date.date()
        signal = 0.002 + day_number * 0.0001
        for symbol_number, symbol in enumerate(symbols):
            base = 100.0 + symbol_number * 10.0
            for cutoff_number, cutoff in enumerate(CUTOFFS):
                fraction = cutoff_number / (len(CUTOFFS) - 1)
                timestamp = datetime.combine(day, cutoff, tzinfo=JST)
                price = base * (1.0 + signal * fraction)
                volume = (100.0 + symbol_number * 10.0) * (1.0 + day_number * 0.01)
                stock_rows.append(
                    _bar_row(symbol, timestamp, price, volume, f"stock-{day}-{symbol}-{cutoff}")
                )
            context_rows.append(
                {
                    "symbol": symbol,
                    "trading_date": trading_date,
                    "sector": sectors[symbol],
                    "prior_close": base * 0.99,
                    "average_daily_trading_value": 100_000_000.0,
                    "prior_expected_return_1d": symbol_number * 0.001,
                    "prior_expected_return_5d": symbol_number * 0.002,
                    "prior_expected_return_20d": symbol_number * 0.003,
                    "prior_downside_quantile": -0.02,
                    "prior_large_loss_probability": 0.10,
                    "prior_uncertainty": 0.02,
                    "prior_rank_pct": (symbol_number + 1) / len(symbols),
                    "prior_model_version": "daily-v1",
                    "prior_feature_version": "daily-feature-v2",
                    "prior_data_snapshot_id": "daily-snapshot",
                    "prior_prediction_as_of": datetime.combine(day, time(8, 50), tzinfo=JST),
                    "is_current_holding": symbol_number in {0, 2},
                    "is_candidate": symbol_number != 0,
                    "available_at": datetime.combine(day, time(8, 50), tzinfo=JST),
                }
            )
        for cutoff_number, cutoff in enumerate(CUTOFFS):
            fraction = cutoff_number / (len(CUTOFFS) - 1)
            timestamp = datetime.combine(day, cutoff, tzinfo=JST)
            market_rows.append(
                _bar_row(
                    "TOPIX",
                    timestamp,
                    2_000.0 * (1.0 + signal * 0.2 * fraction),
                    1_000.0,
                    f"market-{day}-{cutoff}",
                )
            )
            for sector_name in sorted(set(sectors.values())):
                sector_rows.append(
                    _bar_row(
                        sector_name,
                        timestamp,
                        500.0 * (1.0 + signal * 0.4 * fraction),
                        500.0,
                        f"sector-{sector_name}-{day}-{cutoff}",
                    )
                )
    stock = pd.DataFrame(stock_rows)
    market = pd.DataFrame(market_rows)
    sector = pd.DataFrame(sector_rows)
    all_bars = pd.concat((stock, market, sector), ignore_index=True)
    report = _core_capability_report()
    universe = build_morning_universe(
        current_holdings=tuple(
            symbol for number, symbol in enumerate(symbols) if number in {0, 2}
        ),
        candidates=tuple(symbol for number, symbol in enumerate(symbols) if number != 0),
    )
    freezes = tuple(
        MorningFreezeMetadata(
            as_of=datetime.combine(trading_date.date(), time(11, 30), tzinfo=JST),
            provider="deterministic-fixture",
            source_snapshot_ids=(f"snapshot-{trading_date.date()}",),
            source_record_ids=tuple(
                sorted(
                    all_bars.loc[
                        all_bars["timestamp"]
                        .map(lambda value: pd.Timestamp(value).date())
                        .eq(trading_date.date()),
                        "source_record_id",
                    ].astype(str)
                )
            ),
            universe=universe,
            capability_report=report,
        )
        for trading_date in dates
    )
    return stock, pd.DataFrame(context_rows), market, sector, freezes


def _bar_row(
    symbol: str, timestamp: datetime, price: float, volume: float, source_id: str
) -> dict[str, object]:
    return {
        "symbol": symbol,
        "timestamp": timestamp,
        "available_at": timestamp,
        "price": price,
        "volume": volume,
        "trading_value": volume * price,
        "provider": "deterministic-fixture",
        "source_record_id": source_id,
    }


def _feature_and_label_fixture(
    *, periods: int, symbols: tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bars, context, market, sectors, freezes = _morning_inputs(periods=periods, symbols=symbols)
    features = build_morning_features(
        bars,
        daily_context=context,
        market_bars=market,
        sector_bars=sectors,
        capability_report=_core_capability_report(),
        freeze_metadata=freezes,
    ).frame
    labels: list[dict[str, object]] = []
    for _, row in features.iterrows():
        trading_date = pd.Timestamp(row["trading_date"])
        as_of = pd.Timestamp(row["as_of"])
        item: dict[str, object] = {"symbol": row["symbol"], "trading_date": trading_date}
        for horizon in (1, 5, 20):
            end_date = trading_date + pd.offsets.BDay(horizon)
            target = float(row["morning.return_0900_1130"]) * (1 + horizon / 20)
            target += float(row[f"morning.prior_expected_return_{horizon}d"])
            item[f"target_return_{horizon}d"] = target
            item[f"label_entry_at_{horizon}d"] = as_of.to_pydatetime() + timedelta(hours=1)
            item[f"label_end_date_{horizon}d"] = end_date
            item[f"label_end_at_{horizon}d"] = datetime.combine(
                end_date.date(), time(15, 30), tzinfo=JST
            )
            item[f"label_available_at_{horizon}d"] = datetime.combine(
                end_date.date(), time(16), tzinfo=JST
            )
            item[f"label_status_{horizon}d"] = "AVAILABLE"
        labels.append(item)
    return features, pd.DataFrame(labels)


def _label_calendar(labels: pd.DataFrame) -> pd.DatetimeIndex:
    start = pd.to_datetime(labels["trading_date"]).min()
    end = max(
        pd.to_datetime(labels[f"label_end_date_{horizon}d"]).max() for horizon in (1, 5, 20)
    )
    return pd.bdate_range(start, end)


def _decision_oof_fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.bdate_range("2026-07-01", periods=35)
    for horizon in (1, 5, 20):
        for day_number, trading_date in enumerate(dates):
            for symbol_number, symbol in enumerate(("A", "B")):
                prior = symbol_number * 0.01
                revision = (day_number % 5 - 2) * 0.001
                target = prior + revision + ((day_number + symbol_number) % 3 - 1) * 0.0005
                rows.append(
                    {
                        "symbol": symbol,
                        "trading_date": trading_date,
                        "as_of": datetime.combine(trading_date.date(), time(11, 30), tzinfo=JST),
                        "provider": "deterministic-fixture",
                        "source_snapshot_ids": ("morning-source-snapshot",),
                        "source_record_ids": ("morning-source-record",),
                        "reference_price_1130": 100.0 + symbol_number,
                        "average_daily_trading_value": 100_000_000.0,
                        "prior_prediction_as_of": datetime.combine(
                            trading_date.date(), time(8, 50), tzinfo=JST
                        ),
                        "horizon": horizon,
                        "target": target,
                        "label_end": trading_date + pd.offsets.BDay(horizon),
                        "label_available_at": datetime.combine(
                            (trading_date + pd.offsets.BDay(horizon)).date(),
                            time(16),
                            tzinfo=JST,
                        ),
                        "prior_prediction": prior,
                        "final_prediction": prior + revision,
                        "model_family": "ridge",
                        "seed": 17,
                        "prior_downside_quantile": -0.02,
                        "prior_large_loss_probability": 0.1,
                        "prior_uncertainty": 0.02,
                        "prior_model_version": "daily-v1",
                        "prior_feature_version": "daily-feature-v2",
                        "prior_data_snapshot_id": "daily-snapshot",
                        "is_current_holding": symbol == "A",
                        "is_candidate": symbol == "B",
                    }
                )
    return pd.DataFrame(rows)


def _decision_artifacts(
    oof: pd.DataFrame,
) -> tuple[MorningFreezeMetadata, MorningDatasetSnapshot, MorningResearchReport]:
    freeze_date = pd.to_datetime(oof["trading_date"]).sort_values().unique()[-2]
    freeze_as_of = datetime.combine(pd.Timestamp(freeze_date).date(), time(11, 30), tzinfo=JST)
    capability = _core_capability_report()
    freeze = MorningFreezeMetadata(
        as_of=freeze_as_of,
        provider="deterministic-fixture",
        source_snapshot_ids=("morning-source-snapshot",),
        source_record_ids=("morning-source-record",),
        universe=build_morning_universe(current_holdings=("A",), candidates=("B",)),
        capability_report=capability,
    )
    snapshot = MorningDatasetSnapshot(
        snapshot_id="a" * 64,
        created_at=datetime(2026, 8, 24, 12, tzinfo=JST),
        publication_as_of=datetime(2027, 1, 1, tzinfo=JST),
        provider="deterministic-fixture",
        source_snapshot_ids=("morning-source-snapshot",),
        source_record_ids=("morning-source-record",),
        feature_set_version=MORNING_CORE_MANIFEST.feature_set_version,
        feature_manifest_hash=MORNING_CORE_MANIFEST.manifest_hash,
        feature_names=MORNING_CORE_MANIFEST.feature_names,
        frame_hash="b" * 64,
        capability_statuses=tuple(
            sorted((name, status.value) for name, status in capability.capabilities.items())
        ),
        trading_calendar_dates=tuple(
            str(value.date()) for value in pd.bdate_range("2025-01-06", periods=100)
        ),
        rows=1,
        data_start="2025-01-06",
        data_end="2025-03-01",
        parquet_path=Path("morning.parquet"),
        metadata_path=Path("morning.json"),
    )
    snapshot = snapshot.model_copy(
        update={"snapshot_id": _stable_hash(morning_ml._morning_snapshot_identity(snapshot))}
    )
    config = MorningResearchConfig()
    results = tuple(
        MorningModelResult(
            horizon=horizon,
            model_family="ridge",
            seed=17,
            folds=2,
            rows=20,
            dates=10,
            holdings_rows=10,
            candidate_rows=10,
            mean_squared_error=0.01,
            baseline_mean_squared_error=0.02,
            mean_daily_rank_ic=0.5,
            baseline_mean_daily_rank_ic=0.2,
            incremental_rank_ic=0.3,
            revision_win_rate=0.7,
            adds_oos_value=True,
            disposition=MorningModelDisposition.RESEARCH,
            inference_timing_status="UNMEASURED_LIVE_RESEARCH_ONLY",
        )
        for horizon in (1, 5, 20)
    )
    report = MorningResearchReport(
        report_id="PENDING",
        created_at=datetime(2026, 8, 24, 12, tzinfo=JST),
        data_snapshot_id=snapshot.snapshot_id,
        feature_set_version=snapshot.feature_set_version,
        feature_manifest_hash=snapshot.feature_manifest_hash,
        feature_names=snapshot.feature_names,
        code_commit="fixture-commit",
        config=config.model_dump(mode="json"),
        config_hash=config.config_hash,
        holdout_start="2025-04-01",
        results=results,
        neural_and_sequence_capabilities=(),
        oof_rows=len(oof),
        oof_sha256=_frame_hash(oof),
        blocking_reasons=("research-only replay",),
    )
    report = report.model_copy(
        update={"report_id": f"morning-{_stable_hash(_report_identity(report))[:24]}"}
    )
    return freeze, snapshot, report


def _prior_prediction(
    symbol: str, *, expected: float, as_of: datetime | None = None
) -> Prediction:
    return Prediction(
        symbol=symbol,
        as_of=as_of or AS_OF - timedelta(hours=2, minutes=40),
        expected_return_1d=expected,
        expected_return_5d=expected,
        expected_return_20d=expected,
        downside_quantile=0.0,
        large_loss_probability=0.0,
        uncertainty=PredictionUncertainty(standard_error=0.0),
        model_version="daily-v1",
        feature_version="daily-feature-v2",
        data_snapshot_id="daily-snapshot",
    )


def _decision_candidate(symbol: str, prediction: Prediction) -> DecisionCandidate:
    return DecisionCandidate(
        security=Security(symbol=symbol, company_name=f"Company {symbol}", sector="Sector"),
        account_bucket_id="bucket",
        price=Decimal("1000"),
        average_daily_trading_value=Decimal("100000000"),
        prediction=prediction,
    )
