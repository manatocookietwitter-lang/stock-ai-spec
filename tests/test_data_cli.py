from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from typer.testing import CliRunner

from stock_ai.cli import app
from stock_ai.data.contracts import IngestionStatus

runner = CliRunner()


def test_capability_command_needs_no_credential() -> None:
    result = runner.invoke(app, ["data", "capabilities", "--plan", "free"])
    assert result.exit_code == 0
    assert "daily_prices AVAILABLE" in result.stdout
    assert "topix_context BLOCKED_BY_PLAN" in result.stdout
    assert "intraday_morning OUT_OF_SCOPE" in result.stdout


def test_sync_without_credential_fails_closed_without_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("JQUANTS_API_KEY", raising=False)
    result = runner.invoke(
        app,
        [
            "data",
            "sync",
            "--date",
            "2026-08-21",
            "--data-root",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 2
    assert "JQUANTS_API_KEY is required" in result.stderr
    assert "fixture" not in result.stderr.lower()
    assert not list(tmp_path.rglob("*.parquet"))


def test_verify_empty_store_is_explicit(tmp_path: Path) -> None:
    result = runner.invoke(app, ["data", "verify", "--data-root", str(tmp_path)])
    assert result.exit_code == 1
    assert "no immutable objects or production snapshots" in result.stderr

    result = runner.invoke(
        app,
        ["data", "verify", "--data-root", str(tmp_path), "--allow-empty"],
    )
    assert result.exit_code == 0
    assert (
        "verified_objects=0 feature_snapshots=0 dataset_snapshots=0 builds=0 status=OK"
        in result.stdout
    )


def test_goal2_history_and_research_commands_are_exposed() -> None:
    data_help = runner.invoke(app, ["data", "--help"])
    research_help = runner.invoke(app, ["research", "--help"])
    assert data_help.exit_code == 0
    assert "history" in data_help.stdout
    assert research_help.exit_code == 0
    assert "build" in research_help.stdout
    assert "baseline" in research_help.stdout
    assert "e2e" in research_help.stdout


class _Context:
    def __enter__(self) -> _Context:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def test_data_sync_and_history_success_orchestration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("stock_ai.cli.JQuantsV2Client.from_env", lambda **_kwargs: _Context())
    monkeypatch.setattr("stock_ai.cli.DuckDBCatalog", lambda _path: _Context())
    sync_result = SimpleNamespace(
        ingestion_run_id="run-safe",
        status=IngestionStatus.SUCCEEDED,
        source_date=date(2026, 8, 21),
        objects=(),
    )
    monkeypatch.setattr(
        "stock_ai.cli.JQuantsV2Ingestor",
        lambda **_kwargs: SimpleNamespace(
            sync_date=lambda _day, datasets: sync_result if datasets else None
        ),
    )
    sync = runner.invoke(
        app,
        ["data", "sync", "--date", "2026-08-21", "--data-root", str(tmp_path)],
    )
    assert sync.exit_code == 0
    assert "run=run-safe status=SUCCEEDED" in sync.stdout

    history_result = SimpleNamespace(
        start=date(2026, 8, 20),
        end=date(2026, 8, 21),
        listed_files=1,
        downloaded_files=1,
        skipped_files=0,
        ingested_source_dates=2,
        objects=4,
    )
    monkeypatch.setattr(
        "stock_ai.cli.JQuantsV2HistoryIngestor",
        lambda **_kwargs: SimpleNamespace(
            sync_history=lambda *_args, **_kwargs2: history_result
        ),
    )
    history = runner.invoke(
        app,
        [
            "data",
            "history",
            "--start",
            "2026-08-20",
            "--end",
            "2026-08-21",
            "--data-root",
            str(tmp_path),
        ],
    )
    assert history.exit_code == 0
    assert "listed=1 downloaded=1" in history.stdout


def test_research_cli_orchestration_never_creates_an_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cutoff = pd.Timestamp("2026-08-24T11:30:00+09:00")
    features = SimpleNamespace(
        v0=pd.DataFrame({"row": [1]}),
        v1_core=pd.DataFrame(
            {
                "symbol": ["7203"],
                "trading_date": [pd.Timestamp("2026-08-22")],
                "available_at": [cutoff],
            }
        ),
    )
    snapshot = SimpleNamespace(
        snapshot_id="d" * 64,
        rows=1,
        data_start="2026-08-22",
        data_end="2026-08-22",
        label_1230_status=SimpleNamespace(value="BLOCKED_BY_DATA_CAPABILITY"),
    )
    artifacts = SimpleNamespace(
        features=features,
        snapshot=snapshot,
        dataset=pd.DataFrame({"target_return_5d": [0.01]}),
        bundle=SimpleNamespace(universe=pd.DataFrame({"symbol": ["7203"]})),
    )
    monkeypatch.setattr("stock_ai.cli._build_production_artifacts", lambda **_kwargs: artifacts)

    build = runner.invoke(
        app,
        [
            "research",
            "build",
            "--as-of",
            cutoff.isoformat(),
            "--data-root",
            str(tmp_path),
        ],
    )
    assert build.exit_code == 0
    assert "v0_rows=1 v1_rows=1" in build.stdout

    models = (
        SimpleNamespace(
            model_name="CASH",
            folds=1,
            mean_squared_error=0.0,
            mean_daily_rank_ic=None,
            rank_ic_dates=0,
        ),
    )
    report = SimpleNamespace(
        report_id="r" * 64,
        locked_holdout_start="2026-08-01",
        models=models,
    )
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.touch()
    monkeypatch.setattr(
        "stock_ai.cli.load_production_dataset_snapshot",
        lambda _path: (snapshot, artifacts.dataset),
    )
    monkeypatch.setattr(
        "stock_ai.cli.run_production_walk_forward_baselines",
        lambda *_args, **_kwargs: report,
    )
    monkeypatch.setattr(
        "stock_ai.cli.write_production_baseline_report",
        lambda _report, root: root / "report.json",
    )
    baseline = runner.invoke(
        app,
        [
            "research",
            "baseline",
            "--dataset-parquet",
            str(dataset_path),
            "--code-commit",
            "test-commit",
            "--report-root",
            str(tmp_path / "reports"),
        ],
    )
    assert baseline.exit_code == 0
    assert "model=CASH" in baseline.stdout

    proposal = SimpleNamespace(
        proposal_id="proposal-safe",
        model_dump=lambda **_kwargs: {"is_order_instruction": False},
    )
    decision = SimpleNamespace(
        proposal=proposal,
        reference_price_rule="RESEARCH_CLOSE_PROXY; BLOCKED_BY_DATA_CAPABILITY",
        candidate_count=1,
    )
    monkeypatch.setattr(
        "stock_ai.cli.run_research_decision_e2e",
        lambda **_kwargs: decision,
    )
    e2e = runner.invoke(
        app,
        [
            "research",
            "e2e",
            "--as-of",
            cutoff.isoformat(),
            "--code-commit",
            "test-commit",
            "--data-root",
            str(tmp_path),
            "--report-root",
            str(tmp_path / "research-reports"),
            "--candidate-limit",
            "1",
        ],
    )
    assert e2e.exit_code == 0
    assert "no order or execution record" in e2e.stdout
    proposal_path = tmp_path / "research-reports" / "proposals" / "proposal-safe.research.json"
    assert proposal_path.is_file()
    assert '"is_order_instruction": false' in proposal_path.read_text(encoding="utf-8")


def test_cli_input_and_research_failures_are_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_sync = runner.invoke(app, ["data", "sync", "--date", "not-a-date"])
    invalid_history = runner.invoke(
        app,
        ["data", "history", "--start", "bad", "--end", "2026-08-21"],
    )
    assert invalid_sync.exit_code == 2
    assert "--date must use YYYY-MM-DD" in invalid_sync.stderr
    assert invalid_history.exit_code == 2
    assert "--start and --end must use YYYY-MM-DD" in invalid_history.stderr

    monkeypatch.setattr(
        "stock_ai.cli._build_production_artifacts",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("safe build failure")),
    )
    failed_build = runner.invoke(
        app,
        ["research", "build", "--as-of", "2026-08-24T11:30:00+09:00"],
    )
    assert failed_build.exit_code == 2
    assert "production build blocked: safe build failure" in failed_build.stderr

    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.touch()
    monkeypatch.setattr(
        "stock_ai.cli.load_production_dataset_snapshot",
        lambda _path: (_ for _ in ()).throw(RuntimeError("safe snapshot failure")),
    )
    failed_baseline = runner.invoke(
        app,
        [
            "research",
            "baseline",
            "--dataset-parquet",
            str(dataset_path),
            "--code-commit",
            "test-commit",
        ],
    )
    assert failed_baseline.exit_code == 2
    assert "baseline blocked: safe snapshot failure" in failed_baseline.stderr


def test_verify_rejects_objects_without_a_catalog(tmp_path: Path) -> None:
    manifest = (
        tmp_path
        / "raw"
        / "jquants_v2"
        / "daily_prices"
        / "source_date=2026-08-21"
        / ("a" * 64)
        / "manifest.json"
    )
    manifest.parent.mkdir(parents=True)
    manifest.touch()
    verified = runner.invoke(app, ["data", "verify", "--data-root", str(tmp_path)])
    assert verified.exit_code == 1
    assert "catalog is missing" in verified.stderr
