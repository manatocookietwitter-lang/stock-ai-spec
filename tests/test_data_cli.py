from pathlib import Path

import pytest
from typer.testing import CliRunner

from stock_ai.cli import app

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
    assert result.exit_code == 0
    assert "verified_objects=0 status=OK" in result.stdout
