from __future__ import annotations

from typer.testing import CliRunner

from stock_ai.cli import app


def test_deterministic_fixture_end_to_end(tmp_path: object) -> None:
    result = CliRunner().invoke(app, ["fixture-demo", "--snapshot-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "DETERMINISTIC FIXTURE ONLY" in result.output
    assert "features=54" in result.output
    assert "validation_folds=" in result.output
    assert "manual_executions=1" in result.output
    assert "No securities order was submitted" in result.output
