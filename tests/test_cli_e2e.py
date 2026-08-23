from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from typer.testing import CliRunner

from stock_ai.cli import app


def test_deterministic_fixture_end_to_end(tmp_path: object) -> None:
    result = CliRunner().invoke(app, ["fixture-demo", "--snapshot-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "DETERMINISTIC FIXTURE ONLY" in result.output
    assert "features=58" in result.output
    assert "validation_folds=" in result.output
    assert "manual_executions=1" in result.output
    assert "No securities order was submitted" in result.output


def test_installed_console_entry_point_loads_without_source_path_fallback() -> None:
    executable_name = "stock-ai.exe" if os.name == "nt" else "stock-ai"
    executable = Path(sys.executable).with_name(executable_name)
    result = subprocess.run(
        [str(executable), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "never submits orders" in result.stdout
