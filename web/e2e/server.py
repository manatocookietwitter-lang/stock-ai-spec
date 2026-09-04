"""Disposable source-tree server for the Goal 5 browser E2E gate."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo

import uvicorn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from stock_ai.operations import (  # noqa: E402
    OperationalStore,
    bootstrap_goal5_fixture,
    create_app,
)


def main() -> None:
    with TemporaryDirectory(prefix="stock-ai-goal5-e2e-") as directory:
        database = Path(directory) / "operations.sqlite3"
        store = OperationalStore(database)
        jst = ZoneInfo("Asia/Tokyo")
        as_of = datetime(2026, 1, 8, 11, 30, tzinfo=jst)
        bootstrap_goal5_fixture(store, as_of=as_of)
        test_clock = datetime(2026, 1, 8, 12, 0, tzinfo=jst)
        uvicorn.run(
            create_app(
                database,
                static_dir=ROOT / "web" / "dist",
                clock=lambda: test_clock,
            ),
            host="127.0.0.1",
            port=8766,
            log_level="warning",
        )


if __name__ == "__main__":
    main()
