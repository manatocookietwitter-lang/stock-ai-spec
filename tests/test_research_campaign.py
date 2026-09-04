from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from stock_ai.cli import _advanced_campaign_child_command, app
from stock_ai.ml.campaign import (
    CampaignBatchStatus,
    authenticate_batch_artifact,
    create_campaign_manifest,
    discover_batch_artifact,
    load_campaign_build_id,
    load_campaign_manifest,
    reconcile_campaign,
    write_campaign_manifest,
)


def _manifest(tmp_path: Path):  # type: ignore[no-untyped-def]
    return create_campaign_manifest(
        build_id="b" * 64,
        build_manifest_path=tmp_path / "build.json",
        code_commit="abc1234",
        report_root=tmp_path / "reports",
        experiment_registry=tmp_path / "experiments.jsonl",
        horizons=(1, 5),
        model_families=("lightgbm", "xgboost"),
        common_config={
            "target_family": "return",
            "seeds": (17,),
            "tuning_trials": 1,
            "tuning_timeout_seconds": 10,
            "estimator_count": 5,
            "initial_train_periods": 20,
            "validation_periods": 5,
            "step_periods": 5,
            "holdout_periods": 5,
            "run_ablations": False,
            "run_diagnostics": False,
            "max_materialized_oof_rows": 10_000,
            "max_model_fits": 100,
            "feature_names": ("return_1d",),
        },
        now=datetime(2026, 9, 4, tzinfo=UTC),
    )


def test_campaign_manifest_is_atomic_round_trip_and_plan_authenticated(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    assert [batch.batch_id for batch in manifest.batches] == [
        "h1-lightgbm",
        "h1-xgboost",
        "h5-lightgbm",
        "h5-xgboost",
    ]
    assert all(batch.status is CampaignBatchStatus.PENDING for batch in manifest.batches)
    assert all(len(batch.feature_names_hash) == 64 for batch in manifest.batches)

    path = tmp_path / "campaign.json"
    write_campaign_manifest(manifest, path)
    loaded = load_campaign_manifest(path)
    assert loaded.campaign_id == manifest.campaign_id
    assert not list(tmp_path.glob(".campaign.json.*"))

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["common_config"]["estimator_count"] = 99
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        load_campaign_manifest(path)


def test_campaign_manifest_missing_or_invalid_is_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="missing or invalid"):
        load_campaign_manifest(tmp_path / "missing.json")
    invalid = tmp_path / "invalid.json"
    invalid.write_text("not-json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing or invalid"):
        load_campaign_manifest(invalid)


def test_campaign_build_marker_is_authenticated_without_loading_parquet(tmp_path: Path) -> None:
    build_id = "b" * 64
    directory = tmp_path / build_id
    directory.mkdir()
    path = (directory / f"{build_id}.json").resolve()
    payload = {"build_id": build_id, "manifest_path": str(path), "dataset": "metadata-only"}
    payload["metadata_hash"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_campaign_build_id(path) == build_id

    payload["dataset"] = "tampered"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="metadata hash mismatch"):
        load_campaign_build_id(path)


def test_reconcile_marks_stale_running_batch_interrupted(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    batch = manifest.batches[0]
    batch.status = CampaignBatchStatus.RUNNING
    batch.child_pid = 123

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr("stock_ai.ml.campaign._process_is_running", lambda _pid: False)
        reconcile_campaign(manifest)

    assert batch.status is CampaignBatchStatus.INTERRUPTED
    assert batch.child_pid is None
    assert "stopped" in str(batch.last_error)


def test_reconcile_leaves_live_child_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    batch = manifest.batches[0]
    batch.status = CampaignBatchStatus.RUNNING
    batch.child_pid = 123
    monkeypatch.setattr("stock_ai.ml.campaign._process_is_running", lambda _pid: True)

    reconcile_campaign(manifest)

    assert batch.status is CampaignBatchStatus.RUNNING
    assert batch.child_pid == 123


def test_reconcile_authenticates_existing_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    batch = manifest.batches[0]
    oof = tmp_path / "reports" / ("r" * 64) / f"{'r' * 64}.oof.parquet"
    oof.parent.mkdir(parents=True)
    oof.touch()
    oof.with_name(f"{'r' * 64}.json").write_text(
        json.dumps(
            {
                "report": {
                    "config_hash": batch.config_hash,
                    "code_commit": manifest.code_commit,
                }
            }
        ),
        encoding="utf-8",
    )
    report = SimpleNamespace(
        report_id="r" * 64,
        config_hash=batch.config_hash,
        code_commit=manifest.code_commit,
        config=SimpleNamespace(horizons=(1,), model_families=("lightgbm",)),
        feature_names=("return_1d",),
    )
    monkeypatch.setattr(
        "stock_ai.ml.campaign.load_advanced_research_run", lambda _path: (report, object())
    )

    reconcile_campaign(manifest)

    assert batch.status is CampaignBatchStatus.SUCCEEDED
    assert batch.report_id == "r" * 64
    assert batch.oof_path == str(oof.resolve())


def test_batch_authentication_rejects_provenance_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    batch = manifest.batches[0]
    report = SimpleNamespace(
        report_id="r" * 64,
        config_hash="wrong",
        code_commit=manifest.code_commit,
        config=SimpleNamespace(horizons=(1,), model_families=("lightgbm",)),
        feature_names=("return_1d",),
    )
    monkeypatch.setattr(
        "stock_ai.ml.campaign.load_advanced_research_run", lambda _path: (report, object())
    )
    with pytest.raises(RuntimeError, match="config hash mismatch"):
        authenticate_batch_artifact(batch, code_commit=manifest.code_commit, oof_path=tmp_path)
    assert (
        discover_batch_artifact(
            batch, code_commit=manifest.code_commit, report_root=tmp_path / "none"
        )
        is None
    )


def test_reconcile_rejects_missing_previously_succeeded_result(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    manifest.batches[0].status = CampaignBatchStatus.SUCCEEDED
    with pytest.raises(RuntimeError, match="authenticated artifact is missing"):
        reconcile_campaign(manifest)


def test_campaign_child_command_contains_no_credentials(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    batch = manifest.batches[0]
    command = _advanced_campaign_child_command(
        build_manifest=Path(manifest.build_manifest_path),
        code_commit=manifest.code_commit,
        report_root=Path(manifest.report_root),
        experiment_registry=Path(manifest.experiment_registry),
        horizon=batch.horizon,
        model_family=batch.model_family,
        common_config=manifest.common_config,
    )
    joined = " ".join(command)
    assert "JQUANTS_API_KEY" not in joined
    assert "--no-run-ablations" in command
    assert "--no-run-diagnostics" in command
    assert command[1] == "-c"
    assert "from stock_ai.cli import app; app()" in command[2]


class _CompletedProcess:
    last_environment: dict[str, str] | None = None

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.pid = 1234
        environment = _kwargs.get("env")
        if isinstance(environment, dict):
            type(self).last_environment = environment

    def wait(self, timeout: int | None = None) -> int:
        del timeout
        return 0

    def terminate(self) -> None:
        return None


def test_campaign_cli_runs_batches_and_persists_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_manifest = tmp_path / "build.json"
    build_manifest.touch()
    monkeypatch.setattr("stock_ai.cli.load_campaign_build_id", lambda _path: "b" * 64)
    monkeypatch.setattr("stock_ai.cli.subprocess.Popen", _CompletedProcess)
    monkeypatch.setenv("JQUANTS_API_KEY", "must-not-reach-research-child")

    def succeed(manifest, *, batch_ids=None):  # type: ignore[no-untyped-def]
        for batch in manifest.batches:
            if batch_ids is not None and batch.batch_id not in batch_ids:
                continue
            if batch.status is CampaignBatchStatus.RUNNING:
                batch.status = CampaignBatchStatus.SUCCEEDED
                batch.report_id = "r" * 64
                batch.oof_path = str(tmp_path / f"{batch.batch_id}.parquet")
        return manifest

    monkeypatch.setattr("stock_ai.cli.reconcile_campaign", succeed)
    campaign_path = tmp_path / "campaign.json"
    result = CliRunner().invoke(
        app,
        [
            "research",
            "campaign",
            "--build-manifest",
            str(build_manifest),
            "--code-commit",
            "abc1234",
            "--horizons",
            "1,5",
            "--model-families",
            "lightgbm",
            "--campaign-manifest",
            str(campaign_path),
            "--report-root",
            str(tmp_path / "reports"),
            "--experiment-registry",
            str(tmp_path / "experiments.jsonl"),
            "--log-root",
            str(tmp_path / "logs"),
        ],
    )
    assert result.exit_code == 0
    assert "status=SUCCEEDED batches=2" in result.stdout
    loaded = load_campaign_manifest(campaign_path)
    assert all(batch.status is CampaignBatchStatus.SUCCEEDED for batch in loaded.batches)
    assert _CompletedProcess.last_environment is not None
    assert "JQUANTS_API_KEY" not in _CompletedProcess.last_environment


def test_campaign_cli_rejects_manifest_for_different_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_manifest = tmp_path / "build.json"
    build_manifest.touch()
    monkeypatch.setattr("stock_ai.cli.load_campaign_build_id", lambda _path: "b" * 64)
    campaign_path = tmp_path / "campaign.json"
    write_campaign_manifest(_manifest(tmp_path), campaign_path)
    result = CliRunner().invoke(
        app,
        [
            "research",
            "campaign",
            "--build-manifest",
            str(build_manifest),
            "--code-commit",
            "different",
            "--campaign-manifest",
            str(campaign_path),
        ],
    )
    assert result.exit_code == 2
    assert "plan differs" in result.stderr


def test_campaign_cli_rejects_feature_outside_authenticated_v2(tmp_path: Path) -> None:
    build_manifest = tmp_path / "build.json"
    build_manifest.touch()
    result = CliRunner().invoke(
        app,
        [
            "research",
            "campaign",
            "--build-manifest",
            str(build_manifest),
            "--code-commit",
            "abc1234",
            "--feature-names",
            "definitely_not_a_v2_feature",
            "--campaign-manifest",
            str(tmp_path / "campaign.json"),
        ],
    )
    assert result.exit_code == 2
    assert "outside authenticated V2" in result.stderr
    assert not (tmp_path / "campaign.json").exists()
