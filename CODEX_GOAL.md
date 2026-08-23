# Codex Goal Templates

全仕様を1つのGoal本文へコピーしません。各Goalは、リポジトリ内の仕様を読み、指定された段階だけを完了させます。

---

## Goal 1 — 研究基盤と判断エンジンMVP

```text
/goal

Read AGENTS.md, README.md, docs/MASTER_SPEC.md, docs/DATA_CONTRACT.md,
docs/ML_RESEARCH_SPEC.md, docs/FEATURE_CATALOG.md,
docs/DAILY_PORTFOLIO_DECISION_ENGINE.md, docs/IMPLEMENTATION_PLAN.md,
docs/DECISIONS.md, and docs/STATUS.md before editing.

Implement the first usable research and decision-engine MVP described in docs/IMPLEMENTATION_PLAN.md. Complete M0 through M5, M11 through M15, using deterministic fixtures where real credentials are unavailable.

The required end state is:

- repository governance and configuration exist
- point-in-time data contracts and fixture ingestion work
- Feature Registry and Feature Set manifests work
- FeatureSet V0 and V1 Core from docs/FEATURE_CATALOG.md work
- known-value numerical tests for SMA/EMA, MACD, RSI, Bollinger, ADX/DI, ATR/NATR, OBV, and MFI work
- baseline features and labels work
- rule, momentum, Ridge, and one bounded LightGBM baseline work
- V0 versus V1 Core ablation is reported
- purged walk-forward validation works
- a stateful portfolio simulator carries cash and positions between days
- configurable commission, spread, slippage, market impact, and estimated tax effects work
- positions are tracked by symbol and account bucket
- the Daily Portfolio Decision Engine compares HOLD with alternative whole-portfolio targets
- it produces BUY/HOLD/REDUCE/SELL/SKIP proposals in 100-share lots
- it may recommend no trades or all cash
- actual user executions can be recorded manually
- the deterministic end-to-end fixture test passes

Do not build FeatureSet V2/V3, the final PWA, live morning-data integration, neural models, TDnet extraction, or any real-order capability in this Goal.

Keep docs/STATUS.md and docs/DECISIONS.md current. Stop at the Goal 1 boundary and provide a factual completion report.
```

---

## Goal 2 — 高度な機械学習と前場AI

```text
/goal

Read AGENTS.md, README.md, docs/MASTER_SPEC.md, docs/DATA_CONTRACT.md,
docs/ML_RESEARCH_SPEC.md, docs/FEATURE_CATALOG.md,
docs/DAILY_PORTFOLIO_DECISION_ENGINE.md, docs/IMPLEMENTATION_PLAN.md,
docs/DECISIONS.md, docs/STATUS.md, and the completed Goal 1 implementation.

Implement the advanced research scope in docs/ML_RESEARCH_SPEC.md and the relevant milestones in docs/IMPLEMENTATION_PLAN.md.

Add:

- FeatureSet V2 Extended Technical from docs/FEATURE_CATALOG.md
- FeatureSet V3 Data-dependent only where the capability table permits
- feature-family experiments F1 through F14
- fold-local correlation pruning, clipping, ranking, and feature selection
- feature-definition/version regression reports
- 1-day, 5-day, and 20-day models
- LightGBM, XGBoost, and CatBoost regression/ranking challengers
- downside quantile and large-loss models
- morning-session features for both current holdings and new candidates
- morning forecast revision and meta-label experiments
- market-regime features and soft gating
- out-of-fold non-negative stacking
- uncertainty and model-disagreement estimates
- feature ablations, multiple seeds, experiment registry, and stability reports
- bounded neural challengers only after the GBDT baselines are complete
- optional TDnet structured extraction behind a disabled-by-default adapter

Use only point-in-time data and purged expanding walk-forward validation. Never tune on the locked holdout. Do not enable every candidate feature at once or adopt a feature from SHAP/importance alone. Treat unavailable supply-demand, detailed-financial, or intraday inputs as BLOCKED_BY_DATA_CAPABILITY rather than inferred values. A negative result is acceptable. Adopt only models that improve multiple out-of-sample folds after realistic costs without unacceptable turnover or instability.

Integrate accepted predictions with the existing Daily Portfolio Decision Engine. Do not implement real-order submission. Update STATUS and DECISIONS, run the full research test suite, and stop at the Goal 2 boundary.
```

---

## Goal 3 — 自動提案・スマホPWA・フォワード検証

```text
/goal

Read AGENTS.md, README.md, docs/MASTER_SPEC.md, docs/DATA_CONTRACT.md,
docs/DAILY_PORTFOLIO_DECISION_ENGINE.md, docs/UI_SPEC.md,
docs/IMPLEMENTATION_PLAN.md, docs/DECISIONS.md, docs/STATUS.md,
and the completed Goal 1 and Goal 2 implementation.

Implement the automation and product scope in docs/MASTER_SPEC.md, docs/UI_SPEC.md, and docs/IMPLEMENTATION_PLAN.md.

Required end state:

- idempotent scheduled jobs for daily data sync, candidate selection, morning capture, 11:30 freeze, prediction, proposal generation, notification, and end-of-day state update
- fail-closed behavior for stale, missing, inconsistent, or incomplete data
- mobile-first PWA screens and routes defined in docs/UI_SPEC.md, including Home, Today, Decision Review, Execution Record, Ranking, Stock Detail, Validation, and Settings
- proposals centered on current shares, recommended shares, and share difference
- transparent HOLD counterfactual, estimated costs, estimated tax effect, downside, uncertainty, and net improvement
- strict separation of AI proposals, user decisions, actual executions, and Paper results
- manual execution entry and import-ready data interfaces
- monitoring, drift reports, notification hooks, and Windows Task Scheduler registration
- forward paper-decision logging without real-order execution
- deterministic full-stack end-to-end tests and production builds pass

The phone is a dashboard, not the compute host. Do not submit, modify, or cancel any securities order. Update STATUS and DECISIONS, run all checks, and stop with a factual completion report.
```

---

## 1つのGoalだけで進める場合

技術的には可能ですが、推奨しません。実施する場合もGoal本文は短くし、次だけを指定します。

```text
/goal

Treat AGENTS.md and all files under docs as the complete source of truth. Work through docs/IMPLEMENTATION_PLAN.md one milestone at a time, keep docs/STATUS.md and docs/DECISIONS.md current, verify every milestone before continuing, and never implement real-order submission. Continue until all currently implementable milestones pass their stated completion criteria. Stop when credentials, paid data, or future forward-performance history are the only remaining blockers.
```

大量の機能を一度に変更せず、必ずマイルストーン単位でコミット・検証してください。
