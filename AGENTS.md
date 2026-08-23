# AGENTS.md

## Mission

Build a research-grade Japanese-equity decision-support system.

The system generates a proposal every trading day using information available by 11:30 JST. It never places, modifies, or cancels a real securities order. The user makes the final decision and manually submits any order.

## Source of truth

Before editing, read the relevant files:

- Always: `README.md`, `docs/MASTER_SPEC.md`, `docs/STATUS.md`
- Data or ingestion: `docs/DATA_CONTRACT.md`
- Machine learning or validation: `docs/ML_RESEARCH_SPEC.md`
- Feature definitions or indicators: `docs/FEATURE_CATALOG.md`
- Portfolio, cost, tax, or recommendations: `docs/DAILY_PORTFOLIO_DECISION_ENGINE.md`
- UI, routes, interactions, or frontend states: `docs/UI_SPEC.md`
- Milestone work: `docs/IMPLEMENTATION_PLAN.md`
- Ambiguities and changes: `docs/DECISIONS.md`

Priority:

1. Latest explicit user instruction
2. Confirmed decisions in `docs/DECISIONS.md`
3. `docs/MASTER_SPEC.md`
4. Specialist documents
5. Implementation plan
6. Status notes

Do not silently resolve contradictions. Record them in `docs/DECISIONS.md`.

## Non-negotiable product rules

- Decision support only; no real-order submission.
- Freeze alpha inputs at 11:30 JST.
- Evaluate current holdings, eligible new candidates, and cash together.
- Final actions are `BUY`, `HOLD`, `REDUCE`, `SELL`, and `SKIP`.
- Final recommendations are share-quantity centered.
- Initial trading unit is 100 shares.
- Compare every alternative against keeping the current portfolio unchanged.
- Include commission, spread, slippage, market impact, and estimated tax effects.
- Track positions by symbol and account bucket.
- The user manually records or imports actual executions.
- The system may recommend no trades or all cash.

## Non-negotiable research rules

- J-Quants V2 only for the initial official daily-data adapter.
- Every feature must satisfy `available_at <= as_of`.
- Preserve point-in-time universe membership.
- Never use current membership for historical backtests.
- Never use later corrections in earlier predictions.
- Never use random train/test splitting.
- Use purged expanding walk-forward validation and a locked final holdout.
- Keep every experiment, including failed experiments.
- Treat `docs/FEATURE_CATALOG.md` as a candidate pool, not a mandate to enable every feature.
- Implement FeatureSet V0 / V1 / V2 / V3 in stages.
- Never convert RSI, MACD, moving-average crosses, or other indicators directly into final trade Actions.
- Fit clipping, ranking thresholds, correlation pruning, and feature selection inside each training fold only.
- Version every feature formula, parameter set, warm-up rule, implementation, and library version.
- Compare complex models with simple rules and linear baselines.
- A complex model is not adopted merely because its in-sample score is higher.
- A valid negative result is acceptable.
- Never generate or silently use fake production market data.
- Deterministic fixtures are allowed only in tests.

## Engineering defaults

- Python 3.12
- `uv` for Python dependencies
- immutable Parquet for raw research data
- DuckDB for analytical queries
- SQLite WAL for operational state
- FastAPI backend
- React + TypeScript mobile-first PWA
- secrets from environment variables or an OS credential store
- no browser-side provider secrets

Avoid unnecessary file splitting and speculative abstractions.

## Workflow

- The main agent edits files.
- Specialist subagents are read-only reviewers unless explicitly assigned otherwise.
- Update `docs/STATUS.md` after every milestone.
- Update `docs/DECISIONS.md` whenever a product or technical decision changes.
- Create recoverable Git checkpoints after a milestone passes.
- Do not weaken or delete tests to make a milestone pass.
- Stop at the milestone boundary requested by the active Goal.

## Required verification

Use the commands that exist for the current milestone. The target full set is:

```text
uv run ruff check .
uv run mypy src
uv run pytest
uv run pytest --cov
npm run lint
npm run typecheck
npm run test
npm run build
```

The core deterministic end-to-end path is:

```text
fixture ingest
→ point-in-time dataset
→ FeatureSet manifest and indicator numerical checks
→ train
→ validate
→ predict
→ cost/tax estimates
→ stateful current portfolio
→ Daily Portfolio Decision Engine
→ 100-share proposal
→ manual execution record
→ next-day state
```

## Completion report

At the end of each Goal, report:

1. changed files
2. implemented behavior
3. commands executed
4. test results
5. experiment results
6. remaining limitations
7. blocked real-data checks
8. confirmation that no real order was placed
