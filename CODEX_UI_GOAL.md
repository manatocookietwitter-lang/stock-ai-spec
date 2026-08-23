# Codex Goal — Mobile PWA UI

```text
/goal

Read AGENTS.md, README.md, docs/MASTER_SPEC.md, docs/DATA_CONTRACT.md,
docs/DAILY_PORTFOLIO_DECISION_ENGINE.md, docs/ML_RESEARCH_SPEC.md,
docs/FEATURE_CATALOG.md, docs/UI_SPEC.md,
docs/IMPLEMENTATION_PLAN.md, docs/DECISIONS.md, and docs/STATUS.md before editing.

Implement M17 Mobile PWA only. Treat docs/UI_SPEC.md as the complete source of truth for routes, screen roles, actions, interactions, states, data fields, visual rules, tests, and acceptance criteria. Treat docs/ui-reference images as non-normative visual references. When text and images conflict, follow the text specification.

Required routes and screens:

- Home
- Today
- Decision Review
- Execution Record
- Ranking
- Stock Detail
- Validation
- Settings and its defined subpages

Non-negotiable behavior:

- This is decision support only. Do not submit, modify, or cancel any securities order.
- BUY / HOLD / REDUCE / SELL / SKIP must follow the exact share-count definitions in docs/UI_SPEC.md.
- A held position with target shares equal to zero is SELL, never REDUCE.
- Display current shares, recommended shares, and share difference as the primary proposal information.
- Keep AI proposals, user decisions, actual executions, and Paper results as separate immutable or versioned records.
- Saving a user decision must not contact a broker or change actual holdings.
- Actual holdings change only from recorded executions or an approved import/reconciliation path.
- Ranking position and final portfolio Action must be presented as separate concepts.
- Support the same symbol in multiple account buckets.
- Never reuse a stale proposal as today's proposal.
- Fail closed for stale, missing, inconsistent, or incomplete data.
- Never generate unsupported investment reasons.
- No browser-side market-data or credential secrets.

Implementation approach:

1. Inspect the existing frontend and backend contracts.
2. If a required backend endpoint is not yet implemented, add a typed fixture-backed development adapter without creating fake production fallback behavior.
3. Build shared layout, navigation, formatting, action badges, data freshness, and status components first.
4. Complete the Today → Decision Review → Execution Record flow before implementing the remaining screens.
5. Then implement Home, Stock Detail, Ranking, Validation, and Settings.
6. Add loading, empty, stale, error, holiday, and proposal-generation states.
7. Add responsive behavior and accessibility.
8. Run an independent UI/spec review and fix clear mismatches.

Required verification:

- frontend lint
- TypeScript typecheck
- component tests
- route/screen tests
- production build
- deterministic E2E:
  proposal ready → Today → alter one user decision → save decision → record a partial fill → next-day Home reflects only the recorded execution
- a safety test proving that saving a decision does not invoke any broker/order endpoint
- tests for multiple account buckets, insufficient cash, existing holdings above the configured maximum, stale data, no proposal, and SELL versus REDUCE classification

Do not redesign the product into an AI chat, generic stock screener, or automatic trading app. Do not implement unrelated machine-learning research in this Goal.

Update docs/STATUS.md with completed routes, tests, known limitations, and screenshots or visual-review notes. Update docs/DECISIONS.md only when a genuine specification decision is required. Stop at the M17 boundary and provide a factual completion report.
```
