# Goal 5 Local Decision-Support Runbook

更新日: 2026-08-24

## 1. Safety boundary

Goal 5 is a local decision-support application. It records an AI proposal, the user's
decision, and actual fills as three different immutable records. It has no broker login,
order submission, order amendment, or order cancellation route. The application binds
only to `127.0.0.1`.

The fixture workflow is named `DETERMINISTIC_FIXTURE_ONLY`. It is never used as a fallback
when live data, a Morning provider, or an approved model is unavailable. Fixture results
must not be interpreted as profitability evidence.

## 2. Install and build

Use Python 3.12 and Node.js. On Windows paths containing non-ASCII characters, use the
non-editable Python install.

```text
python -m uv sync --all-groups --no-editable
cd web
npm ci
npm run lint
npm run typecheck
npm test
npm run build
npm run test:e2e
cd ..
```

The deterministic E2E uses the installed local Microsoft Edge and a disposable fixture
ledger. It traverses Today → changed user decision → partial fill → apply executions →
next Home against the real local FastAPI surface. It never contacts a broker.

The API serves only the already-built `web/dist` directory. Rebuild it after every PWA
change. The service worker caches only the static shell; `/api/` responses are never cached.

## 3. Explicit local fixture smoke test

```text
stock-ai ops fixture-bootstrap \
  --database .demo-artifacts/goal5/operations.sqlite3 \
  --as-of 2026-08-24T11:30:00+09:00

stock-ai ops serve \
  --database .demo-artifacts/goal5/operations.sqlite3 \
  --static-dir web/dist \
  --port 8765
```

Open `http://127.0.0.1:8765/`. The UI must display the fixture warning. Saving a decision
records user intent only. Record the result actually obtained at the broker on the
execution screen; only those fills can create the next portfolio state.

After all manual results have been recorded, use the Execution screen's
`記録済み約定を実保有へ反映` action, or the equivalent CLI command:

```text
stock-ai ops apply-executions \
  --database data/operations/stock-ai.sqlite3 \
  --next-as-of 2026-08-25T11:30:00+09:00
```

This creates a new actual Portfolio from recorded fills only. It does not contact a
broker and cannot submit an order.

## 4. Live capability gate

```text
stock-ai ops capabilities
stock-ai ops run-daily --database data/operations/stock-ai.sqlite3 --business-date today
```

`run-daily` durably writes `RUNNING` before invoking a configured handler. The handler
receives a stable logical-stage idempotency key and a lock-heartbeat callback. A success
transition is committed only while the same process still owns an unexpired lock. Freshness
timestamps and proposal lineage survive blocked/failed status and successful-stage reuse.
A separately scheduled stage requires
the same workflow version's preceding stage to have succeeded. This repository deliberately ships
without invented live handlers. Until J-Quants history, a chosen Morning provider, an
approved model-registry entry, account state, and calibrated policies are available, the
job records `BLOCKED_BY_DATA_CAPABILITY` and exits non-zero. It does not reuse yesterday's
proposal and does not fall back to fixture data.

The proposal handler must return both the proposal and the exact `DecisionPolicySnapshot`
derived from the `DecisionEngineConfig` used to generate it. Proposal payload, archive evidence,
and policy snapshot are published in one transaction. A missing or mismatched policy fails the
stage; `ops verify` also rejects any proposal whose policy snapshot is absent. The Settings screen
reads limits from this archived policy instead of displaying inferred defaults.

To inspect—but not automatically install—the Task Scheduler definitions:

```text
stock-ai ops scheduler-script \
  --database C:\absolute\path\to\stock-ai.sqlite3 \
  --executable C:\absolute\path\to\stock-ai.exe
```

Review the printed PowerShell before running it manually. It contains the daily data,
candidate, Morning, 11:30 freeze, prediction, proposal, notification and EOD stages plus
monthly challenger research. A configured handler remains required for every stage.

## 5. CSV preview and reconciliation

All CSV imports are preview-first. Conflicts require an explicit confirmation and an
existing immutable record is never overwritten.

Execution CSV columns:

```text
execution_id,decision_id,executed_at,symbol,account_bucket_id,status,side,ordered_shares,filled_shares,average_fill_price,actual_commission,actual_other_cost,tax_withheld
```

Account-state CSV columns:

```text
as_of,record_type,symbol,account_bucket_id,shares,average_acquisition_price,market_price,available_cash,reserved_cash
```

Use one `POSITION` row for each non-zero holding and exactly one `CASH` row for every
account bucket. Position-only reconciliation is rejected because it would silently keep a
stale cash balance. Both available and reserved cash are compared; neither is silently
carried forward. All rows must have one timezone-aware `as_of` later than the current
state. The normalized schema is stable, but the mapping from an SBI or other broker's real
statement remains blocked until a representative file is supplied.

## 6. Paper observations

Every Paper outcome refers to the final archived proposal for one business date. Before any
outcome can be appended, register the verified JPX calendar as an immutable, content-addressed
`PaperCalendarSnapshot` in the same ledger. The calendar must have been frozen before proposal
archival. The store derives the exact next 1 / 5 / 20 sessions from that authenticated calendar
and rejects caller-selected gaps. It also requires proposal archival strictly before the label
endpoint, an actual label-availability time, and an observation no earlier than availability.
Its Champion version must match the archived proposal.
It stores source snapshot IDs, proposal and benchmark return, estimated versus actual cost,
estimated versus audited tax effect, and Champion/Challenger absolute prediction error.
Records are immutable by `outcome_id`; later knowledge cannot rewrite an earlier result.

The Validation performance curve and compounded weekly/monthly readouts use only the
non-overlapping one-session operational return series. Longer forecast horizons may be
audited separately but are not compounded as if they were independent daily P&L. The screen
also exposes a minimum-observation counter, Champion/Challenger error, and a time-ordered
drift check. Model error and drift use only the active exact Champion cohort. Challenger
statistics use only the contiguous exact-version suffix ending at the latest observation; a
version transition or missing Challenger starts a new/empty cohort. The drift check compares adjacent historical error windows;
it never promotes a model automatically and is not evidence of profitability.

## 7. Verify, back up, and restore

Verify SQLite integrity, foreign keys, every immutable content hash, and every
query-driving catalog/identity column:

```text
stock-ai ops verify --database data/operations/stock-ai.sqlite3
```

Create each backup at a new path. Publication uses an atomic no-replace link after the
temporary SQLite copy is verified, so a path concurrently claimed by another process is
not overwritten:

```text
stock-ai ops backup \
  --database data/operations/stock-ai.sqlite3 \
  --destination backups/stock-ai-2026-08-24.sqlite3
```

Restore only while the PWA and scheduled jobs are stopped. The source backup is opened
read-only and verified before copying; the restored database is verified again. Replacement
requires the explicit flag:

```text
stock-ai ops restore \
  --backup backups/stock-ai-2026-08-24.sqlite3 \
  --database data/operations/stock-ai.sqlite3 \
  --confirm-replace
```

If verification fails, keep the damaged file for investigation, restore the latest verified
backup to a separate path first, and do not generate or display a proposal from the damaged
ledger.

## 8. Credentials and logs

`JQUANTS_API_KEY` is read only from the process environment by the J-Quants V2 client. The
PWA Settings API returns only a configured/not-configured boolean. Never paste the value
into CSV, SQLite metadata, Markdown, screenshots, logs, exceptions, fixtures, or Git.
Automation persists stable reason codes rather than provider exception text. The HTTP layer
accepts only localhost Host/Origin values; do not expose the service through a remote proxy.

## 9. Public hosted companion

`hosted/` is a separate companion for checking readiness and the system's safety stop reasons
from a remote URL. It is not a remote copy of the operational ledger. The user approved public
read access on 2026-08-29. Persisting the user-specific safety acknowledgement still requires
ChatGPT sign-in and server-side identity validation.

The hosted worker may persist only minimal user-scoped operating metadata such as acknowledgement
of the safety boundary. Do not upload `JQUANTS_API_KEY`, `data/live`, Production Dataset or model
artifacts, broker CSVs, portfolio records, NISA/tax state, proposals, decisions, or executions.
Until an authenticated and approved sync contract exists, the hosted UI must remain `NO PROPOSAL`
and show the missing capabilities. It must not substitute fixtures, yesterday's proposal, or
synthetic market/account values. Do not expose the local FastAPI service through a reverse proxy
or invent a tunnel to make the hosted worker reach localhost.

Production URL:

```text
https://stock-ai-decision-support.manato0618.chatgpt.site
```

Before treating a release as current, verify the Sites access policy is `public`, the live URL
returns HTTP 200, the page remains `NO PROPOSAL` while required live capabilities are absent,
the Open Graph and X image URLs use the exact production origin, and the D1 `operator_settings`
table exists. Never expose or print provider credentials while performing these checks.

Local verification for the companion:

```text
cd hosted
npm run lint
npm run build
npm audit --omit=dev
npm audit
```
