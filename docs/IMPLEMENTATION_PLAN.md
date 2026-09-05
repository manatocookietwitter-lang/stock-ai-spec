# Implementation Plan

更新日: 2026-08-22  
状態: v0.2

## 方針

全体仕様はMDに保存し、CodexのGoalは段階別に実行する。

推奨:

- Goal 1: 研究基盤 + 判断エンジンMVP
- Goal 2: 高度な機械学習 + 前場AI
- Goal 3: 自動提案 + PWA + フォワード検証

各マイルストーンで:

1. 実装
2. テスト
3. 専門レビュー
4. `STATUS.md` 更新
5. Git checkpoint

を行う。

## M0 — Governance

成果物:

- AGENTS
- 設定
- ロギング
- テスト骨組み
- docs routing
- fixture conventions
- secrets policy

完了:

- 空の設定で起動できる
- fixture smoke test
- lint / typecheck / test commandsが固定
- real order機能が存在しない

## M1 — Data Foundation

成果物:

- J-Quants V2 daily adapter
- immutable raw store
- DuckDB catalog
- pagination / retry / rate limit
- schema validation
- capability table
- trading calendar
- security master
- daily prices and research-adjusted OHLCV
- TOPIX / sector market context
- shares-outstanding history where available
- financial summaries

完了:

- 同じ期間を再取得しても重複しない
- 途中失敗で前回有効データを壊さない
- secretがログへ出ない
- fixture integration test
- capability table distinguishes adjusted OHLCV, breadth, supply-demand, intraday, and detailed financial availability

## M2 — Point-in-time Dataset

成果物:

- historical universe
- available_at
- corporate action handling
- disclosure timing
- delisting handling
- research price / execution price separation
- fixed dataset snapshots

完了:

- leakage invariance tests
- future data mutationが過去へ影響しない
- current universeが過去へ入らない

## M3 — Labels and FeatureSet V0 / V1 Core

成果物:

- 1日 / 5日 / 20日labels
- market / sector / beta residual comparisons
- ranking relevance
- downside labels
- machine-readable Feature Registry
- Feature Set Manifest and hashes
- `docs/FEATURE_CATALOG.md`のFeatureSet V0
- `docs/FEATURE_CATALOG.md`のFeatureSet V1 Core
- price / volume / technical / basic financial / relative / market features
- SMA・主要cross、MACD、RSI、Bollinger、ADX・DI、ATR・NATR
- daily cross-sectional / sector / size ranks
- missingness and warm-up handling

完了:

- feature versioning and definition hashes
- label overlap tests
- fixture dataset reproducibility
- known-value numerical tests for core indicators
- future-price mutation does not change prior features
- current-day close cannot enter the 11:30 daily feature set
- corporate-action adjusted OHLCV tests
- no silent zero-fill for warm-up or missing data
- V1 Core size and enabled definitions are explicit in a manifest

## M4 — Baselines

成果物:

- Cash / HOLD
- Momentum
- rule score
- Ridge
- Elastic Net
- Logistic
- one bounded LightGBM baseline
- V0 vs V1 Core ablation

完了:

- purged expanding walk-forward
- fold reports
- locked holdout enforcement
- 10 / 20 / 30 / 50 bps scenarios

## M5 — Experiment Registry

成果物:

- experiment IDs
- config / seed / commit / data snapshot tracking
- feature definition / feature set / preprocessing version tracking
- adoption / rejection reason
- report generation

完了:

- failed experiments preserved
- identical config reproducible
- accidental holdout tuning blocked

## M6 — GBDT and Feature Family Research

成果物:

- LightGBM regression / ranking
- XGBoost regression / ranking
- CatBoost regression / ranking
- bounded tuning
- multiple seeds
- `FEATURE_CATALOG.md` F1〜F12 family ablations
- FeatureSet V2 Extended Technical
- fold-local correlation pruning and preprocessing
- OOS permutation / SHAP diagnostics
- feature stability and missingness reports

完了:

- baseline and V1 Core comparison
- each feature family evaluated as an incremental experiment
- all-feature-at-once result is not the sole adoption basis
- preprocessing and feature selection fit only on training folds
- stability reports across folds and seeds
- no model or feature adopted solely on in-sample score or importance
- library implementation/version is recorded and numerical regression tests pass
- 5日・20日は3 seedのfull F1〜F12 ablationを行う。1日はdevelopment-only軽量screenを先行し、事前固定した安定性・非重複性・増分OOF・cost・turnover・downside gateをすべて通過した場合だけfull ablationへ戻す

## M7 — Multi-horizon and Downside

成果物:

- separate 1 / 5 / 20 day models
- quantile models
- large-loss classifier
- calibration
- uncertainty estimates

完了:

- Decision Engine-compatible outputs
- uncertainty error analysis
- OOF predictions stored
- 20日を中期の主alpha、5日を短期補助alpha、screen通過時の1日を売買タイミング・短期リスク補助Challengerとして評価する

## M8 — Morning AI

成果物:

- morning data contract
- current holdings always included
- Feature Catalog Morning Core features
- optional Morning Microstructure features when capability exists
- time-of-day volume profile without daily-volume guessing
- forecast revision model
- meta-label comparison
- holdings and candidates evaluated together

完了:

- no post-11:30 input
- current-day close cannot enter morning inference
- unavailable intraday history is marked BLOCKED rather than inferred
- HOLD→REDUCE / SELL and SKIP→BUY can emerge through Decision Engine
- model adds OOS value or is rejected

## M9 — Regime and Ensemble

成果物:

- market regime features
- optional soft gating
- OOF non-negative stacking
- diversity / correlation report
- disagreement penalty

完了:

- stacking trained only on OOF
- stable against seed and period changes

## M10 — Neural and TDnet Challengers

成果物:

- small MLP
- TCN / 1D-CNN
- GRU
- small Transformer
- FeatureSet V3 Data-dependent experiments
- supply-demand and detailed-financial adapters where capability exists
- optional TDnet structured adapter

完了:

- disabled by default unless adopted
- every V3 experiment declares required data capabilities
- no production dependency on LLM
- inference meets timing
- accepted only with robust OOS improvement

## M11 — Stateful Portfolio Simulator

成果物:

- cash
- positions by account bucket
- manual-like execution assumptions
- next-day carry
- 100-share lots
- current portfolio counterfactual

完了:

- repeated simulation deterministic
- current state carried correctly
- HOLD baseline available every day

## M12 — Transaction Cost Engine

成果物:

- configurable commission
- spread
- slippage
- market impact
- SBI zero-commission policy as user setting
- cost model versioning

完了:

- no hardcoded free commission
- cost ablations
- realized execution data can update estimates later

## M13 — Tax and Account Engine

成果物:

- NISA / taxable account buckets
- acquisition price
- unrealized P/L
- YTD realized P/L
- user-supplied loss carryforward
- versioned tax policy
- estimated immediate tax effect
- NISA inputs

完了:

- same symbol in multiple accounts handled
- tax-free vs taxable sale differs
- missing tax inputs produce warnings or conservative behavior
- output labeled as estimate

## M14 — Daily Portfolio Decision Engine

中核。

成果物:

- full-portfolio optimization
- HOLD counterfactual
- replacement gain
- risk / downside / uncertainty
- cost / tax
- no-trade zone
- cash allocation
- continuous and 100-share targets
- BUY / HOLD / REDUCE / SELL / SKIP

完了:

- pairwise swapsだけでない
- current holdings always evaluated
- no-trade and all-cash valid
- 100-share constraints
- available cash never exceeded
- proposal explains HOLD comparison

## M15 — Proposal and Manual Execution

成果物:

- proposal schema
- Japanese human-readable reasons
- manual execution entry
- next-day position update
- import-ready interfaces

完了:

- current→recommended→difference shown
- actual execution can differ from proposal
- next day starts from actual state

## M16 — Automation

成果物:

- idempotent jobs
- data sync
- candidate selection
- morning capture
- 11:30 freeze
- prediction
- proposal
- notification
- EOD update
- monthly challenger training
- process locks
- Windows Task Scheduler registration

完了:

- restart-safe
- stale / missing data fail closed
- last proposal not silently reused
- no order submission

## M17 — Mobile PWA

正本: `docs/UI_SPEC.md`

成果物:

- 共通レイアウトと5項目の下部ナビ
- Home
- Today
- Decision Review
- Execution Record
- Ranking
- Stock Detail
- Validation
- Settings
- loading / empty / stale / error states
- mobile-first responsive design

完了:

- share-centered display
- BUY / HOLD / REDUCE / SELL / SKIP definitions are exact
- AI proposal, user decision, and actual execution are stored separately
- HOLD counterfactual visible
- cost / tax / uncertainty visible
- proposal stop reason visible
- data timestamp and model version visible
- ranking and final portfolio Action are not conflated
- same symbol across multiple account buckets works
- no order button or API that submits a trade
- the UI acceptance criteria in `docs/UI_SPEC.md` pass

## M18 — Import and Reconciliation

成果物:

- CSV mapping layer
- statement / execution import-ready adapters
- differences review
- account state reconciliation

完了:

- imports never overwrite silently
- manual confirmation for conflicts
- actual state remains source for next day

## M19 — Forward Paper Decision Validation

成果物:

- every daily proposal archived before outcome
- real future performance
- model drift
- cost estimate error
- tax estimate audit
- weekly / monthly readout

完了:

- predictions immutable after timestamp
- no hindsight changes
- champion / challenger reports
- minimum observation count tracked

## M20 — Hardening

成果物:

- failure recovery
- backup / restore
- data corruption tests
- security review
- quantitative review
- leakage review
- UI review
- user setup docs

完了:

- full deterministic E2E
- full test suite
- no real-order code path
- limitations documented
- credentials and paid-data blockers listed

## Goal 1の推奨範囲

M0〜M5、M11〜M15。M3ではFeatureSet V0とV1 Coreまでを対象にする。

これで、fixtureを使って:

```text
データ
→ baseline予測
→ current portfolio
→ cost / tax
→ Daily Portfolio Decision Engine
→ 100株提案
→ manual execution
→ next day
```

が通る。

## Goal 2の推奨範囲

M6〜M10。

FeatureSet V2 / V3、特徴量群Ablation、高度なモデルを追加し、Decision Engineへの予測入力を改善する。

## Goal 3の推奨範囲

M16〜M20。

自動提案、PWA、実取引入力、フォワード検証、運用強化。

## プロジェクト完了の定義

コード完成だけでは利益性能は確定しない。

実装完了:

- 仕様に沿う
- テストが通る
- 再現可能
- 漏洩がない
- 提案が自動生成される
- スマホで確認できる
- 実約定を反映できる
- 実注文しない

研究完了:

- baselineと複雑モデルの結果が記録される
- 負の結果も残る
- 最終holdoutを不正利用しない
- forward validationへ移れる

実用性評価:

- 将来のフォワード観測が必要
- 一定期間の実提案と実績を比較
- 取引コスト・税・実装誤差を検証
