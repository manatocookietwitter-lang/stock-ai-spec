# Project Status

更新日: 2026-08-24
状態: `GOAL_1_CORE_MVP_IMPLEMENTED`

## Goal 1で実装したもの

### Repository / runtime

- Python 3.12 + `uv`、`src/` layout、strict mypy、Ruff、pytest、branch coverageを構成
- `.gitignore`と`.env.example`を追加し、credentialをsourceへ保存しない構成にした
- 非ASCIIのWindows pathでeditable `.pth`が壊れる場合の`--no-editable`手順をREADMEへ記録
- installed `stock-ai` console entry pointをtestで直接起動する

### Domain / state

- Security、Account、AccountBucket、Position、CashState、TaxState、MarketSnapshot、Prediction、PredictionUncertainty、TransactionCostEstimate、PortfolioState、TargetPosition、PortfolioProposal、ProposalAction、UserDecision、ExecutionRecordを不変型として実装
- 同一銘柄のNISA / 課税口座同時保有を`(symbol, account_bucket_id)`で分離
- AI提案、ユーザー判断、実約定、次状態を別recordとして保持
- 重複口座状態、矛盾したproposal/action/value、mixed model provenance、NaN/Inf、execution replay・時刻・status矛盾をfail closed
- 実約定費用を取得原価と実現損益へ反映し、税年境界でYTD状態をreset

### Point-in-time features

- `available_at <= as_of`を強制し、naive timestampと11:30時点の同日未完了barを拒否
- Feature Registry、FeatureSet V0（24特徴）/ V1 Core（58特徴）とdefinition / manifest hashを実装
- return、SMA/EMA、distance/slope、crossとcross後経過日数、MACD、RSI、Bollinger、ADX/+DI/-DI、ATR/NATR、volume、trading value、turnover、OBV、MFI、財務、TOPIX/sector relative、market volatility/breadthを実装
- sectorとbreadthをcandidate subsetから再計算せず、明示的PIT context入力から計算
- 必要capability、coverage、有限・正値条件が欠ける場合は`BLOCKED_BY_DATA_CAPABILITY`としてfail closed

### Baseline ML

- immutable Parquet + JSON dataset snapshot、feature manifest hash、1/5/20日target infrastructureを実装
- label endpointの`available_at`を保存し、snapshot cutoff時点で未成熟の将来labelをblankにする
- Goal 1 targetを「調整後終値間absolute returnのfixture研究proxy」と明記
- Momentum baseline、fold内imputation/scaling付きRidge baselineを実装
- expanding walk-forward、overlap purge、horizon以上のembargo、daily cross-sectional Rank IC、locked final holdoutを実装
- random split/shuffle APIを公開せず、0 validation foldは`BLOCKED_BY_VALIDATION`としてfail closed
- code commit/config/feature hashesを含むappend-only experiment recordを実装

### Cost / tax / Daily Portfolio Decision Engine

- commission、spread、slippage、market impactとhard ADV capを持つversioned Transaction Cost Engine
- NISA / 課税口座、源泉徴収方式、年内損益、繰越損失を扱う推定Tax Engine
- 推定税の経済効果、NISA機会費用、決済cash withholdingを別fieldに分離
- 同一bucketの複数売却を一括評価し、損失offset/refund capacityの重複利用を防止
- 現状HOLD反実仮想と全候補target portfolioを比較する上限付き離散optimizer
- available cash、minimum cash、symbol/sector、positions、turnover、liquidity、no-trade thresholdを実装
- 100株単位のtrade deltaを強制し、BUY / HOLD / REDUCE / SELL / SKIPを株数で出力
- policy effective date / bucket policy ID / market price / prediction provenanceを検証
- proposal ID、policy version、tax assumptions、current/recommended market value、人間向け理由を監査fieldとして保存
- 実注文送信機能は存在しない

### CLI

`stock-ai fixture-demo`で次を決定的に一巡する。

```text
fixture OHLCV / 財務 / TOPIX / sector / breadth
→ FeatureSet V1 Core 58特徴
→ point-in-time dataset snapshotと1/5/20日proxy label
→ Ridge predictionとpurged walk-forward validation
→ current portfolio / cash / tax state
→ cost / tax / whole-portfolio proposal
→ target sharesとBUY/HOLD/REDUCE/SELL/SKIP
→ manual execution fixture
→ next portfolio state
```

fixtureはproduction fallbackでも収益性の証拠でもなく、実注文は送信しない。

## 検証結果

2026-08-24に非editable install後、以下を実行した。

```text
python -m uv sync --all-groups --no-editable
python -m uv run --no-sync ruff check .
python -m uv run --no-sync mypy src
python -m uv run --no-sync pytest -q
python -m uv run --no-sync pytest --cov=stock_ai --cov-branch --cov-report=term-missing -q
python -m uv run --no-sync stock-ai --help
python -m uv run --no-sync stock-ai fixture-demo --snapshot-dir .demo-artifacts/final-dataset
```

- Ruff: pass
- mypy strict: pass（22 source files）
- pytest: 63 pass
- branch coverage: 87.08%（設定threshold 85%を通過）
- installed console entry point: pass
- deterministic E2E: pass、58 features、1,360 rows、3 validation folds

read-only specialist reviewを5系統（PIT/leakage、quant、portfolio、tax/cost、software reliability）で実施し、主なhigh findingを回帰test化した。reviewerはfileを変更していない。

## 既知の制約

- Goal 1 labelは12:30 entryやTOPIX/sector excess returnではなく、調整後終値間absolute returnの研究proxy
- baseline uncertaintyは現状training residual RMSEであり、OOF calibrated prediction intervalではない
- productionのhistorical universe membership、上場廃止・売買停止policy、corporate action revision lineageは未接続
- NISA枠の詳細な機会費用、複数broker/複数税policyのrouter、申告税額は未実装。Tax Engineは意思決定用推定のみ
- exact discrete optimizerは小さい明示candidate universe用。上限超過は近似解へ切り替えずfail closed
- datasetのParquet/JSON二ファイルpublishとJSONL experiment registryは単一processのGoal 1実装で、inter-process transaction/lockは未実装
- Git外でartifactを生成する場合は`STOCK_AI_CODE_COMMIT`を明示しないとprovenanceが`UNSET`になる
- fixtureのvalidation結果からprofitabilityを推論しない。最終holdoutはfixture E2Eでも未使用のまま保持する

## 実データ/APIまでblockedの項目

- J-Quants adapter、API key/plan、公式daily price・財務・銘柄master・TOPIX履歴
- 11:30時点の前場価格、出来高、market breadth、同時刻履歴
- PIT universe membership、sector history、shares outstanding/corporate action履歴
- SBI等の保有・約定CSV mapping、実fee条件、実口座・NISA残枠・税状態
- spread/slippage/market-impact calibration、live/paper forward observation
- broker integrationと注文送信は仕様上blockedではなく、製品方針として対象外

## 次のGoal

1. J-Quantsと選定したmorning data sourceのPIT adapter、revision lineage、capability reportを実装する。
2. 12:30 entry / 1・5・20日excess return labelと売買停止・上場廃止policyを確定する。
3. OOF uncertainty calibration、locked holdout運用、Momentum/Ridge以外の比較modelとfeature-family ablationを追加する。
4. broker/口座別cost・tax policy router、NISA capacity model、inter-process artifact transactionを実装する。
5. 大規模universe optimizer、paper/forward validation、状態付きdecision evaluationへ進む。
