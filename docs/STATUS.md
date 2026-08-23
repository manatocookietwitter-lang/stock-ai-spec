# Project Status

更新日: 2026-08-24
状態: `GOAL_2A_JQUANTS_V2_DATA_FOUNDATION_IMPLEMENTED`

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

## Goal 2Aで実装したもの

### J-Quants V2 acquisition

- V2固定base URLと`x-api-key`認証の専用client
- credentialはlive起動時にprocess環境の`JQUANTS_API_KEY`からだけ読込
- V1 token、mail/password、refresh token、設定file、`.env`自動読込を非実装
- `pagination_key`全page取得、循環/page上限検出
- plan別逐次rate limit、`/fins/summary`個別limit、`Retry-After`、429/5xx/network bounded retry
- sanitized errorはstatusとendpointだけを持ち、headerとresponse bodyを含めない
- Free既定: 銘柄master、株価日足、財務summary
- Light明示選択: 営業日calendar、TOPIX日足

### Immutable storage / catalog

- raw / normalizedをcontent-addressed immutable Parquet objectとして保存
- Parquet + manifestを一時directoryで完成後、directory単位でatomic publish
- payload再取得をobject IDでdeduplicateし、訂正payloadを別versionとして保存
- Parquet SHA-256、payload hash、schema version、row数、品質結果をmanifestへ保存
- DuckDBへingestion run、object、run-object対応、品質issueをtransaction記録
- 中断または品質失敗時も既存の正常objectを置換しない
- manifest/path/hash不整合を検出する`stock-ai data verify`

### Normalization / point-in-time

- 全external recordへ`provider/source_endpoint/source_date/received_at/available_at/as_of/payload_hash/schema_version/ingestion_run_id/source_record_hash`を付与
- J-Quants 5文字codeを原値`provider_code`として保持し、内部4文字`symbol`を別保存
- 日足のraw execution参照系列とresearch調整系列、`AdjFactor`、adjustment versionを分離
- 財務開示日時、主要実績・予想値、期末発行株式数・自己株式数を正規化
- 営業日区分0/1/2/3を保持し、equity business dayを1/2だけに限定
- APIが過去訂正時刻を返さないため`available_at = received_at`とし、初回取得前へ値を遡及させない
- DuckDB PIT読取はcutoffまでに受信したnatural key最新版だけを返す

### Quality / capability

- required schema、natural key重複、requested date、issue code、OHLC、volume、adjustment factor、disclosure時刻を検証
- 品質errorはrawとreportを残すがnormalized publishを止め、runを`FAILED`にする
- plan / data capabilityを`AVAILABLE / PARTIAL / BLOCKED_BY_PLAN / BLOCKED_BY_DATA_CAPABILITY / OUT_OF_SCOPE`で明示
- 期末発行株式数、初回取得以前のhistorical universe訂正、coverage未確認breadthを`PARTIAL`として推測補完しない
- 認証なしlive commandは停止し、fixtureへのproduction fallbackを行わない

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
python -m uv run --no-sync stock-ai data capabilities --plan free
```

- Ruff: pass
- mypy strict: pass（28 source files）
- pytest: 78 pass
- branch coverage: 86.11%（設定threshold 85%を通過）
- installed console entry point: pass
- deterministic E2E: pass、58 features、1,360 rows、3 validation folds
- Goal 2A fixture integration: 5 V2 endpoint、raw/normalized各5 object、PIT read pass
- idempotent refetch、non-retroactive correction、atomic failure、tamper、secret non-persistence: pass

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
- J-Quants APIは過去訂正の発生時刻を返さないため、初回取得以前の完全なrevision historyは復元不能
- `available_at = received_at`により安全側へ倒すため、初回backfillを過去の予測日時の入力には使えない
- `/fins/summary`の`ShOutFY`は期末開示値であり、日次shares outstanding seriesではない
- J-Quants planは自動判定せず、CLIで宣言したplanより上のendpointをfail closedする
- historical bulk download最適化と完全なlisting/delisting/corporate-action event lineageは未実装

## 実データ/APIまでblockedの項目

- 実行環境のJ-Quants API key/契約planと、credentialを表示しないlive API smoke結果
- 初回取得以前のprovider correction vintage、完全なlisting/delisting/corporate-action event lineage
- 11:30時点の前場価格、出来高、market breadth、同時刻履歴
- 日次shares outstanding、coverage検証済みmarket breadth、需給履歴
- SBI等の保有・約定CSV mapping、実fee条件、実口座・NISA残枠・税状態
- spread/slippage/market-impact calibration、live/paper forward observation
- broker integrationと注文送信は仕様上blockedではなく、製品方針として対象外

## 次のGoal

1. Goal 2Bでhistorical bulk retrieval、listing/delisting/corporate-action lineage、初回snapshot境界を拡張する。
2. coverage検証済みsector history / market breadth / shares outstanding seriesを接続する。
3. 12:30 entry / 1・5・20日excess return labelと売買停止・上場廃止policyを確定する。
4. morning data source、OOF uncertainty calibration、locked holdout運用を進める。
5. broker/口座別cost・tax policy routerとpaper/forward validationへ進む。注文送信は対象外のままとする。
