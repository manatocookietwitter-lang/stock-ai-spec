# Project Status

更新日: 2026-08-24
状態: `GOAL_3_IMPLEMENTATION_COMPLETE_LIVE_RESEARCH_BLOCKED_BY_PROCESS_CREDENTIAL`

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

## Goal 2で追加実装したもの（実データ履歴取得待ち）

### Historical acquisition

- 公式V2 Bulk List / Get、gzip CSV、signed URLのcredential非送信・非保存を実装
- Bulk file fingerprint単位のDuckDB checkpointを追加し、`RUNNING / FAILED`を安全にresume
- Bulk fileをsource dateごとに分割し、既存のraw / normalized / quality / immutable publish pathを再利用
- `stock-ai data history --start ... --end ... --plan standard`を追加
- 空Bulk list、header-only CSV、範囲外row、圧縮展開上限超過をfail closedし、file全体が完了するまでPIT queryから不可視化
- resume時はcheckpoint件数だけでなく、紐付く成功run/object/fileの完全性を再確認

### Production point-in-time data

- 営業日別master完全一致からhistorical universeとlisting / delisting intervalを構築し、current universeの過去適用を禁止
- source revision vintage（received_at）とderived observation timeを分離し、日足/TOPIX/contextは次営業日11:30、財務は開示時刻でmerge
- `SINGLE_VINTAGE_AS_REVISED / STRICT_AS_KNOWN`をartifactへ固定。前者は研究専用・採用不可、後者は復元不能labelをblock
- provider adjusted OHLCV、AdjFactor、adjustment versionからCorporate Action effective-date lineageを生成
- 期末開示の発行株式数・自己株式数を開示後だけ使い、未開示期間は0補完せずmissing reasonを保存
- PIT全universeからmarket breadthとsector return、eligible / observed / coverageを生成
- 売買停止gapでtechnical rolling windowをreset
- sector/TOPIX/beta returnも営業日gapでresetし、coverage不足breadthをmodel入力前にblock

### Production features / labels / baseline / E2E

- 同一source vintageからV1を一度だけ計算し、厳密subsetとしてFeatureSet V0（24）を生成。大規模履歴の重複計算を削減
- JPX固定calendarで1 / 5 / 20日absolute、TOPIX excess、sector excess、60日causal-beta residual、large-loss labelを生成
- endpoint価格欠損を`DELISTED_NO_EXIT_PRICE / SUSPENDED_NO_EXIT_PRICE`に分離し、次の観測価格へshiftしない
- Standard契約では12:30 labelを`BLOCKED_BY_DATA_CAPABILITY`とし、日足から推測しない
- Production Datasetをcontent hash付きParquet + JSONで固定し、未成熟labelをsnapshot cutoffでblank
- label entryを11:30より後の同日引けへ固定し、locked holdoutへlabel endが跨るdevelopment行をhorizon別にpurge
- Standardの12:30 exact labelはblock。Premiumでもrow単位entry/end/available_at/statusとcoverageを持ち、未成熟値をblank
- source 5 datasetのschema/value hash、列名/dtype、Parquet hash、metadata hashをsnapshot identityへ含める
- V0 / V1 / Datasetの全検証後だけatomic Production Build Manifestを公開し、partial buildを完成扱いしない
- locked holdout外だけでCASH / Momentum / Ridgeのwalk-forwardを行い、daily cross-sectional Rank ICと10 / 20 / 30 / 50 bps signal diagnosticをimmutable report化
- real-data Ridge predictionから最大8候補を選び、paper all-cash portfolioをDaily Portfolio Decision Engineへ渡すresearch-only E2Eを追加
- `stock-ai research build / baseline / e2e`を追加。order / execution recordは生成しない

### Goal 2 regression tests

- Bulk gzip parse、API key header境界、signed downloadへのkey非送信、checkpoint resume、範囲外row fail closed
- FAILED/RUNNING/file未完了runのPIT不可視、空archive、展開size上限、非数値volume/value、catalog/object/checkpoint照合
- current universe非遡及、財務開示時刻、Corporate Action、売買停止・上場廃止label、12:30能力不足
- future price mutation invariance、V0/V1共有値一致、Momentum/Ridge locked holdout、Decision Engine research E2E
- holdout期間価格の1/5/20日model不変、snapshot schema collision/改ざん、report/dataset provenance mismatch、Premium label maturity

## Goal 3で実装したもの（実データ研究run待ち）

### FeatureSet V2 Extended Technical

- V2 ExtendedをV1 Coreの厳密supersetとしてversioned manifest化
- downside volatility、60/120日最大drawdown、return skew、60日左裾quantileを追加
- volume z-score、20/60日売買代金比、OBV 20日傾き、CMF 20を追加
- adjusted openを推測せず、body/range/wick/close location/gapと20/60日breakoutを追加
- V2を一度だけ計算し、同一観測からV1/V0を厳密projectionする
- V0/V1/V2/Datasetを同じsource lineage・observation as-of・revision policy/statusのProduction Build Manifestへ含め、Dataset内feature値と各snapshotの完全一致を公開時・読込時に検証

### GBDT / Learning to Rank / downside

- LightGBM、XGBoost、CatBoostの回帰とLearning to Rank adapterを実装
- 1 / 5 / 20日を独立modelとして扱い、日付内relevance 0〜4でrankerを学習
- LightGBM/XGBoost/CatBoostの10% quantile回帰とabsolute-return大幅下落classifierを実装
- Brier、log loss、expected calibration error、quantile lower-tail rate / pinball lossを保存
- fold-local clipping、median imputation、相関pruningを実装し、validation値でfitしない
- bounded Optunaはtrial数・時間・seed・single-jobをconfigで固定し、strictly earlier tuning期間だけを目的関数に使う
- tuning期間と後続outer model-evaluation OOFをhorizon別に分離し、outer targetが自身のparameter選択へ入らない回帰testを追加
- COMPLETE / PRUNED / FAILを含む全Optuna trialのparameter・score・state・duration・failure reasonを保存し、COMPLETEが0件でも失敗ExperimentRecordへ監査列を残す

### OOF evaluation / ablation / ensemble / uncertainty

- 全model/task/horizon/seedのrow-level OOFを保存し、重複identityと非有限predictionを拒否
- pooled相関ではなく日付別Rank IC、ICIR、NDCG@5/10/20、Precision@K、Top-K targetを計算
- 実FeatureSet V0をF0とし、F1〜F12を直前championへ逐次追加するablationを実装。promotionはstrictly earlier tuning期間、incremental ICは後続outer期間で評価。F8/F9を非重複化し、V2にない需給F11は推測せず`BLOCKED_BY_DATA_CAPABILITY`
- OOS validation permutation importance、missing rate、fold retention率、seed間Rank IC / prediction安定性を診断
- outer OOFを時系列順のstacking fit / uncertainty calibration / final evaluationへ3分割し、label endpointで各境界をpurgeしたうえで非負・総和1 stackingと未接触区間のcoverageを評価
- absolute-return OOFをexpected return / downside quantile / large-loss probabilityへ集約し、予測as-ofより前にlabelが成熟したresidualだけでstandard errorを更新
- 1/5/20日すべてとaware `as_of`・model/feature/data provenanceを必須にした型付き`Prediction`を生成し、未成熟calibration/horizon非整合は明示block
- benchmark-excess targetはabsolute returnへ偽装せず、Decision Engine入力への変換をfail closed

### Research artifact / CLI

- report、library version、commit、config全文/hash、Build/data/V2 feature snapshot、feature definition hash、全trial/fold、選択期間、holdout境界、cost/tax/Decision Engine version、revision policyを監査field化
- OOF ParquetとJSON metadataをcontent-addressed directoryへatomic publishし、Parquet・metadata・report content hashをload時に再検証
- `stock-ai research advanced`は認証済みV0/V1/V2/Dataset Build Manifestだけを入口とし、成功・設定不正・全trial失敗・途中fold失敗・artifact公開失敗をappend-only ExperimentRegistryへ保存。完了済trial/foldとreport/Build/V2 identityを失わない
- model family / horizon / seed / estimator / Optuna / OOF行数 / model-fit数を明示的にboundし、超過時は`BLOCKED_BY_RESOURCE_CAPABILITY`
- locked final holdoutはreport APIからrowを返さず、Goal 3 development reportは常にresearch-only / adoption不可
- API keyが見えない現processではlive Production Datasetを作らず、deterministic fixtureでlibrary数値・漏洩境界・artifact改ざんを検証

## 検証結果

2026-08-24に非editable installを更新後、Goal 2 final gateとして下記を実行した。`uv` executableはPATHにないため、
同じinstalled `uv 0.12.5`をsystem Pythonのmodule entry pointから実行した。

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
- mypy strict: pass（32 source files）
- pytest: 101 pass
- branch coverage: 85.08%（設定threshold 85%を通過）
- installed console entry point: pass
- non-editable packageを`--reinstall-package stock-ai-decision-support`で再buildし、主要source/installed moduleのSHA-256一致を確認
- deterministic E2E: pass、58 features、1,360 rows、3 validation folds
- local `data verify`: pass、catalogとimmutable object 12件を照合。Production artifactはlive履歴未取得のため0件
- Goal 2A fixture integration: 5 V2 endpoint、raw/normalized各5 object、PIT read pass
- idempotent refetch、non-retroactive correction、atomic failure、tamper、secret non-persistence: pass

Goal 3 checkpoint前gate（source tree）:

- Ruff: pass
- mypy strict: pass（34 source files）
- pytest: 136 pass
- branch coverage: 85.74%（設定threshold 85%を通過）
- LightGBM / XGBoost / CatBoost各4 taskの決定的OOF smoke: pass
- V2全追加列のexact formula/warm-up/zero denominator、V0/V1/V2 projection、Production Build observation/content lineage: pass
- locked holdout/outer target mutation invariance、label-maturity purge、OOF/report tamper、simplex、calibration、Decision Engine output contract: pass
- Optuna全FAIL trial監査、完全なfold/trial identity、CLI設定不正を含む失敗registry保存: pass
- 後段family/foldの失敗、非有限fold、artifact write/reload/OSErrorでも完了済監査とBuild/V2 lineageを保持: pass
- installed `stock-ai` / `research advanced --help` / fixture E2E / `data verify` / source-installed SHA-256一致: pass

read-only specialist reviewを5系統（PIT/leakage、quant、portfolio、tax/cost、software reliability）で実施し、主なhigh findingを回帰test化した。reviewerはfileを変更していない。

## 既知の制約

- Goal 1 labelは12:30 entryやTOPIX/sector excess returnではなく、調整後終値間absolute returnの研究proxy
- Goal 1 baseline uncertaintyはtraining residual RMSE。Goal 3はOOF calibrated rank-space intervalを持つが、live/full-scale empirical coverageは未評価
- Goal 2実装とfixture/MockTransport検証は完了したが、現在のlocal catalogはfull historical datasetではないためProduction Dataset / baselineの実データartifactは未生成
- NISA枠の詳細な機会費用、複数broker/複数税policyのrouter、申告税額は未実装。Tax Engineは意思決定用推定のみ
- exact discrete optimizerは小さい明示candidate universe用。上限超過は近似解へ切り替えずfail closed
- JSONL experiment registryは単一process実装でinter-process lock未実装。Goal 2/3 Production snapshotとadvanced reportはdirectory atomic publish + Build Manifestへ移行済み
- Git外でartifactを生成する場合は`STOCK_AI_CODE_COMMIT`を明示しないとprovenanceが`UNSET`になる
- fixtureのvalidation結果からprofitabilityを推論しない。最終holdoutはfixture E2Eでも未使用のまま保持する
- J-Quants APIは過去訂正の発生時刻を返さないため、初回取得以前の完全なrevision historyは復元不能
- `STRICT_AS_KNOWN`ではreceipt時刻以前へbackfillを遡及させず、historical labelをblockする。`SINGLE_VINTAGE_AS_REVISED`は再現可能な研究専用で採用不可
- `/fins/summary`の`ShOutFY`は期末開示値であり、日次shares outstanding seriesではない
- J-Quants planは自動判定せず、CLIで宣言したplanより上のendpointをfail closedする
- Corporate Actionのannouncement時刻と詳細種別はdaily price endpointに存在せず、effective-date adjustment lineageのみ。推測しない
- full JPX履歴のmemory/time実測はAPI key継承後まで未実施。Goal 3は推定OOF行数/model-fit数をfail closedでboundし、horizon/model-family batchを既定にしたが、scale capabilityは実測完了まで`PARTIAL`
- Goal 3 advanced reportはdevelopment OOF専用でlocked holdoutを未開封。liveデータ、複数seed、全horizonの実runと最終holdout評価前にChampion採用しない
- OOF ensembleの不確実性は日付内rank space。絶対return金額幅へのcalibrationはlive OOSが得られるまで未採用

## 実データ/APIまでblockedの項目

- 現在のCodex processから`JQUANTS_API_KEY`を認識できず、full Bulk history取得は環境変数を継承した再起動待ち
- 初回取得以前のprovider correction vintage、完全なlisting/delisting/corporate-action event lineage
- 11:30時点の前場価格、出来高、market breadth、同時刻履歴
- 日次shares outstanding、coverage検証済みmarket breadth、需給履歴
- SBI等の保有・約定CSV mapping、実fee条件、実口座・NISA残枠・税状態
- spread/slippage/market-impact calibration、live/paper forward observation
- broker integrationと注文送信は仕様上blockedではなく、製品方針として対象外

## Goal 2 implementation checkpoint後の継続順

1. 上記final gateを根拠に、外部依存を除くGoal 2 implementation checkpointを作る。
2. 外部blockをSTATUSへ残したままGoal 3（GBDT / LTR / quantile / downside / ablation / bounded Optuna / OOF ensemble / uncertainty）を実装した。
3. Codex再起動後、Standard Bulkで2017-01-04〜契約上の最新確定営業日をcheckpoint / resume取得する。
4. `data verify`、Production V0/V1/Dataset、Momentum/Ridge baseline、research Decision E2Eを実データで実行し、Goal 2 live-data acceptanceを追記する。
5. Goal 3全gateとreview後にrecoverable checkpointを作成し、Goal 4前場AI、続いてGoal 5 PWA / automation / Paper / manual execution recordへ進む。注文送信は対象外のままとする。
