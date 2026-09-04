# Project Status

更新日: 2026-09-05
状態: `GOAL3_INDEPENDENT_RESEARCH_RUNNING`

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
- deterministic fixtureでlibrary数値・漏洩境界・artifact改ざんを検証済み。live catalogは2017〜2020を受入済みだが、全取得期間の確定前なのでProduction Datasetは未生成

## Goal 4で実装したもの（live前場provider・model採用はblocked）

### Morning data / freeze contract

- provider-neutralな09:00〜11:30 bar契約と`MorningFreezeMetadata`を実装
- `available_at <= 当日11:30`、aware JST timestamp、exact cutoff、価格・出来高・売買代金、重複をfail closed検証
- current holdingとdaily candidateのunionを監視universeにし、bar universeとの完全一致を検証
- freeze provider / source snapshot ID / 全source record IDをstock・TOPIX・sector入力と完全一致で検証
- freezeのholding / candidate roleとcapability reportをfeature rowへ完全一致で検証
- live provider未設定時は全capabilityを`BLOCKED_BY_DATA_CAPABILITY`とし、fixtureへfallbackしない
- bar区間volume/valueとsession累積を分離し、日次出来高から前場値を推測しない

### F13 / F14 features

- `morning-core-v1`をmachine-readable Feature Manifestとして実装
- 09:00→09:05 / 09:15 / 09:30 / 10:00 / 11:00 / 11:30 return、前日終値gapを実装
- 同時刻TOPIX / sector relative、high / low / range、realized volatility、VWAP、range位置を実装
- 各cutoff cumulative volumeと、当日を除く過去20session同時刻volume / trading-value progressを実装
- monitored / candidate volume rank、holding / candidate role、freeze済みdaily prior forecastを実装
- `morning-microstructure-v1`をF13 strict supersetとして定義し、quotes / order book / trade frequency能力がある列だけ実装
- 部分F14をcapability固有の`morning-microstructure-v1-<feature-subset-hash>` manifestとして保存・再認証
- exact cutoff欠損、宣言済みmicrostructure列欠損、11:30後のbarを推測補完せずblock

### Morning model / Decision compatibility

- 1 / 5 / 20日daily forecastに対する残差更新datasetを実装し、label entryが11:30より後であることを強制
- label entryを同一session exact 12:30、endをartifactに認証した固定JPX calendarの1 / 5 / 20 session後へ固定し、aware `label_end_at`以前のavailabilityを拒否
- 未成熟labelをpublication cutoffでblankにし、label statusを保存
- fold学習・OOF較正とも`label_available_at < prediction.as_of`を要求し、遅延訂正をendpointへbackdateしない
- no-update daily forecastに対し、固定parameter Ridge / LightGBM / optional small MLPをpurged expanding OOFで比較
- modelごとにMSE、日付別Rank IC、revision win rate、holding / candidate row coverageを保存
- OOS改善がないmodelを`REJECTED`、改善してもdevelopmentでは`RESEARCH`に留め、自動Champion昇格しない
- small MLPは既定`DISABLED`。1D-CNN / TCN / GRU / small Transformerは同期sequence履歴不足で明示block
- `label_end < prediction date`かつ`label_available_at < prediction.as_of`のresidualだけでdownside / large-loss / uncertaintyを更新
- daily priorとmorning revisionの両provenanceを持つ型付き`Prediction`をDaily Portfolio Decision Engineへ接続
- 1 / 5 / 20日、同一as_of、同一prior bundle、全freeze universeを満たすPrediction batchだけを生成
- 認証済みreport / Datasetからresearch-only modelを再fitし、履歴終了後の翌営業日11:30をoutcomeなしで推論
- refit bundleは認証済みfactoryだけで生成し、report / dataset / family / seed / training boundary / estimator contract / fitted state hashを一つのidentityとして再検証
- current 20session profile欠損をimputeせずblockし、feature行content hashを再照合してfreeze済みprice / liquidityと全universeをtyped adapterでEngineへ渡す
- provider / source snapshot+record / role / capability / report ID / price / ADVを`research-only` proposalまで保持し、実Portfolio保有と評価Candidateへ再照合
- Morning modelはActionを直接出さず、current inference batchをEngineへ渡し、別の決定的fixtureでHOLD / SKIPからSELL / BUYへの変化がEngine経由で生じることを確認
- Decision EngineはMorning predictionの混在、Portfolio freeze不一致、auditと評価Candidateのprice / ADV不一致を拒否

### Immutable artifacts / CLI

- Morning Datasetとrow-level OOF reportをschema・値・source ID・manifest・Parquet/metadata hash付きで保存
- content-addressed directoryを一時directory完成後にatomic publishし、再利用前に認証
- `research morning-capabilities`と、明示fixture専用`research morning-fixture`を追加
- fixture commandは`research_only=true / order_instruction=false / live_provider_used=false`を明示
- fixture commandは履歴dataset→OOF→research refit→翌営業日の4銘柄current Predictionまで実行

## Goal 5で実装したもの（local運用完成・live acceptanceはblocked）

### Operational ledger / user workflow

- SQLite WAL台帳へPortfolio、AI提案、version付きユーザー判断、実約定、翌日状態を別recordで保存
- proposal/date、decision/version、execution IDをpayload hash、catalog identity、foreign keyで検証し、同一identityの異内容を拒否
- 提案生成に使った`DecisionEngineConfig`全文をproposal IDと生成時刻へ結びつけた不変policy snapshotとして保存し、proposal / archive証跡 / policyを同一transactionで公開。snapshot欠落・改ざんは提案台帳のintegrity違反として拒否
- 判断reviewは全lineを必須とし、AI推奨または取引なしの検証済み候補だけを許す。bucket別現金とexact cost/taxを再計算し、未再最適化株数は保存しない
- 手動約定は方向・注文/約定株数・時刻を判断と照合し、next Portfolioは実fillだけから原子的に生成。proposal/decisionは上書きしない
- 同一銘柄の複数account bucketを全画面・台帳・照合で分離

### Automation / recovery

- data sync、candidate、Morning capture、11:30 freeze、prediction、proposal、notification、EOD、monthly challengerのstage contractを実装
- process lock/heartbeat、handler前RUNNING、成功確定時のlock owner/expiry再照合、stable logical-stage idempotency key、同一workflow upstream gate、freshness/proposal lineageを保持した成功stageの状態repair、失敗・block記録を実装
- proposal archive完了前の通知を拒否し、stale/error時は同日archive済みでも提案を表示しない
- live handler未設定時は`BLOCKED_BY_DATA_CAPABILITY`で停止し、fixtureや前日提案へfallbackしない
- Windows Task Scheduler登録scriptを表示するが自動実行せず、ユーザーreview後の手動登録に限定
- payload/catalog/FK/SQLite integrity検証、status→same-day archive済みproposal照合、atomic no-replace backup、read-only元と明示確認付きrestore、restore前後検証を実装

### Mobile PWA / API

- React / TypeScript / Viteのmobile-first PWAとFastAPI local APIを実装し、`127.0.0.1`でserve、Host/Originもlocalhostへ限定
- 5項目下部nav、Home、Today、Decision Review、Decision Saved、Execution、Ranking、Stock Detail、Validation、正式capital/model下層を含むSettings、Data Operationsを実装
- Settingsの現金・保有・税状態に加え、最低現金比率、最大保有数、銘柄・業種・turnover・ADV上限、改善閾値等をarchive済みproposalのpolicy snapshotから表示し、未登録値を推測しない
- TodayはSELL/REDUCE/BUYを優先、HOLDを折り畳み、SKIPを変更一覧から除外。株数中心でcurrent→recommended→differenceを表示
- 100株制約・cash/cost/taxを安全な選択候補ごとに再計算し、未再最適化または違反中は保存不可。判断保存と約定記録にはmanual intent headerを必須化
- Ranking候補順位と最終Portfolio Actionを分離し、Stock Detailは同一symbolのaccount bucket tab、加重取得単価、口座別Actionを分離表示
- loading / empty / stale / error / fixture状態、data/model timestamp、model version、no-order disclaimerを表示
- service workerはstatic shellだけをcacheし、`/api/` responseをcacheしない。open tabもvisibility/online/60秒で再取得し、offline時は保持提案を破棄
- CSP、frame拒否、referrer拒否、secret value非表示を実装。broker/order routeは存在しない
- Playwright / Microsoft Edge / disposable fixture台帳でToday→判断変更・保存→一部約定→実保有反映→Homeの実backend込みE2Eを固定gate化
- E2EはNTTのSELL 500株中100株の部分約定を記録し、apply後のHomeで実保有が500株から400株へ変わったことをexact assert

### Import / Paper validation

- 約定CSVと口座状態CSVをpreview-firstで取込み、逆売買・注文株数・累積fill差異をconflict化し、明示確認後だけappend。判断前時刻は確認でも拒否
- 口座状態CSVは非ゼロPOSITIONと全bucketのCASH row、available / reserved cashを必須とし、保有・現金を原子的に照合
- Paper outcomeはarchive前に登録したcontent-addressed不変JPX calendar、exact next-session経路、aware endpoint/availability、archive-before-endpoint、model versionを照合した将来観測だけを不変保存
- 1営業日・1営業日1観測だけを週次/月次compoundし、最大drawdown、minimum observation、active Champion cohort、最新行から連続するexact-version Challenger cohort、同一version隣接窓driftを実装
- minimum observationやdriftからmodelを自動昇格せず、実観測のないfixture画面は空状態を表示
- 起動、Task Scheduler、CSV schema、backup/restore、credential境界を`docs/GOAL5_RUNBOOK.md`へ記録

## 検証結果

### 2026-08-26 認証付きホスト版checkpoint

- ユーザーのremote URL要望を受け、既存localhost PWAを直接公開せず、`hosted/`へ本人限定のSites companionを実装
- ChatGPT sign-inのforwarded user headerをserver側で読み、D1にはuser別の安全境界確認状態だけを保存
- J-Quants API key、`data/live`、Production Dataset、model artifact、実口座CSVをホスト版へ送信する経路は設けていない
- 実データ不足時は`NO PROPOSAL`と不足理由を表示し、fixture、前日提案、推測値へfallbackしない
- no-order、no-broker、local credential境界、2017〜2020受入済み／2021年以降未取得の現状を画面へ明示
- D1 migration `operator_settings`、owner-scoped server action、mobile-first画面、OG preview assetを追加
- Next.js 16.3.2、React 19.2.8、vinext beta.8、Vite 8.2.2、Cloudflare plugin 1.53.1へ更新
- `npm run lint`: pass
- `npm run build`: pass
- `npm audit --omit=dev --json`: vulnerability 0
- `npm audit --json`: development dependencyを含めvulnerability 0
- Sites `create_site`は一度だけ呼び出され、実際にはdirect `structuredContent`でproject IDを返していた。当時のagentが誤って`.result`配下を参照したためIDを永続化できず、deploymentを保留した

### 2026-08-29 認証付きホスト版再開checkpoint

- Sites plugin 0.1.46の正本手順を再読し、SIWC sign-in / sign-outを`target="_top"`のtop-level navigationへ更新
- social preview assetを標準の`public/og.png`へ揃え、Open Graph / X metadataも同一pathへ更新
- `npm run lint`: pass、`npm run build`: pass、`npm audit --json`: vulnerability 0
- 再開時もagent側がdirect `structuredContent.items`を誤って`.result.items`として読んだため0件と判定していた。sessionの非secret監査情報から既存project IDを復元し、重複createせず`.openai/hosting.json`へ永続化した
- ユーザーの明示指示でホスト版の閲覧accessをpublicへ変更。公開内容は安全停止理由・準備状況・no-order境界だけで、本人別確認状態の保存はChatGPT認証を維持
- Sites access policyを`public`へ更新し、version 2を`https://stock-ai-decision-support.manato0618.chatgpt.site`へproduction deploy
- 公開URLはHTTP 200、`株AI` / `NO PROPOSAL`表示、secret値pattern不検出を確認。Open Graph / X imageは同一production originのabsolute URLへ修正済み
- live D1 binding `DB`と`operator_settings` tableを確認。public visitorはreadinessを閲覧でき、user別確認状態の保存だけがserver-side ChatGPT identityを要求する

### 2026-08-25 実データ受け入れ中間checkpoint

- 現processで`JQUANTS_API_KEY`設定済みをboolean確認。credential値は表示・保存していない
- 2026-08-24の実API preflightで、銘柄master 4,444行、日足4,444行、Calendar 1行、TOPIX 1行、財務summary 7行を取得
- 5 datasetのBulk List / Getを実確認。2015・2016年の日足BulkはHTTP 400、2017年以降は取得可能で、実file最古日は2017-01-04
- 観測capabilityはStandard相当の10年履歴と整合するが、APIにplan自己申告endpointがないため契約名そのものは自動断定しない
- REST日足は調整OHLCVを返し、Bulk日足は公式仕様どおりraw OHLCV + `AdjFactor`のみ。D064の公式累積係数式へadapterを更新し回帰testを追加
- 午後session / `AAdjO`は実REST応答・Bulkとも利用不可。exact 12:30 entry labelは`BLOCKED_BY_DATA_CAPABILITY`
- Git対象外の`data/live`へ2017-01-04〜2020-12-30を年次stage取得。Calendar scopeは2017-01-01〜2020-12-31
- Bulk checkpointは`SUCCEEDED=198 / FAILED=1`。FAILED 1件は旧schema期待による初回日足fileで、後続resume成功後も監査記録として保持
- 同一scope再実行で2017年1月は5/5 file、2017年2〜12月は45/45 fileがdownload 0でskipされ、resume完全性を確認
- `data verify --data-root data/live`: 10,808 immutable object、feature 0、dataset 0、build 0、status OK
- catalog品質issue 0件。normalized object集計は日足976 object / 3,862,418行、master 995 / 3,935,157行、財務996 / 76,076行、TOPIX 976 / 976行、Calendar 1,461 / 1,461行。rawも同数
- Production Dataset、Goal 3実データwalk-forward、特徴量・parameter・ensemble選定、Champion固定、locked holdout評価は未実行。holdoutには触れていない
- ユーザー指示により2020年stage完了を切りの良い停止点とした。次回再開点は2021-01-01
- 中間checkpoint最終gate: Ruff合格、mypy strict 43 source合格、Python 192 test合格、branch coverage 85.31%、PWA lint / typecheck / 6 unit test / production build / 1 E2E合格
- 固定日fixtureが実日付を越えて露呈した2件のoperations testを修正。stock詳細APIは任意の`businessDate`を受けて再現可能に参照でき、automation lock testは実時計を使用する
- gate後の`data verify`再実行も10,808 object / status OK。`data/live`はGit対象外で、API key値はsource・log・fixture・Markdownへ保存していない

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

Goal 4 checkpoint前gate（source tree）:

- Ruff: pass
- mypy strict: pass（37 source files）
- pytest: 161 pass
- branch coverage: 85.10%（設定threshold 85%を通過）
- exact 11:30 / provider / source / holding+candidate freeze、post-freeze source、遅延label availability: pass
- F13 exact formula / warm-up / zero-volume、partial F14 manifest / null capability value / artifact tamper: pass
- holdout endpoint purge、pre-fit resource bound、Ridge / LightGBM、3-seed全horizon MLP challenger: pass
- aware label endpoint、snapshot range identity、authenticated report / Dataset refit、model bundle factory/state認証: pass
- outcome-free current inference、全universe typed Prediction、feature evidence hash、Decision Engine exact-freeze E2E: pass
- installed `research morning-capabilities`はlive provider未設定をfail closed、`research morning-fixture`は4 current Predictionを生成: pass
- local `data verify`: 12 immutable object、Production artifact 0件、status OK
- non-editable package再build後、Goal 4主要4 moduleのsource/installed SHA-256一致: pass

Goal 5 final gate（source tree + installed wheel + Microsoft Edge）:

```text
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m mypy src
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
.venv\Scripts\python.exe -m pytest --cov=stock_ai --cov-branch --cov-report=term-missing -q -p no:cacheprovider
cd web
npm run lint
npm run typecheck
npm test -- --run
npm run build
npm run test:e2e
```

- Ruff: pass
- mypy strict: pass（43 source files）
- pytest: 189 pass
- branch coverage: 85.31%（設定threshold 85%を通過）
- frontend ESLint / TypeScript / production build: pass
- Vitest: 6 pass
- Playwright / Microsoft Edge full-stack E2E: 1 pass。Today→判断変更→保存→NTT 500株のSELLを100株部分約定→実保有apply→Home 400株を検証
- non-editable wheelを再buildし、installed `stock-ai --help` / `ops capabilities` / fixture bootstrap / `ops verify` / Task Scheduler script: pass
- installed fixture台帳: Portfolio 1、proposal / archive / Decision policy各1、status 1、ranking 4、integrity `OK`
- installed server: health `ok`、注文送信不可、Settingsは最低現金0.10・最大保有10・銘柄上限0.50等の実policy値を返し、credential valueを返さない
- Goal 5主要6 moduleのsource / installed SHA-256一致: pass
- read-only最終review（PIT/reliability、quant/Paper、portfolio/operations）: material checkpoint blockerなし
- live data、profitability、model採用はこのgateでは評価していない

read-only specialist reviewを5系統（PIT/leakage、quant、portfolio、tax/cost、software reliability）で実施し、主なhigh findingを回帰test化した。reviewerはfileを変更していない。

## 既知の制約

- Goal 1 labelは12:30 entryやTOPIX/sector excess returnではなく、調整後終値間absolute returnの研究proxy
- Goal 1 baseline uncertaintyはtraining residual RMSE。Goal 3はOOF calibrated rank-space intervalを持つが、live/full-scale empirical coverageは未評価
- Goal 2実装とlive acceptanceは完了し、2017-01-04〜2026-08-28のraw/normalized履歴と、現行市場区分が利用可能な2022-04-04以降のProduction Datasetを認証済み。Goal 3実データbaseline / advanced / final holdoutは未実行
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
- 2017〜2026-08-28のJPX履歴取得とfull Production Dataset生成は実測済み。405万8,130行の生成では物理dtype依存hashと一括feature copyを実データで修正し、メモリ境界付き認証まで完了。Goal 3の全family/horizon walk-forward scale capabilityは実run完了まで`PARTIAL`
- Goal 3 advanced reportはdevelopment OOF専用でlocked holdoutを未開封。liveデータ、複数seed、全horizonの実runと最終holdout評価前にChampion採用しない
- OOF ensembleの不確実性は日付内rank space。絶対return金額幅へのcalibrationはlive OOSが得られるまで未採用
- Goal 4 refit bundleとcurrent inferenceは明示的research-only。Goal 5はPaper drift記録を持つが、model weightの承認済み永続registry、live推論時間、十分なlive OOS evidenceが揃うまでChampion採用しない
- Morning datasetのlabel endpointはcallerが供給するJPX session calendarとaware `label_end_at`に依存する。live provider未決のため、fixtureのpandas営業日・15:30 endpointは市場calendar/scheduleの代替でも収益性の証拠でもない
- Goal 5の実データAPI/PWAは引き続きlocalhost専用で、internetへ公開してはならない。本人限定ホスト版companionは実装済みだが、local実データや提案の同期経路は未実装で、Sites project作成結果の外部確認待ち
- PWA内通知だけ利用可能。web push providerは未設定で`BLOCKED_BY_CONFIGURATION`
- daily automation frameworkとTask Scheduler scriptは実装済みだが、live data/Morning/model handlerは外部capability待ちで、未設定のstageは安全に停止する
- Paper集計はlive forward observationが0件のため空状態。fixture proposalから収益性、drift、Champion採否を推論しない

## 実データ/APIまでblockedの項目

- 現在のCodex processは環境変数`JQUANTS_API_KEY`が設定済みであることを認識している。値は表示・保存していない。Standard capabilityと2017-01-04〜2026-08-28のliveデータ品質、年次Bulk resume、Production Buildを実取得で検証済み
- 初回取得以前のprovider correction vintage、完全なlisting/delisting/corporate-action event lineage
- 11:30時点の前場価格、出来高、market breadth、同時刻履歴
- 承認済みMorning model registry entry、live OOS比較、11:30締切内の実推論時間
- 日次shares outstanding、coverage検証済みmarket breadth、需給履歴
- 実fee条件と、local PWAへ手入力する実口座・保有・現金・NISA残枠・税状態。SBI等のCSV mappingはD068により任意で、研究のblockerではない
- spread/slippage/market-impact calibration、live/paper forward observation
- broker integrationと注文送信は仕様上blockedではなく、製品方針として対象外

## Checkpoint後の継続順

1. Goal 2 checkpoint `f5763e7`、Goal 3 checkpoint `d9b094b`、Goal 4 checkpoint `a2a359d`は作成済み。
2. Goal 5の全gateとread-only reviewは完了し、このSTATUSを含む外部依存外のclean checkpointを正本とする。
3. Standard相当Bulkの2017-01-04〜2026-08-28はcheckpoint / resume取得済み。26,030 objectを検証し、Production Build `2fc936a7ca9b939d8016ad3c5efea17c53ffd5264d5ece398a8329bf2f2dfe5f`を固定した。
4. 次はGoal 3 development-only選定と固定、locked holdout単回評価、Goal 4 live OOS、Goal 5 live Paper観測の順に実行し、model採否を追記する。
5. broker integrationと自動売買は今後も実装しない。

## 2026-08-29 実データ研究再開

- ユーザー指示によりSBI口座状態CSVを必須条件から外し、local PWAへの手入力を初期運用の正本とした（D068）
- Goal 3実データ研究は口座入力を待たず、J-Quants capability確認、2021年以降のBulk resume、全履歴verify、Production Dataset生成の順で再開する
- model / feature / hyperparameter / ensemble weightの選択終了前にlocked final holdoutを評価しない

## 2026-08-30 Goal 2 live acceptance完了

- J-Quants V2 Standard capabilityを実取得で確認。`security_master / daily_prices / research_adjusted_ohlcv / trading_calendar / topix_context / financial_summary / bulk_history`は`AVAILABLE`、`shares_outstanding / historical_point_in_time_universe / market_breadth`は`PARTIAL`、exact 12:30 entry labelは`BLOCKED_BY_DATA_CAPABILITY`、前場intradayと需給は今回scope外
- raw最大期間は2017-01-04〜2026-08-28（Calendarは2017-01-01開始）。2021〜2025は各49 file、2026-08-28までは105 fileを追加取得し、2021年stage再実行は49 fileすべてskipしてresumeを確認
- catalogはdaily prices 9,806,149行、financial summary 185,321行、security master 9,878,888行、TOPIX 2,360行、Calendar 3,528行。最終`data verify`は26,030 object、feature snapshot 3、dataset snapshot 1、build 1、status `OK`
- 現行Prime / Standard / Growthかつcommon issueのPIT universeは2022-04-04から利用可能。Production Dataset `3d837e220b4662d2405dafd295b91645813af6e8ffed8cd52a44988b49847a7d`は2022-04-04〜2026-08-27、1,077営業日、4,144銘柄、4,058,130行
- 1 / 5 / 20日absolute labelの`AVAILABLE`は4,004,464 / 3,985,623 / 3,924,593行。売買停止・上場廃止・未成熟は理由別statusで保持し、価格をshiftや推測で補完していない
- Production Buildは`2fc936a7ca9b939d8016ad3c5efea17c53ffd5264d5ece398a8329bf2f2dfe5f`。source cutoffは2026-08-29 23:30 JST、historical revisionは`SINGLE_VINTAGE_AS_REVISED / PARTIAL`のため研究専用で、採用可能性を意味しない
- 実データで市場区分開始境界、provider code優先株衝突、no-trade行、UTC精度、全欠損lineage、一括feature copy、Parquet物理dtype hash、Build全frame同時保持を修正。旧物理dtype hash bundle 4.77GBは削除せず`artifacts/quarantine/production-physical-hash-v1-20260830`へ隔離した
- locked final holdoutにはまだ一度もアクセスしていない。Goal 3の全model / feature / hyperparameter / ensemble weight選択をdevelopment期間だけで完了してから単回評価する
- Goal 2 live acceptance後の全品質gateはRuff / mypy strict（43 source files）/ pytest 195件 / branch coverage 85% / frontend ESLint・TypeScript・Vitest 6件・production build / Microsoft Edge E2E 1件がすべてpass。実データ`data verify`も26,030 object、3 feature snapshot、1 dataset snapshot、1 buildで再度`OK`

## 2026-09-04 Goal 3実データ研究の中断安全化

- 2026-08-30 12:37 JST開始の旧monolithic runは、5日horizonで3 model family、4 task、Optuna、F0〜F12 ablation、diagnosticsを一processへ詰め、12時間超の計算後にhost session終了で停止した。content-addressed advanced reportは未公開、Experiment Registryの成功記録もなく、locked holdoutにはアクセスしていない
- D060の既存方針を実運用化し、`stock-ai research campaign`を追加した。`1 horizon × 1 model family`ごとに独立process、content-addressed report、append-only Experiment Registry、atomic resume manifest、attempt別logを持つ
- 再開時は保存済みartifactをhash / identity / config / commitまで再認証し、成功分をskipする。report公開直後の親停止はartifactから回復し、停止済み`RUNNING`だけを`INTERRUPTED`として再実行する。生存中childは二重起動しない
- ablation後の選択済みfeature subsetを同じbatch境界で再学習できるようにし、V2外・重複featureを拒否してfeature名hashもresume認証へ含めた
- API keyは不要な研究childの環境から除去し、campaign command / manifest / logへも含めない。campaignとlogはGit対象外、実験結果は既存の認証済みartifact境界とExperiment Registryへ保存する
- 親processはBuild markerのidentity / metadata hashを軽量認証してmanifestを先に保存し、各batch childが全Parquet内容を実行直前に完全認証する。model開始前の約20分間にresume状態が存在しない実測上の穴を塞いだ
- 旧runの失敗理由はresource scaleと中断粒度であり、model成績による棄却ではない。development選択は新campaignの完了結果だけから行い、全選択固定までlocked holdoutを開かない

## 2026-09-04 Goal 3 5日foldのメモリ境界修正

- base campaignは1日LightGBM / XGBoost / CatBoostの3 batchを認証済みで完了。5日LightGBM attempt 1はhost process中断でartifact未公開、attempt 2は3 Optuna trialと一部outer foldを完了後、2,826,087行 × 77特徴の`float64`一括copy（1.62 GiB）を確保できず`FAILED`になった
- attempt 2の失敗、完了trial、生成済みfold監査はExperiment Registryへappend済み。content-addressed成功reportは公開されず、5日XGBoost以降も開始していない。locked final holdoutは引き続き未開封
- D071として有限値正規化を列単位の事前確保`float32`行列へ変更し、training-only clip / medianも`float32`、clip / fill / dropをin-place化した。行数・特徴数・target・fold・purge / embargo・tuning範囲は削減していない
- 変更後はAdvanced Research / campaign 44 testをFutureWarningエラー扱いでpass、Ruffとstrict mypy（45 source files）もpass。新code commitを固定した5日 / 20日campaignとして再開し、旧code成功分と新code分のprovenanceを混同しない

## 2026-09-04 Goal 3独立runnerへの引継ぎ

- ユーザー指示により、Codexが30〜60秒周期でPID / CPUを監視する運用を廃止した。以後はユーザーが進捗を尋ねた時だけread-only statusを1回実行する
- 最終の単発process確認ではPython workerが存在せず、campaign `28f0c8df2ed45e5d345271a14ffcb9f7fc288998c4dd710ec10cf6e290b52558`の`h5-lightgbm`だけが保存上`RUNNING`、実効状態`INTERRUPTED`だった。locked holdoutは未開封
- `runner/research-runner.ps1`を追加。campaign / Build / source / dependency provenance、完了artifactを認証し、排他lock、生存worker重複拒否、dead `RUNNING`のatomic `INTERRUPTED`遷移、未完了batchの自動resumeを行う
- `runner/register-research-runner-task.ps1`を追加。hiddenなWindows Task Scheduler jobとしてlogon時と5分周期の再開入口を構成し、Codex終了やWindows再起動後も同一manifestから再開できる
- 研究childへ`JQUANTS_API_KEY`を継承しない。command / task / manifest / runner logにもcredentialを保存しない
- 現行campaignの安全を優先し、最小checkpointはD070の`horizon × model family`単位を維持する。fold / Optuna trialの永続再利用は独立campaign完了後のsource変更境界で追加し、完了済み1日3familyのreportと全失敗監査を保持する
- read-only確認入口: `powershell -NoProfile -File runner/research-runner.ps1 -Action status -Manifest artifacts/campaigns/goal3-base-v3.json`
- Windows PowerShell 5.1のnative引数quote差異と、Task Scheduler権限境界でのGit ownership判定を修正した。repo限定の`safe.directory`をprocess引数へ渡すだけでglobal Git設定は変更していない
- Task Scheduler `StockAI-Goal3-Research`を登録し、初回の独立起動を確認。`h5-lightgbm`はattempt 2の保存状態・実効状態とも`RUNNING`、worker identity一致、task状態`Running`であり、以後は周期監視せずrunnerへ委譲する

## 2026-09-04 Goal 4 / Goal 5 live capability再確認

- 独立runnerの学習process・PIDは照会せず、`stock-ai research morning-capabilities`と`stock-ai ops capabilities`だけをread-only実行した
- Goal 4は`morning_ohlc / intraday_bars / intraday_volume_profile / quotes / order_book / trade_frequency`がすべて`BLOCKED_BY_DATA_CAPABILITY`。前場market-data providerは未設定であり、日足やfixtureから推測しない
- `morning_model_adoption`も、認証済みlive OOS evidenceと承認済みmodel registry entryがないため`BLOCKED_BY_DATA_CAPABILITY`
- Goal 5は`operational_ledger / in_app_notifications`が`AVAILABLE`、remote accessは`LOCALHOST_ONLY`、web pushは`BLOCKED_BY_CONFIGURATION`
- 実運用開始には、Goal 3の固定Championと単回holdout結果、exact 11:30前場provider、local PWAでユーザーが手入力する実保有・account bucket・available / reserved cash・取得単価・税/NISA状態、cost/slippage方針、認証済みJPX Paper calendarが必要。SBI CSVはD068どおり必須ではない
- `order_submission`は`OUT_OF_SCOPE`であり、broker発注・自動売買は実装しない

## 2026-09-04 Goal 3選択固定・単回holdout基盤（隔離ブランチ）

- 実行中のlegacy campaignを変更・停止・再起動せず、次の安全なsource境界で使う実装を隔離branchへ追加した。現行campaign終了まではmainへ取り込まない
- v2 ablation / final-candidate campaignを全artifact hashまで再認証し、3 seed以上のfeature vote、全3 model family × 1 / 5 / 20日 × 4 taskの完全matrixを要求する
- purged development OOFだけからhorizon別feature、expected return、rank、downside quantile、large-loss model、全parameter、非負simplex ensemble weight、uncertaintyを選び、content-addressed selectionへ固定する。ensembleは同じmeta-evaluation区間のbest componentを上回った場合だけ採用する
- 選択後だけ使えるlocked holdout evaluatorを追加した。holdout読込前にselectionと唯一のledger path / evaluator commitを一度だけ結び付け、component単位のtargetなしprediction checkpointからresumeし、完了済みcomponentを再fitしない
- completed reportはselection / Build / Dataset / evaluator / feature definition / prediction hashを再認証する。選択・holdout結果はappend-only Experiment Registryへidempotentに保存し、task固有metricを欠損時0へ置換しない
- full JPX OOFのselection再読込は、最初に全bundleのlogical / Parquet / report hashを順次認証し、ensemble整列passでは同じParquet hashを読込前後に確認してregression / rankingの6列だけをpredicate読込する。quantile / large-loss rowの不要な再materializeを避けるが、認証・期間・行は省略しない
- holdoutの独立runnerとTask Scheduler登録入口を追加した。全development選択固定後の明示登録時だけ単回評価を開始し、Windows再起動後も同じledgerをresumeする。研究workerへAPI keyを渡さず、read-only statusは状態を変更しない
- fixtureで中断・resume・成功済みcomponent skip・改ざん・alternate root / commit拒否・selection schema不整合を検証した。実データlocked holdoutは未開封で、Champion候補もまだ固定していない
- 隔離branchの全品質gateはRuff / strict mypy（48 source files）/ Python 229 test / branch coverage 85.33% / frontend ESLint・TypeScript・Vitest 6件・production build / Microsoft Edge E2E 1件がpass。PowerShell 5.1で両holdout runner scriptの構文とTask Scheduler引数表示も検証した

## 2026-09-05 Goal 3 feature固定→candidate実行ブリッジ（隔離ブランチ）

- ablationのtuning-only seed voteを、horizon別exact feature列、F1〜F12 evidence、Build / Dataset / Feature / campaign / report / code / holdout境界とともにcontent-addressed `DevelopmentFeatureSelectionArtifact`へ先に固定する入口を追加した
- 固定artifactだけから1 / 5 / 20日それぞれのLightGBM / XGBoost / CatBoost × 3 seed以上をv2 campaignとしてrun / resumeし、完了後に全model・parameter・ensemble weight・uncertaintyを固定する二段階CLIを追加した
- candidate専用の独立runnerとWindows Task Scheduler登録scriptを追加した。Codex終了・Windows再起動後も同じmanifest、fold checkpoint、永続Optuna studyから自動再開し、API keyをworkerへ渡さず、locked holdoutを開かない。run再開時はPIDだけでなくPython process名と開始時刻を照合し、PID再利用を生存workerと誤認しない
- `research campaign-status` / `candidate-status`はread-onlyで、v2 campaignのhorizon / model / seedに加えactiveまたは最新のtask / foldとcheckpoint件数を返す。status実行前後にcampaign / progress byteが同一であることをfixture testで確認した
- 独立runnerが実際に使う`python -m stock_ai`入口を追加した。日付切替前後でもGoal 5 E2Eが未来時刻fixtureを作らないよう、APIへaware clock注入口を設け、E2EだけJSTの固定時刻を使用する。production既定は現在JSTのまま、naive clockは拒否する
- 品質gateはRuff / strict mypy（48 source files）/ Python 234 test / branch coverage 85.29% / frontend ESLint・TypeScript・Vitest 6件・production build / Microsoft Edge E2E 1件がpass。candidate / research / holdout runner全scriptはWindows PowerShell 5.1 parserを通し、candidateの実statusがread-onlyかつcredential非表示であることと、Task Schedulerの全研究config明示引数をWindows上で実行確認した
- 現行legacy campaignは変更・停止・再起動しておらず、この隔離sourceはその安全な完了境界後までmainへ取り込まない。実データfeature vote、candidate実行、Champion候補固定、locked holdout単回評価はまだ未実行
