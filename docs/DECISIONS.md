# Decisions and Open Questions

更新日: 2026-08-22  
状態: v0.2

## 確定事項

### D001 — 意思決定支援

本製品は自動売買ではない。AIは実注文しない。最終判断と注文はユーザーが行う。

### D002 — 提案時刻

毎営業日11:30時点で利用可能な情報を凍結し、12:30に向けた提案を作る。

### D003 — 中心目的

おすすめ銘柄ランキングではなく、現在の保有・現金・候補を同時評価して目標ポートフォリオを提案する。

### D004 — 判断対象

現在保有株を毎日必ず再評価する。新規候補だけを前場AIへ入れない。

### D005 — Action

最終Actionは:

- BUY
- HOLD
- REDUCE
- SELL
- SKIP

### D006 — 株数中心

内部では比率・金額を使えるが、画面は現在株数・推奨株数・増減株数を中心にする。

### D007 — 初期売買単位

初期版は100株単位。

### D008 — HOLD反実仮想

乗り換え判断は、代替案と「現在のままHOLD」を比較する。

### D009 — 全体最適化

単純なA→Bペア交換だけでなく、ポートフォリオ全体を同時最適化する。

### D010 — コスト

最低限:

- Commission
- Spread
- Slippage
- Market Impact

を考慮する。

SBIの手数料無料は設定条件であり、ハードコードしない。

### D011 — 税と口座

保有は `symbol × account_bucket` で管理する。

最低限:

- NISA / 特定口座等
- 取得単価
- 保有株数
- 含み益・損
- 年内確定損益
- ユーザー入力の損失繰越
- NISA入力情報

を扱う。

### D012 — 税影響

含み益銘柄を売る場合は、推定即時税影響をHOLDとの比較に含める。

### D013 — 機械学習の役割

LightGBM、XGBoost、CatBoost、前場AI、深層学習等は期待リターン・下振れ・不確実性を予測する。

最終株数を決めるのはDaily Portfolio Decision Engine。

### D014 — M14

M14の名称は `Daily Portfolio Decision Engine`。

### D015 — No-trade

毎日判断するが、毎日売買を提案しない。売買なし・全額現金も有効。

### D016 — 状態付き検証

バックテストは前日の現金・保有株を翌日へ引き継ぐ。

### D017 — 実取引反映

初期版はユーザーが約定株数・価格を手入力。将来CSV取込へ拡張。

### D018 — スマホの役割

スマホは確認・通知・入力・監視。機械学習本体をスマホのバックグラウンドで実行しない。

### D019 — Top-10

Top-10等加重はBaseline。最大10銘柄は暫定制約であり、目的ではない。

### D020 — Point-in-time

`available_at <= as_of` を必須にし、未来情報・生存者バイアスを防ぐ。

### D021 — モデル採用

高度なモデルはChallenger。単純モデルより安定して改善した場合だけ採用。

### D022 — Goal運用

製品全仕様をGoal本文へ詰め込まない。MDを正本にし、Goalは段階別に実行する。

### D023 — 主要画面の役割

- ホーム: 実資産と実保有
- 今日: 当日の株数変更の結論
- ランキング: 予測順位の探索
- 銘柄詳細: 理由とHOLD比較
- 検証: モデルとDecision Engineの実力
- 設定: 資金・口座・税・コスト・データ・モデル

### D024 — 提案・判断・約定の分離

AI提案、ユーザー判断、実際の注文・約定、Paper結果は別レコードとして保存し、互いに上書きしない。

### D025 — UIのAction表示

`current > 0` かつ `target = 0` は必ずSELL。REDUCEは0株より多く残る場合だけ使用する。

### D026 — ランキングの位置付け

ランキング上位は予測上の候補であり、最終BUYを意味しない。最終Actionは保有、現金、税、コスト、リスク、100株制約を含むDecision Engineが決める。

### D027 — UI仕様の正本

画面、ルート、操作、状態、表示データ、UI受入条件は `docs/UI_SPEC.md` を正本とし、`docs/ui-reference/` は非正本の参考画像とする。

### D028 — 広い特徴量候補プール

MACD、RSI、移動平均、Bollinger、ADX、ATR、出来高、財務、相対強度、需給、市場環境、前場指標を広い候補プールとして持つ。

個別候補と計算規則の正本は `docs/FEATURE_CATALOG.md`。

### D029 — 段階別Feature Set

特徴量は次の順で実装・検証する。

- FeatureSet V0: 最小Baseline
- FeatureSet V1 Core: 約40〜60の最優先候補
- FeatureSet V2 Extended Technical
- FeatureSet V3 Data-dependent

候補プール全体を一度にChampionモデルへ投入しない。

### D030 — 指標を固定売買ルールにしない

RSI、MACDクロス、移動平均クロス等は特徴量候補であり、直接BUY / SELLへ変換しない。

最終Actionは予測モデルの出力をDaily Portfolio Decision Engineが現在保有・税・コスト・現金と統合して生成する。

### D031 — Feature Family Ablation

特徴量の採否は、家族単位のOut-of-Sample Ablation、複数fold・複数seed、Decision Engineの税・コスト控除後改善で判断する。

SHAPや単一の重要度だけでは採用しない。

### D032 — Feature Definitionのバージョン管理

特徴量名、式、入力列、パラメータ、warm-up、正規化、利用可能時刻、実装名・版をFeature Registryへ保存する。

TA-Lib等のライブラリは実装手段であり、仕様の正本ではない。

### D033 — Feature選択も時系列分割内

clip、percentile、相関削減、Feature選択等は訓練fold内だけでfitする。検証期間や最終holdout全体を使って選択しない。

### D034 — Goal 1のBaseline境界

2026-08-24の最新Goal指示を、旧`CODEX_GOAL.md`テンプレートより優先する。

Goal 1ではMomentumとRidgeを実装し、LightGBM、Elastic Net、Logistic RegressionはGoal 2以降の比較対象に残す。fixture成績を収益性最適化へ使わない。

### D035 — Goal 1のテクニカル指標実装

FeatureSet V0 / V1 Coreの主要指標はNumPy / pandas上の小さな決定的実装と既知値テストを正本にする。TA-Lib等を将来採用する場合もFeature Definitionのversionを更新し、数値差分を検証する。

### D036 — Goal 1の離散探索境界

Daily Portfolio Decision EngineのMVPは、明示的candidate universe内の100株候補を全体同時評価する上限付き厳密列挙を使う。組合せ数が設定上限を超えた場合、未レビューの近似解へ無言で切り替えずfail closedする。大規模universe向けoptimizerは次Goalの課題とする。

### D037 — Goal 1の税計算精度

初期Tax Engineは、NISAの即時税影響0、課税口座の設定税率、年内確定損益、ユーザー入力の繰越損失を使う意思決定用推定とする。取得費・複雑な損益通算・申告確定額を断定せず、すべて推定として出力する。

### D038 — Goal 1ラベルの意味

実前場・12:30価格が未接続のGoal 1では、1日・5日・20日の調整後終値間absolute returnを研究proxyとして明示する。これは本番の12:30 entryやTOPIX/sector超過リターン予測を意味しない。label endpointの`available_at`がsnapshot cutoffを超える場合はlabelを保存しない。

### D039 — Sector / Breadthの入力境界

sector-relative値とmarket breadthはcandidate subsetから再計算しない。履歴時点で確定した明示的sector contextとmarket contextをFeature Engineへ渡し、coverageまたは必要capability不足時はfail closedする。

### D040 — 税の経済効果と決済現金

推定税負担、NISA機会費用、証券会社の推定源泉徴収による決済現金影響を別フィールドにする。同一口座bucketの複数売却は一括評価し、同じ損失繰越や年内利益を重複利用しない。

### D041 — J-Quants V2専用境界

日次公式データadapterは`https://api.jquants.com/v2`だけを許可し、V1 token、mail/password、refresh token、設定ファイル読込を実装しない。認証値はlive command起動時にprocess環境の`JQUANTS_API_KEY`からだけ読み、header・例外・log・manifest・fixture・Markdownへ値を出力しない。

### D042 — 初回取得以前へ訂正値を遡及させない

J-Quants REST responseは現在返される値の過去訂正時刻を提供しない。したがってGoal 2Aのexternal recordは`available_at = received_at`とし、初回backfillを過去の予測時点から利用可能だった値として扱わない。同一natural keyの後日訂正は新しいimmutable objectとして保存し、PIT queryはcutoffまでに受信した最新版だけを返す。初回取得以前の完全なrevision historyは`PARTIAL` capabilityとする。

### D043 — Content-addressed immutable object

raw / normalizedは`dataset + source_date + payload_hash + schema_version`から識別するimmutable Parquet objectとする。Parquetとmanifestを一時directoryへ完成させてからdirectory単位で公開し、同一payloadの再取得は重複させない。中断・品質失敗・catalog失敗は既存の正常objectを置換しない。

### D044 — Raw価格とresearch調整価格の分離

`/v2/equities/bars/daily`の`O/H/L/C/Vo`をexecution参照用raw系列、`AdjO/AdjH/AdjL/AdjC/AdjVo`をresearch用系列として別columnに保存する。`AdjFactor`とpayload由来の`adjustment_version`を保持し、両者を暗黙に上書きしない。

### D045 — Provider codeと内部symbol

J-Quants V2の5文字`Code`は`provider_code`として原値保存し、末尾のprovider桁を除いた4文字を内部`symbol`とする。数字・英字を含む新証券コードを許可し、整数化しない。

### D046 — Capabilityは推測で補完しない

利用planと実装範囲を`AVAILABLE / PARTIAL / BLOCKED_BY_PLAN / BLOCKED_BY_DATA_CAPABILITY / OUT_OF_SCOPE`で明示する。Goal 2Aでは財務summaryの期末発行株式数を日次株式数へ補間せず、breadthは全銘柄coverage検証前、historical universeは初回取得以前の訂正履歴欠落を理由に`PARTIAL`とする。

### D047 — Goal 2Aの取得単位

Free planでも安全に動く既定datasetは銘柄master、株価日足、財務summaryとする。Light以上の営業日calendarとTOPIXは明示選択する。全endpointは日付単位、逐次rate limit、`pagination_key`追跡、429/5xxのbounded retryで取得し、fixtureへのproduction fallbackは行わない。

### D048 — Bulk履歴のcheckpoint境界

Goal 2の履歴取得は公式J-Quants V2 `/bulk/list`・`/bulk/get`を使う。署名付きURLは保存せず、Bulk keyのhash、size、last-modifiedからfile fingerprintを作る。downloadしたCSVはprovider source dateごとに分割し、Goal 2Aと同じraw / normalized / quality pathへ通す。全date sliceのimmutable publish完了後だけfile checkpointを`SUCCEEDED`にし、`RUNNING`・`FAILED`はresume時に再取得する。

### D049 — Source revision vintageと研究上の利用可能時刻を分離

J-Quantsから受信したexternal recordの`available_at = received_at`はD042どおり維持し、訂正vintageのPIT境界とする。Production Research Datasetを構築する際は、このsource vintageを`source_snapshot_as_of`で固定したうえで、日足・TOPIX・集計contextを「次のJPX営業日11:30」、財務をproviderの`announced_at`で利用可能とするderived observationを別に作る。derived recordにはsourceの`revision_available_at`も残す。これは初回取得以前のprovider訂正履歴を復元するものではなく、単一取得vintageによるhistorical researchという制約を保持する。

### D050 — Goal 2 research universeと欠損exit policy

Goal 2の初期research universeは、各営業日のmaster snapshotに存在する東証Prime / Standard / Growth（market code 0111 / 0112 / 0113）の普通issue（5文字provider code末尾0）とする。master snapshotは営業日ごとの完全一致を必須とし、現在の構成を過去へbackfillしない。5・20日label endpointに価格がない場合は次の観測価格へずらさず、endpoint時点でuniverse外なら`DELISTED_NO_EXIT_PRICE`、universe内なら`SUSPENDED_NO_EXIT_PRICE`としてtargetを欠損にする。

### D051 — 12:30 labelの能力境界

正確な12:30 entry labelは、Premiumで提供されるafternoon-session adjusted openを取得できる場合だけ生成する。Standard以下、または当該列が欠損の場合は日足open/high/low/closeから推測せず`BLOCKED_BY_DATA_CAPABILITY`とする。研究用Decision Engine E2Eで前日raw closeをreference proxyとして使う場合も、12:30価格ではないことをreportへ明記する。

### D052 — Corporate Actionの扱い

研究価格はproviderの`AdjO/AdjH/AdjL/AdjC/AdjVo`と`AdjFactor`を使い、raw取引参照価格と分離する。`AdjFactor != 1`からeffective-date action lineageを生成し、payload hash由来のadjustment versionを保持する。daily endpointが提供しないannouncement時刻・action種別は推測せず欠損理由を残す。将来行の変更が過去特徴へ影響しないこと、分割時もadjusted系列が連続することを回帰testにする。

### D053 — PIT market / sector context coverage

Breadthと業種returnはcandidate集合ではなく、D050の営業日別PIT universe全体から計算する。eligible / observed issue数とcoverage ratioを保存し、既定95%未満の業種returnは欠損にする。閾値は実験設定としてversion化できるが、欠損銘柄を0 returnとして補完しない。売買停止等でsymbolの営業日観測が途切れた場合、technical rolling windowをそのgapでresetする。

## 暫定デフォルト

以下は現在の仮置きで、実験・設定により変更可能。

- 東証プライム普通株
- 最大10銘柄
- 1銘柄10%
- 1業種30%
- 最低現金10%
- 現物買いのみ
- 100株単位
- 1日 / 5日 / 20日予測
- 12:30を主対象
- J-Quants V2を日次公式データの初期アダプターにする

## 未決事項

### Q001 — 投資対象

- 東証プライムのみ
- 東証全普通株
- 流動性上位N銘柄

初期実験の対象を最終確定していない。

### Q002 — 日次候補数

前場監視へ渡す新規候補数。例: 30 / 50 / 100。

### Q003 — 前場ライブデータ源

J-Quantsの契約範囲、別の市場データ、証券ツール等から選ぶ必要がある。

SBIを注文先にしても、データ提供元は別でよい。

### Q004 — J-Quants契約プラン

履歴年数、前場・分足、財務詳細、TDnetの利用可能範囲。

### Q005 — 初期資金と口座構成

- 利用可能資金
- NISA / 特定口座
- 源泉徴収方式
- 現在保有
- 年内確定損益
- 損失繰越

実運用開始時に必要。

### Q006 — 制約値

- 最大保有銘柄数
- 1銘柄上限
- 1業種上限
- 最低現金
- 最大Turnover
- 日次提案金額上限
- 流動性上限

### Q007 — 税ルールの精度

簡易推定、年内損益通算、繰越損失、NISA枠の機会費用をどこまで初期版へ含めるか。

### Q008 — NISAで売る口座選択

同一銘柄をNISAと課税口座で持つ場合に、どちらをREDUCE / SELLするかのユーザー方針。

### Q009 — 予測期間の統合

1日 / 5日 / 20日の重みを手動・OOF学習・市場状態別のどれで決めるか。

### Q010 — コスト推定

Spread / Slippage / Market Impactを、最初は保守的固定値、流動性モデル、実約定学習のどれで始めるか。

### Q011 — 通知

- PWA
- メール
- LINE等
- OS通知

### Q012 — 実行環境

- 常時稼働Windows PC
- クラウド
- Windows + クラウド分離

### Q013 — リモートアクセス

PWAを自宅LAN、VPN、認証付きクラウドのどれで公開するか。

### Q014 — CSV形式

SBI等の保有・約定CSVの実ファイルを取得後にマッピングを確定する。

### Q015 — 提案のユーザー承認記録

ユーザーが提案を採用・一部採用・不採用にした理由を学習・検証へ使うか。

### Q016 — 12:30固定

12:30を主対象にするが、9:00 / 15:30と比較した結果で変更可能にするか。現状は比較実験を残す。

### Q017 — テクニカル実装ライブラリ（Goal 1方針確定）

Goal 1はD035のとおりNumPy / pandas実装とする。Goal 2以降にTA-Lib等を比較採用するかは未決。

### Q018 — V1 Coreの最終有効列

V1 Coreは約40〜60特徴を目安にするが、同じ指標の生値・順位・傾きのどれを初期有効化するかはM3の設計時にmanifestへ固定する。

### Q019 — 需給・市場Breadthのデータ源

信用、空売り、投資部門別売買、52週高安銘柄数、前場同時刻出来高履歴等の提供元と履歴範囲。

## 変更記録

### 2026-08-22

- 自動売買から意思決定支援へ変更
- BUY / HOLD / REDUCE / SELL / SKIPへ変更
- M14をDaily Portfolio Decision Engineへ変更
- 株数中心、100株単位へ変更
- 取引コストと税・口座区分を中核へ追加
- HOLD反実仮想と全体最適化を追加
- MD正本 + 段階別Goal方針を追加
- モバイルPWAの正式UI仕様と参考画像を追加
- AI提案・ユーザー判断・実約定・Paper結果のUI分離を確定
- 一般的なテクニカル・財務・需給・市場・前場指標を候補プールへ追加
- FeatureSet V0 / V1 / V2 / V3の段階導入を追加
- Feature Family Ablation、Feature Registry、数値fixture、fold内選択ルールを追加

### 2026-08-24

- Goal 1の最新指示に合わせ、BaselineをMomentum / Ridgeへ限定
- FeatureSet V0 / V1 Coreの初期manifestを58特徴で固定（cross後経過日数を含む）
- Goal 1のテクニカル実装をNumPy / pandas + 既知値testに決定
- 離散optimizerの上限超過をfail closedに決定
- Goal 1 Tax Engineの推定範囲を固定
- label availability、cross-sectional Rank IC、locked holdoutを追加
- sector / breadthをcandidate subsetから分離した明示入力へ変更
- 税の経済効果、NISA機会費用、源泉徴収cash effectを分離
- Goal 2AをJ-Quants V2専用・環境変数認証・immutable Parquet + DuckDBに固定
- 初回取得以前へprovider訂正値を遡及させず、`available_at = received_at`に固定
- 同一payloadのidempotent再取得、後日訂正の別version、atomic publishを固定
- plan / data capability不足を推測値で補わず明示状態に固定

### D054 — Historical revision policyをartifactへ固定する

J-Quants V2の初回backfillは、初回取得以前に存在した訂正vintageを復元できない。したがってProduction artifactは
`SINGLE_VINTAGE_AS_REVISED`または`STRICT_AS_KNOWN`を必ず記録する。前者は再現可能なas-revised研究に限り、
`historical_revision_status = PARTIAL`かつ`adoption_eligible = false`とする。後者ではsource receipt時刻を
effective availabilityへ含め、復元不能なhistorical labelを`BLOCKED_BY_REVISION_HISTORY`とする。どちらも完全な
correction-PITであるとは表示しない。

### D055 — Production targetは11:30より後に開始し、holdout境界でpurgeする

日足proxy targetのentryはfeature `as_of`と同じ営業日の引け、exitは固定JPX calendar上の1 / 5 / 20営業日後とする。
前日引けから11:30までに実現済みのreturnをtargetへ含めない。locked final holdout開始日以後にlabel endが入る行は、
feature日がholdout前でもdevelopmentから除外する。12:30 exact labelは行単位のentry/end/available_at/statusを持つ場合だけ
利用し、Standard planでは推測しない。

### D056 — Production buildはstaged featureとdatasetのatomic publication markerで完成とする

V0、V1 Core、V2 Extended、Production Datasetはそれぞれcontent-addressed directoryとしてatomic publishし、全snapshotの検証後に
Production Build Manifestを最後にatomic publishする。各snapshotは列名・dtype・値・source frame ID・manifest hashをidentityへ
含める。`data verify`はDuckDB catalog、immutable object、Bulk checkpoint、feature/dataset snapshot、build manifestを相互照合し、
空store、orphan snapshot、partial build、改ざんを成功扱いしない。

### D057 — Goal 2の外部blockと次Goalの進行

API keyが現在processへ継承されていない、契約planに必要endpointがない、providerがrevision historyを提供しない等の外部制約は
`BLOCKED_BY_*`または`PARTIAL`としてSTATUSへ固定する。推測データで埋めず、Goal 2の独立して検証可能な実装と品質gateを完了後、
recoverable checkpointを作り、Goal 3以降の外部制約に依存しない作業を継続する。

### D058 — Goal 3の選択・tuning・stackingはdevelopment OOFだけで行う

LightGBM / XGBoost / CatBoost、回帰 / ranking / quantile / large-loss、feature family ablation、Optuna、ensemble weight、
uncertainty calibrationはpurged expanding developmentだけで比較する。各horizonの前半をhyperparameter tuning専用、後半を
outer model evaluation OOF専用にし、outer targetが自身のmodel parameter選択へ入らないようにする。outer OOFはさらに時系列順に
stacking weight fit、uncertainty calibration、reported metric / coverage評価の3区間へ分離し、各境界をlabel endpointでpurgeする。
Decision互換の逐次uncertainty更新は、予測as-ofより前にlabelが成熟したresidualだけを使う。locked final holdoutはGoal 3 report APIへrow indexを返さず、
model・parameter・feature・ensemble選択へ使用しない。Goal 3 reportはrevision statusにかかわらず`adoption_eligible = false`とし、
最終holdout評価とlive/full-scale acceptanceを別工程にする。

### D059 — benchmark-excess予測をabsolute returnとしてDecision Engineへ渡さない

TOPIX / sector / beta-residual targetはresearch metricとして比較できるが、cash・cost・taxと比較するDaily Portfolio Decision Engineの
期待return fieldへそのまま代入しない。Decision Engine互換OOF出力はabsolute-return modelだけから生成する。OOF ensembleは
日付内rank spaceで非負・総和1 stackingし、その不確実性もrank spaceとして明示する。絶対return幅へのcalibrationはlive OOSで
別途検証する。

### D060 — Goal 3は認証済みV2 buildと明示的resource boundを必須とする

advanced researchは任意のParquet pathを直接信用せず、V0/V1/V2/Datasetを含む認証済みProduction Build Manifestから開始する。
Buildはsource lineageだけでなく、共通のobservation as-of・revision policy/statusと、Datasetへ埋め込んだ各feature値のsnapshot一致を必須にする。
config全文、feature snapshot/definition hash、全Optuna trial、fold結果、選択期間境界、library versionをartifactへ保存する。
全trialがFAILした場合もtrial監査を例外へ保持し、失敗ExperimentRecordへ永続化する。
full JPXで一括materializeする推定OOF行数またはmodel fit数が明示上限を超える場合は、未測定のままOOMへ進まず
`BLOCKED_BY_RESOURCE_CAPABILITY`で停止し、horizon/model-family単位のcontent-addressed batchとして実行する。

### D061 — Goal 4 Morning AIはprovider-neutralな11:30 freezeと日次予測の残差更新にする

前場data providerはQ003のまま未決とし、providerを推測・自動選択しない。実装済みの
`IntradayMorningData`は、`symbol / timestamp / price / volume / trading_value / available_at /
provider / source_record_id`を必須とする。`volume`と`trading_value`は各bar区間値であり、累積値は
同一session内だけで計算する。09:00 / 09:05 / 09:15 / 09:30 / 10:00 / 11:00 / 11:30の
exact barをF13 Morning Coreに必須とし、欠損時に直前値、日足OHLCV、日次出来高から推測しない。
current holdingとdaily candidateのunionを監視universeとしてfreeze metadataへ固定し、現在保有の欠落を
`BLOCKED_BY_DATA_CAPABILITY`にする。

F13は時刻別return、TOPIX/sector relative、range/VWAP、realized volatility、同時刻volume/trading-value
progress、監視対象内rank、daily prior forecastを持つ。同時刻profileは当日を除く直前20sessionだけを使う。
F14 Morning Microstructureはquotes / order book / trade frequencyの各capabilityが明示的に存在する列だけを使い、
不足列を0や合成値で補わない。当日終値と11:30後の値をmorning featureへ入れない。

Morning modelは前日・当日朝にfreeze済みの1 / 5 / 20日daily forecastに対する残差を予測する。
no-update daily forecastを必須baselineにし、固定parameterのRidge / LightGBMと、明示的に有効化したsmall MLPを
purged expanding development OOFで比較する。各modelはOOSでbaselineを改善しなければ`REJECTED`にし、
development結果から自動Champion昇格しない。1D-CNN / TCN / GRU / small Transformerは同期固定間隔の
前場sequence履歴がない間`BLOCKED_BY_DATA_CAPABILITY`、small MLPも既定`DISABLED`とする。

Decision Engine互換の更新値は、対象11:30より前にlabel endを迎え、かつ実際の`label_available_at`も対象11:30より前の
OOF residualだけでdownside、large-loss、standard errorを較正し、daily予測とmorning revisionの両provenanceを
型付き`Prediction`へ残す。ActionはMorning modelが
直接出力せず、更新後Predictionを全保有・候補とともにDaily Portfolio Decision Engineへ渡して初めて決める。
Morning Datasetとrow-level OOF reportはschema・値・source record ID・feature manifest・hashを含む
content-addressed Parquet + JSON directoryとしてatomic publishし、再利用前に認証する。live provider未接続時は
fixtureへfallbackせず、CLI capabilityを`BLOCKED_BY_DATA_CAPABILITY`として表示する。

各sessionの`MorningFreezeMetadata`はprovider、全source snapshot / record ID、holding + candidate universeを持ち、
roleとcapability reportを含めてfeature builder、dataset identity、Prediction batch、research-only proposalまで伝播する。
label entryはexact 12:30、endはartifactに認証した固定JPX calendar上の1 / 5 / 20 session後だけとする。
labelはendpointだけで成熟扱いせず、各fold学習と
Decision較正の双方で`label_available_at < prediction.as_of`を要求する。F14はcapability組合せごとのmanifestを作り、
AVAILABLE宣言済みの11:30値が欠けた場合はblockする。

認証済みreport / Datasetからresearch-only modelを決定的に再fitし、履歴終了後のcurrent freezeをoutcomeなしで推論する
経路を許す。Prediction provenanceはreport / snapshot / selected family / seedから導出し、caller自由文字列を使わない。
この経路は実装検証用で、live OOS evidence・推論時間・承認済みmodel registryが揃うまで採用不可である。

### D062 — Morning labelとDecision freezeは時刻・内容まで認証する

Morning labelは`label_end_date`だけで成熟判定せず、timezone-awareな`label_end_at`を必須にする。
`label_end_at`は認証済みJPX calendarのend sessionと同じ日で、entryより後、かつ
`label_available_at >= label_end_at`でなければならない。過去の取引時間を一律の時刻で推測しない。

current inferenceのPrediction batchは、11:30 feature行全体のcontent hashと、symbol別のreference price / ADVを
不変証跡として持つ。Decision adapterは同じfeature行を再hashし、freeze roleと実Portfolio保有、価格・ADVを照合する。
Decision Engineもaudit価格・ADVと評価Candidateを照合し、research-only proposalへ証跡を保存する。
履歴OOF replayをcurrent proposal用adapterへ渡すこと、認証済みrefit factoryを迂回したmodel bundle生成、
内容identityが一致しないDataset snapshotのresearch/refit利用は拒否する。

### D063 — Goal 5はlocal append-only台帳を実状態の正本にする

Goal 5のAI提案、version付きユーザー判断、実約定、翌日Portfolio、通知、ランキング、Paper将来観測は
SQLite WALの別recordとして保存し、同一identityの異なる内容を拒否する。提案は通知より先にarchiveし、翌日状態は
保存済みユーザー判断ではなく実際のfillだけから生成する。PWA/APIは`127.0.0.1`だけで提供し、mutating APIは
明示的なmanual-record intentを要求する。broker login、注文送信・変更・取消APIは設けない。

各提案は、生成に使った`DecisionEngineConfig`全文をproposal ID / generated-atへ結びつけた不変
`DecisionPolicySnapshot`と必ず同時に保存する。proposal payload、archive時刻証跡、policy snapshotは一つの
SQLite transactionで公開し、いずれかが欠ける場合はproposalを成功・表示可能にしない。Settingsはこの認証済み
snapshotだけを表示し、未登録の制約値をdefaultや推測値で埋めない。

CSV取込は必ずpreview→差分確認→appendとし、既存recordを上書きしない。約定の逆売買、注文株数、累積超過、
判断前時刻をpreviewで照合し、判断前時刻は確認でも上書きできない。口座状態CSVは非ゼロ保有の
`POSITION` rowに加え、全account bucketの`CASH` rowとavailable / reserved cashを必須とし、
保有だけ更新して古い現金を温存することを禁止する。
実broker固有CSV mappingは代表file取得まで決め打ちしない。

Paperは同一営業日の最終archive済み提案に対し、content-addressed不変台帳へ先に登録したJPX calendar、
horizon session経路、aware endpoint、実label availability、archive後の観測時刻、proposal model versionを
照合した実将来値だけを不変追加する。calendarはproposal archiveより前に固定済みであること、proposal営業日から
ちょうど次の1 / 5 / 20 sessionであること、proposal archiveがlabel endpointより前であることを必須とする。
同一営業日・horizonは1観測に限定する。週次・月次return、cost/tax推定誤差、
Champion/Challenger絶対誤差、隣接する過去窓のdrift比を集計するが、最低観測数到達やdrift判定からmodelを自動昇格しない。
運用curveとcompound returnは非重複の1営業日Paper系列だけを使い、5日・20日overlap labelを独立PnLとして複利計算しない。
model誤差・drift・Challenger比較はactiveな同一version cohortだけで計算する。Challengerは最新観測から連続する
同一versionの比較だけを使い、version切替または欠測をまたいで過去比較を再利用せず、比較分母を表示する。
PWA service workerはstatic shellだけをcacheし、`/api/` responseや前日の提案をoffline cacheから再表示しない。

運用backupは新規pathへonline SQLite backupし、全content hash・status→same-day archive済みproposal参照を検証後、
atomic no-replaceで公開する。restoreはPWA/job停止後、明示的な
replacement確認、restore元のread-only検証、restore後の再検証を必須とする。live handlerやdata capability不足時は
jobを`BLOCKED_BY_DATA_CAPABILITY`で停止し、fixture・前日提案・推測dataへfallbackしない。
jobはhandler前にdurable `RUNNING`を記録し、成功確定時にも同じprocessが未失効lockを所有することを再照合する。
freshness/proposal lineageをblock・failure・成功stage reuseで保持し、stable logical-stage idempotency key、heartbeat、同一workflowの
upstream成功を必須とする。APIはDailyStatusが指す提案だけを表示・操作可能にし、Host / Originをlocalhostへ限定する。

### D064 — V2 Bulk日足は公式file-download調整式で研究系列を再構成する

2026-08-25の実取得で、`/equities/bars/daily` RESTは`AdjO / AdjH / AdjL / AdjC / AdjVo`を返す一方、
公式Bulk CSVは仕様どおり`O / H / L / C / Vo / AdjFactor`だけを返すことを確認した。Bulk rawは受信列をそのまま保存し、
存在しない調整値を日次slice内で推測しない。Production build時に固定済みsource vintageの全期間を銘柄別・日付降順に並べ、
公式file-download仕様に従って「現在行より新しい日付の`AdjFactor`の累積積」を価格へ乗算し、出来高を除算する。
`ExRT = 3`のrights issueは公式注意事項どおり出来高側の累積係数から除外する。

再構成methodと全入力系列hashを`adjustment_source / adjustment_version`へ固定する。RESTの調整OHLCVが5列すべて存在する行は
provider値を使い、5列の一部だけが存在するrowはfail closedする。この処理は`SINGLE_VINTAGE_AS_REVISED`の制約を解消せず、
完全なhistorical revision PITや採用可能性を意味しない。

### D065 — 全期間Calendar Bulkは要求scope別にcheckpointする

実取得した`/markets/calendar` Bulk fileは、Bulk Listの要求期間にかかわらず2008-01-01〜2027-12-31の全期間rowを返した。
Calendarだけは要求`start / end`をfile fingerprintのcheckpoint scopeへ含め、要求範囲内rowだけを日付sliceへ公開する。
同一scopeの再実行は全object完全性を確認してskipし、別scopeを誤って完了扱いしない。Calendar以外のdatasetが要求範囲外rowを
返した場合は従来どおりfail closedする。

Production buildはCalendar全体をmaster / daily prices / TOPIXの共通する連続coverageへ境界化し、境界内部でどれか1系統の
営業日が欠ければ日付を黙って落とさず`BLOCKED_BY_DATA_CAPABILITY`にする。

### D066 — ホスト版は本人限定の運用確認面とし、local実データ基盤を置き換えない

ユーザーのremote URL要望を受け、Goal 5 localhost PWAとは別に、ChatGPT sign-inとSites access policyで
所有者本人だけが利用するホスト版control roomを設ける。これは公開マルチユーザー投資助言サービスではなく、
実データ準備状況、安全停止理由、no-order境界を確認する補助面である。broker接続、注文送信、変更、取消は実装しない。

`JQUANTS_API_KEY`、`data/live`、Production Dataset、model artifact、実口座・NISA・税状態CSVはホスト版へ送信しない。
ホスト版D1が保存できるのは認証済みuser IDに紐づく安全境界の確認状態など、機密な市場・口座dataではない
最小の運用metadataに限る。実データ・model・提案の正本は引き続きlocal append-only台帳と認証済みartifactであり、
安全な同期contractを別途承認・実装するまでホスト版は提案を生成せず、前日提案やfixtureを表示しない。

Sites deploymentはowner-only accessを既定とする。Internet公開、workspace全体共有、外部viewer追加は
別の明示承認なしに行わない。localhost FastAPIをreverse proxyでInternetへ公開せず、ホスト版workerから
local serviceへ到達させるための推測tunnelや代替APIも採用しない。
