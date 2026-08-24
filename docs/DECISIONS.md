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

### D056 — Production buildは3 snapshotのatomic publication markerで完成とする

V0、V1 Core、Production Datasetはそれぞれcontent-addressed directoryとしてatomic publishし、三つすべての検証後に
Production Build Manifestを最後にatomic publishする。各snapshotは列名・dtype・値・source frame ID・manifest hashをidentityへ
含める。`data verify`はDuckDB catalog、immutable object、Bulk checkpoint、feature/dataset snapshot、build manifestを相互照合し、
空store、orphan snapshot、partial build、改ざんを成功扱いしない。

### D057 — Goal 2の外部blockと次Goalの進行

API keyが現在processへ継承されていない、契約planに必要endpointがない、providerがrevision historyを提供しない等の外部制約は
`BLOCKED_BY_*`または`PARTIAL`としてSTATUSへ固定する。推測データで埋めず、Goal 2の独立して検証可能な実装と品質gateを完了後、
recoverable checkpointを作り、Goal 3以降の外部制約に依存しない作業を継続する。
