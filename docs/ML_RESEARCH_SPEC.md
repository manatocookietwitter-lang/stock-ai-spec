# Machine Learning and Research Specification

更新日: 2026-08-22  
状態: v0.2

## 1. 機械学習の役割

機械学習は注文やActionを直接決めない。

各銘柄を一定期間保有した場合の期待値・下振れ・不確実性・前場修正を推定し、Daily Portfolio Decision Engineへ渡す。

主な出力:

- 1営業日期待超過リターン
- 5営業日期待超過リターン
- 20営業日期待超過リターン
- 将来リターン分位点
- 大幅下落確率
- 予測不確実性
- 前場情報による予測修正
- 流動性・売買コスト推定に必要な変数

## 2. 予測対象

### Alphaラベル

比較対象を複数持つ。

- 絶対リターン
- TOPIX超過リターン
- 業種超過リターン
- ベータ調整残差リターン
- コスト控除前リターン
- 研究用のコスト控除後リターン

主運用では、12:30を想定した入口と、1日・5日・20日先の出口を比較する。

### Rankingラベル

日付ごとの将来超過リターンを順位化する。

例:

```text
0: 下位20%
1: 20〜40%
2: 40〜60%
3: 60〜80%
4: 上位20%
```

上位Kだけを重視するRanking目的と、連続リターン予測を両方比較する。

### Downsideラベル

- 1日 / 5日 / 20日の最大下落
- 5日 / 20日の10%分位リターン
- -3% / -5%等の大幅下落フラグ
- 期間中最大ドローダウン
- CVaR近似

### Morningラベル

前日モデルで候補になった銘柄と現在保有株について、9:00〜11:30の情報が12:30以降の見通しをどう修正するかを学ぶ。

前場モデルは全銘柄を一から選び直すだけのモデルにしない。

## 3. 特徴量

特徴量候補の正本は `docs/FEATURE_CATALOG.md` とする。

本書では研究方針だけを固定し、個別指標、窓、派生、計算規則、Feature Set段階、必須テストはFeature Catalogを参照する。

### 3.1 段階別Feature Set

- `FeatureSet V0`: 最小Baseline
- `FeatureSet V1 Core`: 約40〜60特徴を目安とする最優先候補
- `FeatureSet V2 Extended Technical`: オシレーター、資金流入、ローソク足等の追加群
- `FeatureSet V3 Data-dependent`: 詳細財務、需給、分足・気配、TDnet等

候補プールは150〜300特徴程度になり得るが、全てをChampionモデルへ投入しない。

### 3.2 V1 Coreの中心

- 1 / 5 / 20 / 60 / 120日リターン
- SMA乖離 20 / 60 / 200、傾き、主要クロス
- 52週高値・安値距離
- MACD / Signal / Histogram / GC・DC
- RSI 14を中心とするRSI群
- Bollinger %B / Band Width
- ADX / +DI / -DI
- ATR / NATR
- 出来高比、売買代金、売買回転率、OBV派生、MFI
- PER / PBR / ROE / 営業利益率
- 売上・営業利益成長、会社予想修正
- TOPIX・業種相対強度
- 市場ボラティリティ・Breadth
- データ能力がある範囲の基本前場特徴

### 3.3 追加候補

Feature Catalogには次を含める。

- SMA / EMAの複数窓、Parabolic SAR、Aroon
- PPO、TRIX、Stochastic、Stoch RSI、CCI、Williams %R、Ultimate Oscillator、CMO
- Chaikin A/D、Chaikin Oscillator、CMF、PVO
- 詳細ボラティリティ、下方リスク、ギャップ、ブレイクアウト、ローソク足形状
- Valuation、Quality、Growth、業績修正・サプライズ
- TOPIX・業種・時価総額グループとの相対値、Beta、相関
- 信用、空売り、投資部門別売買、発行済株式数変化
- 9:00〜11:30の時刻別リターン、VWAP、出来高進捗、Spread、板・約定情報

### 3.4 直接売買ルールにしない

次を固定Actionへ変換しない。

```text
RSI > 70 だからSELL
MACDゴールデンクロスだからBUY
200日線より下だから必ずSKIP
```

指標は予測モデルの候補入力であり、最終株数とActionはDaily Portfolio Decision Engineが決める。

### 3.5 Feature選択

- Feature Family単位のAblationを行う
- 前処理、clip、順位化、重複削減、Feature選択は訓練fold内だけでfitする
- 相関、Permutation importance、SHAP、線形係数安定性は診断に使う
- 重要度だけで採用しない
- 複数fold・複数seed・コスト控除後Decision Engine結果で判断する
- 欠損率、履歴長、データ能力、計算安定性も採用条件にする
- 最終特徴数は固定しない。現時点の目安は約30〜100特徴

### 3.6 Feature Registry

全特徴について次を機械可読に保存する。

```text
name
family
version
inputs
parameters
formula_or_hash
implementation_and_version
warmup_period
output_unit
normalization
availability_rule
required_capabilities
```

Feature Set Manifestには定義hash、前処理版、データ能力、コードcommitを保存する。

## 4. モデル群

### Baselines

- Buy-and-hold / cash
- 単純Momentum
- Momentum + earnings revision
- 等加重ルールスコア
- Ridge
- Elastic Net
- Logistic Regression

### GBDT

- LightGBM Regression
- LightGBM Ranker
- XGBoost Regression
- XGBoost Ranker
- CatBoost Regression
- CatBoost Ranker
- Quantile Regression
- 大幅下落分類

### Multi-horizon

- 1日モデル
- 5日モデル
- 20日モデル
- 期間別の独立学習
- 期間統合はOOF予測だけで学習

### Morning model

- 前日予測を更新する回帰
- Trade / no-tradeのMeta-label比較
- 保有株のHOLD→REDUCE / SELLに有効か検証
- 新規候補のSKIP→BUYに有効か検証

Actionそのものを直接最終出力にしない。

Goal 4の初期Morning modelは、各horizonについて
`post-11:30 realized return - frozen daily forecast`をtargetにした予測修正回帰とする。daily forecastをそのまま使う
no-update baselineと同じvalidation rowで比較し、purged expanding OOFのMSEと日付別Rank ICが改善しないmodelは
`REJECTED`にする。固定parameterのRidge / LightGBMを比較し、small MLPは明示的に有効化したresearch challenger
としてだけ実行する。開発OOFは自動Champion昇格・本番採用の根拠にしない。

downside quantile、large-loss probability、standard errorの初期更新は、各予測11:30より前にlabel endを迎えた
だけでなく、実際に`label_available_at < prediction.as_of`となったOOF residual / outcomeだけを使う。未成熟label、
遅延訂正、同日後場、将来sessionのresidualは学習・較正へ入れない。
Morning labelは同一JPX sessionのexact 12:30から開始し、認証済み固定JPX calendar上の1 / 5 / 20 session後だけを
endpointにする。endpointは日付だけでなくaware `label_end_at`を持ち、availabilityがその時刻より前なら拒否する。
callerが与えたgeneric weekdayや早すぎるavailabilityを営業日・成熟済みとみなさない。
locked final holdoutへlabel endが跨るdevelopment rowをhorizon別にpurgeし、holdout rowはreport APIへ返さない。
current holding / candidate flagは監視roleであり、holdingをdatasetから除外してcandidateだけを採点しない。

small MLPを有効化する場合は1 / 5 / 20日すべてと最低3 seedを必須とし、全horizon / seedのOOF改善を満たさない限り
challenger全体を`REJECTED`にする。認証済みdevelopment report / datasetからのresearch-only再fitと、履歴終了後の
current 11:30 featureに対するoutcome-free inferenceは許すが、live OOS evidence、推論時間計測、model registry承認前に
production採用しない。
再fit bundleはreport / dataset / family / seed / training boundary / fitted state hashを束ねたresearch-only identityを持つ。
current featureは全freeze universe、role、provider、source ID、capabilityと完全一致し、20session profileを含む必須値が
実値で揃う場合だけ推論する。Decision Engine adapterはfeature content hashを再検証し、freeze roleと実Portfolio保有を
照合して、freeze済み11:30 price / liquidityとresearch lineageをproposalまで残す。

### Regime

まず市場状態を通常モデルの特徴量に入れる。

追加価値が確認できた場合だけ:

- 通常
- 高ボラ
- 急落

の専門モデルとSoft Gatingを比較する。

### Ensemble

- 各モデルのOOF予測を保存
- 日付内順位または標準化
- 非負・合計1の制約付きStacking
- モデル間予測相関
- 不一致ペナルティ
- 不確実性推定

### Neural challengers

GBDT完成後に限定して試す。

- small MLP
- 1D-CNN
- TCN
- GRU
- small Transformer

主用途は前場の時系列。

巨大モデルは使わず、複数シード・推論時間・安定性を必ず評価する。

### TDnet

任意の追加実験。

LLMに株価上昇を直接尋ねない。

抽出候補:

- 売上予想修正
- 営業利益予想修正
- EPS修正
- 上方 / 下方修正
- 増配 / 減配
- 自社株買い
- 増資・希薄化
- 特別利益・損失
- 減損
- 予想撤回

ルール抽出を優先し、LLM利用はキャッシュ・監査・無効化可能にする。本番提案の必須依存にしない。

## 5. 検証

### 分割

- ランダム分割禁止
- expanding walk-forward
- overlapping labelsのpurge
- horizon以上のembargo
- 最終holdoutを固定
- holdoutは最後まで特徴量・モデル・パラメータ選択に使わない

### モデル指標

- Spearman Rank IC
- ICIR
- NDCG@5 / 10 / 20
- Precision@K
- Top-K超過リターン
- Calibration
- Coverage
- 不確実性と誤差の対応
- モデル間不一致

### 戦略指標

Decision Engineを通した状態付き検証で評価する。

- 税・コスト控除後リターン
- Sharpe / Sortino
- 最大ドローダウン
- CVaR
- Profit Factor
- Turnover
- Cash Utilization
- Tax Drag
- Trading-cost Drag
- Replacement Gain
- Hold / Reduce / Sell Regret
- Decision Reversal
- 業種・時価総額集中
- 年・銘柄・相場依存

### コストシナリオ

最低限:

- 10 bps
- 20 bps
- 30 bps
- 50 bps
- no fill
- partial fill相当の実装誤差
- spread拡大
- slippage増加
- market impact増加

本システムは手動注文だが、提案性能の検証では実装可能性と価格差をストレスする。

## 6. 採用基準

複雑なモデルは次を満たす場合だけChampion候補にする。

- 複数walk-forward foldで改善
- realistic cost後でも改善
- 単純Baselineを上回る
- Turnoverだけで見かけの性能を作っていない
- 特定年・業種・銘柄に依存しない
- 複数seedで大きく崩れない
- 推論時間が11:30提案に間に合う
- Decision Engineを通した純改善がある
- 説明不能なデータ漏洩兆候がない

利益が出なかった場合も、正しい検証結果として保存する。

## 7. 実験管理

全実験に次を保存する。

```text
experiment_id
hypothesis
data_snapshot
feature_set
model
parameters
seed
folds
metrics
cost scenario
tax policy
Decision Engine version
adopted / rejected
rejection reason
```

失敗した実験を削除しない。

試行数を記録し、必要に応じてPBO・Deflated Sharpe Ratio等の多重試行対策を追加する。

## 7.1 Feature Family実験

`docs/FEATURE_CATALOG.md`のF0〜F14を、直前Champion Feature Setに対する増分として実行する。

主な比較群:

- MA・長期位置
- MACD
- RSI・オシレーター
- Bollinger
- ADX・DI
- Volatility・Downside
- Volume・Money Flow
- Valuation・Quality・Growth
- Revision・Surprise
- Relative・Market Context
- Supply / Demand
- Candle / Breakout
- Morning Core / Microstructure

全特徴を同時に足した一度の結果だけで採否を決めない。

## 8. 固定実験一覧

| ID | 実験 |
|---|---|
| E0 | Cash / hold baseline |
| E1 | 単純Momentum |
| E2 | Ridge / Elastic Net |
| E3 | LightGBM regression |
| E4 | LightGBM Ranker |
| E5 | XGBoost Ranker |
| E6 | CatBoost Ranker |
| E7 | FeatureSet V1 Core構築・ablation |
| E8 | FeatureSet V2 Extended Technical ablation |
| E9 | 1日 / 5日 / 20日統合 |
| E10 | Quantile / downside |
| E11 | 前場予測更新 |
| E12 | 保有株を含むMorning Meta-model |
| E13 | Market regime / soft gating |
| E14 | OOF ensemble |
| E15 | small MLP |
| E16 | TCN / 1D-CNN |
| E17 | GRU |
| E18 | small Transformer |
| E19 | FeatureSet V3 Data-dependent / TDnet structured features |
| E20 | 09:00 / 12:30 / 15:30比較 |
| E21 | 日次 / 週次判断比較 |
| E22 | Decision Engine variant比較 |
| E23 | 税・コスト・No-trade ablation |

## 9. 未来情報漏洩テスト

必須:

- 未来価格を変更しても過去特徴量が変わらない
- 開示前に財務値が現れない
- 後日の訂正が過去時点へ遡らない
- 現在の上場区分が過去ユニバースへ影響しない
- 11:30以降の値が11:30提案へ入らない
- 同時刻の約定に未確定価格を使わない
- ランダム分割APIを公開しない
- holdout評価が途中のモデル選択へ入らない
- Feature選択・clip・percentile・相関削減を検証期間でfitしない
- 当日後場・終値が11:30の日次テクニカルへ入らない
- 週次需給値が公表前の日へ前方補完されない
- 株式分割後の調整が過去時点の利用可能性を壊さない
- Feature Definitionまたはライブラリ版が変わった場合にfeature_versionが変わる
