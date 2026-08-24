# Data Contract

更新日: 2026-08-22  
状態: v0.2

## 1. 目的

市場データ、財務データ、前場データ、保有口座、予測、提案、実約定を、未来情報漏洩なく再現可能な形で保存する。

## 2. 共通メタデータ

外部データ由来のレコードには最低限次を持たせる。

```text
provider
source_endpoint
source_date
received_at
available_at
as_of
payload_hash
schema_version
ingestion_run_id
```

定義:

- `received_at`: システムが受信した日時
- `available_at`: 当時のユーザーが利用可能になった最早日時
- `as_of`: 予測・提案が参照する時点
- `payload_hash`: 同一データ判定と監査用

## 3. Point-in-time不変条件

- すべての特徴量は `available_at <= as_of`
- 現在の上場銘柄一覧を過去へ適用しない
- 上場前の銘柄を含めない
- 上場廃止銘柄を履歴から消さない
- 後から訂正された値を訂正前時点へ遡って使わない
- 財務情報は開示日だけでなく利用可能時刻を管理する
- 11:30提案では11:30より後の情報を使わない
- 同時刻で使えない価格を同時刻約定に使わない
- 株式分割・併合などの調整後研究価格と、実際の取引用価格を分離する

## 4. 保存層

```text
data/raw/
受信したままの不変データ

data/normalized/
型・列名・コード体系を統一したデータ

data/features/
特徴量スナップショット

data/datasets/
学習・検証用の固定スナップショット

artifacts/models/
学習済みモデルとモデルカード

artifacts/reports/
実験・バックテスト・提案レポート
```

推奨:

- raw: Parquet
- analytical catalog: DuckDB
- operational state: SQLite WAL

## 5. 主なエンティティ

### SecurityMaster

```text
symbol
company_name
security_type
market_segment
sector
listing_date
delisting_date
lot_size
valid_from
valid_to
```

### DailyMarketData

```text
symbol
trading_date
open
high
low
close
volume
trading_value
adjustment_factor
adjusted_open
adjusted_high
adjusted_low
adjusted_close
adjusted_volume
adjustment_method_version
provider metadata
```

テクニカル指標は、複数日系列の整合を保つ研究用調整OHLCVから計算する。実注文参考価格・金額計算には未調整の実価格を使う。

`adjusted_volume`を生成する場合は、株式分割等に対する調整式を明示し、provider値か自前計算かを記録する。推測できない場合は欠損にする。

### IntradayMorningData

```text
symbol
timestamp
price
volume
trading_value
vwap
bid
ask
spread
quote_state
provider metadata
```

分足・気配が提供されない場合は、取得できない列を推測しない。該当実験を能力不足として記録する。

Goal 4のprovider-neutral contractでは、上記に加えて最低限次を必須とする。

```text
available_at
provider
source_record_id
```

`volume / trading_value`は各bar区間値とし、session累積値と混同しない。09:00〜11:30 JST外のbar、
11:30 freeze後に受信したbar、naive timestamp、symbol/timestamp重複、非有限・非正price、負volume/valueは
publishしない。current holdingとcandidateのunionを`MorningFreezeMetadata`へ固定し、当日のbar universeと
完全一致を検証する。freezeにはprovider、source snapshot ID、全source record IDを固定し、stock / TOPIX / sector
barと完全一致させる。historical同時刻profile、quotes、order book、trade frequencyは独立capabilityであり、
日足から復元しない。

Morning supervised datasetとresearch reportはcontent-addressed directoryとして保存する。Dataset identityには
ordered schema / dtype / row values、全source record ID、11:30 feature manifest、provider、capability status、
固定JPX trading calendar全文、publication as-ofを含める。label entryは同一sessionのexact 12:30、1 / 5 / 20日endは
その認証済みcalendar上のsessionだけを許す。timezone-awareな`label_end_at`も認証し、availabilityがその時刻より
前なら拒否する。publication時点で未成熟のlabelは値をblankにしてstatusを残す。
OOF reportはdaily prior / morning revision / final prediction / label endをrow単位で保存し、Parquet hashとmetadata
hashを再読込時に検証する。
walk-forwardの学習・較正には`label_end_date < prediction date`と`label_available_at < prediction.as_of`の両方を必須とする。
global publication時点までに取得済みでも、各rowの11:30より後に受信したsourceはそのrowへ入れない。
freezeのholding / candidate roleとcapability reportはfeature rowへ完全一致させる。Prediction batchからDecision Engineへは
provider、全source snapshot / record ID、role、capability、research report ID、11:30 feature content hash、
reference price / liquidityを落とさず渡し、実Portfolio保有と評価Candidateへ再照合し、
Goal 4 development出力はproposalにも`research-only`として残す。

### MarketContextData

```text
trading_date_or_timestamp
market_id
index_open
index_high
index_low
index_close
index_volume
index_trading_value
advancing_issues
declining_issues
unchanged_issues
new_52week_high_count
new_52week_low_count
cross_sectional_return_dispersion
average_pairwise_correlation
sector_return_dispersion
provider metadata
```

利用できない市場Breadth・相関項目は、当時利用可能なPoint-in-time構成銘柄から計算し、計算対象ユニバースとfeature versionを保存する。

### SharesOutstandingSnapshot

```text
symbol
valid_from
valid_to
shares_outstanding
free_float_or_proxy
market_cap
available_at
provider metadata
```

売買回転率、時価総額グループ、発行済株式数変化等に使う。現在値を過去へ適用しない。

### SupplyDemandData

```text
symbol
observation_period
published_at
available_at
margin_long_balance
margin_short_balance
margin_ratio
short_sale_ratio
reported_short_position
investor_flow_category
investor_net_flow
shares_outstanding_change
provider metadata
```

週次・不定期値を日次へ利用する場合は、公表後だけ前方保持し、観測からの経過日数を別特徴として持つ。

### FinancialDisclosure

```text
symbol
period
disclosure_type
disclosed_at
available_at
revision_id
is_correction
revenue
operating_profit
net_income
eps
forecast values
provider metadata
```

### CorporateAction

```text
symbol
action_type
announcement_at
effective_date
ratio
cash_amount
provider metadata
```

### PositionLot

保有は銘柄だけでなく口座バケットごとに管理する。

```text
position_id
symbol
account_bucket_id
shares
average_acquisition_price
acquisition_date
market_price
market_value
unrealized_pnl
book_value
updated_at
```

### AccountBucket

```text
account_bucket_id
broker
account_type
withholding_mode
available_cash
reserved_cash
realized_gain_ytd
realized_loss_ytd
loss_carryforward_user_input
nisa_book_value
nisa_annual_capacity_user_input
nisa_lifetime_capacity_user_input
fee_policy_id
tax_policy_id
updated_at
```

対応口座は拡張可能にし、初期対象は少なくとも:

- NISA
- taxable_specified
- taxable_general
- unknown/manual

### ActualExecution

ユーザーが実際に行った売買を記録する。

```text
execution_id
executed_at
symbol
account_bucket_id
side
shares
price
commission
other_cost
tax_withheld
source
recorded_at
```

`source` は `manual`, `csv_import`, `statement_import` など。

### Prediction

```text
prediction_id
as_of
symbol
horizon
expected_excess_return
downside_quantiles
large_loss_probability
uncertainty
morning_revision
model_version
feature_version
data_snapshot_id
```

### PortfolioProposal

```text
proposal_id
as_of
generated_at
current_portfolio_snapshot_id
target_portfolio_snapshot_id
model_bundle_version
cost_policy_id
tax_policy_id
status
```

### ProposalLine

```text
proposal_id
symbol
account_bucket_id
current_shares
recommended_shares
share_difference
action
current_weight
target_weight
estimated_required_or_released_cash
hold_expected_value
proposed_expected_value
estimated_transaction_cost
estimated_tax_effect
net_expected_improvement
downside_contribution
uncertainty
reasons
```

### FeatureDefinition

```text
feature_name
feature_family
feature_version
stage
input_columns
parameters
formula_or_formula_hash
implementation_name
implementation_version
warmup_period
output_unit
normalization_options
availability_rule
required_capabilities
created_at
code_commit
```

個別指標の正本は `docs/FEATURE_CATALOG.md`。同名特徴でも式・窓・調整方法・実装版が変わればfeature versionを更新する。

### FeatureObservation

```text
as_of
symbol
feature_name
feature_version
value
is_missing
missing_reason
available_at
source_snapshot_ids
feature_run_id
```

欠損値を0へ暗黙変換しない。`missing_reason`は少なくとも履歴不足、通常欠損、能力不足、計算失敗を区別する。

### FeatureSetManifest

```text
feature_set_id
feature_set_version
feature_definition_hashes
preprocessing_version
required_capabilities
training_only_fit_rules
created_at
code_commit
manifest_hash
```

### ExperimentRun

```text
experiment_id
created_at
code_commit
config_hash
data_snapshot_id
feature_version
model_type
parameters
seed
fold_results
aggregate_results
decision
rejection_reason
```

失敗した実験も削除しない。

## 6. データ能力表

データ契約やプランによって利用可能範囲が異なるため、システムは能力を明示する。

```text
daily_prices
daily_adjusted_ohlcv
financial_summary
detailed_financials
company_forecasts
shares_outstanding_history
market_breadth
sector_history
margin_balance
short_sale_data
investor_flows
morning_ohlc
intraday_bars
intraday_volume_profile
quotes
order_book
trade_frequency
tdnet_text
historical_depth
live_delay
```

能力が不足する場合:

- 推測しない
- 偽データへ切り替えない
- 実験を `BLOCKED_BY_DATA_CAPABILITY` として記録する
- UIへ不足内容を表示する

## 7. データ品質

最低限の検査:

- 重複
- 日付欠落
- 型不一致
- 異常な価格・出来高
- 銘柄コード不整合
- 取引日不整合
- 調整係数不整合
- 財務訂正の時系列
- 取得途中失敗
- 古い最終更新
- 保有と実約定の不一致
- テクニカル指標のwarm-up不足
- 株式分割前後の調整OHLCV不整合
- Feature Definitionと出力列の不一致
- feature version / manifest hash不整合
- 週次需給データの公表前利用
- ライブラリ更新による特徴量値の予期しない変化

取得途中で失敗した場合、以前の有効スナップショットを破壊しない。

## 8. 欠損

- 欠損を無言で0にしない
- 元値と欠損フラグを分ける
- モデルごとの補完規則をバージョン管理する
- 本番推論で学習時に存在しなかった列欠落があれば失敗させる
- データ能力不足と通常の欠損を区別する

## 9. 秘密情報

- APIキーは環境変数またはOSの資格情報ストア
- フロントエンドへ配布しない
- ログ・例外・テストスナップショットへ含めない
- テストは決定的なfixtureを使う

## 10. 初期データ方針

- 公式日次データ用の初期アダプターはJ-Quants V2
- 前場ライブデータ提供元は未確定
- SBIは手動注文先の想定であり、市場データ提供元と同一である必要はない
- ユーザー保有情報は初期版では手入力またはCSV
- 将来の取引履歴・残高CSV取込を前提にデータ層を設計する
- 特徴量候補・計算規則は `docs/FEATURE_CATALOG.md` を正本にする
- TA-Lib等のライブラリは実装手段であり、Feature Definitionと数値fixtureを仕様の正本にする
