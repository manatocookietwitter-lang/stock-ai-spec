# Feature Candidate Catalog

更新日: 2026-08-22  
状態: v0.1  
位置付け: 候補プールの正本。最終採用特徴量一覧ではない。

## 1. 目的

本書は、株価予測モデルへ投入し得る一般的なテクニカル、価格・出来高、財務、需給、市場環境、前場特徴量を広く定義する。

重要な方針:

- 一般的に有名な指標は、まず候補として持つ。
- 有名であることを採用理由にはしない。
- 単純な売買ルールへ直結させない。
- 似た情報を持つ派生特徴を無制限に増やさない。
- 最終採用は、Point-in-timeのOut-of-Sample検証、特徴量群Ablation、安定性、コスト控除後のDecision Engine改善で決める。
- 候補プール全体は150〜300特徴程度になり得るが、Championモデルへ全て投入することを意味しない。
- 初期Champion候補は、データ能力に応じて約40〜60特徴の`FeatureSet V1 Core`から開始する。

機械学習は、RSIやMACDの閾値を固定売買命令として使うのではなく、他の特徴量と組み合わせて将来リターン・下振れ・不確実性を推定する。

## 2. Feature Setの段階

### FeatureSet V0 — 最小Baseline

複雑なテクニカル指標を追加する前の比較基準。

- 1 / 5 / 20 / 60 / 120日リターン
- 20 / 60日実現ボラティリティ
- 20日平均出来高比
- 売買代金・流動性
- 52週高値距離
- TOPIX相対5 / 20 / 60日
- 業種相対5 / 20 / 60日
- 基本財務成長・収益性・予想修正

### FeatureSet V1 Core — 最優先候補

最初から実装・検証する候補。

- リターン 1 / 5 / 20 / 60 / 120日
- SMA乖離 20 / 60 / 200
- 移動平均傾き・主要クロス
- 52週高値・安値距離
- MACD / Signal / Histogram / GC・DC
- RSI 14を中心とするRSI群
- Bollinger %B / Band Width
- ADX / +DI / -DI
- ATR / NATR
- 出来高比・売買代金・売買回転率
- OBV派生・MFI
- PER / PBR / ROE / 営業利益率
- 売上・営業利益成長
- 会社予想修正
- TOPIX・業種相対強度
- 市場ボラティリティ・Breadth
- 利用可能な基本前場特徴

目標は約40〜60特徴。全パラメータ派生を一度に有効化しない。

### FeatureSet V2 Extended Technical

V1に対する追加価値を特徴量群単位で検証する。

- 全SMA / EMA候補
- Parabolic SAR
- Aroon
- PPO / TRIX
- Stochastic / Stoch RSI
- CCI / Williams %R / Ultimate Oscillator / CMO
- Chaikin A/D / Chaikin Oscillator / CMF / PVO
- 詳細ローソク足形状
- 限定したローソク足パターン

### FeatureSet V3 Data-dependent

データ契約・履歴・Point-in-time品質が満たされる場合だけ追加する。

- 詳細BS / PL / CF
- FCF Yield / EV系指標 / ROIC相当
- 信用取引・空売り関連
- 投資部門別売買
- 発行済株式数変化
- 分足・気配・板・約定頻度
- TDnet構造化イベント

能力不足の場合は推測せず、`BLOCKED_BY_DATA_CAPABILITY`として記録する。

## 3. 共通計算ルール

### 3.1 As-ofと利用可能時刻

11:30提案で使う日次テクニカルは、原則として前営業日までに確定し、かつ`available_at <= 11:30`のデータから計算する。

当日9:00〜11:30の値動きは、日次テクニカルへ混ぜず、前場特徴量として別系統で計算する。

必須不変条件:

```text
feature.available_at <= proposal.as_of
```

### 3.2 研究価格と実価格

- 複数日をまたぐ価格・テクニカル計算: 株式分割等を整合させた研究用調整OHLCV
- 注文参考価格、売買金額、約定シミュレーション: 未調整の実価格
- 調整方法と係数はバージョン管理する
- 調整済み終値だけでなく、必要な指標ではOHLC全体の整合を取る
- 出来高の調整有無と方法も明示する

### 3.3 直接ルール化の禁止

次のような固定判断を製品ルールにしない。

```text
RSI > 70 だからSELL
MACDゴールデンクロスだからBUY
200日線より下だから必ずSKIP
```

これらはモデル入力、Ablation対象、説明候補であり、最終ActionはDaily Portfolio Decision Engineが決める。

### 3.4 クロスの定義

ゴールデンクロス例:

```text
short_t > long_t
and
short_t-1 <= long_t-1
```

デッドクロス例:

```text
short_t < long_t
and
short_t-1 >= long_t-1
```

クロス系は最低限次を分ける。

- 当日クロスフラグ
- 現在の上下関係
- 2系列間の距離
- 各系列の傾き
- クロス後経過営業日
- クロス時の価格・ボラ・出来高状態

過去にクロスが存在しない場合は0で埋めず、欠損と欠損フラグを持つ。

### 3.5 傾き

候補として次を比較する。

- 1日差分を価格または指標水準で正規化
- 5営業日のOLS傾き
- 20営業日のOLS傾き
- 日付内順位化した傾き

式、lookback、正規化方式をFeature Definitionへ保存する。

### 3.6 順位・標準化

各特徴量について、必要に応じて次を別候補にする。

- 生値
- 日付内市場順位
- 業種内順位
- 時価総額グループ内順位
- 自銘柄過去分布内のz-scoreまたはpercentile

前処理のfitは各訓練fold内だけで行い、検証・holdout全体を使って閾値や分位点を決めない。

### 3.7 欠損・Warm-up

- lookback不足を0にしない
- IPO直後など履歴不足を明示する
- 値と欠損フラグを分ける
- 最低履歴日数をFeature Definitionへ保存する
- データ欠損と契約能力不足を区別する
- 本番推論時の未定義列は失敗させる

### 3.8 実装ライブラリ

TA-Lib等は計算実装の候補であり、仕様の正本ではない。

各特徴量について最低限保存する。

```text
feature_name
feature_family
feature_version
input_columns
parameters
formula_or_formula_hash
implementation_name
implementation_version
warmup_period
output_unit
normalization
availability_rule
required_capabilities
```

主要指標は、決定的fixture上で独立実装または既知値と照合する。ライブラリ更新で過去特徴が無言で変わらないようにする。

## 4. トレンド・移動平均系

### 4.1 SMA

候補窓:

- SMA 5
- SMA 20
- SMA 25
- SMA 50
- SMA 60
- SMA 75
- SMA 100
- SMA 200

各窓について候補:

- SMA水準
- `close / SMA - 1`の乖離率
- SMAの1日変化
- SMAの5日・20日傾き
- 株価がSMAより上か下か
- SMAからの距離の日付内順位

### 4.2 EMA

候補窓:

- EMA 5
- EMA 12
- EMA 20
- EMA 26
- EMA 50
- EMA 200

各窓についてSMAと同様の乖離・傾き・上下関係を候補にする。

### 4.3 MAスプレッド・クロス

優先組合せ:

- SMA 5 / 20
- SMA 20 / 60
- SMA 25 / 75
- SMA 50 / 200
- EMA 12 / 26
- EMA 20 / 50
- EMA 50 / 200

候補出力:

- `short / long - 1`
- short slope
- long slope
- ゴールデンクロス
- デッドクロス
- クロス後経過営業日
- クロスの強さ
- クロス時出来高比
- クロス時ボラティリティ

すべての短期・長期組合せを総当たりしない。追加組合せは個別実験にする。

### 4.4 長期位置

- 200日線より上 / 下
- 200日線乖離率
- 50日線と200日線の距離
- 52週高値からの距離
- 52週安値からの距離
- 52週高値更新フラグ
- 52週安値更新フラグ

52週は初期実装では250営業日相当を標準とし、定義を設定可能にする。

### 4.5 その他トレンド

- Parabolic SAR
- SARと価格の距離
- SAR反転フラグ
- Aroon Up
- Aroon Down
- Aroon Oscillator

V2追加群として扱う。

## 5. MACD系

標準パラメータ:

```text
fast = 12
slow = 26
signal = 9
```

候補:

- MACD
- Signal
- Histogram
- MACDゴールデンクロス
- MACDデッドクロス
- クロス後経過営業日
- MACDが0より上 / 下
- MACDの0ライン上抜け / 下抜け
- MACDの1日・5日傾き
- Signalの1日・5日傾き
- Histogramの水準
- Histogramの1日・5日変化
- Histogramの符号
- MACD / close
- Signal / close
- Histogram / close
- MACD、Signal、Histogramの日付内順位

例:

```text
macd_gc = 1
macd_days_since_gc = 2
macd_histogram_pct_price = 0.008
macd_slope_5d > 0
macd_above_zero = 0
```

MACDクロス単独を売買ルールにしない。

## 6. RSI・過熱 / 売られすぎ

候補窓:

- RSI 7
- RSI 14
- RSI 21

候補:

- RSI水準
- 1日変化
- 5日変化
- 5日傾き
- RSI > 70
- RSI > 80
- RSI < 30
- RSI < 20
- 70を下回った直後
- 30を上回った直後
- 50ライン上抜け / 下抜け
- 過去60日・250日内percentile
- 日付内市場順位
- 業種内順位

V1ではRSI 14を中心にし、7・21の追加価値をAblationで確認する。

## 7. Bollinger Band

標準:

```text
lookback = 20
standard deviation = 2
```

候補:

- 中央線
- 上限2σ
- 下限2σ
- `%B = (close - lower) / (upper - lower)`
- Band Width
- +1σ / +2σ / +3σに対する位置
- -1σ / -2σ / -3σに対する位置
- 上限ブレイク
- 下限ブレイク
- バンド内へ戻ったフラグ
- Band Widthの1日・5日変化
- バンド幅拡大
- バンド幅縮小
- スクイーズpercentile
- %Bの日付内・自銘柄内順位

スクイーズの閾値は全期間固定値ではなく、訓練期間内の自銘柄過去分布等から決める候補を比較する。

## 8. モメンタム系

### 8.1 リターン

- 1日
- 3日
- 5日
- 10日
- 20日
- 60日
- 120日
- 250日

候補派生:

- 単純リターン
- 対数リターン
- 直近1か月を除く12か月相当モメンタム
- 日付内順位
- 業種内順位
- ボラティリティ調整モメンタム

### 8.2 その他

- ROC 5 / 10 / 20 / 60
- Momentum
- PPO
- PPO Signal / Histogram
- TRIX
- TRIX Signal

V1ではリターンを基礎にし、PPO・TRIXはV2追加群とする。

## 9. トレンド強度

標準候補:

- ADX 14
- +DI 14
- -DI 14
- DX 14

派生:

- +DI - -DI
- +DI / -DI比率
- +DI上抜け / 下抜け
- ADXの1日・5日変化
- ADX上昇中フラグ
- ADX > 20
- ADX > 25
- ADX > 30
- `price trend sign × ADX`
- ADXの日付内・自銘柄内順位

閾値フラグは補助特徴であり、固定売買ルールにしない。

## 10. オシレーター系

V2候補。RSIとの重複を前提に、特徴量群Ablationで追加価値を確認する。

- Stochastic %K
- Stochastic %D
- %K / %Dゴールデンクロス・デッドクロス
- Stoch RSI
- CCI
- Williams %R
- Ultimate Oscillator
- CMO

初期標準パラメータ候補:

```text
Stochastic: 14, 3, 3
Stoch RSI: 14, 3, 3
CCI: 20
Williams %R: 14
Ultimate Oscillator: 7, 14, 28
CMO: 14
```

各指標について水準、変化、傾き、閾値cross、日付内順位を必要最小限で比較する。

## 11. ボラティリティ・リスク

### 11.1 実現ボラティリティ

- 5日標準偏差
- 20日標準偏差
- 60日標準偏差
- 120日標準偏差
- 年率換算版
- 日付内順位
- 業種内順位

### 11.2 Range系

- True Range
- ATR 14
- NATR 14
- ATR / close
- Intraday Range `(high - low) / close`
- Parkinson等の高安ベース推定はV2比較候補

### 11.3 下振れ

- Downside volatility 20 / 60 / 120
- 20日最大ドローダウン
- 60日最大ドローダウン
- 120日最大ドローダウン
- 上昇日のボラティリティ
- 下落日のボラティリティ
- 上昇日 / 下落日ボラ比
- Overnight gap volatility
- Gap Down頻度
- 左裾の歪度・下方quantile候補

同一の下振れをラベルと特徴量の両方で扱う場合、lookbackは必ず過去方向だけにする。

## 12. 出来高・資金流入

### 12.1 基礎出来高

- 当日出来高
- 5日平均出来高
- 20日平均出来高
- 60日平均出来高
- 当日 / 20日平均
- 5日平均 / 60日平均
- 出来高z-score
- 出来高急増フラグ
- 売買代金
- 売買代金20 / 60日平均
- 売買代金順位
- 売買回転率
- 流動性percentile

11:30提案の日次特徴では「当日」は前営業日を指す。当日前場出来高は前場特徴へ分ける。

### 12.2 資金流入指標

- OBV
- OBVの5 / 20日傾き
- OBVの自銘柄z-score
- OBV / 平均出来高等の正規化候補
- Chaikin A/D
- Chaikin A/D傾き
- Chaikin Oscillator
- CMF 20
- MFI 14
- PVO
- PVO Signal / Histogram

OBVやA/Dの生水準は銘柄間比較に不向きな場合があるため、差分・傾き・自銘柄内標準化を優先する。

## 13. 値動きの位置・ローソク足

### 13.1 価格位置

- 当日高値からの距離
- 当日安値からの距離
- 終値の当日レンジ内位置
- 前日高値突破
- 前日安値割れ
- 20日高値突破
- 60日高値突破
- 52週高値更新
- 52週安値更新
- Gap Up
- Gap Down
- 窓の大きさ
- 窓を埋めたか

### 13.2 ローソク足形状

- 実体サイズ / close
- 実体サイズ / range
- 上ヒゲ比率
- 下ヒゲ比率
- 陽線 / 陰線
- 終値位置
- 連続上昇日数
- 連続下落日数
- 連続陽線・陰線日数

### 13.3 パターン

最初から全パターンを入れない。V2で限定比較する。

- Doji
- Bullish / Bearish Engulfing
- Three White Soldiers
- Three Black Crows

パターン名だけでなく、実体・ヒゲ・ギャップ等の連続特徴を優先する。

## 14. バリュエーション

Point-in-timeで利用可能な財務値・会社予想を用いる。

候補:

- PER
- 会社予想EPSベースForward PER相当
- PBR
- Earnings Yield `1 / PER`
- Book-to-price `1 / PBR`
- 配当利回り
- FCF Yield
- EV / EBITDA相当
- PSR

注意:

- 負の利益で意味が崩れる場合を別フラグ化する
- 逆数特徴は0除算・符号・クリップを明示する
- 市場・業種・時価総額グループ内順位を候補にする
- 詳細財務がない場合にEVやFCFを推測しない

## 15. 収益性・Quality

候補:

- ROE
- ROA
- ROIC相当
- 営業利益率
- 経常利益率
- 純利益率
- 粗利益率
- 営業CF Margin
- FCF Margin
- 自己資本比率
- Debt / Equity
- Net Debt
- Accruals候補
- 利益率の安定性
- ROE / 利益率の業種内順位

ROIC等は構成項目と計算式を明示し、データ能力に応じて無効化する。

## 16. 成長

候補:

- 売上YoY
- 営業利益YoY
- 経常利益YoY
- 純利益YoY
- EPS成長
- 3年売上成長
- 3年営業利益成長
- 3年EPS成長
- 営業利益率改善幅
- FCF成長
- 成長の加速・減速
- 市場・業種内順位

四半期比較では季節性を考慮し、原則として前年同期比を優先する。

## 17. 業績修正・サプライズ

重点候補:

- 売上予想修正率
- 営業利益予想修正率
- 経常利益予想修正率
- 純利益予想修正率
- EPS修正率
- 上方修正フラグ
- 下方修正フラグ
- 増配
- 減配
- 自社株買い
- 増資
- 希薄化
- 実績 vs 会社予想
- 実績 vs 前回会社予想
- 決算進捗率
- 過去予想誤差
- 決算発表からの営業日数
- 修正発表からの営業日数
- 訂正開示フラグ

開示日時と`available_at`を必ず使い、後日の訂正を過去へ遡って反映しない。

## 18. TOPIX・業種・時価総額との比較

### 18.1 相対リターン

- TOPIX相対1日
- TOPIX相対5日
- TOPIX相対20日
- TOPIX相対60日
- 業種相対1日
- 業種相対5日
- 業種相対20日
- 業種相対60日
- 時価総額グループ相対5 / 20 / 60日

### 18.2 横断順位

- 市場内順位
- 業種内順位
- 時価総額グループ内順位
- 各順位の1 / 5 / 20日変化

### 18.3 市場感応度

- Beta 60 / 120 / 250日
- TOPIXとの相関 20 / 60 / 120日
- Downside beta候補
- 残差モメンタム
- 業種残差モメンタム

Beta・相関は最低観測数、外れ値処理、推定窓を明示する。

## 19. 需給

データ能力がある場合のV3候補。

- 信用買い残
- 信用売り残
- 信用倍率
- 信用残の前週比
- 信用買い残 / 流通株式等の正規化候補
- 空売り比率
- 空売り残高
- 空売り残高変化
- 投資部門別売買
- 自社株買い実施・進捗
- 発行済株式数変化
- 浮動株変化候補

週次・不定期データを日次へ前方補完する場合、`available_at`以降だけ有効にし、観測からの経過日数も特徴として持つ。

## 20. 市場環境

個別銘柄特徴と別に、各提案日の市場状態を表す。

- TOPIX 1 / 5 / 20 / 60日リターン
- TOPIX 20 / 60日ボラティリティ
- TOPIX RSI
- TOPIX MACD / Signal / Histogram
- TOPIX MA乖離
- 上昇銘柄比率
- 下落銘柄比率
- 騰落レシオ候補
- 52週高値銘柄数・比率
- 52週安値銘柄数・比率
- 市場売買代金
- 業種間リターン分散
- 銘柄間リターン分散
- 銘柄間平均相関
- 市場Breadth
- 高ボラ相場フラグ
- 急落相場フラグ
- 流動性悪化フラグ

市場状態はまず通常モデルの入力に使う。専門モデルへのHard切替は追加価値が確認された場合だけ行う。

## 21. 前場用特徴量

当日9:00〜11:30の専用Feature Set。現在保有株と新規候補の双方へ適用する。

### 21.1 時刻別リターン

- 前日終値→9:00 Gap
- 9:00→9:05
- 9:00→9:15
- 9:00→9:30
- 9:00→10:00
- 9:00→11:00
- 9:00→11:30
- 各時点のTOPIX相対
- 各時点の業種相対

### 21.2 レンジ・VWAP

- 前場高値
- 前場安値
- 前場レンジ
- 前場実現ボラティリティ
- 前場VWAP
- 価格とVWAPの乖離
- 前引けの前場レンジ内位置
- 高値から前引けまでの下落
- 安値から前引けまでの反発

### 21.3 出来高・流動性

- 9:05 / 9:15 / 9:30 / 10:00 / 11:00 / 11:30時点出来高
- 過去同時刻平均に対する出来高進捗
- 売買代金進捗
- 前場候補内出来高順位
- Spread
- Spread percentile
- Bid / Ask
- Quote state
- 板偏り
- 約定頻度
- 無約定時間

同時刻平均を作るには過去の同粒度履歴が必要。履歴がない場合に日次出来高から推測しない。

### 21.4 前場評価変化

- 前日順位→11:30順位の変化
- 5日期待値修正
- 20日期待値修正
- 下振れ確率修正
- 不確実性修正
- 候補内順位
- 保有株内順位

予測修正そのものはMorning modelの出力であり、同じ日の最終ラベルを入力へ漏らさない。

## 22. Feature Family実験

全特徴を一度に追加せず、家族単位で比較する。

| ID | 追加群 | 目的 |
|---|---|---|
| F0 | V0基礎 | 最低Baseline |
| F1 | MA・長期位置 | トレンド情報の追加価値 |
| F2 | MACD | 複合トレンド・転換情報 |
| F3 | RSI・オシレーター | 過熱・反転情報 |
| F4 | Bollinger | 位置とボラ収縮・拡大 |
| F5 | ADX・DI | トレンド強度 |
| F6 | Volatility・Downside | リスク推定改善 |
| F7 | Volume・Money Flow | 資金流入・流動性 |
| F8 | Valuation・Quality・Growth | 財務情報 |
| F9 | Revision・Surprise | 業績変化 |
| F10 | Relative・Market Context | 市場共通要因除去 |
| F11 | Supply / Demand | 信用・空売り・発行株式 |
| F12 | Candle / Breakout | 値動き形状 |
| F13 | Morning Core | 前場確認 |
| F14 | Morning Microstructure | 気配・板・約定 |

各実験は、直前Champion Feature Setへの増分として評価する。全特徴を同時に足した結果だけで採否を決めない。

## 23. 重複整理と選択

### 23.1 重複候補

特に重複しやすい組合せ:

- RSI / Stoch RSI / CCI / Williams %R / CMO
- SMA乖離 / EMA乖離 / MACD / PPO
- ATR / NATR / Range / 実現ボラティリティ
- OBV / A/D / CMF / MFI
- PER / Earnings Yield
- PBR / Book-to-price

### 23.2 整理方法

- 訓練fold内だけで相関クラスタを作る
- ほぼ同一の式・パラメータを削る
- Spearman相関の高い候補を家族内で比較する
- 欠損率・履歴長・計算安定性を考慮する
- Permutation importance、SHAP、係数安定性を診断に使う
- SHAPや重要度だけで採用しない
- OOS Ablationで純改善を確認する
- Feature選択処理自体もfold内でfitする

### 23.3 採用条件

特徴量または特徴量群は、次を満たす場合だけChampion候補へ残す。

- 複数walk-forward foldで改善
- 複数seedで大きく崩れない
- Rank IC等だけでなくDecision Engineの税・コスト控除後結果も改善
- Turnover増加だけで見かけのリターンを作っていない
- 特定年・業種・少数銘柄への依存が弱い
- 欠損・データ遅延・プラン制約に耐える
- 説明可能なPoint-in-time経路を持つ

最終特徴数に固定目標を置かない。現時点の目安は約30〜100特徴だが、検証結果を優先する。

## 24. 必須テスト

### 24.1 数値テスト

- SMA / EMA既知値
- MACD 12-26-9既知値
- RSI 14既知値
- Bollinger %B / Width既知値
- ADX / DI既知値
- ATR / NATR既知値
- OBV / MFI既知値
- クロス当日・前日境界
- days-since-cross
- IPO等のwarm-up不足
- 株式分割前後の研究系列整合

### 24.2 Leakageテスト

- 将来価格を変更しても過去特徴が変わらない
- 当日後場・終値が11:30特徴へ入らない
- 開示前の財務・修正値が現れない
- 週次需給データが公表前に現れない
- 後日の訂正が過去特徴へ遡らない
- 訓練外期間を使ってpercentile・clip・相関削減をfitしない

### 24.3 再現性

- 同じsnapshot・version・configで同一値
- ライブラリ更新時に差分レポート
- Feature Set manifestのhash保存
- 並列実行でも順序依存しない

### 24.4 データ能力

- 必要列がなければ明示的にBLOCKED
- 欠損を別指標から推測しない
- TA-Lib等がなくてもproduction fallbackとして偽値を作らない
- 指標計算失敗で前回値を無言再利用しない

## 25. Feature Registry契約

各特徴量定義は機械可読なRegistryへ登録する。

例:

```yaml
name: tech.macd.histogram_pct_price
family: macd
version: 1
stage: v1_core
inputs:
  - adjusted_close
parameters:
  fast: 12
  slow: 26
  signal: 9
formula: "MACD_histogram / adjusted_close"
warmup_period: 35
output_unit: ratio
availability_rule: "completed_daily_bar_only"
normalization:
  - raw
  - cross_sectional_rank
required_capabilities:
  - daily_adjusted_ohlcv
```

Feature Set Manifestには次を保存する。

```text
feature_set_id
feature_set_version
feature_definition_hashes
preprocessing_version
required_capabilities
created_at
code_commit
```

## 26. UI説明への利用

銘柄詳細のプラス・マイナス要因は、モデル寄与と構造化特徴に基づく。

許可例:

```text
20日相対モメンタムが市場上位5%
MACD Histogramが5日連続で改善
前場出来高進捗が過去20日平均の1.8倍
```

禁止:

- 取得していない情報の生成
- 「MACDがGCだから必ず上がる」等の断定
- In-sample重要度だけを根拠にする
- 特徴量名だけを投資助言のように表示する

## 27. 参考実装資料

以下は指標実装・データ項目確認の参考であり、予測力を保証する根拠ではない。

- TA-Lib Functions: https://ta-lib.org/functions/
- TA-Lib Python function catalog: https://github.com/TA-Lib/ta-lib-python/blob/master/docs/funcs.md
- J-Quants: https://jpx-jquants.com/
- J-Quants Pro 財務情報関連: https://pro.jpx-jquants.com/datasets/5
- J-Quants Pro 株価・前場関連: https://pro.jpx-jquants.com/datasets/9

データ項目・契約範囲は実装時点の公式仕様を確認し、`Data Capability Table`を正とする。
