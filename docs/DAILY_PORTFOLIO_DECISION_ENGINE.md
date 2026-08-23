# Daily Portfolio Decision Engine

更新日: 2026-08-22  
マイルストーン: M14  
状態: 中核仕様 v0.1

## 1. 目的

毎営業日11:30に、現在ポートフォリオをHOLDする場合と、保有調整・乗り換え・新規購入・現金化を組み合わせた代替ポートフォリオを比較し、12:30に向けた目標株数を提案する。

「予測順位上位10銘柄を買う」はBaselineであり、主目的ではない。

## 2. 入力

### 現在状態

- 口座ごとの保有銘柄
- 現在株数
- 平均取得単価
- 時価
- 含み益・含み損
- 利用可能現金
- 年内確定損益
- ユーザー入力の損失繰越
- NISA関連情報
- 未反映の実約定
- 現在の業種・銘柄集中

### 予測

各銘柄について:

- 1日 / 5日 / 20日期待超過リターン
- 下振れ分位点
- 大幅下落確率
- 不確実性
- 前場予測修正
- 流動性

### コスト

- Commission
- Spread
- Slippage
- Market Impact

### 税

- Account type
- Acquisition price
- Realized gain/loss YTD
- User-supplied loss carryforward
- Tax policy version
- NISA policy inputs

## 3. 判断宇宙

必ず含める:

- すべての現在保有銘柄
- 各口座バケット
- 新規候補
- 現金
- 反映待ち状態

新規候補の数は流動性と日次モデルで絞ってよい。

## 4. HOLD反実仮想

毎日の第一比較は、現在ポートフォリオを変えない場合である。

```text
HoldScenario
=
現在株数を維持
+ 新規取引コスト0
+ 即時実現税0
+ 現在構成の期待リターン・リスク
```

代替提案は必ずHoldScenarioを上回る必要がある。

## 5. 目的関数

概念式:

```text
NetUtility(Target)

=
Expected Future Wealth(Target)
- Portfolio Risk Penalty
- Downside Penalty
- Prediction Uncertainty Penalty
- Incremental Commission
- Incremental Spread
- Incremental Slippage
- Incremental Market Impact
- Estimated Immediate Tax Effect
- Turnover Penalty
```

比較:

```text
Net Improvement
=
NetUtility(Target)
- Utility(HoldScenario)
```

実装では同じコストや税を二重控除しないよう、各項の定義をデータ契約とテストで固定する。

## 6. 乗り換え判断

基本思想:

```text
乗り換え後の期待資産
- HOLDした場合の期待資産
- 売却側の増分コスト
- 購入側の増分コスト
- 増分推定税影響
- リスク・不確実性増加
```

B株の期待値からコストを引くだけではない。

AをHOLDする価値を反実仮想に置く。

単純な1対1交換だけでなく、次を同時比較する。

- Aを一部売ってBを買う
- Aを一部売ってBとCへ分散
- Aを維持して現金からBを買う
- Aを全売却して現金化
- 複数保有を縮小して1銘柄を追加
- 何もしない
- 全額または高比率を現金にする

## 7. No-trade Zone

毎日再評価しても、毎日取引を提案しない。

```text
Net Improvement
<= Minimum Improvement
   + Uncertainty Buffer
   + Implementation Buffer

→ HOLD / SKIP
```

CostとTaxはNetUtility内で控除する。閾値式で再度同じ値を二重に控除しない。

ノートレード帯は次を防ぐ。

- 小さな順位変化による頻繁な入替
- モデル誤差より小さい改善での売買
- 100株単位への丸めによる逆効果
- BUYとSELLの短期反転

## 8. 制約

初期の暫定値。すべて設定可能にする。

- 現物買いのみ
- 空売りなし
- レバレッジなし
- 最大保有銘柄数10
- 1銘柄最大10%
- 1業種最大30%
- 最低現金10%
- 100株単位
- 利用可能現金を超えない
- 売却可能株数を超えない
- 日次提案金額上限
- 流動性に対する注文金額上限
- 最低改善閾値
- 最大Turnover

現金は有効な目標配分である。

## 9. 連続最適化と株数化

内部で金額・比率を使ってよいが、最終提案は株数。

禁止:

- 単純な比率四捨五入だけで終了
- 現金不足を後処理で無視
- 100株化後に制約を再確認しない

推奨処理:

```text
1. 連続値の候補ポートフォリオ
2. 100株単位の候補集合を生成
3. 現金・口座・銘柄・業種制約を再評価
4. 離散候補のNetUtilityを比較
5. HOLDとの差が閾値以下なら取引なし
```

最適化アルゴリズムはBaselineと複数方式を比較し、性能・再現性・計算時間で選ぶ。

## 10. Action変換

口座バケット単位で判定する。

```text
target > current + band
→ BUY

abs(target - current) <= band
→ HOLD

0 < target < current - band
→ REDUCE

target == 0 and current > 0
→ SELL

target == 0 and current == 0
→ SKIP
```

100株単位ではbandは通常0株または設定された最小差分。

同一銘柄を複数口座で持つ場合:

- 銘柄集計表示
- 口座別の具体提案

を両方出す。

## 11. Transaction Cost Engine

### Commission

- Brokerとfee policyから計算
- SBIの無料条件はユーザー設定
- 0円をハードコードしない
- 条件未確認時は保守的設定

### Spread

基本候補:

```text
estimated_spread_cost
=
half_spread × shares
```

データ不足時は流動性区分に基づく保守的推定を使用し、推定であることを明示する。

### Slippage

- 銘柄のボラティリティ
- 注文方法
- 取引時刻
- 過去のユーザー実約定
- 前場終値と実約定差

から推定する。

### Market Impact

- 注文金額 / 通常売買代金
- 注文株数 / 通常出来高
- 流動性区分

を使う。

推定式はバージョン管理し、実約定から更新できるようにする。

## 12. Tax Engine

Tax Engineは機械学習と分離する。

出力:

- 売却による推定実現益・損
- 推定即時税影響
- 損失利用の推定影響
- NISA扱い
- Tax policy version
- 推定誤差・前提

最低限の入力:

- account type
- withholding mode
- shares
- acquisition price
- proposed sell shares
- expected sell price
- realized gain/loss YTD
- user-supplied loss carryforward
- NISA information
- effective date

税務申告額を断定せず、設定に基づく意思決定用推定と明示する。

### NISA

設定上の基本:

- 対象利益の即時税影響0
- NISA内損失の扱いは課税口座と別
- 枠の機会費用は任意パラメータ
- ユーザー入力がない枠情報を推測しない

## 13. 前場再評価

現在保有株と新規候補を両方再評価する。

可能な変化:

```text
HOLD → HOLD
HOLD → BUY
HOLD → REDUCE
HOLD → SELL
SKIP → BUY
SKIP → SKIP
```

前場悪化だけで機械的に売らず、HOLD反実仮想と全体最適化を通す。

## 14. 11:30以降

Alpha入力は11:30で凍結する。

本プロジェクトは実注文しないため、11:30以降は:

- 提案の表示
- データ時点警告
- 価格乖離警告
- ユーザー判断支援

に限定する。

12:30直前の大幅価格変化がある場合は、提案を無効または再確認必須として表示できるが、新しいAlphaを裏で混ぜて説明不能な判断に変えない。

## 15. 出力

各提案行:

```text
symbol
company
account_bucket
current_shares
recommended_shares
share_difference
action
current_market_value
recommended_market_value
estimated_cash_required_or_released
hold_expected_value
proposed_expected_value
estimated_commission
estimated_spread
estimated_slippage
estimated_market_impact
estimated_tax_effect
net_expected_improvement
downside_risk
uncertainty
reason_codes
human_readable_reasons
as_of
model_version
decision_engine_version
```

## 16. 説明

説明は予測寄与だけでなく、HOLDとの差を示す。

例:

```text
REDUCE 200株

主な理由
- 前場後の5日期待値が低下
- 下振れ寄与が上昇
- 同業種集中を縮小
- 売却税を考慮しても全体効用が改善

HOLD比の純改善推定: +xx円
```

数値の不確実性を隠さない。

## 17. 状態付きバックテスト

毎日、実際の状態を引き継ぐ。

```text
Day 1 current state
→ proposal
→ simulated/manual-like execution assumptions
→ resulting cash and positions
→ Day 2 current state
```

検証では提案時点の価格と実行価格差、100株制約、現金不足、部分的実装を反映する。

比較指標:

- Replacement Gain
- Hold Regret
- Reduce Regret
- Sell Regret
- No-trade Value
- Decision Reversal Rate
- Target-share Implementation Error
- Tax Drag
- Trading-cost Drag
- Turnover
- Cash Utilization

## 18. Baselines

- HOLD only
- Cash only
- Top-10 equal weight
- Weekly top-10
- Current-position unaware optimization
- Cost-free optimization
- Tax-free optimization
- No morning update
- Continuous weights without lot constraints

主エンジンはこれらと同じデータ期間・同じコスト前提で比較する。

## 19. 安全な失敗

次の場合は新しい提案を出さず、前回提案を再利用しない。

- 11:30時点のデータが欠損
- 現在保有が不明
- 利用可能現金が不明
- 口座区分が不明で税影響が重大
- モデル版が未承認
- コスト推定に必要な価格が欠損
- データ時点が古い
- 保有と実約定が不一致

UIへ「提案停止理由」を表示する。
