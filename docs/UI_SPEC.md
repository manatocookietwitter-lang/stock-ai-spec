# Mobile PWA UI Specification

更新日: 2026-08-22  
版: v0.2  
状態: Codex実装用の正式UI仕様  
対象: React + TypeScriptのモバイルファーストPWA

## 0. この文書の位置付け

この文書は、株式意思決定支援アプリの画面構成、表示内容、操作、状態、データ要件、UI受入条件を定義する。

製品目的、判断ロジック、機械学習、税・コスト計算の正本は次を参照する。

- `docs/MASTER_SPEC.md`
- `docs/DAILY_PORTFOLIO_DECISION_ENGINE.md`
- `docs/DATA_CONTRACT.md`
- `docs/ML_RESEARCH_SPEC.md`
- `docs/FEATURE_CATALOG.md`
- `docs/DECISIONS.md`

本書と参考画像が矛盾した場合は本書を優先する。参考画像は見た目と情報密度の参考であり、端末フレーム、iOSステータスバー、仮の数値、仮の会社名、仮のモデル名を実装仕様とはみなさない。

## 1. UIの目的

このアプリは自動売買アプリではない。

毎営業日11:30時点で作られたAI提案を、ユーザーがスマホで確認し、最終判断を保存し、証券会社で手動注文した結果を後から記録するための意思決定支援UIである。

UIの役割を次のように固定する。

| 画面 | 主な役割 |
|---|---|
| ホーム | 現在の実資産と保有状態を確認する |
| 今日 | 今日変更すべき株数の結論を見る |
| ランキング | 銘柄の予測順位を探索する |
| 銘柄詳細 | AI提案の理由とHOLD比較を確認する |
| 検証 | 予測モデルとDecision Engineの実力を確認する |
| 設定 | 資金・口座・税・コスト・データ・モデルを管理する |

「ホーム＝資産確認、今日＝結論、銘柄詳細＝理由、ランキング＝探索、検証＝実力確認」という役割を崩さない。

## 2. 非交渉のUIルール

- 実注文を送信、変更、取消するUIを作らない。
- 「判断を保存」は証券会社への注文ではない。
- AI提案、ユーザー判断、実際の約定、Paper結果を別データとして扱う。
- AI提案をユーザー操作で上書きしない。
- 最終表示は比率より株数を優先する。
- 初期版は100株単位で提案・入力する。
- 現在保有株を毎日必ず表示・評価対象に含める。
- 同一銘柄をNISAと特定口座で持つ場合は口座別提案を確認できるようにする。
- ランキング上位とBUY提案を同義にしない。
- データ更新時刻、提案生成時刻、モデル版を確認可能にする。
- 古い提案を当日の提案として自動再利用しない。
- 取得していない根拠やニュースを、もっともらしいAI文章として生成しない。
- エラー、欠損、時点不整合時は提案を表示せず、理由を明示する。

## 3. Actionの正式定義

Actionは口座バケット単位で判定する。

| Action | 現在株数 | 推奨株数 | 表示例 |
|---|---:|---:|---|
| BUY | 0株以上 | 現在より100株以上多い | `200株 → 300株 / BUY 100株` |
| HOLD | 1株以上 | 100株単位への最適化後に現在と同じ | `200株 → 200株 / HOLD` |
| REDUCE | 1株以上 | 0株より多く、現在より少ない | `300株 → 200株 / REDUCE 100株` |
| SELL | 1株以上 | 0株 | `200株 → 0株 / SELL 200株` |
| SKIP | 0株 | 0株 | `0株 → 0株 / SKIP` |

注意:

- `100株 → 0株` はREDUCEではなくSELL。
- BUYは新規購入と買い増しの両方を含む。
- HOLDは「良い銘柄」の意味ではなく、取引コスト・税・不確実性を含めると株数変更の価値が不足する状態も含む。
- SKIPはランキングが低いという意味だけではない。現金、業種上限、相関、税、コスト、100株単位などを含めた全体最適化の結果である。

## 4. 画面構成とルート

```text
下部ナビ
ホーム / 今日 / ランキング / 検証 / 設定

ホーム                         /
├─ 保有株一覧
└─ 銘柄詳細                    /stocks/:symbol

今日                           /today
├─ 判断を確認                  /today/review
├─ 保存済み判断                /today/decision
└─ 実行結果を記録              /today/executions

ランキング                     /ranking
└─ 銘柄詳細                    /stocks/:symbol

検証                           /validation
├─ 詳細検証                    /validation/details
└─ モデル詳細                  /models/:modelId

設定                           /settings
├─ 資金                        /settings/capital
├─ 証券会社                    /settings/broker
├─ 口座・税金                  /settings/accounts
├─ 取引コスト・制約            /settings/trading
├─ AI判断                      /settings/decision
├─ データ・運用状態            /settings/data
└─ モデル                      /settings/model
```

下部ナビは5項目で固定する。

- ホーム
- 今日
- ランキング
- 検証
- 設定

主要タブでは常時表示する。銘柄詳細、モデル詳細、設定下層は上部の戻る操作を主とし、下部ナビは非表示にしてよい。判断確認画面では下部ナビを表示し、主要ボタンをその直上に固定できる。

## 5. 共通の時間・状態モデル

アプリは日次処理状態を次のいずれかで表す。

```text
PRE_MARKET
MORNING_ANALYSIS
FREEZING_INPUTS
GENERATING_PROPOSAL
PROPOSAL_READY
USER_DECISION_SAVED
EXECUTION_PENDING
EXECUTION_RECORDED
MARKET_CLOSED
HOLIDAY
STALE_DATA
DATA_ERROR
MODEL_ERROR
```

ユーザー向け表示例:

| 内部状態 | 表示 |
|---|---|
| PRE_MARKET | 次回の前場分析を待っています |
| MORNING_ANALYSIS | 前場を分析中 |
| FREEZING_INPUTS | 11:30時点のデータを確定中 |
| GENERATING_PROPOSAL | 今日の提案を作成中 |
| PROPOSAL_READY | 今日の提案が完成しました |
| USER_DECISION_SAVED | あなたの判断を保存済み |
| EXECUTION_PENDING | 実行結果の記録待ち |
| EXECUTION_RECORDED | 実行結果を反映済み |
| HOLIDAY | 本日は休場日です |
| STALE_DATA | 最新データを確認できないため提案を停止しました |
| DATA_ERROR | データ取得に失敗しました |
| MODEL_ERROR | AI提案の作成に失敗しました |

当日の提案を生成できなかった場合、前日の提案を自動表示してはならない。履歴として見る場合は、日付と「過去の提案」を明確に表示する。

## 6. 共通デザインルール

### 6.1 色

| 用途 | 色の役割 |
|---|---|
| 背景 | 白またはごく薄いグレー |
| 基本文字 | 黒〜濃いグレー |
| 青 | 選択状態、HOLD、リンク、主要操作 |
| 緑 | 利益、BUY、良好 |
| 赤 | 損失、REDUCE、SELL、警告 |
| 灰 | SKIP、無効、補助情報 |

色だけで意味を伝えず、必ずラベルまたはアイコンを併用する。

### 6.2 禁止する見た目

- グラデーション
- AIチャット風の吹き出し
- 小さな数値ごとのカード乱用
- 過剰な影
- 派手な発光・ネオン
- 1画面内の複数の主要CTA
- 長い説明文を一覧へ常時表示
- iPhone端末枠やOSステータスバーの再現

### 6.3 レイアウト

- モバイルファースト。
- 360〜430px幅を基準にする。
- デスクトップでは中央寄せし、画面に応じて最大幅を拡張する。
- 一覧はカードではなく、見出し＋罫線区切りを基本にする。
- 大きなまとまりだけをセクションカードにする。
- 主要CTAは画面下部に置き、必要ならSafe Areaの上に固定する。
- タップ領域は44px以上を目安にする。
- 下部ナビはSafe Areaに対応する。
- 数値は桁が揃って見える設定を使う。
- 金額と株数は改行せず読みやすく保つ。

### 6.4 数値書式

- 円: `1,482,300円`
- 株数: `300株`
- 差分: `+100株` / `-200株`
- 損益: `+23,200円  +4.21%`
- 順位: `12位 / 486`
- パーセンタイル: `上位7%`
- 時刻: `11:36`
- 日付: `2026年8月24日（月）`
- 不明値: `—`
- 推定値には必要に応じて `推定` または `目安` を付ける。

## 7. 共通コンポーネント

最低限、次を再利用可能なコンポーネントとして設計する。

- `AppHeader`
- `BottomNavigation`
- `PageStatusBanner`
- `SegmentedControl`
- `ActionBadge`
- `MoneyValue`
- `ShareChange`
- `DataFreshness`
- `SectionHeader`
- `ListRow`
- `ProposalRow`
- `HoldingRow`
- `RankingRow`
- `MetricCard`
- `MetricTable`
- `ChartPanel`
- `InfoPopover`
- `ReviewQuantitySelector`
- `StickyPrimaryAction`
- `EmptyState`
- `ErrorState`
- `SkeletonState`

汎用カードコンポーネントをすべてに適用せず、リスト表示と大きなセクション表示を分ける。

# 8. ホーム

## 8.1 目的

現在の実際の資産、現金、保有株、資産推移を確認する。

AI提案だけでまだ約定していない売買を、実保有へ反映してはならない。

## 8.2 表示順

1. 日付・市場状態・更新時刻
2. 総資産
3. 今日の損益
4. 現金と株式の内訳
5. 資産推移
6. 保有株一覧
7. データ更新情報

## 8.3 ヘッダー

```text
8月24日（月）11:42
前場終了
```

右側または補助行に、現在の処理状態を表示する。

## 8.4 資産概要

```text
総資産
1,482,300円

今日
+12,400円  +0.84%
```

下に2列で表示する。

```text
現金
312,000円

株式
1,170,300円
```

## 8.5 資産推移

期間選択:

```text
1週間 / 1か月 / 3か月 / 1年 / 全期間
```

初期選択は1か月。

チャート下に次を表示する。

```text
現在
1,482,300円

期間損益
+82,300円  +5.88%
```

`TOPIXと比較`で比較線を追加できる。初期表示では資産線を主とし、比較線を常時詰め込まない。

## 8.6 保有株一覧

実際に保有している銘柄だけを表示する。

行の最低表示:

```text
トヨタ 7203
特定口座
200株
評価額 574,000円
取得比 +4.2%
HOLD
```

同一銘柄を複数口座で保有する場合:

- 銘柄合計の行を表示
- 展開すると口座別の株数とActionを表示
- 銘柄詳細では `合計 / NISA / 特定` を切替可能

一覧のActionは、当日有効な提案が存在するときだけ表示する。未生成、エラー、古い提案の場合はActionを表示せず状態を示す。

## 8.7 更新情報

```text
株価更新 11:42
保有状態更新 8/23 15:40
提案更新 11:36
```

株価、保有、提案の更新時刻を混同しない。

## 8.8 操作

- 保有株行をタップ → 銘柄詳細
- 期間をタップ → チャート更新
- TOPIX比較をタップ → 比較表示切替
- 「一覧を見る」 → 保有株一覧の全件表示

# 9. 今日

## 9.1 目的

11:30時点の最終提案から、「今日、何株変更するか」という結論だけを素早く確認する。

## 9.2 表示対象

初期表示では次を優先する。

1. SELL
2. REDUCE
3. BUY
4. HOLDは折りたたみ
5. SKIPは原則表示しない

SKIPはランキングまたは銘柄詳細で確認する。

## 9.3 ヘッダー

```text
8月24日（月）
更新 11:36
モデル ensemble-v12
```

モデル名は実データから表示し、モック値を固定しない。

## 9.4 提案行

```text
SELL
A社 / 特定
200株 → 0株
-200株
```

```text
REDUCE
ソニー / 特定
300株 → 200株
-100株
```

```text
BUY
B社 / NISA
0株 → 100株
+100株
```

```text
HOLD
トヨタ / 特定
200株 → 200株
変更なし
```

行タップで銘柄詳細へ移動する。

## 9.5 下部サマリー

```text
変更 3件

推定売却額 860,000円
推定購入額 281,000円
推定取引コスト 2,400円
推定税影響 18,600円
提案後現金 888,000円
```

変更件数はBUY、REDUCE、SELLの行数で数える。HOLDとSKIPは含めない。

## 9.6 CTA

```text
判断を確認
```

提案が未完成、古い、無効、エラーの場合は無効化し、理由を表示する。

## 9.7 状態別表示

- 分析前: `次回の分析を待っています`
- 前場中: `前場を分析中`
- 11:30直後: `今日の提案を作成中`
- 完成: 提案一覧
- 休場: `本日は休場日です`
- データ不足: `今日は提案を作成できませんでした`
- 古い提案: `この提案は当日データではありません`

# 10. 今日の判断確認

## 10.1 目的

AI提案とユーザーの最終判断を分離して保存する。

ここで保存しても注文しない。

## 10.2 行の構造

```text
A社 / 特定口座

AI提案
SELL 200株

あなたの判断
[ SELL 200株 ▼ ]
```

```text
ソニー / 特定口座

AI提案
REDUCE 100株

あなたの判断
[ 取引しない ▼ ]
```

```text
B社 / NISA

AI提案
BUY 100株

参考価格
2,810円

必要資金目安
281,000円

あなたの判断
[ BUY 100株 ▼ ]
```

## 10.3 選択肢

現在株数、100株単位、利用可能現金、口座、NISA枠、最大保有、売却可能株数を考慮して候補を生成する。

例:

- 取引しない
- BUY 100株
- BUY 200株
- REDUCE 100株
- REDUCE 200株
- SELL 全株

成立しない候補は表示しないか無効化し、理由を示す。

## 10.4 再計算

ユーザーが判断を変更するたび、次を再計算する。

```text
判断後の保有銘柄数
推定現金
推定株式評価額
推定購入額
推定売却額
推定取引コスト
推定税影響
制約違反
```

例:

```text
利用可能現金を83,000円超えています
B社の購入を100株減らしてください
```

## 10.5 保存

主要ボタン:

```text
判断を保存
```

直前または確認ダイアログに必ず表示する。

```text
保存しても証券会社への注文は行われません
```

保存後:

- AI提案は変更しない
- ユーザー判断を別レコードとして保存
- 修正保存時も履歴を残す
- 実行結果の記録待ちへ遷移できる

# 11. 実行結果を記録

## 11.1 目的

ユーザーが証券会社で手動注文した後、実際の注文・約定結果を記録し、翌日の保有状態へ反映する。

## 11.2 状態

- 未注文
- 注文済み・未約定
- 一部約定
- 全約定
- 取消
- 失効

## 11.3 入力項目

```text
銘柄
口座
ユーザー判断
実行状態
売買方向
約定株数
平均約定単価
約定日時
実売買手数料
その他費用
メモ（任意）
```

一部約定では複数約定を追加できる設計にし、表示用に加重平均を計算する。

## 11.4 重要なデータ分離

```text
AI提案
→ ユーザー判断
→ 実際の注文・約定
```

これらを別レコードとして保存し、互いに上書きしない。

翌日の保有状態はAI提案でもユーザー判断でもなく、確定した実行結果を基準にする。

# 12. ランキング

## 12.1 目的

モデルが予測した銘柄順位を探索する。

ランキングは最終売買提案ではない。

## 12.2 タブ

```text
総合 / 5日 / 20日 / 前場
```

総合は採用中アンサンブルの統合順位。各期間タブは対応モデルの順位。

## 12.3 検索

```text
銘柄名・コードで検索
```

銘柄名の部分一致と銘柄コード検索に対応する。

## 12.4 行表示

```text
1
A社 1234
総合1位 / 486
上位0.2%
未保有
BUY候補
```

```text
2
B社 5678
総合2位 / 486
上位0.4%
特定口座 200株保有
HOLD
```

表示する2種類の状態を分離する。

- 予測順位
- 現在のポートフォリオAction

3位でもSKIPは正常である。

## 12.5 BUY候補ラベル

`BUY候補`は予測順位上の候補であり、最終ActionのBUYとは別ラベルにする。

混同を避けるため、可能なら表示名を次のように区別する。

- `候補`
- `最終提案 BUY`

情報アイコン:

```text
順位は銘柄の予測評価です。
最終提案は保有、現金、税、コスト、リスクも考慮します。
```

## 12.6 ページング

初期は20〜50件単位で表示し、仮想化またはページングを使う。全486件を一度にDOMへ描画しない。

# 13. 銘柄詳細

## 13.1 目的

価格、保有、AI評価、提案株数、判断理由、現在維持との比較を確認する。

## 13.2 ヘッダー

```text
トヨタ 7203
2,870円
+32円  +1.13%
株価更新 11:42
```

戻るボタンを置く。お気に入りは任意機能であり、MVP必須ではない。

## 13.3 口座切替

同一銘柄を複数口座で持つ場合:

```text
合計 / NISA / 特定
```

合計では銘柄全体、各口座では具体的な提案を表示する。

## 13.4 保有情報

```text
現在保有 200株
平均取得 2,754円
評価額 574,000円
含み損益 +23,200円  +4.21%
口座 特定
```

未保有銘柄では保有情報を簡略化する。

## 13.5 AI提案

```text
AI提案 HOLD
現在 200株
推奨 200株
変更 なし
```

または:

```text
AI提案 REDUCE
現在 300株
推奨 200株
変更 -100株
```

## 13.6 評価サマリー

```text
総合 12位 / 486
5日 上位7%
20日 上位3%
前場 良好
下振れ 低
不確実性 中
```

前日からの変化も確認可能にする。

```text
前日評価 上位5%
11:30評価 上位7%
前場でやや低下
```

## 13.7 判断理由

主なプラス要因とマイナス要因を3件前後に絞る。

理由は次のいずれかに根拠を持つこと。

- モデル寄与値
- `docs/FEATURE_CATALOG.md`に登録されたテクニカル・出来高・財務・相対・前場特徴量
- 構造化された財務・市場特徴量
- 取得済みの開示情報
- Decision Engineの制約・コスト・税結果

取得していない事実をLLMに生成させない。MACD、RSI、移動平均クロス等を表示する場合も、「それだけで上昇する」と断定せず、登録済み特徴の観測値またはモデル寄与として表示する。

例:

```text
主なプラス要因
・20日相対モメンタムが強い
・営業利益予想の上方修正
・業種内順位が高い
```

```text
主なマイナス要因
・短期モメンタムが過熱
・PERの業種内順位が高い
・前場出来高が平常より弱い
```

## 13.8 HOLD比較

見出し:

```text
現在維持 vs 提案ポートフォリオ
```

銘柄単独のA→B交換だけに見せず、ポートフォリオ全体の比較を表示する。

```text
現在のまま保有       推定5日価値 +2.8%
提案ポートフォリオ   推定5日価値 +3.4%
売却コスト                         -0.1%
購入コスト                         -0.1%
推定税影響                         -0.3%
リスク差                           -0.0%
純改善                             +0.1%
最低改善基準                       +0.4%

→ HOLDを推奨
```

主要な乗り換え先がある場合だけ補足する。

```text
主な代替先
B社 100株
C社 100株
現金 82,000円
```

## 13.9 株価チャート

期間:

```text
1日 / 1週 / 1月 / 3月 / 1年
```

MVPでは価格推移を中心にし、複雑なテクニカル指標を詰め込まない。

# 14. 検証

## 14.1 目的

モデルの予測能力と、Decision Engineを含む最終提案能力を分けて確認する。

## 14.2 モード

```text
提案Paper / 実運用 / 過去検証
```

- 提案Paper: AI提案をすべて実行した仮想成績
- 実運用: ユーザー判断と実約定に基づく成績
- 過去検証: 状態付きウォークフォワードバックテスト

実運用データがない間はタブを無効または空状態にする。

## 14.3 期間

```text
1か月 / 3か月 / 1年 / 全期間
```

## 14.4 上部指標

```text
提案戦略 +12.35%
TOPIX +3.21%
超過 +9.14%
```

提案戦略が次のどちらかを明記する。

- 取引コスト控除後
- 推定税影響控除後

## 14.5 パフォーマンスチャート

- 提案戦略とTOPIXを比較
- 凡例を明示
- 0%線を表示
- 日付範囲と最終更新を表示
- 欠損期間を線で補間して隠さない

## 14.6 Decision Engine中心指標

```text
最大ドローダウン
純乗換改善
売買しなかった価値
Turnover
取引コスト影響
推定税影響
短期判断反転率
現金利用率
```

## 14.7 モデル診断

「AI上位10銘柄」は主成績ではなくモデル診断として表示する。

```text
予測上位10銘柄
平均5日後 +1.42%
勝率 62.0%
Rank IC 0.036
```

## 14.8 モデル状態

```text
LightGBM   採用中
XGBoost    比較中
CatBoost   比較中
前場モデル 採用中
深層学習   研究中
```

正式状態:

- CHAMPION
- ACTIVE_COMPONENT
- CHALLENGER
- RESEARCH
- REJECTED
- DISABLED
- BLOCKED_BY_DATA

色と短い日本語ラベルを併用する。

## 14.9 詳細検証

詳細画面では次を表示可能にする。

- fold別成績
- 年別成績
- 業種別成績
- コスト感応度
- 税あり・なし比較
- 前場モデルあり・なし
- HOLD baseline比較
- 上位10等加重baseline比較
- モデル版、データ版、特徴量版

# 15. 設定

## 15.1 設定トップ

設定トップはセクション一覧と現在値を簡潔に表示する。

### 資金

```text
運用対象資金 1,500,000円
最低現金 10%
1日提案金額上限 500,000円
```

### 証券会社

```text
証券会社 SBI証券
```

証券会社ログインや注文機能は作らない。

### 口座・税金

```text
特定口座 使用
NISA 使用
```

下層で次を設定する。

- 源泉徴収方式
- 年内確定利益
- 年内確定損失
- 繰越損失
- NISA成長投資枠残
- NISA非課税保有限度額関連のユーザー入力
- 税ポリシー版

税額は意思決定用の推定であることを表示する。

### 取引コスト・制約

```text
売買手数料 0円
手数料設定 SBIゼロ革命
条件確認済み
単元 100株
最大保有 10銘柄
1銘柄上限 10%
1業種上限 30%
取引コスト推定 標準
```

SBI無料条件をユーザーが確認した日を記録可能にする。0円を固定値にしない。

下層では次を設定・確認する。

- Commission policy
- Spread model
- Slippage model
- Market impact model
- 最大Turnover
- 流動性上限
- 最低改善閾値
- 不確実性バッファ

### AI判断

```text
判断時刻 11:30
推奨取引時刻 12:30 後場寄り
提案方式 Daily Portfolio Decision Engine
```

`第一売買候補`ではなく`推奨取引時刻`を使う。

### データ・運用状態

```text
J-Quants 接続済み
日次データ 8/24 01:42
前場データ 8/24 11:30
最終提案 8/24 11:36
自動処理 正常
```

詳細では、データ能力、欠損、エラー履歴、最終成功ジョブを確認可能にする。

APIキー自体を画面へ表示しない。

### モデル

```text
Champion ensemble-v12
学習日 8/1
学習データ最終日 7/31
検証状態 承認済み
```

モデル名は実データから表示する。

# 16. フロントエンドデータ契約

バックエンドの正式契約は `docs/DATA_CONTRACT.md` を優先する。UIは最低限次の形のデータを受け取れるようにする。

## 16.1 AppStatus

```text
businessDate
marketState
pipelineState
dataAsOf
morningDataAsOf
proposalGeneratedAt
portfolioUpdatedAt
modelBundleVersion
isStale
blockingReason
```

## 16.2 PortfolioSummary

```text
asOf
totalAssets
cashValue
equityValue
dailyPnlAmount
dailyPnlPct
periodPnlAmount
periodPnlPct
holdingsCount
```

## 16.3 HoldingView

```text
symbol
companyName
accountBucketId
accountType
shares
averageAcquisitionPrice
currentPrice
marketValue
unrealizedPnlAmount
unrealizedPnlPct
latestAction
recommendedShares
proposalId
```

## 16.4 DailyProposal

```text
proposalId
asOf
generatedAt
modelBundleVersion
status
lines
estimatedSellValue
estimatedBuyValue
estimatedTransactionCost
estimatedTaxEffect
estimatedCashAfter
changeCount
```

## 16.5 ProposalLine

```text
symbol
companyName
accountBucketId
accountType
currentShares
recommendedShares
shareDifference
action
referencePrice
estimatedRequiredOrReleasedCash
holdExpectedValue
proposedExpectedValue
estimatedTransactionCost
estimatedTaxEffect
netExpectedImprovement
minimumImprovementThreshold
downsideLevel
uncertaintyLevel
positiveReasons
negativeReasons
```

## 16.6 UserDecision

```text
decisionId
proposalId
version
savedAt
lines
estimatedCashAfter
estimatedTransactionCost
estimatedTaxEffect
constraintViolations
```

各行:

```text
proposalLineId
selectedAction
selectedShareDifference
selectedTargetShares
userNote
```

## 16.7 ExecutionRecord

```text
executionRecordId
decisionId
symbol
accountBucketId
status
side
orderedShares
filledShares
averageFillPrice
executedAt
actualCommission
actualOtherCost
source
```

## 16.8 RankingRow

```text
symbol
companyName
rankType
rank
totalUniverse
percentile
candidateStatus
portfolioAction
currentShares
accountTypes
updatedAt
```

## 16.9 ValidationSummary

```text
mode
period
strategyReturn
topixReturn
excessReturn
maxDrawdown
turnover
replacementGain
noTradeValue
costDrag
taxDrag
decisionReversalRate
cashUtilization
rankingDiagnostics
modelStatuses
updatedAt
```

# 17. エラー・空状態

すべての主要画面に、loading、empty、error、staleの状態を実装する。

## 17.1 データ不足

```text
必要なデータが揃っていないため、今日の提案を作成できませんでした
不足: 前場出来高データ
```

## 17.2 モデル失敗

```text
AI提案の作成に失敗しました
前日の提案は再利用していません
```

## 17.3 保有未登録

```text
保有株が登録されていません
手入力またはCSV取込で登録してください
```

## 17.4 実運用履歴なし

```text
実運用データはまだありません
実行結果を記録すると表示されます
```

## 17.5 制約超過

既存保有が最大10銘柄を超えていてもデータを拒否しない。

```text
現在の保有は設定上限を18銘柄超えています
全保有を評価し、段階的な調整案を作成します
```

# 18. アクセシビリティ

- Actionは色だけで区別しない。
- チャートには数値要約を付ける。
- フォーカス表示を消さない。
- キーボード操作が可能なコンポーネントを使う。
- アイコンのみのボタンにはラベルを付ける。
- 小さな補助文字でも可読性を保つ。
- 損益のプラス・マイナスを色と符号の両方で示す。
- 動きの多いアニメーションを使わない。

# 19. レスポンシブ動作

## モバイル

- 1列中心
- 下部ナビ固定
- チャートは横幅100%
- 判断確認の主要CTAは下部固定可

## タブレット・デスクトップ

- ホームは資産概要と保有一覧を2カラムにしてよい
- 検証はチャートと指標を2カラムにしてよい
- 最大幅を設け、全幅に引き伸ばさない
- 下部ナビは維持しても、左ナビへ切替してもよいが、ルートと名称は変えない

# 20. 実装しないもの

- 証券会社への注文送信
- 注文取消・訂正
- 証券会社ログイン
- AIチャット画面
- ニュースフィード
- SNS投稿
- 信用取引・空売りUI
- レバレッジ設定
- 高頻度取引画面
- 税務申告書作成
- 取得していない理由の自由生成
- モデルをユーザーが自由に本番昇格する機能

# 21. 参考画像

参考画像は `docs/ui-reference/` に置く。

| 画像 | 対応画面 |
|---|---|
| `home.png` | ホーム |
| `today.png` | 今日 |
| `decision-review.png` | 今日の判断確認 |
| `ranking.png` | ランキング |
| `stock-detail.png` | 銘柄詳細 |
| `validation.png` | 検証 |
| `settings.png` | 設定 |

参考画像の誤りやモック上の矛盾は本書で修正済み。

主な修正:

- `100株 → 0株` はREDUCEではなくSELL
- 最大保有10銘柄と保有28銘柄が混在するモック値は統一しない。実際には制約超過状態として扱う
- ランキング上位を最終BUYとみなさない
- 一対一の乗換比較ではなく全体ポートフォリオ比較を中心にする
- 検証は上位10銘柄だけでなくDecision Engineの税・コスト控除後成績を中心にする
- 判断保存後の実行結果記録を追加する

# 22. Codex実装順

UI実装は次の順で進める。

1. ルート、共通レイアウト、BottomNavigation
2. ActionBadge、数値書式、更新時刻、共通状態
3. Today + fixture API
4. Decision Review + 制約再計算
5. Execution Record
6. Home
7. Stock Detail
8. Ranking
9. Validation
10. Settings
11. エラー・空状態
12. レスポンシブ・アクセシビリティ
13. API統合
14. visual reviewと受入テスト

最初から全画面を静的モックとして量産せず、Todayのデータフローを通してから横展開する。

# 23. 必須テスト

## 23.1 コンポーネント

- Action定義と色・ラベル
- 金額・株数・損益書式
- 100株単位のselector
- loading / empty / error / stale
- 同一銘柄の複数口座表示

## 23.2 画面

- TodayのAction別グルーピング
- HOLD折りたたみ
- SKIPがToday初期一覧へ出ない
- 判断変更時の現金・税・コスト再計算
- 制約違反表示
- 保存しても注文されないこと
- 実行結果が翌日保有へ反映されること
- ランキング順位とActionが独立していること
- 銘柄詳細のHOLD比較
- 検証のPaper / 実運用 / 過去検証分離
- 設定の秘密情報非表示

## 23.3 E2E

```text
11:30提案生成済みfixture
→ Today表示
→ 判断確認
→ AI提案を一部変更
→ 判断保存
→ 実行結果を一部約定として記録
→ 翌日のホーム保有へ反映
```

## 23.4 安全性

- order、submit order、cancel orderに相当するAPI呼出が存在しない
- `判断を保存`で外部証券サービスへ通信しない
- 古い提案を当日提案として表示しない
- データ欠損時に架空値を表示しない
- APIキーがブラウザbundleやログへ含まれない

# 24. UI受入条件

Codexは次を満たすまでUIマイルストーンを完了扱いにしない。

- 5つの下部ナビが動作する
- 全主要ルートが直接開ける
- Todayが株数中心で表示される
- BUY / HOLD / REDUCE / SELL / SKIP定義が正しい
- SELLとREDUCEを0株基準で誤判定しない
- AI提案とユーザー判断を別保存する
- 判断保存で注文が発生しない
- 実際の約定を別記録できる
- 同一銘柄の複数口座に対応する
- Homeは実保有だけを表示する
- Rankingと最終Actionを混同しない
- Stock DetailでHOLDと提案全体を比較できる
- Validationでモデル診断と最終提案成績を分離する
- Settingsで資金、口座、税、コスト、制約、データ、モデルを確認できる
- 更新時刻とモデル版が表示される
- stale / missing / error時にfail closed表示になる
- モバイル幅で横スクロールが発生しない
- frontend lint、typecheck、test、production buildが通る
- 参考画像と大きく異なる場合は差分理由を `docs/DECISIONS.md` に記録する

