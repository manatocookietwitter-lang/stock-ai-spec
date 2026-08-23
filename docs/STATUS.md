# Project Status

更新日: 2026-08-22

## 現在地

`SPEC_UI_AND_FEATURES_CONSOLIDATED`

コード実装前。現在仕様をMDへ統合した段階。

## 完了

- 製品目的を意思決定支援へ固定
- 11:30提案 / 12:30向け判断を固定
- 現在保有株と新規候補の同時評価
- BUY / HOLD / REDUCE / SELL / SKIP
- 株数中心・100株単位
- M14 Daily Portfolio Decision Engine
- HOLD反実仮想
- 全体最適化
- Transaction Cost Engine
- Tax and Account Engine
- 状態付き検証
- 機械学習研究範囲
- Feature Candidate Catalog（テクニカル・財務・需給・市場・前場）
- FeatureSet V0 / V1 Core / V2 Extended / V3 Data-dependent
- MACD・RSI・MA・Bollinger・ADX・ATR・出来高系等の候補定義
- Feature Family Ablation、Feature Registry、数値fixture方針
- 自動提案とスマホPWAの役割
- `docs/UI_SPEC.md` v0.2
- 7画面の参考画像整理
- UI Action定義、状態、AI提案・ユーザー判断・実約定・Paper結果の分離、受入条件
- 段階別Codex Goal構成
- 未決事項一覧

## 未実装

- Git repository scaffold
- Python package
- Data adapters
- Point-in-time dataset
- Feature Registry / Feature Set manifests
- FeatureSet V0 / V1 Coreの実装
- FeatureSet V2 / V3の候補実装とablation
- Labels
- Models
- Backtest
- Cost / tax code
- Decision Engine
- Proposal generator
- Automation
- PWA実装
- CSV import
- Forward validation

## テスト

まだ実行していない。

## 次の推奨作業

`CODEX_GOAL.md` の Goal 1 を実行する。

Goal 1の中心E2E:

```text
fixture ingest
→ point-in-time dataset
→ FeatureSet V0 / V1 Core
→ known-value indicator tests
→ baseline model
→ stateful current portfolio
→ transaction cost
→ tax estimate
→ Daily Portfolio Decision Engine
→ 100-share proposal
→ manual execution record
→ next-day state
```

## ブロッカーではないもの

実データ契約や現在保有情報がなくても、決定的fixtureを使ってGoal 1のコードとテストは実装可能。

## 後で必要な外部情報

- J-Quants API key / plan
- 前場データ提供元
- 需給・市場Breadth・同時刻出来高履歴のデータ提供元
- TA-Lib等の指標実装ライブラリ方針
- V1 Core manifestの最終有効列
- 実際の口座区分
- 現在保有
- 利用可能現金
- SBI fee条件
- 税設定
- 実際のCSV形式
- 実行環境
- 通知方式

## 重要な制限

- コードが完成しても将来利益は確定しない
- forward performanceは時間経過後にしか評価できない
- 税計算は意思決定用推定
- 実注文はユーザーが行う
