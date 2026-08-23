# Stock AI Decision Support — Specification Bundle

更新日: 2026-08-22  
状態: 仕様整理版 v0.3（UI・特徴量仕様統合、実装前）

## このリポジトリの目的

このプロジェクトは、毎営業日11:30時点で利用可能な情報を使い、現在の保有株・現金・税金・取引コスト・新規候補を同時に評価して、12:30に向けた投資提案を作る意思決定支援システムです。

AIやアプリは実注文を出しません。最終判断と注文はユーザーが行います。

## 仕様の正本

仕様と実装の優先順位は次の通りです。

1. ユーザーが明示した最新の変更
2. `docs/DECISIONS.md` の確定事項
3. `docs/MASTER_SPEC.md`
4. 各専門仕様（Decision Engine / Data / ML / Feature / UI）
5. `docs/IMPLEMENTATION_PLAN.md`
6. `docs/STATUS.md`

文書同士が矛盾した場合は、勝手に補完せず `docs/DECISIONS.md` に論点を記録します。

## ファイル構成

- `AGENTS.md`
  - Codexが毎回守る短い恒久ルール
  - 全仕様を詰め込まない
- `CODEX_GOAL.md`
  - Codexに貼る短い `/goal` のテンプレート
  - 3段階に分けて実装する
- `docs/MASTER_SPEC.md`
  - 製品全体の現在仕様
- `docs/DATA_CONTRACT.md`
  - データ、時点管理、保有・口座・提案のデータ契約
- `docs/ML_RESEARCH_SPEC.md`
  - 機械学習、ラベル、モデル、検証、実験方針
- `docs/FEATURE_CATALOG.md`
  - MACD、RSI、移動平均、Bollinger、ADX、ATR、出来高、財務、需給、市場環境、前場特徴の候補プールと計算規則
- `docs/DAILY_PORTFOLIO_DECISION_ENGINE.md`
  - M14の中核判断ロジック、税・コスト・株数最適化
- `docs/UI_SPEC.md`
  - モバイルPWAの画面、操作、状態、データ要件、UI受入条件
- `docs/ui-reference/`
  - 非正本の画面参考画像。矛盾時はMDを優先
- `docs/IMPLEMENTATION_PLAN.md`
  - M0〜M20の実装順序と完了条件
- `docs/DECISIONS.md`
  - 確定事項と未決事項
- `docs/STATUS.md`
  - 進行状況、テスト結果、次の作業

## Codexでの使い方

1. このフォルダ一式を空のGitリポジトリへ置く。
2. Codexをリポジトリのルートで開く。
3. 最初に `AGENTS.md` と `docs/` を読ませる。
4. `CODEX_GOAL.md` の Goal 1 を貼る。
5. 各Goal終了時に、Codexが `docs/STATUS.md` と `docs/DECISIONS.md` を更新したことを確認する。
6. Goal 2、Goal 3へ進む。

## なぜGoal本文へ全仕様を貼らないか

`/goal` は、達成したい結果・制約・検証条件を短く示すために使います。詳細な製品仕様、データ契約、機械学習実験、税・コスト計算はリポジトリ内のMDを正本にします。

Goalは「どの文書を読み、どのマイルストーンを、どの完了条件まで進めるか」を指定する役割に限定します。

## 現時点の最重要原則

> 毎日11:30に、現在のポートフォリオを維持する場合と、売却・買い増し・新規購入を組み合わせた代替ポートフォリオを比較し、12:30に向けて何株を増やす・維持する・減らす・売るべきか提案する。

最終表示は次の形式を中心にします。

```text
現在 300株
推奨 400株
BUY 100株
```

内部では金額・比率・期待リターン・リスクで最適化して構いませんが、提案は100株単位の株数を中心に表示します。

## 特徴量の追加方針

一般によく使われる指標は広い候補プールとして保存しますが、全てを最初から最終モデルへ投入しません。

```text
FeatureSet V0 最小Baseline
→ FeatureSet V1 Core 約40〜60候補
→ FeatureSet V2 Extended Technical
→ FeatureSet V3 Data-dependent
```

MACD、RSI、移動平均クロス等を固定BUY / SELLルールにせず、Point-in-timeのwalk-forward検証と特徴量群Ablationで本当に改善したものだけ残します。詳細は `docs/FEATURE_CATALOG.md` を参照します。

## Goal 1 MVP の実行

Python 3.12 と `uv` を使用する。

```text
uv sync --all-groups
uv run ruff check .
uv run mypy src
uv run pytest
uv run pytest --cov
uv run stock-ai fixture-demo
```

Windows でリポジトリの親パスに非ASCII文字があり、editable install の `.pth` が壊れる環境では、次の非editable同期を使用する。

```text
uv sync --all-groups --no-editable
uv run --no-sync stock-ai fixture-demo
```

`fixture-demo` は明示的な決定的fixtureだけを使い、次を一巡する。

```text
fixture OHLCV / 財務 / TOPIX
→ Point-in-time FeatureSet V1 Core（54特徴）
→ 1日・5日・20日ラベル
→ Ridge baselineとpurged expanding validation
→ 現在保有・現金・口座別状態
→ 取引コストと推定税影響
→ Daily Portfolio Decision Engine
→ 100株単位の提案
→ 手動約定fixture
→ 翌日状態
```

fixture は実データ欠損時の本番fallbackではなく、正しさと再現性の検証専用である。CLIも実注文を送信しない。

## 現在のパッケージ構成

- `src/stock_ai/domain`: 口座、保有、予測、提案、ユーザー判断、実約定の不変型
- `src/stock_ai/data`: `available_at <= as_of` のPoint-in-time検証
- `src/stock_ai/features`: Feature Registry、V0/V1 manifest、指標計算
- `src/stock_ai/ml`: dataset snapshot、1/5/20日label、Momentum/Ridge、時系列検証、実験記録
- `src/stock_ai/decision`: コスト、税、全体ポートフォリオ比較、状態遷移
- `tests`: 外部API不要の決定的fixtureテスト

APIキーや認証情報をソースへ置かない。ローカル設定は `.env.example` をコピーして使い、`.env` はGitへ追加しない。
