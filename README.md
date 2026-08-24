# Stock AI Decision Support — Specification Bundle

更新日: 2026-08-24
状態: Goal 3本格ML研究基盤実装済み（live履歴runはcredential継承待ち）

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
- `docs/JQUANTS_V2_RUNBOOK.md`
  - Goal 2Aの取得、保存、検証、障害対応、PIT制約

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
fixture OHLCV / 財務 / TOPIX / sector / breadth
→ Point-in-time FeatureSet V1 Core（58特徴）
→ 1日・5日・20日ラベル（Goal 1では調整後終値間リターンの研究proxy）
→ Ridge baseline、purged expanding validation、未使用locked holdout
→ 現在保有・現金・口座別状態
→ 取引コストと推定税影響
→ Daily Portfolio Decision Engine
→ 100株単位の提案
→ 手動約定fixture
→ 翌日状態
```

fixture は実データ欠損時の本番fallbackではなく、正しさと再現性の検証専用である。CLIも実注文を送信しない。

Feature Engineは候補銘柄集合からsector値やbreadthを再計算しない。明示的なpoint-in-time
sector contextとmarket breadthを入力し、必要capabilityが欠ける場合は
`BLOCKED_BY_DATA_CAPABILITY`としてfail closedする。

## Goal 2A J-Quants V2データ基盤

公式J-Quants API V2から1日単位で取得し、schema・key・日付・OHLC・調整係数を検証してから、content-addressed immutable ParquetとDuckDB catalogへ保存する。

```text
stock-ai data capabilities --plan free
stock-ai data sync --date 2026-08-21 --plan free --data-root data
stock-ai data history --start 2017-01-04 --end 2026-08-21 --plan standard --data-root data
stock-ai data verify --data-root data
stock-ai research build --as-of 2026-08-24T11:30:00+09:00 --plan standard --data-root data
stock-ai research baseline --dataset-parquet <content-addressed-parquet> --code-commit <commit>
stock-ai research advanced --build-manifest <production-build-manifest.json> --code-commit <commit>
```

Free planの既定取得は銘柄master、株価日足、財務summary。Light以上の営業日calendarとTOPIXは`--datasets`で明示する。live取得はprocess環境の`JQUANTS_API_KEY`がない場合に停止し、fixtureへfallbackしない。

株価はexecution参照用raw系列とresearch用調整系列を分け、同じpayloadの再取得は重複させない。訂正値は新versionとして残す。APIが過去訂正時刻を返さないためsource objectは`available_at = received_at`とし、初回取得より前の時点へbackfill値を遡及させない。as-revised単一vintageは研究専用かつ採用不可として明示し、V0/V1/V2/Datasetはsource-frame ID付きのatomic Build Manifestで固定する。詳細は`docs/JQUANTS_V2_RUNBOOK.md`を参照する。

## Goal 3 本格ML研究基盤

FeatureSet V2 Extendedと、LightGBM / XGBoost / CatBoostの回帰・Learning to Rank、
1/5/20日、quantile / large-loss、bounded Optuna、feature-family ablation、OOF ensemble、
uncertainty calibrationを実装した。hyperparameter選択はstrictly earlier tuning期間、model評価は後続outer OOF、
stacking・uncertainty calibration・reported coverageはさらに3つの時系列区間へ分離する。
locked final holdoutは開かない。`research advanced`の成果物は常にresearch-onlyで、実注文を生成しない。

## 現在のパッケージ構成

- `src/stock_ai/domain`: 口座、保有、予測、提案、ユーザー判断、実約定の不変型
- `src/stock_ai/data`: J-Quants V2 client、品質検証、immutable Parquet、DuckDB、PIT読取
- `src/stock_ai/features`: Feature Registry、V0/V1/V2 manifest、指標計算
- `src/stock_ai/ml`: dataset snapshot、1/5/20日label、GBDT/LTR/downside、OOF ensemble、時系列検証、実験記録
- `src/stock_ai/decision`: コスト、税、全体ポートフォリオ比較、状態遷移
- `tests`: 外部API不要の決定的fixtureテスト

APIキーや認証情報をsource、log、例外、fixture、Markdown、Git履歴へ置かない。live clientはprocess環境の`JQUANTS_API_KEY`だけを読み、`.env`や設定fileを自動読込しない。
