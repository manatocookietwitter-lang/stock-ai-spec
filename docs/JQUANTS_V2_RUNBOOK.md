# J-Quants V2 Data Runbook

更新日: 2026-08-24
対象: Goal 2 J-Quants V2実データ基盤

## 境界

- 使用するbase URLは`https://api.jquants.com/v2`だけ
- 認証値はprocess環境の`JQUANTS_API_KEY`からだけ読む
- repository内の`.env`や設定fileはclientが読み込まない
- credential値を標準出力、標準error、例外本文、log、fixture、Parquet、manifest、DuckDB、Markdownへ保存しない
- 認証がない場合は停止し、fixtureや別providerへfallbackしない
- broker発注、自動売買、前場realtime処理は存在しない

公式仕様参照:

- [J-Quants API V2 Python client](https://github.com/J-Quants/jquants-api-client-python)
- [Response pagination](https://jpx-jquants.com/en/spec/pagination)
- [Rate limits](https://jpx-jquants.com/en/spec/rate-limits)
- [Data update timing](https://jpx-jquants.com/en/spec/data-update)

## 実装endpoint

| Dataset | V2 endpoint | 最低plan | 既定取得 |
|---|---|---:|---:|
| 銘柄master | `/equities/master` | Free | Yes |
| 株価日足 | `/equities/bars/daily` | Free | Yes |
| 財務summary | `/fins/summary` | Free | Yes |
| 営業日calendar | `/markets/calendar` | Light | No |
| TOPIX日足 | `/indices/bars/daily/topix` | Light | No |

`stock-ai data capabilities --plan <plan>`はAPIを呼ばず、plan・実装・data capabilityの境界を表示する。

## 取得前の準備

API keyはrepository外でprocess環境へ設定する。`.env.example`は変数名の一覧だけであり、値を記入してcommitしない。本clientは`.env`を自動読込しない。

非ASCIIを含むWindows pathでは非editable installを使用する。

```text
python -m uv sync --all-groups --no-editable
```

## 1日分の取得

Free planの既定3 dataset:

```text
stock-ai data sync --date 2026-08-21 --plan free --data-root data
```

Light以上でcalendarとTOPIXも取得する場合:

```text
stock-ai data sync --date 2026-08-21 --plan light --data-root data --datasets security_master,daily_prices,financial_summary,trading_calendar,topix
```

出力はrun ID、状態、source date、object数だけで、request headerやresponse bodyは表示しない。

## 履歴取得と再開

Light以上のV2 Bulkを日付範囲で取得する。

```text
stock-ai data history --start 2017-01-04 --end 2026-08-21 --plan standard --data-root data
```

Bulk Listが空、CSVがheaderだけ、rowが範囲外、品質検証に失敗した場合は成功checkpointを作らない。
file fingerprint単位のcheckpointが`SUCCEEDED`で、紐付く全objectとfileが存在する場合だけ`--resume`でskipする。
`RUNNING / FAILED` runや、file全体のcheckpointが未完了なobjectはPIT読取へ公開しない。

## 保存構造

```text
data/
  raw/jquants_v2/<dataset>/source_date=YYYY-MM-DD/<object_id>/
    data.parquet
    manifest.json
  normalized/jquants_v2/<dataset>/source_date=YYYY-MM-DD/<object_id>/
    data.parquet
    manifest.json
  catalog.duckdb
  features/<feature-set-version>/<snapshot-id>/
  datasets/production/<snapshot-id>/
  builds/production/<build-id>/
```

各external recordは次を持つ。

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
source_record_hash
```

manifestはParquet SHA-256、row数、schema version、品質結果を持つ。object directoryは内容アドレス付きで、一度公開したものを上書きしない。

## Point-in-time方針

source objectは`available_at = received_at`とする。これはAPIが過去訂正の実施時刻を返さないためである。

- 初回backfillを、それ以前の予測日時で既知だった値として使わない
- 同じpayloadの再取得は既存objectを参照して重複しない
- 同一natural keyの内容が変われば新objectとして保存する
- DuckDB PIT読取は`available_at <= cutoff`の最新版だけを返す
- 初回取得以前のprovider revision historyは復元できず、capabilityは`PARTIAL`
- Production buildはrevision policyを固定する。`SINGLE_VINTAGE_AS_REVISED`は再現可能な研究専用で採用不可、
  `STRICT_AS_KNOWN`はreceipt時刻をavailabilityへ含め、復元不能なhistorical labelをblockする

## 株価系列

- raw execution参照: `raw_open/raw_high/raw_low/raw_close/raw_volume`
- research調整系列: `research_open/research_high/research_low/research_close/research_volume`
- 調整情報: `adjustment_factor/adjustment_version`

raw系列とresearch系列は別columnに保存し、暗黙に置換しない。

## 品質検証

normalized publish前に次を検証する。

- required columnとsource schema version
- dataset natural key重複
- requested date外のrow
- 5文字英数字のprovider issue code
- OHLCの有限・正値・high/low関係
- volume非負、adjustment factor正値
- disclosure date/time

品質error時も取得済みraw objectと品質reportは残すが、normalized objectは公開せずrunを`FAILED`にする。

## 完全性検証

```text
stock-ai data verify --data-root data
```

catalogと全manifest/Parquet hash/row数/object ID、成功Bulk checkpoint、V0/V1/V2/Dataset、最終Build Manifestを相互照合する。
不完全object、orphan snapshot、partial build、path/identity/hash/row数不一致、空storeは非zeroで停止する。新規の空storeを
意図的に確認する場合だけ`--allow-empty`を付ける。

Production Research artifactは次で生成・検証する。

```text
stock-ai research build --as-of 2026-08-24T11:30:00+09:00 --plan standard --data-root data
stock-ai data verify --data-root data
stock-ai research baseline --dataset-parquet <content-addressed-parquet> --code-commit <git-commit>
stock-ai research advanced --build-manifest <production-build-manifest.json> --code-commit <git-commit>
stock-ai research e2e --as-of 2026-08-24T11:30:00+09:00 --code-commit <git-commit> --plan standard
```

baselineは隣接metadata、Parquet hash、content ID、source lineageを再検証する。advanced researchはさらに
V0/V1/V2/Datasetの最終Build Manifestを入口にし、認証済みV2 snapshotとdatasetの同一lineageを必須とする。
任意にrenameしたParquetは受け付けない。

## 障害時

- 429、500、502、503、504、network一時障害はbounded exponential retry
- plan rate limitと`/fins/summary`個別limitの小さい方で逐次throttle
- `Retry-After`の秒形式とHTTP-date形式を尊重する。設定上限を超える待機要求は早期再試行せず安全にabortする
- pagination keyの循環とpage上限を検出する
- publish中断時は検証済み一時directoryだけを削除し、既存objectは保持する
- DuckDBにはcredentialやresponse bodyではなく例外class名だけを`error_code`として残す

## 既知の制約

- J-Quants契約planは自動判定せずCLI引数で宣言する
- 初回取得以前の訂正vintageと完全な上場・廃止event lineageは復元できない
- `ShOutFY`は期末開示値であり日次発行株式数ではない
- breadthは全銘柄coverageを別途確認するまで`PARTIAL`
- full JPX規模ではpandas/scikit-learnのmemory gateを実データで確認するまでscale capabilityは`PARTIAL`
- 前場・分足、需給、TDnet、broker連携はGoal 2対象外
