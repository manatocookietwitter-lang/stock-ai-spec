import { type ReactNode, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ActionBadge, Metric, PageHeader, StatePanel, yen } from "../components";
import { useApiData } from "../hooks";
import type { StockDetail } from "../types";

export function StockDetailPage(): ReactNode {
  const { symbol = "" } = useParams();
  const { data, error, loading } = useApiData<StockDetail>(`/api/v1/stocks/${symbol}`);
  const [selectedBucket, setSelectedBucket] = useState("ALL");
  if (loading) return <StatePanel title="銘柄情報を読み込んでいます" />;
  if (error || !data) return <StatePanel tone="error" title="銘柄情報を取得できません" detail={error} />;

  const bucketIds = Array.from(new Set([
    ...data.positions.map((item) => item.account_bucket_id),
    ...data.proposalLines.map((item) => item.accountBucketId),
  ])).sort();
  const selectedPositions = selectedBucket === "ALL"
    ? data.positions
    : data.positions.filter((item) => item.account_bucket_id === selectedBucket);
  const selectedLines = selectedBucket === "ALL"
    ? data.proposalLines
    : data.proposalLines.filter((item) => item.accountBucketId === selectedBucket);
  const company = data.proposalLines[0]?.companyName || symbol;
  const totalShares = selectedPositions.reduce((sum, item) => sum + item.shares, 0);
  const weightedAverage = totalShares === 0
    ? null
    : selectedPositions.reduce(
      (sum, item) => sum + Number(item.average_acquisition_price) * item.shares,
      0,
    ) / totalShares;

  return (
    <div className="page">
      <Link className="back-link" to="/today">← 戻る</Link>
      <PageHeader eyebrow={symbol} title={company} meta={selectedLines.length === 1 ? yen(selectedLines[0].referencePrice) : undefined} />
      {bucketIds.length > 1 && <div className="account-tabs" role="tablist">
        <button role="tab" aria-selected={selectedBucket === "ALL"} className={selectedBucket === "ALL" ? "active" : ""} onClick={() => setSelectedBucket("ALL")}>合計</button>
        {bucketIds.map((bucketId) => <button role="tab" aria-selected={selectedBucket === bucketId} className={selectedBucket === bucketId ? "active" : ""} onClick={() => setSelectedBucket(bucketId)} key={bucketId}>{bucketId}</button>)}
      </div>}
      <section className="section">
        <div className="section-title"><div><span>ACTUAL POSITION</span><h2>{selectedBucket === "ALL" ? "実保有合計" : `実保有 / ${selectedBucket}`}</h2></div></div>
        {selectedPositions.length === 0 ? <StatePanel title="未保有" /> : <div className="metric-grid metric-grid--2">
          <Metric label="現在保有" value={`${totalShares}株`} />
          <Metric label="加重平均取得" value={yen(weightedAverage)} />
        </div>}
      </section>
      {selectedLines.length === 0 ? <StatePanel title="当日の提案はありません" /> : selectedLines.map((line) => <section className="section" key={line.lineId}>
        <div className="proposal-focus">
          <div><span>AI提案 / {line.accountBucketId}</span><ActionBadge action={line.action} /></div>
          <strong>{line.currentShares}株 → {line.recommendedShares}株</strong>
          <p>{line.shareDifference === 0 ? "変更なし" : `${line.shareDifference > 0 ? "+" : ""}${line.shareDifference}株`}</p>
        </div>
        <div className="reason-columns">
          <div><strong>主な要因</strong>{line.positiveReasons.length ? <ul>{line.positiveReasons.map((reason) => <li key={reason}>{reason}</li>)}</ul> : <p>構造化された追加理由はありません。</p>}</div>
          {line.negativeReasons.length > 0 && <div><strong>注意要因</strong><ul>{line.negativeReasons.map((reason) => <li key={reason}>{reason}</li>)}</ul></div>}
        </div>
        <dl className="summary-list">
          <div><dt>この口座行のコスト</dt><dd>-{yen(line.estimatedTransactionCost)}</dd></div>
          <div><dt>この口座行の推定税影響</dt><dd>{yen(line.estimatedTaxEffect)}</dd></div>
        </dl>
      </section>)}
      {data.proposal && <section className="section counterfactual">
        <div className="section-title"><div><span>COUNTERFACTUAL</span><h2>現在維持 vs 提案全体</h2></div></div>
        <dl className="summary-list">
          <div><dt>現在のまま保有</dt><dd>{yen(data.proposal.holdExpectedValue)}</dd></div>
          <div><dt>提案ポートフォリオ</dt><dd>{yen(data.proposal.proposedExpectedValue)}</dd></div>
          <div className="emphasis"><dt>全体の純改善</dt><dd>{yen(data.proposal.netImprovement)}</dd></div>
        </dl>
      </section>}
    </div>
  );
}
